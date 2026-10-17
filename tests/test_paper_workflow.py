import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from unifiedgroupgen.cli import build_models
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.training import file_hash

ROOT = Path(__file__).resolve().parents[1]


def script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(dataset='mp_20', kind='space'):
    descriptor = Descriptor(kind, 1, (OrbitSpec('a', 29), OrbitSpec('a', 29)))
    c = compile_descriptor(descriptor)
    state = c.prior(dtype=torch.float64).tolist()
    return {'id': dataset, 'dataset': dataset, 'descriptor': descriptor.to_dict(), 'state': state,
            'properties': {'formation_energy': -1.0, 'band_gap': None}}


def test_campaign_uses_train_quantiles_and_matched_descriptor_prior(tmp_path, monkeypatch):
    prepare = script('prepare_paper_experiments')
    runner = script('evaluate_joint')
    monkeypatch.setattr(prepare, 'ROOT', tmp_path)
    rows = [record() for _ in range(5)]
    for i, row in enumerate(rows):
        row['id'] = str(i)
        row['properties']['formation_energy'] = i-4.0
    condition = prepare.build_campaign({'mp_20': rows}, {'mp_20': 0}, 'conditions', tmp_path, [142], 2, 4, 1)
    assert len(condition['jobs']) == 9
    assert condition['targets']['mp_20/formation_energy']['targets'] == pytest.approx([-3.6, -2, -.4])
    assert not any('band_gap' in job['properties'] for job in condition['jobs'])
    controls = prepare.build_campaign({'mp_20': rows}, {'mp_20': 0}, 'controls', tmp_path, [142], 2, 4, 1)
    assert len(controls['jobs']) == 3
    assert {j['seed'] for j in controls['jobs']} == {142}
    assert len({j['descriptor'] for j in controls['jobs']}) == 1
    path = tmp_path/'campaign.json'
    path.write_text(json.dumps(controls))
    args = SimpleNamespace(campaign=str(path), skip_test=True, checkpoint='best.pt', device='cpu')
    plan = runner.plan(args, [])
    assert len(plan) == 3 and any('--prior' in cmd for _, cmd in plan)
    controls['jobs'][1]['name'] = controls['jobs'][0]['name']
    path.write_text(json.dumps(controls))
    with pytest.raises(ValueError, match='unique safe names'):
        runner.plan(args, [])


def test_matched_training_keeps_group_vocabulary_and_filters_only_inputs(tmp_path, monkeypatch):
    prepare = script('prepare_paper_experiments')
    monkeypatch.setattr(prepare, 'ROOT', tmp_path)
    config = {'dataset_domains': {'mp_20': 0, 'c2db_51': 2}, 'kinds': ['space', 'layer'],
              'properties': {'source_domain': {'num_classes': 3, 'context': True}, 'band_gap': {'type': 'scalar'}},
              'model': {'flow_hidden': 512},
              **{f'{s}_files': [f'joint/{d}/{s}.jsonl' for d in ('mp_20', 'c2db_51')]
                 for s in ('train', 'val', 'test')}}
    outputs = prepare.matched_configs(config, tmp_path, [8, 29], {'mp_20': {'band_gap'}, 'c2db_51': set()})
    bulk = yaml.safe_load((tmp_path/outputs['mp_20']).read_text())
    assert bulk['kinds'] == config['kinds'] and bulk['model'] == config['model']
    assert bulk['properties']['source_domain']['num_classes'] == 3
    assert bulk['joint_split_datasets'] == ['mp_20']
    assert bulk['test_files'] == ['joint/mp_20/test.jsonl']
    assert bulk['allowed_elements'] == [8, 29]
    layer = yaml.safe_load((tmp_path/outputs['c2db_51']).read_text())
    assert 'band_gap' not in layer['properties'] and 'source_domain' in layer['properties']


def test_actual_campaign_prior_observation_and_resume_on_cpu(tmp_path):
    prepare = script('prepare_paper_experiments')
    properties = {'formation_energy': {'type': 'scalar'},
                  'source_domain': {'type': 'categorical', 'num_classes': 3, 'context': True}}
    config = {'kind': 'space', 'groups': [('space', 1)], 'properties': properties,
              'allowed_elements': [29], 'max_atoms': 2, 'max_orbits': 2, 'batch_size': 1, 'edge_budget': 100,
              'dataset_domains': {'mp_20': 0}, 'require_data_audit': True,
              'data_audit': str(tmp_path/'audit.json'),
              'model': {'hidden': 16, 'proposal_layers': 1, 'flow_layers': 1, 'heads': 4, 'frequencies': 2}}
    row = record()
    hashes = {}
    for split in ('train', 'val', 'test'):
        path = tmp_path/f'{split}.jsonl'
        path.write_text(json.dumps(row)+'\n')
        config[f'{split}_files'] = [str(path)]
        hashes[split] = file_hash(path)
    (tmp_path/'audit.json').write_text(json.dumps({'passed': True, 'inputs': {
        split: [{'path': config[f'{split}_files'][0], 'sha256': hashes[split]}]
        for split in ('train', 'val', 'test')}}))
    proposal, flow = build_models(config, 'cpu')
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.zero_()
    checkpoint = tmp_path/'best.pt'
    torch.save({'config': config, 'proposal': proposal.state_dict(), 'flow': flow.state_dict(), 'epoch': 0,
                'train_split_sha256': [hashes['train']], 'val_split_sha256': [hashes['val']]}, checkpoint)
    with tempfile.TemporaryDirectory(prefix='test_paper_campaign_', dir=ROOT/'artifacts') as folder:
        folder = Path(folder)
        campaign = prepare.build_campaign({'mp_20': [row]}, {'mp_20': 0}, 'controls', folder, [142], 1, 1, 1)
        campaign['train_sha256'] = {config['train_files'][0]: hashes['train']}
        path = folder/'campaign.json'; path.write_text(json.dumps(campaign))
        command = [sys.executable, str(ROOT/'scripts/evaluate_joint.py'), '--checkpoint', str(checkpoint),
                   '--campaign', str(path), '--skip-test', '--device', 'cpu', '--output', str(folder/'run')]
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
        assert result.returncode == 0, result.stdout+'\n'+result.stderr
        state = json.loads((folder/'run/pipeline.json').read_text())
        outputs = {}
        for name, job in state['jobs'].items():
            outputs[name.split('_case000_')[1].split('_s')[0]] = json.loads(
                (folder/'run'/job['directory']/'samples.jsonl').read_text())
        assert outputs['prior']['state'] == outputs['flow']['state']  # zero flow, same random initial state
        assert outputs['prior']['generator'] == 'legal_prior'
        assert outputs['fixedlattice']['validation']['observation_max_residual'] == 0
        result = subprocess.run(command+['--resume'], capture_output=True, text=True, timeout=60)
        assert result.returncode == 0 and result.stdout.count('[REUSE]') == 3
        observation = ROOT/campaign['jobs'][-1]['observations']
        observation.write_text(observation.read_text()+'\n')
        changed = subprocess.run(command+['--resume'], capture_output=True, text=True, timeout=60)
        assert changed.returncode != 0 and 'settings changed' in changed.stderr


def test_independent_calculator_keeps_targets_absent_and_supports_layers(tmp_path, monkeypatch):
    from ase.calculators.emt import EMT
    from pymatgen.core import Lattice, Structure
    assess = script('assess_structures')
    monkeypatch.setattr(assess, 'ROOT', tmp_path)
    monkeypatch.setattr(assess, 'root_path', lambda p: Path(p) if Path(p).is_absolute() else tmp_path/p)
    cif = tmp_path/'cu.cif'
    Structure(Lattice.cubic(3.6), ['Cu'], [[0, 0, 0]]).to(filename=str(cif))
    task = {'job': 'mp_20_unconditional', 'index': 0, 'kind': 'space', 'cif': 'cu.cif', 'sha256': file_hash(cif)}
    value = assess.assess_one(task, EMT(), tmp_path, False, False, .05, 5,
        {'elemental_eV_per_atom': {'Cu': -1.0},
         'phase_entries': [{'composition': {'Cu': 1}, 'energy_eV': -1.0}]})
    assert np.isfinite(value['total_energy_eV']) and 'band_gap' not in value
    assert value['formation_energy'] == pytest.approx(value['total_energy_eV']+1)
    assert value['energy_above_hull'] == pytest.approx(value['formation_energy'])
    assert not value['relaxation_success']
    task['kind'] = 'layer'
    value = assess.assess_one(task, EMT(), tmp_path, True, True, 10, 1)
    assert value['relaxation_success'] and not value['cell_relaxed']
    assert (tmp_path/value['relaxed_cif']).exists()
    cif.write_text(cif.read_text()+'\n')
    with pytest.raises(ValueError, match='CIF changed'):
        assess.assess_one(task, EMT(), tmp_path, False, False, .05, 5)


def test_paper_png_unknowns_and_denominators_are_preserved(tmp_path):
    paper = script('build_paper_results')
    rows = pd.DataFrame([
        {'model': 'Engineering fixture', 'dataset': 'mp_20', 'mode': 'unconditional', 'coordinate_dof': 1,
         'joint_valid': True, 'exact_target_group': True, 'legacy_structural_valid': True, 'composition_valid': True},
        {'model': 'Engineering fixture', 'dataset': 'mp_20', 'mode': 'unconditional', 'coordinate_dof': None,
         'joint_valid': None, 'exact_target_group': None}])
    rates = paper.grouped_rates(rows, ['model', 'dataset'])
    assert rates.iloc[0].joint_valid == .5 and rates.iloc[0].joint_valid_unknown == 1
    low, high = paper.wilson(64, 64)
    assert .9 < low < 1 and high == pytest.approx(1)
    table = pd.DataFrame([{'stable_rate': None, 'sun_rate': None, 'mode': 'unconditional',
                          'dataset': 'mp_20', 'model': 'Engineering fixture',
                          'cov_precision': None, 'cov_recall': None, 'valid_unique_novel_rate': None}])
    files = paper.render(table, rows, tmp_path)
    assert len(files) == 3
    assert all(path.suffix == '.png' for path in tmp_path.iterdir())
    assert not any('condition' in f or 'physical' in f for f in files)


def test_external_baseline_import_counts_missing_cif_as_failure(tmp_path):
    from pymatgen.core import Lattice, Structure
    baseline = script('import_baseline_samples')
    Structure(Lattice.cubic(3.6), ['Cu'], [[0, 0, 0]]).to(filename=str(tmp_path/'cu.cif'))
    manifest = tmp_path/'baseline.csv'
    pd.DataFrame([{'index': 0, 'status': 'generated', 'cif': 'cu.cif'},
                  {'index': 1, 'status': 'generated', 'cif': 'missing.cif'},
                  {'index': 2, 'status': 'failed', 'cif': ''}]).to_csv(manifest, index=False)
    output = tmp_path/'imported'; output.mkdir()
    rows = baseline.convert(manifest, 'mp_20', 3, output, {})
    assert len(rows) == 3 and [r['status'] for r in rows] == ['generated', 'failed', 'failed']
    assert rows[0]['descriptor']['number'] is None
    assert not rows[0]['validation']  # no invented exact-target or chemistry success
    with pytest.raises(ValueError, match='every attempted index'):
        baseline.convert(manifest, 'mp_20', 4, output, {})


def test_metrics_collection_rejects_duplicate_or_changed_rows(tmp_path, monkeypatch):
    from pymatgen.core import Lattice, Structure
    metric = script('evaluate_generation')
    paper = script('build_paper_results')
    monkeypatch.setattr(paper, 'ROOT', tmp_path)
    cif = tmp_path/'sample_00000.cif'
    Structure(Lattice.cubic(3), ['Na', 'Cl'], [[0, 0, 0], [.5, .5, .5]]).to(filename=str(cif))
    samples = tmp_path/'samples.jsonl'
    samples.write_text(json.dumps({'index': 0, 'status': 'generated',
        'descriptor': {'kind': 'space', 'number': 1}, 'properties_requested': {'band_gap': 1},
        'validation': {'exact_target_group': False}})+'\n')
    rows, report = metric.evaluate_samples(samples, {'rows': [], 'fingerprints': [], 'requested': 0},
        {}, 'space', assessments={0: {'band_gap': 1.1}}, use_fingerprints=False)
    report.update(samples_sha256=file_hash(samples), dataset='mp_20', sampling={'seed': 142, 'guidance': 1})
    path = tmp_path/'mp_20_conditional.csv'; pd.DataFrame(rows).to_csv(path, index=False)
    report['metrics_csv_sha256'] = file_hash(path)
    summary = {'state': 'complete', 'reference_identity': 'user_selected_references',
        'train_sha256': [], 'test_sha256': [], 'evaluation_code_sha256': {}, 'versions': {},
        'jobs': {'mp_20_conditional': report}}
    path = tmp_path/'summary.json'; path.write_text(json.dumps(summary))
    experiment = {'label': 'Engineering fixture', 'metrics': str(path)}
    table, attempts, provenance = paper.collect({'experiments': [experiment]})
    assert table.iloc[0].band_gap_mae == pytest.approx(.1)
    assert attempts.iloc[0].predicted_band_gap == pytest.approx(1.1)
    assert len(provenance) == 1
    with pytest.raises(ValueError, match='Duplicate sample'):
        paper.collect({'experiments': [experiment, experiment]})
    row_path = tmp_path/'mp_20_conditional.csv'; row_path.write_text(row_path.read_text()+'\n')
    with pytest.raises(ValueError, match='file changed'):
        paper.collect({'experiments': [experiment]})


def test_assessment_cli_resume_does_not_duplicate_completed_tasks(tmp_path):
    from pymatgen.core import Lattice, Structure
    cif = tmp_path/'cu.cif'
    Structure(Lattice.cubic(3.6), ['Cu'], [[0, 0, 0]]).to(filename=str(cif))
    tasks = tmp_path/'tasks.json'
    tasks.write_text(json.dumps({'tasks': [
        {'job': 'mp_20_unconditional', 'index': 0, 'kind': 'space', 'cif': str(cif), 'sha256': file_hash(cif)},
        {'dataset': 'mp_20', 'id': 'cu', 'kind': 'space', 'cif': str(cif), 'sha256': file_hash(cif)}]}))
    with tempfile.TemporaryDirectory(prefix='test_assessment_', dir=ROOT/'artifacts') as folder:
        command = [sys.executable, str(ROOT/'scripts/assess_structures.py'), '--tasks', str(tasks),
                   '--calculator-factory', 'ase.calculators.emt:EMT', '--evaluator', 'EMT engineering fixture',
                   '--relax', '--fmax', '10', '--steps', '1', '--output', folder]
        result = subprocess.run(command, capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stdout+'\n'+result.stderr
        report = json.loads((Path(folder)/'summary.json').read_text())
        assert report['assessed'] == 2 and report['errors'] == 0
        checksum = file_hash(Path(folder)/'assessments.jsonl')
        result = subprocess.run(command+['--resume'], capture_output=True, text=True, timeout=90)
        assert result.returncode == 0, result.stderr
        assert file_hash(Path(folder)/'assessments.jsonl') == checksum
        reference = pd.read_csv(Path(folder)/'references.csv')
        assert not reference.iloc[0].relaxation_success and 'band_gap' not in reference
        cif.write_text(cif.read_text()+'\n')
        result = subprocess.run(command+['--resume'], capture_output=True, text=True, timeout=90)
        assert result.returncode != 0 and 'Input CIF changed' in result.stderr
