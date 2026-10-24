import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT/'scripts/legacy_eval'


def load(name):
    spec = importlib.util.spec_from_file_location('legacy_'+name, TOOLS/(name+'.py'))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(TOOLS))
    return module


def test_ten_job_plan_is_cpu_only_read_only_and_tests_precede_sampling(monkeypatch, capsys):
    module = load('campaign')
    output = ROOT/'artifacts/evaluations/legacy_plan_test_not_created'
    assert not output.exists()
    monkeypatch.setattr(sys, 'argv', ['campaign', '--project-root', str(ROOT), '--cgdit-root', str(ROOT.parent),
        '--predictor-root', 'unused', '--output', str(output)])
    monkeypatch.setattr(module.subprocess, 'Popen', lambda *a, **k: pytest.fail('Plan started a child process'))
    module.main()
    plan = json.loads(capsys.readouterr().out)
    assert not output.exists() and plan['total_sampling_attempts'] == 40960
    jobs = plan['jobs']
    assert len(jobs) == 33 and [j['stage'] for j in jobs[:3]] == ['test']*3
    samples = [j for j in jobs if j['stage'] == 'sample']
    assert len(samples) == 10
    assert [j['name'] for j in samples[:4]] == ['M1_mp_20_unconditional', 'M1_mp_20_conditional',
                                              'M2_mpts_52_unconditional', 'M2_mpts_52_conditional']
    for j in samples:
        cmd = j['command']
        assert cmd[cmd.index('--device')+1] == 'cpu'
        assert cmd[cmd.index('--count')+1] == '4096'
        assert 'train' not in cmd
    for job in jobs:
        assert 'main15_seed42' not in json.dumps(job)


def test_token_decomposition_matches_actual_masked_training_loss():
    from unifiedgroupgen.proposal import DescriptorTransformer
    from unifiedgroupgen.symmetry import Descriptor, OrbitSpec
    module = load('components')
    model = DescriptorTransformer([('space', 2)], {'gap': {}}, hidden=8, layers=1, heads=2,
                                  max_atoms=8, max_orbits=4, allowed_elements=[8, 14]).eval()
    descriptors = [Descriptor('space', 2, (OrbitSpec('i', 14),)),
                   Descriptor('space', 2, (OrbitSpec('a', 8), OrbitSpec('i', 14)))]
    props = [{'gap': 1.}, {'gap': None}]
    with torch.inference_mode():
        actual = module.descriptor_batch(model, descriptors, props)
        expected = model.loss_batch(descriptors, props)
    for row, target, descriptor in zip(actual, expected, descriptors):
        assert row['descriptor_loss'] == pytest.approx(float(target), abs=1e-6)
        assert sum(p['contribution'] for p in row['parts'].values()) == pytest.approx(float(target), abs=1e-6)
        assert row['parts']['G']['tokens'] == row['parts']['STOP']['tokens'] == 1
        assert row['parts']['W']['tokens'] == row['parts']['Z']['tokens'] == len(descriptor.orbits)


def test_predictor_bridge_keeps_failure_and_noncontiguous_attempt_indices():
    module = load('properties')
    def graphs(arrays, **kwargs):
        assert kwargs['num_workers'] == 1
        return ['graph-a', 'graph-c'], [0, 2], [None, 'graph failed', None]
    def predict(checkpoint, target, data, indices, count, batch_size, device):
        assert device == 'cpu' and count == 3
        result = np.full(count, np.nan); result[indices] = [-1., -2.]
        return result
    backend = SimpleNamespace(build_graphs=graphs, predict_graphs=predict)
    predictors = {name: {'checkpoint': 'unused', 'target': name, 'graph_settings': {}}
                  for name in ['formation_energy', 'band_gap']}
    items = [{'index': 9, '_crystal': {}}, {'index': 100, 'parse_error': 'bad CIF'},
             {'index': 23, '_crystal': {}}, {'index': 4, '_crystal': {}}]
    rows = module.infer_items(items, predictors, backend, 8)
    assert [r['index'] for r in rows] == [9, 100, 23, 4]
    assert rows[0]['formation_energy'] == -1 and rows[3]['formation_energy'] == -2
    assert 'formation_energy' not in rows[1] and rows[2]['formation_energy'] is None
    assert rows[2]['graph_error'] == 'graph failed'


def test_property_hit_denominators_and_missing_hull_are_not_hidden():
    module = load('properties')
    rows = [{'formation_energy': -1., 'band_gap': 1.}, {'formation_energy': None}, {}]
    reference = [{'formation_energy': -1.1, 'band_gap': 1.2}]
    tolerances = {'formation_energy': .06, 'band_gap': .45}
    report = module.summarize(rows, reference, {'formation_energy': -1., 'energy_above_hull': 0.}, tolerances)
    assert report['properties']['formation_energy']['hit_rate_all'] == pytest.approx(1/3)
    assert report['properties']['formation_energy']['hit_rate_predicted'] == 1
    assert report['targets_unassessed'] == ['energy_above_hull']
    assert report['all_requested_targets_joint_hit_rate'] is None
    report = module.summarize(rows, reference, {'formation_energy': -1., 'band_gap': 1.}, tolerances)
    assert report['all_requested_targets_joint_hit_rate'] == pytest.approx(1/3)


def test_predictor_inventory_checks_targets_scaling_graph_policy_and_final_arrays(tmp_path):
    import yaml
    module = load('properties')
    for short, target in [('fe', 'formation_energy_per_atom'), ('bg', 'band_gap')]:
        run = tmp_path/short/'seed123'; run.mkdir(parents=True)
        torch.save({'hyper_parameters': {'target_prop': target, 'target_mean': -1., 'target_std': 2.}}, run/'model.ckpt')
        (run/'hparams.yaml').write_text(yaml.safe_dump({'data': {'niggli': True, 'primitive': False,
                                                              'graph_method': 'crystalnn', 'max_atoms': 20}}))
        np.save(run/'test_preds.npy', [-1., 1.]); np.save(run/'test_targets.npy', [-1.1, .9])
    backend = SimpleNamespace(resolve_checkpoint=lambda run: run/'model.ckpt')
    info = module.predictors_at(tmp_path, 123, backend)
    assert info['formation_energy']['target_std'] == 2
    assert info['band_gap']['graph_settings']['niggli'] is True
    assert info['band_gap']['saved_test_array_agreement']['mae'] == pytest.approx(.1)
    torch.save({'hyper_parameters': {'target_prop': 'wrong'}}, tmp_path/'bg/seed123/model.ckpt')
    with pytest.raises(ValueError, match='target is not'):
        module.predictors_at(tmp_path, 123, backend)


def test_full_component_entry_records_flow_scale_and_unsupported_elements(tmp_path, monkeypatch):
    from unifiedgroupgen.cli import build_models
    from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor
    module = load('components')
    records = []
    for i, element in enumerate([14, 14, 8]):
        descriptor = Descriptor('space', 2, (OrbitSpec('i', element),))
        records.append({'id': str(i), 'dataset': 'mp_20', 'descriptor': descriptor.to_dict(),
                        'state': compile_descriptor(descriptor).prior().tolist(), 'properties': {}})
    data = tmp_path/'test.jsonl'
    data.write_text('\n'.join(json.dumps(r) for r in records), encoding='utf-8')
    config = {'groups': [('space', 2)], 'kind': 'space', 'properties': {}, 'max_atoms': 8, 'max_orbits': 4,
              'allowed_elements': [14], 'prepared_root': str(tmp_path), 'test_files': [str(data)],
              'model': {'hidden': 8, 'proposal_layers': 1, 'flow_layers': 1, 'heads': 2, 'frequencies': 1}}
    proposal, flow = build_models(config, 'cpu')
    checkpoint = tmp_path/'model.pt'
    torch.save({'config': config, 'proposal': proposal.state_dict(), 'flow': flow.state_dict(), 'epoch': 0,
                'runtime_options': {'coordinate_length_scale': 3.}}, checkpoint)
    output = tmp_path/'result'
    monkeypatch.setattr(sys, 'argv', ['components', '--project-root', str(ROOT), '--checkpoint', str(checkpoint),
                                    '--output', str(output), '--batch-size', '2'])
    module.main()
    report = json.loads((output/'summary.json').read_text())
    assert report['requested'] == 3 and report['overall']['evaluated'] == 2
    assert report['unsupported'][0]['id'] == '2'
    metrics = report['overall']
    assert metrics['flow'] == pytest.approx(metrics['lattice']+metrics['coordinates']/9)
    assert report['sources']['mp_20']['evaluated'] == 2


def test_standard_cli_consumes_prior_work_property_tolerances(tmp_path, monkeypatch):
    import yaml
    spec = importlib.util.spec_from_file_location('legacy_standard_metrics', ROOT/'scripts/evaluate_generation.py')
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    from unifiedgroupgen.symmetry import Descriptor, OrbitSpec
    descriptor = Descriptor('space', 1, (OrbitSpec('a', 11), OrbitSpec('a', 17)))
    record = {'id': 'ref', 'dataset': 'mp_20', 'descriptor': descriptor.to_dict(),
              'state': [0.]*9+[.5]*3, 'properties': {}}
    structure = module.record_structure(record)
    structure.to(filename=str(tmp_path/'sample_00000.cif'))
    reference = tmp_path/'test.jsonl'; reference.write_text(json.dumps(record)+'\n')
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'train_files': [str(reference)], 'test_files': [str(reference)]}))
    samples = tmp_path/'samples.jsonl'
    samples.write_text(json.dumps({'index': 0, 'status': 'generated', 'descriptor': descriptor.to_dict(),
        'properties_requested': {'formation_energy': -1., 'band_gap': 1.}, 'validation': {}})+'\n')
    assessments = tmp_path/'predictions.csv'
    assessments.write_text('job,index,formation_energy,band_gap\nmp_20_explicit,0,-0.92,1.3\n')
    metadata = tmp_path/'metadata.json'
    metadata.write_text(json.dumps({'evaluator': 'test', 'units': {'band_gap': 'eV'},
        'property_tolerances': {'formation_energy': .06, 'band_gap': .45, 'energy_above_hull': .03}}))
    monkeypatch.setattr(module, 'artifact_path', lambda p: Path(p))
    output = tmp_path/'metrics'; figures = tmp_path/'figures'
    monkeypatch.setattr(sys, 'argv', ['evaluate_generation', '--samples', str(samples), '--dataset', 'mp_20',
        '--reference-config', str(config), '--output', str(output), '--figures', str(figures),
        '--no-fingerprints', '--assessments', str(assessments), '--assessment-metadata', str(metadata)])
    module.main()
    report = json.loads((output/'summary.json').read_text())['jobs']['mp_20_explicit']['external']
    assert report['properties']['formation_energy']['valid_hit_count'] == 0
    assert report['properties']['band_gap']['valid_hit_count'] == 1
    assert report['properties']['band_gap']['tolerance'] == .45


def test_campaign_resumes_completed_stages_and_rejects_changed_budget(tmp_path, monkeypatch):
    import types
    import yaml
    module = load('campaign')
    root = tmp_path/'project'; (root/'configs').mkdir(parents=True)
    (root/'src/unifiedgroupgen').mkdir(parents=True)
    for name, run, config_name, epoch, _ in module.MODELS:
        data = root/'data'/name; data.mkdir(parents=True)
        for split in ('train', 'val', 'test'):
            (data/(split+'.jsonl')).write_text('test')
        (root/'configs'/f'{config_name}.yaml').write_text(yaml.safe_dump({'prepared_root': str(data)}))
        checkpoint = root/'artifacts/runs'/run/'model.best.pt'; checkpoint.parent.mkdir(parents=True)
        torch.save({'epoch': epoch-1, 'config': {}, 'train_split_sha256': [module.digest(data/'train.jsonl')],
                    'val_split_sha256': [module.digest(data/'val.jsonl')]}, checkpoint)
    cgdit = tmp_path/'cgdit'
    for path in ['cgdit/evaluation/generated_properties.py', 'cgdit/prop_models/gnn_models/m3gnet.py',
                 'cgdit/common/data_utils.py', 'cgdit/prop_models/gnn_models/_layers.py']:
        p = cgdit/path; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('code')
    package = types.ModuleType('cgdit'); package.__path__ = []
    evaluation = types.ModuleType('cgdit.evaluation'); evaluation.generated_properties = SimpleNamespace()
    monkeypatch.setitem(sys.modules, 'cgdit', package)
    monkeypatch.setitem(sys.modules, 'cgdit.evaluation', evaluation)
    monkeypatch.setattr(sys.modules['properties'], 'predictors_at', lambda *a: {'checkpoint': 'fixed'})
    calls = []
    class Child:
        pid = 123
        def __init__(self, command, **kwargs):
            calls.append(command)
            assert kwargs['env']['CUDA_VISIBLE_DEVICES'] == ''
            assert kwargs['env']['OMP_NUM_THREADS'] == '1'
            output = Path(command[command.index('--output')+1]); output.mkdir(parents=True)
            (output/'summary.json').write_text('{"state":"complete"}')
        def wait(self):
            return 0
    monkeypatch.setattr(module.subprocess, 'Popen', Child)
    output = root/'artifacts/evaluations/campaign'
    command = ['campaign', '--project-root', str(root), '--cgdit-root', str(cgdit),
               '--predictor-root', str(cgdit/'predictors'), '--output', str(output), '--run', '--stages', 'test']
    monkeypatch.setattr(sys, 'argv', command); module.main()
    assert len(calls) == 3
    assert json.loads((output/'pipeline.json').read_text())['state'] == 'stages_complete'
    calls.clear()
    monkeypatch.setattr(sys, 'argv', command+['--resume']); module.main()
    assert not calls and not (output/'pipeline.lock').exists()
    monkeypatch.setattr(sys, 'argv', command+['--resume', '--count', '2048'])
    with pytest.raises(ValueError, match='identical'):
        module.main()
    assert not calls
