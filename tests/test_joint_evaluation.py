import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import pytest
import torch

from unifiedgroupgen.cli import build_models
from unifiedgroupgen.evaluation import summarize_attempts
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.training import file_hash

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('evaluate_joint', ROOT/'scripts/evaluate_joint.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


def test_evaluation_plan_has_no_training_and_keeps_domain_conditions():
    args = SimpleNamespace(checkpoint='best.pt', device='cpu', seed=142, count=256, steps=64, guidance=2)
    jobs = evaluation.plan(args, ['a', 'b', 'c'])
    assert len(jobs) == 7
    assert all('train' not in command for _, command in jobs)
    for name, command in jobs[1:]:
        properties = json.loads(command[command.index('--properties')+1])
        assert command[command.index('--count')+1] == '256'
        if name.endswith('_unconditional'):
            assert set(properties) == {'source_domain'}
        if name.startswith('mpts_52'):
            assert 'band_gap' not in properties
        if name.startswith('c2db_51'):
            assert properties['source_domain'] == 2
            assert command[command.index('--kind')+1] == 'layer'


def test_reuse_requires_all_attempts_and_cif_files(tmp_path):
    rows = [{'index': 0, 'status': 'failed', 'reason': 'test'}]
    (tmp_path/'samples.jsonl').write_text(json.dumps(rows[0])+'\n')
    (tmp_path/'summary.json').write_text(json.dumps(summarize_attempts(rows)))
    with pytest.raises(ValueError, match='Incomplete'):
        evaluation.completed_outputs(tmp_path, 'sample', 2)
    assert len(evaluation.completed_outputs(tmp_path, 'sample', 1)) == 2
    rows[0] = {'index': 0, 'status': 'generated', 'validation': {}}
    (tmp_path/'samples.jsonl').write_text(json.dumps(rows[0])+'\n')
    (tmp_path/'summary.json').write_text(json.dumps(summarize_attempts(rows)))
    with pytest.raises(FileNotFoundError):
        evaluation.completed_outputs(tmp_path, 'sample', 1)


def test_joint_summary_retains_unknown_and_failed_denominators():
    valid = dict(group_operation_closure=True, distance_valid=True, orbit_multiplicity_regular=True,
                 no_coincident_orbits=True, metric_residual=0.0)
    result = summarize_attempts([
        {'status': 'generated', 'validation': {**valid, 'exact_target_group': True}},
        {'status': 'generated', 'validation': {**valid, 'exact_target_group': None}},
        {'status': 'failed'},
    ])
    assert result['exact_group_rate_over_all_attempts'] == pytest.approx(1/3)
    assert result['exact_group_unidentified_generated'] == 1
    assert result['geometry_screen_success'] == 2
    assert result['geometry_and_exact_group_success'] == 1
    assert not any(result['external_validation_performed_counts'].values())


def test_complete_cpu_shared_evaluation_and_verified_resume(tmp_path):
    """Exercise actual subprocess CLI, checkpoint loading, 3 domains and native SG/LG checks."""
    properties = {name: {'type': 'scalar'} for name in ('formation_energy', 'band_gap', 'energy_above_hull')}
    properties['source_domain'] = {'type': 'categorical', 'num_classes': 3, 'context': True}
    config = {'kind': 'space', 'groups': [('space', 1), ('layer', 1)], 'properties': properties,
              'allowed_elements': [14], 'max_atoms': 2, 'max_orbits': 1, 'batch_size': 2, 'edge_budget': 100,
              'model': {'hidden': 16, 'proposal_layers': 1, 'flow_layers': 1, 'heads': 4, 'frequencies': 2},
              'dataset_domains': {'mp_20': 0, 'mpts_52': 1, 'c2db_51': 2},
              'require_data_audit': True, 'data_audit': str(tmp_path/'audit.json')}
    tests = []
    for dataset, _, kind, _ in evaluation.TARGETS:
        descriptor = Descriptor(kind, 1, (OrbitSpec('a', 14),))
        record = {'id': dataset, 'dataset': dataset, 'properties': {'formation_energy': -1.0},
                  'descriptor': descriptor.to_dict(), 'state': compile_descriptor(descriptor).prior().tolist()}
        path = tmp_path/f'{dataset}.jsonl'
        path.write_text(json.dumps(record)+'\n')
        tests.append(path)
    config['test_files'] = [str(path) for path in tests]
    audit = {'passed': True, 'inputs': {'train': [{'sha256': 'train'}], 'val': [{'sha256': 'val'}],
             'test': [{'path': str(path), 'sha256': file_hash(path)} for path in tests]}}
    (tmp_path/'audit.json').write_text(json.dumps(audit))
    proposal, flow = build_models(config, 'cpu')
    # A stationary flow isolates orchestration from untrained-network numerical blow-ups.
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.zero_()
    checkpoint = tmp_path/'model.best.pt'
    torch.save({'config': config, 'proposal': proposal.state_dict(), 'flow': flow.state_dict(), 'epoch': 0,
                'train_split_sha256': ['train'], 'val_split_sha256': ['val']}, checkpoint)
    artifacts = ROOT/'artifacts'
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='test_joint_evaluation_', dir=artifacts) as folder:
        output = Path(folder)/'evaluation'
        command = [sys.executable, str(ROOT/'scripts/evaluate_joint.py'), '--checkpoint', str(checkpoint),
                   '--output', str(output), '--device', 'cpu', '--count', '1', '--steps', '1']
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, timeout=240)
        assert result.returncode == 0, result.stdout+'\n'+result.stderr
        summary = json.loads((output/'summary.json').read_text())
        assert summary['test']['evaluated_records'] == 3
        assert len(summary['generation']) == 6
        assert all(group['attempted'] == 1 for group in summary['generation'].values())
        assert all(group['generated'] == 1 for group in summary['generation'].values()), {
            name: (output/group['directory']/'samples.jsonl').read_text()
            for name, group in summary['generation'].items()}
        status = json.loads((output/'pipeline.json').read_text())
        assert status['state'] == 'complete'
        for job in list(status['jobs'].values())[1:]:
            assert json.loads((output/job['directory']/'sampling.status.json').read_text())['state'] == 'complete'
        result = subprocess.run(command+['--resume'], cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stderr
        assert result.stdout.count('[REUSE]') == 7
        assert '[START]' not in result.stdout
        # A modified artifact may not be silently reused.
        report_path = output/status['jobs']['test']['directory']/'result.json'
        report_path.write_text(report_path.read_text()+'\n')
        result = subprocess.run(command+['--resume'], cwd=ROOT, capture_output=True, text=True, timeout=60)
        assert result.returncode != 0
        assert 'Completed outputs changed' in result.stderr
