import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest
import yaml


def load(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_capacity_plan_never_starts_a_process_or_writes(tmp_path, monkeypatch, capsys):
    module = load('benchmark_batch_capacity')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: pytest.fail('plan started a process'))
    monkeypatch.setattr(sys, 'argv', ['benchmark_batch_capacity.py', '--output', 'artifacts/benchmarks/test'])
    module.main()
    plan = json.loads(capsys.readouterr().out)
    assert not plan['formal_training_started']
    assert plan['initial_edge_budget'] == 200000 and plan['conditional_edge_budget'] == 400000
    assert not (tmp_path/'artifacts').exists()


def test_incomplete_study_blocks_benchmark_before_gpu_access(tmp_path, monkeypatch):
    module = load('benchmark_batch_capacity')
    state = tmp_path/'artifacts/loss_balance_4090_seed42/status.json'
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({'state': 'running'}))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: pytest.fail('started before evaluation complete'))
    monkeypatch.setattr(sys, 'argv', ['benchmark_batch_capacity.py', '--output', 'artifacts/benchmarks/test', '--run'])
    with pytest.raises(SystemExit):
        module.main()
    assert not (tmp_path/'artifacts/benchmarks').exists()


def test_edge_pressure_and_commands_preserve_original_config(tmp_path):
    module = load('benchmark_batch_capacity')
    report = {'batch_size': 256, 'batch_sizes': [120, 256], 'batch_edges': [190000, 10000], 'edge_budget': 200000}
    assert module.edge_limited(report)
    report['batch_edges'][0] = 100
    assert not module.edge_limited(report)  # A partial tail batch alone is not a budget limit.
    report['batch_sizes'][1] = 120
    assert module.edge_limited(report)  # Multiple partial batches cannot all be tail batches.
    random = module.command(tmp_path/'config.yaml', tmp_path/'pool.jsonl', tmp_path/'r.json', 128, 400000, 'random')
    stress = module.command(tmp_path/'config.yaml', tmp_path/'pool.jsonl', tmp_path/'s.json', 128, 400000, 'largest')
    assert '--full-pass' in random and '--records' in random
    assert '--records' not in stress and stress[stress.index('--selection')+1] == 'largest'
    assert '--no-checkpoint' in random and '--edge-budget' in random
    assert not any('train_main15' in item for item in random+stress)


def test_benchmark_full_pass_respects_runtime_edge_cap_on_cpu(tmp_path, monkeypatch, capsys):
    module = load('benchmark')
    config = tmp_path/'toy.yaml'
    config.write_text(yaml.safe_dump({'dataset': 'synthetic', 'kind': 'space', 'groups': [['space', 1]],
        'properties': {}, 'allowed_elements': [14], 'max_atoms': 4, 'max_orbits': 4,
        'model': {'hidden': 8, 'heads': 2, 'proposal_layers': 1, 'flow_layers': 1, 'frequencies': 1},
        'composition_probability': 0., 'observation_probability': 0., 'edge_budget': 100}))
    original = config.read_bytes()
    monkeypatch.setattr(module, 'artifact_path', lambda path: Path(path))
    output = tmp_path/'report.json'
    monkeypatch.setattr(sys, 'argv', ['benchmark.py', '--config', str(config), '--atoms', '2',
        '--batch-size', '4', '--edge-budget', '4', '--full-pass', '--batches', '1', '--steps', '1',
        '--warmup', '1', '--threads', '1', '--device', 'cpu', '--output', str(output)])
    module.main()
    report = json.loads(output.read_text())
    assert report['batch_sizes'] == [2, 2] and report['batch_edges'] == [4, 4]
    assert report['measured_steps'] == 2 and report['measured_records'] == 4
    assert report['warmup'] == 2 and report['requested_warmup'] == 1
    assert report['edge_budget'] == 4 and config.read_bytes() == original


def test_sweep_retests_edge_pressure_and_packs_diagnostics_without_training_inputs(tmp_path, monkeypatch):
    module = load('benchmark_batch_capacity')
    state = tmp_path/'artifacts/loss_balance_4090_seed42/status.json'
    state.parent.mkdir(parents=True)
    state.write_text(json.dumps({'state': 'complete'}))
    config = tmp_path/'configs/main15/mp20_all.yaml'
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({'train_files': ['train.jsonl']}))
    (tmp_path/'train.jsonl').write_text('{"id": "a"}\n{"id": "b"}\n')
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'SIZES', {'mp20_all': [4]})
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **kw: SimpleNamespace(stdout=''))
    calls = []
    def benchmark(config, pool, folder, size, budget, mode, environment):
        calls.append((budget, mode))
        return {'batch_size': size, 'edge_budget': budget, 'batch_sizes': [1, 1],
            'batch_edges': [120000, 120000], 'measured_records': 2,
            'records_per_second': 100., 'seconds_per_step': .02,
            'peak_cuda_bytes': 1, 'peak_cuda_reserved_bytes': 2}
    monkeypatch.setattr(module, 'benchmark', benchmark)
    monkeypatch.setattr(sys, 'argv', ['benchmark_batch_capacity.py', '--output', 'artifacts/benchmarks/test', '--run'])
    module.main()
    assert calls == [(200000, 'random'), (200000, 'largest'), (400000, 'random'), (400000, 'largest')]
    report = json.loads((tmp_path/'artifacts/benchmarks/test/summary.json').read_text())
    assert not report['formal_training_started'] and len(report['rows']) == 2
    assert len({r['pool_sha256'] for r in report['rows']}) == 1
    with zipfile.ZipFile(tmp_path/'artifacts/benchmarks/test.zip') as archive:
        assert any(name.endswith('summary.json') for name in archive.namelist())
        assert not any(name.endswith(('.jsonl', '.pt')) for name in archive.namelist())
