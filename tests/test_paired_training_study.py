"""The paired study must compare identical inputs and preserve failures/unknowns."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch


def study_module():
    path = Path(__file__).resolve().parents[1]/'scripts/paired_training_study.py'
    spec = importlib.util.spec_from_file_location('paired_training_study', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pairing_loads_shared_weights_before_optimizer_and_restores_rng(tmp_path, monkeypatch):
    module = study_module()
    proposal, flow = torch.nn.Linear(2, 2), torch.nn.Linear(2, 1)
    initial = {'proposal': {k: v.clone() for k, v in proposal.state_dict().items()},
               'flow': {k: v.clone() for k, v in flow.state_dict().items()}, 'rng': 'shared_rng'}
    expected_hash = module.weight_hash((proposal, flow))
    with torch.no_grad():
        proposal.weight.add_(3)
    calls = []
    monkeypatch.setattr(module, 'restore_rng', lambda value: calls.append(value))
    def original(*args):
        assert module.weight_hash((args[4], args[5])) == expected_hash
        assert calls == ['shared_rng']
        return 'trained'
    training = SimpleNamespace(run_training=original, batch_indices=lambda *a, **k: [[0]])
    module.instrument_training(training, initial, {'fitted_config': {'x': 1},
                               'initial_weights_sha256': expected_hash}, tmp_path/'trace.jsonl', False)
    assert training.run_training(None, {'x': 1}, [], [], proposal, flow) == 'trained'
    with pytest.raises(ValueError, match='configuration'):
        training.run_training(None, {'x': 2}, [], [], proposal, flow)


def test_batch_trace_does_not_log_validation_or_consume_rng(tmp_path, monkeypatch):
    module = study_module()
    monkeypatch.setattr(module, 'rng_hash', lambda: 'rng')
    training = SimpleNamespace(run_training=lambda *a: None, batch_indices=lambda *a, **k: [[1], [0]])
    trace = tmp_path/'pairing.jsonl'
    module.instrument_training(training, {}, {}, trace, True)
    records = [{'id': 'a', 'dataset': 'mp_20'}, {'id': 'b', 'dataset': 'mp_20'}]
    training.batch_indices(records, 1, 10, shuffle=False)
    assert not trace.exists()
    actual = training.batch_indices(records, 1, 10)
    saved = json.loads(trace.read_text())
    assert saved['batch_order_sha256'] == module.digest([[('mp_20', 'b')], [('mp_20', 'a')]])
    assert saved['records'] == 2 and actual == [[1], [0]]


def test_changed_input_or_frozen_code_blocks_run(tmp_path, monkeypatch):
    module = study_module()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    path = tmp_path/'input.jsonl'
    path.write_text('original')
    manifest = {'files': {'input.jsonl': module.file_hash(path)}, 'frozen_files': {},
                'controller_sha256': module.file_hash(module.__file__)}
    module.verify_inputs(tmp_path, manifest)
    path.write_text('changed')
    with pytest.raises(ValueError, match='input changed'):
        module.verify_inputs(tmp_path, manifest)


def test_resumed_training_preserves_checkpoint_rng_instead_of_resetting(tmp_path, monkeypatch):
    module = study_module()
    def forbidden(*args):
        raise AssertionError('Resuming must not restore initialization RNG')
    monkeypatch.setattr(module, 'restore_rng', forbidden)
    training = SimpleNamespace(run_training=lambda *args: 'resume', batch_indices=lambda *a, **k: [])
    module.instrument_training(training, {}, {'fitted_config': {}}, tmp_path/'trace', True)
    assert training.run_training(None, {}, [], [], None, None) == 'resume'


def test_unknown_identifications_and_failures_stay_in_rate_denominator():
    module = study_module()
    result = module.rate_summary([True, False, None, None])
    assert result['rate'] == .25 and result['attempted'] == 4
    assert result['unknown'] == 2
    assert result['wilson95'][0] < .25 < result['wilson95'][1]


def test_summary_handles_all_failed_generations_and_missing_coverage(tmp_path):
    module = study_module()
    history = [{'epoch': i, 'train': {'descriptor': 1., 'flow': 2.},
                'val': {'descriptor': 1., 'flow': 2.} if i == 1 else None,
                'val_loss': 3. if i == 1 else None} for i in range(2)]
    metric = {'attempted': 2, 'generated': 0, 'coverage': {'cov_recall': None, 'cov_precision': None},
              'rule_checks': {'target_group_containment': module.rate_summary([None, None])},
              'validity_intervals': {key: module.rate_summary([None, None])
                                     for key in ('joint_valid', 'exact_target_group')}}
    for key in ('joint_valid', 'exact_target_group', 'unique', 'novel_against_reference'):
        metric[key] = {'rate_over_all_attempts': 0.}
    for dataset in module.DATASETS:
        for version in ('before', 'after'):
            folder = tmp_path/'runs'/dataset/version
            folder.mkdir(parents=True)
            (folder/'model.history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
            (folder/'pairing.jsonl').write_text(''.join(json.dumps({'batch_order_sha256': str(i),
                'rng_sha256': str(i)})+'\n' for i in range(2)))
            module.write(tmp_path/'evaluations'/dataset/version/'complete.json', {'jobs': {'mp_20_unconditional': metric}})
    module.summarize(tmp_path, {'epochs': 2})
    report = json.loads((tmp_path/'comparison.json').read_text())
    assert report['datasets']['joint']['batch_order_and_epoch_rng_paired']
    assert (tmp_path/'figures/paired_generation_quality.png').read_bytes().startswith(b'\x89PNG')
