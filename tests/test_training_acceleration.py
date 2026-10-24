"""Compare optimized arithmetic with the published dense equations and gradients."""

import copy
import importlib.util
import json
from pathlib import Path
import random
import sys
import pytest
import torch
import yaml
from unifiedgroupgen.cli import build_models, fit_training_config
from unifiedgroupgen.conditioning import attach_dataset_domains
from unifiedgroupgen.constraints import AffineObservation
from unifiedgroupgen.flow import OrbitFlow
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.training import batch_indices, joint_loss, prepare_training_records


def script(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('device,dtype', [('cpu', torch.float64), ('cpu', torch.float32),
                                         ('cuda', torch.float64), ('cuda', torch.float32),
                                         ('cuda', torch.bfloat16)])
@pytest.mark.parametrize('checkpoint', [False, True])
def test_full_velocity_and_all_gradients_match_dense_reference(device, dtype, checkpoint):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA BF16 unavailable')
    torch.manual_seed(42)
    charts = [compile_descriptor(d) for d in [
        Descriptor('space', 1, tuple(OrbitSpec('a', 14) for _ in range(5))),
        Descriptor('space', 136, (OrbitSpec('a', 22), OrbitSpec('f', 8))),
        Descriptor('layer', 37, (OrbitSpec('r', 14), OrbitSpec('r', 8))),
        Descriptor('plane', 1, (OrbitSpec('a', 6),)),
        Descriptor('space', 225, (OrbitSpec('a', 14),))]]
    geometry_dtype = torch.float32 if dtype == torch.bfloat16 else dtype
    states = [c.prior(device, geometry_dtype).requires_grad_() for c in charts]
    other_states = [z.detach().clone().requires_grad_() for z in states]
    properties = [{'gap': 1.0}, {'gap': None}, {'gap': .5}, {'gap': 0.0}, {'gap': 2.0}]
    model = OrbitFlow({'gap': {}}, hidden=16, layers=2, frequencies=3, cond_dropout=0,
                      checkpoint_layers=checkpoint, edge_chunk_size=7).to(device=device, dtype=geometry_dtype)
    reference = copy.deepcopy(model)
    script('benchmark').install_reference(reference)
    # The empty observation reproduces the old per-graph zero-feature path.
    observations = [None, AffineObservation.fixed(charts[1].dof, [0], [0.2]), None, None, None]
    reference_observations = [o or AffineObservation.fixed(c.dof, [], []) for c, o in zip(charts, observations)]
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        actual = model.forward_batch(charts, states, [.3]*len(charts), properties, observations=observations)
        expected = reference.forward_batch(charts, other_states, [.3]*len(charts), properties,
                                           observations=reference_observations)
        actual_loss = sum(v.square().mean() for v in actual)
        expected_loss = sum(v.square().mean() for v in expected)
    actual_loss.backward(); expected_loss.backward()
    if dtype == torch.float64:
        atol, rtol = 1e-10, 1e-8
    elif dtype == torch.bfloat16:
        # Reassociated BF16 GEMMs round at different points; test bounded error,
        # not bitwise trajectory identity. Geometry still stays FP32.
        atol, rtol = 2e-3, .05
    else:
        atol, rtol = 1e-5, 2e-4
    def compare(a, b):
        assert torch.isfinite(a).all() and torch.isfinite(b).all()
        assert (a-b).abs().max() <= atol+rtol*b.abs().max()
    for a, b in zip(actual, expected):
        compare(a, b)
    for a, b in zip(states, other_states):
        compare(a.grad, b.grad)
    assert list(model.state_dict()) == list(reference.state_dict())
    for (name, a), (_, b) in zip(model.named_parameters(), reference.named_parameters()):
        if a.grad is not None or b.grad is not None:
            compare(torch.zeros_like(a) if a.grad is None else a.grad,
                    torch.zeros_like(b) if b.grad is None else b.grad)


def test_factorization_failures_are_not_silently_ignored(monkeypatch):
    c = compile_descriptor(Descriptor('space', 225, (OrbitSpec('a', 14),)))
    model = OrbitFlow({}, hidden=8, layers=1, frequencies=1)
    original = torch.linalg.cholesky_ex
    def failing(*args, **kwargs):
        factor, info = original(*args, **kwargs)
        return factor, info+1
    monkeypatch.setattr(torch.linalg, 'cholesky_ex', failing)
    with pytest.raises(torch.linalg.LinAlgError, match='factorization'):
        model(c, c.prior(), .3)


def rows():
    records = []
    for i, (dataset, kind, element, value) in enumerate([
            ('mp_20', 'space', 14, 0.), ('mp_20', 'space', 8, 2.),
            ('c2db_51', 'layer', 14, 100.), ('c2db_51', 'layer', 8, 104.)]):
        d = Descriptor(kind, 1, (OrbitSpec('a', element),))
        records.append({'id': str(i), 'dataset': dataset, 'descriptor': d.to_dict(),
                        'state': compile_descriptor(d).prior().tolist(), 'properties': {'gap': value}})
    return records


def test_prepared_records_preserve_losses_batch_order_and_rng():
    config = {'kind': 'space', 'kinds': ['space', 'layer'], 'groups': [('space', 1), ('layer', 1)],
              'properties': {'gap': {}}, 'model': {'hidden': 8, 'proposal_layers': 1,
              'flow_layers': 1, 'heads': 2, 'frequencies': 1}, 'max_orbits': 4}
    proposal, flow = build_models(config, 'cpu')
    records = rows(); cached = prepare_training_records(records)
    def run(data):
        random.seed(42); torch.manual_seed(42)
        batches = batch_indices(data, 2, 100)
        losses = joint_loss(proposal, flow, data, 0, 0)
        return batches, losses, random.getstate(), torch.get_rng_state()
    a, b = run(records), run(cached)
    assert a[0] == b[0] and a[2] == b[2] and torch.equal(a[3], b[3])
    assert all(torch.equal(a[1][key], b[1][key]) for key in a[1])
    bad = copy.deepcopy(records); bad[0]['state'][0] = float('nan')
    with pytest.raises(ValueError, match='Invalid training state'):
        prepare_training_records(bad)


def test_benchmark_uses_train_scalers_domains_and_main15_probabilities(tmp_path, monkeypatch):
    module = script('benchmark')
    config = {'dataset': 'joint', 'kind': 'space', 'groups': [['space', 1], ['layer', 1]],
              'dataset_domains': {'mp_20': 0, 'c2db_51': 1},
              'properties': {'gap': {'per_domain': True},
                             'source_domain': {'type': 'categorical', 'context': True, 'num_classes': 2}},
              'composition_probability': 0., 'observation_probability': 0., 'precision': 'bf16',
              'batch_size': 2, 'max_atoms': 8, 'max_orbits': 4, 'edge_budget': 100,
              'model': {'hidden': 8, 'proposal_layers': 1, 'flow_layers': 1, 'heads': 2, 'frequencies': 1},
              'train_files': ['train.jsonl'], 'val_files': ['val.jsonl']}
    data = rows()
    for split in ['train', 'val']:
        (tmp_path/f'{split}.jsonl').write_text('\n'.join(json.dumps(r) for r in data))
    timing = copy.deepcopy(data[:1]); timing[0]['properties']['gap'] = 999
    (tmp_path/'timing.jsonl').write_text(json.dumps(timing[0]))
    (tmp_path/'config.yaml').write_text(yaml.safe_dump(config))
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'root_path', lambda p: tmp_path/p)
    monkeypatch.setattr(module, 'artifact_path', lambda p: tmp_path/p)
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--config', 'config.yaml', '--records', 'timing.jsonl',
        '--device', 'cpu', '--steps', '1', '--warmup', '1', '--output', 'report.json'])
    module.main()
    report = json.loads((tmp_path/'report.json').read_text())
    fitted = fit_training_config(config, attach_dataset_domains(data, config))
    assert report['properties'] == fitted['properties']
    assert report['allowed_elements'] == [8, 14]
    assert report['properties']['gap']['domain_scalers']['0']['shift'] == 1
    assert report['composition_probability'] == report['observation_probability'] == 0
    assert report['precision'] == 'fp32' and report['requested_precision'] == 'bf16'
    assert report['record_ids'] == [['0']]
    assert report['train_split_sha256'] != report['timing_pool_sha256']
    assert not list(tmp_path.glob('*.pt'))


def test_main15_forwards_runtime_options_without_mutating_configs(monkeypatch, capsys):
    module = script('train_main15')
    monkeypatch.setattr(sys, 'argv', ['train_main15', '--models', 'mp20_u', 'joint_all', '--plan',
                                    '--no-checkpoint', '--edge-chunk-size', '32768'])
    module.main()
    jobs = json.loads(capsys.readouterr().out)
    assert len(jobs) == 2
    for job in jobs:
        assert '--no-checkpoint' in job['command']
        assert job['command'][job['command'].index('--edge-chunk-size')+1] == '32768'
        assert job['command'][job['command'].index('--batch-size')+1] == '256'
        assert job['command'][job['command'].index('--edge-budget')+1] == '400000'
        assert job['command'][job['command'].index('--coordinate-length-scale')+1] == '3.0'
