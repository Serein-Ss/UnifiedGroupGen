import copy
import json
import random
from types import SimpleNamespace

import pytest
import torch

from unifiedgroupgen.cli import build_models
from unifiedgroupgen.loss_diagnostics import gradient_diagnostics
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.training import joint_loss, run_training


def models_and_records():
    config = {'groups': [('space', 2), ('layer', 37)], 'properties': {},
              'model': {'hidden': 8, 'proposal_layers': 1, 'flow_layers': 1,
                        'heads': 2, 'frequencies': 1}, 'max_orbits': 4}
    proposal, flow = build_models(config, 'cpu')
    records = []
    for i, descriptor in enumerate((Descriptor('space', 2, (OrbitSpec('i', 14),)),
                                    Descriptor('layer', 37, (OrbitSpec('r', 8),)))):
        compiled = compile_descriptor(descriptor)
        records.append({'id': str(i), 'dataset': descriptor.kind,
                        'descriptor': descriptor.to_dict(), 'state': compiled.prior().tolist(),
                        'properties': {}})
    return config, proposal, flow, records


def reset_rng():
    random.seed(42)
    torch.manual_seed(42)


def test_fixed_length_scale_changes_only_coordinate_objective_and_gradient():
    reset_rng()
    _, proposal, flow, records = models_and_records()
    reset_rng()
    raw = joint_loss(proposal, flow, records, 0, 0)
    reset_rng()
    normalized = joint_loss(proposal, flow, records, 0, 0, 3.0)
    for key in ('descriptor', 'lattice', 'coordinates'):
        assert torch.equal(raw[key], normalized[key])
    assert torch.allclose(normalized['flow'], raw['lattice']+raw['coordinates']/9)
    parameters = list(flow.parameters())
    actual = torch.autograd.grad(normalized['flow'].mean(), parameters, retain_graph=True, allow_unused=True)
    expected = torch.autograd.grad((raw['lattice']+raw['coordinates']/9).mean(), parameters, allow_unused=True)
    for a, b in zip(actual, expected):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.allclose(a, b, atol=1e-6, rtol=1e-5)


def test_gradient_diagnostic_is_observational_and_reconstructs_total():
    reset_rng()
    _, proposal, flow, records = models_and_records()
    losses = joint_loss(proposal, flow, records, 0, 0, 3.0)
    state, python_state = torch.get_rng_state().clone(), random.getstate()
    weights = {'descriptor': 0.7, 'flow': 1.2}
    report = gradient_diagnostics(proposal, flow, losses, weights, 3.0)
    assert torch.equal(state, torch.get_rng_state()) and python_state == random.getstate()
    parameters = list(proposal.parameters())+list(flow.parameters())
    assert all(p.grad is None for p in parameters)
    objective = weights['descriptor']*losses['descriptor'].mean()+weights['flow']*losses['flow'].mean()
    objective.backward()
    expected = torch.sqrt(sum(p.grad.square().sum() for p in parameters if p.grad is not None))
    assert report['reconstructed_total_preclip_norm'] == pytest.approx(float(expected), rel=2e-5)
    assert report['coordinates_weighted_norm'] == pytest.approx(1.2*report['coordinates_raw_norm']/9)


def test_zero_coordinate_dof_is_finite():
    reset_rng()
    _, proposal, flow, records = models_and_records()
    fixed = Descriptor('space', 2, (OrbitSpec('a', 14),))
    compiled = compile_descriptor(fixed)
    records[0].update(descriptor=fixed.to_dict(), state=compiled.prior().tolist())
    losses = joint_loss(proposal, flow, records[:1], 0, 0, 3)
    assert losses['coordinates'].item() == 0
    report = gradient_diagnostics(proposal, flow, losses, {'descriptor': 1, 'flow': 1}, 3)
    assert report['coordinates_raw_norm'] == 0
    assert report['shared_lattice_coordinate_cosine'] is None


def test_diagnostics_preserve_optimizer_updates_and_log_raw_and_normalized(tmp_path):
    reset_rng()
    config, proposal, flow, records = models_and_records()
    original = (copy.deepcopy(proposal.state_dict()), copy.deepcopy(flow.state_dict()))
    config.update(batch_size=2, epochs=2, validation_interval=2, coordinate_length_scale=3.0,
                  composition_probability=0, observation_probability=0, learning_rate=0.001)
    train, val = tmp_path/'train.jsonl', tmp_path/'val.jsonl'
    train.write_text('train'); val.write_text('val')
    args = SimpleNamespace(device='cpu', resume=None, batch_size=None, epochs=None)
    checkpoints = []
    for interval in (0, 1):
        proposal.load_state_dict(original[0]); flow.load_state_dict(original[1]); reset_rng()
        cfg = {**config, 'gradient_diagnostics_interval': interval}
        out = tmp_path/f'diagnostics_{interval}.pt'
        run_training(args, cfg, records, records, proposal, flow, [train], [val], out)
        checkpoints.append(torch.load(out, weights_only=False))
    for key in ('proposal', 'flow'):
        for name in checkpoints[0][key]:
            assert torch.equal(checkpoints[0][key][name], checkpoints[1][key][name])
    history = [json.loads(s) for s in (tmp_path/'diagnostics_1.history.jsonl').read_text().splitlines()]
    for row in history:
        assert row['train']['coordinates_dimensionless'] == pytest.approx(row['train']['coordinates']/9)
        assert row['train']['flow'] == pytest.approx(row['train']['lattice']+row['train']['coordinates']/9)
        assert 0 <= row['gradient_clipped_fraction'] <= 1
    assert len((tmp_path/'diagnostics_1.gradients.jsonl').read_text().splitlines()) == 2


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_invalid_length_scale_is_rejected(scale):
    _, proposal, flow, records = models_and_records()
    with pytest.raises(ValueError, match='coordinate_length_scale'):
        joint_loss(proposal, flow, records, 0, 0, scale)
