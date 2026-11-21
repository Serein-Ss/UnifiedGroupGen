"""Generative backbones share the same state, objective and event sampler."""
from dataclasses import replace

import pytest
import torch

from unifiedgroupgen.common.data_utils import target_from_record
from unifiedgroupgen.experiment_config import compose_experiment, training_definition
from unifiedgroupgen.generation.unified.sampling import sample_batch
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl
from unifiedgroupgen.group_constraints.flow_state import initial
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.pl_data.unified.path import sample_path
from unifiedgroupgen.pl_data.unified.tasks import task_target
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_modules.unified.objectives import loss
from unifiedgroupgen.tests.unit.test_joint_flow_v2 import config, record


BACKBONES = ('cspnet', 'equiformer_v3', 'esen', 'mace')


@pytest.fixture(autouse=True)
def torch_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


def definition(backbone, task):
    cfg = config(backbone, task)
    cfg['allowed_elements'] = sorted({
        8, *(z for dataset in ('mp_20', 'c2db_51')
             for z in target_from_record(record(dataset)).state.elements),
    })
    cfg['model'].update(radial_basis=8, heads=4, lmax=2, mmax=2)
    return cfg


@pytest.mark.parametrize('backbone', BACKBONES)
def test_mixed_space_layer_dng_csp_forward_and_backward(backbone):
    torch.manual_seed(9)
    cfg = definition(backbone, 'dng')
    model = build_joint_model(cfg)
    samples = []
    for kind in ('mp_20', 'c2db_51'):
        for task in ('dng', 'csp'):
            target = task_target(target_from_record(record(kind)), definition(backbone, task), 7)
            samples.append(sample_path(target, 7, time=.87, allowed_elements=cfg['allowed_elements']))
    parts = loss(model, samples)
    parts['loss'].backward()
    assert all(torch.isfinite(value) for value in parts.values())
    assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
               for parameter in model.backbone.parameters())
    assert all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
               if parameter.grad is not None)


@pytest.mark.parametrize('backbone', BACKBONES)
@pytest.mark.parametrize('dataset', ['mp_20', 'c2db_51'])
def test_backbone_frame_equivariance_and_empty_graph(backbone, dataset):
    torch.manual_seed(6)
    cfg = definition(backbone, 'csp')
    model = build_joint_model(cfg).eval()
    target = task_target(target_from_record(record(dataset)), cfg, 11)
    state = sample_path(target, 11, time=1., allowed_elements=cfg['allowed_elements']).state
    empty = initial(state.spec, chemistry=state.chemistry)
    rotation, _ = torch.linalg.qr(torch.randn(3, 3))
    rotation[:, 0] *= torch.linalg.det(rotation).sign()
    transformed = replace(state, frame=rotation, origin=torch.tensor([.2, -.1, .5]))
    context = torch.randn(1, cfg['model']['hidden'])
    with torch.no_grad():
        vectors, features = model.backbone.forward_states([state], context)
        moved, moved_features = model.backbone.forward_states([empty, transformed], context.repeat(2, 1))
    torch.testing.assert_close(moved_features, features, rtol=3e-4, atol=1e-4)
    expected = vectors if backbone == 'cspnet' else vectors @ rotation.T
    torch.testing.assert_close(moved, expected, rtol=3e-4, atol=1e-4)


@pytest.mark.parametrize('backbone', BACKBONES)
@pytest.mark.parametrize('task', ['dng', 'csp'])
def test_configs_and_event_generation_from_empty(backbone, task):
    cfg = training_definition(compose_experiment('full_mp20_eqv2_' + task,
                    overrides=[f'model=joint_flow_{backbone}']))
    assert cfg['model']['backbone'] == backbone and cfg['generation_task'] == task
    torch.manual_seed(16)
    model = build_joint_model(definition(backbone, task)).eval()
    chemistry = ChemicalControl.from_inputs(counts={8: 1}) if task == 'csp' else ChemicalControl()
    results = sample_batch(model, [catalog_spec('space', 1)], seeds=[8], steps=3,
                           budget=6, epsilon=.005, chemistries=[chemistry])
    assert len(results) == 1
    result = results[0]
    assert result.events and result.events[0]['atoms_before'] == 0
    assert result.state.spec == catalog_spec('space', 1)
    if task == 'csp':
        assert all(z == 8 for z in result.state.elements)
        assert result.state.num_atoms <= 1
