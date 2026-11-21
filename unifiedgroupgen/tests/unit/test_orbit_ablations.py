"""Paired orbit-ablation controls and real no-training execution paths."""
from collections import Counter

import pytest
import torch
from omegaconf import OmegaConf

from unifiedgroupgen.common.data_utils import target_from_record
from unifiedgroupgen.experiment_config import compose_experiment, training_definition
from unifiedgroupgen.generation.unified.sampling import sample_one
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl
from unifiedgroupgen.group_constraints.flow_state import advance, assign
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.pl_data.unified.path import sample_path
from unifiedgroupgen.pl_data.unified.tasks import task_target
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_modules.unified.objectives import loss
from unifiedgroupgen.tests.unit.test_joint_flow_v2 import config, populated, record


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('dataset', ['mp20', 'c2db51', 'joint'])
@pytest.mark.parametrize('task', ['dng', 'csp'])
def test_hydra_ablation_cohort_keeps_data_backbone_and_budgets(dataset, task):
    experiment = f'full_{dataset}_eqv2_{task}'
    original = training_definition(compose_experiment(experiment))
    experiments = {}
    for selector, mode in [('control', 'joint'), ('fixed_orbits', 'fixed_orbits'),
                           ('no_geometry_feedback', 'no_geometry_feedback')]:
        group = compose_experiment(experiment, overrides=[f'ablation={selector}'])
        definition = training_definition(group)
        assert definition['experiment_id'] == f'{experiment}_{selector}'
        assert definition['orbit_ablation'] == mode
        assert definition['early_stopping'] is None
        assert definition['model']['backbone'] == 'equiformer_v2'
        expected = {key: value for key, value in original.items()
                    if key not in ('early_stopping', 'experiment_id')}
        actual = {key: value for key, value in definition.items()
                  if key not in ('early_stopping', 'experiment_id', 'orbit_ablation')}
        assert actual == expected
        experiments[selector] = OmegaConf.to_container(group.runtime, resolve=True)
    assert experiments['control'] == experiments['fixed_orbits'] == experiments['no_geometry_feedback']


@pytest.mark.parametrize('dataset,task', [('mp_20', 'dng'), ('c2db_51', 'csp')])
def test_fixed_teacher_preserves_edit_schedule_and_delays_geometry(dataset, task):
    cfg = config(task=task)
    endpoint = task_target(target_from_record(record(dataset)), cfg, 41)
    options = dict(allowed_elements=cfg['allowed_elements'], budget=24)
    start = sample_path(endpoint, 41, time=0, orbit_ablation='fixed_orbits', **options)
    freeze = start.trajectory[-1]['time']
    last_edit = start.trajectory[-2]['time']
    before_time = (last_edit + freeze) / 2
    joint = sample_path(endpoint, 41, time=before_time, **options)
    fixed = sample_path(endpoint, 41, time=before_time, orbit_ablation='fixed_orbits', **options)
    blind = sample_path(endpoint, 41, time=before_time, orbit_ablation='no_geometry_feedback', **options)
    assert fixed.trajectory == joint.trajectory == blind.trajectory
    assert fixed.state.letters == joint.state.letters and fixed.state.elements == joint.state.elements
    assert fixed.remaining == joint.remaining and fixed.next_action == joint.next_action
    assert not fixed.state.frozen
    torch.testing.assert_close(blind.state.coordinates, joint.state.coordinates)
    torch.testing.assert_close(fixed.state.metric, start.state.metric, atol=0, rtol=0)
    torch.testing.assert_close(fixed.metric_anchor, fixed.state.metric, atol=0, rtol=0)
    torch.testing.assert_close(fixed.coordinate_anchor, fixed.state.coordinates, atol=0, rtol=0)
    at_freeze = sample_path(endpoint, 41, time=freeze, orbit_ablation='fixed_orbits', **options)
    assert at_freeze.state.frozen
    torch.testing.assert_close(at_freeze.state.coordinates, fixed.state.coordinates, atol=0, rtol=0)
    finished = sample_path(endpoint, 41, time=1, orbit_ablation='fixed_orbits', **options)
    reference = sample_path(endpoint, 41, time=1, **options)
    assert finished.state.letters == reference.state.letters and finished.state.elements == reference.state.elements
    torch.testing.assert_close(finished.state.coordinates, reference.state.coordinates, atol=0, rtol=0)
    torch.testing.assert_close(finished.state.metric, reference.state.metric, atol=0, rtol=0)

    model = build_joint_model(dict(cfg, orbit_ablation='fixed_orbits')).eval()
    metrics = loss(model, [fixed])
    assert metrics['continuous'].item() == 0 and torch.isfinite(metrics['loss'])
    metrics = loss(model, [at_freeze])
    assert metrics['continuous'].item() > 0 and torch.isfinite(metrics['loss'])


@pytest.mark.parametrize('kind,number', [('space', 2), ('layer', 78)])
def test_no_geometry_feedback_is_invariant_to_live_lattice_and_positions(kind, number, monkeypatch):
    torch.manual_seed(17)
    model = build_joint_model(dict(config(), orbit_ablation='no_geometry_feedback')).eval()
    baseline = build_joint_model(config()).eval()
    baseline.load_state_dict(model.state_dict(), strict=True)
    assert model.backbone.__class__ == baseline.backbone.__class__
    state = populated(kind, number)
    other = advance(state, torch.ones_like(state.metric)*.7,
                    torch.linspace(-.2,.3,state.coordinates.numel()).reshape_as(state.coordinates), 1.)
    assert not torch.allclose(state.metric, other.metric)
    assert not torch.allclose(state.coordinates, other.coordinates)
    calls = 0
    forward = model.backbone.forward_states

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return forward(*args, **kwargs)

    monkeypatch.setattr(model.backbone, 'forward_states', counted)
    with torch.no_grad():
        a, b = model.forward_batch([state, other], [.3, .3])
        original_a, original_b = baseline.forward_batch([state, other], [.3, .3])
        chemical = model(assign(state, 0, 8), .3)
    assert calls == 2  # One backbone forward for each packed call.
    assert a['actions'] == b['actions'] == original_a['actions']
    assert {'birth','replace','freeze'} <= {action.kind for action in a['actions']}
    for key in ('action_logits', 'remaining', 'element_logits'):
        torch.testing.assert_close(a[key], b[key], atol=0, rtol=0)
    assert not torch.allclose(a['metric_velocity_scaled'], b['metric_velocity_scaled'])
    assert not torch.allclose(original_a['action_logits'], original_b['action_logits'])
    assert not torch.allclose(a['action_logits'], chemical['action_logits'])


def test_fixed_sampler_builds_own_scaffold_from_empty_before_geometry_flow():
    torch.manual_seed(5)
    model = build_joint_model(dict(config(task='csp'), orbit_ablation='fixed_orbits',
                                   max_atoms=1, max_orbits=1)).eval()
    trace = []

    def observe(index, stage, state, time, step, event, reason):
        trace.append((stage, state, event))

    result = sample_one(model, catalog_spec('space', 1), steps=8, epsilon=.005,
                        generator=torch.Generator().manual_seed(22),
                        chemistry=ChemicalControl.from_inputs(counts={8: 1}),
                        trajectory_callback=observe)
    assert trace[0][0] == 'initial' and trace[0][1].num_atoms == 0
    assert result.state.frozen and result.state.status == 'done'
    assert Counter(result.state.elements) == {8: 1}
    edits = [event['action']['kind'] for _, _, event in trace if event]
    assert edits[0] == 'birth' and edits[-1] == 'freeze' and edits.count('freeze') == 1
    assert len(edits) <= model.event_budget
    for cursor, (stage, state, _) in enumerate(trace):
        if not state.frozen:
            torch.testing.assert_close(state.metric, trace[0][1].metric, atol=0, rtol=0)
            if stage == 'continuous':
                torch.testing.assert_close(state.coordinates, trace[cursor-1][1].coordinates, atol=0, rtol=0)
    fixed = next(state for _, state, event in trace if event and event['action']['kind'] == 'freeze')
    assert fixed.letters == result.state.letters
    assert not torch.allclose(fixed.metric, result.state.metric)


@pytest.mark.parametrize('mode', ['fixed_orbits', 'no_geometry_feedback'])
def test_checkpoint_restores_ablation_with_the_same_parameter_structure(tmp_path, mode):
    from unifiedgroupgen.pl_modules.unified.module import checkpoint_format, load_joint_checkpoint

    settings = dict(config(), orbit_ablation=mode)
    model = build_joint_model(settings)
    path = tmp_path/'model.pt'
    torch.save(dict(format=checkpoint_format(settings), config=settings, model=model.state_dict()), path)
    checkpoint = load_joint_checkpoint(path)
    restored = build_joint_model(checkpoint['config'])
    restored.load_state_dict(checkpoint['model'], strict=True)
    assert restored.orbit_ablation == mode
    assert restored.state_dict().keys() == build_joint_model(config()).state_dict().keys()
