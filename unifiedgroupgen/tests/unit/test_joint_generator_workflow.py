"""Actual sampler events, maximal nonmagnetic groups and versioned checkpoint recovery."""
from pathlib import Path
import pytest
import torch
from torch import nn
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.group_constraints.state import legal_actions, JointAction
from unifiedgroupgen.generation.joint import sample_joint
from unifiedgroupgen.group_constraints.identification import validate_joint_state, _plane_to_flat_layer
from unifiedgroupgen.pl_modules.joint_training import fit_joint, load_joint_checkpoint
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.joint import evaluation_record
from unifiedgroupgen.tests.unit.test_joint_generator import real_record, preserve_joint_process_state


class Oracle(nn.Module):
    """Known conditional generator, so sampler correctness is independent of training quality."""
    def __init__(self):
        super().__init__()
        self.parameter = nn.Parameter(torch.zeros((),dtype=torch.float64))
        self.allowed_elements, self.max_atoms, self.max_orbits = (14,), 16, 4

    def forward(self,state,time,properties=None,actions=None):
        options = legal_actions(state,self.allowed_elements,self.max_atoms,self.max_orbits) if actions is None else tuple(actions)
        desired = (JointAction('birth',letter='a') if state.step == 0 else
                   JointAction('element',orbit=0,element=14) if state.step == 1 else JointAction('end'))
        logits = state.metric.new_full((len(options),),-100.)
        if desired in options:
            logits[options.index(desired)] = 100.
        return dict(metric_anchor=torch.zeros_like(state.metric),
                    coordinate_anchor=torch.zeros_like(state.coordinates),moment_anchor=None,
                    remaining=state.metric.new_tensor(max(0,3-state.step) if state.status == 'active' else 0),
                    actions=options,action_logits=logits)


def test_event_sampler_stops_at_events_and_recovers_requested_maximal_group():
    result = sample_joint(Oracle(),catalog_spec('space',225),steps=16,budget=8,
                          generator=torch.Generator().manual_seed(12))
    assert result.state.status == 'done' and result.reason == 'complete'
    assert [entry['action']['kind'] for entry in result.events] == ['birth','element','end']
    assert all(a['time'] < b['time'] for a,b in zip(result.events,result.events[1:]))
    report = validate_joint_state(result.state,exact=True)
    assert report['satisfies_requested_operations'] and report['exact_target_group'] is True
    assert report['distance_valid']
    record = evaluation_record(result,0,12,exact=True)
    assert record['status'] == 'generated' and record['accepted'] and record['cif_sha256']


def test_no_end_policy_cannot_be_forced_into_success_at_budget_boundary():
    class NoEnd(Oracle):
        def forward(self,state,time,properties=None,actions=None):
            result = super().forward(state,time,properties,actions)
            result['remaining'] = state.metric.new_tensor(state.budget-state.step)
            result['action_logits'].fill_(-100.)
            desired = JointAction('birth',letter='a') if not state.num_atoms else JointAction('delete',orbit=0)
            if desired in result['actions']:
                result['action_logits'][result['actions'].index(desired)] = 100.
            return result
    result = sample_joint(NoEnd(),catalog_spec('space',225),steps=16,budget=6,
                          generator=torch.Generator().manual_seed(12))
    assert result.state.status == 'failed' and result.reason == 'event_budget'
    assert len(result.events) == 6 and result.events[-1]['action']['kind'] == 'fail'


def test_single_atom_periodic_self_image_distance_is_an_acceptance_condition():
    from dataclasses import replace
    result = sample_joint(Oracle(),catalog_spec('space',1),steps=16,budget=8,
                          generator=torch.Generator().manual_seed(12))
    metric = torch.zeros_like(result.state.metric)
    metric[[0,3,5]] = -6
    result.state = replace(result.state,metric=metric)
    report = validate_joint_state(result.state)
    assert result.state.num_atoms == 1 and report['satisfies_requested_operations']
    assert report['minimum_distance_angstrom'] < .6 and not report['distance_valid']
    record = evaluation_record(result,0,12)
    assert not record['accepted'] and record['status'] == 'rejected'


def test_plane_classification_is_bijective_for_all_seventeen_groups():
    mapping = _plane_to_flat_layer()
    assert len(mapping) == 17 and set(mapping.values()) == set(range(1,18))


@pytest.mark.parametrize('batch_size,prefix_steps',[(2,2),(1,1)])
def test_native_lightning_checkpoint_and_resume_keep_one_backbone(tmp_path,batch_size,prefix_steps):
    from pytorch_lightning import seed_everything
    config = dict(model=dict(family='joint_symmetry_v1',hidden=16,layers=1,heads=4),
                  allowed_elements=list(range(1,119)),properties={}, max_atoms=192,max_orbits=52,
                  event_budget=256,birth_scale=.2,cond_dropout=0.,learning_rate=1e-3)
    train = [real_record('mp_20'),real_record('c2db_51')]
    val = [real_record('mp_20','val'),real_record('c2db_51','val')]
    runtime = dict(batch_size=batch_size,edge_budget=200000)
    seed_everything(142,workers=True)
    uninterrupted = tmp_path/'uninterrupted/model.pt'
    fit_joint(config,train,val,uninterrupted,epochs=3,runtime=runtime,max_steps=3)
    expected = load_joint_checkpoint(uninterrupted)['model']
    seed_everything(142,workers=True)
    output = tmp_path/'model.pt'
    trainer = fit_joint(config,train,val,output,epochs=2,runtime=runtime,max_steps=prefix_steps)
    assert trainer.global_step == prefix_steps
    saved = load_joint_checkpoint(output)
    assert saved['config']['magnetic_training'] is False
    rebuilt = build_joint_model(saved['config'])
    rebuilt.load_state_dict(saved['model'],strict=True)
    native = load_joint_checkpoint(tmp_path/'last.ckpt')
    assert native['model'].keys() == saved['model'].keys()
    continued = fit_joint(config,train,val,output,epochs=3,runtime=runtime,max_steps=3,
                          resume=tmp_path/'last.ckpt')
    assert continued.global_step == 3
    actual = load_joint_checkpoint(output)['model']
    for name in expected:
        torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)
