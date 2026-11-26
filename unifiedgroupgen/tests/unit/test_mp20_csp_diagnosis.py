import json
from dataclasses import replace

import torch

from scripts.mp20_csp_diagnose import (
    evaluate_arm, initialized_orbits, load_conditioned_sampler, select_group,
)
from unifiedgroupgen.common.data_utils import target_from_record
from unifiedgroupgen.evaluation.joint_csp import prepare_csp_manifest, target_view
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl
from unifiedgroupgen.group_constraints.flow_state import apply, initial
from unifiedgroupgen.group_constraints.state import JointAction
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.tests.unit.test_joint_flow_v2 import record


class ConstantVelocity(torch.nn.Module):
    event_budget = 24
    birth_scale = .2

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward_batch(self, states, times, properties, actions):
        return [dict(metric_velocity_scaled=torch.ones_like(state.metric)*(1-t),
                     coordinate_velocity_scaled=torch.ones_like(state.coordinates)*(1-t),
                     remaining=0.) for state, t in zip(states, times)]


def test_group_allocation_pairs_first_candidate_and_does_not_wrap_wide():
    ranks = tuple(range(20))
    assert [select_group('blind_wide', ranks, k, 4) for k in range(20)] == list(ranks)
    assert [select_group('blind_top4_repeat', ranks, k, 4) for k in range(20)] == list(range(4))*5
    assert select_group('blind_wide', ranks[:2], 2, 4) is None
    assert select_group('blind_top4_repeat', ranks[:2], 2, 4) == 0


def test_orbit_conditioning_and_reference_metric_leave_positions_free():
    model = ConstantVelocity().eval()
    chemistry = ChemicalControl.from_inputs(counts={14: 1})
    spec = catalog_spec('space', 1)
    target = initial(spec, chemistry=chemistry)
    target = apply(target, JointAction('birth', letter=spec.templates[0].letter, element=14))
    target = replace(target, metric=torch.zeros_like(target.metric))
    rngs = [torch.Generator().manual_seed(23) for _ in range(2)]
    free, locked = [initialized_orbits(target, chemistry, model, rng, lock)
                    for rng, lock in zip(rngs, (False, True))]
    assert free.frozen and locked.frozen and locked.letters == target.letters
    assert torch.equal(free.coordinates, locked.coordinates)
    assert not torch.equal(locked.coordinates, target.coordinates)
    sampler = load_conditioned_sampler()
    result = sampler(model, [spec], generators=[rngs[1]], initial_states=[locked],
                     lock_metric=True, chemistries=[chemistry], steps=2, epsilon=.05)[0]
    assert result.state.status == 'done' and result.events == []
    assert torch.equal(result.state.metric, target.metric)
    assert not torch.equal(result.state.coordinates, locked.coordinates)


def test_matched_only_rmsd_and_fixed_failure_denominators(tmp_path):
    ref = record('mp_20')
    manifest = prepare_csp_manifest([ref], 20)
    view = target_view(ref)
    state = target_from_record(ref).state
    assert state.spec.kind == 'space'
    path = tmp_path / 'samples.jsonl'
    with path.open('w', encoding='utf-8') as f:
        for k in range(20):
            f.write(json.dumps(dict(view, dataset='mp_20', case_id=ref['id'], index=k, candidate=k,
                case_manifest_sha256=manifest.digest, status='generated' if k == 2 else 'failed'))+'\n')
    metrics, hits = evaluate_arm(path, [ref], manifest, {1: 2., 20: 40.})
    assert metrics['1']['match_rate'] == 0. and metrics['1']['matched_only_normalized_rmse'] is None
    assert metrics['1']['candidate_failure_rate'] == metrics['1']['case_no_output_rate'] == 1.
    assert metrics['20']['match_rate'] == 1. and metrics['20']['matched_only_normalized_rmse'] < 1e-5
    assert metrics['20']['candidate_failure_rate'] == .95 and metrics['20']['case_no_output_rate'] == 0.
    assert metrics['1']['sampling_seconds'] == 2. and metrics['20']['sampling_seconds'] == 40.
    assert hits[1] == set() and hits[20] == {ref['id']}
