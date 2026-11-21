"""Nonmagnetic rod-group constraints and generation without training."""
import json
from math import sqrt

import pytest
import torch

from unifiedgroupgen.common.data_utils import radius_graph
from unifiedgroupgen.common.structures import load_structure
from unifiedgroupgen.generation.trajectory import TrajectoryWriter
from unifiedgroupgen.generation.unified.sampling import sample_batch
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl
from unifiedgroupgen.group_constraints.flow_state import JointAction, actions as legal_actions
from unifiedgroupgen.group_constraints.operations import compile_request, operation_spec
from unifiedgroupgen.group_constraints.unified import SymmetrySpec
from unifiedgroupgen.joint import evaluation_record


def test_cartesian_screw_closure_matches_catalog():
    # IT E No. 43, p31: a 120-degree rotation and an axial third of a cell.
    screw = dict(rotation=[[1, 0, 0], [0, -.5, -sqrt(3)/2], [0, sqrt(3)/2, -.5]],
                 translation=[1/3, 0, 0])
    spec = compile_request(dict(kind='rod', generators=[screw], periodic=[True, False, False]))
    assert spec == operation_spec('rod', 43)
    assert spec != operation_spec('rod', 44)
    assert spec == SymmetrySpec.from_dict(spec.to_dict())
    assert len(spec.operations) == 3
    assert [(w.multiplicity, w.dof) for w in spec.templates] == [(3, 3)]
    with pytest.raises(ValueError, match='Periodic mask'):
        compile_request(dict(kind='rod', generators=[screw], periodic=[False, False, True]))
    with pytest.raises(ValueError, match='one periodic axis'):
        SymmetrySpec.from_dict(dict(spec.to_dict(), periodic=[True, True, False]))


def test_rod_catalog_stabilizer_and_metric():
    from unifiedgroupgen.group_constraints.representation import audit_catalog
    rows = list(audit_catalog(kinds=('rod',)))
    assert len(rows) == 331
    assert all(row['passed'] and row['lattice_dof'] == 1 for row in rows)


def test_rod_graph_wraps_axial_images_only():
    # The second pair would be neighbors if the transverse Å coordinate wrapped.
    coords = torch.tensor([[.02, 0., 0.], [.98, 0., 0.], [.02, 1., 0.], [.02, 0., 1.]])
    cell = torch.diag(torch.tensor([3., 1., 1.]))[None]
    edges, shifts, neighbors, vectors = radius_graph(
        coords, cell, torch.tensor([4]), torch.tensor([[True, False, False]]), cutoff=.4)
    assert set(map(tuple, edges.T.tolist())) == {(0, 1), (1, 0)}
    assert neighbors.tolist() == [2]
    assert shifts[:, 1:].count_nonzero() == 0
    assert set(shifts[:, 0].tolist()) == {-1, 1}
    torch.testing.assert_close(vectors.norm(dim=1), torch.full((2,), .12))


class ScriptedRodHeads(torch.nn.Module):
    """Deterministic choices drive the real sampler through legal edits."""
    allowed_elements = (8, 14)
    max_atoms = 24
    max_orbits = 6
    event_budget = 8
    birth_scale = 2.

    def __init__(self, program):
        super().__init__()
        self.marker = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
        self.program = program

    def forward_batch(self, states, times, properties=None, actions=None):
        result = []
        for state, time in zip(states, times):
            support = legal_actions(state, self.allowed_elements, self.max_atoms, self.max_orbits)
            action_logits = state.metric.new_full((len(support),), -torch.inf)
            remaining = 0. if state.frozen else 30.
            if not state.frozen:
                action_logits[support.index(self.program[state.step])] = 0.
            element_logits = state.metric.new_full((len(state.letters), 119), -torch.inf)
            element_logits[:, 8] = 0.
            result.append(dict(actions=support, action_logits=action_logits,
                remaining=state.metric.new_tensor(remaining), element_logits=element_logits,
                metric_velocity_scaled=torch.full_like(state.metric, .1 * (1-time)),
                coordinate_velocity_scaled=state.coordinates * state.coordinates.new_tensor([0., .1, .1]) * (1-time)))
        return result


@pytest.mark.parametrize('task', ['dng', 'csp'])
def test_rod_sampler_from_empty_to_valid_cif(task, tmp_path):
    number = 42 if task == 'dng' else 43
    spec = operation_spec('rod', number)
    general = next(w.letter for w in spec.templates if w.dof == 3)
    if task == 'dng':
        axial = next(w.letter for w in spec.templates if w.dof == 1)
        program = (JointAction('birth', letter=axial), JointAction('replace', 0, general),
                   JointAction('split', 0, general), JointAction('delete', 1), JointAction('freeze'))
        chemistry = ChemicalControl()
    else:
        chemistry = ChemicalControl.from_inputs(counts={8: 3, 14: 3})
        program = (JointAction('birth', letter=general, element=8),
                   JointAction('birth', letter=general, element=14), JointAction('freeze'))
    metric_values = []
    with TrajectoryWriter(tmp_path, 1) as writer:
        writer.start(0, 21, spec, number, chemistry, {})
        def record(index, stage, state, time, step, event, reason):
            decoded = state.decoded()
            residual = state.compiled.residuals(decoded['coordinates'], decoded['atomic_numbers'], decoded['lattice'])
            assert residual['elements'] and residual['coordinates'] < 1e-8
            assert state.metric.shape == (1,)
            metric_values.append(float(state.metric[0]))
            writer.record(index, stage, state, time, step, event, reason)
        result, = sample_batch(ScriptedRodHeads(program), [spec], seeds=[21], steps=8,
                               chemistries=[chemistry], trajectory_callback=record)
        row = evaluation_record(result, 0, 21, requested_group=number)
        writer.finish(0, row)
    assert result.state.status == 'done' and result.state.frozen
    assert metric_values[-1] > metric_values[0]
    assert row['accepted'], row.get('validation')
    assert row['periodic'] == [True, False, False]
    assert row['orbit_labels'] == (['3b'] if task == 'dng' else ['3a', '3a'])
    assert row['current_counts'] == ({8: 3} if task == 'dng' else {8: 3, 14: 3})
    exported = load_structure(row, tmp_path)
    assert exported.lattice.pbc == spec.periodic
    assert exported.lattice.a == pytest.approx(float(result.state.decoded()['lattice'][0, 0]), abs=1e-7)
    from unifiedgroupgen.group_constraints.identification import validate_joint_state
    exact = validate_joint_state(result.state, exact=True)
    assert exact['satisfies_requested_operations']
    assert exact['exact_target_group'] is None and not exact['maximal_group_checked']
    records = [json.loads(line) for line in writer.path(0).read_text(encoding='utf-8').splitlines()]
    frames = [item for item in records if item['record_type'] == 'frame']
    assert frames[0]['stage'] == 'initial' and frames[0]['num_atoms'] == 0
    assert frames[-1]['stage'] == 'final' and frames[-1]['num_atoms'] == sum(row['current_counts'].values())
    assert [item['time'] for item in frames] == sorted(item['time'] for item in frames)
    structural = [event for event in result.events if event['action']['kind'] != 'element']
    assert [event['action']['kind'] for event in structural] == [action.kind for action in program]
    assert [event['atoms_after'] for event in structural] == ([1, 3, 6, 3, 3] if task == 'dng' else [3, 6, 6])
    if task == 'csp':
        assert all(min(item['remaining_inventory'].values()) >= 0 for item in frames)
