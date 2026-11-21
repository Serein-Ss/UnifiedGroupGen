"""Recovery stays auxiliary and exact/strict symmetry rates use all requests."""
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from unifiedgroupgen.evaluation.common import check_symmetry, specified_symmetry_summary
from unifiedgroupgen.experiment_config import compose_experiment, training_definition
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl, orbit_counts
from unifiedgroupgen.group_constraints.flow_state import JointAction, apply, initial
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.pl_data.unified.path import FlowSample
from unifiedgroupgen.pl_data.unified.recovery import recovery_example
from unifiedgroupgen.pl_modules.unified.objectives import loss


@pytest.mark.parametrize('task', ['csp', 'dng'])
def test_legal_recovery_does_not_edit_teacher_path(task):
    spec = catalog_spec('space', 2)
    w = next(w for w in spec.templates if w.dof)
    gen = torch.Generator().manual_seed(12)
    start = initial(spec, gen, 24)
    first = JointAction('birth', letter=w.letter, element=8)
    second = JointAction('birth', letter=w.letter, element=14)
    state = apply(start, first, gen)
    endpoint = apply(state, second, gen)
    counts = orbit_counts(spec, endpoint.letters, endpoint.elements)
    if task == 'csp':
        state = replace(state, chemistry=ChemicalControl.from_inputs(counts=counts))
    else:
        second = replace(second, element=0)
    sample = FlowSample(state, .4, endpoint.metric, state.coordinates, (8,), 2, second, {}, 'probe')
    result = recovery_example(sample, endpoint, 31, [8,14], 20, 12, .2)
    assert result is not None
    mutant = result
    assert mutant.correction in mutant.support and mutant.time == sample.time
    assert len(mutant.state.letters) == len(state.letters)+1
    restored = apply(mutant.state, mutant.correction)
    assert restored.letters == state.letters and restored.elements == state.elements
    torch.testing.assert_close(restored.metric, state.metric, atol=0, rtol=0)
    torch.testing.assert_close(restored.coordinates, state.coordinates, atol=0, rtol=0)
    if task=='csp':
        assert min(dict(mutant.state.chemistry.remaining(spec, mutant.state.letters, mutant.state.elements)).values()) >= 0

    logit = torch.tensor(0., requires_grad=True)
    calls = []
    def forward(states, times, properties, actions=None):
        calls.append(actions)
        support = (second,) if actions is None else actions[0]
        logits = logit * torch.arange(float(len(support)))
        return [dict(metric_velocity_scaled=state.metric * 0, coordinate_velocity_scaled=state.coordinates * 0,
                     actions=support, action_logits=logits, remaining=logit.exp())]
    model = SimpleNamespace(forward_batch=forward)
    baseline = loss(model, [sample])
    assert len(calls)==1 and 'orbit_recovery' not in baseline
    trained = loss(model, [sample], recovery=[result], recovery_weight=.1)
    torch.testing.assert_close(trained['loss'], baseline['loss']+.1*trained['orbit_recovery'])
    for key in ('continuous','element','jump'):
        torch.testing.assert_close(trained[key], baseline[key])
    trained['orbit_recovery'].backward()
    assert torch.isfinite(logit.grad) and logit.grad != 0


def test_default_weight_is_compatible_and_hydra_override_is_wired():
    name = 'full_mp20_eqv2_csp'
    base = training_definition(compose_experiment(name))
    changed = training_definition(compose_experiment(name, overrides=['train.orbit_recovery_weight=0.1']))
    assert 'orbit_recovery_weight' not in base
    assert changed.pop('orbit_recovery_weight')==.1
    assert changed == base


def test_cubic_exact_and_strict_lower_number():
    structure = Structure(Lattice.cubic(4), ['Si'], [[0,0,0]])
    for group, expected in ((221,'exact'), (1,'strict_supergroup')):
        row = dict(target_group=group, basic_valid=True)
        check_symmetry(row, {}, {}, structure, 'space', True)
        assert row['operation_relation']==expected
        assert row['target_operation_containment'] and row['valid_operation_containment']
        assert row['recovered_group']==221
    reflected = dict(rotation=(-np.eye(3)).tolist(),translation=[.5,0,0])
    row = dict(target_group=None,basic_valid=True)
    check_symmetry(row, {}, dict(operations=[reflected]), structure, 'space', True)
    assert row['operation_relation']=='violated' and not row['target_operation_containment']


def test_layer_export_restores_operation_origin():
    structure = Structure(Lattice.orthorhombic(3,4,25), ['Si'], [[0,0,.5]])
    mirrored = dict(rotation=np.diag([1,1,-1]).tolist(),translation=[0,0,4])
    identity = dict(rotation=np.eye(3).tolist(),translation=[0,0,0])
    row = dict(target_group=None,basic_valid=False)
    record = dict(symmetry=dict(operations=[identity,mirrored]),coordinates=[[0,0,2]])
    check_symmetry(row, record, {}, structure, 'layer', True)
    assert row['operation_relation']=='strict_supergroup'
    assert row['target_operation_containment'] and not row['valid_operation_containment']


def test_failed_and_unknown_requests_remain_in_denominator():
    rows = [dict(symmetry_requested=True,exact_target_group=True,strict_supergroup=False,
                 target_operation_containment=True,valid_operation_containment=True),
            dict(symmetry_requested=True,operation_relation='unresolved',target_operation_containment=True),
            dict(symmetry_requested=True,exact_target_group=False,target_operation_containment=False)]
    report = specified_symmetry_summary(rows, True)
    assert report['attempts']==3 and report['unresolved']==1
    assert report['rates']['exact_target_group']['rate']==pytest.approx(1/3)
    assert report['rates']['target_operation_containment']['rate']==pytest.approx(2/3)




def _finished_orbit(group, free):
    from unifiedgroupgen.generation.types import JointSample

    spec = catalog_spec('space', group)
    state = initial(spec, torch.Generator().manual_seed(25), dtype=torch.float64)
    template = next(orbit for orbit in spec.templates if orbit.dof == 3)
    basis, offset = map(state.metric.new_tensor, template.arrays())
    coordinates = offset + basis @ state.metric.new_tensor(free)
    state = replace(state, metric=torch.zeros_like(state.metric), letters=(template.letter,),
                    elements=(14,), coordinates=coordinates, frozen=True, status='done')
    return JointSample(state, 1., [], 'complete', 0, 1e-4)


def test_cif_acceptance_allows_strict_supergroup():
    from unifiedgroupgen.common.structures import load_structure
    from unifiedgroupgen.joint import evaluation_record

    sample = _finished_orbit(1, [0., 0., 0.])
    row = evaluation_record(sample, 0, 25, requested_group=1, verify_cif_operations=True)
    assert row['accepted'] and row['rejection_reasons'] == []
    assert row['validation']['cif_operation_check']['target_operation_containment']
    result = dict(target_group=1, basic_valid=True)
    check_symmetry(result, row, {}, load_structure(row, '.'), 'space', True)
    assert result['operation_relation'] == 'strict_supergroup'


def test_cif_acceptance_rejects_readback_operation_violation(monkeypatch):
    from unifiedgroupgen.common.structures import load_structure
    from unifiedgroupgen.joint import evaluation_record

    def translated_site(*args):
        restored = load_structure(*args)
        restored.translate_sites([0], [.07, 0., 0.])
        return restored

    monkeypatch.setattr('unifiedgroupgen.common.structures.load_structure', translated_site)
    sample = _finished_orbit(2, [.16, .19, .23])
    row = evaluation_record(sample, 0, 25, requested_group=2, verify_cif_operations=True)
    assert row['status'] == 'rejected' and not row['accepted']
    assert row['validation']['satisfies_requested_operations']
    assert row['validation']['structure_valid']
    assert not row['validation']['cif_operation_check']['target_operation_containment']
    assert row['rejection_reasons'] == ['operation_violation']


def test_cif_acceptance_reports_geometry_rejection_separately(tmp_path):
    import json
    from unifiedgroupgen.evaluate import evaluate
    from unifiedgroupgen.joint import evaluation_record

    sample = _finished_orbit(2, [.04, 0., 0.])
    row = evaluation_record(sample, 0, 25, requested_group=2, verify_cif_operations=True)
    assert row['status'] == 'rejected' and not row['accepted']
    assert row['validation']['cif_operation_check']['target_operation_containment']
    assert not row['validation']['structure_valid']
    assert row['rejection_reasons'] == ['structure_invalid']

    ledger = tmp_path / 'samples.jsonl'
    ledger.write_text(json.dumps(row) + "\n", encoding="utf-8")
    _, report, _ = evaluate(ledger, 'dng', 'space', tmp_path / 'evaluate',
                            use_fingerprints=False, expected_attempts=1)
    rates = report['specified_symmetry']['rates']
    assert rates['target_operation_containment']['passed'] == 1
    assert rates['structure_valid']['passed'] == 0
    assert rates['strict_supergroup']['passed'] == 0
    assert all(rate['total'] == 1 for rate in rates.values())



def test_contact_repair_preserves_a_layer_mirror_through_cif_readback():
    from unifiedgroupgen.generation.types import JointSample
    from unifiedgroupgen.generation.unified.contact_repair import repair_short_contacts
    from unifiedgroupgen.group_constraints.operations import compile_request
    from unifiedgroupgen.joint import evaluation_record

    spec = compile_request(dict(kind='layer', generators=[
        dict(rotation=np.diag([1, 1, -1]).tolist(), translation=[0., 0., 2.])]))
    state = initial(spec, torch.Generator().manual_seed(7), dtype=torch.float32)
    template = next(w for w in spec.templates if w.multiplicity == 2)
    coordinates = state.metric.new_tensor([[.13, .22, .985], [.13, .22, 1.015]])
    state = replace(state, metric=torch.zeros_like(state.metric), letters=(template.letter,),
                    elements=(14,), coordinates=coordinates, frozen=True, status='done')
    source = state.coordinates.clone()
    rng = torch.random.get_rng_state()
    repaired, report = repair_short_contacts(state)
    assert report['applied'] and report['initial_pair_distance_angstrom'] < .6
    assert report['final_pair_distance_angstrom'] >= .6
    assert repaired.spec is state.spec and repaired.metric is state.metric
    assert repaired.letters == state.letters and repaired.elements == state.elements
    assert torch.equal(state.coordinates, source) and torch.equal(torch.random.get_rng_state(), rng)
    row = evaluation_record(JointSample(repaired, 1., [], 'complete', 0, 1e-4), 0, 1,
                            verify_cif_operations=True)
    assert row['accepted'] and row['validation']['cif_operation_check']['target_operation_containment']
    unchanged, no_report = repair_short_contacts(repaired)
    assert unchanged is repaired and no_report is None


def test_contact_repair_leaves_short_periodic_self_image_rejected():
    from unifiedgroupgen.generation.unified.contact_repair import repair_short_contacts

    sample = _finished_orbit(1, [0., 0., 0.])
    state = sample.state
    metric = state.metric.new_tensor([-6., -6., -6., 0., 0., 0.])
    state = replace(state, metric=metric)
    repaired, report = repair_short_contacts(state)
    assert repaired is state and not report['applied']
    assert report['outcome'] == 'periodic_self_contact'



def test_layer_recheck_does_not_wrap_near_zero_out_of_plane():
    lattice = Lattice.from_parameters(1.59300590, 1.59300592, 20., 90., 90., 119.99999958)
    # CIF-equivalent fractional coordinates for a central site and a reflected pair.
    structure = Structure(lattice, ['Re', 'Re', 'C'],
                          [[.66666669, .33333334, .54918454],
                           [.33333334, .66666669, .45081546], [0., 0., .5]])
    row = dict(target_group=72, basic_valid=True)
    check_symmetry(row, {}, {}, structure, 'layer', False)
    assert row['target_operation_containment']
    assert row['operation_residual_angstrom'] < 1e-6
