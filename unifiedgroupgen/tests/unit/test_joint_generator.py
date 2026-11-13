"""Nonmagnetic unified constraints, actual mixed paths, gradients and legacy isolation."""
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
from unifiedgroupgen.group_constraints.unified import catalog_spec, compile_constraints, SymmetrySpec, Operation, OrbitTemplate
from unifiedgroupgen.group_constraints.state import initial_state, legal_actions, apply_action, JointAction
from unifiedgroupgen.pl_data.joint_path import JointTarget, target_from_record, sample_training_path
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_modules.objectives.joint_generator import joint_generator_loss
from unifiedgroupgen.common.configuration import compose_experiment, training_definition


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def preserve_joint_process_state():
    state = random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.get_num_threads()
    yield
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    torch.set_num_threads(state[3])


def real_record(dataset, split='train'):
    with (ROOT/'data/dataset/full_verified_v1'/dataset/(split+'.jsonl')).open(encoding='utf-8') as stream:
        return json.loads(next(stream))


def invariant(state, tolerance=2e-5):
    decoded = state.decoded()
    report = state.compiled.residuals(decoded['coordinates'], decoded['atomic_numbers'], decoded['lattice'])
    assert report['elements']
    assert report['coordinates'] < tolerance
    for op in state.spec.operations:
        r = decoded['lattice'].new_tensor(op.rotation)
        metric = decoded['lattice'].T@decoded['lattice']
        torch.testing.assert_close(r.T@metric@r, metric, atol=tolerance, rtol=tolerance)


@pytest.mark.parametrize('kind,count', [('space',230), ('layer',80), ('plane',17)])
def test_operator_compiler_covers_every_nonmagnetic_catalog_group(kind, count):
    generator = torch.Generator().manual_seed(91)
    for number in range(1, count+1):
        spec = catalog_spec(kind, number)
        assert SymmetrySpec.from_dict(spec.to_dict()) == spec
        # Construction validates every chart, including special positions.
        for w in spec.templates:
            assert compile_constraints(spec,(w.letter,)).num_atoms == w.multiplicity
        state = initial_state(spec, dtype=torch.float64, generator=generator)
        template = spec.templates[0]
        state = apply_action(state, JointAction('birth', letter=template.letter), generator)
        invariant(state, 1e-7)
        projected = state.compiled.project_coordinates(state.coordinates, wrap=False)
        torch.testing.assert_close(state.compiled.project_coordinates(projected, wrap=False), projected)


def test_layer_height_is_real_and_plane_is_strictly_planar():
    for kind, expected in [('layer',2.3), ('plane',0.)]:
        state = initial_state(catalog_spec(kind, 1), dtype=torch.float64)
        state = apply_action(state, JointAction('birth', letter='a'))
        state.coordinates[:] = torch.tensor([[.2,.3,2.3]])
        assert float(state.decoded()['coordinates'][0,2]) == pytest.approx(expected)


def test_explicit_point_and_rod_actions_share_the_same_compiler():
    identity = tuple(tuple(float(x) for x in row) for row in np.eye(3))
    reflection = ((-1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
    template = OrbitTemplate('g', (identity, reflection), ((0.,0.,0.),(0.,0.,0.)))
    for kind, mask in [('point',(False,False,False)), ('rod',(True,False,False))]:
        spec = SymmetrySpec(kind, 0, mask, 3, (Operation(identity,(0.,0.,0.)),
                            Operation(reflection,(0.,0.,0.))), (template,))
        state = apply_action(initial_state(spec, dtype=torch.float64), JointAction('birth',letter='g'))
        invariant(state, 1e-7)


def test_invalid_explicit_orbit_is_rejected_before_generation():
    identity = tuple(tuple(float(x) for x in row) for row in np.eye(3))
    reflection = ((-1.,0.,0.),(0.,1.,0.),(0.,0.,1.))
    with pytest.raises(ValueError,match='not invariant|invariant affine'):
        SymmetrySpec('point',0,(False,False,False),3,
                     (Operation(identity,(0.,0.,0.)),Operation(reflection,(0.,0.,0.))),
                     (OrbitTemplate('wrong',(identity,),((0.,0.,0.),)),))


def test_edit_kernels_preserve_symmetry_and_copy_surviving_raw_coordinates():
    state = initial_state(catalog_spec('space',2), dtype=torch.float64, budget=16)
    domain = (8,14)
    for action in [JointAction('birth',letter='i'), JointAction('element',0,element=14),
                   JointAction('birth',letter='a'), JointAction('element',1,element=8)]:
        state = apply_action(state, action, check=legal_actions(state,domain,max_atoms=8))
        invariant(state)
    before = state.coordinates[:2].clone()
    replaced = state.coordinates[2:].clone()
    state = apply_action(state, JointAction('replace',1,'b',14), check=legal_actions(state,domain,max_atoms=8))
    torch.testing.assert_close(state.coordinates[:2], before, atol=0, rtol=0)
    torch.testing.assert_close(state.coordinates[2:], replaced, atol=0, rtol=0)
    invariant(state)
    state = apply_action(state, JointAction('delete',orbit=1))
    torch.testing.assert_close(state.coordinates, before, atol=0, rtol=0)
    state = apply_action(state, JointAction('end'))
    assert state.status == 'done' and not legal_actions(state, domain)
    with pytest.raises(ValueError,match='stopped'):
        apply_action(state, JointAction('delete',0))


def test_orbit_split_uses_parent_geometry_and_preserves_all_old_atoms():
    state = apply_action(initial_state(catalog_spec('space',2),dtype=torch.float64),JointAction('birth',letter='i'))
    parent = state.coordinates.clone()
    action = JointAction('split',orbit=0,letter='i')
    assert action in legal_actions(state,(14,),max_atoms=8)
    split = apply_action(state,action,birth_scale=0.)
    assert split.num_atoms == 4
    torch.testing.assert_close(split.coordinates[:2],parent,rtol=0,atol=0)
    torch.testing.assert_close(split.coordinates[2:],parent,rtol=0,atol=0)
    invariant(split)


@pytest.mark.parametrize('dataset', ['mp_20','c2db_51'])
def test_real_data_adapter_and_actual_conditional_endpoint(dataset):
    record = real_record(dataset)
    before = json.dumps(record,sort_keys=True)
    target = target_from_record(record,torch.float64)
    assert target.state.moments is None
    domain = tuple(sorted(set(target.state.elements)))
    for seed in (3,17,29):
        sample = sample_training_path(target, seed, time=1., allowed_elements=domain)
        assert sample.remaining == 0 and sample.state.status == 'done'
        invariant(sample.state,1e-7)
        a, b = sample.state.decoded(), target.state.decoded()
        torch.testing.assert_close(a['lattice'],b['lattice'])
        def rows(decoded):
            x = decoded['coordinates'].numpy().copy()
            periodic = np.asarray(target.state.spec.periodic)
            x[:,periodic] = (x[:,periodic]+1e-8)%1
            return sorted((int(z), *np.round(p,6)) for z,p in zip(decoded['atomic_numbers'],x))
        assert rows(a) == rows(b)
        middle = sample_training_path(target, seed, time=.61, allowed_elements=domain)
        invariant(middle.state)
        if middle.next_action:
            assert middle.next_action in legal_actions(middle.state,domain)
    assert json.dumps(record,sort_keys=True) == before


def test_one_backbone_trains_on_mixed_real_space_and_layer_states():
    targets = [target_from_record(real_record(name)) for name in ('mp_20','c2db_51')]
    allowed = sorted({z for target in targets for z in target.state.elements})
    config = dict(allowed_elements=allowed, max_atoms=192, max_orbits=52, properties={},
                  model=dict(family='joint_symmetry_v1', hidden=32,layers=2,heads=4),cond_dropout=0.)
    model = build_joint_model(config)
    samples = [sample_training_path(target, 28+i,time=.63,allowed_elements=allowed) for i,target in enumerate(targets)]
    loss = joint_generator_loss(model,samples)
    loss['loss'].backward()
    for name in ('coordinate_head.weight','metric_head.weight','count_head.weight','action_head.3.weight',
                 'blocks.0.qkv.weight','template_site_encoder.0.weight'):
        gradient = dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all() and gradient.norm() > 0, name
    assert model.moment_head.weight.grad is None


def test_coordinated_element_swaps_receive_conditional_path_supervision():
    state = initial_state(catalog_spec('space',1),dtype=torch.float64)
    for element in (8,14):
        state = apply_action(state,JointAction('birth',letter='a'))
        state = apply_action(state,JointAction('element',orbit=len(state.letters)-1,element=element))
    target = JointTarget(state,'two_species',{})
    for seed in range(64):
        for time in (.2,.4,.6,.8):
            sample = sample_training_path(target,seed,time=time,allowed_elements=(8,14))
            if sample.next_action and sample.next_action.kind == 'swap':
                assert sample.next_action in legal_actions(sample.state,(8,14))
                invariant(sample.state)
                endpoint = sample_training_path(target,seed,time=1.,allowed_elements=(8,14))
                assert endpoint.state.status == 'done' and sorted(endpoint.state.elements) == [8,14]
                return
    pytest.fail('Coordinated swaps have no teacher-path support')


def test_skeleton_scores_distinguish_equal_size_orbits_with_different_free_directions():
    torch.manual_seed(9318)
    model = build_joint_model(dict(allowed_elements=[8,14],properties={},cond_dropout=0.,
                                   model=dict(family='joint_symmetry_v1',hidden=16,layers=1,heads=4)))
    state = initial_state(catalog_spec('space',47))
    state.metric.requires_grad_(True)
    actions = (JointAction('birth',letter='y'),JointAction('birth',letter='w'))
    first,second = (state.spec.template(a.letter) for a in actions)
    assert first.multiplicity == second.multiplicity and first.dof == second.dof
    np.testing.assert_array_equal(first.arrays()[1],second.arrays()[1])
    assert not np.array_equal(first.arrays()[0],second.arrays()[0])
    logits = model(state,.5,actions=actions)['action_logits']
    assert logits[0] != logits[1], 'Distinct affine directions are invisible to the skeleton scorer'
    sensitivity = torch.autograd.grad(logits[0]-logits[1],state.metric)[0]
    assert torch.isfinite(sensitivity).all() and sensitivity[[0,3,5]].norm() > 0


def test_rate_weighting_recovers_rates_not_unweighted_next_action_frequencies():
    # posterior paths have equal mass but one/four remaining events respectively
    log_rates = torch.tensor([0.2,-0.4],dtype=torch.float64,requires_grad=True)
    rates = log_rates.exp()
    conditional = .5*(rates.sum()-1*log_rates[0])+.5*(rates.sum()-4*log_rates[1])
    expected = rates.sum()-(torch.tensor([.5,2.],dtype=torch.float64)*log_rates).sum()
    torch.testing.assert_close(torch.autograd.grad(conditional,log_rates,retain_graph=True)[0],
                               torch.autograd.grad(expected,log_rates)[0])


@pytest.mark.parametrize('experiment', ['joint_symmetry_mp20','joint_symmetry_c2db51','joint_symmetry_joint'])
def test_new_selectors_are_explicit_opt_in_and_have_no_source_domain(experiment):
    config = training_definition(compose_experiment(experiment,ROOT,('task=plan','device=cpu')))
    assert config['model']['family'] == 'joint_symmetry_v1'
    assert config['precision'] == 'fp32'
    assert 'source_domain' not in config['properties']
    assert 'joint_symmetry_v1' in config['data_audit']
