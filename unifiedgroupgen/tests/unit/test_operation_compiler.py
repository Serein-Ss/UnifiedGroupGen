"""Independent catalog crosschecks, compiled fields and real joint workflows."""
from collections import Counter
import json
import numpy as np
import pytest
import torch
from unifiedgroupgen.group_constraints.operations import compile_operations, operation_spec
from unifiedgroupgen.group_constraints.unified import catalog_spec, SymmetrySpec
from unifiedgroupgen.group_constraints.state import initial_state, apply_action, legal_actions, JointAction
from unifiedgroupgen.pl_data.operation_path import match_orbit
from unifiedgroupgen.pl_data.joint_path import target_from_record, sample_training_path
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_modules.constraint_layers import project_scalars, project_vectors
from unifiedgroupgen.pl_modules.objectives.joint_generator import joint_generator_loss
from unifiedgroupgen.tests.unit.test_joint_generator import real_record, invariant, preserve_joint_process_state


GROUPS = [(kind, n) for kind, count in [('space', 230), ('layer', 80), ('plane', 17)]
          for n in range(1, count+1)]


@pytest.mark.parametrize('kind,number', GROUPS)
def test_computed_atlas_matches_every_catalog_orbit(kind, number):
    spec = operation_spec(kind, number)
    reference = catalog_spec(kind, number)
    assert spec.number == 0
    assert Counter((w.multiplicity, w.dof) for w in spec.templates) == Counter(
        (w.multiplicity, w.dof) for w in reference.templates)
    # Match actual generic coordinates, not only counts/histograms.
    for w in reference.templates:
        a, b = w.arrays()
        x = b+np.einsum('niq,q->ni', a, np.array([.137031, .293117, 1.419071])[:w.dof])
        letter, fitted, order = match_orbit(spec, x, tolerance=1e-7)
        assert spec.template(letter).dof == w.dof
        delta = fitted-x[order]
        delta[:, spec.periodic] -= np.round(delta[:, spec.periodic])
        assert np.max(np.abs(delta)) < 1e-7


def test_compilation_uses_no_special_position_table(monkeypatch):
    from unifiedgroupgen.group_constraints import operations as module
    from unifiedgroupgen.group_constraints import unified
    from unifiedgroupgen.group_constraints import compiler
    module._atlas.cache_clear()
    def forbidden(*args, **kwargs):
        raise AssertionError('A special-position table was consulted')
    monkeypatch.setattr(unified, 'wp_arrays', forbidden)
    monkeypatch.setattr(unified, 'group_table', forbidden)
    monkeypatch.setattr(unified, 'catalog_spec', forbidden)
    monkeypatch.setattr(compiler, 'group_table', forbidden)
    mirror = dict(rotation=[[-1,0,0],[0,1,0],[0,0,1]], translation=[0,0,0])
    spec = compile_operations([mirror], 'space')
    assert Counter((w.multiplicity,w.dof) for w in spec.templates) == Counter({(1,2):2,(2,3):1})


def test_equivalent_generators_and_serialization_are_identical():
    x = dict(rotation=[[-1,0,0],[0,1,0],[0,0,1]], translation=[0,0,0])
    y = dict(rotation=[[1,0,0],[0,-1,0],[0,0,1]], translation=[0,0,0])
    first = compile_operations([x,y], 'plane')
    second = compile_operations([y,x,x]+[o.to_dict() for o in first.operations], 'plane')
    assert first == second == SymmetrySpec.from_dict(first.to_dict())
    assert first == SymmetrySpec.from_dict(dict(kind='plane',generators=[x,y]))


@pytest.mark.parametrize('extra', [dict(forbidden=['inversion']), dict(periodic=[True,True,True]),
                                  dict(dimension=3), dict(magnetic=True)])
def test_unsupported_or_inconsistent_requests_are_not_ignored(extra):
    with pytest.raises(ValueError):
        SymmetrySpec.from_dict(dict(kind='plane', generators=[], **extra))


@pytest.mark.parametrize('operation,kind,message', [
    (dict(rotation=[[1,1,0],[0,1,0],[0,0,1]],translation=[0,0,0]),'space','close'),
    (dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation=[0,0,1]),'layer','close'),
    (dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation=[0,0,1]),'plane','z=0'),
    (dict(rotation=[[2,0,0],[0,1,0],[0,0,1]],translation=[0,0,0]),'space','adapted'),
    (dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation=[2**.5,0,0]),'space','rational'),
])
def test_invalid_operation_requests_fail_explicitly(operation, kind, message):
    with pytest.raises(ValueError, match=message):
        compile_operations([operation],kind)


@pytest.mark.parametrize('kind,number', [('space',14),('space',225),('layer',80),('plane',17)])
def test_compiled_feature_fields_are_exact_and_differentiable(kind,number):
    torch.set_num_threads(1)
    spec = operation_spec(kind,number)
    w = max(spec.templates,key=lambda w:w.dof)
    state = apply_action(initial_state(spec,dtype=torch.float64),JointAction('birth',letter=w.letter))
    state.metric.requires_grad_()
    lattice = state.compiled.lattice(state.metric)
    scalar = torch.randn(state.num_atoms,5,dtype=torch.float64,requires_grad=True)
    vector = torch.randn(state.num_atoms,2,3,dtype=torch.float64,requires_grad=True)
    s = project_scalars(scalar,state.compiled)
    v = project_vectors(vector,state.compiled,lattice)
    torch.testing.assert_close(project_vectors(v,state.compiled,lattice),v)
    for op, permutation in zip(spec.operations,state.compiled.permutations()):
        q = lattice@lattice.new_tensor(op.rotation)@torch.linalg.inv(lattice)
        torch.testing.assert_close(s[list(permutation)],s)
        torch.testing.assert_close(v[list(permutation)],torch.einsum('ab,ncb->nca',q,v))
    (s.square().sum()+v.square().sum()).backward()
    for value in (scalar,vector,state.metric):
        assert value.grad is not None and torch.isfinite(value.grad).all()


@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
def test_recompiled_real_target_has_legal_paths_and_loss_gradients(dataset):
    torch.set_num_threads(1)
    row = real_record(dataset)
    before = json.dumps(row,sort_keys=True)
    target = target_from_record(row,operation_compiler=True)
    assert json.dumps(row,sort_keys=True) == before
    assert target.state.spec.number == 0
    domain = tuple(sorted(set(target.state.elements)))
    model = build_joint_model(dict(model=dict(family='joint_symmetry_v1',hidden=32,layers=2,heads=4,
                                  constraint_layers=True,operation_compiler=True),allowed_elements=domain))
    samples = [sample_training_path(target,seed,time=t,allowed_elements=domain)
               for seed,t in [(3,0.),(4,.3),(5,.7),(6,1.)]]
    for sample in samples:
        invariant(sample.state)
        if sample.remaining:
            assert sample.next_action in legal_actions(sample.state,domain)
    losses = joint_generator_loss(model,samples)
    losses['loss'].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.constraint_layers.parameters())
    output = model(samples[1].state,.3)
    assert 'constraint_fields' in output


@pytest.mark.parametrize('kind',['space','layer','plane'])
def test_every_legal_edit_uses_compiled_constraints(kind):
    spec = operation_spec(kind,2)
    state = initial_state(spec,dtype=torch.float64,budget=16)
    general = max(spec.templates,key=lambda w:w.dof)
    state = apply_action(state,JointAction('birth',letter=general.letter))
    state = apply_action(state,JointAction('element',0,element=14))
    for action in legal_actions(state,(8,14),max_atoms=32,max_orbits=8):
        result = apply_action(state,action,check=legal_actions(state,(8,14),32,8))
        invariant(result,1e-7)


def test_new_configurations_leave_legacy_defaults_unchanged():
    from unifiedgroupgen.common.configuration import compose_experiment,training_definition
    for name in ('operation_joint','operation_mp20','operation_c2db51'):
        config = training_definition(compose_experiment(name))
        assert config['model']['operation_compiler'] and config['model']['constraint_layers']
    old = training_definition(compose_experiment('joint_symmetry_joint'))
    assert not old['model'].get('operation_compiler',False)
    assert not old['model'].get('constraint_layers',False)
    model = build_joint_model(dict(model=dict(family='joint_symmetry_v1',hidden=16,layers=1,heads=4),allowed_elements=[14]))
    assert not any('constraint_layers' in key for key in model.state_dict())


def test_fixed_orbit_retains_atomwise_public_latent_residuals():
    torch.set_num_threads(1)
    torch.manual_seed(21)
    spec = operation_spec('space',225)
    fixed = next(w for w in spec.templates if w.dof == 0 and w.multiplicity > 1)
    state = apply_action(initial_state(spec),JointAction('birth',letter=fixed.letter))
    state.coordinates.requires_grad_()
    model = build_joint_model(dict(model=dict(family='joint_symmetry_v1',hidden=32,layers=2,heads=4,
                                  constraint_layers=True),allowed_elements=[14]))
    output = model(state,.5)
    # Physical positions are fixed, but the ambient anchor must still represent
    # unequal offsets and atomwise bridge residuals. Scalar pooling loses both.
    difference = output['coordinate_anchor'][0]-output['coordinate_anchor'][1]
    assert difference.abs().max() > 1e-6
    gradient = torch.autograd.grad(difference.square().sum(),state.coordinates)[0]
    assert gradient.abs().sum() > 1e-8
    assert output['constraint_fields']['vectors'].abs().max() == 0


@pytest.mark.parametrize('kind,number',[('space',225),('layer',80),('plane',17)])
def test_event_sampler_operates_on_computed_orbits(kind,number):
    from unifiedgroupgen.generation.joint import sample_joint
    from unifiedgroupgen.tests.unit.test_joint_generator_workflow import Oracle
    torch.set_num_threads(1)
    spec = operation_spec(kind,number)
    letter = min(spec.templates,key=lambda w:(w.multiplicity,w.dof)).letter
    class ComputedOracle(Oracle):
        def forward(self,state,time,properties=None,actions=None):
            result = super().forward(state,time,properties,actions)
            desired = (JointAction('birth',letter=letter) if state.step == 0 else
                       JointAction('element',orbit=0,element=14) if state.step == 1 else JointAction('end'))
            result['action_logits'].fill_(-100.)
            if desired in result['actions']:
                result['action_logits'][result['actions'].index(desired)] = 100.
            return result
    result = sample_joint(ComputedOracle(),spec,steps=16,budget=8,generator=torch.Generator().manual_seed(12))
    assert result.state.status == 'done' and result.reason == 'complete'
    assert [row['action']['kind'] for row in result.events] == ['birth','element','end']
    invariant(result.state,1e-7)
    if kind == 'space':
        from unifiedgroupgen.joint import evaluation_record
        row = evaluation_record(result,0,12,exact=True,requested_group=number)
        assert row['accepted'] and row['status'] == 'generated' and row['cif_sha256']


def test_operation_mode_native_checkpoint_resumes(tmp_path):
    from pytorch_lightning import seed_everything
    from unifiedgroupgen.pl_modules.joint_training import fit_joint,load_joint_checkpoint
    torch.set_num_threads(1)
    config = dict(model=dict(family='joint_symmetry_v1',hidden=16,layers=1,heads=4,
                            operation_compiler=True,constraint_layers=True),
                  allowed_elements=list(range(1,119)),properties={},max_atoms=192,max_orbits=52,
                  event_budget=256,birth_scale=.2,learning_rate=1e-3,cond_dropout=0.)
    train = [real_record('mp_20'),real_record('c2db_51')]
    val = [real_record('mp_20','val'),real_record('c2db_51','val')]
    runtime = dict(batch_size=2,edge_budget=200000)
    seed_everything(72,workers=True)
    output = tmp_path/'model.pt'
    trainer = fit_joint(config,train,val,output,epochs=2,runtime=runtime,max_steps=1)
    assert trainer.global_step == 1
    saved = load_joint_checkpoint(output)
    model = build_joint_model(saved['config'])
    model.load_state_dict(saved['model'],strict=True)
    continued = fit_joint(config,train,val,output,epochs=3,runtime=runtime,max_steps=2,resume=tmp_path/'last.ckpt')
    assert continued.global_step == 2


def test_compile_cli_reads_operations_and_persists_complete_spec(tmp_path,capsys,monkeypatch):
    from unifiedgroupgen.common.configuration import artifact_path
    import unifiedgroupgen.joint as entry
    monkeypatch.setattr(entry,'artifact_path',lambda value:artifact_path(value,root=tmp_path))
    from unifiedgroupgen.joint import main
    request,output = tmp_path/'request.json',tmp_path/'compiled.json'
    request.write_text(json.dumps(dict(kind='plane',generators=[])),encoding='utf-8')
    main(['compile','--symmetry',str(request),'--output',str(output)])
    spec = SymmetrySpec.from_dict(json.loads(output.read_text(encoding='utf-8')))
    assert spec.kind == 'plane' and [(w.multiplicity,w.dof) for w in spec.templates] == [(1,2)]
    assert json.loads(capsys.readouterr().out)['source'] == 'computed_fixed_loci'


@pytest.mark.parametrize('device',['cpu','cuda:0'])
def test_plane_joint_loss_backward_on_cpu_and_cuda(device):
    if device.startswith('cuda') and not torch.cuda.is_available():
        pytest.skip('CUDA is not available')
    from unifiedgroupgen.group_constraints.compiler import Descriptor,OrbitSpec,compile_descriptor
    torch.set_num_threads(1)
    general = catalog_spec('plane',17).templates[0].letter
    descriptor = Descriptor('plane',17,(OrbitSpec(general,14),))
    old = compile_descriptor(descriptor)
    record = dict(id='plane_fixture',descriptor=descriptor.to_dict(),
                  state=old.prior(generator=torch.Generator().manual_seed(71)).tolist())
    target = target_from_record(record,operation_compiler=True)
    model = build_joint_model(dict(model=dict(family='joint_symmetry_v1',hidden=16,layers=1,heads=4,
                                  constraint_layers=True,operation_compiler=True),allowed_elements=[14]),device)
    samples = [sample_training_path(target,seed,time=t,allowed_elements=(14,)).to(device)
               for seed,t in [(14,.4),(17,.8)]]
    loss = joint_generator_loss(model,samples)['loss']
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
