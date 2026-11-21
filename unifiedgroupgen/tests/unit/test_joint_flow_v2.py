"""New mixed-flow contracts and actual packed-core equivalence; no quality claims."""
from dataclasses import replace
import json
from pathlib import Path
import pytest
import torch
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.common.data_utils import target_from_record
from unifiedgroupgen.pl_data.unified.tasks import task_target
from unifiedgroupgen.pl_data.unified.path import sample_path
from unifiedgroupgen.group_constraints.flow_state import initial,apply,assign,actions,JointAction,coordinate_tangent
from unifiedgroupgen.pl_modules.unified.objectives import loss
from unifiedgroupgen.generation.unified.sampling import sample_batch
from unifiedgroupgen.group_constraints.unified import catalog_spec

ROOT=Path(__file__).resolve().parents[3]

@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)

def config(backend='equiformer_v2',task='dng',property_conditioning=False):
    return dict(model=dict(family='joint_flow_v2',backbone=backend,hidden=16,layers=1,channels=16,
        heads=4,lmax=2,mmax=1,cutoff=4.,max_neighbors=32,use_gate_act=True,task_controls=True),
        allowed_elements=list(range(1,119)),properties={'band_gap':dict(type='scalar')} if property_conditioning else {},
        generation_task=task,chemical_set_probability=0.,cond_dropout=0.,max_atoms=40,max_orbits=12,event_budget=24,
        birth_scale=.2,learning_rate=1e-3)

def record(dataset,split='train'):
    path=ROOT/'unifiedgroupgen/tests/fixtures/current_training_records.json'
    return json.loads(path.read_text(encoding='utf-8'))[dataset][split]

def populated(kind='space',number=1):
    s=initial(catalog_spec(kind,number),budget=24)
    w=next(w for w in s.spec.templates if w.dof)
    s=apply(s,JointAction('birth',letter=w.letter),torch.Generator().manual_seed(2))
    return s

@pytest.mark.parametrize('backend',['equiformer_v2','joint_atom_graph'])
@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
@pytest.mark.parametrize('task,conditioned',[('dng',False),('csp',False),('dng',True),('csp',True)])
def test_real_tasks_finite_backward(backend,dataset,task,conditioned):
    cfg=config(backend,task,conditioned);model=build_joint_model(cfg)
    target=task_target(target_from_record(record(dataset)),cfg,12)
    samples=[sample_path(target,k,time=t,allowed_elements=cfg['allowed_elements'],budget=24,max_atoms=40,max_orbits=12)
             for k,t in [(3,.2),(4,.67),(5,.92)]]
    values=loss(model,samples);values['loss'].backward()
    assert model.last_packed_graphs==3
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    assert model.metric_head.weight.grad is not None
    if conditioned:
        assert model.property_encoder.encoders['band_gap'][0].weight.grad is not None

@pytest.mark.parametrize('backend',['equiformer_v2','joint_atom_graph'])
def test_packed_matches_single_and_does_not_cross_graphs(backend):
    torch.manual_seed(2);m=build_joint_model(config(backend)).eval()
    states=[initial(catalog_spec('space',2),budget=24),populated('space',1),populated('layer',1),populated('plane',1),populated('rod',54)]
    with torch.no_grad():
        batch=m.forward_batch(states,[.3]*len(states))
        single=[m(s,.3) for s in states]
    for a,b in zip(batch,single):
        for key in ['metric_velocity_scaled','coordinate_velocity_scaled','element_logits','remaining','action_logits']:
            torch.testing.assert_close(a[key],b[key],atol=2e-5,rtol=2e-4)

@pytest.mark.parametrize('kind,number',[('space',2),('layer',78),('plane',17),('rod',75)])
def test_orbit_path_endpoint_and_empty_source(kind,number):
    s=populated(kind,number);s=assign(s,0,8)
    from unifiedgroupgen.common.data_utils import JointTarget
    target=JointTarget(s,'fixture',{})
    start=sample_path(target,31,time=0,allowed_elements=[8],budget=24)
    end=sample_path(target,31,time=1,allowed_elements=[8],budget=24)
    assert start.state.num_atoms==0 and not start.state.elements
    assert end.state.frozen and end.remaining==0 and all(end.state.elements)
    torch.testing.assert_close(end.state.coordinates,s.coordinates)
    assert end.state.compiled.residuals(end.state.coordinates,end.state.decoded()['atomic_numbers'],end.state.decoded()['lattice'])['elements']

@pytest.mark.parametrize('kind,number',[('space',1),('rod',43)])
def test_equiformer_rotation_and_tangent_covariance(kind,number):
    torch.manual_seed(9);m=build_joint_model(config()).eval();s=populated(kind,number)
    # Periodic self-images contribute even for a single isolated site.
    q,_=torch.linalg.qr(torch.tensor([[1.,2.,3.],[2.,-1.,.4],[.2,.8,-2.]]))
    if torch.det(q)<0:q[:,0]*=-1
    rotated=replace(s,frame=q)
    with torch.no_grad():a=m(s,.4);b=m(rotated,.4)
    for key in ['metric_velocity_scaled','coordinate_velocity_scaled','element_logits','remaining','action_logits']:
        torch.testing.assert_close(a[key],b[key],atol=3e-5,rtol=3e-4)

def test_batch_sampling_from_empty_retains_attempts_and_symmetry():
    m=build_joint_model(config()).eval()
    result=sample_batch(m,[catalog_spec('space',1),catalog_spec('layer',1),catalog_spec('plane',1)],
                        seeds=[7,8,9],steps=3,budget=10,epsilon=.005)
    assert len(result)==3
    for r in result:
        assert r.events and r.events[0]['atoms_before']==0
        if r.state.status=='done':
            d=r.state.decoded()
            residual=r.state.compiled.residuals(d['coordinates'],d['atomic_numbers'],d['lattice'])
            assert residual['elements'] and residual['coordinates']<1e-4
            assert not any(z==0 for z in r.state.elements)

def test_shell_selection_keeps_symmetry_related_neighbors():
    from unifiedgroupgen.common.data_utils import radius_graph
    f=torch.zeros((1,3));cell=torch.eye(3)[None]*3
    e,n,_,v=radius_graph(f,cell,torch.tensor([1]),torch.tensor([[True]*3]),3.1,1,16,keep_shells=True)
    assert len(v)==6

def test_new_config_selectors():
    from unifiedgroupgen.experiment_config import compose_experiment,training_definition
    for dataset in ['mp20','c2db51']:
        for task in ['dng','property','csp']:
            c=training_definition(compose_experiment(f'flow_{dataset}_{task}'))
            assert c['model']['family']=='joint_flow_v2' and c['model']['operation_compiler']
            assert c['generation_task']==('dng' if task=='property' else task)
        for task in ('dng','csp'):
            c=training_definition(compose_experiment(f'full_{dataset}_eqv2_{task}',
                                                     overrides=['conditioning=formation_energy']))
            assert c['generation_task']==task and set(c['properties'])=={'formation_energy'}
            assert c['cond_dropout']==.1


@pytest.mark.parametrize('task',['dng','csp'])
@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
def test_property_condition_does_not_change_the_chemical_path(task,dataset):
    from unifiedgroupgen.conditional.tasks import eligible_records
    row=record(dataset)
    endpoint=target_from_record(row)
    unconditioned=config(task=task)
    conditioned=config(task=task,property_conditioning=True)
    a=task_target(endpoint,unconditioned,7)
    b=task_target(endpoint,conditioned,7)
    assert a.properties=={} and b.properties=={'band_gap':row['properties']['band_gap']}
    assert a.state.chemistry==b.state.chemistry
    assert bool(b.state.chemistry.counts)==(task=='csp')
    assert eligible_records([row,dict(properties={'band_gap':None})],conditioned)==[row]
    for time in (0.,.45,.8,1.):
        first=sample_path(a,7,time=time,allowed_elements=unconditioned['allowed_elements'])
        second=sample_path(b,7,time=time,allowed_elements=conditioned['allowed_elements'])
        assert first.trajectory==second.trajectory
        assert first.state.elements==second.state.elements
        torch.testing.assert_close(first.state.metric,second.state.metric,rtol=0,atol=0)
        torch.testing.assert_close(first.state.coordinates,second.state.coordinates,rtol=0,atol=0)


def test_property_is_a_conditioning_option_not_a_generation_mode():
    invalid = config(task='property', property_conditioning=True)
    with pytest.raises(ValueError, match='independent task'):
        build_joint_model(invalid)


def test_nontrivial_rotation_translation_and_atom_permutation():
    from unifiedgroupgen.group_constraints.flow_state import from_state
    torch.manual_seed(13);m=build_joint_model(config()).eval()
    s=initial(catalog_spec('space',1),budget=24)
    for z in [8,14,16]:
        s=apply(s,JointAction('birth',letter='a'),torch.Generator().manual_seed(z))
        s=assign(s,len(s.elements)-1,z)
    s=replace(s,coordinates=s.metric.new_tensor([[.13,.27,.41],[.44,.18,.63],[.72,.66,.21]]))
    rotation,_=torch.linalg.qr(torch.randn(3,3))
    if torch.det(rotation)<0:rotation[:,0]*=-1
    transformed=replace(s,frame=rotation,origin=torch.tensor([2.,-3.,5.]))
    with torch.no_grad():
        a=m(s,.43);b=m(transformed,.43)
    for key in ['metric_velocity_scaled','coordinate_velocity_scaled','element_logits','remaining','action_logits']:
        torch.testing.assert_close(a[key],b[key],atol=4e-5,rtol=5e-4)
    assert a['coordinate_velocity_scaled'].abs().max()>.01
    perm=[2,0,1]
    permuted=replace(s,elements=tuple(s.elements[i] for i in perm),coordinates=s.coordinates[perm])
    with torch.no_grad():c=m(permuted,.43)
    torch.testing.assert_close(a['metric_velocity_scaled'],c['metric_velocity_scaled'],atol=4e-5,rtol=5e-4)
    torch.testing.assert_close(a['element_logits'][perm],c['element_logits'],atol=4e-5,rtol=5e-4)
    torch.testing.assert_close(a['coordinate_velocity_scaled'][perm],c['coordinate_velocity_scaled'],atol=4e-5,rtol=5e-4)

def test_new_native_format_and_optimizer_roundtrip_without_training(tmp_path):
    from unifiedgroupgen.pl_modules.unified.module import JointGeneratorModule,load_joint_checkpoint,FLOW_FORMAT
    cfg=config('joint_atom_graph')
    module=JointGeneratorModule(cfg)
    optimizer=module.configure_optimizers()
    output=tmp_path/'last.ckpt'
    saved=dict(epoch=2,global_step=42,state_dict=module.state_dict(),
        joint_state=dict(format=FLOW_FORMAT,config=cfg),optimizer_states=[optimizer.state_dict()])
    torch.save(saved,output)
    value=load_joint_checkpoint(output)
    assert value['format']==FLOW_FORMAT
    restored=build_joint_model(value['config']);restored.load_state_dict(value['model'],strict=True)
    other=torch.optim.AdamW(restored.parameters(),lr=.2)
    other.load_state_dict(torch.load(output,weights_only=False)['optimizer_states'][0])
    assert other.param_groups[0]['lr']==optimizer.param_groups[0]['lr']
    for key,tensor in module.model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key],tensor,rtol=0,atol=0)

def test_oracle_sampler_reaches_real_endpoint_without_reference_at_birth():
    # Oracle is a numerical sampler fixture, not a learned generation result.
    from unifiedgroupgen.common.data_utils import JointTarget
    endpoint=populated('space',2);endpoint=assign(endpoint,0,8)
    class Oracle(torch.nn.Module):
        allowed_elements=(8,);max_atoms=40;max_orbits=12;event_budget=8;birth_scale=.2
        def __init__(self):
            super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(()))
        def forward_batch(self,states,times,properties=None,actions=None):
            result=[]
            for s in states:
                support=() if s.frozen else ((JointAction('birth',letter=endpoint.letters[0]),) if not s.letters else (JointAction('freeze'),))
                logits=s.metric.new_full((len(s.letters),119),-torch.inf);logits[:,8]=0
                result.append(dict(metric_velocity_scaled=endpoint.metric-s.metric,
                    coordinate_velocity_scaled=endpoint.coordinates-s.coordinates if s.num_atoms else s.coordinates,
                    element_logits=logits,remaining=s.metric.new_tensor(0 if s.frozen else 2-s.step),
                    actions=support,action_logits=s.metric.new_zeros(len(support))))
            return result
    results=sample_batch(Oracle(),[endpoint.spec]*3,steps=32,epsilon=1e-4,budget=8,seeds=[1,2,3])
    for r in results:
        assert r.state.status=='done'
        torch.testing.assert_close(r.state.coordinates,endpoint.coordinates,atol=.002,rtol=.002)
        assert r.state.elements==(8,)

def test_all_structural_kernels_are_legal_and_preserve_survivors():
    s=populated('space',2);s=assign(s,0,8)
    w=next(w for w in s.spec.templates if w.dof)
    child=apply(s,JointAction('split',orbit=0,letter=w.letter),torch.Generator().manual_seed(4))
    torch.testing.assert_close(child.coordinates[:s.num_atoms],s.coordinates,atol=0,rtol=0)
    assert child.elements==(8,0)
    fixed=next(w for w in s.spec.templates if not w.dof)
    changed=apply(child,JointAction('replace',1,fixed.letter),torch.Generator().manual_seed(5))
    assert changed.elements==(8,0)
    deleted=apply(changed,JointAction('delete',orbit=1))
    torch.testing.assert_close(deleted.coordinates,s.coordinates,atol=0,rtol=0)
    frozen=apply(deleted,JointAction('freeze'))
    assert frozen.frozen and frozen.status=='active'


@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
def test_real_v2_endpoint_cif_roundtrip(dataset):
    from unifiedgroupgen.group_constraints.flow_state import from_state
    from unifiedgroupgen.generation.types import JointSample
    from unifiedgroupgen.joint import evaluation_record
    target=target_from_record(record(dataset))
    state=from_state(target.state,frozen=True)
    sample=JointSample(state,1.,[],'real_endpoint_fixture',0,1e-4)
    row=evaluation_record(sample,0,1)
    assert row['status']=='generated' and row['accepted']
    assert row['cif_sha256'] and row['constraint_residuals']['elements']
