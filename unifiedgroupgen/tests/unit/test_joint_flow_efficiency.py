"""Acceleration must retain graph topology, FP32 geometry and shared gradients."""
import pytest
import torch
from unifiedgroupgen.tests.unit.test_joint_flow_v2 import config,record
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_data.joint_path import target_from_record
from unifiedgroupgen.pl_data.task_path import task_target
from unifiedgroupgen.joint_flow.path import sample_path
from unifiedgroupgen.joint_flow.objective import loss
from unifiedgroupgen.joint_flow.state import metric_tangent,coordinate_tangent
from unifiedgroupgen.pl_modules.backbones.periodic_graph import radius_graph


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('shells',[False,True])
@pytest.mark.parametrize('periodic',[(True,True,True),(True,True,False)])
def test_vectorized_neighbors_equal_individual_target_sort(shells,periodic):
    torch.manual_seed(8)
    coords=torch.rand((9,3));cell=torch.tensor([[3.,0,0],[.7,3.2,0],[0,0,5.]])
    e,shifts,_,vectors=radius_graph(coords,cell[None],torch.tensor([9]),torch.tensor([periodic]),4.,5,16,shells)
    import itertools
    bounds=torch.ceil(4.*torch.linalg.vector_norm(torch.linalg.inv(cell),dim=0)).long()+1
    bounds=torch.where(torch.tensor(periodic),bounds,torch.zeros_like(bounds))
    images=torch.tensor(list(itertools.product(*(range(-n,n+1) for n in bounds.tolist()))))
    source,target=torch.meshgrid(torch.arange(9),torch.arange(9),indexing='ij');source=source.flatten();target=target.flatten()
    distances=torch.linalg.vector_norm((coords[source,None]-coords[target,None]+images)@cell,dim=-1)
    pairs,im=((distances<4.)&(distances>1e-6)).nonzero(as_tuple=True)
    selected=[]
    for atom in range(9):
        ids=(target[pairs]==atom).nonzero().flatten()
        ordered=ids[distances[pairs[ids],im[ids]].argsort(stable=True)]
        if shells and len(ordered)>5:
            boundary=distances[pairs[ordered[4]],im[ordered[4]]]
            ordered=ordered[distances[pairs[ordered],im[ordered]]<=boundary+1e-5]
        else:ordered=ordered[:5]
        selected.append(ordered)
    ids=torch.cat(selected)
    assert torch.equal(e,torch.stack((source[pairs[ids]],target[pairs[ids]])))
    assert torch.equal(shifts,images[im[ids]])
    torch.testing.assert_close(vectors,(coords[e[0]]+shifts-coords[e[1]])@cell,atol=0,rtol=0)


@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
def test_checkpointed_core_matches_outputs_and_gradients(dataset):
    cfg=config();torch.manual_seed(9);plain=build_joint_model(cfg)
    cfg2={**cfg,'model':{**cfg['model'],'checkpoint_layers':True}}
    checked=build_joint_model(cfg2);checked.load_state_dict(plain.state_dict())
    target=task_target(target_from_record(record(dataset)),cfg,5)
    samples=[sample_path(target,5,time=.8,allowed_elements=cfg['allowed_elements'])]
    rng=torch.get_rng_state();a=loss(plain,samples);a['loss'].backward();torch.set_rng_state(rng)
    b=loss(checked,samples);b['loss'].backward()
    torch.testing.assert_close(a['loss'],b['loss'],atol=1e-5,rtol=1e-5)
    for (name,p),(other,q) in zip(plain.named_parameters(),checked.named_parameters()):
        assert name==other and (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA AMP/fused execution')
@pytest.mark.parametrize('backend',['equiformer_v2','joint_atom_graph'])
@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
def test_cuda_bf16_geometry_remains_float32(backend,dataset):
    cfg=config(backend);m=build_joint_model(cfg,'cuda:0')
    target=task_target(target_from_record(record(dataset)),cfg,31)
    samples=[sample_path(target,31,time=.8,allowed_elements=cfg['allowed_elements']).to('cuda:0')]
    s=samples[0].state
    with torch.autocast('cuda',dtype=torch.bfloat16):
        decoded=s.decoded();value=loss(m,samples)
        assert decoded['lattice'].dtype==decoded['coordinates'].dtype==torch.float32
        assert metric_tangent(s,torch.zeros_like(s.metric)).dtype==torch.float32
        assert coordinate_tangent(s,torch.zeros_like(s.coordinates)).dtype==torch.float32
    optimizer=torch.optim.AdamW(m.parameters(),lr=1e-4,fused=True)
    value['loss'].backward();optimizer.step()
    assert all(torch.isfinite(p).all() for p in m.parameters())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_fast_profile_resolves_without_changing_model_definition():
    from unifiedgroupgen.common.configuration import compose_experiment,training_definition
    a=compose_experiment('flow_mp20_dng')
    b=compose_experiment('flow_mp20_dng',overrides=['runtime=joint_flow_fast'])
    assert training_definition(a)==training_definition(b)
    assert a.train.chemical_set_probability==0.
    assert b.runtime.precision=='bf16' and b.runtime.fused_adamw


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA training precision integration')
@pytest.mark.parametrize('precision,mode',[('bf16','bf16-mixed'),('fp32','32-true')])
def test_lightning_fused_checkpoint_and_resume(tmp_path,monkeypatch,precision,mode):
    from unifiedgroupgen.pl_modules.joint_training import fit_joint,load_joint_checkpoint,JointGeneratorModule
    original=JointGeneratorModule.configure_gradient_clipping;norms=[]
    def capture(module,*args,**kwargs):
        original(module,*args,**kwargs)
        norm=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in module.parameters() if p.grad is not None]))
        norms.append(float(norm))
    monkeypatch.setattr(JointGeneratorModule,'configure_gradient_clipping',capture)
    cfg=config('equiformer_v2');cfg['gradient_clip']=.01
    runtime=dict(batch_size=2,edge_budget=200000,precision=precision,fused_adamw=True,log_every_n_steps=25)
    train=[record('mp_20'),record('c2db_51')];val=[record('mp_20','val'),record('c2db_51','val')]
    path=tmp_path/'model.pt'
    trainer=fit_joint(cfg,train,val,path,device='cuda:0',epochs=1,max_steps=1,runtime=runtime)
    assert trainer.precision==mode
    assert norms and max(norms)<=.010001
    assert load_joint_checkpoint(path)['runtime']['fused_adamw']
    resumed=fit_joint(cfg,train,val,path,device='cuda:0',epochs=2,max_steps=2,runtime=runtime,resume=tmp_path/'last.ckpt')
    assert resumed.global_step==2
    assert len(norms)==2 and max(norms)<=.010001
