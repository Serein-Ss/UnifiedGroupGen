"""Opt-in current-model CUDA training checks. Not part of default unit collection.
Run only when explicitly authorizing optimizer updates / short training.
"""
import pytest
import torch
from unifiedgroupgen.tests.unit.test_joint_flow_v2 import config,record

@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA training precision integration')
@pytest.mark.parametrize('precision,mode',[('bf16','bf16-mixed'),('fp32','32-true')])
def test_lightning_fused_checkpoint_and_resume(tmp_path,monkeypatch,precision,mode):
    from unifiedgroupgen.pl_modules.unified.module import fit_joint,load_joint_checkpoint,JointGeneratorModule
    original=JointGeneratorModule.configure_gradient_clipping;norms=[]
    def capture(module,*args,**kwargs):
        original(module,*args,**kwargs)
        norm=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in module.parameters() if p.grad is not None]))
        norms.append(float(norm))
    monkeypatch.setattr(JointGeneratorModule,'configure_gradient_clipping',capture)
    cfg=config('equiformer_v2');cfg['gradient_clip']=.01
    runtime=dict(batch_size=2,precision=precision,fused_adamw=True,log_every_n_steps=25)
    train=[record('mp_20'),record('c2db_51')];val=[record('mp_20','val'),record('c2db_51','val')]
    path=tmp_path/'model.pt'
    trainer=fit_joint(cfg,train,val,path,device='cuda:0',epochs=1,max_steps=1,runtime=runtime)
    assert trainer.precision==mode
    assert norms and max(norms)<=.010001
    assert load_joint_checkpoint(path)['runtime']['fused_adamw']
    resumed=fit_joint(cfg,train,val,path,device='cuda:0',epochs=2,max_steps=2,runtime=runtime,resume=tmp_path/'last.ckpt')
    assert resumed.global_step==2
    assert len(norms)==2 and max(norms)<=.010001
