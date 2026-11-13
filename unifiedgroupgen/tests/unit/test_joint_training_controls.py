"""Scheduler/callback state tests without model construction, fit, or backward."""
import copy
from types import SimpleNamespace
import pytest
import torch
from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
from unifiedgroupgen.pl_modules import joint_training as training


def optimizer_and_schedule():
    parameter=torch.nn.Parameter(torch.tensor(2.))
    module=SimpleNamespace(config=dict(learning_rate=.001,lr_schedule=dict(factor=.5,patience=1,
        threshold=.0001,min_lr=.00001)),runtime={},parameters=lambda:iter([parameter]))
    result=training.JointGeneratorModule.configure_optimizers(module)
    return parameter,result['optimizer'],result['lr_scheduler']['scheduler']


def test_plateau_scheduler_decays_and_preserves_state_without_parameter_updates():
    parameter,optimizer,scheduler=optimizer_and_schedule()
    scheduler.step(3.);scheduler.step(3.)
    opt_state=copy.deepcopy(optimizer.state_dict());state=copy.deepcopy(scheduler.state_dict())
    scheduler.step(3.);assert optimizer.param_groups[0]['lr']==.0005
    other,opt,resumed=optimizer_and_schedule();opt.load_state_dict(opt_state);resumed.load_state_dict(state)
    resumed.step(3.)
    assert opt.param_groups[0]['lr']==optimizer.param_groups[0]['lr']
    assert scheduler.state_dict()==resumed.state_dict()
    assert parameter.item()==other.item()==2. and not optimizer.state


def test_legacy_config_keeps_plain_optimizer():
    parameter=torch.nn.Parameter(torch.tensor(2.))
    stub=SimpleNamespace(config={},runtime={},parameters=lambda:iter([parameter]))
    assert isinstance(training.JointGeneratorModule.configure_optimizers(stub),torch.optim.AdamW)


def test_early_stopping_patience_is_restored_without_fit(tmp_path):
    config=dict(early_stopping=dict(patience=2,min_delta=.0001))
    callbacks=training.joint_callbacks(config,tmp_path/'model.pt')
    early=next(c for c in callbacks if isinstance(c,EarlyStopping))
    assert not early._check_on_train_epoch_end and early.check_finite
    assert not early._evaluate_stopping_criteria(torch.tensor(3.))[0]
    assert not early._evaluate_stopping_criteria(torch.tensor(3.))[0]
    restored=EarlyStopping(monitor='val/loss',patience=2,min_delta=.0001,check_on_train_epoch_end=False)
    restored.load_state_dict(early.state_dict())
    assert early._evaluate_stopping_criteria(torch.tensor(3.))[0]
    assert restored._evaluate_stopping_criteria(torch.tensor(3.))[0]
    assert any(isinstance(c,ModelCheckpoint) and c.monitor=='val/loss' for c in callbacks)


def checkpoint(completed=True):
    counters=dict(ready=1,started=1,processed=0,completed=0)
    batch=dict(ready=4,started=4,processed=4,completed=4 if completed else 2)
    return dict(epoch=0,loops={'fit_loop':{'epoch_progress':dict(total=counters.copy(),current=counters.copy()),
        'epoch_loop.batch_progress':dict(is_last_batch=completed,total=batch.copy(),current=batch.copy()),
        'epoch_loop.scheduler_progress':{},'epoch_loop.automatic_optimization.optim_progress':{},
        'epoch_loop.val_loop.batch_progress':dict(is_last_batch=True,total=counters.copy(),current=counters.copy())}})


def test_pre_scheduler_checkpoint_replays_exactly_once_and_final_save_does_not(monkeypatch):
    _,optimizer,scheduler=optimizer_and_schedule()
    config=dict(model={'family':'joint_flow_v2'},lr_schedule={'patience':1})
    module=SimpleNamespace(config=config,runtime={},split_hashes={},lr_schedulers=lambda:scheduler,
        trainer=SimpleNamespace(num_training_batches=4,callback_metrics={'val/loss':torch.tensor(3.)}))
    monkeypatch.setattr(training,'_collect_rng_states',lambda:{'fixture':True})
    monkeypatch.setattr(training,'_set_rng_states',lambda state:None)
    saved=checkpoint();training.JointGeneratorModule.on_save_checkpoint(module,saved)
    assert saved['epoch']==1 and saved['joint_state']['pending_lr_metric']==3.
    module._pending_rng=None;module._pending_lr_metric=None
    training.JointGeneratorModule.on_load_checkpoint(module,saved)
    training.JointGeneratorModule.on_train_start(module)
    assert scheduler.last_epoch==1 and module._pending_lr_metric is None
    training.JointGeneratorModule.on_train_start(module);assert scheduler.last_epoch==1
    final=checkpoint();training.JointGeneratorModule.on_save_checkpoint(module,final)
    assert final['joint_state']['pending_lr_metric'] is None
    partial=checkpoint(False);training.JointGeneratorModule.on_save_checkpoint(module,partial)
    assert partial['joint_state']['pending_lr_metric'] is None
    assert not optimizer.state


def test_validation_seed_is_repeatable_and_does_not_advance_training_rng(monkeypatch):
    initial=torch.get_rng_state().clone()
    monkeypatch.setattr(training,'_collect_rng_states',lambda:{'cpu':torch.get_rng_state().clone()})
    monkeypatch.setattr(training,'_set_rng_states',lambda state:torch.set_rng_state(state['cpu']))
    module=SimpleNamespace(config={'validation_seed':918},_validation_rng=None)
    try:
        values=[]
        for _ in range(2):
            training.JointGeneratorModule.on_validation_epoch_start(module);values.append(torch.rand(5))
            training.JointGeneratorModule.on_validation_epoch_end(module)
            assert torch.equal(initial,torch.get_rng_state())
        assert torch.equal(*values)
    finally:torch.set_rng_state(initial)


def test_strict_fp32_disables_tf32_without_cuda_execution(monkeypatch):
    calls=[];matmul=SimpleNamespace(allow_tf32=True);cudnn=SimpleNamespace(allow_tf32=True)
    monkeypatch.setattr(training.torch,'set_float32_matmul_precision',calls.append)
    monkeypatch.setattr(training.torch.backends.cuda,'matmul',matmul)
    monkeypatch.setattr(training.torch.backends,'cudnn',cudnn)
    training.strict_fp32(dict(precision='fp32',strict_fp32=True))
    assert calls==['highest'] and not matmul.allow_tf32 and not cudnn.allow_tf32
    with pytest.raises(ValueError):training.strict_fp32(dict(precision='bf16',strict_fp32=True))
    monkeypatch.setenv('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE','1')
    with pytest.raises(ValueError,match='Disable'):training.strict_fp32(dict(precision='fp32',strict_fp32=True))


def test_fit_wiring_uses_only_fake_model_and_fake_trainer(tmp_path,monkeypatch):
    import json
    calls=[]
    class FakeModule:
        def __init__(self,*args):self.model=SimpleNamespace(state_dict=lambda:{})
    class FakeTrainer:
        def __init__(self,**options):
            calls.append(options);self.callbacks=options['callbacks'];self.current_epoch=61
            self.global_step=123;self.interrupted=False;self.callback_metrics={'val/loss':torch.tensor(2.)}
        def fit(self,*args,**kwargs):
            for cb in self.callbacks:
                if isinstance(cb,EarlyStopping):cb.stopped_epoch=60
                if isinstance(cb,ModelCheckpoint):cb.best_model_score=torch.tensor(2.)
        def save_checkpoint(self,path):path.write_bytes(b'fixture, not trained weights')
    monkeypatch.setattr(training,'JointGeneratorModule',FakeModule)
    monkeypatch.setattr(training,'JointDataModule',lambda *args:None)
    monkeypatch.setattr(training.pl,'Trainer',FakeTrainer)
    monkeypatch.setattr(training,'strict_fp32',lambda runtime:calls.append(dict(runtime)))
    config=dict(model={'family':'joint_flow_v2','task_controls':True},generation_task='dng',
        properties={},group_prior={},lr_schedule={'factor':.5,'patience':10},
        early_stopping={'patience':60,'min_delta':.0001})
    path=tmp_path/'model.pt'
    training.fit_joint(config,[{'id':'train'}],[{'id':'val'}],path,device='cpu',epochs=1000,
                       runtime={'precision':'fp32','strict_fp32':True})
    assert calls[0]['strict_fp32'] and calls[1]['precision']=='32-true' and calls[1]['max_epochs']==1000
    assert any(isinstance(c,EarlyStopping) for c in calls[1]['callbacks'])
    status=json.loads(path.with_suffix('.status.json').read_text())
    assert status['stop_reason']=='early_stopping' and status['epochs_completed']==61
    assert status['max_epochs']==1000 and status['best_validation_loss']==2.
