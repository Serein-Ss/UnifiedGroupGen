"""Exercise real Lightning callback binding/dispatch without entering Trainer.fit."""
import copy
import io
import json
import random

import numpy as np
import pytest
import torch
import pytorch_lightning as pl
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.states import TrainerFn, RunningStage

from unifiedgroupgen.pl_modules.progress import TrainingProgress, cache_progress_losses, progress_callbacks


class Probe(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.weight=torch.nn.Parameter(torch.tensor(1.))
        self.runtime={'batch_size':8}

    def training_step(self,*args,**kwargs):
        raise AssertionError('No training step is permitted in callback binding tests')


def bound_trainer(tmp_path,monkeypatch,tty=False):
    class Terminal(io.StringIO):
        def isatty(self):return tty
    monkeypatch.setattr('sys.stdout',Terminal())
    callbacks=progress_callbacks(tmp_path/'model.pt')
    trainer=pl.Trainer(accelerator='cpu',devices=1,logger=False,callbacks=callbacks,
        enable_checkpointing=False,enable_progress_bar=tty,enable_model_summary=False,
        default_root_dir=str(tmp_path),max_epochs=1000,num_sanity_val_steps=0)
    module=Probe();trainer.strategy.connect(module);module.trainer=trainer
    trainer.strategy.optimizers=[torch.optim.AdamW(module.parameters(),lr=.0003)]
    trainer.state.fn=TrainerFn.FITTING;trainer.state.stage=RunningStage.TRAINING
    trainer.fit_loop.max_batches=1;trainer.fit_loop.epoch_loop.val_loop._max_batches=[1]
    trainer.fit_loop.epoch_progress.current.completed=105
    trainer.fit_loop.epoch_loop.automatic_optimization.optim_progress.optimizer.step.total.completed=356160
    trainer.callback_metrics.update({'train/loss_epoch':torch.tensor(10.),'train/loss_step':torch.tensor(8.),
        'val/loss':torch.tensor(8.6),'train/continuous_epoch':torch.tensor(.2),'val/continuous':torch.tensor(.3),
        'train/element_epoch':torch.tensor(2.),'val/element':torch.tensor(2.1),
        'train/jump_epoch':torch.tensor(7.8),'val/jump':torch.tensor(6.2)})
    # These are the real connector methods used by Trainer._run, not assignments
    # emulating them. They must remain callable under the no-training guard.
    trainer._callback_connector._attach_model_callbacks()
    trainer._callback_connector._attach_model_logging_functions()
    callback=next(c for c in trainer.callbacks if isinstance(c,TrainingProgress))
    assert callback.log.__self__ is module and callback.log_dict.__self__ is module
    call._call_callback_hooks(trainer,'setup',stage='fit')
    return trainer,module,callback


def records(tmp_path):
    return [json.loads(line) for line in (tmp_path/'model.steps.jsonl').read_text(encoding='utf-8').splitlines()]


@pytest.mark.parametrize('tty',[False,True])
def test_native_binding_start_batch_validation_and_finish_without_training(tmp_path,monkeypatch,tty):
    trainer,module,callback=bound_trainer(tmp_path,monkeypatch,tty)
    before=copy.deepcopy(module.state_dict());optimizer=copy.deepcopy(trainer.optimizers[0].state_dict())
    rng=torch.get_rng_state().clone();py=random.getstate();numpy=np.random.get_state()
    call._call_callback_hooks(trainer,'on_train_start')
    call._call_callback_hooks(trainer,'on_train_epoch_start')
    cache_progress_losses(module,'train',dict(continuous=torch.tensor(.2),element=torch.tensor(2.),jump=torch.tensor(5.8)))
    call._call_callback_hooks(trainer,'on_train_batch_end',{'loss':torch.tensor(8.)},[{}],0)
    trainer.state.stage=RunningStage.VALIDATING
    call._call_callback_hooks(trainer,'on_validation_start')
    call._call_callback_hooks(trainer,'on_validation_batch_start',[{}],0,0)
    cache_progress_losses(module,'validation',dict(continuous=torch.tensor(.1),element=torch.tensor(.2),jump=torch.tensor(.45)))
    call._call_callback_hooks(trainer,'on_validation_batch_end',torch.tensor(.75),[{}],0,0)
    call._call_callback_hooks(trainer,'on_validation_end')
    trainer.state.stage=RunningStage.TRAINING
    call._call_callback_hooks(trainer,'on_train_epoch_end')
    call._call_callback_hooks(trainer,'on_train_end')
    call._call_callback_hooks(trainer,'on_fit_end')
    call._call_callback_hooks(trainer,'teardown',stage='fit')
    data=records(tmp_path)
    assert [r['event'] for r in data]==['start','step','step','epoch_end','complete']
    assert data[0]['epoch']==105 and data[0]['global_step']==356160
    assert data[1]['continuous']==pytest.approx(.2) and data[2]['jump']==pytest.approx(.45)
    assert data[-2]['loss']==10. and data[-2]['validation_loss']==pytest.approx(8.6)
    assert callable(callback.log) and callable(callback.log_dict) and callback.state_dict()=={}
    assert callback._step_log.stream is None
    assert trainer.global_step==356160 and trainer.current_epoch==105
    assert trainer.optimizers[0].state_dict()==optimizer
    assert all(torch.equal(value,module.state_dict()[key]) for key,value in before.items())
    assert module.weight.grad is None and torch.equal(torch.get_rng_state(),rng)
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],numpy[1])
    assert not list(tmp_path.glob('*.ckpt')) and not (tmp_path/'model.pt').exists()


@pytest.mark.parametrize('tty',[False,True])
def test_native_interrupt_retains_original_exception_after_callback_binding(tmp_path,monkeypatch,tty):
    trainer,module,callback=bound_trainer(tmp_path,monkeypatch,tty)
    error=RuntimeError('original error must survive callback cleanup')
    def fail_before_first_batch():
        call._call_callback_hooks(trainer,'on_train_start')
        raise error
    with pytest.raises(RuntimeError) as raised:
        call._call_and_handle_interrupt(trainer,fail_before_first_batch)
    assert raised.value is error
    data=records(tmp_path)
    assert [r['event'] for r in data]==['start','failed'] and data[-1]['error']==str(error)
    assert callable(callback.log) and callable(callback.log_dict) and callback.state_dict()=={}
    assert callback._step_log.stream is None
    assert not list(tmp_path.glob('*.ckpt'))


def test_bound_callback_can_append_after_failure_without_losing_prior_records(tmp_path,monkeypatch):
    trainer,module,callback=bound_trainer(tmp_path,monkeypatch)
    call._call_callback_hooks(trainer,'on_train_start')
    call._call_callback_hooks(trainer,'on_exception',RuntimeError('fixture'))
    previous=(tmp_path/'model.steps.jsonl').read_bytes()
    trainer,module,resumed=bound_trainer(tmp_path,monkeypatch)
    call._call_callback_hooks(trainer,'on_train_start')
    call._call_callback_hooks(trainer,'on_fit_end')
    assert (tmp_path/'model.steps.jsonl').read_bytes().startswith(previous)
    assert [r['event'] for r in records(tmp_path)]==['start','failed','start','complete']
    assert callback.state_dict()==resumed.state_dict()=={}
    assert callback._step_log.stream is None and resumed._step_log.stream is None
