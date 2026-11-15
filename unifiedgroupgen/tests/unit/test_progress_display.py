"""Read-only progress regression; no Trainer.fit or optimizer updates."""
import io
import json
import random
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unifiedgroupgen.common.progress import AlignedTqdm, PhaseClock, Progress, StepLog, track, watch
from unifiedgroupgen.pl_modules.progress import TrainingProgress, CompactProgressBar, progress_callbacks, cache_progress_losses


def rows(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def test_redirected_output_is_compact_and_step_file_has_no_terminal_codes(tmp_path, capsys):
    path=tmp_path/'sampling.progress.jsonl'
    with Progress('Generate',2,log=path,unit='structure') as display:
        display.observe(event='sampling_step',sampling_time=.5,t='50%',atoms=3)
        display.update(accepted=1,failed=0)
        display.update(accepted=1,failed=1)
    console=capsys.readouterr().out
    assert re.search(r'Generate\s+\|\s+2/2\s+\| complete',console)
    assert '\r' not in console and '\x1b' not in console
    data=rows(path)
    assert all(r['time'].endswith('+08:00') for r in data)
    assert [r['event'] for r in data]==['start','sampling_step','step','step','complete']
    assert [r['completed'] for r in data]==[0,0,1,2,2]
    assert data[1]['sampling_time']==.5 and data[-2]['failed']==1
    assert '\r' not in path.read_text() and '\x1b' not in path.read_text()


def test_tty_uses_in_place_refresh_without_drawing_into_log(tmp_path, monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):return True
    terminal=Terminal()
    monkeypatch.setattr('sys.stdout',terminal)
    with Progress('Evaluate',1,log=tmp_path/'metrics.progress.jsonl') as display:
        display.update()
        display.bar.refresh()
    assert '\r' in terminal.getvalue() and '100%' in terminal.getvalue()
    assert rows(tmp_path/'metrics.progress.jsonl')[-1]['event']=='complete'


def test_inner_sampling_steps_are_buffered_but_terminal_events_flush(tmp_path):
    calls=[]
    log=StepLog(tmp_path/'progress.jsonl')
    log.stream=SimpleNamespace(write=lambda text:calls.append('write'),
                               flush=lambda:calls.append('flush'),close=lambda:calls.append('close'))
    log.flushed+=60
    log.write('Generate','sampling_step',sampling_time=.5)
    assert calls==['write']
    log.write('Generate','complete')
    assert calls==['write','write','flush']
    log.close()
    assert calls[-1]=='close' and log.stream is None


@pytest.mark.parametrize('exception',[RuntimeError('fixture'),KeyboardInterrupt()])
def test_exception_keeps_partial_count_and_closes_log(tmp_path, exception):
    path=tmp_path/'steps.jsonl'
    with pytest.raises(type(exception)):
        with Progress('Evaluate',3,log=path) as display:
            display.update()
            raise exception
    assert display.log.stream is None
    assert rows(path)[-1]['completed']==1
    assert rows(path)[-1]['event']==('failed' if isinstance(exception,RuntimeError) else 'interrupted')


def test_tracking_keeps_order_counts_continues_and_all_rng_states(tmp_path):
    py=random.getstate();np_state=np.random.get_state();state=torch.get_rng_state().clone()
    seen=[]
    for item in track([2,0,1],'Metrics',log=tmp_path/'metrics.jsonl'):
        seen.append(item)
        if not item:continue
    assert seen==[2,0,1] and rows(tmp_path/'metrics.jsonl')[-1]['completed']==3
    assert random.getstate()==py and torch.equal(state,torch.get_rng_state())
    current=np.random.get_state()
    assert current[0]==np_state[0] and np.array_equal(current[1],np_state[1]) and current[2:]==np_state[2:]


def trainer():
    return SimpleNamespace(is_global_zero=True,sanity_checking=False,current_epoch=9,
        global_step=30529,num_training_batches=3392,max_epochs=1000,
        optimizers=[SimpleNamespace(param_groups=[{'lr':.00015}])],
        callback_metrics={'train/loss_epoch':torch.tensor(10.),'train/loss_step':torch.tensor(8.),
                          'val/loss':torch.tensor(8.6)})


def test_lightning_records_every_batch_validation_and_append_only_resume(tmp_path,capsys):
    output=tmp_path/'model.pt';t=trainer();cb=TrainingProgress(output)
    module=SimpleNamespace(runtime={'batch_size':8})
    rng=torch.get_rng_state().clone()
    cb.on_train_start(t,module)
    cache_progress_losses(module,'train',dict(continuous=torch.tensor(.2),element=torch.tensor(1.),jump=torch.tensor(1.8)))
    cb.on_train_batch_end(t,module,{'loss':torch.tensor(3.,requires_grad=True)},[{},{}],0)
    t.global_step+=1
    cb.on_train_batch_end(t,module,torch.tensor(2.),[],1)
    cache_progress_losses(module,'validation',dict(continuous=torch.tensor(.1),element=torch.tensor(.2),jump=torch.tensor(.7)))
    cb.on_validation_batch_end(t,module,torch.tensor(1.),[{}],0)
    cb.on_train_epoch_end(t,module)
    cb.on_fit_end(t,None)
    path=output.with_suffix('.steps.jsonl');previous=path.read_bytes();data=rows(path)
    assert data[0]['global_step']==30529 and data[0]['epoch']==9
    assert [r['batch'] for r in data if r['phase']=='train' and r['event']=='step']==[1,2]
    assert next(r for r in data if r['phase']=='validation')['loss']==1.
    first=next(r for r in data if r['event']=='step' and r['phase']=='train')
    assert first['continuous']==pytest.approx(.2) and first['element']==1.
    assert first['jump']==pytest.approx(1.8) and first['batch_size']==8 and first['actual_batch_size']==2
    assert next(r for r in data if r['phase']=='validation')['jump']==pytest.approx(.7)
    assert re.search(r'Epoch 10/1000 \(bs=8\)\s+\| step=\s*30530',capsys.readouterr().out)
    resumed=TrainingProgress(output);resumed.on_train_start(t,module);resumed.on_fit_end(t,module)
    assert path.read_bytes().startswith(previous)
    assert torch.equal(rng,torch.get_rng_state()) and cb.state_dict()=={}
    assert not list(tmp_path.glob('*.ckpt')) and not output.exists()


def test_nonfinite_loss_is_valid_json_and_error_is_recorded(tmp_path):
    t=trainer();cb=TrainingProgress(tmp_path/'model.pt')
    cb.on_train_batch_end(t,None,torch.tensor(float('nan')),[],0)
    cb.on_exception(t,None,RuntimeError('original failure'))
    data=rows(tmp_path/'model.steps.jsonl')
    assert data[0]['loss'] is None and data[-1]['event']=='failed'
    assert data[-1]['error']=='original failure'


def test_rank_nonzero_does_not_write_duplicate_records(tmp_path,capsys):
    t=trainer();t.is_global_zero=False
    cb=TrainingProgress(tmp_path/'model.pt')
    cb.on_train_start(t,None);cb.on_train_batch_end(t,None,torch.tensor(1.),[],0)
    cb.on_train_epoch_end(t,None);cb.on_fit_end(t,None)
    assert not list(tmp_path.iterdir()) and not capsys.readouterr().out


@pytest.mark.parametrize('tty',[False,True])
def test_native_lightning_bar_is_only_enabled_for_a_terminal(tmp_path,monkeypatch,tty):
    monkeypatch.setattr('unifiedgroupgen.pl_modules.progress.interactive',lambda:tty)
    callbacks=progress_callbacks(tmp_path/'model.pt')
    assert sum(isinstance(cb,CompactProgressBar) for cb in callbacks)==int(tty)
    assert all(cb.state_dict()=={} for cb in callbacks)
    if tty:
        fields=callbacks[-1].get_metrics(trainer(),None)
        assert set(fields)=={'loss','geom','elem','jump','lr'} and 'v_num' not in fields


def test_progress_observer_does_not_change_joint_sampling_results_or_rng(tmp_path):
    from unifiedgroupgen.tests.unit.test_joint_flow_v2 import config
    from unifiedgroupgen.pl_modules.factory import build_joint_model
    from unifiedgroupgen.group_constraints.unified import catalog_spec
    from unifiedgroupgen.generation.unified.sampling import sample_batch
    torch.manual_seed(91);model=build_joint_model(config('equiformer_v2','dng'))
    specs=[catalog_spec(kind,1) for kind in ('space','layer','plane')]
    before=torch.get_rng_state().clone()
    baseline=sample_batch(model,specs,seeds=[7,8,9],steps=3,budget=6,epsilon=.005)
    after=torch.get_rng_state().clone()
    # The unchanged native EqV2 core uses the global RNG for edge frames.
    # Replay that existing stream as well as the sampler's per-structure seeds.
    torch.set_rng_state(before)
    with Progress('Generate',3,log=tmp_path/'sampling.progress.jsonl') as display:
        def observe(i,stage,state,clock,step,event,reason):
            display.observe(event='sampling_step',sample_index=i,sampling_time=clock,stage=stage)
        observed=sample_batch(model,specs,seeds=[7,8,9],steps=3,budget=6,epsilon=.005,trajectory_callback=observe)
        display.update(3)
    for a,b in zip(baseline,observed):
        torch.testing.assert_close(a.state.metric,b.state.metric,rtol=0,atol=0)
        torch.testing.assert_close(a.state.coordinates,b.state.coordinates,rtol=0,atol=0)
        assert a.state.elements==b.state.elements and a.state.letters==b.state.letters
        assert a.time==b.time and a.events==b.events and a.reason==b.reason
    assert torch.equal(after,torch.get_rng_state()) and model.training


def test_existing_job_viewer_is_read_only_and_ctrl_c_does_not_change_status(tmp_path,monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):return True
    terminal=Terminal();monkeypatch.setattr('sys.stdout',terminal)
    path=tmp_path/'model.status.json'
    path.write_text(json.dumps(dict(state='running',phase='training',epoch=53,max_epochs=1000,
        global_step=181750,loss=10.71,validation_loss=8.68,updated=1794672666,batch_size=8)),encoding='utf-8')
    original=path.read_bytes();stamp=path.stat().st_mtime_ns
    def interrupt(seconds):raise KeyboardInterrupt()
    monkeypatch.setattr('unifiedgroupgen.common.progress.time.sleep',interrupt)
    watch([path])
    assert path.read_bytes()==original and path.stat().st_mtime_ns==stamp
    assert '181750' in terminal.getvalue() and 'training' in terminal.getvalue() and '(bs=8)' in terminal.getvalue()


@pytest.mark.parametrize('state',['complete','failed','interrupted'])
def test_existing_job_viewer_exits_when_every_task_is_finished(tmp_path,monkeypatch,state):
    class Terminal(io.StringIO):
        def isatty(self):return True
    terminal=Terminal();monkeypatch.setattr('sys.stdout',terminal)
    path=tmp_path/'model.status.json'
    path.write_text(json.dumps(dict(state=state,max_epochs=1000,
        **({'epochs_completed':60} if state=='complete' else {'epoch':60}))),encoding='utf-8')
    monkeypatch.setattr('unifiedgroupgen.common.progress.time.sleep',lambda _:pytest.fail('Do not wait after completion'))
    watch([path])
    assert '60/1000' in terminal.getvalue()


def test_existing_job_viewer_requires_a_terminal_and_creates_no_files(tmp_path,monkeypatch):
    monkeypatch.setattr('unifiedgroupgen.common.progress.interactive',lambda:False)
    with pytest.raises(ValueError,match='interactive terminal'):
        watch([tmp_path/'missing.json'])
    assert not list(tmp_path.iterdir())


def test_all_phase_bars_and_metric_columns_are_aligned_and_pipe_separated():
    rendered=[]
    for label,total,completed in [('Train 54/1000 (bs=8)',3392,1290),('Validate 1000/1000 (bs=128)',1131,250),
                                   ('Generate',256,160),('Evaluate structures',256,218)]:
        bar=AlignedTqdm(total=total,initial=completed,desc=label,file=io.StringIO(),
                        ncols=300,dynamic_ncols=False)
        bar.set_postfix(dict(loss='10.7100',val='8.6800',geom='0.2376',elem='4.9743',jump='5.5013',lr='0.00015'),refresh=False)
        rendered.append(str(bar));bar.close()
    assert '(bs=128)' in rendered[1]
    for marker in ('% |','| time=','| ETA=','| loss=','| val=','| geom=','| elem=','| jump=','| lr='):
        assert len({line.index(marker) for line in rendered})==1
    assert all(', loss=' not in line and ', val=' not in line for line in rendered)


@pytest.mark.parametrize('mode',['fit','validate'])
def test_native_validation_bar_updates_count_and_batch_loss_without_fit(monkeypatch,mode):
    class Terminal(io.StringIO):
        def isatty(self):return True
    terminal=Terminal();monkeypatch.setattr('sys.stdout',terminal)
    t=trainer();t.state=SimpleNamespace(fn=mode);t.max_steps=-1
    t.num_training_batches=4;t.num_val_batches=[2];t.num_sanity_val_batches=[1]
    module=SimpleNamespace(runtime={'batch_size':8})
    cache_progress_losses(module,'train',dict(continuous=torch.tensor(.2),element=torch.tensor(2.),jump=torch.tensor(5.8)))
    cb=CompactProgressBar(refresh_rate=1,leave=False)
    before=torch.get_rng_state().clone()
    cb.setup(t,module,'fit' if mode=='fit' else 'validate')
    if mode=='fit':
        cb.on_train_start(t,module);cb.on_train_epoch_start(t,module)
        header=cb._epoch_bar
        initial_header=str(header)
        assert re.search(r'\| loss=\s*- \| val=\s*-',initial_header)
        assert [abs(bar.pos) for bar in (header,cb.train_progress_bar)]==[0,1]
        assert header.desc=='Epoch 10/1000 (bs=8)'
        assert getattr(cb,'_val_progress_bar',None) is None
        content=terminal.getvalue()
        assert content.index('Epoch 10/1000 (bs=8)')<content.index('Train') and 'Validate' not in content
        cb.on_train_batch_end(t,module,{'loss':torch.tensor(8.)},[],0)
        training=str(cb.train_progress_bar)
    cb.on_validation_start(t,module)
    if mode=='fit':
        assert [abs(bar.pos) for bar in (header,cb.train_progress_bar,cb.val_progress_bar)]==[0,1,2]
        assert not header.disable and not cb.train_progress_bar.disable and not cb.val_progress_bar.disable
    cb.on_validation_batch_start(t,module,[],0)
    cache_progress_losses(module,'validation',dict(continuous=torch.tensor(.1),element=torch.tensor(.2),jump=torch.tensor(.45)))
    cb.on_validation_batch_end(t,module,torch.tensor(.75),[],0)
    validation=str(cb.val_progress_bar)
    assert isinstance(cb.val_progress_bar,AlignedTqdm)
    assert cb.val_progress_bar.n==1 and cb.val_progress_bar.total==2
    assert ('Validate' if mode=='fit' else 'Validate 10/1000 (bs=8)') in validation and '50%' in validation
    assert re.search(r'\| loss=\s*0.7500',validation)
    assert cb.get_metrics(t,module)['geom']=='0.1000' and cb.get_metrics(t,module)['jump']=='0.4500'
    assert cb.get_metrics(t,module)['loss']=='0.7500' and 'val' not in cb.get_metrics(t,module)
    if mode=='fit':
        assert training.index('% |')==validation.index('% |')
    cb.on_validation_end(t,module)
    assert cb.get_metrics(t,module)['loss']=='8.0000' and cb.get_metrics(t,module)['geom']=='0.2000'
    if mode=='fit':
        assert cb.val_progress_bar.disable and re.search(r'\| loss=\s*8.6000',str(cb.val_progress_bar))
        assert not header.disable and not cb.train_progress_bar.disable
        completed=cb.val_progress_bar
        cb.on_train_epoch_end(t,module)
        assert re.search(r'\| loss=\s*10.0000',str(cb.train_progress_bar))
        assert re.search(r'\| val=\s*8.6000',str(header))
        for marker in ('| step=','| loss=','| val=','| best=','| early=','| lr='):
            assert initial_header.index(marker)==str(header).index(marker)
        t.current_epoch+=1
        before_next_epoch=len(terminal.getvalue())
        cb.on_train_epoch_start(t,module)
        assert cb._epoch_bar is header and 'Epoch 11/1000 (bs=8)'==header.desc
        assert cb.val_progress_bar is completed and completed.disable
        cb.on_train_epoch_end(t,module)
        assert 'Validate' not in terminal.getvalue()[before_next_epoch:]
        cb.on_train_end(t,None)
        assert header.disable and cb.train_progress_bar.disable and cb.val_progress_bar.disable
    assert cb.state_dict()=={} and torch.equal(before,torch.get_rng_state())


@pytest.mark.parametrize('exception',[RuntimeError('fixture'),KeyboardInterrupt()])
@pytest.mark.parametrize('during_validation',[False,True])
def test_three_row_display_closes_cleanly_on_exception_without_touching_training(monkeypatch,exception,during_validation):
    class Terminal(io.StringIO):
        def isatty(self):return True
    monkeypatch.setattr('sys.stdout',Terminal())
    t=trainer();t.state=SimpleNamespace(fn='fit');t.num_val_batches=[2]
    module=SimpleNamespace(runtime={'batch_size':8})
    cb=CompactProgressBar(refresh_rate=1,leave=False);cb.setup(t,module,'fit')
    before=torch.get_rng_state().clone()
    cb.on_train_start(t,module);cb.on_train_epoch_start(t,module)
    if during_validation:cb.on_validation_start(t,module)
    cb.on_exception(t,module,exception);cb.teardown(t,module,'fit')
    bars=(cb._epoch_bar,cb.train_progress_bar,getattr(cb,'_val_progress_bar',None))
    assert all(bar is None or bar.disable for bar in bars)
    assert t.global_step==30529 and cb.state_dict()=={} and torch.equal(before,torch.get_rng_state())


def test_interactive_epoch_summary_updates_header_without_appending_another_line(tmp_path,monkeypatch):
    class Terminal(io.StringIO):
        def isatty(self):return True
    terminal=Terminal();monkeypatch.setattr('sys.stdout',terminal)
    t=trainer();t.state=SimpleNamespace(fn='fit');t.num_val_batches=[2]
    module=SimpleNamespace(runtime={'batch_size':8})
    display=CompactProgressBar(refresh_rate=1,leave=False)
    logger=TrainingProgress(tmp_path/'model.pt');t.callbacks=[logger,display]
    display.setup(t,module,'fit');display.on_train_start(t,module);display.on_train_epoch_start(t,module)
    previous=terminal.getvalue()
    logger.on_train_epoch_end(t,module)
    assert terminal.getvalue()==previous
    display.on_train_epoch_end(t,module)
    assert re.search(r'\| val=\s*8.6000',str(display._epoch_bar))
    assert re.search(r'\| best=\s*8.6000',str(display._epoch_bar))
    logger.on_fit_end(t,module);display.on_train_end(t,module)
    assert rows(tmp_path/'model.steps.jsonl')[0]['event']=='epoch_end'


def test_loss_observations_leave_step_returns_parameters_and_gradients_unchanged():
    from unifiedgroupgen.pl_modules.unified.module import JointGeneratorModule
    module=torch.nn.Linear(1,1)
    values={key:module.weight.sum()*factor for key,factor in [('continuous',1),('element',2),('jump',3)]}
    values['loss']=sum(values.values())
    module.objective=lambda *args,**kwargs:values
    module.log_dict=lambda *args,**kwargs:None
    before={k:v.clone() for k,v in module.state_dict().items()}
    train=JointGeneratorModule.training_step(module,[{}],0)
    validation=JointGeneratorModule.validation_step(module,[{}],0)
    assert train is values['loss'] and torch.equal(validation,values['loss']) and not validation.requires_grad
    assert all(not value.requires_grad for parts in module._progress_losses.values() for value in parts.values())
    assert set(before)==set(module.state_dict()) and module.weight.grad is None
    for k,v in before.items():torch.testing.assert_close(v,module.state_dict()[k],rtol=0,atol=0)


def test_epoch_summary_uses_current_best_before_checkpoint_callback_and_current_early_count(tmp_path,capsys):
    import copy
    from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
    t=trainer();module=SimpleNamespace(runtime={'batch_size':8})
    early=EarlyStopping('val/loss',patience=60,min_delta=.0001,check_on_train_epoch_end=False)
    checkpoint=ModelCheckpoint(dirpath=tmp_path,monitor='val/loss',mode='min')
    checkpoint.best_model_score=torch.tensor(9.)
    early.best_score=torch.tensor(9.);early.wait_count=6
    t.callbacks=[early,checkpoint]
    early._evaluate_stopping_criteria(t.callback_metrics['val/loss'])
    old=copy.deepcopy(checkpoint.state_dict());early_old=copy.deepcopy(early.state_dict())
    cb=TrainingProgress(tmp_path/'model.pt')
    cb.on_train_epoch_end(t,module)
    assert checkpoint.state_dict()==old and early.state_dict()==early_old
    checkpoint.best_model_score=t.callback_metrics['val/loss'].clone()
    t.callback_metrics['val/loss']=torch.tensor(8.8)
    early._evaluate_stopping_criteria(t.callback_metrics['val/loss'])
    cb.on_train_epoch_end(t,module)
    cb.on_fit_end(t,module)
    data=[r for r in rows(tmp_path/'model.steps.jsonl') if r['event']=='epoch_end']
    assert data[0]['best_validation_loss']==pytest.approx(8.6) and data[0]['early_stopping']['wait_count']==0
    assert data[1]['best_validation_loss']==pytest.approx(8.6) and data[1]['early_stopping']==dict(wait_count=1,patience=60)
    text=capsys.readouterr().out
    assert re.search(r'\| best=\s*8.6000 \| early=\s*0/60',text)
    assert re.search(r'\| early=\s*1/60',text)
    assert not list(tmp_path.glob('*.ckpt'))


def test_early_stopping_min_delta_does_not_hide_a_new_checkpoint_best(tmp_path):
    from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint
    t=trainer();t.callback_metrics['val/loss']=torch.tensor(8.59995)
    early=EarlyStopping('val/loss',patience=60,min_delta=.0001,check_on_train_epoch_end=False)
    early.best_score=torch.tensor(8.6)
    checkpoint=ModelCheckpoint(dirpath=tmp_path,monitor='val/loss',mode='min')
    checkpoint.best_model_score=torch.tensor(8.6)
    t.callbacks=[early,checkpoint]
    early._evaluate_stopping_criteria(t.callback_metrics['val/loss'])
    cb=TrainingProgress(tmp_path/'model.pt');cb.on_train_epoch_end(t,None);cb.on_fit_end(t,None)
    row=rows(tmp_path/'model.steps.jsonl')[0]
    assert row['best_validation_loss']==pytest.approx(8.59995) and row['early_stopping']['wait_count']==1


def test_missing_parts_are_not_invented_and_zero_csp_element_loss_is_visible():
    cb=CompactProgressBar();t=trainer()
    assert cb.get_metrics(t,None)['elem']=='-'
    module=SimpleNamespace(runtime={'batch_size':4})
    cache_progress_losses(module,'train',dict(continuous=torch.tensor(.2),element=torch.tensor(0.),jump=torch.tensor(4.)))
    assert cb.get_metrics(t,module)['elem']=='0.0000'


def test_epoch_summary_marks_disabled_early_stopping_and_logs_epoch_parts(tmp_path,capsys):
    t=trainer();t.callbacks=[]
    t.callback_metrics.update({'train/continuous_epoch':torch.tensor(.2),'val/continuous':torch.tensor(.3),
                               'train/element_epoch':torch.tensor(0.),'val/element':torch.tensor(0.),
                               'train/jump_epoch':torch.tensor(4.),'val/jump':torch.tensor(4.1)})
    cb=TrainingProgress(tmp_path/'model.pt');cb.on_train_epoch_end(t,None);cb.on_fit_end(t,None)
    record=rows(tmp_path/'model.steps.jsonl')[0]
    assert record['early_stopping'] is None and record['loss_parts']['train']['element']==0.
    assert record['loss_parts']['val']['jump']==pytest.approx(4.1)
    assert re.search(r'\| early=\s*off',capsys.readouterr().out)


def test_phase_clock_excludes_paused_intervals_and_resets_without_sleep(monkeypatch):
    now=[1000.]
    monkeypatch.setattr('unifiedgroupgen.common.progress.time.monotonic',lambda:now[0])
    clock=PhaseClock();now[0]+=50
    assert clock.elapsed==0.
    clock.reset();now[0]+=7;clock.pause();now[0]+=100
    assert clock.elapsed==7.
    clock.pause();clock.resume();clock.resume();now[0]+=3
    assert clock.elapsed==10.
    clock.reset(running=False);now[0]+=50
    assert clock.elapsed==0.


@pytest.mark.parametrize('finish_train',[False,True])
def test_native_stage_times_rate_and_eta_freeze_during_validation(monkeypatch,finish_train):
    class Terminal(io.StringIO):
        def isatty(self):return True
    monkeypatch.setattr('sys.stdout',Terminal())
    now=[1000.]
    monkeypatch.setattr('unifiedgroupgen.common.progress.time.monotonic',lambda:now[0])
    t=trainer();t.state=SimpleNamespace(fn='fit');t.num_training_batches=4;t.num_val_batches=[2]
    t.num_sanity_val_batches=[0]
    module=SimpleNamespace(runtime={'batch_size':8})
    cache_progress_losses(module,'train',dict(continuous=torch.tensor(.2),element=torch.tensor(2.),jump=torch.tensor(5.8)))
    cb=CompactProgressBar(refresh_rate=1,leave=False);cb.setup(t,module,'fit')
    rng=torch.get_rng_state().clone()
    cb.on_train_start(t,module);cb.on_train_epoch_start(t,module)
    now[0]+=10
    cb.on_train_batch_end(t,module,torch.tensor(8.),[],3 if finish_train else 0)
    train_bar=cb.train_progress_bar;before=str(train_bar);before_values=train_bar.format_dict
    assert before_values['elapsed']==10.
    assert before_values['rate']==pytest.approx((4 if finish_train else 1)/10)
    cb.on_validation_start(t,module);cb.on_validation_batch_start(t,module,[],0)
    now[0]+=90
    cache_progress_losses(module,'validation',dict(continuous=torch.tensor(.1),element=torch.tensor(.2),jump=torch.tensor(.45)))
    cb.on_validation_batch_end(t,module,torch.tensor(.75),[],0)
    assert str(train_bar)==before
    assert train_bar.format_dict['elapsed']==10. and train_bar.format_dict['rate']==before_values['rate']
    val=cb.val_progress_bar
    assert val.format_dict['elapsed']==90. and val.format_dict['rate']==pytest.approx(1/90)
    assert '01:30' in str(val)
    train_line=str(train_bar);val_line=str(val)
    assert [i for i,c in enumerate(train_line) if c=='|']==[i for i,c in enumerate(val_line) if c=='|']
    assert train_line.index('| time=')<70 and train_line.index('| loss=')<120
    cb.on_validation_end(t,module);now[0]+=5
    assert train_bar.format_dict['elapsed']==(10. if finish_train else 15.)
    assert val.phase_clock.elapsed==90.
    if not finish_train:
        cb.on_train_batch_end(t,module,torch.tensor(8.),[],1)
        assert train_bar.format_dict['rate']==pytest.approx(2/15)
    cb.on_train_epoch_end(t,module);final=train_bar.format_dict.copy();now[0]+=100
    assert train_bar.format_dict['elapsed']==final['elapsed'] and train_bar.format_dict['rate']==final['rate']
    cb.on_train_epoch_start(t,module)
    assert train_bar.format_dict['elapsed']==0.
    cb.on_train_end(t,module)
    assert cb.state_dict()=={} and torch.equal(rng,torch.get_rng_state())


def test_noninteractive_logs_keep_separate_cumulative_phase_seconds(tmp_path,monkeypatch):
    now=[1000.]
    monkeypatch.setattr('unifiedgroupgen.common.progress.time.monotonic',lambda:now[0])
    t=trainer();t.num_training_batches=4;module=SimpleNamespace(runtime={'batch_size':8})
    cb=TrainingProgress(tmp_path/'model.pt');cb.on_train_start(t,module);cb.on_train_epoch_start(t,module)
    now[0]+=10;cb.on_train_batch_end(t,module,torch.tensor(8.),[{}],0)
    cb.on_validation_start(t,module);now[0]+=90
    cb.on_validation_batch_end(t,module,torch.tensor(.75),[{}],0);cb.on_validation_end(t,module)
    now[0]+=5;cb.on_train_batch_end(t,module,torch.tensor(8.),[{}],3)
    cb.on_validation_start(t,module);now[0]+=30
    cb.on_validation_batch_end(t,module,torch.tensor(.75),[{}],1);cb.on_validation_end(t,module)
    now[0]+=100;cb.on_train_epoch_end(t,module);cb.on_fit_end(t,module)
    data=rows(tmp_path/'model.steps.jsonl')
    assert [r['phase_seconds'] for r in data if r['event']=='step' and r['phase']=='train']==[10.,15.]
    assert [r['phase_seconds'] for r in data if r['event']=='step' and r['phase']=='validation']==[90.,120.]
    summary=next(r for r in data if r['event']=='epoch_end')
    assert summary['train_seconds']==15. and summary['validation_seconds']==120.


def test_standalone_validation_logger_never_starts_training_clock(tmp_path,monkeypatch):
    now=[1000.]
    monkeypatch.setattr('unifiedgroupgen.common.progress.time.monotonic',lambda:now[0])
    cb=TrainingProgress(tmp_path/'model.pt');t=trainer()
    cb.on_validation_start(t,None);now[0]+=3
    cb.on_validation_batch_end(t,None,torch.tensor(.5),[{}],0);cb.on_validation_end(t,None)
    now[0]+=20;cb._step_log.close()
    assert cb.train_clock.elapsed==0. and cb.validation_clock.elapsed==3.
    assert rows(tmp_path/'model.steps.jsonl')[0]['phase_seconds']==3.
