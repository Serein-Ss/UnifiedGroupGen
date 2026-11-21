"""Compact native Lightning display plus per-batch, append-only observations."""
import sys

from pytorch_lightning.callbacks import Callback, TQDMProgressBar, EarlyStopping, ModelCheckpoint
from tqdm.auto import tqdm

from unifiedgroupgen.common.progress import AlignedTqdm, BAR_FORMAT, PhaseClock, StepLog, format_metrics, interactive, scalar


PARTS = ('continuous', 'element', 'jump')
PHASE_BAR_FORMAT = ('{desc:<10.10} | {percentage:3.0f}% |{bar:18}| {n_fmt:>5}/{total_fmt:<5}'
                    ' | time={elapsed:>8} | ETA={remaining:>8} | {rate_fmt:>10}{fields}')


def cache_progress_losses(module, phase, losses):
    """Keep detached observations only; do not change step returns or checkpoint state."""
    if not hasattr(module, '_progress_losses'):
        module._progress_losses = {}
    module._progress_losses[phase] = {key: losses[key].detach() for key in PARTS if key in losses}


def _parts(module, phase):
    values = getattr(module, '_progress_losses', {}).get(phase, {})
    return {key: scalar(values.get(key)) for key in PARTS}


def _batch_size(module):
    return getattr(module, 'runtime', {}).get('batch_size', '-')


def _label(phase, trainer, module):
    return f'{phase} {trainer.current_epoch+1}/{trainer.max_epochs} (bs={_batch_size(module)})'


def _epoch_monitor(trainer):
    callbacks = getattr(trainer, 'callbacks', ())
    checkpoint = next((cb for cb in callbacks if isinstance(cb, ModelCheckpoint) and cb.monitor), None)
    early = next((cb for cb in callbacks if isinstance(cb, EarlyStopping)), None)
    monitor = checkpoint.monitor if checkpoint else early.monitor if early else 'val/loss'
    scores = [scalar(trainer.callback_metrics.get(monitor))]
    if checkpoint is not None:
        # ModelCheckpoint runs after ordinary epoch callbacks. Include this
        # epoch's metric without mutating its state or reporting an old best.
        scores.append(scalar(checkpoint.best_model_score))
    elif early is not None:
        scores.append(scalar(early.best_score))
    scores = [value for value in scores if value is not None]
    mode = checkpoint.mode if checkpoint else early.mode if early else 'min'
    best = (max(scores) if mode == 'max' else min(scores)) if scores else None
    stopping = dict(wait_count=early.wait_count, patience=early.patience) if early else None
    return best, stopping, monitor


class CompactProgressBar(TQDMProgressBar):
    def _new_bar(self, description, position, leave=False, compact=False):
        return AlignedTqdm(desc=description, position=position, disable=self.is_disabled,
                           leave=leave, dynamic_ncols=True, file=sys.stdout, smoothing=0,
                           phase_timing=True,bar_format=PHASE_BAR_FORMAT if compact else BAR_FORMAT)

    def init_train_tqdm(self):
        return self._new_bar('Train',3*self.process_position+1,compact=True)

    def init_validation_tqdm(self):
        has_main_bar=self.trainer.state.fn!='validate'
        position=3*self.process_position+2 if has_main_bar else 3*self.process_position
        return self._new_bar('Validate',position,not has_main_bar,compact=has_main_bar)

    def init_test_tqdm(self):
        return self._new_bar('Test',2*self.process_position,True)

    def init_predict_tqdm(self):
        return self._new_bar('Predict',2*self.process_position,True)

    def on_train_start(self, trainer, pl_module):
        self._epoch_bar=AlignedTqdm(desc=_label('Epoch',trainer,pl_module),
            position=3*self.process_position,disable=self.is_disabled,leave=False,
            dynamic_ncols=True,file=sys.stdout,bar_format='{desc:<24.24}{fields}')
        super().on_train_start(trainer,pl_module)

    def update_epoch_header(self, trainer, pl_module, completed=False):
        header=getattr(self,'_epoch_bar',None)
        if header is None or header.disable:
            return
        best,stopping,_=_epoch_monitor(trainer)
        show=lambda value: f'{value:.4f}' if value is not None else '-'
        values=dict(step=trainer.global_step,loss='-',val='-')
        if completed:
            metrics=trainer.callback_metrics
            values.update(loss=show(scalar(metrics.get('train/loss_epoch',metrics.get('train/loss')))),
                          val=show(scalar(metrics.get('val/loss'))))
        values.update(best=show(best),early=f'{stopping["wait_count"]}/{stopping["patience"]}' if stopping else 'off',
                      lr=f'{trainer.optimizers[0].param_groups[0]["lr"]:.2g}' if trainer.optimizers else '-')
        header.set_description_str(_label('Epoch',trainer,pl_module),refresh=False)
        header.set_postfix(values,refresh=True)

    def on_train_epoch_start(self, trainer, pl_module):
        super().on_train_epoch_start(trainer, pl_module)
        if self.is_enabled:
            self.update_epoch_header(trainer,pl_module)
            self.train_progress_bar.set_description('Train',refresh=True)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_train_batch_end(trainer,pl_module,outputs,batch,batch_idx)
        if self.train_progress_bar.total is not None and batch_idx+1>=self.train_progress_bar.total:
            self.train_progress_bar.phase_clock.pause()
        if self.is_enabled and self._should_update(batch_idx+1,self.train_progress_bar.total):
            self.update_epoch_header(trainer,pl_module)

    def on_validation_start(self, trainer, pl_module):
        train=getattr(self,'_train_progress_bar',None)
        if isinstance(train,AlignedTqdm) and train.phase_clock is not None:
            train.phase_clock.pause()
            train.refresh()
        previous=getattr(self,'_val_progress_bar',None)
        if previous is not None and not trainer.sanity_checking:
            previous.close()
        self._validation_batch_loss=None
        self._validation_active=True
        super().on_validation_start(trainer,pl_module)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        super().on_validation_batch_start(trainer,pl_module,batch,batch_idx,dataloader_idx)
        if self.is_enabled:
            label='Validate' if getattr(self,'_epoch_bar',None) is not None else _label('Validate',trainer,pl_module)
            if self.val_progress_bar.desc.rstrip(': ')!=label:
                self.val_progress_bar.set_description(label,refresh=True)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        loss=outputs.get('loss') if isinstance(outputs,dict) else outputs
        self._validation_batch_loss=scalar(loss)
        if self.is_enabled:
            self.val_progress_bar.set_postfix(self.get_metrics(trainer,pl_module),refresh=False)
        super().on_validation_batch_end(trainer,pl_module,outputs,batch,batch_idx,dataloader_idx)

    def on_validation_end(self, trainer, pl_module):
        clock=getattr(self.val_progress_bar,'phase_clock',None)
        if clock is not None:
            clock.pause()
        self._validation_batch_loss=None
        self._validation_active=False
        if self.is_enabled:
            self.val_progress_bar.set_postfix(self.epoch_metrics(trainer,'val'),refresh=True)
        super().on_validation_end(trainer,pl_module)
        train=getattr(self,'_train_progress_bar',None)
        if isinstance(train,AlignedTqdm) and train.phase_clock is not None and train.total is not None and train.n<train.total:
            train.phase_clock.resume()

    def epoch_metrics(self, trainer, phase):
        metrics=trainer.callback_metrics
        result=dict(loss='-',geom='-',elem='-',jump='-',lr='-')
        for label,key in [('loss','loss'),('geom','continuous'),('elem','element'),('jump','jump')]:
            value=scalar(metrics.get(f'{phase}/{key}_epoch',metrics.get(f'{phase}/{key}')))
            if value is not None:
                result[label]=f'{value:.4f}'
        if trainer.optimizers:
            result['lr']=f'{trainer.optimizers[0].param_groups[0]["lr"]:.2g}'
        return result

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_progress_bar.phase_clock.pause()
        super().on_train_epoch_end(trainer,pl_module)
        if self.is_enabled:
            self.train_progress_bar.set_postfix(self.epoch_metrics(trainer,'train'),refresh=True)
            self.update_epoch_header(trainer,pl_module,completed=True)

    def _close_display(self):
        for name in ('_val_progress_bar','_train_progress_bar'):
            bar=getattr(self,name,None)
            if bar is not None:
                if isinstance(bar,AlignedTqdm) and bar.phase_clock is not None:
                    bar.phase_clock.pause()
                bar.close()
        header=getattr(self,'_epoch_bar',None)
        if header is not None:
            header.leave=True
            header.close()

    def on_train_end(self, trainer, pl_module):
        self._close_display()

    def on_exception(self, trainer, pl_module, exception):
        self._close_display()

    def teardown(self, trainer, pl_module, stage):
        self._close_display()

    def get_metrics(self, trainer, pl_module):
        values = trainer.callback_metrics
        phase='validation' if getattr(self,'_validation_active',False) else 'train'
        result = dict(loss='-',geom='-',elem='-',jump='-',lr='-')
        loss=(getattr(self,'_validation_batch_loss',None) if phase=='validation' else
              scalar(values.get('train/loss_step',values.get('train/loss'))))
        if loss is not None:
            result['loss']=f'{loss:.4f}'
        parts=_parts(pl_module,phase)
        for label,key in [('geom','continuous'),('elem','element'),('jump','jump')]:
            value=parts[key]
            if value is not None:
                result[label]=f'{value:.4f}'
        if trainer.optimizers:
            result['lr'] = f'{trainer.optimizers[0].param_groups[0]["lr"]:.2g}'
        return result


class TrainingProgress(Callback):
    """Stateless callback: never changes checkpoint keys, RNG or optimization."""
    def __init__(self, output):
        # Lightning binds callback.log/log_dict to the module during setup.
        self._step_log = StepLog(output.with_suffix('.steps.jsonl'))
        self.train_clock,self.validation_clock=PhaseClock(),PhaseClock()

    def on_train_epoch_start(self, trainer, pl_module):
        self.train_clock.reset()
        self.validation_clock.reset(running=False)
        self._last_train_batch=0
        self._train_epoch_active=True

    def on_validation_start(self, trainer, pl_module):
        self.train_clock.pause()
        self.validation_clock.resume()

    def on_validation_end(self, trainer, pl_module):
        self.validation_clock.pause()
        if getattr(self,'_train_epoch_active',False) and self._last_train_batch<trainer.num_training_batches:
            self.train_clock.resume()

    def on_train_start(self, trainer, pl_module):
        if trainer.is_global_zero:
            self._step_log.write('train', 'start', epoch=trainer.current_epoch, global_step=trainer.global_step,
                           batch_size=_batch_size(pl_module))

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._last_train_batch=batch_idx+1
        if self._last_train_batch>=trainer.num_training_batches:
            self.train_clock.pause()
        if trainer.is_global_zero:
            loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
            self._step_log.write('train', epoch=trainer.current_epoch, batch=batch_idx+1,
                           batches=trainer.num_training_batches, global_step=trainer.global_step,
                           loss=scalar(loss), lr=trainer.optimizers[0].param_groups[0]['lr'],
                           batch_size=_batch_size(pl_module), actual_batch_size=batch.num_graphs if hasattr(batch,'num_graphs') else len(batch),
                           phase_seconds=self.train_clock.elapsed, **_parts(pl_module,'train'))

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if trainer.is_global_zero and not trainer.sanity_checking:
            loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
            self._step_log.write('validation', epoch=trainer.current_epoch, batch=batch_idx+1,
                           dataloader=dataloader_idx, global_step=trainer.global_step, loss=scalar(loss),
                           batch_size=_batch_size(pl_module), actual_batch_size=batch.num_graphs if hasattr(batch,'num_graphs') else len(batch),
                           phase_seconds=self.validation_clock.elapsed, **_parts(pl_module,'validation'))

    def on_train_epoch_end(self, trainer, pl_module):
        self.train_clock.pause()
        self.validation_clock.pause()
        self._train_epoch_active=False
        if not trainer.is_global_zero:
            return
        metrics = trainer.callback_metrics
        train = scalar(metrics.get('train/loss_epoch', metrics.get('train/loss')))
        val = scalar(metrics.get('val/loss'))
        lr = trainer.optimizers[0].param_groups[0]['lr']
        best,stopping,monitor = _epoch_monitor(trainer)
        parts = {phase: {key: scalar(metrics.get(f'{phase}/{key}_epoch',metrics.get(f'{phase}/{key}')))
                         for key in PARTS} for phase in ('train','val')}
        self._step_log.write('train', 'epoch_end', epoch=trainer.current_epoch, global_step=trainer.global_step,
                       loss=train, validation_loss=val, lr=lr, batch_size=_batch_size(pl_module),
                       loss_parts=parts, best_validation_loss=best,early_stopping=stopping,validation_monitor=monitor,
                       train_seconds=self.train_clock.elapsed,validation_seconds=self.validation_clock.elapsed)
        show = lambda value: f'{value:.4f}' if value is not None else '-'
        label=_label('Epoch',trainer,pl_module)
        early=f'{stopping["wait_count"]}/{stopping["patience"]}' if stopping else 'off'
        displayed=any(isinstance(cb,CompactProgressBar) and getattr(cb,'_epoch_bar',None) is not None
                      and cb.is_enabled for cb in getattr(trainer,'callbacks',()))
        if not displayed:
            tqdm.write(f'{label:<24.24}'+format_metrics(dict(step=trainer.global_step,loss=show(train),
                       val=show(val),best=show(best),early=early,lr=f'{lr:.2g}')),file=sys.stdout)

    def on_fit_end(self, trainer, pl_module):
        self.train_clock.pause()
        self.validation_clock.pause()
        if trainer.is_global_zero:
            self._step_log.write('train', 'complete', epoch=trainer.current_epoch, global_step=trainer.global_step)
        self._step_log.close()

    def on_exception(self, trainer, pl_module, exception):
        self.train_clock.pause()
        self.validation_clock.pause()
        if trainer.is_global_zero:
            state = 'interrupted' if isinstance(exception, (KeyboardInterrupt, SystemExit)) else 'failed'
            self._step_log.write('train', state, epoch=trainer.current_epoch,
                           global_step=trainer.global_step, error=str(exception))
        self._step_log.close()


def progress_callbacks(output):
    callbacks = [TrainingProgress(output)]
    if interactive():
        callbacks.append(CompactProgressBar(refresh_rate=10, leave=False))
    return callbacks
