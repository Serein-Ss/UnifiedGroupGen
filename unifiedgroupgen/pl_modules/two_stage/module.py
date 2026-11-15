"""Two-stage Lightning module and its artifact callback."""
from pathlib import Path
import json, math, random, time
from datetime import datetime, timedelta, timezone
import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning.utilities import grad_norm
from pytorch_lightning.loggers import WandbLogger
from lightning_fabric.utilities.seed import _collect_rng_states as rng_state, _set_rng_states as restore_rng
from unifiedgroupgen.pl_modules.objectives.joint import joint_loss
from unifiedgroupgen.tests.gradients import gradient_diagnostics

def save_status(output, status):
    path = output.with_suffix('.status.json')
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
    temporary.replace(path)


class UnifiedGroupGenModule(pl.LightningModule):
    keys = ('descriptor', 'flow', 'lattice', 'coordinates')

    def __init__(self, proposal, flow, config, runtime, hashes, output):
        super().__init__()
        self.proposal, self.flow = proposal, flow
        self.config, self.runtime, self.hashes, self.output = config, runtime, hashes, Path(output)
        self.weights = config.get('loss_weights', {'descriptor': 1., 'flow': 1.})
        self.best = float('inf')
        self.pending_plateau = None
        self.pending_rng = None
        self.val_rng = None
        self.save_hyperparameters({'config': config, 'runtime': runtime, 'split_hashes': hashes})

    def objective(self, records, validation=False):
        losses = joint_loss(self.proposal, self.flow, records,
                            0 if validation else self.config.get('composition_probability', .5),
                            0 if validation else self.config.get('observation_probability', .25),
                            self.runtime['coordinate_length_scale'])
        total = self.weights['descriptor']*losses['descriptor'].mean()+self.weights['flow']*losses['flow'].mean()
        if not torch.isfinite(total):
            raise FloatingPointError('Nonfinite loss in records: '+str([row['id'] for row in records]))
        return losses, total

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        # joint_loss packs CPU chart endpoints and transfers the packed tensor itself.
        return batch

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.config.get('learning_rate', 1e-3))
        interval = self.runtime['validation_interval']
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.config.get('lr_patience', 10)//interval, factor=.5)
        return {'optimizer': optimizer, 'lr_scheduler': {'scheduler': scheduler, 'monitor': 'val/loss',
                'interval': 'epoch', 'frequency': interval, 'strict': False}}

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def on_train_epoch_start(self):
        # PL 2.5 saves epoch checkpoints before its plateau-scheduler update.
        # Replay only that recorded pending call after native optimizer/scheduler restore.
        if self.pending_plateau is not None:
            self.lr_schedulers().step(self.pending_plateau)
            self.pending_plateau = None
        if self.pending_rng is not None:
            restore_rng(self.pending_rng)
            self.pending_rng = None
        self.epoch_started = time.perf_counter()
        self.train_totals = torch.zeros(4, dtype=torch.float64, device=self.device)
        self.norm_totals = torch.zeros(2, dtype=torch.float64, device=self.device)
        self.train_seen, self.train_batches = 0, 0
        self.validated, self.val_loss = None, None
        self.validation_seconds = 0.
        self.training_seconds = None
        interval = self.runtime['validation_interval']
        self.trainer.check_val_every_n_epoch = 1 if self.current_epoch+1 == self.trainer.max_epochs else interval
        if self.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.device)

    def _log_parts(self, phase, losses, total, count):
        values = {phase+'/'+key: value.mean() for key, value in losses.items()}
        values[phase+'/coordinates_dimensionless'] = losses['coordinates'].mean()/self.runtime['coordinate_length_scale']**2
        values[phase+'/loss'] = total
        self.log_dict(values, on_step=False, on_epoch=True, batch_size=count)

    def training_step(self, batch, batch_idx):
        losses, total = self.objective(batch)
        self.train_totals += torch.stack([losses[key].detach().sum() for key in self.keys]).double()
        self.train_seen += len(batch)
        self.train_batches += 1
        self._log_parts('train', losses, total, len(batch))
        self.log('train/loss_step', total, on_step=True, on_epoch=False, prog_bar=True, batch_size=len(batch))
        every = self.config.get('gradient_diagnostics_interval', 0)
        self.diagnostic = (gradient_diagnostics(self.proposal, self.flow, losses, self.weights,
                           self.runtime['coordinate_length_scale'])
                           if every and (batch_idx == 0 or (batch_idx+1) % every == 0) else None)
        if self.diagnostic is not None:
            self.diagnostic.update(epoch=self.current_epoch, step=batch_idx+1,
                                   records=[{'dataset': r.get('dataset'), 'id': r['id']} for r in batch])
        if (batch_idx+1) % 25 == 0 or batch_idx == 0:
            save_status(self.output, {'event': 'progress', 'state': 'running', 'phase': 'train',
                        'epoch': self.current_epoch, 'step': batch_idx+1, 'steps': self.trainer.num_training_batches,
                        'loss': float(total.detach()), 'seconds': time.perf_counter()-self.epoch_started,
                        'completed_epochs': self.current_epoch})
        return total

    def on_before_optimizer_step(self, optimizer):
        norm = grad_norm(self, 2)['grad_2.0_norm_total'].to(self.device)
        if not torch.isfinite(norm):
            raise FloatingPointError('Nonfinite gradient')
        limit = self.config.get('gradient_clip', 10.)
        self.norm_totals += torch.stack((norm.detach().double(), (norm.detach() > limit).double()))
        if self.diagnostic is not None:
            self.diagnostic.update(total_preclip_norm=float(norm), clip_threshold=limit, clipped=bool(norm > limit))
            for key in ('descriptor_weighted_norm', 'lattice_weighted_norm', 'coordinates_weighted_norm'):
                self.log('gradients/'+key, self.diagnostic[key], on_step=True, on_epoch=False)


    def on_validation_epoch_start(self):
        if self.device.type == 'cuda':
            torch.cuda.synchronize(self.device)
        self.validation_is_fit = self.trainer.state.fn.value == 'fit'
        self.training_seconds = time.perf_counter()-self.epoch_started if self.validation_is_fit else 0.
        self.validation_started = time.perf_counter()
        self.val_rng = rng_state()
        random.seed(918); torch.manual_seed(918)
        self.val_totals = torch.zeros(4, dtype=torch.float64, device=self.device)
        self.val_seen = 0
        if self.validation_is_fit:
            save_status(self.output, {'state': 'running', 'phase': 'validation', 'epoch': self.current_epoch})

    def validation_step(self, batch, batch_idx):
        losses, total = self.objective(batch, validation=True)
        self.val_totals += torch.stack([losses[key].detach().sum() for key in self.keys]).double()
        self.val_seen += len(batch)
        self._log_parts('val', losses, total, len(batch))
        return total.detach()

    def on_validation_epoch_end(self):
        self.validated = dict(zip(self.keys, (self.val_totals/self.val_seen).cpu().tolist()))
        self.val_loss = sum(self.weights[key]*self.validated[key] for key in ('descriptor', 'flow'))
        if self.validation_is_fit:
            self.best = min(self.best, self.val_loss)
        restore_rng(self.val_rng)
        self.val_rng = None
        self.validation_seconds = time.perf_counter()-self.validation_started

    def on_test_epoch_start(self):
        self.test_rng = rng_state()
        random.seed(918); torch.manual_seed(918)

    def test_step(self, batch, batch_idx):
        losses, total = self.objective(batch, validation=True)
        self._log_parts('test', losses, total, len(batch))

    def on_test_epoch_end(self):
        restore_rng(self.test_rng)

    def on_save_checkpoint(self, checkpoint):
        # Lightning 2.5.5 saves loops/optimizers/callbacks, but not global RNG by default.
        checkpoint['ugg_state'] = {'format_version': 2, 'config': self.config, 'runtime': self.runtime,
                                   'hashes': self.hashes, 'best': self.best, 'rng': rng_state(),
                                   'wandb_id': self.logger.version if isinstance(self.logger, WandbLogger) else None,
                                   'pending_plateau': self.val_loss if self.val_loss is not None and
                                   self.lr_schedulers().last_epoch < (self.current_epoch+1)//self.runtime['validation_interval'] else None}

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get('ugg_state', {})
        if state.get('format_version') != 2:
            raise ValueError('Older training-state format; use initialize_from for weights-only initialization')
        if state['config'] != self.config or state['runtime'] != self.runtime or state['hashes'] != self.hashes:
            raise ValueError('Resume configuration, runtime or split hashes differ')
        self.best, self.pending_rng = state['best'], state['rng']
        self.pending_plateau = state.get('pending_plateau')


class TrainingArtifacts(pl.Callback):
    def on_train_epoch_end(self, trainer, module):
        if module.device.type == 'cuda':
            torch.cuda.synchronize(module.device)
        train_seconds = module.training_seconds or (time.perf_counter()-module.epoch_started)
        training = dict(zip(module.keys, (module.train_totals/module.train_seen).cpu().tolist()))
        for parts in (training, module.validated):
            if parts is not None:
                parts['coordinates_dimensionless'] = parts['coordinates']/module.runtime['coordinate_length_scale']**2
                parts['flow_raw'] = parts['lattice']+parts['coordinates']
        norm, clipped = module.norm_totals.cpu().tolist()
        row = {'epoch': trainer.current_epoch, 'train': training, 'val': module.validated,
               'train_loss': sum(module.weights[k]*training[k] for k in ('descriptor', 'flow')),
               'val_loss': module.val_loss, 'validated': module.validated is not None,
               'best_val_loss': module.best if np.isfinite(module.best) else None,
               'training_seconds': train_seconds, 'validation_seconds': module.validation_seconds,
               'learning_rate': trainer.optimizers[0].param_groups[0]['lr'],
               'gradient_norm_mean': norm/module.train_batches, 'gradient_clipped_fraction': clipped/module.train_batches,
               'coordinate_length_scale_angstrom': module.runtime['coordinate_length_scale'],
               'runtime_options': module.runtime, 'train_batches': module.train_batches,
               'mean_actual_batch': module.train_seen/module.train_batches,
               'train_records_per_second': module.train_seen/train_seconds,
               'train_records': len(trainer.datamodule.records), 'val_records': len(trainer.datamodule.validation),
               'peak_cuda_bytes': torch.cuda.max_memory_allocated(module.device) if module.device.type == 'cuda' else 0,
               'peak_cuda_reserved_bytes': torch.cuda.max_memory_reserved(module.device) if module.device.type == 'cuda' else 0}
        row['checkpoint_seconds'] = None
        row['checkpoint_timing_source'] = 'Lightning SimpleProfiler'
        row['seconds'] = train_seconds+module.validation_seconds
        row['wall_seconds'] = time.perf_counter()-module.epoch_started
        row['other_seconds'] = None
        with module.output.with_suffix('.history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        module.log_dict({'timing/train_seconds': train_seconds, 'timing/validation_seconds': module.validation_seconds,
                         'memory/peak_allocated_gib': row['peak_cuda_bytes']/2**30,
                         'gradients/mean_preclip_norm': row['gradient_norm_mean'],
                         'gradients/clipped_fraction': row['gradient_clipped_fraction'], 'lr': row['learning_rate']},
                        on_step=False, on_epoch=True)
        save_status(module.output, {'state': 'running', 'phase': 'epoch_saved',
                    'completed_epochs': trainer.current_epoch+1, 'last': row})
        val = f'{module.val_loss:.4f}' if module.val_loss is not None else 'skipped'
        message = (f'Epoch {trainer.current_epoch+1}/{trainer.max_epochs} | train={row["train_loss"]:.4f} '
                   f'| val={val} | train_t={train_seconds:.1f}s | val_t={module.validation_seconds:.1f}s '
                   f'| save_t=profiler')
        with (module.output.parent/'train.log').open('a', encoding='utf-8') as stream:
            stream.write(message+'\n')
