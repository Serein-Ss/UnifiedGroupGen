import json
import random
import time
from pathlib import Path
import torch
import pytorch_lightning as pl
from unifiedgroupgen.tests.gradients import gradient_diagnostics
from unifiedgroupgen.pl_modules.objectives.joint import joint_loss
from unifiedgroupgen.tests.legacy.training_before_native.common.checkpoints import save_status

from unifiedgroupgen.tests.legacy.training_before_native.common.rng import rng_state, restore_rng

class UnifiedGroupGenModule(pl.LightningModule):
    keys = ('descriptor', 'flow', 'lattice', 'coordinates')

    def __init__(self, proposal, flow, config, runtime, hashes, output):
        super().__init__()
        self.proposal, self.flow = proposal, flow
        self.config, self.runtime, self.hashes, self.output = config, runtime, hashes, Path(output)
        self.weights = config.get('loss_weights', {'descriptor': 1., 'flow': 1.})
        self.best, self.stale, self.last_validation = float('inf'), 0, -1
        self.scheduler = None
        self.pending_state = None
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
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.config.get('lr_patience', 10), factor=.5)
        if self.pending_state:
            self.scheduler.load_state_dict(self.pending_state['scheduler'])
        return optimizer

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def on_train_epoch_start(self):
        if self.pending_rng is not None:
            restore_rng(self.pending_rng)
            self.pending_rng = None
        self.epoch_started = time.perf_counter()
        self.train_totals = torch.zeros(4, dtype=torch.float64, device=self.device)
        self.norm_totals = torch.zeros(2, dtype=torch.float64, device=self.device)
        self.train_seen, self.train_batches = 0, 0
        self.validated, self.val_loss, self.improved = None, None, False
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
        norm = torch.nn.utils.clip_grad_norm_(self.parameters(), float('inf'), error_if_nonfinite=True)
        limit = self.config.get('gradient_clip', 10.)
        self.norm_totals += torch.stack((norm.detach().double(), (norm.detach() > limit).double()))
        if self.diagnostic is not None:
            self.diagnostic.update(total_preclip_norm=float(norm), clip_threshold=limit, clipped=bool(norm > limit))
            with self.output.with_suffix('.gradients.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(self.diagnostic, allow_nan=False)+'\n')
            for key in ('descriptor_weighted_norm', 'lattice_weighted_norm', 'coordinates_weighted_norm'):
                self.log('gradients/'+key, self.diagnostic[key], on_step=True, on_epoch=False)

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.get('gradient_clip', 10.), error_if_nonfinite=True)

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

    def on_validation_epoch_end(self):
        self.validated = dict(zip(self.keys, (self.val_totals/self.val_seen).cpu().tolist()))
        self.val_loss = sum(self.weights[key]*self.validated[key] for key in ('descriptor', 'flow'))
        if self.validation_is_fit:
            self.improved = self.val_loss < self.best
            gap = self.current_epoch-self.last_validation
            self.best, self.stale = (self.val_loss, 0) if self.improved else (self.best, self.stale+gap)
            if not self.scheduler.is_better(self.val_loss, self.scheduler.best):
                self.scheduler.num_bad_epochs += gap-1
            self.scheduler.step(self.val_loss)
            self.last_validation = self.current_epoch
            patience = self.config.get('early_stopping_patience', 30)
            if patience and self.stale >= patience:
                self.trainer.should_stop = True
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
        checkpoint['ugg_state'] = {'config': self.config, 'runtime': self.runtime, 'hashes': self.hashes,
                                   'best': self.best, 'stale': self.stale, 'last_validation': self.last_validation,
                                   'scheduler': self.scheduler.state_dict(), 'rng': rng_state()}

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get('ugg_state')
        if state is None:
            raise ValueError('Not a UnifiedGroupGen Lightning checkpoint; use --initialize-from for legacy weights')
        if state['config'] != self.config or state['runtime'] != self.runtime or state['hashes'] != self.hashes:
            raise ValueError('Resume configuration, runtime or split hashes differ')
        self.best, self.stale, self.last_validation = state['best'], state['stale'], state['last_validation']
        self.pending_state, self.pending_rng = state, state['rng']
