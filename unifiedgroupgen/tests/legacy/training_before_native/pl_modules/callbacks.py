import json
import sys
import time
import numpy as np
import torch
import pytorch_lightning as pl
from unifiedgroupgen.tests.legacy.training_before_native.common.checkpoints import save_checkpoint, save_status

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
        started = time.perf_counter()
        trainer.save_checkpoint(module.output.parent/'last.ckpt')
        if module.improved:
            trainer.save_checkpoint(module.output.parent/'best.ckpt')
        inference = {'config': module.config, 'proposal': module.proposal.state_dict(), 'flow': module.flow.state_dict(),
                     'epoch': trainer.current_epoch, 'runtime_options': module.runtime, 'best_val_loss': module.best,
                     'trainer_backend': 'lightning', 'resume_checkpoint': 'last.ckpt', **module.hashes}
        save_checkpoint(module.output, inference)
        if module.improved:
            save_checkpoint(module.output.with_suffix('.best.pt'), {**inference, 'resume_checkpoint': 'best.ckpt'})
        row['checkpoint_seconds'] = time.perf_counter()-started
        row['seconds'] = train_seconds+module.validation_seconds
        row['wall_seconds'] = time.perf_counter()-module.epoch_started
        row['other_seconds'] = row['wall_seconds']-row['seconds']-row['checkpoint_seconds']
        with module.output.with_suffix('.history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        module.log_dict({'timing/train_seconds': train_seconds, 'timing/validation_seconds': module.validation_seconds,
                         'timing/checkpoint_seconds': row['checkpoint_seconds'],
                         'memory/peak_allocated_gib': row['peak_cuda_bytes']/2**30,
                         'gradients/mean_preclip_norm': row['gradient_norm_mean'],
                         'gradients/clipped_fraction': row['gradient_clipped_fraction'], 'lr': row['learning_rate']},
                        on_step=False, on_epoch=True)
        save_status(module.output, {'state': 'running', 'phase': 'epoch_saved',
                    'completed_epochs': trainer.current_epoch+1, 'last': row})
        val = f'{module.val_loss:.4f}' if module.val_loss is not None else 'skipped'
        message = (f'Epoch {trainer.current_epoch+1}/{trainer.max_epochs} | train={row["train_loss"]:.4f} '
                   f'| val={val} | train_t={train_seconds:.1f}s | val_t={module.validation_seconds:.1f}s '
                   f'| save_t={row["checkpoint_seconds"]:.1f}s')
        with (module.output.parent/'train.log').open('a', encoding='utf-8') as stream:
            stream.write(message+'\n')
        if not sys.stdout.isatty():
            print(message, flush=True)
