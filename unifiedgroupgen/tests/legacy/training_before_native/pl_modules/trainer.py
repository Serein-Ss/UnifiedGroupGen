import json
import sys
from pathlib import Path
import torch
import pytorch_lightning as pl
from unifiedgroupgen.common.source import file_hash
from unifiedgroupgen.tests.legacy.training_before_native.common.checkpoints import save_status

from unifiedgroupgen.run import build_loggers
from unifiedgroupgen.run import resume_checkpoint, runtime_options
from unifiedgroupgen.tests.legacy.training_before_native.common.rng import rng_state
from unifiedgroupgen.pl_data.datamodule import CrystalDataModule
from unifiedgroupgen.run import UnifiedGroupGenModule
from unifiedgroupgen.run import TrainingArtifacts

def run_lightning(args, config, records, validation, proposal, flow, train_paths, val_paths, output, settings,
                  finish_logging=True):
    output = Path(output)
    if not args.resume and any(path.exists() for path in (output, output.parent/'last.ckpt', output.parent/'best.ckpt',
                                                         output.parent/'logs/metrics.csv')):
        raise FileExistsError('Checkpoint already exists: '+str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {'train_split_sha256': [file_hash(p) for p in train_paths],
              'val_split_sha256': [file_hash(p) for p in val_paths]}
    resume, previous = resume_checkpoint(args.resume)
    runtime = runtime_options(args, config, flow, previous, hashes)
    metadata_path = output.parent/'run.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if resume and metadata.get('runtime_options') != runtime:
            raise ValueError('Resume runtime differs from run.json')
        metadata['runtime_options'] = runtime
        metadata_path.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    if getattr(args, 'initialize_from', None):
        if resume:
            raise ValueError('initialize-from and resume are mutually exclusive')
        from unifiedgroupgen.run import load_pretrained_models
        load_pretrained_models(proposal, flow, args.initialize_from)
    flow.checkpoint_layers, flow.edge_chunk_size = runtime['checkpoint_layers'], runtime['edge_chunk_size']
    data = CrystalDataModule(records, validation, runtime, getattr(args, 'seed', 42))
    module = UnifiedGroupGenModule(proposal, flow, config, runtime, hashes, output)
    mode = config.get('precision', 'fp32')
    if mode not in ('fp32', 'bf16', 'fp16'):
        raise ValueError('precision must be fp32, bf16 or fp16')
    if mode != 'fp32' and not str(args.device).startswith('cuda'):
        raise ValueError('Mixed precision requires CUDA')
    if mode == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('CUDA device does not support bf16')
    epochs = args.epochs if args.epochs is not None else config.get('epochs', 100)
    if epochs < 1:
        raise ValueError('epochs must be positive')
    loggers = build_loggers(args, output, settings, resume)
    initial_rng = rng_state()
    module.pending_rng = initial_rng
    trainer = pl.Trainer(accelerator='gpu' if str(args.device).startswith('cuda') else 'cpu',
                         devices=[torch.device(args.device).index or 0] if str(args.device).startswith('cuda') else 1,
                         max_epochs=epochs, precision={'fp32': '32-true', 'bf16': 'bf16-mixed', 'fp16': '16-mixed'}[mode],
                         logger=loggers, callbacks=[TrainingArtifacts()], default_root_dir=str(output.parent),
                         enable_checkpointing=False, enable_model_summary=False, num_sanity_val_steps=0,
                         check_val_every_n_epoch=runtime['validation_interval'], reload_dataloaders_every_n_epochs=1,
                         use_distributed_sampler=False, log_every_n_steps=25, enable_progress_bar=sys.stdout.isatty(),
                         gradient_clip_val=config.get('gradient_clip', 10.), inference_mode=False)
    success = False
    try:
        trainer.fit(module, datamodule=data, ckpt_path=str(resume) if resume else None)
        if trainer.interrupted:
            raise KeyboardInterrupt('Trainer interrupted; resume from the last completed epoch checkpoint')
        result = {'state': 'complete', 'reason': 'early_stopping' if trainer.should_stop else 'maximum_epochs',
                  'completed_epochs': trainer.current_epoch, 'best_val_loss': module.best,
                  'train_records': len(records), 'val_records': len(validation), 'trainer_backend': 'lightning',
                  'verification': 'Training finished; material stability and property performance not certified'}
        save_status(output, result)
        success = True
    except KeyboardInterrupt:
        save_status(output, {'state': 'interrupted', 'trainer_backend': 'lightning',
                             'reason': 'Trainer interrupted; resume from the last completed epoch checkpoint'})
        raise
    except BaseException as error:
        save_status(output, {'state': 'failed', 'trainer_backend': 'lightning', 'error': str(error)})
        raise
    finally:
        if settings.wandb_mode != 'disabled' and (finish_logging or not success):
            import wandb
            if wandb.run is not None:
                wandb.finish(exit_code=0 if success else 1)
    return trainer
