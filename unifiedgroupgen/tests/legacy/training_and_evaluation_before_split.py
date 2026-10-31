"""Hydra and native Lightning training orchestration."""
from pathlib import Path
import sys
if __package__ in (None, ''):
    sys.path[0] = str(Path(__file__).resolve().parent.parent)
import json, math, random, time, uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import hydra
import numpy as np
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger, WandbLogger
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.utilities import grad_norm
from lightning_fabric.utilities.seed import _collect_rng_states as rng_state, _set_rng_states as restore_rng
from unifiedgroupgen.common.configuration import project_root, resolve_path, RunSettings
from unifiedgroupgen.common.source import file_hash
from unifiedgroupgen.pl_modules.factory import build_models
from unifiedgroupgen.pl_modules.objectives.joint import joint_loss
from unifiedgroupgen.conditional.properties import fit_property_scalers
from unifiedgroupgen.pl_data.dataset import load_training_splits
from unifiedgroupgen.pl_data.datamodule import CrystalDataModule
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
            with self.output.with_suffix('.gradients.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps(self.diagnostic, allow_nan=False)+'\n')
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
        if not sys.stdout.isatty():
            print(message, flush=True)


def training_output(settings, experiment, output=None, resume=None):
    """Explicit historical outputs stay explicit; new default runs get a real start date."""
    if not experiment or Path(experiment).name != experiment or experiment in ('.', '..') or ':' in experiment or '\\' in experiment:
        raise ValueError('Experiment must be a single directory name')
    if output:
        output = Path(output).expanduser()
        output = (output if output.is_absolute() else settings.project_root/output).resolve()
    elif resume:
        output = Path(resume).resolve().parent/'last.ckpt'
    else:
        started = datetime.now(timezone(timedelta(hours=8)))
        directory = settings.jobs/experiment/started.strftime('%y-%m-%d-%H-%M-%S')
        directory.mkdir(parents=True, exist_ok=False)
        output = directory/'last.ckpt'
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata = output.parent/'run.json'
    if resume and not metadata.exists():
        raise ValueError('Resume needs a Lightning run.json; use --initialize-from to start from legacy weights')
    if not metadata.exists():
        started = datetime.now(timezone(timedelta(hours=8)))
        metadata.write_text(json.dumps({'experiment': experiment, 'started_at': started.isoformat(),
                            'directory_date_kind': 'training_run_start', 'trainer': 'pytorch-lightning',
                            'wandb_mode': settings.wandb_mode, 'wandb_id': uuid.uuid4().hex,
                            'output': str(output)}, indent=2)+'\n', encoding='utf-8')
    elif not resume:
        raise FileExistsError('Training directory already has run.json; use --resume: '+str(output.parent))
    return output


def build_loggers(args, output, settings, resume):
    # Lightning's CSV writer overwrites an existing metrics.csv. Keep every resume session.
    version = 'resume_'+uuid.uuid4().hex[:12] if resume else ''
    csv_logger = CSVLogger(save_dir=str(output.parent), name='logs', version=version)
    loggers = [csv_logger]
    if settings.wandb_mode != 'disabled':
        settings.wandb_dir.mkdir(parents=True, exist_ok=True)
        metadata_file = output.parent/'run.json'
        metadata = json.loads(metadata_file.read_text(encoding='utf-8')) if metadata_file.exists() else {}
        loggers.append(WandbLogger(project=getattr(args, 'wandb_project', None) or 'UnifiedGroupGen',
                                  name=metadata.get('experiment', output.parent.name),
                                  id=metadata.get('wandb_id'), save_dir=str(settings.wandb_dir),
                                  offline=settings.wandb_mode == 'offline', mode=settings.wandb_mode,
                                  log_model=False))
    return loggers


def resume_checkpoint(path):
    resume = None
    previous = None
    if path:
        resume = Path(path).resolve()
        previous = torch.load(resume, map_location='cpu', weights_only=False)
        if previous.get('ugg_state',{}).get('format_version') != 2:
            raise ValueError('Not a native Lightning checkpoint; use initialize_from for weights-only initialization')
    return resume, previous


def runtime_options(args, config, flow, previous, hashes):
    runtime = (dict(previous['ugg_state']['runtime']) if previous else
               {'checkpoint_layers': flow.checkpoint_layers, 'edge_chunk_size': flow.edge_chunk_size,
                'batch_size': config.get('batch_size', 8), 'edge_budget': config.get('edge_budget', 200000),
                'validation_interval': config.get('validation_interval', 5),
                'coordinate_length_scale': config.get('coordinate_length_scale', 3.)})
    for flag, key in [('checkpoint', 'checkpoint_layers'), ('edge_chunk_size', 'edge_chunk_size'),
                      ('batch_size', 'batch_size'), ('edge_budget', 'edge_budget'),
                      ('validation_interval', 'validation_interval'), ('coordinate_length_scale', 'coordinate_length_scale')]:
        value = getattr(args, flag, None)
        if value is not None:
            runtime[key] = value
    for key in ('batch_size', 'edge_budget', 'edge_chunk_size', 'validation_interval'):
        if not isinstance(runtime[key], int) or isinstance(runtime[key], bool) or runtime[key] < 1:
            raise ValueError(key+' must be a positive integer')
    if not np.isfinite(runtime['coordinate_length_scale']) or runtime['coordinate_length_scale'] <= 0:
        raise ValueError('Coordinate length scale must be finite and positive')
    for key in ('composition_probability', 'observation_probability'):
        if not 0 <= config.get(key, .5 if key == 'composition_probability' else .25) <= 1:
            raise ValueError(key+' must lie in [0,1]')
    if previous and (previous['ugg_state']['config'] != config or previous['ugg_state']['runtime'] != runtime
                     or previous['ugg_state']['hashes'] != hashes):
        raise ValueError('Resume configuration, runtime or split hashes differ')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('This trainer supports a single CPU or CUDA device')
    diagnostic_interval = config.get('gradient_diagnostics_interval', 0)
    if not isinstance(diagnostic_interval, int) or isinstance(diagnostic_interval, bool) or diagnostic_interval < 0:
        raise ValueError('gradient_diagnostics_interval must be a nonnegative integer')
    return runtime


def build_callbacks(config, runtime, directory):
    callbacks = [TrainingArtifacts(), LearningRateMonitor(logging_interval='epoch'),
        ModelCheckpoint(dirpath=directory, filename='best', monitor='val/loss', mode='min', save_top_k=1,
                        save_on_train_epoch_end=False, auto_insert_metric_name=False, enable_version_counter=False),
        ModelCheckpoint(dirpath=directory, save_top_k=0, save_last=True, every_n_epochs=1,
                        save_on_train_epoch_end=True, enable_version_counter=False)]
    patience = config.get('early_stopping_patience', 30)
    if patience:
        callbacks.append(EarlyStopping(monitor='val/loss', mode='min',
            patience=math.ceil(patience/runtime['validation_interval']), check_on_train_epoch_end=False))
    return callbacks


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
                         logger=loggers, callbacks=build_callbacks(config, runtime, output.parent), default_root_dir=str(output.parent),
                         profiler=SimpleProfiler(dirpath=output.parent, filename="profiler"), enable_model_summary=False, num_sanity_val_steps=0,
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


def fit_training_config(config, records):
    """Use the same TRAIN-only element domain and property scalers in training and timing."""
    config = dict(config)
    if config.get('element_policy', 'training') == 'training' and not config.get('allowed_elements'):
        config['allowed_elements'] = sorted({o['atomic_number'] for r in records for o in r['descriptor']['orbits']})
    config['properties'] = fit_property_scalers(records, config.get('properties', {}))
    if config.get('dataset_domains'):
        config['domain_label_support'] = {}
        for dataset, domain in config['dataset_domains'].items():
            rows = [r for r in records if r['dataset'] == dataset]
            config['domain_label_support'][str(domain)] = {
                'kinds': sorted({r['descriptor']['kind'] for r in rows}),
                'property_counts': {name: sum(r['properties'].get(name) is not None for r in rows)
                                    for name, specification in config['properties'].items() if not specification.get('context')}}
    return config

def load_pretrained_models(proposal, flow, path):
    """Weights only: accept inference .pt or native Lightning .ckpt, strictly."""
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if 'proposal' in saved and 'flow' in saved:
        proposal_weights, flow_weights = saved['proposal'], saved['flow']
    elif 'state_dict' in saved and 'ugg_state' in saved:
        weights = saved['state_dict']
        proposal_weights = {k.removeprefix('proposal.'): v for k, v in weights.items() if k.startswith('proposal.')}
        flow_weights = {k.removeprefix('flow.'): v for k, v in weights.items() if k.startswith('flow.')}
    else:
        raise ValueError('Pretrained weights must be a UnifiedGroupGen inference .pt or native .ckpt')
    proposal.load_state_dict(proposal_weights, strict=True)
    flow.load_state_dict(flow_weights, strict=True)

def train_args(args, config=None, resolved_config=None):
    """Common Python entry; no training logic is duplicated in command adapters."""

    root = project_root()
    if config is None:
        config = OmegaConf.to_container(OmegaConf.load(resolve_path(args.config, root)), resolve=True)
    records, validation, paths = load_training_splits(config, root, getattr(args, 'train', None), getattr(args, 'val', None))
    config = fit_training_config(config, records)
    args.resume = resolve_path(args.resume, root) if args.resume else None
    if getattr(args, 'initialize_from', None):
        args.initialize_from = resolve_path(args.initialize_from, root)
    proposal, flow = build_models(config, 'cpu')
    settings = RunSettings.load(root, getattr(args, 'wandb_mode', None))
    output = training_output(settings, args.experiment or config.get('experiment_id', 'training'), args.output, args.resume)
    if not args.resume:
        meta = output.parent/'run.json'
        metadata = json.loads(meta.read_text(encoding='utf-8'))
        metadata.update(seed=args.seed)
        if getattr(args, 'config', None):
            metadata['source_config_sha256'] = file_hash(resolve_path(args.config, root))
        meta.write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    if resolved_config is not None:
        # Preserve launch settings separately for each resume, too.
        conf = output.parent/'.hydra'
        conf.mkdir(exist_ok=True)
        filename = 'config.yaml' if not args.resume else 'resume_'+datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'.yaml'
        OmegaConf.save(resolved_config, conf/filename, resolve=True)
        OmegaConf.save(OmegaConf.create(config), conf/'training_config.yaml')
    print(json.dumps({'event': 'run_directory', 'trainer': 'lightning', 'output': str(output),
                      'wandb_mode': settings.wandb_mode}), flush=True)
    return run_lightning(args, config, records, validation, proposal, flow,
                         paths['train'], paths['val'], output, settings)


def execute(cfg: DictConfig):
    if cfg.task not in ('train', 'plan'):
        return execute_analysis(cfg)
    settings = RunSettings.load(project_root(), cfg.wandb.mode)
    if cfg.resume and cfg.initialize_from:
        raise ValueError('resume restores a run; initialize_from starts a new run. Choose one.')
    config = OmegaConf.to_container(cfg.experiment, resolve=True)
    profile = config.pop('runtime_profile', {})
    from hydra.core.hydra_config import HydraConfig
    selected = HydraConfig.get().runtime.choices.get('experiment', 'training') if HydraConfig.initialized() else 'training'
    name = cfg.experiment_name or config.get('experiment_id') or selected
    runtime = OmegaConf.to_container(cfg.runtime, resolve=True)
    if not cfg.resume:
        defaults = {**{key: config.get(key) for key in runtime}, **profile}
        runtime = {key: value if value is not None else defaults.get(key) for key, value in runtime.items()}
    cfg.runtime = OmegaConf.create(runtime)
    device = cfg.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = SimpleNamespace(config=None, trainer='lightning', experiment=name, device=device, seed=cfg.seed,
                           epochs=cfg.epochs, output=cfg.output, resume=cfg.resume, initialize_from=cfg.initialize_from,
                           wandb_mode=cfg.wandb.mode, wandb_project=cfg.wandb.project, train=None, val=None, **runtime)
    if cfg.task == 'plan':
        print(OmegaConf.to_yaml(cfg, resolve=True))
        print(json.dumps({'project_root':str(settings.project_root),'hydra_jobs':str(settings.jobs),'wandb_dir':str(settings.wandb_dir)}))
        print('PLAN ONLY: no model training, generation or output directory creation.')
        return
    if cfg.task != 'train':
        raise ValueError('task must be train or plan')
    torch.set_num_threads(cfg.threads)
    pl.seed_everything(cfg.seed, workers=True)
    return train_args(args, config, cfg)




# Evaluation and generation entry points share the same Hydra configuration.
import csv
import os
import subprocess
import hashlib
import importlib.metadata
from unifiedgroupgen.common.configuration import load_experiments, runtime_pythonpath, source_key
from unifiedgroupgen.common.structures import ReferenceBank, counts_of, inline_cif_bytes, load_structure, read_records, sha256, write_json
from unifiedgroupgen.evaluation.common import EvaluationContext, evaluate_records, build_summary, geometry, operation_check, descriptor_metadata
from unifiedgroupgen.evaluation.utils import numeric, logical_and, composition_validity, fingerprints
from unifiedgroupgen.evaluation.conditional import evaluate_properties
from unifiedgroupgen.common.reports import write_tables, figures, campaign_summary
from unifiedgroupgen.common.source import package_sources
from unifiedgroupgen.pl_modules.factory import load_model_checkpoint
ROOT=project_root()
TASKS = ('components', 'dng', 'csp', 'fixed_descriptor', 'group')
def reference_metrics(bank, kind, use_fingerprints, cache):
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pymatgen', 'matminer', 'smact', 'pyxtal', 'spglib')}
    identity = {'files': bank.identity(), 'kind': kind, 'fingerprints': use_fingerprints, 'versions': versions,
        'code': {Path(p).name: sha256(p) for p in (
            Path(__file__), Path(__file__).parent/'evaluation/common.py', Path(__file__).parent/'evaluation/utils.py', Path(__file__).parent/'common/structures.py',
            Path(__file__).parent/'group_constraints/compiler.py',
            Path(__file__).parent/'tests/representation.py')}}
    key = __import__('hashlib').sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache)/(key+'.json')
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['identity'] != identity:
            raise ValueError('Reference cache identity mismatch')
        return saved
    rows, features = [], []
    for index, (record, base) in enumerate(bank.rows):
        row = {'id': str(record.get('id', index)), **descriptor_metadata(record, kind)}
        row['recovered_group'] = row.get('declared_group')  # Audited reference label, explicitly identified in protocol.
        try:
            structure = load_structure(record, base)
            row.update(geometry(structure, kind))
            row['elements'] = sorted(counts_of(structure))
            row['chemical_valid'], row['chemical_error'] = composition_validity(structure)
            row['basic_valid'] = logical_and(row['periodic_distance_valid'], row['chemical_valid'])
            if use_fingerprints:
                pair = fingerprints(structure, kind)
                features.append([p.tolist() for p in pair])
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, ZeroDivisionError, np.linalg.LinAlgError) as error:
            row['evaluation_error'] = str(error)
        rows.append(row)
        if (index+1)%100==0:
            print(json.dumps({'phase': 'reference', 'completed': index+1, 'requested': len(bank.rows)}), flush=True)
    saved = {'identity': identity, 'rows': rows, 'fingerprints': features}
    write_json(path, saved)
    return saved


def evaluate(samples, task, kind, output, reference_files=(), novelty_files=(), dataset=None, request=None,
             assessments=None, assessment_metadata=None, use_fingerprints=True, reidentify=True,
             cache=None, expected_attempts=None, embeddings=None, reference_assessments=None, *, conditional_enabled=False):
    if task not in ('csp', 'dng') or kind not in ('space', 'layer'):
        raise ValueError('task=csp/dng and kind=space/layer are required')
    samples, output = Path(samples).resolve(), Path(output).resolve()
    request = request or {}
    records = read_records(samples)
    if not records or len({r['index'] for r in records}) != len(records):
        raise ValueError('A nonempty ledger with unique attempt indices is required')
    if expected_attempts is not None and len(records)!=expected_attempts:
        raise ValueError('Attempt count differs from the requested budget')
    refs, train = ReferenceBank(reference_files, dataset), ReferenceBank(novelty_files, dataset)
    predictions, metadata = assessments or {}, assessment_metadata or {}
    if predictions and (not metadata.get('evaluator') or metadata.get('samples_sha256')!=sha256(samples)):
        raise ValueError('Independent assessments require evaluator identity and matching samples_sha256')
    if reference_assessments and metadata.get('reference_sha256')!=refs.identity():
        raise ValueError('Reference predictions do not bind the exact reference files')
    reference = reference_metrics(refs, kind, use_fingerprints, cache or output/'reference_cache') if refs.rows else {'rows': [], 'fingerprints': []}
    if task=='csp' and any(str(r.get('case_id', r.get('reference_id'))) not in refs.by_id for r in records):
        raise ValueError('Each CSP attempt must name an existing reference case, including failures')
    context = EvaluationContext(task, kind, dataset, request, refs, train, use_fingerprints, reidentify, novelty_files, samples)
    tolerances = request.get('property_tolerances', {'formation_energy': .1, 'band_gap': .2, 'energy_above_hull': .05})
    started = time.perf_counter()
    rows, features = evaluate_records(records, context)
    properties = evaluate_properties(rows, predictions, metadata, tolerances, reference_assessments, enabled=conditional_enabled)
    embedding_data = json.loads(Path(embeddings).read_text(encoding='utf-8')) if embeddings else None
    report = build_summary(rows, features, reference, properties, context, embedding_data)
    elapsed = time.perf_counter()-started
    generation_times = [r['generation_seconds'] for r in rows if r.get('generation_seconds') is not None]
    report['efficiency'] = {'metric_seconds_excluding_reference_cache':elapsed,'generation_seconds':numeric(generation_times),
        'generation_time_coverage':len(generation_times),'generated_structures_per_second':report['generation']['generated']/sum(generation_times)
            if len(generation_times)==len(rows) and sum(generation_times)>0 else None}
    write_tables(output, rows, report, kind)
    return rows,report,reference['rows']



def evaluate_task(args):
    if not args.conditional_enabled and (args.assessments or args.assessment_metadata or args.reference_assessments):
        raise ValueError('Enable evaluation.conditional.enabled for property predictions')
    if bool(args.assessments)!=bool(args.assessment_metadata):
        raise ValueError('Assessments CSV and evaluator metadata must be supplied together')
    inputs = [args.samples, *args.reference_files, *args.novelty_files,
              *[p for p in (args.request,args.assessments,args.assessment_metadata,args.reference_assessments,args.embeddings) if isinstance(p,Path)]]
    records = read_records(args.samples)
    for r in records:
        if 'cif' in r:
            continue
        if r.get('cif_path'):
            p=Path(r['cif_path']); inputs.append(p if p.is_absolute() else args.samples.parent/p)
        elif r.get('status')=='generated' and (args.samples.parent/f"sample_{r['index']:05d}.cif").exists():
            inputs.append(args.samples.parent/f"sample_{r['index']:05d}.cif")
    from unifiedgroupgen.common.source import package_sources
    code = package_sources()
    identity = {'inputs':{str(p.resolve()):sha256(p) for p in inputs},
                'code':{source_key(p,ROOT):sha256(p) for p in code},
                'options':{k:v for k,v in vars(args).items() if k not in ('resume','output','cache') and not isinstance(v,Path) and k not in ('reference_files','novelty_files')}}
    manifest = args.output/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text(encoding='utf-8'))
        if not args.resume or saved['identity']!=identity:
            raise ValueError('Existing output has another identity, or --resume was not supplied; use a new directory')
        if all((args.output/p).exists() and sha256(args.output/p)==h for p,h in saved['outputs'].items()):
            print('Verified existing evaluation; no work repeated.'); return
        raise ValueError('Completed output was modified or removed; preserve it and use a new output directory')
    def prediction_csv(path,index_key):
        predictions={}
        if path is None: return predictions
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                index=row.pop(index_key)
                if index in predictions:
                    raise ValueError('Duplicate prediction index')
                predictions[index]={}
                for name,value in row.items():
                    if name=='prediction_error':
                        predictions[index][name]=value
                        continue
                    try: predictions[index][name]=float(value) if value else None
                    except ValueError: pass
        return predictions
    predictions=prediction_csv(args.assessments,'index')
    reference_predictions=prediction_csv(args.reference_assessments,'id')
    if set(predictions)-{str(r['index']) for r in records}:
        raise ValueError('Predictions contain indices outside the attempt ledger')
    metadata=json.loads(args.assessment_metadata.read_text(encoding='utf-8')) if args.assessment_metadata else {}
    for name, expected in metadata.get('structure_files_sha256', {}).items():
        actual = sha256(Path(name)) if Path(name).is_file() else None
        if actual != expected: raise ValueError('Property predictor input CIF changed: '+name)
    if args.assessments and metadata.get('predictions_sha256') and metadata['predictions_sha256']!=sha256(args.assessments):
        raise ValueError('Prediction CSV hash mismatch')
    if args.reference_assessments and metadata.get('reference_predictions_sha256')!=sha256(args.reference_assessments):
        raise ValueError('Reference prediction CSV hash mismatch')
    request=args.request if isinstance(args.request,dict) else json.loads(args.request.read_text(encoding='utf-8')) if args.request else {}
    rows,report,refs=evaluate(args.samples,args.task,args.kind,args.output,args.reference_files,args.novelty_files,
        args.dataset,request,predictions,metadata,not args.no_fingerprints,not args.no_reidentify,args.cache,args.expected_attempts,args.embeddings,reference_predictions,conditional_enabled=args.conditional_enabled)
    for row in refs:
        row.update({'predicted_'+p:v for p,v in reference_predictions.get(row['id'],{}).items()})
    figures(rows,report,refs,args.output)
    outputs=[p for p in args.output.iterdir() if p.suffix in ('.csv','.json','.png') and p.name!='manifest.json']
    write_json(manifest,{'identity':identity,'outputs':{p.name:sha256(p) for p in outputs},'state':'complete'})
    print(json.dumps({'state':'complete','requested':len(rows),'generated':report['generation']['generated'],'output':str(args.output)},ensure_ascii=False))


def build_jobs(config, count, cases, k, conditional_count, seed, group_count=0,
               tasks=None, conditional_tasks=None, target_properties=None, root=None):
    root = project_root() if root is None else Path(root)
    jobs=[{'name':'components','task':'components'}]
    tests=[r for p in config['test_files'] for r in read_records(resolve_path(root/p,root))]
    jobs[0]['test_records']=len(tests)
    train=[r for p in config['train_files'] for r in read_records(resolve_path(root/p,root))]
    domains=config.get('dataset_domains') or {config['dataset']:None}
    properties=[p for p,spec in config.get('properties',{}).items() if not spec.get('context')]
    if target_properties is not None:
        properties=[p for p in properties if p in target_properties]
    for dataset,domain in domains.items():
        kinds={r['descriptor']['kind'] for r in tests if r['dataset']==dataset and r.get('descriptor')}
        if len(kinds)>1:
            raise ValueError(f'A dataset must have one declared structure kind: {dataset}')
        kind=config.get('dataset_kinds', {}).get(dataset) or next(iter(kinds), config.get('kind'))
        if kind not in ('space', 'layer'):
            raise ValueError(f'Missing space/layer metadata for dataset: {dataset}')
        context={'source_domain':domain} if domain is not None else {}
        selected=sorted((r for r in tests if r['dataset']==dataset),key=lambda r:hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
        if cases: selected=selected[:cases]
        base={'dataset':dataset,'kind':kind,'properties':context,'count':count,'case_ids':[str(r['id']) for r in selected],'k':k}
        def add(suffix,task,targets=None,**extra):
            job={**base,'name':dataset+'_'+suffix,'task':task,'properties':{**context,**(targets or {})},**extra}
            if task!='csp':
                job.pop('case_ids');job.pop('k')
            jobs.append(job)
        add('dng','dng');add('csp','csp');add('fixed_descriptor','csp',fixed_descriptor=True)
        if group_count:
            groups=list(range(1,81 if kind=='layer' else 231))
            add('group_sweep','dng',groups=groups,per_group=group_count,count=len(groups)*group_count)
        domain_train=[r for r in train if r['dataset']==dataset]
        for p in properties:
            values=[r['properties'][p] for r in domain_train if r['properties'].get(p) is not None and np.isfinite(r['properties'][p])]
            for i,value in enumerate(sorted(set(map(float,np.quantile(values,[.1,.5,.9])))) if values else []):
                target={p:value}
                add(f'dng_{p}_q{i}','dng',target,count=conditional_count,conditional=True)
                add(f'csp_{p}_q{i}','csp',target,conditional=True)
        if len(properties)>1:
            complete=sorted([r for r in domain_train if all(r['properties'].get(p) is not None and np.isfinite(r['properties'][p]) for p in properties)],
                            key=lambda r:(r['properties'][properties[0]],str(r['id'])))
            for i,q in enumerate((.1,.5,.9)) if complete else []:
                r=complete[round(q*(len(complete)-1))];target={p:r['properties'][p] for p in properties}
                add(f'dng_joint_q{i}','dng',target,count=conditional_count,conditional=True)
                add(f'csp_joint_q{i}','csp',target,conditional=True)
    selected=set(tasks or TASKS)
    conditional_tasks=set(conditional_tasks if conditional_tasks is not None else ('dng','csp'))
    def selected_job(job):
        category=('components' if job['task']=='components' else 'group' if job.get('group') or job.get('groups') else
                  'fixed_descriptor' if job.get('fixed_descriptor') else job['task'])
        return category in selected and (not job.get('conditional') or job['task'] in conditional_tasks)
    return [j for j in jobs if selected_job(j)]


def build_plan(args, root):
    catalog = load_experiments(getattr(args, 'experiments', root/'configs/experiments.yaml'), root)
    plan=[]
    for name in args.models:
        entry = catalog[name]
        if args.checkpoint_root is not None:
            directory=args.checkpoint_root/name
            checkpoint=directory/'best.ckpt' if (directory/'best.ckpt').exists() else directory/'model.best.pt'
        elif entry.get('checkpoint'):
            checkpoint=resolve_path(entry['checkpoint'], root)
        else:
            runs=RunSettings.load(root).jobs/name
            candidates=sorted([*runs.glob('*/model.best.pt'), *runs.glob('*/best.ckpt')], key=lambda p:(p.parent.name,p.name=='best.ckpt'))
            checkpoint=candidates[-1] if candidates else RunSettings.load(root).jobs/name/'model.best.pt'
        if checkpoint.exists():
            saved=load_model_checkpoint(checkpoint,map_location='cpu',weights_only=False)
            config=saved['config']
            if config.get('experiment_id')!=name: raise ValueError('Checkpoint/model identity mismatch')
            checkpoint_hash=sha256(checkpoint)
            if args.run and not args.plan:
                from unifiedgroupgen.tests.components import checked_config
                checked_config(saved)
        else:
            config=entry['config']
            checkpoint_hash=None
        data_hashes={p:sha256(resolve_path(root/p,root)) for split in ('train','val','test') for p in config[split+'_files']}
        for job in build_jobs(config,args.count,args.csp_cases,args.csp_k,args.conditional_count,args.seed,args.per_group,
                              args.tasks,args.conditional_tasks,args.target_properties, root=root):
            plan.append({'model':name,'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_hash,'data_sha256':data_hashes,**job})
    total=sum((len(j['case_ids'])*j['k'] if j['task']=='csp' else j.get('count',0)) for j in plan if j['task']!='components')
    options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('action','run','resume','plan','status','stage','verify_generated')}
    identity={'options':options,'jobs':plan,'code':{source_key(p,root):sha256(p) for p in
        package_sources()}}
    return plan, total, identity

def run_command(command,log,env):
    with log.open('a',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u',*map(str,command)],cwd=project_root(),env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def verify_files(root,outputs):
    for name,expected in outputs.items():
        path=resolve_path(root/name,root)
        if not path.is_file() or sha256(path)!=expected:
            raise ValueError(f'Saved output missing or modified: {path}')


def output_hashes(root):
    return {source_key(p,root):sha256(p) for p in root.rglob('*')
            if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.png','.cif')}

def show_status(args):
    state=json.loads((args.output/'pipeline.json').read_text(encoding='utf-8'))
    result={'state':state['state'],'pid':state['pid'],'current':state.get('current'),
        'stage':state.get('stage'),'phase':state.get('phase'),
        'complete':sum(r['state']=='complete' for r in state['jobs']),'total':len(state['identity']['jobs']),
        'generated_jobs':sum(r.get('generation',{}).get('state')=='complete' for r in state['jobs']),
        'generation_jobs_total':sum(j['task']!='components' for j in state['identity']['jobs']),
        'last_completed':[{'name':r['name'],'seconds':r.get('seconds')} for r in state['jobs'] if r['state']=='complete'][-5:],
        'error':state.get('error')}
    status=args.output/state.get('current','')/'sampling.status.json'
    if status.exists(): result['sampling']=json.loads(status.read_text(encoding='utf-8'))
    print(json.dumps(result,ensure_ascii=False,indent=2));return


def verify_generated(args, plan, identity):
    saved=json.loads((args.output/'pipeline.json').read_text(encoding='utf-8'))
    if saved['identity']!=identity: raise ValueError('Generation verification requires identical model/data/options/code')
    generated=failed=cifs=inline_cifs=0
    for job in plan:
        if job['task']=='components': continue
        key=job['model']+'/'+job['name'];target=args.output/key
        entry=next((r for r in saved['jobs'] if r['name']==key),{})
        receipt=entry.get('generation',{})
        if receipt.get('state')!='complete': raise ValueError(f'Generation incomplete: {key}')
        verify_files(target,receipt['outputs'])
        records=read_records(target/'samples.jsonl')
        generated+=sum(r['status']=='generated' for r in records)
        failed+=sum(r['status']!='generated' for r in records)
        cifs+=sum(p.endswith('.cif') for p in receipt['outputs'])
        for record in records:
            if 'cif' in record:
                inline_cif_bytes(record);inline_cifs+=1
    print(json.dumps({'state':'verified','requested':generated+failed,'generated':generated,'failed':failed,
        'cif_files_checked':cifs,'cif_strings_checked':inline_cifs,'regeneration_started':False},indent=2));return


def write_plan(args, plan, total, identity):
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'plan.json',{'identity':identity,'generation_attempts':total,'formal_tasks_started':False})
    with (args.output/'plan.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=['model','job','task','dataset','kind','test_records','candidates_per_case','requested_structures','guidance','properties']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for j in plan:
            writer.writerow({'model':j['model'],'job':j['name'],'task':j['task'],'dataset':j.get('dataset'),
                'kind':j.get('kind'),'test_records':len(j.get('case_ids',[])) if j['task']=='csp' else j.get('test_records'),
                'candidates_per_case':j.get('k') if j['task']=='csp' else None,
                'requested_structures':len(j['case_ids'])*j['k'] if j['task']=='csp' else j.get('count',0),
                'guidance':args.guidance if j.get('conditional') else 0.,'properties':json.dumps(j.get('properties',{}),sort_keys=True)})
    print(json.dumps({'models':len(args.models),'jobs':len(plan),'generation_attempts':total,'plan':str(args.output/'plan.json'),
                      'checkpoints_missing':sorted({j['model'] for j in plan if j['checkpoint_sha256'] is None}),
                      'formal_tasks_started':False},ensure_ascii=False,indent=2));return



def evaluate_suite(args):
    if args.status:
        show_status(args); return
    if min(args.count,args.csp_k,args.conditional_count,args.steps,args.threads,args.test_batch_size)<1 or min(args.per_group,args.csp_cases)<0:
        raise ValueError('Budgets must be positive; per-group and csp-cases may be zero')
    if bool(args.cgdit_root)!=bool(args.predictor_root): raise ValueError('Provide both predictor paths')
    plan, total, identity = build_plan(args, ROOT)
    if args.verify_generated:
        verify_generated(args, plan, identity); return
    if not args.run or args.plan:
        write_plan(args, plan, total, identity); return
    if any(j['checkpoint_sha256'] is None for j in plan): raise ValueError('Missing checkpoints; only a plan was possible')
    if args.device.startswith('cuda'):
        # Refuse competing GPU work; never stop or alter another task.
        check=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        if check.stdout.strip(): raise RuntimeError('GPU has active compute processes; evaluate later or use --device cpu')
    args.output.mkdir(parents=True,exist_ok=True)
    pipeline=args.output/'pipeline.json'
    state={'state':'running','pid':os.getpid(),'identity':identity,'jobs':[],'generation_attempts':total,'stage':args.stage}
    if pipeline.exists():
        old=json.loads(pipeline.read_text(encoding='utf-8'))
        if not args.resume or old['identity']!=identity: raise ValueError('Use --resume with identical inputs/options or a new output directory')
        state['jobs']=old['jobs']
    if args.stage=='metrics':
        for job in plan:
            if job['task']=='components': continue
            key=job['model']+'/'+job['name']
            entry=next((r for r in state['jobs'] if r['name']==key),{})
            if entry.get('generation',{}).get('state')!='complete':
                raise ValueError(f'Generate all requested structures first: {key}; metrics stage never samples')
            verify_files(args.output/key,entry['generation']['outputs'])
    lock=args.output/'evaluation.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    env={**os.environ,'PYTHONPATH':runtime_pythonpath(),'MKL_THREADING_LAYER':'SEQUENTIAL',
         'OMP_NUM_THREADS':str(args.threads),'MKL_NUM_THREADS':str(args.threads),'PYTHONIOENCODING':'utf-8'}
    try:
        for job in plan:
            key=job['model']+'/'+job['name'];target=args.output/key
            if args.stage=='generate' and job['task']=='components': continue
            old=next((r for r in state['jobs'] if r['name']==key),None)
            if old and old['state']=='complete':
                verify_files(target,old['outputs'])
                continue
            target.mkdir(parents=True,exist_ok=True)
            entry=old or {'name':key}
            if old is None: state['jobs'].append(entry)
            entry.update(state='running',started=time.time());state.update(current=key,updated=time.time());write_json(pipeline,state)
            write_json(target/'job.json',job)
            print('[START] '+key,flush=True)
            run_job(job, args, target, entry, state, pipeline, env)
        if args.stage=='generate':
            state['state']='complete' if len(state['jobs'])==len(plan) and all(j['state']=='complete' for j in state['jobs']) else 'generation_complete'
        else:
            state['state']='complete'
            from unifiedgroupgen.common.reports import campaign_summary
            campaign_summary(args.output,state)
    except BaseException as error:
        state.update(state='failed',error=str(error));raise
    finally:
        state['updated']=time.time();write_json(pipeline,state);lock.unlink()




def analysis_options(cfg, section):
    """Resolve YAML paths once at the public entry, including relocated logical paths."""
    values = OmegaConf.to_container(cfg[section], resolve=True)
    paths = ('samples', 'output', 'cache', 'request', 'assessments', 'assessment_metadata',
             'reference_assessments', 'embeddings', 'experiments', 'checkpoint_root',
             'cgdit_root', 'predictor_root', 'reference_cache', 'model_cache', 'predictor_config', 'checkpoint')
    for key in paths:
        if values.get(key) is not None and not isinstance(values[key],dict):
            values[key] = resolve_path(values[key], ROOT)
    for key in ('reference_files', 'novelty_files'):
        if key in values:
            values[key] = [resolve_path(p, ROOT) for p in values[key]]
    values.update(device=cfg.device if cfg.device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'),
                  threads=cfg.threads, seed=cfg.seed)
    return SimpleNamespace(**values)


def inference_task(args):
    """Load a checkpoint once for a component check or a resumable evaluation sample job."""
    from unifiedgroupgen.tests.components import checked_config, components
    from unifiedgroupgen.tests.legacy.generation_before_entry.tasks import generate
    if args.batch_size < 1 or args.steps < 1:
        raise ValueError('Batch size and flow steps must be positive')
    saved = load_model_checkpoint(args.checkpoint, map_location='cpu', weights_only=False)
    config = checked_config(saved)
    job = args.job
    if job.get('checkpoint_sha256') and job['checkpoint_sha256'] != sha256(args.checkpoint):
        raise ValueError('Checkpoint changed after the evaluation plan was created')
    identity = {'checkpoint_sha256':sha256(args.checkpoint), 'job':job, 'seed':args.seed,
                'steps':args.steps, 'guidance':args.guidance,
                'test':{p:sha256(resolve_path(p,ROOT)) for p in config['test_files']},
                'structure_storage':'inline_cif_utf8_v1',
                'code':{source_key(p,ROOT):sha256(p) for p in package_sources()}}
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output/'identity.json'
    if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8')) != identity:
        raise ValueError('Inference identity changed; preserve results and choose a new output directory')
    write_json(receipt, identity)
    proposal, flow = build_models(config, args.device)
    proposal.load_state_dict(saved['proposal'], strict=True)
    flow.load_state_dict(saved['flow'], strict=True)
    proposal.eval(); flow.eval()
    if job['task'] == 'components':
        return components(proposal, flow, saved, args.output, args.device, args.batch_size, args.seed)
    return generate(proposal, flow, config, job, args.output, args.device, args.seed, args.steps, args.guidance)


def run_worker(options, task, values, target, log, env):
    """Isolate each task's native scientific runtime; all workers enter through run.py."""
    config_path = target/(log.stem+'.yaml')
    write = {'defaults':['_self_',{'override hydra/job_logging':'disabled'}],
             'hydra':{'job':{'chdir':False},'run':{'dir':str(target)},'output_subdir':None},
             'task':task, 'device':options.device, 'seed':options.seed, 'threads':options.threads, **values}
    OmegaConf.save(OmegaConf.create(write), config_path)
    run_command(['-m','unifiedgroupgen.run','--config-path',target,'--config-name',config_path.stem], log, env)


def evaluate_metrics(job, args, target, state, pipeline, env):
    config = load_model_checkpoint(job['checkpoint'], map_location='cpu', weights_only=False)['config']
    property_args = {}
    if args.conditional_enabled and args.predictor_root and (job['dataset']=='mp_20' or args.allow_proxy_transfer):
        state['phase']='properties'; write_json(pipeline,state)
        spec = OmegaConf.to_container(OmegaConf.load(ROOT/'configs/property_predictor/default.yaml'), resolve=True)
        spec.update(backend='cgdit', samples=str(target/'samples.jsonl'), dataset=job['dataset'], kind=job['kind'],
                    cgdit_root=str(args.cgdit_root), predictor_root=str(args.predictor_root),
                    predictor_seed=args.predictor_seed, output=str(target/'proxy'),
                    reference_files=config['test_files'], reference_cache=str(args.output/'proxy_reference_cache'),
                    allow_proxy_transfer=args.allow_proxy_transfer)
        if not (target/'proxy/evaluator.json').exists():
            run_worker(args,'predict_properties',{'property_predictor':spec},target,target/'properties.log',env)
        property_args = {'assessments':str(target/'proxy/predictions.csv'),
                         'assessment_metadata':str(target/'proxy/evaluator.json'),
                         'reference_assessments':str(target/'proxy/reference_predictions.csv')}
    request = {'properties':{p:v for p,v in job['properties'].items() if p!='source_domain'}, 'group':job.get('group')}
    if job.get('groups'): request['sampling_task']='balanced_group_sweep'
    if job['task']=='csp': request.update(case_ids=job['case_ids'], attempts_per_case=job['k'])
    write_json(target/'request.json',request)
    spec = OmegaConf.to_container(OmegaConf.load(ROOT/'configs/evaluation/default.yaml'),resolve=True)
    spec.update(samples=str(target/'samples.jsonl'), task=job['task'], kind=job['kind'], dataset=job['dataset'],
                reference_files=config['test_files'], novelty_files=config['train_files'],
                request=str(target/'request.json'), output=str(target/'metrics'), cache=str(args.output/'reference_cache'),
                expected_attempts=len(job['case_ids'])*job['k'] if job['task']=='csp' else job['count'],
                resume=True, no_fingerprints=args.no_fingerprints, **property_args)
    spec['conditional']['enabled']=args.conditional_enabled
    state['phase']='metrics'; write_json(pipeline,state)
    run_worker(args,'evaluate',{'evaluation':spec},target,target/'metrics.log',env)


def run_job(job, args, target, entry, state, pipeline, env):
    spec = {'checkpoint':job['checkpoint'], 'job':job, 'output':str(target),
            'steps':args.steps, 'guidance':args.guidance if job.get('conditional') else 0.,
            'batch_size':args.test_batch_size}
    if job['task']=='components':
        state['phase']='components'; write_json(pipeline,state)
        run_worker(args,'check_components',{'component_check':spec},target,target/'components.log',env)
    elif args.stage in ('all','generate'):
        receipt=entry.get('generation',{})
        if receipt.get('state')=='complete':
            verify_files(target,receipt['outputs'])
        else:
            state['phase']='generate'; write_json(pipeline,state)
            started=time.perf_counter()
            run_worker(args,'generate',{'generation':spec},target,target/'inference.log',env)
            entry['generation']={'state':'complete','seconds':time.perf_counter()-started,'outputs':output_hashes(target)}
        entry['state']='generated'; state['updated']=time.time(); write_json(pipeline,state)
    if args.stage=='generate':
        print('[GENERATED] '+entry['name'],flush=True); return
    started=time.perf_counter()
    if job['task']!='components': evaluate_metrics(job,args,target,state,pipeline,env)
    seconds=time.time()-entry['started'] if job['task']=='components' else time.perf_counter()-started
    entry.update(state='complete',evaluation_seconds=seconds,
                 seconds=entry.get('generation',{}).get('seconds',0)+seconds, outputs=output_hashes(target))
    state['updated']=time.time(); write_json(pipeline,state)
    print('[DONE] '+entry['name'],flush=True)


def execute_analysis(cfg):
    """Single public dispatch for generation, metric calculation and component checks."""
    torch.set_num_threads(cfg.threads)
    pl.seed_everything(cfg.seed,workers=True)
    if cfg.task == 'evaluate':
        args = analysis_options(cfg,'evaluation')
        args.conditional_enabled = args.conditional['enabled']
        del args.conditional
        if args.samples is None or args.output is None:
            raise ValueError('evaluation.samples and evaluation.output are required')
        if args.task not in ('csp','dng') or args.kind not in ('space','layer'):
            raise ValueError('Choose evaluation.task=csp/dng and evaluation.kind=space/layer')
        return evaluate_task(args)
    if cfg.task == 'evaluate_suite':
        args = analysis_options(cfg,'evaluation_suite')
        args.run=args.action=='run'; args.plan=args.action=='plan'
        args.status=args.action=='status'; args.verify_generated=args.action=='verify'
        args.models=args.models or list(load_experiments(args.experiments,ROOT))
        if set(args.models)-set(load_experiments(args.experiments,ROOT)):
            raise ValueError('Unknown experiment in evaluation_suite.models')
        return evaluate_suite(args)
    if cfg.task == 'check_components':
        args=analysis_options(cfg,'component_check')
        args.job={'task':'components'}
        return inference_task(args)
    if cfg.task == 'generate':
        args=analysis_options(cfg,'generation')
        if getattr(args,'job',None) is not None:
            return inference_task(args)
        from unifiedgroupgen.tests.legacy.generation_before_entry.pipeline import sample
        args.properties=json.dumps(args.properties)
        if isinstance(args.composition,dict): args.composition=json.dumps(args.composition)
        return sample(args)
    if cfg.task == 'predict_properties':
        from unifiedgroupgen.common.predictors.service import predict
        return predict(analysis_options(cfg,'property_predictor'))
    if cfg.task == 'export_cif':
        from unifiedgroupgen.common.structures import export_cif
        args=analysis_options(cfg,'export')
        print(json.dumps(export_cif(args.samples,args.index,args.output)))
        return
    raise ValueError('task must be train, plan, generate, check_components, evaluate, evaluate_suite, predict_properties or export_cif')


@hydra.main(version_base='1.3', config_path=str(project_root()/'configs'), config_name='run')
def main(cfg: DictConfig):
    execute(cfg)


if __name__ == '__main__':
    main()
