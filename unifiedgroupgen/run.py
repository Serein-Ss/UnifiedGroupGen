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
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.profilers import SimpleProfiler
from pytorch_lightning.utilities import grad_norm
from lightning_fabric.utilities.seed import _collect_rng_states as rng_state, _set_rng_states as restore_rng
from unifiedgroupgen.common.configuration import project_root, resolve_path, RunSettings, training_definition
from unifiedgroupgen.common.source import file_hash
from unifiedgroupgen.pl_modules.factory import build_models
from unifiedgroupgen.pl_modules.two_stage.module import save_status, UnifiedGroupGenModule, TrainingArtifacts
from unifiedgroupgen.common.progress import interactive
from unifiedgroupgen.pl_modules.progress import progress_callbacks
from unifiedgroupgen.pl_modules.objectives.joint import joint_loss
from unifiedgroupgen.conditional.properties import fit_property_scalers
from unifiedgroupgen.pl_data.dataset import load_training_splits
from unifiedgroupgen.pl_data.datamodule import CrystalDataModule
from unifiedgroupgen.tests.gradients import gradient_diagnostics








def training_output(settings, experiment, output=None, resume=None):
    """Explicit historical outputs stay explicit; new default runs get a real start date."""
    if not experiment or Path(experiment).name != experiment or experiment in ('.', '..') or ':' in experiment or '\\' in experiment:
        raise ValueError('Experiment must be a single directory name')
    if output:
        output = Path(output).expanduser()
        output = (output if output.is_absolute() else settings.project_root/output).resolve()
    elif resume:
        output = Path(resume).resolve().parent/'model.pt'
    else:
        started = datetime.now(timezone(timedelta(hours=8)))
        directory = settings.jobs/experiment/started.strftime('%y-%m-%d-%H-%M-%S')
        directory.mkdir(parents=True, exist_ok=False)
        output = directory/'model.pt'
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.name != 'model.pt':
        raise ValueError('Training output must be named model.pt')
    if not resume and any((output.parent/name).exists() for name in
                          ('model.pt', 'model.best.pt', 'model.history.jsonl', 'model.status.json', 'model.steps.jsonl', 'train.log')):
        raise FileExistsError('Training results already exist; resume or choose a new run directory')
    return output


def build_loggers(args, output, settings, resume):
    loggers = []
    if settings.wandb_mode != 'disabled':
        settings.wandb_dir.mkdir(parents=True, exist_ok=True)
        saved = torch.load(resume, map_location='cpu', weights_only=False) if resume else {}
        run_id = saved.get('ugg_state', {}).get('wandb_id') or uuid.uuid4().hex
        loggers.append(WandbLogger(project=getattr(args, 'wandb_project', None) or 'UnifiedGroupGen',
                                  name=getattr(args, 'experiment', None) or output.parent.parent.name,
                                  id=run_id, save_dir=str(settings.wandb_dir),
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


def build_callbacks(config, runtime, directory, logging=True):
    best = ModelCheckpoint(dirpath=directory, filename='model.best', monitor='val/loss', mode='min', save_top_k=1,
                        save_on_train_epoch_end=False, auto_insert_metric_name=False, enable_version_counter=False)
    last = ModelCheckpoint(dirpath=directory, save_top_k=0, save_last=True, every_n_epochs=1,
                           save_on_train_epoch_end=True, enable_version_counter=False)
    best.FILE_EXTENSION = last.FILE_EXTENSION = '.pt'
    last.CHECKPOINT_NAME_LAST = 'model'
    callbacks = [TrainingArtifacts(), best, last]
    if logging:
        callbacks.append(LearningRateMonitor(logging_interval='epoch'))
    patience = config.get('early_stopping_patience', 30)
    if patience:
        callbacks.append(EarlyStopping(monitor='val/loss', mode='min',
            patience=math.ceil(patience/runtime['validation_interval']), check_on_train_epoch_end=False))
    return callbacks


def run_lightning(args, config, records, validation, proposal, flow, train_paths, val_paths, output, settings,
                  finish_logging=True):
    output = Path(output)
    if not args.resume and any(path.exists() for path in (output, output.with_suffix('.steps.jsonl'), output.parent/'model.best.pt', output.parent/'last.ckpt', output.parent/'best.ckpt',
                                                         output.parent/'logs/metrics.csv')):
        raise FileExistsError('Checkpoint already exists: '+str(output))
    output.parent.mkdir(parents=True, exist_ok=True)
    hashes = {'train_split_sha256': [file_hash(p) for p in train_paths],
              'val_split_sha256': [file_hash(p) for p in val_paths]}
    resume, previous = resume_checkpoint(args.resume)
    runtime = runtime_options(args, config, flow, previous, hashes)
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
    if loggers:
        loggers[0].log_hyperparams({'training': config, 'runtime': runtime, 'split_hashes': hashes,
                                  'seed': getattr(args, 'seed', 42),
                                  'launch': getattr(args, 'resolved_config', None),
                                  'model_directory': str(output.parent)})
    initial_rng = rng_state()
    module.pending_rng = initial_rng
    trainer = pl.Trainer(accelerator='gpu' if str(args.device).startswith('cuda') else 'cpu',
                         devices=[torch.device(args.device).index or 0] if str(args.device).startswith('cuda') else 1,
                         max_epochs=epochs, precision={'fp32': '32-true', 'bf16': 'bf16-mixed', 'fp16': '16-mixed'}[mode],
                         logger=loggers or False, callbacks=build_callbacks(config, runtime, output.parent, bool(loggers))+progress_callbacks(output), default_root_dir=str(output.parent),
                         profiler=SimpleProfiler(), enable_model_summary=False, num_sanity_val_steps=0,
                         check_val_every_n_epoch=runtime['validation_interval'], reload_dataloaders_every_n_epochs=1,
                         use_distributed_sampler=False, log_every_n_steps=25, enable_progress_bar=interactive(),
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
    if config.get('prior'):
        from unifiedgroupgen.generation.prior import fit_prior
        config['prior'] = fit_prior(records, config['prior'])
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
    if config.get('model', {}).get('family') in ('joint_symmetry_v1', 'joint_flow_v2'):
        from unifiedgroupgen.pl_modules.joint_training import run_joint_training
        return run_joint_training(args, config, records, validation, paths, resolved_config)
    proposal, flow = build_models(config, 'cpu')
    if hasattr(flow, 'backbone_identity'):
        config['backbone_identity'] = flow.backbone_identity
    settings = RunSettings.load(root, getattr(args, 'wandb_mode', None))
    output = training_output(settings, args.experiment or config.get('experiment_id', 'training'), args.output, args.resume)
    args.resolved_config = OmegaConf.to_container(resolved_config, resolve=True) if resolved_config is not None else None
    print(f'Training | output={output} | logging={settings.wandb_mode}', flush=True)
    return run_lightning(args, config, records, validation, proposal, flow,
                         paths['train'], paths['val'], output, settings)


def execute(cfg: DictConfig):
    logging = cfg.logging if 'logging' in cfg else cfg.wandb
    settings = RunSettings.load(project_root(), logging.mode)
    if cfg.resume and cfg.initialize_from:
        raise ValueError('resume restores a run; initialize_from starts a new run. Choose one.')
    config = training_definition(cfg)
    profile = config.pop('runtime_profile', {})
    from hydra.core.hydra_config import HydraConfig
    selected = HydraConfig.get().runtime.choices.get('experiment', 'training') if HydraConfig.initialized() else 'training'
    name = cfg.experiment_name or config.get('experiment_id') or selected
    runtime = OmegaConf.to_container(cfg.runtime, resolve=True)
    if cfg.resume and 'data' in cfg:
        # Native checkpoints own execution settings unless the launch explicitly overrides them.
        overrides = HydraConfig.get().overrides.task if HydraConfig.initialized() else []
        explicit = {item.split('=', 1)[0].lstrip('+') for item in overrides}
        runtime = {key: value if ('runtime' in explicit or 'runtime.'+key in explicit
                   or (key == 'coordinate_length_scale' and 'loss.coordinate_length_scale' in explicit)
                   or (key == 'validation_interval' and 'train.validation_interval' in explicit)) else None
                   for key, value in runtime.items()}
    if not cfg.resume:
        defaults = {**{key: config.get(key) for key in runtime}, **profile}
        runtime = {key: value if value is not None else defaults.get(key) for key, value in runtime.items()}
    cfg.runtime = OmegaConf.create(runtime)
    device = cfg.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = SimpleNamespace(config=None, trainer='lightning', experiment=name, device=device, seed=cfg.seed,
                           epochs=cfg.epochs, output=cfg.output, resume=cfg.resume, initialize_from=cfg.initialize_from,
                           wandb_mode=logging.mode, wandb_project=logging.project, train=None, val=None, **runtime)
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




@hydra.main(version_base='1.3', config_path=str(project_root()/'configs'), config_name='run')
def main(cfg: DictConfig):
    execute(cfg)


if __name__ == '__main__': main()
