"""Hydra entry for the current joint-flow model; no legacy-model dispatch."""
from pathlib import Path
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
import hydra
import torch
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.loggers import WandbLogger
from unifiedgroupgen.common.configuration import configuration_root, project_root, resolve_path, RunSettings
from unifiedgroupgen.experiment_config import training_definition
from unifiedgroupgen.conditional.properties import fit_property_scalers
from unifiedgroupgen.pl_data.dataset import load_training_splits

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
        options = {}
        if 'joint_state' in saved:
            state = saved['joint_state']
            same_directory = state.get('wandb_run_directory') == str(output.parent.resolve())
            previous = state.get('wandb_id') if same_directory and settings.wandb_mode == 'online' else None
            run_id = previous or uuid.uuid4().hex
            # Offline sessions get separate IDs; old SDKs cannot reliably resume them.
            options['resume'] = ('allow' if previous else 'never') if settings.wandb_mode == 'online' else None
        else:
            run_id = uuid.uuid4().hex
        loggers.append(WandbLogger(project=getattr(args, 'wandb_project', None) or 'UnifiedGroupGen',
                                  name=getattr(args, 'experiment', None) or output.parent.parent.name,
                                  id=run_id, save_dir=str(settings.wandb_dir),
                                  offline=settings.wandb_mode == 'offline', mode=settings.wandb_mode,
                                  log_model=False, **options))
    return loggers

def fit_training_config(config, records):
    """Use the same TRAIN-only element domain and property scalers in training and timing."""
    config = dict(config)
    if config.get('element_policy', 'training') == 'training' and not config.get('allowed_elements'):
        config['allowed_elements'] = sorted({o['atomic_number'] for r in records for o in r['descriptor']['orbits']})
    config['properties'] = fit_property_scalers(records, config.get('properties', {}))
    return config

def train_args(args, config):
    root = project_root()
    records, validation, paths = load_training_splits(config, root, getattr(args, 'train', None), getattr(args, 'val', None))
    if config.get('data_audit'):
        from unifiedgroupgen.pl_data.validation import verify_training_audit
        verify_training_audit(resolve_path(config['data_audit'], root), config, paths)
    config = fit_training_config(config, records)
    args.resume = resolve_path(args.resume, root) if args.resume else None
    if getattr(args, 'initialize_from', None):
        args.initialize_from = resolve_path(args.initialize_from, root)
    from unifiedgroupgen.pl_modules.unified.module import run_joint_training
    return run_joint_training(args, config, records, validation, paths)

def execute(cfg: DictConfig):
    logging = cfg.logging
    settings = RunSettings.load(project_root(), logging.mode)
    if cfg.resume and cfg.initialize_from:
        raise ValueError('resume restores a run; initialize_from starts a new run. Choose one.')
    config = training_definition(cfg)
    from hydra.core.hydra_config import HydraConfig
    selected = HydraConfig.get().runtime.choices.get('experiment', 'training') if HydraConfig.initialized() else 'training'
    name = config.get('experiment_id') or selected
    cfg.experiment_name = name
    runtime = OmegaConf.to_container(cfg.runtime, resolve=True)
    if cfg.resume:
        # Inherit execution settings from the checkpoint unless explicitly overridden.
        overrides = HydraConfig.get().overrides.task if HydraConfig.initialized() else []
        explicit = {item.split('=', 1)[0].lstrip('+') for item in overrides}
        runtime = {key: value if 'runtime' in explicit or 'runtime.' + key in explicit else None
                   for key, value in runtime.items()}
    cfg.runtime = OmegaConf.create(runtime)
    device = cfg.device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args = SimpleNamespace(trainer='lightning', experiment=name, device=device, seed=cfg.seed,
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
    return train_args(args, config)

@hydra.main(version_base='1.3', config_path=str(configuration_root()/'configs'), config_name='run')
def main(cfg: DictConfig):
    execute(cfg)

if __name__ == '__main__': main()
