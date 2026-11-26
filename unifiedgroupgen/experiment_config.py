"""Compose the current training experiment from domain-specific YAML settings."""
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from unifiedgroupgen.common.configuration import configuration_root


def training_definition(cfg):
    groups = OmegaConf.to_container(cfg, resolve=True)
    model = dict(groups['model'])
    limits = {key: model.pop(key) for key in ('max_atoms', 'max_orbits')}
    # An omitted auxiliary weight and an explicit zero define the same run.
    if groups['train'].get('orbit_recovery_weight') == 0:
        groups['train'].pop('orbit_recovery_weight')
    if groups['train'].get('geometry_objective') is None:
        groups['train'].pop('geometry_objective', None)
    for key in ('short_contact_weight', 'layer_log_area_weight', 'layer_height_weight', 'terminal_contact_repair'):
        if groups['train'].get(key) is None:
            groups['train'].pop(key, None)
    name = groups['experiment_name']
    if 'orbit_ablation' in groups['train']:
        mode = groups['train']['orbit_ablation']
        suffix = 'control' if mode == 'joint' else mode
        name = f"{name or 'training'}_{suffix}"
    return {
        **groups['data'], **limits, 'model': model, **groups['conditioning'],
        **groups['optim'], **groups['train'],
        'experiment_id': name,
    }


def compose_experiment(name, root=None, overrides=()):
    """Use the training entry's Hydra defaults without launching a run."""
    root = Path(root).resolve() if root is not None else configuration_root()
    hydra = GlobalHydra.instance()
    previous = hydra.hydra
    hydra.clear()
    try:
        with initialize_config_dir(config_dir=str(root/'configs'), version_base='1.3'):
            return compose(config_name='run', overrides=[f'experiment={name}', *overrides])
    finally:
        hydra.hydra = previous
