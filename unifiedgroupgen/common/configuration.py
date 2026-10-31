"""Shared workspace paths, YAML experiments and .env settings; no training side effects."""
import os
from types import SimpleNamespace
from dataclasses import dataclass
from pathlib import Path

import yaml
from omegaconf import OmegaConf


def package_parent():
    """Parent of the actual imported package, including an installed wheel."""
    return Path(__file__).resolve().parents[2]


def runtime_pythonpath():
    """Keep workers on this implementation, preserving other explicit search paths."""
    paths = [str(package_parent()), *os.environ.get('PYTHONPATH', '').split(os.pathsep)]
    return os.pathsep.join(dict.fromkeys(p for p in paths if p))


def project_root():
    explicit=os.environ.get('UGG_PROJECT_ROOT') or os.environ.get('PROJECT_ROOT')
    if explicit: return Path(explicit).expanduser().resolve()
    for candidate in (Path.cwd(),*Path.cwd().parents):
        if (candidate/'configs/run.yaml').is_file() and (candidate/'unifiedgroupgen').is_dir():
            return candidate.resolve()
    editable=Path(__file__).resolve().parents[2]
    return editable if (editable/'configs').is_dir() else Path.cwd().resolve()


def resolve_path(value, root=None):
    """Resolve a path; location mounts are data configuration, not dataset logic."""
    root=Path(root).resolve() if root is not None else project_root()
    settings = root/'configs/paths.yaml'
    options = (yaml.safe_load(settings.read_text(encoding='utf-8')) or {}) if settings.is_file() else {}
    recorded = os.fspath(value).replace('\\', '/')
    native = Path(value).expanduser()
    if not (native.is_absolute() and native.resolve().is_relative_to(root)):
        for before, after in sorted(options.get('recorded_roots', {}).items(), key=lambda item: -len(item[0])):
            before = before.replace('\\', '/').rstrip('/')
            if recorded == before or recorded.startswith(before+'/'):
                value = root/after/recorded[len(before):].lstrip('/')
                break
    path=Path(value).expanduser()
    path=(path if path.is_absolute() else root/path).resolve()
    package=Path(__file__).resolve().parents[1]
    mounts = options.get('mounts', {})
    aliases = [(root/before, root/after) for before, after in mounts.items()]
    aliases.append((root/'unifiedgroupgen', package))
    for before,after in sorted(aliases, key=lambda pair: -len(pair[0].parts)):
        before=before.resolve()
        if path.is_relative_to(before):
            target=after/path.relative_to(before)
            return target.resolve()
    return path


def source_key(path, root=None):
    """Stable source labels also work when installed outside the data workspace."""
    path=Path(path).resolve();root=Path(root).resolve() if root is not None else project_root()
    if path.is_relative_to(root): return path.relative_to(root).as_posix()
    package=Path(__file__).resolve().parents[1]
    if path.is_relative_to(package): return 'unifiedgroupgen/'+path.relative_to(package).as_posix()
    return str(path)


def artifact_path(value, root=None):
    root = Path(root).resolve() if root is not None else project_root()
    path = resolve_path(value, root)
    if not path.is_relative_to(root):
        raise ValueError('Generated artifacts must be inside UnifiedGroupGen')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def root_path(value):
    return resolve_path(value)


def model_result_directory(checkpoint, category, task):
    """Keep generated structures and metrics beside the checkpoint that produced them."""
    if category not in ('generate', 'evaluate'):
        raise ValueError('Result category must be generate or evaluate')
    if not task or Path(task).name != task or task in ('.', '..') or any(c in task for c in '/\\:'):
        raise ValueError('Result task must be a single directory name')
    return resolve_path(checkpoint).parent/category/task


def evaluation_directory(samples):
    """Infer the matching metric task from <run>/generate/<task>/samples.jsonl."""
    samples = resolve_path(samples)
    if samples.parent.parent.name != 'generate':
        raise ValueError('Set evaluation.output for samples outside a model generate/task directory')
    return samples.parent.parent.parent/'evaluate'/samples.parent.name


def load_yaml(path, root=None):
    value = OmegaConf.to_container(OmegaConf.load(resolve_path(path, root)), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a YAML mapping: {path}')
    return value


def training_definition(cfg):
    """Compose functional groups into the flat model/data contract used by checkpoints.

    Execution overrides stay in runtime, as in the original audited experiments.
    This preserves both the audit fingerprint and inference checkpoint schema.
    """
    if 'data' not in cfg:
        return OmegaConf.to_container(cfg.experiment, resolve=True)
    groups = OmegaConf.to_container(cfg, resolve=True)
    data = dict(groups['data'])
    context = data.pop('context_properties', {})
    overrides = data.pop('property_overrides', {})
    condition = dict(groups['conditioning'])
    properties = {name: {**spec, **overrides} for name, spec in condition.pop('properties').items()}
    condition['properties'] = {**properties, **context}
    model = dict(groups['model'])
    limits = {key: model.pop(key) for key in ('max_atoms', 'max_orbits')}
    loss = dict(groups['loss'])
    loss.pop('coordinate_length_scale')  # A runtime loss setting, saved with every run.
    definition = {**data, **limits, 'model': model, **condition, **groups['optim'],
                  **loss, **groups['train'], **groups['experiment'], **groups.get('learning', {})}
    if groups.get('prior'): definition['prior'] = groups['prior']
    return definition


def compose_experiment(name, root=None, overrides=()):
    """Use the same Hydra defaults list as the training entry, without launching it."""
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    root = Path(root).resolve() if root is not None else project_root()
    # Catalogs may also be read from the Hydra evaluation entry. Restore its loader.
    state = GlobalHydra.instance().hydra
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=str(root/'configs'), version_base='1.3'):
            return compose(config_name='run', overrides=['experiment='+name, *overrides])
    finally:
        GlobalHydra.instance().hydra = state


def load_experiments(path, root=None):
    """Preserve catalog order; Hydra selectors and standalone YAML inputs share one API."""
    root = Path(root).resolve() if root is not None else project_root()
    entries = load_yaml(path, root).get('experiments')
    if not isinstance(entries, dict) or not entries:
        raise ValueError('The experiment catalog needs a nonempty experiments mapping')
    result = {}
    for name, entry in entries.items():
        if not isinstance(entry, dict) or not (entry.get('experiment') or entry.get('config')):
            raise ValueError(f'Experiment {name} needs a Hydra experiment or config path')
        if entry.get('experiment'):
            composed = compose_experiment(entry['experiment'], root, entry.get('overrides', []))
            config_path = root/'configs/experiment'/f"{entry['experiment']}.yaml"
            config = training_definition(composed)
            runtime = OmegaConf.to_container(composed.runtime, resolve=True)
        else:
            config_path = resolve_path(entry['config'], root)
            config = load_yaml(config_path, root)
            runtime = entry.get('runtime', {})
        if isinstance(runtime, str):
            runtime = load_yaml(runtime, root)
        if not isinstance(runtime, dict):
            raise ValueError(f'Experiment {name} runtime must be a mapping or YAML path')
        result[name] = {**entry, 'config_path': config_path, 'config': config,
                        'runtime': dict(runtime)}
    return result


def task_options(cfg, section):
    """Resolve an entry's configured input/output paths without importing training code."""
    values=OmegaConf.to_container(cfg[section],resolve=True)
    paths={'samples','output','cache','request','assessments','assessment_metadata','reference_assessments',
           'embeddings','experiments','checkpoint_root','cgdit_root','predictor_root','reference_cache',
           'model_cache','predictor_config','checkpoint','reuse_manifest'}
    for key in paths & values.keys():
        if values[key] is not None and not isinstance(values[key],dict):
            values[key]=resolve_path(values[key])
    for key in ('reference_files','novelty_files'):
        if key in values: values[key]=[resolve_path(p) for p in values[key]]
    values.update(device=cfg.get('device','cpu'),seed=cfg.get('seed',142),threads=cfg.get('threads',2))
    return SimpleNamespace(**values)


@dataclass(frozen=True)
class RunSettings:
    project_root: Path
    jobs: Path
    wandb_dir: Path
    wandb_mode: str

    @classmethod
    def load(cls, root, mode=None):
        from dotenv import dotenv_values
        root = Path(root).resolve()
        values = {**dotenv_values(root/'.env'), **os.environ}
        flag = str(values.get('USE_WANDB_LOGGING', '0'))
        if flag not in ('0', '1'):
            raise ValueError('USE_WANDB_LOGGING must be 1 (online) or 0 (offline)')
        def directory(key, default):
            value = Path(values.get(key) or default).expanduser()
            return (value if value.is_absolute() else root/value).resolve()
        return cls(root, directory('HYDRA_JOBS', 'outputs'), directory('WANDB_DIR', 'wandb'),
                   mode or ('online' if flag == '1' else 'offline'))
