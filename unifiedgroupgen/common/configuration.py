"""Shared workspace paths, YAML loading and .env settings."""
import os
from dataclasses import dataclass
from pathlib import Path

from omegaconf import OmegaConf


def package_parent():
    """Parent of the actual imported package, including an installed wheel."""
    return Path(__file__).resolve().parents[2]


def runtime_pythonpath():
    """Keep workers on this implementation, preserving other explicit search paths."""
    paths = [str(package_parent()), *os.environ.get('PYTHONPATH', '').split(os.pathsep)]
    return os.pathsep.join(dict.fromkeys(p for p in paths if p))


def project_root():
    """Read the project root from the required workspace .env file."""
    return RunSettings.load().project_root


def configuration_root():
    """Prefer YAML shipped with the active checkout or frozen source."""
    source = package_parent()
    return source if (source/'configs/run.yaml').is_file() else project_root()


def resolve_path(value, root=None):
    """Resolve current paths literally, relative to the workspace when needed."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = Path(root).resolve() if root is not None else project_root()
    return (base / path).resolve()


def source_key(path, root=None):
    """Stable source labels also work when installed outside the data workspace."""
    path=Path(path).resolve();root=Path(root).resolve() if root is not None else project_root()
    if path.is_relative_to(root): return path.relative_to(root).as_posix()
    package=Path(__file__).resolve().parents[1]
    if path.is_relative_to(package): return 'unifiedgroupgen/'+path.relative_to(package).as_posix()
    return str(path)


def artifact_path(value, root=None):
    settings = RunSettings.load() if root is None else None
    root = Path(root).resolve() if root is not None else settings.project_root
    path = resolve_path(value, root)
    # Frozen HPC sources deliberately mount the live data/result directories.
    roots = [root, *((root/name).resolve() for name in ('data', 'outputs', 'intermediate', 'wandb')
                     if (root/name).is_symlink())]
    if settings is not None:
        roots.extend((settings.jobs, settings.wandb_dir))
    if not any(path.is_relative_to(base) for base in roots):
        raise ValueError('Generated artifacts must be inside UnifiedGroupGen or .env output/log directories')
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


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


def sampling_files(output):
    """Use distinct sidecars for explicit ledger filenames."""
    output = Path(output)
    prefix = '' if output.name == 'samples.jsonl' else output.name+'.'
    return output.parent/(prefix+'summary.json'), output.parent/(prefix+'sampling.status.json')


def load_yaml(path, root=None):
    value = OmegaConf.to_container(OmegaConf.load(resolve_path(path, root)), resolve=True)
    if not isinstance(value, dict):
        raise ValueError(f'Expected a YAML mapping: {path}')
    return value


@dataclass(frozen=True)
class RunSettings:
    project_root: Path
    jobs: Path
    wandb_dir: Path
    wandb_mode: str

    @classmethod
    def load(cls, root=None, mode=None):
        """Require all runtime settings in .env; never overlay shell values.

        UGG_PROJECT_ROOT only selects the directory containing .env for installed
        packages and frozen workers. Otherwise use the source checkout directory.
        """
        from dotenv import dotenv_values
        directory = Path(root if root is not None else os.environ.get('UGG_PROJECT_ROOT') or package_parent())
        env_file = directory.expanduser().resolve()/'.env'
        if not env_file.is_file():
            raise FileNotFoundError(f'Required environment file does not exist: {env_file}')
        values = dotenv_values(env_file, encoding='utf-8-sig', interpolate=False)
        required = ('PROJECT_ROOT', 'HYDRA_JOBS', 'WANDB_DIR', 'USE_WANDB_LOGGING')
        missing = [key for key in required if not values.get(key) or not values[key].strip()]
        if missing:
            raise ValueError(f'{env_file}: missing required settings: {", ".join(missing)}')
        root = Path(values['PROJECT_ROOT']).expanduser()
        if not root.is_absolute() or not root.is_dir():
            raise ValueError(f'{env_file}: PROJECT_ROOT must be an existing absolute directory')
        root = root.resolve()
        flag = values['USE_WANDB_LOGGING']
        if flag not in ('0', '1'):
            raise ValueError(f'{env_file}: USE_WANDB_LOGGING must be 1 (online) or 0 (offline)')
        if mode not in (None, 'online', 'offline', 'disabled'):
            raise ValueError('Logging mode must be online, offline or disabled')
        def output_directory(key):
            value = values[key].replace('${PROJECT_ROOT}', str(root))
            if '$' in value:
                raise ValueError(f'{env_file}: {key} only supports ${{PROJECT_ROOT}} interpolation')
            path = Path(value).expanduser()
            path = (path if path.is_absolute() else root/path).resolve()
            if any(p.exists() and not p.is_dir() for p in (path, *path.parents)):
                raise ValueError(f'{env_file}: {key} must point to a directory, not a file')
            return path
        return cls(root, output_directory('HYDRA_JOBS'), output_directory('WANDB_DIR'),
                   mode or ('online' if flag == '1' else 'offline'))
