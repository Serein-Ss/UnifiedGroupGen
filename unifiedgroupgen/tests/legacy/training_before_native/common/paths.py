"""Workspace paths with optional, explicit location mounts from configs/paths.yaml."""
import os
from pathlib import Path
import yaml


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
    path=Path(value).expanduser()
    path=(path if path.is_absolute() else root/path).resolve()
    package=Path(__file__).resolve().parents[1]
    settings = root/'configs/paths.yaml'
    mounts = (yaml.safe_load(settings.read_text(encoding='utf-8')) or {}).get('mounts', {}) if settings.is_file() else {}
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
