"""Bind optional upstream cores to their actual installed source and resource bytes."""
import hashlib
import importlib
from importlib.metadata import version
from pathlib import Path


def backbone_identity(backbone):
    module = importlib.import_module(type(backbone.core).__module__)
    directory = Path(module.__file__).parent
    files = sorted(p for p in directory.rglob('*') if p.suffix in ('.py', '.json', '.pt'))
    packages = ['torch', 'numpy', 'torch-geometric', 'torch-scatter', 'torch-sparse', 'einops']
    if '.official.fairchem.' in module.__name__:
        packages += ['e3nn', 'opt-einsum-fx']
    return {'core': type(backbone.core).__module__+'.'+type(backbone.core).__name__,
            'versions': {name: version(name) for name in packages},
            'files': {p.relative_to(directory).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}
