"""Enumerate the installed implementation for reproducible evaluation identities."""
import hashlib
from pathlib import Path
from unifiedgroupgen.common.configuration import resolve_path


def package_sources():
    root = Path(__file__).resolve().parents[1]
    files = [p for p in root.rglob('*.py')
             if not any(part in ('__pycache__', 'build') for part in p.relative_to(root).parts)
             and not any(p.is_relative_to(root/'tests'/name) for name in ('unit','legacy','reference_workflows','workflows'))
             and p != root/'tests/reference_training.py' and p.name != 'setup.py']
    return sorted(files + list((root/'evaluation/resources').glob('*.json'))
                  + list((root/'common/predictors').glob('*.json'))
                  + list((root/'pl_modules/backbones/official').glob('*.json')))


def file_hash(path):
    digest = hashlib.sha256()
    with resolve_path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()
