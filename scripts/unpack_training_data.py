"""Verify bundled bytes, extract full data, and rebind audited paths to this checkout."""

import argparse
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rebind(value, mappings):
    if isinstance(value, dict):
        return {rebind(key, mappings): rebind(item, mappings) for key, item in value.items()}
    if isinstance(value, list):
        return [rebind(item, mappings) for item in value]
    if isinstance(value, str):
        normalized = value.replace('\\', '/')
        for previous, current in mappings:
            if normalized == previous or normalized.startswith(previous + '/'):
                return str(current / normalized[len(previous):].lstrip('/'))
    return value


def unpack(bundle, root):
    root = root.resolve()
    manifest = json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))
    expected = {item['path']: item for item in manifest['files']}
    mappings = [(manifest['original_root'].replace('\\', '/'), root),
                (manifest['original_data_root'].replace('\\', '/'), root / 'raw_data')]
    # Repeat imports are idempotent; refuse to overwrite changed training data.
    for name, item in expected.items():
        path = root / name
        if not path.resolve().is_relative_to(root):
            raise ValueError(f'Unsafe inventory path: {name}')
        if path.exists() and path.suffix not in ('.json',):
            if sha256(path) != item['sha256']:
                raise ValueError(f'Existing data differ from the published bundle: {name}')
    root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile() as archive:
        digest = hashlib.sha256()
        for item in manifest['parts']:
            part = bundle / item['name']
            if part.stat().st_size != item['bytes'] or sha256(part) != item['sha256']:
                raise ValueError(f'Corrupt or incomplete dataset part: {part.name}')
            with part.open('rb') as stream:
                for block in iter(lambda: stream.read(1024 * 1024), b''):
                    archive.write(block)
                    digest.update(block)
        if digest.hexdigest() != manifest['archive_sha256']:
            raise ValueError('Combined dataset archive hash differs')
        archive.seek(0)
        with tarfile.open(fileobj=archive, mode='r:gz') as data:
            members = data.getmembers()
            names = [member.name for member in members]
            if len(names) != len(expected) or set(names) != set(expected):
                raise ValueError('Archive inventory differs from the manifest')
            for member in members:
                target = (root / member.name).resolve()
                if not target.is_relative_to(root) or not member.isfile():
                    raise ValueError(f'Unsafe archive member: {member.name}')
                target.parent.mkdir(parents=True, exist_ok=True)
                with data.extractfile(member) as source, target.open('wb') as output:
                    digest = hashlib.sha256()
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        output.write(block)
                        digest.update(block)
                if digest.hexdigest() != expected[member.name]['sha256']:
                    raise ValueError(f'Extracted file hash differs: {member.name}')
    for name in expected:
        if name.endswith('.json'):
            path = root / name
            original = json.loads(path.read_text(encoding='utf-8'))
            current = rebind(original, mappings)
            if current != original:
                path.write_text(json.dumps(current, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    receipt = {'passed': True, 'archive_sha256': manifest['archive_sha256'], 'files_verified': len(expected),
               'total_source_records': manifest['total_source_records'], 'root': str(root),
               'scope': 'Published bytes verified; metadata paths rebound; training gates must also pass.'}
    path = root / 'artifacts/dataset_import.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args()
    print(json.dumps(unpack(ROOT / 'datasets', args.root)))


if __name__ == '__main__':
    main()
