"""Portable, hash-verified ZIP bundles for explicitly selected project files."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import zipfile


def digest(content):
    return hashlib.sha256(content).hexdigest()


def pack(root, files, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    contents = {}
    for value in files:
        path = Path(value)
        path = (path if path.is_absolute() else root/path).resolve()
        if not path.is_relative_to(root) or not path.is_file() or path == output:
            raise ValueError(f'Bundle inputs must be files inside the root: {path}')
        name = path.relative_to(root).as_posix()
        if name == 'bundle.manifest.json' or name in contents:
            raise ValueError(f'Duplicate or reserved bundle path: {name}')
        contents[name] = path
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = {name: {'sha256': digest(path.read_bytes()), 'bytes': path.stat().st_size}
                for name, path in contents.items()}
    with zipfile.ZipFile(output, 'x', zipfile.ZIP_DEFLATED) as archive:
        for name, path in contents.items():
            archive.write(path, name)
        archive.writestr('bundle.manifest.json', json.dumps(manifest, sort_keys=True))
    return manifest


def unpack(archive, root):
    root = Path(root).resolve()
    with zipfile.ZipFile(archive) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError('Duplicate bundle members')
        manifest = json.loads(source.read('bundle.manifest.json'))
        if set(names) != set(manifest) | {'bundle.manifest.json'}:
            raise ValueError('Bundle contents differ from its manifest')
        targets = {}
        for name, expected in manifest.items():
            relative = PurePosixPath(name)
            target = (root/name).resolve()
            if '\\' in name or ':' in name or relative.is_absolute() or '..' in relative.parts or not target.is_relative_to(root):
                raise ValueError(f'Unsafe bundle path: {name}')
            content = source.read(name)
            if len(content) != expected['bytes'] or digest(content) != expected['sha256']:
                raise ValueError(f'Bundle hash mismatch: {name}')
            if target.exists() and (not target.is_file() or digest(target.read_bytes()) != expected['sha256']):
                raise ValueError(f'Refusing to replace a different existing input: {target}')
            targets[target] = content
    # Validate all members before writing anything.
    for path, content in targets.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_bytes(content)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=['pack', 'unpack'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--files', nargs='+')
    args = parser.parse_args()
    if args.mode == 'pack' and not args.files:
        parser.error('pack needs --files')
    result = pack(args.root, args.files, args.archive) if args.mode == 'pack' else unpack(args.archive, args.root)
    print(json.dumps({'files': len(result), 'mode': args.mode, 'archive': str(args.archive)}))


if __name__ == '__main__':
    main()
