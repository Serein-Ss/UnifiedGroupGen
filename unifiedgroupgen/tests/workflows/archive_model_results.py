"""Quarantine explicitly selected results, preserving relative paths and file hashes."""
import argparse
import hashlib
import json
import os
from pathlib import Path


def filesystem(path):
    """Use extended Windows paths for nested test artifacts, without system changes."""
    path=Path(path).absolute()
    return Path('\\\\?\\'+str(path)) if os.name=='nt' and not str(path).startswith('\\\\?\\') else path


def digest(path):
    value = hashlib.sha256()
    with filesystem(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def checked_paths(root, archive, plan, resume=False):
    root, archive = Path(root).resolve(), Path(archive).resolve()
    if not archive.is_relative_to(root) or archive == root:
        raise ValueError('Archive must be a separate directory inside the project')
    protected = [(root/p).resolve() for p in plan['protected']]
    entries = []
    for entry in plan['moves']:
        source = (root/entry['source']).resolve()
        if not source.is_relative_to(root) or source == root or source.is_relative_to(archive):
            raise ValueError('Source escapes the project or selects the archive')
        if source == root/'.git' or source.is_relative_to(root/'.git'):
            raise ValueError('Git history is outside this cleanup')
        if any(source.is_relative_to(p) or p.is_relative_to(source) for p in protected):
            raise ValueError('Move overlaps a protected current input or model: '+str(source))
        destination = archive/source.relative_to(root)
        if not filesystem(source).exists() and not (resume and filesystem(destination).exists()):
            raise FileNotFoundError(source)
        if filesystem(destination).exists() and (not resume or filesystem(source).exists()):
            raise FileExistsError(destination)
        if any(source.is_relative_to(s) or s.is_relative_to(source) for s, _, _ in entries):
            raise ValueError('Overlapping move sources')
        entries.append((source, destination, entry['reason']))
    return root, archive, entries


def move_results(root, archive, plan, resume=False):
    root, archive, entries = checked_paths(root, archive, plan, resume)
    archive.mkdir(parents=True, exist_ok=True)
    journal = archive/'moves.jsonl'
    if journal.exists() and not resume:
        raise FileExistsError('An archive journal already exists; inspect it before resuming')
    previous=[json.loads(line) for line in journal.read_text(encoding='utf-8').splitlines()] if journal.exists() else []
    receipts={r['source']:r for r in previous if r['state']=='prepared'}
    complete={r['source'] for r in previous if r['state']=='verified'}
    with journal.open('a' if resume else 'x', encoding='utf-8') as stream:
        for source, destination, reason in entries:
            key=source.relative_to(root).as_posix()
            if key in complete:
                continue
            receipt=receipts.get(key)
            if receipt is None:
                physical=filesystem(source)
                inventory=[]
                physical_root=filesystem(root)
                resolved_parents={}
                boundary=physical if physical.is_dir() else physical.parent
                for path in ([physical] if physical.is_file() else sorted(p for p in physical.rglob('*') if p.is_file())):
                    if path.parent not in resolved_parents:
                        resolved_parents[path.parent]=path.parent.resolve()
                    resolved_path=path.resolve() if path.is_symlink() else resolved_parents[path.parent]/path.name
                    if not resolved_path.is_relative_to(boundary):
                        raise ValueError('Refusing to traverse a link outside the selected result')
                    inventory.append({'source':path.relative_to(physical_root).as_posix(),
                                      'bytes':path.stat().st_size,'sha256':digest(path)})
                receipt={'source':key,'reason':reason,'state':'prepared','files':inventory}
                receipts[key]=receipt
                stream.write(json.dumps(receipt,ensure_ascii=False)+'\n');stream.flush()
            inventory=receipt['files']
            if filesystem(source).exists():
                filesystem(destination.parent).mkdir(parents=True, exist_ok=True)
                filesystem(source).rename(filesystem(destination))
            for item in inventory:
                target = archive/item['source']
                if filesystem(target).stat().st_size != item['bytes'] or digest(target) != item['sha256']:
                    raise ValueError('Archived bytes changed: '+str(target))
            stream.write(json.dumps({'source': receipt['source'], 'state': 'verified',
                                     'files': len(inventory)}, ensure_ascii=False)+'\n'); stream.flush()
            print(json.dumps({'moved': receipt['source'], 'files': len(inventory)}, ensure_ascii=False), flush=True)
    result = {'state': 'verified', 'moves': len(entries), 'files': sum(len(r['files']) for r in receipts.values()),
              'bytes':sum(f['bytes'] for r in receipts.values() for f in r['files']),
              'original_files_deleted': False, 'scope': 'Quarantine only; no scientific result certified'}
    (archive/'migration.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    _, _, entries = checked_paths(args.root, args.archive, plan, args.resume)
    result = move_results(args.root, args.archive, plan, args.resume) if args.run else {'moves': len(entries), 'started': False}
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
