"""Prepare and verify short-experiment moves; filesystem moves use native PowerShell."""
import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from zipfile import ZipFile

from import_training_outputs import digest, safe_relative, save_json, NOTE


def snapshot(source, target):
    files = []
    for path in sorted(source.rglob('*')):
        if path.is_symlink() or getattr(path, 'is_junction', lambda: False)():
            raise ValueError('Refuse to move linked files or directories: ' + str(path))
        if path.is_file():
            files.append({'path': path.relative_to(source).as_posix(), 'size': path.stat().st_size,
                          'sha256': digest(path)})
    return {'source': str(source), 'target': str(target), 'files': files}


def verify_files(directory, files):
    for entry in files:
        target = directory.joinpath(*safe_relative(entry['path']).parts)
        if not target.resolve().is_relative_to(directory.resolve()):
            raise ValueError('File escapes experiment directory')
        if target.stat().st_size != entry['size'] or digest(target) != entry['sha256']:
            raise ValueError('Hash mismatch: ' + str(target))


def prepare(root, archive_path):
    root = Path(root).resolve()
    output = root/'intermediate/short_experiments'
    now = datetime.now(timezone(timedelta(hours=8)))
    stamp = now.strftime('%y-%m-%d-%H-%M-%S')
    local = [
        ('acceleration_pairing', 'artifacts/paired_training_20261023', ''),
        ('loss_balance_local', 'artifacts/loss_balance_20261023', ''),
        ('engineering_smoke', 'artifacts/smoke', ''),
        ('engineering_pretrain_validation', 'artifacts/pretrain_validation', ''),
        ('benchmark_local', 'artifacts/benchmarks', ''),
        ('loss_balance_4090', 'artifacts/analysis/loss_balance_4090_seed42_20261024', 'analysis'),
        ('runtime_pilot_selected_3A', 'artifacts/analysis/selected_3A_20261024_161510', 'analysis'),
        ('benchmark_4090', 'artifacts/analysis/batch_throughput_20261024', 'analysis'),
    ]
    remote = [
        ('loss_balance_4090', 'artifacts/loss_balance_4090_seed42/'),
        ('runtime_pilot_mp20_threads8', 'artifacts/runtime_pilots/mp20_all_t8_20261023_162050/'),
        ('runtime_pilot_selected_3A', 'artifacts/runtime_pilots/selected_3A_20261024_161510/'),
        ('benchmark_4090', 'artifacts/benchmarks/'),
    ]
    extras = {
        'runtime_pilot_mp20_threads8': ['artifacts/runtime_pilots/mp20_all_t8_20261023_162050.driver.log',
                                      'artifacts/runtime_pilots/mp20_pilot_audit_20261023_164151.zip'],
        'runtime_pilot_selected_3A': ['artifacts/runtime_pilots/selected_3A_20261024_161510.driver.log',
                                     'artifacts/runtime_pilots/selected_3A_20261024_161510.zip',
                                     'artifacts/runtime_pilots/selected_3A_20261024_161510.zip.sha256'],
    }
    names = sorted({name for name, *_ in local + remote})
    for name in names:
        if (output/name/stamp).exists():
            raise FileExistsError(str(output/name/stamp))
    for _, old, _ in local:
        if not (root/old).is_dir():
            raise FileNotFoundError(str(root/old))
    for old in ('artifacts/paired_training_20261023', 'artifacts/loss_balance_20261023'):
        if json.loads((root/old/'status.json').read_text(encoding='utf-8'))['state'] != 'complete':
            raise ValueError('Study still active: ' + old)
    plan = {'root': str(root), 'output': str(output), 'imported_at': now.isoformat(),
            'stamp': stamp, 'date_note': NOTE, 'state': 'preparing', 'archive': str(Path(archive_path).resolve()),
            'groups': names, 'moves': [], 'remote': [], 'aliases': {}}
    for name, old, tail in local:
        source, target = root/old, output/name/stamp/tail
        print('[HASH LOCAL] ' + old, flush=True)
        plan['moves'].append(snapshot(source, target))
        plan['aliases'][old] = target.relative_to(root).as_posix()
    with ZipFile(archive_path) as archive:
        prefix = 'UnifiedGroupGen/'
        available = {i.filename: i for i in archive.infolist() if not i.is_dir()}
        jobs = []
        for name, source in remote:
            selected = [(n, n[len(prefix+source):]) for n in available if n.startswith(prefix+source)]
            selected += [(prefix+n, 'provenance/'+Path(n).name) for n in extras.get(name, []) if prefix+n in available]
            if not selected:
                raise ValueError('Missing archive experiment: ' + source)
            for original, relative in selected:
                safe_relative(original); safe_relative(relative)
            jobs.append((name, source, selected))
        for name, source, selected in jobs:
            destination = output/name/stamp
            destination.mkdir(parents=True, exist_ok=False)
            files = []
            for original, relative in selected:
                target = destination.joinpath(*safe_relative(relative).parts)
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError('Archive output escapes experiment directory')
                target.parent.mkdir(parents=True, exist_ok=True)
                import hashlib
                checksum = hashlib.sha256()
                with archive.open(original) as incoming, target.open('xb') as outgoing:
                    for block in iter(lambda: incoming.read(8*1024*1024), b''):
                        outgoing.write(block); checksum.update(block)
                files.append({'path': relative, 'source_member': original,
                              'size': target.stat().st_size, 'sha256': checksum.hexdigest()})
            verify_files(destination, files)
            plan['remote'].append({'experiment': name, 'directory': str(destination), 'files': files})
            plan['aliases'].setdefault(source.rstrip('/'), destination.relative_to(root).as_posix())
            print('[VERIFIED ZIP] ' + name + ': ' + str(len(files)) + ' files', flush=True)
    output.mkdir(parents=True, exist_ok=True)
    plan['state'] = 'ready_for_native_move'
    path = output/('migration_'+stamp+'.json')
    save_json(path, plan)
    print('MOVE_PLAN=' + str(path), flush=True)
    return path


def finalize(path):
    path = Path(path).resolve()
    plan = json.loads(path.read_text(encoding='utf-8'))
    if plan['state'] != 'ready_for_native_move':
        raise ValueError('Migration is not ready to finalize')
    root = Path(plan['root']).resolve()
    output = root/'intermediate/short_experiments'
    grouped = {name: [] for name in plan['groups']}
    for move in plan['moves']:
        source, target = Path(move['source']), Path(move['target'])
        if source.exists():
            raise ValueError('Old source was not moved: ' + str(source))
        verify_files(target, move['files'])
        name = target.relative_to(output).parts[0]
        base = output/name/plan['stamp']
        for entry in move['files']:
            grouped[name].append({**entry, 'path': (target/entry['path']).relative_to(base).as_posix(),
                                  'source_path': str(source/entry['path'])})
    for remote in plan['remote']:
        verify_files(Path(remote['directory']), remote['files'])
        grouped[remote['experiment']].extend(remote['files'])
    summaries = []
    for name, files in grouped.items():
        destination = output/name/plan['stamp']
        histories = []
        for entry in files:
            if entry['path'].endswith('.history.jsonl'):
                rows = [json.loads(s) for s in (destination/entry['path']).read_text(encoding='utf-8').splitlines() if s.strip()]
                histories.append({'path': entry['path'], 'rows': len(rows),
                                  'train_records': rows[-1].get('train_records') if rows else None})
        manifest = {'experiment': name, 'imported_at': plan['imported_at'], 'date_note': NOTE,
                    'directory_date_kind': 'import_time', 'training_started_at': None,
                    'histories': histories, 'files': files, 'state': 'verified'}
        save_json(destination/'migration_manifest.json', manifest)
        (destination/'IMPORT_TIME_NOTE.txt').write_text(NOTE+'\n', encoding='utf-8')
        summaries.append({'experiment': name, 'directory': destination.relative_to(output).as_posix(),
                          'files': len(files), 'bytes': sum(f['size'] for f in files), 'histories': histories})
    aliases = output/'path_aliases.json'
    # The local and remote benchmark collections share a historical relative name.
    # Existing local callers retain their local collection; remote data has its own explicit directory.
    for move in plan['moves']:
        plan['aliases'][Path(move['source']).relative_to(root).as_posix()] = Path(move['target']).relative_to(root).as_posix()
    prior = json.loads(aliases.read_text(encoding='utf-8')) if aliases.exists() else {}
    if any(key in prior and prior[key] != value for key, value in plan['aliases'].items()):
        raise ValueError('Conflicting historical path aliases')
    save_json(aliases, {**prior, **plan['aliases']})
    plan.update(state='verified', summary=summaries)
    save_json(path, plan)
    print(json.dumps({'state': 'verified', 'groups': len(summaries),
                      'files': sum(s['files'] for s in summaries),
                      'bytes': sum(s['bytes'] for s in summaries)}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--finalize', type=Path)
    args = parser.parse_args()
    if args.finalize:
        finalize(args.finalize)
    elif args.archive:
        prepare(Path(__file__).resolve().parents[1], args.archive)
    else:
        parser.error('--archive or --finalize is required')


if __name__ == '__main__':
    main()
