"""Copy complete historical training runs from a server ZIP; never overwrite sources.

Directory dates describe the import, not the unknown historical training start.
"""
import argparse
import csv
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

REQUIRED = ('model.pt', 'model.best.pt', 'model.history.jsonl', 'model.status.json')
MAIN15 = ('mp20_u', 'c2db51_u', 'joint_u', 'mp20_all', 'c2db51_all', 'joint_all',
          'mp20_ef', 'mp20_eg', 'mp20_eh', 'c2db51_ef', 'c2db51_eg', 'c2db51_eh',
          'joint_ef', 'joint_eg', 'joint_eh')
NOTE = '历史导入时间，非训练开始时间'


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def safe_relative(name):
    relative = PurePosixPath(name)
    if relative.is_absolute() or '..' in relative.parts or ':' in name or '\\' in name:
        raise ValueError('Unsafe archive path: ' + name)
    return relative


def catalog(archive):
    names = [entry.filename for entry in archive.infolist() if not entry.is_dir()]
    if len(names) != len(set(names)):
        raise ValueError('Duplicate archive members')
    marker = 'artifacts/runs/main15_seed42/runner.json'
    matches = [name for name in names if name.endswith(marker)]
    if len(matches) != 1:
        raise ValueError('Expected one main15 seed42 runner')
    prefix = matches[0][:-len(marker)]
    available = {name[len(prefix):] for name in names if name.startswith(prefix)}
    runner = json.loads(archive.read(prefix + marker))
    main_jobs = {job['name']: job for job in runner['jobs']}
    specs = [(name, f'artifacts/runs/main15_seed42/{name}/', f'configs/main15/{name}.yaml',
              'main15', main_jobs.get(name)) for name in MAIN15]
    result = []
    for name, source, config, family, job in specs:
        for filename in REQUIRED:
            if source + filename not in available:
                raise ValueError('Missing training artifact: ' + source + filename)
        status = json.loads(archive.read(prefix + source + 'model.status.json'))
        if status.get('state') != 'complete':
            raise ValueError('Training is not complete: ' + name)
        history = [json.loads(line) for line in
                   archive.read(prefix + source + 'model.history.jsonl').decode().splitlines() if line.strip()]
        epochs = [row['epoch'] for row in history]
        if set(epochs) != set(range(status['completed_epochs'])):
            raise ValueError('Missing or unexpected training epochs: ' + name)
        raw_files = [(path, path[len(source):]) for path in sorted(available) if path.startswith(source)]
        for path, relative in raw_files:
            safe_relative(relative)
        files = [(path, relative) for path, relative in raw_files if relative in REQUIRED or relative == 'train.log']
        log = f'artifacts/runs/main15_seed42/{name}.log'
        if not any(target == 'train.log' for _, target in files) and log in available:
            files.append((log, 'train.log'))
        if not any(target == 'train.log' for _, target in files):
            raise ValueError('Missing training log: '+name)
        if config not in available:
            raise ValueError('Missing source configuration: ' + config)
        for path, target in files:
            safe_relative(path)
            safe_relative(target)
        result.append({'experiment': name, 'family': family, 'source_directory': source,
                       'training_status': status, 'history_rows': len(history),
                       'duplicate_history_rows': len(epochs) - len(set(epochs)),
                       'recorded_training_job': job, 'source_config': archive.read(prefix+config).decode('utf-8'), 'files': files})
    return prefix, result


def import_runs(source, output, imported_at=None):
    source, output = Path(source).resolve(), Path(output).resolve()
    imported_at = imported_at or datetime.now(timezone(timedelta(hours=8)))
    if imported_at.tzinfo is None:
        raise ValueError('Import timestamp must include a timezone')
    stamp = imported_at.astimezone(timezone(timedelta(hours=8))).strftime('%y-%m-%d-%H-%M-%S')
    with ZipFile(source) as archive:
        prefix, runs = catalog(archive)
        for run in runs:
            if (output / run['experiment'] / stamp).exists():
                raise FileExistsError('Import directory already exists: ' + run['experiment'] + '/' + stamp)
        print('Hashing source ZIP...', flush=True)
        archive_sha = digest(source)
        registry = {'schema': 1, 'state': 'importing', 'imported_at': imported_at.isoformat(),
                    'directory_date_kind': 'import_time', 'date_note': NOTE,
                    'training_started_at': None, 'archive': str(source), 'archive_sha256': archive_sha,
                    'models': [], 'scope': '15 complete training runs; excludes short studies and separate evaluation campaigns',
                    'output_root': str(output)}
        output.mkdir(parents=True, exist_ok=True)
        audit = output.parent/'intermediate/training_imports'/stamp
        audit.mkdir(parents=True, exist_ok=False)
        index = audit/'index.json'
        if index.exists():
            raise FileExistsError(str(index))
        save_json(index, registry)
        for run in runs:
            destination = output / run['experiment'] / stamp
            destination.mkdir(parents=True, exist_ok=False)
            entries = []
            for path, relative in run['files']:
                target = destination.joinpath(*safe_relative(relative).parts)
                if not target.resolve().is_relative_to(destination.resolve()):
                    raise ValueError('Output escapes run directory')
                target.parent.mkdir(parents=True, exist_ok=True)
                checksum = hashlib.sha256()
                with archive.open(prefix + path) as incoming, target.open('xb') as outgoing:
                    for block in iter(lambda: incoming.read(8 * 1024 * 1024), b''):
                        checksum.update(block)
                        outgoing.write(block)
                expected = checksum.hexdigest()
                if digest(target) != expected:
                    raise ValueError('Copied file hash mismatch: ' + str(target))
                entries.append({'source_member': prefix + path, 'path': relative,
                                'size': target.stat().st_size, 'sha256': expected})
            manifest = {key: value for key, value in run.items() if key != 'files'}
            manifest.update(imported_at=imported_at.isoformat(), directory_date_kind='import_time',
                            date_note=NOTE, training_started_at=None, archive=str(source),
                            archive_sha256=archive_sha, files=entries, state='verified')
            manifest_path = audit/(run['experiment']+'.json')
            save_json(manifest_path, manifest)
            registry['models'].append({'experiment': run['experiment'], 'family': run['family'],
                                       'directory': destination.relative_to(output).as_posix(),
                                       'completed_epochs': run['training_status']['completed_epochs'],
                                       'files': len(entries), 'manifest': str(manifest_path), 'manifest_sha256': digest(manifest_path)})
            save_json(index, registry)
            print('[VERIFIED] ' + run['experiment'] + ': ' + str(len(entries)) + ' files', flush=True)
        registry['state'] = 'complete'
        save_json(index, registry)
        with index.with_suffix('.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(registry['models'][0]))
            writer.writeheader()
            writer.writerows(registry['models'])
    verify_import(index)
    return index


def verify_import(index):
    index = Path(index).resolve()
    registry = json.loads(index.read_text(encoding='utf-8'))
    if registry['state'] != 'complete' or len(registry['models']) != len(MAIN15):
        raise ValueError('Import is incomplete')
    count = 0
    output = Path(registry.get('output_root', index.parent)).resolve()
    for run in registry['models']:
        directory = output.joinpath(*safe_relative(run['directory']).parts)
        if not directory.resolve().is_relative_to(output):
            raise ValueError('Run escapes output root')
        path = Path(run['manifest']) if 'manifest' in run else directory/'import_manifest.json'
        if digest(path) != run['manifest_sha256']:
            raise ValueError('Manifest hash mismatch: ' + str(path))
        manifest = json.loads(path.read_text(encoding='utf-8'))
        for entry in manifest['files']:
            target = directory.joinpath(*safe_relative(entry['path']).parts)
            if not target.resolve().is_relative_to(directory.resolve()):
                raise ValueError('File escapes run directory')
            if digest(target) != entry['sha256'] or target.stat().st_size != entry['size']:
                raise ValueError('Artifact hash mismatch: ' + str(target))
            count += 1
    return {'models': len(registry['models']), 'verified_files': count, 'state': 'verified'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--verify', type=Path)
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_import(args.verify)))
        return
    if not args.archive:
        parser.error('--archive is required')
    root = Path(__file__).resolve().parents[1]
    env_path = root / '.env'
    settings = dict(line.split('=', 1) for line in env_path.read_text(encoding='utf-8').splitlines()
                    if line.strip() and not line.lstrip().startswith('#') and '=' in line) if env_path.exists() else {}
    configured = os.environ.get('HYDRA_JOBS') or settings.get('HYDRA_JOBS', '').strip().strip('"\'')
    output = args.output or (Path(configured) if configured else root / 'outputs')
    if args.run:
        print(import_runs(args.archive, output))
    else:
        with ZipFile(args.archive) as archive:
            _, runs = catalog(archive)
            print(json.dumps({'output': str(output), 'models': [run['experiment'] for run in runs],
                              'date_note': NOTE, 'started': False}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
