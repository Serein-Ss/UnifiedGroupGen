"""Export/restore the four HPC composition runs using only the standard library."""
import argparse
from datetime import datetime, timezone, timedelta
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import zipfile


MODELS = ('mp20_u_comp_v2', 'c2db51_u_comp_v2', 'joint_u_comp_v2',
          'joint_u_comp_source_control_v2')
FORMAT = 'ugg-hpc-new4-training-v1'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def name_part(value):
    if not value or value in ('.', '..') or any(c in value for c in '/\\:'):
        raise ValueError('Expected a single campaign directory name')
    return value


def files_under(folder, excluded=()):
    for directory, dirs, names in os.walk(folder, followlinks=False):
        dirs[:] = sorted(d for d in dirs if d not in excluded and d != '__pycache__'
                         and not (Path(directory)/d).is_symlink())
        for name in sorted(names):
            path = Path(directory)/name
            if name != '.env' and not path.is_symlink() and path.suffix != '.pyc':
                yield path


def export_training(root, campaign_path, output=None):
    root, campaign_path = Path(root).resolve(), Path(campaign_path).resolve()
    folder = campaign_path.parent
    if not campaign_path.is_relative_to(root):
        raise ValueError('Campaign must be inside the project')
    campaign = read(campaign_path)
    name = name_part(campaign['name'])
    if folder != root/'intermediate/hpc_runs'/name:
        raise ValueError('Campaign folder does not match its name')
    selected = set()
    best_hashes = {}
    required = ('model.best.pt', 'model.pt', 'model.history.jsonl', 'model.status.json')
    for model in MODELS:
        print('VERIFY MODEL', model, flush=True)
        run = root/'outputs'/model/name
        for filename in required:
            path = run/filename
            if not path.is_file() or path.is_symlink() or path.stat().st_size == 0:
                raise ValueError('Missing training output: '+str(path))
        if read(folder/'state'/('train_'+model+'.json')).get('state') != 'complete':
            raise ValueError('Training phase is not complete: '+model)
        if read(run/'model.status.json').get('state') != 'complete':
            raise ValueError('Training status is not complete: '+model)
        pipeline = read(folder/'evaluation'/model/'pipeline.json')
        hashes = {job['checkpoint_sha256'] for job in pipeline['identity']['jobs']}
        best_hashes[model] = digest(run/'model.best.pt')
        if hashes != {best_hashes[model]}:
            raise ValueError('Best checkpoint differs from evaluated weights: '+model)
        selected.update(files_under(run, ('generate', 'evaluate', 'wandb', '.git')))
        selected.add(folder/'state'/('train_'+model+'.json'))
    for filename in ('campaign.json', 'profile.json', 'new_catalog.yaml', 'submitted.json', 'recovery.json'):
        if (folder/filename).is_file():
            selected.add(folder/filename)
    for path in (folder/'logs').glob('*'):
        if path.is_file() and ('train' in path.name or path.name.startswith('audit-')):
            selected.add(path)
    for model in MODELS:
        if (folder/'audits'/(model+'.json')).is_file():
            selected.add(folder/'audits'/(model+'.json'))
    source = folder/'source'
    if not (source/'configs').is_dir():
        raise ValueError('Frozen source/configs is missing')
    for part in ('configs', 'unifiedgroupgen', 'scripts', 'submit_files'):
        selected.update(files_under(source/part, ('.git', 'data', 'outputs', 'intermediate', 'wandb')))
    # Exclude bundled pretrained binaries; the requested four native checkpoints stay included.
    selected = {p for p in selected if not p.is_relative_to(source) or p.suffix not in ('.pt', '.ckpt')}
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime('%Y%m%d_%H%M%S_%f')
    output = Path(output).resolve() if output else root.parent/'UnifiedGroupGen_exports'/f'{name}_new4_training_{stamp}.zip'
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_suffix('.zip.partial')
    if output.exists() or partial.exists():
        raise FileExistsError(output)
    manifest = {'format': FORMAT, 'campaign': name, 'source_revision': campaign['revision'],
                'models': list(MODELS), 'best_checkpoint_sha256': best_hashes, 'files': {}}
    with zipfile.ZipFile(partial, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=1) as archive:
        for path in sorted(selected):
            if path.is_symlink() or not path.resolve().is_relative_to(root):
                raise ValueError('External or linked input: '+str(path))
            relative = path.relative_to(root).as_posix()
            before = path.stat()
            target = zipfile.ZipInfo(relative) if path.suffix in ('.pt', '.ckpt') else relative
            value = hashlib.sha256(); size = 0
            if path.suffix in ('.pt', '.ckpt'):
                print('PACK', relative, flush=True)
            with path.open('rb') as source_stream, archive.open(target, 'w', force_zip64=True) as destination:
                for block in iter(lambda: source_stream.read(1024*1024), b''):
                    value.update(block); size += len(block); destination.write(block)
            after = path.stat()
            if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                raise ValueError('Input changed while exporting: '+str(path))
            manifest['files'][relative] = {'sha256': value.hexdigest(), 'bytes': size,
                                         'mtime_ns': before.st_mtime_ns}
        for model in MODELS:
            if manifest['files'][f'outputs/{model}/{name}/model.best.pt']['sha256'] != best_hashes[model]:
                raise ValueError('Checkpoint changed during export: '+model)
        archive.writestr('training_manifest.json', json.dumps(manifest, ensure_ascii=False, indent=2))
    partial.rename(output)
    checksum = digest(output)
    output.with_suffix('.zip.sha256').write_text(checksum+'  '+output.name+'\n', encoding='utf-8')
    print('ZIP:', output, '\nSIZE GiB:', round(output.stat().st_size/1024**3, 3), '\nSHA256:', checksum)
    return output


def restore_training(root, archive_path, results=None):
    root, archive_path = Path(root).resolve(), Path(archive_path).resolve()
    checksum = digest(archive_path)
    if checksum != archive_path.with_suffix('.zip.sha256').read_text(encoding='utf-8').split()[0]:
        raise ValueError('Downloaded archive checksum mismatch')
    targets = {}
    with zipfile.ZipFile(archive_path) as archive:
        manifest = json.loads(archive.read('training_manifest.json'))
        if manifest.get('format') != FORMAT or manifest.get('models') != list(MODELS):
            raise ValueError('Not the expected four-model training bundle')
        name = name_part(manifest['campaign'])
        prefixes = tuple(f'outputs/{model}/{name}/' for model in MODELS)+(f'intermediate/hpc_runs/{name}/',)
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest['files'])|{'training_manifest.json'}:
            raise ValueError('Unexpected or duplicate archive entries')
        for member, receipt in manifest['files'].items():
            parts = PurePosixPath(member).parts
            if (not member.startswith(prefixes) or PurePosixPath(member).is_absolute()
                    or member != PurePosixPath(member).as_posix()
                    or any(p in ('.', '..', '.git', '.env') or ':' in p for p in parts) or '\\' in member
                    or stat.S_ISLNK(archive.getinfo(member).external_attr >> 16)):
                raise ValueError('Unsafe archive path: '+member)
            target = root.joinpath(*parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError('Target escapes project: '+member)
            value = hashlib.sha256(); size = 0
            with archive.open(member) as stream:
                for block in iter(lambda: stream.read(1024*1024), b''):
                    value.update(block); size += len(block)
            if value.hexdigest() != receipt['sha256'] or size != receipt['bytes']:
                raise ValueError('Bundle member checksum mismatch: '+member)
            if target.exists():
                if not target.is_file() or target.is_symlink() or digest(target) != receipt['sha256']:
                    raise FileExistsError('Existing file differs; nothing imported: '+str(target))
            targets[member] = target
        for model in MODELS:
            for filename in ('model.best.pt', 'model.pt', 'model.history.jsonl', 'model.status.json'):
                member = f'outputs/{model}/{name}/{filename}'
                if member not in targets:
                    raise ValueError('Missing required training member: '+member)
            best = manifest['files'][f'outputs/{model}/{name}/model.best.pt']['sha256']
            if best != manifest['best_checkpoint_sha256'][model]:
                raise ValueError('Best-checkpoint receipt mismatch: '+model)
        if results:
            results = Path(results)
            if digest(results) != results.with_suffix('.zip.sha256').read_text(encoding='utf-8').split()[0]:
                raise ValueError('Evaluation result archive checksum mismatch')
            with zipfile.ZipFile(results) as previous:
                original = json.loads(previous.read('manifest.json'))
                if original['campaign'] != name or original['source_revision'] != manifest['source_revision']:
                    raise ValueError('Training and evaluation campaigns differ')
                for model in MODELS:
                    pipeline = json.loads(previous.read(f'models/{model}/campaign_evaluation/pipeline.json'))
                    hashes = {job['checkpoint_sha256'] for job in pipeline['identity']['jobs']}
                    if hashes != {manifest['best_checkpoint_sha256'][model]}:
                        raise ValueError('Training weights differ from local evaluated checkpoint: '+model)
        created = skipped = 0
        for member, target in targets.items():
            if target.exists():
                skipped += 1
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name+'.import-partial')
            with temporary.open('xb') as destination, archive.open(member) as source_stream:
                shutil.copyfileobj(source_stream, destination, length=1024*1024)
            if digest(temporary) != manifest['files'][member]['sha256']:
                raise ValueError('Restored file checksum mismatch: '+member)
            if target.exists():
                raise FileExistsError('Target appeared during import: '+str(target))
            temporary.rename(target)
            modified = manifest['files'][member]['mtime_ns']
            os.utime(target, ns=(modified, modified))
            created += 1
    receipt = root/'intermediate/server_training_imports'/archive_path.stem
    receipt.mkdir(parents=True, exist_ok=True)
    (receipt/'training_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    result = {'archive': str(archive_path), 'archive_sha256': checksum, 'campaign': name,
              'source_revision': manifest['source_revision'], 'models': list(MODELS),
              'files_restored': created, 'identical_files_skipped': skipped,
              'matched_to_local_evaluation_archive': bool(results)}
    (receipt/'import_verification.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print('CONFIGS/LOGS:', root/'intermediate/hpc_runs'/name)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    exporter = sub.add_parser('export')
    exporter.add_argument('--root', type=Path, required=True)
    exporter.add_argument('--campaign', type=Path, required=True)
    exporter.add_argument('--output', type=Path)
    importer = sub.add_parser('restore')
    importer.add_argument('--root', type=Path, required=True)
    importer.add_argument('--archive', type=Path, required=True)
    importer.add_argument('--results', type=Path)
    args = parser.parse_args()
    if args.action == 'export':
        export_training(args.root, args.campaign, args.output)
    else:
        restore_training(args.root, args.archive, args.results)
