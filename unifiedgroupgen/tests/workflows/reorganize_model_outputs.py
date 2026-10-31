"""One-time migration of existing main15 results to checkpoint-local task directories."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
from zipfile import ZipFile, ZIP_DEFLATED


TRAINING_FILES = {'model.best.pt', 'model.pt', 'model.history.jsonl', 'model.status.json', 'train.log'}


def digest(path):
    checksum = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8*1024*1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def inside(path, parent):
    path = Path(path).resolve()
    if not path.is_relative_to(parent.resolve()) or path == parent.resolve():
        raise ValueError('Migration path escapes its verified root: '+str(path))
    return path


def pack_ledger(source, target):
    """Embed exact exported CIF bytes, preserving indices, failures and original CIF hashes."""
    target.parent.mkdir(parents=True, exist_ok=True)
    counts = {'records': 0, 'generated': 0, 'failed': 0, 'inline_cifs': 0}
    seen = set()
    with source.open(encoding='utf-8') as incoming, target.open('x', encoding='utf-8', newline='\n') as outgoing:
        for line in incoming:
            if not line.strip():
                continue
            row = json.loads(line)
            if row['index'] in seen:
                raise ValueError('Duplicate index in '+str(source))
            seen.add(row['index'])
            if 'cif' not in row and row.get('status') == 'generated':
                cif = Path(row.get('cif_path', f"sample_{row['index']:05d}.cif"))
                cif = inside(cif if cif.is_absolute() else source.parent/cif, source.parent)
                payload = cif.read_bytes()
                if row.get('cif_sha256') and hashlib.sha256(payload).hexdigest() != row['cif_sha256']:
                    raise ValueError('Original CIF hash differs: '+str(cif))
                row['cif'] = payload.decode('utf-8')
            if 'cif' in row:
                checksum = hashlib.sha256(row['cif'].encode('utf-8')).hexdigest()
                if row.get('cif_sha256') and checksum != row['cif_sha256']:
                    raise ValueError('Inline CIF hash differs: '+str(source))
                row['cif_sha256'] = checksum
                counts['inline_cifs'] += 1
                row.pop('cif_path', None)
            counts['records'] += 1
            counts['generated' if row.get('status') == 'generated' else 'failed'] += 1
            outgoing.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    return {**counts, 'source_sha256': digest(source), 'samples_sha256': digest(target)}


def migrate(root, stamp, run=False):
    root = Path(root).resolve()
    output = root/'outputs'
    inventory = list(csv.DictReader((output/'model_inventory.csv').open(encoding='utf-8-sig')))
    runs = {row['name']: inside(root/row['directory'], output) for row in inventory}
    known = {*runs, 'baselines_256_local_seed142', 'main15_evaluation', 'validation',
             'model_inventory.csv', 'model_inventory.json', 'evaluation_locations.json', 'README.md'}
    unknown = {p.name for p in output.iterdir()}-known
    if unknown:
        raise ValueError('Unclassified outputs must be reviewed first: '+str(sorted(unknown)))
    checkpoints = {str(p): digest(p) for directory in runs.values()
                   for p in (directory/'model.pt', directory/'model.best.pt')}
    for row in inventory:
        directory = runs[row['name']]
        for filename in TRAINING_FILES:
            if not (directory/filename).is_file():
                raise ValueError('Missing training result '+str(directory/filename))
        for filename, key in [('model.best.pt', 'best_checkpoint_sha256'), ('model.pt', 'last_checkpoint_sha256')]:
            if row.get(key) and checkpoints[str(directory/filename)] != row[key]:
                raise ValueError('Checkpoint differs from inventory: '+row['name'])
    tasks, components = [], []
    baseline = output/'baselines_256_local_seed142'
    if baseline.exists():
        for ledger in baseline.rglob('samples.jsonl'):
            relative = ledger.relative_to(baseline)
            model, dataset, mode = relative.parts[1:4]
            name = f'baseline256_{dataset}_{mode}_seed142'
            tasks.append((ledger.parent, runs[model]/'generate'/name, runs[model]/'evaluate'/name))
        for summary in baseline.glob('*/*/test/summary.json'):
            components.append((summary.parent, runs[summary.parent.parent.name]/'evaluate'/'baseline256_components_seed142'))
    campaign = output/'main15_evaluation'
    if campaign.exists():
        for ledger in campaign.rglob('samples.jsonl'):
            parts = ledger.relative_to(campaign).parts
            model, task = parts[1:3]
            name = f'diagnostic256_{task}_seed142'
            tasks.append((ledger.parent, runs[model]/'generate'/name, runs[model]/'evaluate'/name))
        for summary in campaign.glob('*/*/components/summary.json'):
            components.append((summary.parent, runs[summary.parent.parent.name]/'evaluate'/'diagnostic256_components_seed142'))
    receipt = {'models': len(runs), 'checkpoints_before': checkpoints, 'tasks': len(tasks),
               'component_checks': len(components), 'state': 'plan', 'path_map': {}}
    if not run:
        return receipt
    audit = inside(root/'intermediate'/'output_layout'/stamp, root)
    audit.mkdir(parents=True, exist_ok=False)
    mappings, results = receipt['path_map'], []
    def label(path):
        return path.relative_to(root).as_posix()
    def copy_tree(source, target):
        if target.exists():
            raise FileExistsError(str(target))
        shutil.copytree(source, target)
        for old in source.rglob('*'):
            if old.is_file() and digest(old) != digest(target/old.relative_to(source)):
                raise ValueError('Copied file differs: '+str(old))
    for source, generated, evaluated in tasks:
        if generated.exists() or evaluated.exists():
            raise FileExistsError('Task result already exists: '+str(generated))
        counts = pack_ledger(source/'samples.jsonl', generated/'samples.jsonl')
        for name in ('summary.json', 'sampling.status.json'):
            if (source/name).exists():
                shutil.copy2(source/name, generated/name)
        if (source/'metrics').is_dir():
            copy_tree(source/'metrics', evaluated)
            mappings[label(source/'metrics')] = label(evaluated)
            (evaluated/'storage_migration.json').write_text(json.dumps({**counts,
                'checkpoint_sha256': checkpoints[str(generated.parent.parent/'model.best.pt')],
                'scope': 'Storage migration only; metric values unchanged, no recomputation. Original receipts are historical.'}, indent=2))
        mappings[label(source)] = label(generated)
        results.append({'source': label(source), 'generate': label(generated), **counts})
    for source, target in components:
        target.mkdir(parents=True, exist_ok=False)
        for name in ('summary.json', 'records.jsonl', 'status.json'):
            if (source/name).exists():
                shutil.copy2(source/name, target/name)
        mappings[label(source)] = label(target)
    for source in baseline.glob('*/reference_cache'):
        copy_tree(source, audit/'reference_cache')
        mappings[label(source)] = label(audit/'reference_cache')
    for name in ('validation', 'model_inventory.csv', 'model_inventory.json', 'evaluation_locations.json', 'README.md'):
        source = output/name
        if source.is_dir():
            copy_tree(source, audit/name)
        elif source.is_file():
            shutil.copy2(source, audit/name)
        mappings[label(source)] = label(audit/name)
    # Validate all newly packed records and unchanged checkpoints before removing any old file.
    for task in results:
        ledger = root/task['generate']/'samples.jsonl'
        rows = [json.loads(line) for line in ledger.read_text(encoding='utf-8').splitlines()]
        if len(rows) != task['records'] or sum(r.get('status') == 'generated' for r in rows) != task['generated']:
            raise ValueError('Candidate denominator changed')
        for row in rows:
            if 'cif' in row and hashlib.sha256(row['cif'].encode('utf-8')).hexdigest() != row['cif_sha256']:
                raise ValueError('Packed CIF changed')
    if any(digest(path) != checksum for path, checksum in checkpoints.items()):
        raise ValueError('A checkpoint changed during migration')
    # Keep old control receipts outside results, in one compressed audit file, not thousands of CIFs.
    obsolete = [p for p in output.iterdir() if p.name not in runs]
    obsolete += [p for directory in runs.values() for p in directory.iterdir()
                 if p.name not in TRAINING_FILES | {'generate', 'evaluate'}]
    files = []
    for item in obsolete:
        inside(item, output)
        files += [item] if item.is_file() else [p for p in item.rglob('*') if p.is_file()]
    with ZipFile(audit/'previous_control_records.zip', 'x', compression=ZIP_DEFLATED) as archive:
        for file in files:
            inside(file, output)
            if file.suffix != '.cif' and 'reference_cache' not in file.parts:
                archive.write(file, file.relative_to(output).as_posix())
    with ZipFile(audit/'previous_control_records.zip') as archive:
        if archive.testzip() is not None:
            raise ValueError('Control-record archive verification failed')
    for file in files:
        inside(file, output).unlink()
    for item in obsolete:
        if item.is_dir():
            for folder in sorted((p for p in item.rglob('*') if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
                inside(folder, output).rmdir()
            item.rmdir()
    receipt.update(state='verified', results=results, checkpoints_after={p: digest(p) for p in checkpoints},
                   directory_dates='Historical model dates are import times, not training start times',
                   scope='Storage only; all candidate slots and CIF content preserved; no model, training or metric changes')
    (audit/'migration.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2), encoding='utf-8')
    return receipt


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--stamp', required=True)
    parser.add_argument('--run', action='store_true')
    args = parser.parse_args()
    result = migrate(args.root, args.stamp, args.run)
    print(json.dumps({k: v for k, v in result.items() if k not in ('checkpoints_before', 'checkpoints_after', 'path_map', 'results')}, ensure_ascii=False))
