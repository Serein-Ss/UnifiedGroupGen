"""Package the verified full datasets without checkpoints or credentials."""

import argparse
import gzip
import hashlib
import json
from pathlib import Path
import tarfile

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-data', type=Path, default=ROOT.parent / 'data')
    parser.add_argument('--verified', action='store_true', help='Package the new certified revision as an additive bundle')
    args = parser.parse_args()
    files = {}
    folders = tuple(f'artifacts/full_verified_v1/{d}' for d in ('mp_20', 'mpts_52', 'c2db_51')) if args.verified else (
        'artifacts/full/mp_20', 'artifacts/full_tol0p01/mpts_52', 'artifacts/full_official_tol0p1_v7/c2db_51')
    for folder in folders:
        for split in ('train', 'val', 'test'):
            for suffix in ('', '.report.json', '.manifest.json', '.rejected.jsonl'):
                name = f'{folder}/{split}.jsonl{suffix}'
                files[name] = ROOT / name
        for path in (ROOT / folder).glob('audit*.json'):
            files[path.relative_to(ROOT).as_posix()] = path
    joint_root = 'artifacts/joint_verified_v1' if args.verified else 'artifacts/full_joint_sourcegrouped'
    for path in (ROOT / joint_root).rglob('*'):
        if path.is_file() and path.suffix in ('.json', '.jsonl'):
            files[path.relative_to(ROOT).as_posix()] = path
    if args.verified:
        for path in (ROOT/'artifacts/full_verified_v1').glob('*'):
            if path.is_file():files[path.relative_to(ROOT).as_posix()] = path
    else:
        for split in ('train', 'val', 'test'):
            name = f'artifacts/source_recovery/c2db_official/{split}.csv'
            files[name] = ROOT / name
        for name in ('artifacts/source_recovery/c2db_official/recovery.json',
                     'artifacts/source_recovery/c2db_alternatives/mlip_arena_c2db.db'):
            files[name] = ROOT / name
        for dataset in ('mp_20', 'mpts_52', 'c2db_51'):
            for split in ('train', 'val', 'test'):
                files[f'raw_data/{dataset}/{split}.csv'] = args.source_data / dataset / f'{split}.csv'
    destination = ROOT / 'datasets' / 'source_integrity_v1' if args.verified else ROOT/'datasets'
    if args.verified and (destination/'manifest.json').exists():
        raise ValueError('Certified bundle already exists; preserve it or choose a new revision')
    destination.mkdir(exist_ok=True)
    archive = destination / 'full_training.tar.gz'
    inventory = []
    with archive.open('wb') as raw, gzip.GzipFile(fileobj=raw, mode='wb', mtime=0) as compressed:
        with tarfile.open(fileobj=compressed, mode='w|') as bundle:
            for name, path in sorted(files.items()):
                item = bundle.gettarinfo(str(path), arcname=name)
                item.mtime = item.uid = item.gid = 0
                item.uname = item.gname = ''
                with path.open('rb') as stream:
                    bundle.addfile(item, stream)
                inventory.append({'path': name, 'bytes': path.stat().st_size, 'sha256': sha256(path)})
                print(name, flush=True)
    parts = []
    with archive.open('rb') as stream:
        index = 0
        while block := stream.read(48 * 1024 * 1024):
            part = destination / f'full_training.tar.gz.part{index:03d}'
            part.write_bytes(block)
            parts.append({'name': part.name, 'bytes': len(block), 'sha256': sha256(part)})
            index += 1
    source_records = {d: sum(json.loads((ROOT/f'artifacts/full_verified_v1/{d}/{split}.jsonl.report.json').read_text())['accepted'] for split in ('train','val','test')) for d in ('mp_20','mpts_52','c2db_51')} if args.verified else {'mp_20': 45229, 'mpts_52': 40476, 'c2db_51': 16905}
    joint_counts = json.loads((ROOT/joint_root/'manifest.json').read_text())['report']['counts'] if args.verified else {'train': 82185, 'val': 10190, 'test': 10235}
    manifest = {'version': 2 if args.verified else 1, 'original_root': str(ROOT), 'original_data_root': str(args.source_data.resolve()),
                'archive_sha256': sha256(archive), 'parts': parts, 'files': inventory,
                'source_records': source_records, 'total_source_records': sum(source_records.values()),
                'joint_split_records': joint_counts,
                'scope': 'Certified additive revision; requires original bundle for raw data/recovery evidence; 2 quarantined inputs excluded.' if args.verified else 'Full prepared sources, grouped joint splits, raw CSVs and original C2DB recovery evidence.'}
    (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    archive.unlink()  # The complete archive is reconstructed from its versioned parts.
    print(json.dumps({'parts': len(parts), 'compressed_bytes': sum(p['bytes'] for p in parts),
                      'files': len(inventory), 'records': sum(source_records.values())}), flush=True)


if __name__ == '__main__':
    main()
