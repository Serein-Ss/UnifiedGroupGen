"""Import a finished server snapshot and bind reused results to exact source bytes.

Never imports server source over local code. Previous presentation files are kept
in a ZIP outside assets; the current manifest is the authoritative result source.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
from zipfile import ZipFile, ZIP_DEFLATED

ROOT = Path(__file__).resolve().parents[1]


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--snapshot', default='20261019_completed245')
    args = parser.parse_args()
    target = (ROOT/'artifacts/server_snapshots'/args.snapshot).resolve()
    if target.exists():
        raise ValueError('Snapshot already exists; use its verified manifest or a new snapshot name')
    if not target.is_relative_to((ROOT/'artifacts/server_snapshots').resolve()):
        raise ValueError('Snapshot must remain inside artifacts/server_snapshots')
    run = 'artifacts/runs/joint_space_layer_large_seed42/'
    evaluation = 'artifacts/evaluations/joint_best_seed142/'
    target.mkdir(parents=True)
    with ZipFile(args.archive) as z:
        matches = [n for n in z.namelist() if n.endswith(run+'model.status.json')]
        if len(matches) != 1:
            raise ValueError('Expected exactly one joint training run')
        prefix = matches[0][:-len(run+'model.status.json')]
        status = json.loads(z.read(prefix+run+'model.status.json'))
        if status['state'] != 'complete':
            raise ValueError('Training snapshot is not complete')
        pipeline = json.loads(z.read(prefix+evaluation+'pipeline.json'))
        if pipeline['state'] != 'complete':
            raise ValueError('Evaluation snapshot is not complete')
        imported = []
        for member in z.infolist():
            if member.is_dir() or not member.filename.startswith(prefix):
                continue
            rel = member.filename[len(prefix):]
            if not (rel.startswith(run) or rel.startswith(evaluation) or rel == 'artifacts/runs/joint_resume.log'):
                continue
            parts = PurePosixPath(rel)
            if parts.is_absolute() or '..' in parts.parts:
                raise ValueError('Unsafe archive path')
            dest = target.joinpath(*parts.parts)
            dest.parent.mkdir(parents=True, exist_ok=True)
            with z.open(member) as source, dest.open('wb') as output:
                shutil.copyfileobj(source, output)
            imported.append(rel)
        jobs = []
        for name, job in pipeline['jobs'].items():
            if job['state'] != 'complete':
                raise ValueError('Incomplete evaluation job: '+name)
            for filename, checksum in job['output_sha256'].items():
                candidate = (target/evaluation/job['directory']/filename).resolve()
                if not candidate.is_relative_to((target/evaluation).resolve()) or digest(candidate) != checksum:
                    raise ValueError('Evaluation artifact mismatch: '+str(candidate))
            jobs.append({'job': name, 'verified_outputs': len(job['output_sha256'])})
        best_sha = digest(target/run/'model.best.pt')
        if best_sha != pipeline['identity']['checkpoint_sha256']:
            raise ValueError('Best checkpoint does not match evaluation')
        # Bind unchanged dataset statistics to every prepared train/val/test file.
        data_checks = []
        for dataset, expected_test in zip(('mp_20', 'mpts_52', 'c2db_51'), pipeline['identity']['test_sha256']):
            for split in ('train', 'val', 'test'):
                rel = f'artifacts/full_joint_sourcegrouped/{dataset}/{split}.jsonl'
                sha = hashlib.sha256(z.read(prefix+rel)).hexdigest()
                if sha != digest(ROOT/rel) or (split == 'test' and sha != expected_test):
                    raise ValueError('Prepared dataset changed; recompute statistics: '+rel)
                data_checks.append({'path': rel, 'sha256': sha})
        code = {name: hashlib.sha256(z.read(prefix+name)).hexdigest() == sha
                for name, sha in pipeline['identity']['code_sha256'].items()}
        # Current server code may have advanced since generation; report it rather
        # than treating a newer source checkout as evidence of a new sampling run.
        imported_path = ROOT/'artifacts/imported_server/joint_best_seed142'
        existing = []
        for name, job in pipeline['jobs'].items():
            for filename, sha in job['output_sha256'].items():
                path = imported_path/job['directory']/filename
                existing.append(path.exists() and digest(path) == sha)
        history = [json.loads(line) for line in (target/run/'model.history.jsonl').read_text().splitlines() if line.strip()]
        history = sorted({row['epoch']: row for row in history}.values(), key=lambda row: row['epoch'])
        if [r['epoch'] for r in history] != list(range(status['completed_epochs'])):
            raise ValueError('History is incomplete or not contiguous')
        validation = [r for r in history if r.get('val_loss') is not None]
        best = min(validation, key=lambda r: r['val_loss'])
        if best['val_loss'] != status['best_val_loss'] or best['epoch'] != pipeline['checkpoint_epoch']:
            raise ValueError('Best validation and evaluated checkpoint disagree')
        save(target/'history.json', history)
        manifest = {'archive_name': args.archive.name, 'archive_sha256': digest(args.archive),
            'snapshot': target.relative_to(ROOT).as_posix(), 'training': status,
            'history': (target/'history.json').relative_to(ROOT).as_posix(),
            'history_sha256': digest(target/'history.json'), 'validation_points': len(validation),
            'best_epoch_1based': best['epoch']+1, 'best_checkpoint_sha256': best_sha,
            'last_checkpoint_sha256': digest(target/run/'model.pt'),
            'evaluation': (target/evaluation).relative_to(ROOT).as_posix(),
            'evaluation_jobs_verified': jobs, 'evaluation_identical_to_previous_import': all(existing),
            'archived_code_matches_generation': code, 'dataset_splits_verified': data_checks,
            'statistics_policy': 'Training history refreshed. Generation/test metrics remain valid only for the byte-identical best checkpoint and verified output/reference files. No new sampling or property/stability certification.',
            'files_imported': len(imported)}
        save(target/'verification.json', manifest)
        save(ROOT/'assets/results/current_snapshot.json', manifest)
    print(json.dumps({k: manifest[k] for k in ('snapshot', 'validation_points', 'best_epoch_1based', 'evaluation_identical_to_previous_import', 'files_imported')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
