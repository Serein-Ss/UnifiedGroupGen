"""Pack loss-study evidence for local analysis, excluding weights and raw datasets."""

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import zipfile

ROOT = Path(os.environ.get('UGG_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()


def file_hash(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def archive_results(study, destination, allow_incomplete=False):
    study, destination = study.resolve(), destination.resolve()
    status = json.loads((study/'status.json').read_text(encoding='utf-8'))
    if status.get('state') != 'complete' and not allow_incomplete:
        raise ValueError('Study is not complete; use --allow-incomplete only for a diagnostic snapshot')
    if destination.is_relative_to(study):
        raise ValueError('Archive must be outside the study directory')
    if destination.exists():
        raise FileExistsError(destination)
    suffixes = {'.json', '.jsonl', '.csv', '.png', '.cif', '.md', '.log', '.txt', '.yaml', '.yml', '.py'}
    files = [p for p in sorted(study.rglob('*')) if p.is_file()
             and p.suffix.lower() in suffixes
             and not {'initial', 'reference_metrics', '__pycache__'}.intersection(p.relative_to(study).parts)]
    if any(not p.resolve().is_relative_to(study) for p in files):
        raise ValueError('Study contains a file pointing outside its directory')
    manifest = {'study': study.name, 'state': status.get('state'), 'complete': status.get('state') == 'complete',
                'scope': 'Loss/gradient histories, paired traces, metrics, PNG, generated CIF and frozen code; no weights/raw data',
                'excluded': ['model/initial weights', 'raw training data', 'reference fingerprint caches'],
                'files': [{'path': p.relative_to(study).as_posix(), 'bytes': p.stat().st_size,
                           'sha256': file_hash(p)} for p in files]}
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in files:
            archive.write(path, study.name+'/'+path.relative_to(study).as_posix())
        archive.writestr('audit_manifest.json', json.dumps(manifest, indent=2))
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError('Archive CRC validation failed')
    digest = file_hash(destination)
    destination.with_suffix(destination.suffix+'.sha256').write_text(
        digest+'  '+destination.name+'\n', encoding='utf-8')
    return {'archive': str(destination), 'bytes': destination.stat().st_size,
            'sha256': digest, 'files': len(files), 'complete': manifest['complete']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', required=True)
    parser.add_argument('--output')
    parser.add_argument('--allow-incomplete', action='store_true')
    args = parser.parse_args()
    study = (ROOT/args.study).resolve()
    output = (ROOT/args.output).resolve() if args.output else ROOT/'artifacts/exports'/(
        study.name+'_audit_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'.zip')
    print(json.dumps(archive_results(study, output, args.allow_incomplete), indent=2), flush=True)


if __name__ == '__main__':
    main()
