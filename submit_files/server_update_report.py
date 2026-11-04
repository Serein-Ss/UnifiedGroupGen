"""Read code/campaign updates into a portable report; never fetch, commit or run jobs."""
import argparse
from collections import Counter
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import subprocess
import zipfile


def command(args, root):
    try:
        result = subprocess.run(args, cwd=root, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=20,
                                env={**os.environ, 'GIT_OPTIONAL_LOCKS': '0'})
        return {'returncode': result.returncode, 'stdout': result.stdout.strip(),
                'stderr': result.stderr.strip()}
    except (OSError, subprocess.TimeoutExpired) as error:
        return {'returncode': None, 'stdout': '', 'stderr': str(error)}


def collect(root, campaign):
    root, campaign = Path(root).resolve(), Path(campaign).resolve()
    if not campaign.is_relative_to(root) or not campaign.is_file():
        raise ValueError('Choose an existing campaign.json inside the project')
    folder = campaign.parent
    result = {'root': str(root), 'campaign': {}, 'phases': {}, 'evaluations': {},
              'files': {}, 'warnings': []}
    rows = []

    def inspect(path, contents=False):
        path = Path(path).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            result['warnings'].append('Outside report scope: '+str(path))
            return None
        if not path.is_file():
            result['files'][relative] = {'exists': False}
            return None
        stat = path.stat()
        entry = {'exists': True, 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
        data = path.read_bytes() if contents else None
        if data is not None:
            entry['sha256'] = hashlib.sha256(data).hexdigest()
        result['files'][relative] = entry
        return data

    def read(path):
        data = inspect(path, contents=True)
        if data is None:
            result['warnings'].append('Missing report input: '+str(path))
            return {}
        try:
            return json.loads(data)
        except (ValueError, UnicodeError) as error:
            result['warnings'].append('Unreadable JSON: '+str(path)+'; '+str(error))
            return {}

    result['campaign'] = read(campaign)
    if not result['campaign'].get('name') or not result['campaign'].get('revision'):
        raise ValueError('Campaign report requires a name and frozen revision')
    for path in sorted((folder/'state').glob('*.json')):
        result['phases'][path.stem] = read(path)
    for path in sorted((folder/'evaluation').glob('*/pipeline.json')):
        pipeline = read(path)
        entries = pipeline.get('jobs', [])
        expected = len(pipeline.get('identity', {}).get('jobs', []))
        result['evaluations'][path.parent.name] = {
            'reported_state': pipeline.get('state'), 'expected_tasks': expected,
            'recorded_tasks': len(entries),
            'task_states': dict(Counter(e.get('state', 'missing') for e in entries))}
        if pipeline.get('state') == 'complete' and (
                expected != len(entries) or any(e.get('state') != 'complete' for e in entries)):
            result['warnings'].append('Complete pipeline has incomplete task records: '+str(path))
        data = inspect(path.with_name('summary.csv'), contents=True)
        if data is not None:
            rows.extend(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))))
        else:
            result['warnings'].append('Missing metric summary: '+str(path.with_name('summary.csv')))
        for entry in entries:
            generated = entry.get('directory')
            if generated:
                for name in ('sampling.status.json', 'samples.jsonl'):
                    inspect(Path(generated)/name)
            metrics = entry.get('metrics', {}).get('metric_directory') or entry.get('metric_directory')
            if metrics:
                inspect(Path(metrics)/'summary.json', contents=True)
                inspect(Path(metrics)/'samples.csv')
    for phase in result['phases'].values():
        if phase.get('phase') == 'train' and phase.get('model'):
            run = root/'outputs'/phase['model']/result['campaign']['name']
            for name in ('model.best.pt', 'model.pt', 'model.status.json', 'model.history.jsonl'):
                inspect(run/name)
    for path in sorted((folder/'logs').glob('*')):
        if path.suffix in ('.out', '.err', '.log'):
            inspect(path)
    git = ['git', '--no-pager', '-c', 'safe.directory='+str(root)]
    result['git'] = {key: command(git+args, root) for key, args in {
        'head': ['rev-parse', 'HEAD'], 'branch': ['branch', '--show-current'],
        'tracked_changes': ['status', '--short', '--untracked-files=no'],
        'cached_upstream_difference': ['rev-list', '--left-right', '--count', 'HEAD...@{upstream}'],
        'log': ['log', '-20', '--format=%h %aI %s', '--name-status']}.items()}
    result['queue'] = command(['yhq'], root)
    return result, rows


def changes(previous, current):
    before = {p: value for p, value in previous.get('files', {}).items() if value['exists']}
    after = {p: value for p, value in current['files'].items() if value['exists']}
    return {'baseline_available': bool(previous),
            'code_commit_changed': bool(previous) and previous.get('git', {}).get('head') != current['git']['head'],
            'added': sorted(set(after)-set(before)),
            'removed': sorted(set(before)-set(after)),
            'modified': sorted(p for p in before.keys() & after.keys() if before[p] != after[p])}


def export_report(root, campaign, output):
    result, rows = collect(root, campaign)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    latest = output/'latest.json'
    previous = json.loads(latest.read_text(encoding='utf-8')) if latest.exists() else {}
    if previous.get('root') != result['root'] or previous.get('campaign', {}).get('name') != result['campaign'].get('name'):
        previous = {}
    now = datetime.now(timezone(timedelta(hours=8)))
    result.update(captured_at=now.isoformat(), host=socket.gethostname(), changes=changes(previous, result))
    delta = result['changes']
    lines = ['# Server update snapshot', '', 'Captured: '+result['captured_at'],
             'Host: '+result['host'], 'Code HEAD: '+result['git']['head']['stdout'],
             'Campaign frozen revision: '+str(result['campaign'].get('revision')), '',
             'Phase records: '+json.dumps(dict(Counter(p.get('state', 'missing') for p in result['phases'].values()))),
             'Metrics rows: '+str(len(rows)), 'Previous snapshot available: '+str(bool(previous)),
             'Changed code commit: '+str(delta['code_commit_changed']),
             'File entries added/modified/removed: '+str([len(delta[k]) for k in ('added', 'modified', 'removed')]),
             '', '## Tracked uncommitted changes', '```', result['git']['tracked_changes']['stdout'] or '(none)', '```',
             '', '## Queue at capture', '```', result['queue']['stdout'] or result['queue']['stderr'], '```',
             '', '## Result file changes']
    for key in ('added', 'modified', 'removed'):
        lines.extend([key+': '+p for p in delta[key]])
    lines.extend(['', '## Scope',
        'Git upstream counts use the cached local remote ref; no network fetch was performed.',
        'Task states are reported records, not a fresh recomputation or full artifact verification.',
        'Selected campaign/state/pipeline/metric summaries are hashed; large ledgers, checkpoints and logs use size/mtime only.',
        'This is not a recursive inventory of all directories. Missing selected inputs are recorded in snapshot.json.',
        'No .env, environment variables, checkpoint contents, raw datasets or log contents are exported.'])
    table = io.StringIO(newline='')
    writer = csv.DictWriter(table, fieldnames=sorted({key for row in rows for key in row}) or ['model'])
    writer.writeheader(); writer.writerows(rows)
    content = json.dumps(result, ensure_ascii=False, indent=2)+'\n'
    stem = 'server-update-'+now.strftime('%Y%m%d_%H%M%S_%f')
    archive = output/(stem+'.zip')
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr('snapshot.json', content)
        bundle.writestr('report.md', '\n'.join(lines)+'\n')
        bundle.writestr('git_log.txt', result['git']['log']['stdout']+'\n')
        bundle.writestr('metrics.csv', table.getvalue().encode('utf-8-sig'))
    (output/(stem+'.json')).write_text(content, encoding='utf-8')
    temporary = latest.with_suffix('.json.tmp')
    temporary.write_text(content, encoding='utf-8'); temporary.replace(latest)
    checksum = hashlib.sha256(archive.read_bytes()).hexdigest()
    archive.with_suffix('.zip.sha256').write_text(checksum+'  '+archive.name+'\n', encoding='utf-8')
    return archive, result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--campaign', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=Path('intermediate/server_reports'))
    args = parser.parse_args()
    archive, result = export_report(args.root, args.campaign, args.output)
    print('REPORT:', archive)
    print('PHASES:', dict(Counter(p.get('state', 'missing') for p in result['phases'].values())))
    print('CHANGES:', json.dumps(result['changes'], ensure_ascii=False))
