"""Evaluate an existing shared checkpoint without starting or resuming training."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import argparse
import csv
import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
os.environ['PYTHONPATH'] = str(ROOT/'src')

TARGETS = (
    ('mp_20', 0, 'space', {'formation_energy': -1.0, 'band_gap': 1.0}),
    ('mpts_52', 1, 'space', {'formation_energy': -1.0, 'energy_above_hull': 0.0}),
    ('c2db_51', 2, 'layer', {'formation_energy': -1.0, 'band_gap': 1.0}),
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def save_json(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def plan(args, test_files):
    jobs = [('test', ['scripts/evaluate_checkpoint.py', '--checkpoint', args.checkpoint,
                     '--device', args.device, '--records', *test_files])]
    if getattr(args, 'skip_test', False):
        jobs = []
    if getattr(args, 'campaign', None):
        campaign = json.loads(Path(args.campaign).read_text(encoding='utf-8'))
        if campaign.get('schema') != 1 or not campaign.get('jobs'):
            raise ValueError('Campaign requires schema=1 and a nonempty jobs list')
        names = set()
        for item in campaign['jobs']:
            dataset, domain, kind, _ = next(t for t in TARGETS if t[0] == item['dataset'])
            name = item['name']
            if not name.startswith(dataset+'_') or not name.replace('_', '').isalnum() or name in names:
                raise ValueError('Campaign names must be unique safe names prefixed by their dataset')
            names.add(name)
            if item['count'] < 1 or item['steps'] < 1:
                raise ValueError('Campaign count and steps must be positive')
            properties = {'source_domain': domain, **item.get('properties', {})}
            if properties['source_domain'] != domain:
                raise ValueError('Campaign source domain does not match dataset')
            command = ['-m', 'unifiedgroupgen.cli', '--seed', str(item['seed']), 'sample',
                '--checkpoint', args.checkpoint, '--device', args.device, '--kind', kind,
                '--properties', json.dumps(properties), '--guidance', str(item['guidance']),
                '--count', str(item['count']), '--steps', str(item['steps'])]
            for field in ('descriptor', 'observations', 'composition'):
                if item.get(field):
                    command += ['--'+field, str(item[field])]
            if item.get('prior'):
                command.append('--prior')
            jobs.append((name, command))
        return jobs
    for dataset, domain, kind, target in TARGETS:
        for mode in ('unconditional', 'conditional'):
            properties = {'source_domain': domain, **(target if mode == 'conditional' else {})}
            jobs.append((f'{dataset}_{mode}', ['-m', 'unifiedgroupgen.cli', '--seed', str(args.seed),
                'sample', '--checkpoint', args.checkpoint, '--device', args.device, '--kind', kind,
                '--properties', json.dumps(properties), '--guidance', str(args.guidance if mode == 'conditional' else 0),
                '--count', str(args.count), '--steps', str(args.steps)]))
    return jobs


def completed_outputs(directory, name, count):
    """Validate completion before allowing a job to be reused on resume."""
    if name == 'test':
        report = json.loads((directory/'result.json').read_text(encoding='utf-8'))
        if report['requested_records'] <= 0:
            raise ValueError('Empty held-out test evaluation')
        return {'result.json': digest(directory/'result.json')}
    rows = [json.loads(line) for line in (directory/'samples.jsonl').read_text(encoding='utf-8').splitlines() if line]
    if len(rows) != count or [r['index'] for r in rows] != list(range(count)):
        raise ValueError(f'Incomplete generation attempt accounting: {name}')
    if any(r['status'] not in ('generated', 'failed') for r in rows):
        raise ValueError(f'Unknown generation status: {name}')
    from unifiedgroupgen.evaluation import summarize_attempts
    summary = summarize_attempts(rows)
    if json.loads((directory/'summary.json').read_text()) != summary:
        raise ValueError(f'Generation summary does not match attempt records: {name}')
    paths = [directory/'samples.jsonl', directory/'summary.json']
    paths += [directory/f"sample_{r['index']:05d}.cif" for r in rows if r['status'] == 'generated']
    return {path.name: digest(path) for path in paths}


def summarize(output, state):
    from unifiedgroupgen.evaluation import summarize_attempts
    report = {'checkpoint_sha256': state['identity']['checkpoint_sha256'],
              'checkpoint_epoch': state['checkpoint_epoch'], 'generation': {},
              'scope': 'Unconditional means property-unconditional within a fixed source domain and kind. '
                       'Targets are demonstration conditions; these outputs do not measure property accuracy, '
                       'chemical validity, uniqueness, novelty or physical stability.'}
    table = []
    for name, job in state['jobs'].items():
        directory = output/job['directory']
        if name == 'test':
            report['test'] = json.loads((directory/'result.json').read_text())
            continue
        rows = [json.loads(line) for line in (directory/'samples.jsonl').read_text().splitlines() if line]
        result = summarize_attempts(rows)
        result['directory'] = job['directory']
        result['properties_requested'] = json.loads(job['command'][job['command'].index('--properties')+1])
        report['generation'][name] = result
        table.append({'job': name, **{key: result[key] for key in
            ('attempted', 'generated', 'failed', 'exact_group_success', 'exact_group_unidentified_generated',
             'geometry_screen_success', 'geometry_and_exact_group_success')},
            'distance_valid': result['counts']['distance_valid'],
            'no_coincident_orbits': result['counts']['no_coincident_orbits']})
    save_json(output/'summary.json', report)
    with (output/'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(table[0]))
        writer.writeheader()
        writer.writerows(table)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default='artifacts/runs/joint_space_layer_large_seed42/model.best.pt')
    parser.add_argument('--output', default='artifacts/evaluations/joint_best_seed142')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--count', type=int, default=256, help='Attempts per dataset and mode; six jobs in total')
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--seed', type=int, default=142)
    parser.add_argument('--guidance', type=float, default=2.0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--campaign', help='JSON jobs produced by prepare_paper_experiments.py')
    parser.add_argument('--skip-test', action='store_true', help='Generation only; held-out input audit still applies')
    parser.add_argument('--plan', action='store_true', help='Print the seven jobs without loading weights or launching work')
    args = parser.parse_args()
    if args.campaign:
        args.campaign = str((ROOT/args.campaign).resolve())
    if args.count < 1 or args.steps < 1:
        parser.error('count and steps must be positive')
    if args.plan:
        tests = [f'artifacts/full_joint_sourcegrouped/{dataset}/test.jsonl' for dataset, *_ in TARGETS]
        print(json.dumps(plan(args, tests), indent=2))
        return
    import torch
    from unifiedgroupgen.cli import root_path, artifact_path
    from evaluate_checkpoint import check_heldout_inputs
    checkpoint_path = root_path(args.checkpoint)
    checkpoint_hash = digest(checkpoint_path)
    saved = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    config = saved['config']
    expected_domains = {dataset: domain for dataset, domain, *_ in TARGETS}
    domains = config.get('dataset_domains', {})
    if not domains or any(expected_domains.get(key) != value for key, value in domains.items()):
        raise ValueError('Checkpoint calculation domains do not match the evaluated sources')
    if domains != expected_domains and not args.campaign:
        raise ValueError('A separate-source matched model requires an explicit campaign for its trained domain')
    paths = [root_path(path) for path in config['test_files']]
    audit = check_heldout_inputs(saved, paths)
    if audit is None:
        raise ValueError('Shared evaluation requires an audited checkpoint and complete held-out files')
    if args.campaign:
        campaign = json.loads(Path(args.campaign).read_text(encoding='utf-8'))
        hashes = campaign.get('train_sha256', {})
        if not hashes or list(hashes.values()) != saved['train_split_sha256']:
            raise ValueError('Campaign targets/cases were not derived from this checkpoint training split')
        if any(digest(root_path(path)) != checksum for path, checksum in hashes.items()):
            raise ValueError('Campaign training files changed')
    checkpoint_epoch = saved['epoch']
    del saved
    commands = plan(args, [str(path) for path in paths])
    if any(name != 'test' and not any(name.startswith(dataset+'_') for dataset in domains)
           for name, _ in commands):
        raise ValueError('Campaign requests a source domain outside this checkpoint training data')
    code_paths = sorted((ROOT/'src/unifiedgroupgen').glob('*.py')) + [Path(__file__), ROOT/'scripts/evaluate_checkpoint.py']
    identity = {'checkpoint_sha256': checkpoint_hash,
                'test_sha256': [digest(path) for path in paths],
                'code_sha256': {str(path.relative_to(ROOT)): digest(path) for path in code_paths},
                'commands': [[name, command] for name, command in commands],
                'versions': {name: importlib.metadata.version(name) for name in ('torch', 'numpy', 'pyxtal', 'spglib', 'pymatgen')}}
    auxiliary = {Path(args.campaign)} if args.campaign else set()
    for _, command in commands:
        for flag in ('--descriptor', '--observations'):
            if flag in command:
                auxiliary.add((ROOT/command[command.index(flag)+1]).resolve())
    identity['auxiliary_sha256'] = {str(path): digest(path) for path in sorted(auxiliary)}
    output = artifact_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    status_path = output/'pipeline.json'
    if status_path.exists():
        state = json.loads(status_path.read_text(encoding='utf-8'))
        if not args.resume:
            raise ValueError('Output already has a run; use --resume or a new --output')
        if state['identity'] != identity:
            raise ValueError('Checkpoint, data, code, environment or evaluation settings changed; use a new --output')
    else:
        if any(output.iterdir()):
            raise ValueError('Output must be empty for a new evaluation')
        state = {'identity': identity, 'checkpoint_epoch': checkpoint_epoch, 'heldout_audit': audit, 'jobs': {}}
    lock = output/'pipeline.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    state.update(state='running', pid=os.getpid(), updated=datetime.datetime.now(datetime.timezone.utc).isoformat())
    state.pop('error', None)
    save_json(status_path, state)
    try:
        for name, command in commands:
            previous = state['jobs'].get(name)
            if previous and previous['state'] == 'complete':
                count = int(command[command.index('--count')+1]) if name != 'test' else args.count
                actual = completed_outputs(output/previous['directory'], name, count)
                if actual != previous['output_sha256']:
                    raise ValueError(f'Completed outputs changed for {name}; use a new output directory')
                print(f'[REUSE] {name}', flush=True)
                continue
            # Keep interrupted outputs. A retry starts a fresh, separately recorded attempt.
            parent = output/name
            parent.mkdir(exist_ok=True)
            attempt = 1
            while (parent/f'attempt_{attempt:03d}').exists():
                attempt += 1
            directory = parent/f'attempt_{attempt:03d}'
            directory.mkdir()
            destination = directory/'result.json' if name == 'test' else directory
            command = [*command, '--output', str(destination)]
            job = {'state': 'running', 'directory': str(directory.relative_to(output)), 'command': command}
            state['jobs'][name] = job
            state['phase'] = name
            save_json(status_path, state)
            print(f'[START] {name}: {directory}', flush=True)
            with (directory/'stdout.log').open('w', encoding='utf-8') as out, (directory/'stderr.log').open('w', encoding='utf-8') as err:
                process = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT, stdout=out, stderr=err)
                job['pid'] = process.pid
                save_json(status_path, state)
                try:
                    code = process.wait()
                except BaseException:
                    process.terminate()
                    process.wait()
                    raise
            if code:
                job.update(state='failed', exit_code=code)
                raise RuntimeError(f'{name} exited with {code}; inspect {directory}/stderr.log')
            count = int(command[command.index('--count')+1]) if name != 'test' else args.count
            job['output_sha256'] = completed_outputs(directory, name, count)
            job.update(state='complete', exit_code=0)
            save_json(status_path, state)
            print(f'[DONE] {name}', flush=True)
        if digest(checkpoint_path) != checkpoint_hash:
            raise ValueError('Checkpoint changed during evaluation; do not use these results')
        summarize(output, state)
        state.update(state='complete', phase='complete')
    except BaseException:
        state.update(state='failed', error=traceback.format_exc())
        raise
    finally:
        state['updated'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        save_json(status_path, state)
        lock.unlink()
    print(f'[COMPLETE] {output}/summary.json', flush=True)


if __name__ == '__main__':
    main()
