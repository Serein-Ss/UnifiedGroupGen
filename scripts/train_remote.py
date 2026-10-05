"""Portable full-data reference and shared-model training, evaluation and generation."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ['PYTHONPATH'] = str(ROOT / 'src')


def jobs(stage, resume, checkpoint=None, edge_chunk_size=None):
    commands = []
    runtime = ([] if checkpoint is None else ['--checkpoint' if checkpoint else '--no-checkpoint'])
    if edge_chunk_size is not None:
        runtime += ['--edge-chunk-size', str(edge_chunk_size)]
    if stage in ('all', 'references'):
        commands.append(('references', ['scripts/train_local.py', *(['--resume'] if resume else []), *runtime]))
    if stage in ('all', 'main'):
        directory = 'artifacts/runs/joint_space_layer_large_seed42'
        model = f'{directory}/model.pt'
        train = ['-m', 'unifiedgroupgen.cli', '--seed', '42', 'train', '--config',
                 'configs/joint_space_layer_large.yaml', '--device', 'cuda', '--output', model]
        if resume and (ROOT / model).exists():
            train += ['--resume', model]
        train += runtime
        commands.append(('joint_train', train))
        best = f'{directory}/model.best.pt'
        tests = [f'artifacts/full_joint_sourcegrouped/{dataset}/test.jsonl'
                 for dataset in ('mp_20', 'mpts_52', 'c2db_51')]
        commands.append(('joint_test', ['scripts/evaluate_checkpoint.py', '--checkpoint', best,
                                      '--device', 'cuda', '--records', *tests, '--output', f'{directory}/test.json']))
        for dataset, domain, kind, target in (
            ('mp_20', 0, 'space', {'formation_energy': -1.0, 'band_gap': 1.0}),
            ('mpts_52', 1, 'space', {'formation_energy': -1.0, 'energy_above_hull': 0.0}),
            ('c2db_51', 2, 'layer', {'formation_energy': -1.0, 'band_gap': 1.0}),
        ):
            for mode in ('unconditional', 'conditional'):
                properties = {'source_domain': domain, **(target if mode == 'conditional' else {})}
                commands.append((f'{dataset}_{mode}', ['-m', 'unifiedgroupgen.cli', '--seed', '142', 'sample',
                                  '--checkpoint', best, '--device', 'cuda', '--kind', kind,
                                  '--properties', json.dumps(properties), '--guidance', '2' if mode == 'conditional' else '0',
                                  '--count', '64', '--steps', '64', '--output', f'{directory}/{dataset}/{mode}']))
    return commands


def completed_joint_training():
    """Allow completed-run resumes only with unchanged audited configuration and data."""
    import torch
    import yaml
    from unifiedgroupgen.preflight import check_training_gate
    config = yaml.safe_load((ROOT / 'configs/joint_space_layer_large.yaml').read_text(encoding='utf-8'))
    directory = ROOT / 'artifacts/runs/joint_space_layer_large_seed42'
    state = directory / 'model.status.json'
    if not state.exists() or json.loads(state.read_text())['state'] != 'complete':
        return False
    check_training_gate(config, [ROOT / p for p in config['train_files']], [ROOT / p for p in config['val_files']], ROOT)
    saved = torch.load(directory / 'model.pt', map_location='cpu', weights_only=False)
    for key, value in config.items():
        if key == 'properties':
            matches = all(saved['config'][key][name].get(k) == v
                          for name, specification in value.items() for k, v in specification.items())
        else:
            matches = saved['config'].get(key) == value
        if not matches:
            raise ValueError(f'Completed checkpoint configuration changed: {key}')
    audit = json.loads((ROOT / config['data_audit']).read_text())
    for split in ('train', 'val'):
        if saved[f'{split}_split_sha256'] != [item['sha256'] for item in audit['inputs'][split]]:
            raise ValueError('Completed checkpoint data changed')
    if not (directory / 'model.best.pt').exists():
        raise ValueError('Completed run is missing its best checkpoint')
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'main', 'references'), default='all')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan', action='store_true', help='Print commands without launching training')
    parser.add_argument('--checkpoint', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--edge-chunk-size', type=int)
    args = parser.parse_args()
    if args.edge_chunk_size is not None and args.edge_chunk_size < 1:
        parser.error('edge-chunk-size must be positive')
    commands = jobs(args.stage, args.resume, args.checkpoint, args.edge_chunk_size)
    if args.plan:
        print(json.dumps([{'name': name, 'command': [sys.executable, '-u', *command]} for name, command in commands], indent=2))
        return
    if not (ROOT / 'artifacts/dataset_import.json').exists():
        raise ValueError('Run python scripts/unpack_training_data.py before remote training')
    folder = ROOT / 'artifacts/runs/remote_pipeline'
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder / 'pipeline.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode())
    os.close(descriptor)
    status = {'pid': os.getpid(), 'state': 'running', 'stage': args.stage, 'jobs': [],
              'started': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def update():
        temporary = folder / 'pipeline.json.tmp'
        temporary.write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
        temporary.replace(folder / 'pipeline.json')
    try:
        for name, command in commands:
            if args.resume and name == 'joint_train' and completed_joint_training():
                status['jobs'].append({'name': name, 'state': 'complete', 'reused': True})
                update()
                continue
            print(f'[START] {name}', flush=True)
            job = {'name': name, 'state': 'running', 'command': command}
            status['jobs'].append(job)
            with (folder / f'{name}.stdout.log').open('a', encoding='utf-8') as out, \
                 (folder / f'{name}.stderr.log').open('a', encoding='utf-8') as err:
                process = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT, stdout=out, stderr=err)
                job['pid'] = process.pid
                update()
                code = process.wait()
            job.update(state='complete' if code == 0 else 'failed', exit_code=code)
            update()
            if code:
                raise RuntimeError(f'{name} exited with {code}; inspect {folder / (name + ".stderr.log")}')
            print(f'[DONE] {name}', flush=True)
        status['state'] = 'complete'
        status['scope'] = 'Full-data training, held-out loss and generation attempts completed; physical properties/stability require separate evaluation.'
    except BaseException:
        status.update(state='failed', error=traceback.format_exc())
        raise
    finally:
        status['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        update()
        lock.unlink()


if __name__ == '__main__':
    main()
