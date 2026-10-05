"""Run the full MP20 and MPTS52 large models sequentially, with held-out evaluation."""

import os

# Set before importing NumPy/PyTorch: this environment ships two OpenMP runtimes.
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import argparse
import datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
os.environ['PYTHONPATH'] = str(ROOT/'src')


def timestamp():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--edge-chunk-size', type=int)
    parser.add_argument('--validation-interval', type=int)
    args = parser.parse_args()
    if args.edge_chunk_size is not None and args.edge_chunk_size < 1:
        parser.error('edge-chunk-size must be positive')
    if args.validation_interval is not None and args.validation_interval < 1:
        parser.error('validation-interval must be positive')
    directory = ROOT/'artifacts/runs/local_large_seed42'
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/'runner.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    status_path = directory/'runner.json'
    status = {'pid': os.getpid(), 'started': timestamp(), 'state': 'running', 'jobs': []}
    def update():
        temporary = status_path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temporary.replace(status_path)
    def run(name, command):
        job = {'name': name, 'started': timestamp(), 'command': command, 'state': 'running'}
        status['jobs'].append(job)
        update()
        with (directory/f'{name}.stdout.log').open('a', encoding='utf-8') as out, \
             (directory/f'{name}.stderr.log').open('a', encoding='utf-8') as err:
            process = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT, stdout=out, stderr=err)
            job['pid'] = process.pid
            update()
            code = process.wait()
        job.update({'exit_code': code, 'finished': timestamp(), 'state': 'complete' if code == 0 else 'failed'})
        update()
        if code:
            raise RuntimeError(f'{name} exited with {code}; inspect its stderr log')
    try:
        for dataset, prepared, target in (
            ('mp20', 'artifacts/full/mp_20', {'formation_energy': -1.0, 'band_gap': 1.0}),
            ('mpts52', 'artifacts/full_tol0p01/mpts_52', {'formation_energy': -1.0, 'energy_above_hull': 0.0}),
        ):
            model_path = f'artifacts/runs/local_large_seed42/{dataset}/model.pt'
            command = ['-m', 'unifiedgroupgen.cli', '--seed', '42', 'train',
                       '--config', f'configs/{dataset}_local.yaml', '--device', 'cuda', '--output', model_path]
            if args.checkpoint is not None:
                command += ['--checkpoint' if args.checkpoint else '--no-checkpoint']
            if args.edge_chunk_size is not None:
                command += ['--edge-chunk-size', str(args.edge_chunk_size)]
            if args.validation_interval is not None:
                command += ['--validation-interval', str(args.validation_interval)]
            if args.resume and (ROOT/model_path).exists():
                saved_status = json.loads((ROOT/model_path).with_suffix('.status.json').read_text())
                if saved_status['state'] != 'complete':
                    command += ['--resume', model_path]
                else:
                    command = None
            if command:
                run(f'{dataset}_train', command)
            best = str(Path(model_path).with_suffix('.best.pt'))
            run(f'{dataset}_test', ['scripts/evaluate_checkpoint.py', '--checkpoint', best,
                                  '--records', f'{prepared}/test.jsonl', '--device', 'cuda',
                                  '--output', f'artifacts/runs/local_large_seed42/{dataset}/test.json'])
            for name, properties, guidance in [('unconditional', {}, '0'), ('conditional', target, '2')]:
                run(f'{dataset}_{name}', ['-m', 'unifiedgroupgen.cli', '--seed', '142', 'sample',
                                         '--checkpoint', best, '--device', 'cuda', '--count', '64',
                                         '--steps', '64', '--properties', json.dumps(properties),
                                         '--guidance', guidance,
                                         '--output', f'artifacts/runs/local_large_seed42/{dataset}/{name}'])
        status['state'] = 'complete'
        status['scope'] = 'Two complete 3D reference training runs, held-out loss evaluation and 64 samples per mode; shared 2D/3D training is tracked by full_pipeline/pipeline.json'
    except BaseException:
        status['state'] = 'failed'
        status['error'] = traceback.format_exc()
        raise
    finally:
        status['finished'] = timestamp()
        update()
        lock.unlink()


if __name__ == '__main__':
    main()
