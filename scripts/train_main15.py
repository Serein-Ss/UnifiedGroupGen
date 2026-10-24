"""Sequential, locked runner for the 15 approved property models; never starts control fine-tuning."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import json
import hashlib
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.main_experiments import configurations, RUNTIME_BY_DATASET


def main():
    configs = configurations(ROOT)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(configs), default=list(configs))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--coordinate-length-scale', type=float, default=3.0)
    parser.add_argument('--checkpoint', action=argparse.BooleanOptionalAction, default=False,
                        help='Activation recomputation only; model checkpoints are always saved')
    parser.add_argument('--edge-chunk-size', type=int, default=32768)
    args = parser.parse_args()
    if len(set(args.models)) != len(args.models):
        parser.error('Duplicate model names')
    if args.threads < 1:
        parser.error('threads must be positive')
    if not math.isfinite(args.coordinate_length_scale) or args.coordinate_length_scale <= 0:
        parser.error('coordinate-length-scale must be finite and positive')
    if args.edge_chunk_size < 1:
        parser.error('edge-chunk-size must be positive')
    runtime = {name: {**RUNTIME_BY_DATASET[name.split('_')[0]],
                     'threads': args.threads,
                     'coordinate_length_scale': args.coordinate_length_scale,
                     'checkpoint_layers': args.checkpoint,
                     'edge_chunk_size': args.edge_chunk_size,
                     'validation_interval': 5} for name in configs}
    jobs = []
    for name in args.models:
        output = f'artifacts/runs/main15_seed{args.seed}/{name}/model.pt'
        options = runtime[name]
        command = ['-m', 'unifiedgroupgen.cli', '--seed', str(args.seed),
                   '--threads', str(args.threads), 'train',
                   '--config', f'configs/main15/{name}.yaml', '--device', args.device,
                   '--output', output, '--validation-interval', '5',
                   '--batch-size', str(options['batch_size']),
                   '--edge-budget', str(options['edge_budget']),
                   '--coordinate-length-scale', str(args.coordinate_length_scale)]
        if args.resume and (ROOT/output).exists():
            command += ['--resume', output]
        command.append('--checkpoint' if args.checkpoint else '--no-checkpoint')
        command += ['--edge-chunk-size', str(args.edge_chunk_size)]
        jobs.append({'name': name, 'command': command, 'output': output,
                     'runtime_options': options,
                     'target_properties': [key for key, value in configs[name]['properties'].items()
                                           if not value.get('context', False)],
                     'cond_dropout': configs[name]['cond_dropout']})
    if args.plan:
        print(json.dumps(jobs, indent=2))
        return
    subprocess.run([sys.executable, 'scripts/prepare_main15.py', '--verify-only'], cwd=ROOT, check=True)
    directory = ROOT/f'artifacts/runs/main15_seed{args.seed}'
    directory.mkdir(parents=True, exist_ok=True)
    lock = directory/'runner.lock'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    status = {'state': 'running', 'pid': os.getpid(), 'jobs': jobs}
    def save():
        temporary = directory/'runner.json.tmp'
        temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temporary.replace(directory/'runner.json')
    try:
        identity = {'seed': args.seed, 'configs': {name: hashlib.sha256(
            (ROOT/f'configs/main15/{name}.yaml').read_bytes()).hexdigest() for name in configs},
                    'runtime_options': runtime}
        identity_path = directory/'identity.json'
        if identity_path.exists() and json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('Run directory belongs to different experiment configurations or runtime options')
        if not identity_path.exists() and any((directory/name/'model.pt').exists() for name in configs):
            raise ValueError('Existing model checkpoints have no verified runtime identity')
        identity_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
        for job in jobs:
            path = ROOT/job['output']
            if args.resume and path.with_suffix('.status.json').exists():
                prior = json.loads(path.with_suffix('.status.json').read_text(encoding='utf-8'))
                if prior.get('state') == 'complete':
                    if not path.exists():
                        raise ValueError('Completed model status has no checkpoint: '+str(path))
                    job['state'] = 'complete'; save()
                    continue
            job['state'] = 'running'; save()
            print('[START] '+job['name'], flush=True)
            with (directory/f'{job["name"]}.log').open('a', encoding='utf-8') as stream:
                env = {**os.environ, 'PYTHONPATH': str(ROOT/'src'),
                       'MKL_THREADING_LAYER': 'SEQUENTIAL', 'OMP_NUM_THREADS': str(args.threads),
                       'MKL_NUM_THREADS': str(args.threads), 'PYTHONIOENCODING': 'utf-8'}
                process = subprocess.Popen([sys.executable, '-u', *job['command']], cwd=ROOT,
                                           stdout=stream, stderr=subprocess.STDOUT, env=env)
                job['pid'] = process.pid; save()
                job['exit_code'] = process.wait()
            job['state'] = 'complete' if job['exit_code'] == 0 else 'failed'; save()
            if job['exit_code']:
                raise RuntimeError('Training failed: '+job['name'])
            print('[DONE] '+job['name'], flush=True)
        status['state'] = 'complete'
    except BaseException as error:
        status.update(state='failed', error=str(error))
        raise
    finally:
        save()
        lock.unlink()


if __name__ == '__main__':
    main()
