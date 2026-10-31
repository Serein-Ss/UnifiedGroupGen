"""Plan or serially train the approved configurations with Lightning and dated runs."""
import argparse
import json
import os
import subprocess
import sys

from unifiedgroupgen.tests.legacy.experiments import configurations, RUNTIME_BY_DATASET
from unifiedgroupgen.tests.legacy.paths import project_root, resolve_path, module_command
from unifiedgroupgen.common.configuration import RunSettings
from unifiedgroupgen.common.source import file_hash


def plan_jobs(root, args):
    configs = configurations(root)
    jobs = []
    for name in args.models:
        runtime = {**RUNTIME_BY_DATASET[name.split('_')[0]], 'coordinate_length_scale': 3.,
                   'checkpoint_layers': False, 'edge_chunk_size': 32768, 'validation_interval': 5}
        config = root/f'unifiedgroupgen/tests/legacy/configs/main15/{name}.yaml'
        command = [sys.executable, '-u', '-m', 'unifiedgroupgen.tests.legacy.cli', '--seed', str(args.seed),
                   '--threads', str(args.threads), 'train', '--trainer', 'lightning',
                   '--config', str(config), '--experiment', name, '--device', args.device,
                   '--batch-size', str(runtime['batch_size']), '--edge-budget', str(runtime['edge_budget']),
                   '--coordinate-length-scale', '3', '--no-checkpoint', '--edge-chunk-size', '32768',
                   '--validation-interval', '5']
        if args.epochs is not None:
            command += ['--epochs', str(args.epochs)]
        if args.wandb_mode:
            command += ['--wandb-mode', args.wandb_mode]
        jobs.append({'name': name, 'command': command, 'runtime_options': runtime,
                     'config_sha256': file_hash(config),
                     'target_properties': [k for k, v in configs[name]['properties'].items() if not v.get('context')]})
    return jobs


def main():
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=list(configurations(root)), default=list(configurations(root)))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--run', action='store_true')
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if args.threads < 1 or len(args.models) != len(set(args.models)):
        parser.error('Positive threads and unique model names are required')
    jobs = plan_jobs(root, args)
    settings = RunSettings.load(root, args.wandb_mode)
    if not args.run or args.plan:
        print(json.dumps({'jobs_root': str(settings.jobs), 'wandb_mode': settings.wandb_mode,
                          'directory_date': 'Actual start time allocated by each child; not plan time',
                          'jobs': jobs, 'started': False}, indent=2))
        return
    settings.jobs.mkdir(parents=True, exist_ok=True)
    lock = settings.jobs/f'lightning_schedule_seed{args.seed}.lock'
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    try:
        execute(root, args, jobs, settings)
    finally:
        lock.unlink()


def execute(root, args, jobs, settings):
    subprocess.run(module_command([sys.executable, 'scripts/prepare_main15.py', '--verify-only']), cwd=root, check=True)
    for job in jobs:
        if args.resume:
            candidates = []
            for path in (settings.jobs/job['name']).glob('*/run.json'):
                meta = json.loads(path.read_text(encoding='utf-8'))
                if (meta.get('trainer') == 'pytorch-lightning' and meta.get('seed') == args.seed
                        and meta.get('source_config_sha256') == job['config_sha256']
                        and meta.get('runtime_options') == job['runtime_options']
                        and (path.parent/'last.ckpt').is_file()):
                    candidates.append(path)
            if candidates:
                chosen = max(candidates, key=lambda p: p.parent.name).parent
                status = chosen/'model.status.json'
                if status.exists() and json.loads(status.read_text(encoding='utf-8')).get('state') == 'complete':
                    print('[COMPLETE] '+job['name']+' '+str(chosen), flush=True)
                    continue
                job['command'] += ['--resume', str(chosen/'last.ckpt')]
        print('[START] '+job['name'], flush=True)
        subprocess.run(job['command'], cwd=root, check=True)
        print('[DONE] '+job['name'], flush=True)


if __name__ == '__main__':
    main()
