"""Test and sample one main15 checkpoint across its trained source domains."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.cli import root_path, artifact_path
from unifiedgroupgen.data import read_jsonl
from unifiedgroupgen.training import file_hash


def jobs(config, checkpoint, count, conditional_count, steps, guidance, seed):
    result = [('test', ['scripts/evaluate_checkpoint.py', '--checkpoint', checkpoint,
                       '--records', *config['test_files']])]
    datasets = config.get('dataset_domains') or {config['dataset']: None}
    properties = [name for name, spec in config.get('properties', {}).items() if not spec.get('context')]
    train = [r for p in config['train_files'] for r in read_jsonl(root_path(p))] if properties else []
    for dataset, domain in datasets.items():
        kind = 'layer' if dataset == 'c2db_51' else 'space'
        context = {'source_domain': domain} if domain is not None else {}
        def add(name, target, scale, size):
            command = ['-m', 'unifiedgroupgen.cli', '--seed', str(seed), 'sample',
                '--checkpoint', checkpoint, '--kind', kind, '--properties', json.dumps({**context, **target}),
                '--guidance', str(scale), '--count', str(size), '--steps', str(steps)]
            result.append((dataset+'_'+name, command))
        add('unconditional', {}, 0, count)
        for name in properties:
            values = [r['properties'][name] for r in train if r['dataset'] == dataset
                      and r['properties'].get(name) is not None and np.isfinite(r['properties'][name])]
            if not values:
                continue
            for index, value in enumerate(sorted(set(float(x) for x in np.quantile(values, [.1, .5, .9])))):
                add(f'{name}_{index}', {name: value}, guidance, conditional_count)
        if len(properties) > 1:
            complete = [r for r in train if r['dataset'] == dataset
                        and all(r['properties'].get(p) is not None and np.isfinite(r['properties'][p]) for p in properties)]
            if complete:
                # Choose actual TRAIN property tuples, never unrelated marginal targets or TEST labels.
                complete.sort(key=lambda r: (r['properties'][properties[0]], r['id']))
                for index, fraction in enumerate((.1, .5, .9)):
                    row = complete[round(fraction*(len(complete)-1))]
                    add(f'joint_{index}', {p: row['properties'][p] for p in properties}, guidance, conditional_count)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--count', type=int, default=4096)
    parser.add_argument('--conditional-count', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--guidance', type=float, default=2)
    parser.add_argument('--seed', type=int, default=142)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--standard-metrics', action='store_true')
    args = parser.parse_args()
    if min(args.count, args.conditional_count, args.steps) < 1:
        parser.error('Sampling budgets must be positive')
    checkpoint = root_path(args.checkpoint)
    config = torch.load(checkpoint, map_location='cpu', weights_only=False)['config']
    name = config['experiment_id']
    queue = jobs(config, str(checkpoint), args.count, args.conditional_count, args.steps, args.guidance, args.seed)
    if args.plan:
        print(json.dumps(queue, indent=2))
        return
    output = artifact_path(args.output)
    if output.exists():
        raise ValueError('Use a new evaluation output directory')
    output.mkdir(parents=True)
    report = {'checkpoint_sha256': file_hash(checkpoint), 'experiment': name,
              'state': 'running', 'jobs': [], 'property_accuracy': None}
    try:
        for tag, command in queue:
            target = output/tag
            destination = target/'result.json' if tag == 'test' else target
            command += ['--device', args.device, '--output', str(destination)]
            entry = {'name': tag, 'command': command, 'state': 'running'}
            report['jobs'].append(entry)
            print('[START] '+tag, flush=True)
            with (output/f'{tag}.log').open('w', encoding='utf-8') as stream:
                subprocess.run([sys.executable, '-u', *command], cwd=ROOT, check=True, stdout=stream,
                               stderr=subprocess.STDOUT, env={**os.environ, 'PYTHONPATH': str(ROOT/'src')})
            entry['state'] = 'complete'
            if tag != 'test':
                entry['summary'] = json.loads((target/'summary.json').read_text(encoding='utf-8'))
                entry['samples_sha256'] = file_hash(target/'samples.jsonl')
                if args.standard_metrics:
                    dataset = 'c2db_51' if tag.startswith('c2db_51_') else 'mp_20'
                    subprocess.run([sys.executable, 'scripts/evaluate_generation.py', '--samples',
                        str(target/'samples.jsonl'), '--dataset', dataset, '--reference-config',
                        f'configs/main15/{name}.yaml', '--output', str(target/'metrics'),
                        '--figures', str(target/'figures'), '--reidentify'], cwd=ROOT, check=True,
                        env={**os.environ, 'PYTHONPATH': str(ROOT/'src')})
            print('[DONE] '+tag, flush=True)
        report['state'] = 'complete'
    except BaseException as error:
        report.update(state='failed', error=str(error))
        raise
    finally:
        (output/'summary.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
