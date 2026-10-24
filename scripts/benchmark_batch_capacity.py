"""Sweep real-data training throughput and stress memory; never launch formal training."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

import yaml

ROOT = Path(__file__).resolve().parents[1]
SIZES = {'mp20_all': (64, 96, 128, 160, 192, 256),
         'c2db51_all': (64, 128, 256, 512, 768, 1024),
         'joint_all': (64, 96, 128, 160, 192, 256)}


def edge_limited(report):
    underfilled = sum(size < report['batch_size'] for size in report['batch_sizes'])
    return underfilled > 1 or any(
        size < report['batch_size'] and edges >= .9*report['edge_budget']
        for size, edges in zip(report['batch_sizes'], report['batch_edges']))


def command(config, pool, output, size, budget, mode):
    job = [sys.executable, '-u', str(ROOT/'scripts/benchmark.py'), '--config', str(config),
           '--device', 'cuda', '--seed', '42', '--threads', '4', '--batch-size', str(size),
           '--edge-budget', str(budget), '--no-checkpoint', '--edge-chunk-size', '32768',
           '--warmup', '6', '--output', str(output)]
    if mode == 'random':
        job += ['--records', str(pool), '--selection', 'random', '--full-pass']
    else:
        job += ['--selection', 'largest', '--batches', '6', '--steps', '18']
    return job


def benchmark(config, pool, folder, size, budget, mode, environment):
    name = f'{config.stem}_b{size}_e{budget}_{mode}'
    output, log = folder/(name+'.json'), folder/(name+'.log')
    job = command(config, pool, output, size, budget, mode)
    print('[START] '+name, flush=True)
    started = time.perf_counter()
    with log.open('w', encoding='utf-8') as stream:
        result = subprocess.run(job, cwd=ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT)
    receipt = {'command': job, 'exit_code': result.returncode, 'process_seconds': time.perf_counter()-started}
    (folder/(name+'.command.json')).write_text(json.dumps(receipt, indent=2), encoding='utf-8')
    if result.returncode:
        tail = log.read_text(encoding='utf-8', errors='replace')
        if 'out of memory' not in tail.lower():
            raise RuntimeError('Benchmark failed for a reason other than OOM; inspect '+str(log))
        print('[OOM] '+name, flush=True)
        return None
    report = json.loads(output.read_text(encoding='utf-8'))
    print(f"[DONE] {name}: {report['records_per_second']:.2f} samples/s; "
          f"allocated {report['peak_cuda_bytes']/2**30:.2f} GiB; "
          f"reserved {report['peak_cuda_reserved_bytes']/2**30:.2f} GiB", flush=True)
    return report


def summarize(folder, rows):
    (folder/'summary.json').write_text(json.dumps({
        'rows': rows, 'formal_training_started': False,
        'scope': 'Fixed-pool complete-pass throughput plus full-TRAIN high-edge stress. '
                 'Candidates require five-epoch checks; no convergence or quality claim.'}, indent=2), encoding='utf-8')
    with (folder/'summary.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True)
    parser.add_argument('--pool-size', type=int, default=4096)
    parser.add_argument('--wait-study', default='artifacts/loss_balance_4090_seed42')
    parser.add_argument('--run', action='store_true', help='Without this flag, print the plan only')
    args = parser.parse_args()
    if args.pool_size < 1:
        parser.error('pool-size must be positive')
    folder = (ROOT/args.output).resolve()
    if not folder.is_relative_to((ROOT/'artifacts/benchmarks').resolve()):
        parser.error('Output must be under artifacts/benchmarks')
    if not args.run:
        print(json.dumps({'batch_sizes': SIZES, 'initial_edge_budget': 200000,
            'conditional_edge_budget': 400000, 'pool_size': args.pool_size,
            'threads': 4, 'activation_checkpointing': False, 'edge_chunk_size': 32768,
            'formal_training_started': False}, indent=2))
        return
    state = json.loads((ROOT/args.wait_study/'status.json').read_text(encoding='utf-8'))
    if state.get('state') != 'complete':
        parser.error('The previous evaluation is not complete; no benchmark was started')
    gpu = subprocess.run(['nvidia-smi', '-i', '0', '--query-compute-apps=pid', '--format=csv,noheader'],
                         capture_output=True, text=True, check=True)
    if any(line.strip().isdigit() for line in gpu.stdout.splitlines()):
        parser.error('GPU 0 has an active compute process; no process was stopped')
    folder.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, 'MKL_THREADING_LAYER': 'SEQUENTIAL', 'OMP_NUM_THREADS': '4',
        'MKL_NUM_THREADS': '4', 'PYTHONPATH': str(ROOT/'src'), 'PYTHONIOENCODING': 'utf-8',
        'CUDA_VISIBLE_DEVICES': '0'}
    rows = []
    for model, sizes in SIZES.items():
        config = ROOT/f'configs/main15/{model}.yaml'
        settings = yaml.safe_load(config.read_text(encoding='utf-8'))
        records = [json.loads(line) for path in settings['train_files']
                   for line in (ROOT/path).read_text(encoding='utf-8').splitlines() if line.strip()]
        selected = random.Random(20261024).sample(records, min(len(records), args.pool_size))
        pool = folder/(model+'_timing_pool.jsonl')
        pool.write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in selected), encoding='utf-8')
        pool_hash = hashlib.sha256(pool.read_bytes()).hexdigest()
        for size in sizes:
            budgets = [200000]
            for budget in budgets:
                random_report = benchmark(config, pool, folder, size, budget, 'random', environment)
                stress = (benchmark(config, pool, folder, size, budget, 'largest', environment)
                          if random_report else None)
                if random_report and random_report['measured_records'] != len(selected):
                    raise ValueError('Incomplete timing-pool pass; throughput comparison rejected')
                limited = any(edge_limited(r) for r in (random_report, stress) if r)
                if budget == 200000 and limited and stress is not None:
                    budgets.append(400000)
                row = {'model': model, 'batch_size': size, 'edge_budget': budget,
                    'status': 'OK' if random_report and stress else 'OOM',
                    'samples_per_second': random_report['records_per_second'] if random_report else None,
                    'seconds_per_step': random_report['seconds_per_step'] if random_report else None,
                    'mean_actual_batch': sum(random_report['batch_sizes'])/len(random_report['batch_sizes']) if random_report else None,
                    'max_stress_edges': max(stress['batch_edges']) if stress else None,
                    'edge_limited': limited, 'peak_allocated_GiB': max(
                        (r['peak_cuda_bytes']/2**30 for r in (random_report, stress) if r), default=None),
                    'peak_reserved_GiB': max(
                        (r['peak_cuda_reserved_bytes']/2**30 for r in (random_report, stress) if r), default=None),
                    'pool_records': len(selected), 'pool_sha256': pool_hash}
                rows.append(row)
                summarize(folder, rows)
    # The sampled training inputs remain local; only compact diagnostics are packed.
    archive = folder.with_suffix('.zip')
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(folder.iterdir()):
            if path.is_file() and not path.name.endswith('_timing_pool.jsonl'):
                bundle.write(path, arcname=folder.name+'/'+path.name)
    print('[COMPLETE] benchmarks only: '+str(folder), flush=True)
    print('[AUDIT PACKAGE] '+str(archive), flush=True)


if __name__ == '__main__':
    main()
