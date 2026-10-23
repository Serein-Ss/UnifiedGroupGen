"""Run exactly one five-epoch timing pilot, separate from all main15 output directories."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def summarize(output, process_seconds):
    history = [json.loads(s) for s in (output/'model.history.jsonl').read_text(encoding='utf-8').splitlines() if s]
    if [r['epoch'] for r in history] != list(range(5)):
        raise ValueError('Pilot requires exactly five completed epochs starting from initialization')
    if [r['epoch']+1 for r in history if r['validated']] != [5]:
        raise ValueError('Pilot validation schedule must be epochs 1-4 skipped, epoch 5 validated')
    keys = ('training_seconds', 'validation_seconds', 'checkpoint_seconds', 'other_seconds', 'wall_seconds')
    rows = [{ 'epoch': r['epoch']+1, **{k: r[k] for k in keys}, 'val_loss': r['val_loss'],
              'allocated_GiB': r['peak_cuda_bytes']/2**30,
              'reserved_GiB': r['peak_cuda_reserved_bytes']/2**30} for r in history]
    with (output/'timing.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = {'epochs': 5, 'rows': rows, 'totals': {k: sum(r[k] for r in rows) for k in keys},
              'process_seconds': process_seconds,
              'process_overhead_outside_epoch_timing_seconds': process_seconds-sum(r['wall_seconds'] for r in rows),
              'warm_training_median_epochs_2_to_4': statistics.median(r['training_seconds'] for r in rows[1:4]),
              'validation_epochs': [5], 'scope': 'One timing pilot, not main15 training or a quality/convergence result'}
    (output/'timing.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print('\nEpoch | train_s | validation_s | save_s | other_s | wall_s | reserved_GiB', flush=True)
    for r in rows:
        print(f'{r["epoch"]:5d} | '+ ' | '.join(f'{r[k]:.3f}' for k in (*keys[:-1], 'wall_seconds', 'reserved_GiB')), flush=True)
    print(json.dumps({k: v for k, v in report.items() if k != 'rows'}, indent=2), flush=True)
    return report


def command(config, output, args):
    return [sys.executable, '-u', '-m', 'unifiedgroupgen.cli', '--seed', '42', '--threads', str(args.threads),
            'train', '--config', str(config), '--device', args.device, '--epochs', '5',
            '--validation-interval', '5', '--output', str(output/'model.pt'),
            '--checkpoint' if args.checkpoint else '--no-checkpoint', '--edge-chunk-size', str(args.edge_chunk_size)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['mp20_all', 'c2db51_all', 'joint_all'], required=True)
    parser.add_argument('--threads', type=int, required=True)
    parser.add_argument('--checkpoint', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--edge-chunk-size', type=int, default=32768)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan', action='store_true')
    args = parser.parse_args()
    if min(args.threads, args.edge_chunk_size) < 1:
        parser.error('threads and edge-chunk-size must be positive')
    output = (ROOT/args.output).resolve()
    if not output.is_relative_to((ROOT/'artifacts/runtime_pilots').resolve()):
        parser.error('Pilot output must be under artifacts/runtime_pilots, separate from main15 and paired runs')
    config = ROOT/f'configs/main15/{args.model}.yaml'
    job = command(config, output, args)
    plan = {'state': 'planned', 'model': args.model, 'epochs': 5, 'threads': args.threads,
            'config_sha256': hashlib.sha256(config.read_bytes()).hexdigest(), 'command': job}
    if args.plan:
        print(json.dumps(plan, indent=2))
        return
    output.mkdir(parents=True, exist_ok=False)
    environment = {**os.environ, 'MKL_THREADING_LAYER': 'SEQUENTIAL', 'OMP_NUM_THREADS': str(args.threads),
                   'MKL_NUM_THREADS': str(args.threads), 'PYTHONPATH': str(ROOT/'src'), 'PYTHONIOENCODING': 'utf-8'}
    plan.update(state='running', pid=os.getpid())
    def save():
        (output/'pilot.json').write_text(json.dumps(plan, indent=2), encoding='utf-8')
    save()
    print('[START] one model, exactly five epochs: '+args.model, flush=True)
    started = time.perf_counter()
    try:
        subprocess.run(job, cwd=ROOT, env=environment, check=True)
        summarize(output, time.perf_counter()-started)
        plan.update(state='complete')
        print('[COMPLETE] '+str(output), flush=True)
    except BaseException as error:
        plan.update(state='failed', error=str(error))
        raise
    finally:
        save()


if __name__ == '__main__':
    main()
