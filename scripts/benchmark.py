"""Measure configured joint model capacity, one-step cost, and peak GPU memory."""

import argparse
import json
import random
import statistics
import time
import torch
import yaml
from unifiedgroupgen.cli import build_models, root_path, artifact_path
from unifiedgroupgen.data import read_jsonl
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.training import joint_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--records')
    parser.add_argument('--descriptor', help='Explicit synthetic descriptor for multiplicity/memory stress tests')
    parser.add_argument('--atoms', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--steps', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--shuffle-records', action='store_true')
    parser.add_argument('--no-checkpoint', action='store_true')
    parser.add_argument('--edge-chunk-size', type=int)
    parser.add_argument('--profile', help='Optional profiler report from a separate, untimed step')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.steps < 1 or args.warmup < 1 or args.batch_size < 1:
        parser.error('steps, warmup and batch-size must be positive')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    config = yaml.safe_load(root_path(args.config).read_text(encoding='utf-8'))
    proposal, flow = build_models(config, args.device)
    if args.no_checkpoint:
        flow.checkpoint_layers = False
    if args.edge_chunk_size is not None:
        if args.edge_chunk_size < 1:
            parser.error('edge-chunk-size must be positive')
        flow.edge_chunk_size = args.edge_chunk_size
    if args.records:
        records = read_jsonl(root_path(args.records))
        if args.shuffle_records:
            random.Random(args.seed).shuffle(records)
        records = records[:args.batch_size]
    else:
        # P1 is the largest independent-coordinate case; this is a resource test.
        descriptor = (Descriptor.from_dict(json.loads(root_path(args.descriptor).read_text())) if args.descriptor
                      else Descriptor('space', 1, tuple(OrbitSpec('a', 14) for _ in range(args.atoms))))
        compiled = compile_descriptor(descriptor)
        if len(descriptor.orbits) > config['max_orbits'] or compiled.num_atoms > config['max_atoms']:
            raise ValueError('Synthetic P1 atom count exceeds descriptor orbit budget')
        records = [{'id': str(i), 'descriptor': descriptor.to_dict(), 'properties': {},
                    'state': compiled.prior().tolist()} for i in range(args.batch_size)]
    optimizer = torch.optim.AdamW(list(proposal.parameters())+list(flow.parameters()), lr=1e-4)
    cuda = args.device.startswith('cuda')
    precision = config.get('precision', 'fp32')
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=cuda and precision == 'fp16')
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=cuda and precision == 'fp16')
    elapsed = []
    peak = 0
    def training_step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(torch.device(args.device).type, dtype=dtype, enabled=cuda and precision != 'fp32'):
            losses = joint_loss(proposal, flow, records, 0.5, 0.25)
            loss = losses['descriptor'].mean()+losses['flow'].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Benchmark loss is nonfinite')
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(list(proposal.parameters())+list(flow.parameters()), 10, error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
    for step in range(args.steps+args.warmup):
        if cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        training_step()
        if cuda:
            torch.cuda.synchronize()
            peak = max(peak, torch.cuda.max_memory_allocated())
        duration = time.perf_counter()-started
        if step >= args.warmup:
            elapsed.append(duration)
        elif step == 0:
            cold_seconds = duration
    report = {'config': args.config, 'proposal_parameters': sum(p.numel() for p in proposal.parameters()),
              'flow_parameters': sum(p.numel() for p in flow.parameters()), 'batch_size': len(records),
              'atoms': [compile_descriptor(Descriptor.from_dict(r['descriptor'])).num_atoms for r in records],
              'seconds_per_step': sum(elapsed)/len(elapsed), 'peak_cuda_bytes': peak,
              'median_seconds_per_step': statistics.median(elapsed), 'step_seconds': elapsed,
              'records_per_second': len(records)/statistics.median(elapsed),
              'checkpoint_layers': flow.checkpoint_layers, 'edge_chunk_size': flow.edge_chunk_size,
              'seed': args.seed, 'threads': args.threads, 'warmup': args.warmup,
              'record_ids': [r['id'] for r in records],
              'cold_step_seconds': cold_seconds,
              'device': torch.cuda.get_device_name() if cuda else 'cpu', 'precision': precision,
              'torch': torch.__version__, 'scope': 'Short training-step benchmark; not convergence or material validation'}
    report['total_parameters'] = report['proposal_parameters']+report['flow_parameters']
    artifact_path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report))
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as profile:
            training_step()
            if cuda:
                torch.cuda.synchronize()
        rows = [{'name': e.key, 'count': e.count, 'cpu_self_us': e.self_cpu_time_total,
                 'device_total_us': (e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total)}
                for e in profile.key_averages()]
        artifact_path(args.profile).write_text(json.dumps({'events': rows, 'scope': 'Untimed profiled step; not a runtime estimate'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
