"""Time configured training batches without saving model weights."""

import argparse
import hashlib
import json
import random
import statistics
import time
from types import MethodType
import torch
import yaml
from unifiedgroupgen.cli import build_models, fit_training_config, root_path, artifact_path, ROOT
from unifiedgroupgen.conditioning import attach_dataset_domains
from unifiedgroupgen.data import read_jsonl
from unifiedgroupgen.preflight import check_training_gate
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.training import batch_indices, file_hash, joint_loss, prepare_training_records


def reference_message(self, h, edges, edge_index, degree, chunk_size=8192):
    """Published dense-concatenation arithmetic for paired kernel comparisons."""
    aggregate = torch.zeros_like(h)
    for start in range(0, edges.shape[0], chunk_size):
        source, target = edge_index[:, start:start+chunk_size]
        messages = self.edge(torch.cat((h[source], h[target], edges[start:start+chunk_size]), -1))
        aggregate = aggregate.index_add(0, source, messages.to(aggregate.dtype))
    return self.norm(h+self.node(torch.cat((h, aggregate/degree[:, None].clamp_min(1)), -1)))


def reference_projection(self, compiled, candidates, lattices, states, node_graph):
    """Published padded full-q equations; other training plumbing stays shared."""
    import numpy as np
    max_atoms, max_q = max(c.num_atoms for c in compiled), max(c.coordinate_dof for c in compiled)
    if not max_q:
        return [z.new_zeros(0) for z in states], None
    count = len(compiled)
    basis = np.zeros((count, max_atoms, 3, max_q))
    nodes = np.full((count, max_atoms), len(candidates), dtype=np.int64)
    offset = 0
    for i, c in enumerate(compiled):
        basis[i, :c.num_atoms, :, :c.coordinate_dof] = c.coordinate_basis
        nodes[i, :c.num_atoms] = range(offset, offset+c.num_atoms)
        offset += c.num_atoms
    a = torch.as_tensor(basis, device=states[0].device, dtype=states[0].dtype)
    candidate = torch.cat((candidates.to(a.dtype), a.new_zeros((1, 3))))[torch.as_tensor(nodes, device=a.device)]
    metric = lattices.transpose(1, 2) @ lattices
    gram = torch.einsum('biaq,bac,bicr->bqr', a, metric, a)
    rhs = torch.einsum('biaq,bac,bic->bq', a, metric, candidate)
    unused = torch.arange(max_q, device=a.device)[None] >= torch.tensor(
        [c.coordinate_dof for c in compiled], device=a.device)[:, None]
    solution, info = torch.linalg.solve_ex(gram+torch.diag_embed(unused.to(a.dtype)), rhs[..., None], check_errors=False)
    return [solution[i, :c.coordinate_dof, 0] for i, c in enumerate(compiled)], info


def install_reference(flow):
    for layer in flow.layers:
        layer.forward = MethodType(reference_message, layer)
    flow._project_coordinates = MethodType(reference_projection, flow)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--records', nargs='+', help='Optional timing pool; scalers use configured TRAIN files')
    parser.add_argument('--descriptor', help='Explicit synthetic descriptor for resource stress tests')
    parser.add_argument('--atoms', type=int, help='Explicit synthetic P1 atom count; default uses real TRAIN batches')
    parser.add_argument('--batch-size', type=int, help='Default: configured batch size')
    parser.add_argument('--batches', type=int, default=1)
    parser.add_argument('--selection', choices=['first', 'random', 'largest'], default='first')
    parser.add_argument('--shuffle-records', action='store_true', help='Alias for --selection random')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--checkpoint', action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument('--edge-chunk-size', type=int)
    parser.add_argument('--implementation', choices=['optimized', 'reference'], default='optimized')
    parser.add_argument('--profile', help='Separate untimed CPU/CUDA operator profile')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if min(args.steps, args.warmup, args.batches, args.threads) < 1 or (args.batch_size is not None and args.batch_size < 1):
        parser.error('steps, warmup, batches, threads and batch-size must be positive')
    if args.edge_chunk_size is not None and args.edge_chunk_size < 1:
        parser.error('edge-chunk-size must be positive')
    if args.atoms is not None and args.atoms < 1:
        parser.error('atoms must be positive')
    if args.records and (args.descriptor or args.atoms is not None):
        parser.error('Choose real records or an explicit synthetic resource test')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed); random.seed(args.seed)
    config = yaml.safe_load(root_path(args.config).read_text(encoding='utf-8'))
    batch_size = args.batch_size or config.get('batch_size', 8)
    hashes = {}
    synthetic = args.descriptor is not None or args.atoms is not None
    if synthetic:
        descriptor = (Descriptor.from_dict(json.loads(root_path(args.descriptor).read_text())) if args.descriptor
                      else Descriptor('space', 1, tuple(OrbitSpec('a', 14) for _ in range(args.atoms))))
        c = compile_descriptor(descriptor)
        if len(descriptor.orbits) > config['max_orbits'] or c.num_atoms > config['max_atoms']:
            raise ValueError('Synthetic descriptor exceeds configured budgets')
        dataset = next(iter(config.get('dataset_domains', {})), config.get('dataset', 'synthetic'))
        records = [{'id': str(i), 'dataset': dataset, 'descriptor': descriptor.to_dict(),
                    'properties': {}, 'state': c.prior().tolist()} for i in range(batch_size)]
        records = attach_dataset_domains(records, config)
    else:
        prepared = config.get('prepared_root', f"artifacts/{config['dataset']}")
        train_paths = [root_path(p) for p in config.get('train_files', [f'{prepared}/train.jsonl'])]
        val_paths = [root_path(p) for p in config.get('val_files', [f'{prepared}/val.jsonl'])]
        check_training_gate(config, train_paths, val_paths, ROOT)
        training = attach_dataset_domains([r for p in train_paths for r in read_jsonl(p)], config)
        config = fit_training_config(config, training)
        timing_paths = [root_path(p) for p in args.records] if args.records else train_paths
        records = (attach_dataset_domains([r for p in timing_paths for r in read_jsonl(p)], config)
                   if args.records else training)
        hashes = {'train_split_sha256': [file_hash(p) for p in train_paths],
                  'val_split_sha256': [file_hash(p) for p in val_paths],
                  'timing_pool_sha256': [file_hash(p) for p in timing_paths]}
    if not records:
        raise ValueError('Timing pool is empty')
    records = prepare_training_records(records)
    selection = 'random' if args.shuffle_records else args.selection
    batches = batch_indices(records, batch_size, config.get('edge_budget', 200000), shuffle=selection != 'first')
    if selection == 'largest':
        batches.sort(key=lambda b: sum(records[i]['_training'][1].num_atoms*(records[i]['_training'][1].num_atoms-1)
                                       for i in b), reverse=True)
    batches = [[records[i] for i in b] for b in batches[:args.batches]]
    proposal, flow = build_models(config, args.device)
    if args.implementation == 'reference':
        install_reference(flow)
    if args.checkpoint is not None:
        flow.checkpoint_layers = args.checkpoint
    if args.edge_chunk_size is not None:
        flow.edge_chunk_size = args.edge_chunk_size
    parameters = list(proposal.parameters())+list(flow.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.get('learning_rate', 1e-3))
    cuda = args.device.startswith('cuda')
    requested_precision = config.get('precision', 'fp32')
    precision = requested_precision if cuda else 'fp32'
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support configured bf16')
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=cuda and precision == 'fp16')
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=cuda and precision == 'fp16')
    weights = config.get('loss_weights', {'descriptor': 1.0, 'flow': 1.0})
    def training_step(records):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(torch.device(args.device).type, dtype=dtype, enabled=cuda and precision != 'fp32'):
            losses = joint_loss(proposal, flow, records, config.get('composition_probability', 0.5),
                                config.get('observation_probability', 0.25))
            loss = weights['descriptor']*losses['descriptor'].mean()+weights['flow']*losses['flow'].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Benchmark loss is nonfinite')
        scaler.scale(loss).backward(); scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(parameters, config.get('gradient_clip', 10), error_if_nonfinite=True)
        scaler.step(optimizer); scaler.update()
    elapsed, measured_records, peak, reserved = [], 0, 0, 0
    for step in range(args.steps+args.warmup):
        batch = batches[step % len(batches)]
        if cuda:
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        training_step(batch)
        if cuda:
            torch.cuda.synchronize()
            peak = max(peak, torch.cuda.max_memory_allocated())
            reserved = max(reserved, torch.cuda.max_memory_reserved())
        duration = time.perf_counter()-started
        if step >= args.warmup:
            elapsed.append(duration); measured_records += len(batch)
        elif step == 0:
            cold_seconds = duration
    report = {'config': args.config, 'implementation': args.implementation,
              'proposal_parameters': sum(p.numel() for p in proposal.parameters()),
              'flow_parameters': sum(p.numel() for p in flow.parameters()),
              'batch_size': batch_size, 'edge_budget': config.get('edge_budget', 200000),
              'batch_sizes': [len(b) for b in batches], 'selection': selection,
              'atoms': [[r['_training'][1].num_atoms for r in b] for b in batches],
              'record_ids': [[r['id'] for r in b] for b in batches],
              'seconds_per_step': statistics.mean(elapsed), 'median_seconds_per_step': statistics.median(elapsed),
              'step_seconds': elapsed, 'records_per_second': measured_records/sum(elapsed),
              'peak_cuda_bytes': peak, 'peak_cuda_reserved_bytes': reserved,
              'checkpoint_layers': flow.checkpoint_layers, 'edge_chunk_size': flow.edge_chunk_size,
              'composition_probability': config.get('composition_probability', 0.5),
              'observation_probability': config.get('observation_probability', 0.25),
              'properties': config.get('properties', {}), 'allowed_elements': config.get('allowed_elements'),
              'data_revision': config.get('data_revision'), 'config_sha256': file_hash(root_path(args.config)),
              'fitted_config_sha256': hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest(),
              'seed': args.seed, 'threads': args.threads, 'warmup': args.warmup, 'cold_step_seconds': cold_seconds,
              'device': torch.cuda.get_device_name(torch.device(args.device)) if cuda else 'cpu',
              'precision': precision, 'requested_precision': requested_precision, 'torch': torch.__version__,
              **hashes, 'scope': 'Short training-step timing, no weights saved; reference compares dense messages '
              'and full-q projection with shared updated data plumbing; not full old-release timing or convergence.'}
    report['total_parameters'] = report['proposal_parameters']+report['flow_parameters']
    report['synthetic_resource_test'] = synthetic
    artifact_path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)
    if args.profile:
        activities = [torch.profiler.ProfilerActivity.CPU]
        if cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        with torch.profiler.profile(activities=activities) as profile:
            training_step(batches[0])
            if cuda:
                torch.cuda.synchronize()
        rows = [{'name': e.key, 'count': e.count, 'cpu_self_us': e.self_cpu_time_total,
                 'device_total_us': e.device_time_total if hasattr(e, 'device_time_total') else e.cuda_time_total}
                for e in profile.key_averages()]
        artifact_path(args.profile).write_text(json.dumps({'events': rows, 'scope': 'Separate untimed step'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
