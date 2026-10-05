"""Batched joint training, deterministic validation, and resumable checkpoints."""

import hashlib
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from .constraints import AffineObservation
from .flow import flow_loss_batch
from .symmetry import Descriptor, compile_descriptor


def batch_indices(records, batch_size, edge_budget, shuffle=True):
    """Size buckets reduce variance while bounding the total full-graph edge count."""
    if batch_size < 1 or edge_budget < 1:
        raise ValueError('Batch size and edge budget must be positive')
    sizes = [compile_descriptor(Descriptor.from_dict(r['descriptor'])).num_atoms for r in records]
    order = list(range(len(records)))
    if shuffle:
        random.shuffle(order)
        order = [j for start in range(0, len(order), batch_size*32)
                 for j in sorted(order[start:start+batch_size*32], key=lambda i: sizes[i])]
    batches, current, edges = [], [], 0
    for index in order:
        required = sizes[index]*(sizes[index]-1)
        if required > edge_budget:
            raise ValueError(f"Single structure {records[index]['id']} needs {required} edges, budget={edge_budget}")
        if current and (len(current) >= batch_size or edges+required > edge_budget):
            batches.append(current)
            current, edges = [], 0
        current.append(index)
        edges += required
    if current:
        batches.append(current)
    if shuffle:
        random.shuffle(batches)
    return batches


def joint_loss(proposal, flow, records, composition_probability=0.5, observation_probability=0.25):
    descriptors = [Descriptor.from_dict(r['descriptor']) for r in records]
    compiled = [compile_descriptor(d) for d in descriptors]
    like = next(flow.parameters())
    cpu_endpoints = [torch.tensor(r['state'], dtype=like.dtype) for r in records]
    endpoints = torch.cat(cpu_endpoints).to(like.device).split([z.numel() for z in cpu_endpoints])
    compositions, observations = [], []
    for r, c, endpoint in zip(records, compiled, cpu_endpoints):
        if endpoint.shape != (c.dof,) or not torch.isfinite(endpoint).all():
            raise ValueError(f"Invalid training state in {r['id']}")
        elements, counts = np.unique(c.elements, return_counts=True)
        composition = dict(zip(elements.tolist(), counts.tolist())) if random.random() < composition_probability else None
        observation = None
        if random.random() < observation_probability:
            indices = (list(range(c.lattice_dof)) if random.random() < 0.5
                       else random.sample(range(c.dof), max(1, c.dof//3)))
            observation = AffineObservation.fixed(c.dof, indices, endpoint[indices].detach().cpu())
        compositions.append(composition)
        observations.append(observation)
    properties = [r['properties'] for r in records]
    descriptor = proposal.loss_batch(descriptors, properties, compositions)
    continuous = flow_loss_batch(flow, compiled, endpoints, properties, observations, compositions)
    return {'descriptor': descriptor, 'flow': continuous['loss'],
            'lattice': continuous['lattice'], 'coordinates': continuous['coordinates']}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def save_checkpoint(path, checkpoint):
    temporary = path.with_suffix(path.suffix+'.tmp')
    torch.save(checkpoint, temporary)
    temporary.replace(path)


def save_status(output, status):
    path = output.with_suffix('.status.json')
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
    temporary.replace(path)


def run_training(args, config, records, validation, proposal, flow, train_paths, val_paths, output):
    if output.exists() and not args.resume:
        raise ValueError(f"Checkpoint exists: {output}; use --resume or a new --output")
    parameters = list(proposal.parameters())+list(flow.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.get('learning_rate', 1e-3))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=config.get('lr_patience', 10), factor=0.5)
    precision = config.get('precision', 'fp32')
    if precision not in ('fp32', 'bf16', 'fp16'):
        raise ValueError('precision must be fp32, bf16, or fp16')
    if precision != 'fp32' and not args.device.startswith('cuda'):
        raise ValueError('Mixed precision requires CUDA; use fp32 for CPU diagnostics')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        raise ValueError('This GPU does not support bf16; use fp32 or fp16')
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('cuda', enabled=precision == 'fp16')
    else:
        scaler = torch.cuda.amp.GradScaler(enabled=precision == 'fp16')
    dtype = torch.bfloat16 if precision == 'bf16' else torch.float16
    hashes = {'train_split_sha256': [file_hash(p) for p in train_paths],
              'val_split_sha256': [file_hash(p) for p in val_paths]}
    start_epoch, best, stale = 0, float('inf'), 0
    runtime = {'checkpoint_layers': flow.checkpoint_layers, 'edge_chunk_size': flow.edge_chunk_size}
    if args.resume:
        saved = torch.load(args.resume, map_location=args.device, weights_only=False)
        if saved['config'] != config or any(saved[key] != value for key, value in hashes.items()):
            raise ValueError('Resume configuration/scalers or split hashes differ from the checkpoint')
        proposal.load_state_dict(saved['proposal'])
        flow.load_state_dict(saved['flow'])
        optimizer.load_state_dict(saved['optimizer'])
        if 'scheduler' in saved:
            scheduler.load_state_dict(saved['scheduler'])
            scaler.load_state_dict(saved['scaler'])
        start_epoch = saved['epoch']+1
        best, stale = saved.get('best_val_loss', best), saved.get('stale_epochs', 0)
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['torch_rng'].cpu())
        if args.device.startswith('cuda') and saved.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in saved['cuda_rng']])
        runtime.update(saved.get('runtime_options', {}))
    if getattr(args, 'checkpoint', None) is not None:
        runtime['checkpoint_layers'] = args.checkpoint
    if getattr(args, 'edge_chunk_size', None) is not None:
        runtime['edge_chunk_size'] = args.edge_chunk_size
    if runtime['edge_chunk_size'] < 1:
        raise ValueError('edge_chunk_size must be positive')
    flow.checkpoint_layers, flow.edge_chunk_size = runtime['checkpoint_layers'], runtime['edge_chunk_size']
    batch_size = args.batch_size if args.batch_size is not None else config.get('batch_size', 8)
    edge_budget = config.get('edge_budget', 200000)
    epochs = args.epochs if args.epochs is not None else config.get('epochs', 100)
    if epochs < 1:
        raise ValueError('Total training epochs must be positive')
    patience = config.get('early_stopping_patience', 30)
    weights = config.get('loss_weights', {'descriptor': 1.0, 'flow': 1.0})
    if not 0 <= config.get('composition_probability', 0.5) <= 1 or not 0 <= config.get('observation_probability', 0.25) <= 1:
        raise ValueError('Condition probabilities must lie in [0,1]')
    print(json.dumps({'event': 'training_start', 'proposal_parameters': sum(p.numel() for p in proposal.parameters()),
                      'flow_parameters': sum(p.numel() for p in flow.parameters()), 'precision': precision,
                      'batch_size': batch_size, 'edge_budget': edge_budget,
                      'train_records': len(records), 'val_records': len(validation),
                      'start_epoch': start_epoch, 'maximum_epochs': epochs, 'runtime_options': runtime}), flush=True)
    completed = start_epoch
    reason = 'maximum_epochs'
    for epoch in range(start_epoch, epochs):
        started = time.perf_counter()
        if args.device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
        proposal.train(); flow.train()
        totals = {key: 0.0 for key in ('descriptor', 'flow', 'lattice', 'coordinates')}
        gradient_norms = []
        batches = batch_indices(records, batch_size, edge_budget)
        save_status(output, {'state': 'running', 'phase': 'train', 'epoch': epoch,
                             'step': 0, 'steps': len(batches), 'completed_epochs': completed})
        for step, indices in enumerate(batches, 1):
            chunk = [records[i] for i in indices]
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=torch.device(args.device).type, dtype=dtype, enabled=precision != 'fp32'):
                losses = joint_loss(proposal, flow, chunk, config.get('composition_probability', 0.5),
                                    config.get('observation_probability', 0.25))
                loss = weights['descriptor']*losses['descriptor'].mean()+weights['flow']*losses['flow'].mean()
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite batch loss: {[r['id'] for r in chunk]}")
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            norm = torch.nn.utils.clip_grad_norm_(parameters, config.get('gradient_clip', 10), error_if_nonfinite=True)
            scaler.step(optimizer); scaler.update()
            gradient_norms.append(float(norm))
            for key in totals:
                totals[key] += float(losses[key].detach().sum())
            if step % 25 == 0 or step == len(batches):
                progress = {'event': 'progress', 'state': 'running', 'phase': 'train',
                            'epoch': epoch, 'step': step, 'steps': len(batches),
                            'loss': float(loss.detach()), 'seconds': time.perf_counter()-started,
                            'completed_epochs': completed}
                print(json.dumps(progress), flush=True)
                save_status(output, progress)
        training = {key: value/len(records) for key, value in totals.items()}
        proposal.eval(); flow.eval()
        totals = {key: 0.0 for key in totals}
        save_status(output, {'state': 'running', 'phase': 'validation', 'epoch': epoch,
                             'completed_epochs': completed})
        python_rng = random.getstate()
        devices = [torch.device(args.device).index or 0] if args.device.startswith('cuda') else []
        with torch.random.fork_rng(devices=devices):
            random.seed(918); torch.manual_seed(918)
            with torch.no_grad():
                for indices in batch_indices(validation, batch_size, edge_budget, shuffle=False):
                    losses = joint_loss(proposal, flow, [validation[i] for i in indices], 0, 0)
                    for key in totals:
                        totals[key] += float(losses[key].sum())
        random.setstate(python_rng)
        validated = {key: value/len(validation) for key, value in totals.items()}
        val_loss = weights['descriptor']*validated['descriptor']+weights['flow']*validated['flow']
        if not np.isfinite(val_loss):
            raise FloatingPointError('Nonfinite validation loss')
        improved = val_loss < best
        best, stale = (val_loss, 0) if improved else (best, stale+1)
        scheduler.step(val_loss)
        if args.device.startswith('cuda'):
            torch.cuda.synchronize()
        history = {'epoch': epoch, 'train': training, 'val': validated, 'val_loss': val_loss,
                   'best_val_loss': best, 'seconds': time.perf_counter()-started,
                   'learning_rate': optimizer.param_groups[0]['lr'], 'gradient_norm_mean': float(np.mean(gradient_norms)),
                   'train_records': len(records), 'val_records': len(validation),
                   'runtime_options': runtime,
                   'peak_cuda_bytes': torch.cuda.max_memory_allocated() if devices else 0}
        print(json.dumps(history), flush=True)
        with output.with_suffix('.history.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(history)+'\n')
        checkpoint = {'config': config, 'epoch': epoch, 'proposal': proposal.state_dict(), 'flow': flow.state_dict(),
                      'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(),
                      'best_val_loss': best, 'stale_epochs': stale, 'python_rng': random.getstate(),
                      'runtime_options': runtime,
                      'numpy_rng': np.random.get_state(), 'torch_rng': torch.get_rng_state(),
                      'cuda_rng': torch.cuda.get_rng_state_all() if devices else None, **hashes,
                      'verification': 'trained; material stability and property performance not certified'}
        save_checkpoint(output, checkpoint)
        if improved:
            save_checkpoint(output.with_suffix('.best.pt'), checkpoint)
        completed = epoch+1
        save_status(output, {'state': 'running', 'phase': 'epoch_saved', 'completed_epochs': completed,
                             'best_val_loss': best, 'last': history})
        if patience and stale >= patience:
            print(json.dumps({'event': 'early_stop', 'epoch': epoch}), flush=True)
            reason = 'early_stopping'
            break
    result = {'state': 'complete', 'reason': reason, 'completed_epochs': completed,
              'best_val_loss': best, 'train_records': len(records), 'val_records': len(validation),
              'verification': 'Training finished; material stability and property performance not certified'}
    save_status(output, result)
    print(json.dumps({'event': 'training_complete', **result}), flush=True)
