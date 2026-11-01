"""Calibrate real TRAIN batches; discard all temporary weights afterwards."""
import gc
import random
import time
from pathlib import Path

import torch

from submit_files.hpc5090 import CATALOG, read, write
from unifiedgroupgen.common.configuration import load_experiments, project_root
from unifiedgroupgen.pl_data.dataset import batch_indices, load_training_splits, prepare_training_records
from unifiedgroupgen.pl_modules.factory import build_models
from unifiedgroupgen.pl_modules.objectives.joint import joint_loss
from unifiedgroupgen.run import fit_training_config


CANDIDATES = {
    'mp20': [(256, 400000), (384, 500000), (512, 600000)],
    'c2db51': [(768, 200000), (1024, 300000), (1536, 400000)],
    'joint': [(256, 400000), (384, 500000), (512, 600000)],
}


def select_trial(trials):
    valid = [t for t in trials if t['state'] == 'pass']
    if not valid:
        raise RuntimeError('No batch passed the finite-gradient and memory checks')
    return max(valid, key=lambda t: t['records_per_second'])


def trial(config, records, batch_size, edge_budget, cap):
    random.seed(42); torch.manual_seed(42)
    proposal, flow = build_models(config, 'cuda')
    flow.edge_chunk_size = 32768
    flow.checkpoint_layers = False
    params = [*proposal.parameters(), *flow.parameters()]
    optimizer = torch.optim.AdamW(params, lr=config['learning_rate'])
    batches = batch_indices(records, batch_size, edge_budget, rng=random.Random(42))
    # Spread timing batches over the size distribution and stress the largest graphs.
    ordered = sorted(batches, key=lambda b: sum(records[i]['_training'][1].num_atoms**2 for i in b))
    measured = [ordered[int((len(ordered)-1)*q)] for q in (.1, .35, .65, .9)]
    stress = ordered[-2:]
    durations = []; seen = 0
    torch.cuda.reset_peak_memory_stats()
    for step, indices in enumerate([measured[0]]*2+measured+stress):
        batch = [records[i] for i in indices]
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(); started = time.perf_counter()
        with torch.autocast('cuda', dtype=torch.bfloat16):
            parts = joint_loss(proposal, flow, batch, config['composition_probability'],
                               config['observation_probability'], 3.)
            weights = config['loss_weights']
            loss = weights['descriptor']*parts['descriptor'].mean()+weights['flow']*parts['flow'].mean()
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite calibration loss')
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, config['gradient_clip'], error_if_nonfinite=True)
        optimizer.step(); torch.cuda.synchronize()
        if 2 <= step < 6:
            durations.append(time.perf_counter()-started); seen += len(batch)
    peak = torch.cuda.max_memory_reserved()
    return {'batch_size': batch_size, 'edge_budget': edge_budget,
            'state': 'pass' if peak <= cap else 'memory_margin_failed',
            'peak_reserved_gib': peak/2**30, 'peak_allocated_gib': torch.cuda.max_memory_allocated()/2**30,
            'records_per_second': seen/sum(durations), 'timed_records': seen,
            'estimated_train_epoch_seconds': len(records)/(seen/sum(durations)),
            'timing_scope': '4 TRAIN batches after 2 warmups; 2 largest-edge stress batches excluded from throughput'}


def profile(campaign):
    output = Path(campaign['folder'])/'profile.json'
    if output.exists():
        saved = read(output)
        if saved['revision'] != campaign['revision']:
            raise ValueError('Calibration source revision changed')
        if set(saved['selected']) == set(CANDIDATES):
            print('Using the completed calibration; matched training batches remain fixed.', flush=True)
            return
    torch.set_num_threads(4)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError('Calibration requires the allocated BF16-capable GPU')
    catalog = load_experiments(CATALOG)
    total = torch.cuda.get_device_properties(0).total_memory
    cap = min(.85*total, total-2*2**30)
    result = {'revision': campaign['revision'], 'device': torch.cuda.get_device_name(0), 'memory_limit_gib': cap/2**30,
              'temporary_weights_discarded': True, 'selected': {}, 'trials': {}}
    for family, candidates in CANDIDATES.items():
        config = catalog[family+'_u_comp_v2']['config']
        records, _, _ = load_training_splits(config, project_root())
        config = fit_training_config(config, records)
        records = prepare_training_records(records)
        trials = []
        for batch_size, edge_budget in candidates:
            try:
                row = trial(config, records, batch_size, edge_budget, cap)
            except torch.cuda.OutOfMemoryError:
                row = {'batch_size': batch_size, 'edge_budget': edge_budget, 'state': 'oom'}
            trials.append(row)
            gc.collect(); torch.cuda.empty_cache()
            print(family, row, flush=True)
        result['trials'][family] = trials
        result['selected'][family] = select_trial(trials)
        write(output, result)
        del records
        gc.collect()
    print('Both joint runs use the SAME selected joint batch/edge budget.', flush=True)
