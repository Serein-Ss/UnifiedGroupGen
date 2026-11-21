"""Compare joint-flow backbones on the same cached MP-20 intermediate states."""
import argparse
import json
import random
from pathlib import Path
from statistics import median
from time import perf_counter

import torch

from unifiedgroupgen.common.configuration import project_root
from unifiedgroupgen.experiment_config import compose_experiment, training_definition
from unifiedgroupgen.pl_data.dataset import CrystalDataset, load_training_splits
from unifiedgroupgen.pl_data.unified.path import sample_path
from unifiedgroupgen.pl_data.unified.tasks import task_target
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_modules.unified.module import strict_fp32
from unifiedgroupgen.pl_modules.unified.objectives import loss
from unifiedgroupgen.run import fit_training_config


BACKBONES = ('cspnet', 'equiformer_v3', 'esen', 'mace')
# Match total trainable parameters on the MP-20 training vocabulary.
MP20_MATCHED_CHANNELS = dict(cspnet=288, equiformer_v3=64, esen=78, mace=38)
MP20_PARAMETER_TARGET = 2_605_631
PARAMETER_TOLERANCE = .02


def prepared_batches(config, records, device):
    order = list(range(len(records)))
    random.Random(101).shuffle(order)
    data = CrystalDataset([records[i] for i in order[:96]], config['model']['operation_compiler'])
    batch, batches = [], []
    for index, row in enumerate(data.records):
        target = row['_joint_target']
        if not 10 <= target.state.num_atoms <= 32:
            continue
        seed = 101 + index
        sample = sample_path(task_target(target, config, seed), seed,
                             time=(.65, .85)[index % 2], allowed_elements=config['allowed_elements'],
                             max_atoms=config['max_atoms'], max_orbits=config['max_orbits'],
                             budget=config['event_budget'], birth_scale=config['birth_scale'])
        if sample.state.num_atoms:
            batch.append(sample.to(device))
            if len(batch) == 2:
                batches.append(batch)
                batch = []
        if len(batches) == 6:
            return batches
    raise ValueError('Insufficient nonempty MP-20 training samples for the benchmark')


def measure(step, batches, warmup, steps, device):
    for i in range(warmup):
        step(batches[i % len(batches)])
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    elapsed = []
    for i in range(steps):
        start = perf_counter()
        step(batches[i % len(batches)])
        torch.cuda.synchronize(device)
        elapsed.append(perf_counter() - start)
    return dict(median_ms=1000*median(elapsed), structures_per_second=2*steps/sum(elapsed),
                peak_allocated_gib=torch.cuda.max_memory_allocated(device)/2**30,
                elapsed_seconds=elapsed)


def compare(config, batches, device, warmup, steps, parameter_target=None):
    torch.manual_seed(42)
    model = build_joint_model(config, device).train()
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if parameter_target is not None and abs(parameters - parameter_target) > parameter_target * PARAMETER_TOLERANCE:
        name = config['model']['backbone']
        raise ValueError(f'{name} has {parameters:,} trainable parameters, outside the matched budget '
                         f'of {parameter_target:,} +/- {PARAMETER_TOLERANCE:.0%}')
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['learning_rate'], fused=True)

    def update(batch):
        optimizer.zero_grad(set_to_none=True)
        loss(model, batch, config['jump_weight'])['loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['gradient_clip'], error_if_nonfinite=True)
        optimizer.step()

    training = measure(update, batches, warmup, steps, device)
    optimizer.zero_grad(set_to_none=True)
    del optimizer
    model.eval()

    def predict(batch):
        model.forward_batch([x.state for x in batch], [x.time for x in batch], [x.properties for x in batch])

    with torch.inference_mode():
        inference = measure(predict, batches, warmup, steps, device)
    return dict(parameters=parameters,
                model=config['model'], training=training, prediction=inference)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--parameter-matched', action='store_true',
                        help='Use the MP-20 channel widths calibrated to 2.61M trainable parameters')
    args = parser.parse_args()
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available() or args.steps < 1 or args.warmup < 1:
        parser.error('Use a CUDA device and positive step and warmup counts')
    torch.set_num_threads(4)
    strict_fp32(dict(precision='fp32', strict_fp32=True))
    root = project_root()
    result = dict(device=torch.cuda.get_device_name(device), precision='strict_fp32',
                  batch_size=2, warmup=args.warmup, steps=args.steps,
                  workload='Identical MP-20 intermediate batches; preprocessing and path sampling excluded',
                  datasets={})
    target = MP20_PARAMETER_TARGET if args.parameter_matched else None
    if target is not None:
        result['parameter_budget'] = dict(target=target, relative_tolerance=PARAMETER_TOLERANCE,
                                          channels=MP20_MATCHED_CHANNELS)
    for task in ('dng', 'csp'):
        experiment = 'full_mp20_eqv2_' + task
        base = training_definition(compose_experiment(experiment))
        records, _, _ = load_training_splits(base, root)
        batches = prepared_batches(fit_training_config(base, records), records, device)
        task_result = dict(batches=[[dict(sample=x.identity, atoms=x.state.num_atoms, time=x.time)
                                     for x in batch] for batch in batches], backbones={})
        result['datasets'][task] = task_result
        for backbone in BACKBONES:
            overrides = [f'model=joint_flow_{backbone}']
            if target is not None:
                overrides.append(f'model.channels={MP20_MATCHED_CHANNELS[backbone]}')
            config = training_definition(compose_experiment(experiment, overrides=overrides))
            config = fit_training_config(config, records)
            stats = compare(config, batches, device, args.warmup, args.steps, parameter_target=target)
            torch.cuda.empty_cache()
            task_result['backbones'][backbone] = stats
            print(task, backbone, stats['parameters'], {phase: round(stats[phase]['median_ms'], 2)
                                  for phase in ('training','prediction')}, flush=True)
    with args.output.open('x', encoding='utf8') as out:
        json.dump(result, out, indent=2)


if __name__ == '__main__':
    main()
