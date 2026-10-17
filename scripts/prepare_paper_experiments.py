"""Prepare reproducible evaluation campaigns and matched-split training controls; never train here."""

import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.cli import artifact_path, root_path
from unifiedgroupgen.data import read_jsonl
from unifiedgroupgen.symmetry import Descriptor, compile_descriptor, metric_chart
from unifiedgroupgen.training import file_hash


def select_cases(records, count):
    """Prefer different groups and free dimensions, then deterministic material IDs. TRAIN only."""
    ordered = sorted(records, key=lambda r: hashlib.sha256(str(r['id']).encode()).hexdigest())
    buckets = {}
    for record in ordered:
        description = record['descriptor']
        k_dof = len(metric_chart(description['kind'], description['number'])[2])
        q_dof = len(record['state'])-k_dof
        if q_dof:
            buckets.setdefault((description['number'], q_dof), record)
    return list(buckets.values())[:count]


def build_campaign(records, domains, suite, output, seeds, count, steps, cases):
    jobs, targets = [], {}
    for dataset, domain in domains.items():
        rows = records[dataset]
        def add(tag, seed, properties=None, guidance=0, **extra):
            jobs.append(dict(name=f'{dataset}_{tag}_s{seed}', dataset=dataset, seed=seed,
                count=count, steps=steps, properties=properties or {}, guidance=guidance, **extra))
        if suite == 'unconditional':
            for seed in seeds:
                add('unconditional', seed)
        elif suite == 'conditions':
            # Each scalar is varied independently. Joint feasibility is a separate experiment.
            for name in ('formation_energy', 'band_gap', 'energy_above_hull'):
                values = [r['properties'][name] for r in rows if r['properties'].get(name) is not None
                          and np.isfinite(r['properties'][name])]
                if not values:
                    continue
                values = sorted(set(float(x) for x in np.quantile(values, [.1, .5, .9])))
                targets[f'{dataset}/{name}'] = dict(training_label_count=len([r for r in rows
                    if r['properties'].get(name) is not None]), targets=values, quantiles=[.1, .5, .9])
                for i, value in enumerate(values):
                    for guidance in (0, 1, 2):
                        for seed in seeds:
                            add(f'{name}_t{i}_g{guidance}', seed, {name: value}, guidance)
        elif suite == 'controls':
            for i, record in enumerate(select_cases(rows, cases)):
                c = compile_descriptor(Descriptor.from_dict(record['descriptor']))
                descriptor_path = output/f'{dataset}_case{i:03d}.descriptor.json'
                descriptor_path.write_text(json.dumps(record['descriptor'], indent=2), encoding='utf-8')
                indices = list(range(c.lattice_dof))
                observation_path = output/f'{dataset}_case{i:03d}.observation.json'
                observation_path.write_text(json.dumps({'indices': indices,
                    'values': [record['state'][j] for j in indices]}, indent=2), encoding='utf-8')
                extra = {'descriptor': str(descriptor_path.relative_to(ROOT)).replace('\\', '/')}
                for seed in seeds:
                    add(f'case{i:03d}_flow', seed, **extra)
                    add(f'case{i:03d}_prior', seed, prior=True, **extra)
                    add(f'case{i:03d}_fixedlattice', seed,
                        observations=str(observation_path.relative_to(ROOT)).replace('\\', '/'), **extra)
                targets[f'{dataset}/case{i:03d}'] = {'source_split': 'train', 'id': record['id'],
                    'coordinate_dof': c.coordinate_dof, 'lattice_dof': c.lattice_dof,
                    'purpose': 'Matched descriptor and random seed; not unseen-rule generalization'}
                compile_descriptor.cache_clear()
        else:
            raise ValueError('Unknown suite')
    return {'schema': 1, 'suite': suite, 'jobs': jobs, 'targets': targets,
            'total_generation_attempts': sum(j['count'] for j in jobs),
            'scope': 'Seeds repeat sampling of one checkpoint, not independent model training. '
                     'Prior controls fix descriptor/elements and bypass only the continuous learned model.'}


def matched_configs(config, output, elements, label_support=None):
    result = {}
    for dataset in config['dataset_domains']:
        control = copy.deepcopy(config)
        control['dataset'] = f'matched_{dataset}'
        control['joint_split_datasets'] = [dataset]
        control['dataset_domains'] = {dataset: config['dataset_domains'][dataset]}
        # Keep the same group/element vocabulary and architecture, including unused categories.
        control['allowed_elements'] = elements
        if label_support is not None:
            control['properties'] = {name: spec for name, spec in control.get('properties', {}).items()
                                     if spec.get('context') or name in label_support[dataset]}
        for split in ('train', 'val', 'test'):
            control[f'{split}_files'] = [p for p in config[f'{split}_files'] if Path(p).parent.name == dataset]
        control['data_audit'] = str((output/f'{dataset}.audit.json').relative_to(ROOT)).replace('\\', '/')
        path = output/f'{dataset}.yaml'
        path.write_text(yaml.safe_dump(control, sort_keys=False), encoding='utf-8')
        result[dataset] = str(path.relative_to(ROOT)).replace('\\', '/')
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/joint_space_layer_large.yaml')
    parser.add_argument('--suite', choices=['unconditional', 'conditions', 'controls', 'matched-training'], required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--seeds', nargs='+', type=int, default=[142, 143, 144])
    parser.add_argument('--count', type=int, default=256, help='Attempts per job; use 4096 for formal unconditional runs')
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--cases', type=int, default=8, help='TRAIN descriptors per dataset for matched continuous controls')
    args = parser.parse_args()
    if min(args.count, args.steps, args.cases) < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Budgets must be positive and sampling seeds distinct')
    output = artifact_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory; experiment definitions are immutable')
    output.mkdir(parents=True, exist_ok=True)
    config = yaml.safe_load(root_path(args.config).read_text(encoding='utf-8'))
    paths = [root_path(p) for p in config['train_files']]
    records = {dataset: [] for dataset in config['dataset_domains']}
    for path in paths:
        for record in read_jsonl(path):
            records[record['dataset']].append(record)
    if args.suite == 'matched-training':
        elements = sorted({o['atomic_number'] for rows in records.values() for r in rows for o in r['descriptor']['orbits']})
        support = {dataset: {name for r in rows for name, value in r['properties'].items() if value is not None}
                   for dataset, rows in records.items()}
        report = {'configs': matched_configs(config, output, elements, support), 'requires_retraining': True,
                  'scope': 'Same grouped split, group/element vocabulary and backbone dimensions. '
                           'Unlabeled property encoders are omitted per source (e.g. MPTS52 band gap); '
                           'report actual parameters. Separate models see fewer records; report steps, '
                           'best epochs and aggregate parameters as well as quality.'}
    else:
        report = build_campaign(records, config['dataset_domains'], args.suite, output,
                                args.seeds, args.count, args.steps, args.cases)
    report['train_sha256'] = {str(p.relative_to(ROOT)).replace('\\', '/'): file_hash(p) for p in paths}
    report['config_sha256'] = file_hash(root_path(args.config))
    (output/'campaign.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'output': str(output), 'jobs': len(report.get('jobs', [])),
                      'total_generation_attempts': report.get('total_generation_attempts')}))


if __name__ == '__main__':
    main()
