"""Full prepared-data statistics, exhaustive catalog checks, and independent source re-encoding."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
from collections import defaultdict
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yaml
from scipy.optimize import linear_sum_assignment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.cli import artifact_path, root_path
from unifiedgroupgen.representation_audit import audit_record, audit_catalog, compiled_chart, expand_numpy
from unifiedgroupgen.symmetry import Descriptor
from unifiedgroupgen.standard_metrics import quantiles
from unifiedgroupgen.training import file_hash


def compare_records(saved, fresh):
    c = compiled_chart(Descriptor.from_dict(saved['descriptor']))
    d = compiled_chart(Descriptor.from_dict(fresh['descriptor']))
    lattice, first, h = expand_numpy(c, saved['state'])
    _, second, other_h = expand_numpy(d, fresh['state'])
    if c.num_atoms != d.num_atoms or Counter_elements(c) != Counter_elements(d):
        raise ValueError('Source re-encoding changed atom counts/composition')
    if c.descriptor.number != d.descriptor.number:
        raise ValueError('Source re-encoding recovered a different group')
    # A common origin shift and same-species assignment are allowed; positions are
    # compared at the saved standard cell, while lattice strain is reported separately.
    element = min(set(c.elements), key=lambda e: np.count_nonzero(c.elements == e))
    anchor = first[np.flatnonzero(c.elements == element)[0]]
    best = None
    for candidate in second[d.elements == element]:
        adjusted = second+(anchor-candidate)
        distances = []
        for species in np.unique(c.elements):
            delta = first[c.elements == species, None]-adjusted[None, d.elements == species]
            delta[..., c.periodic] -= np.round(delta[..., c.periodic])
            cost = np.linalg.norm(delta @ lattice.T, axis=-1)
            rows, cols = linear_sum_assignment(cost)
            distances.extend(cost[rows, cols])
        score = float(np.sqrt(np.mean(np.square(distances))))
        if best is None or score < best[0]:
            best = score, float(max(distances))
    relative = float(np.linalg.norm(h-other_h)/np.linalg.norm(h))
    return {'source_reencode_rms_angstrom': best[0], 'source_reencode_max_angstrom': best[1],
            'source_reencode_metric_relative_error': relative,
            'source_reencode_passed': best[1] < 1e-5 and relative < 1e-5}


def Counter_elements(compiled):
    values, counts = np.unique(compiled.elements, return_counts=True)
    return dict(zip(values, counts))


def source_worker(item):
    from unifiedgroupgen.data import convert_row
    from unifiedgroupgen.symmetry import compile_descriptor
    result = convert_row(item)
    compile_descriptor.cache_clear()
    return result


def record_worker(record):
    try:
        result = audit_record(record)
    except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
        result = {'id': record['id'], 'dataset': record['dataset'], 'passed': False, 'reason': str(error)}
    return record, result


def check_sources(candidates, config, folder, workers):
    from unifiedgroupgen.data import _prepare_worker
    rows, items = [], []
    for dataset, chosen in candidates.items():
        print(json.dumps({'phase': 'source_lookup', 'dataset': dataset, 'selected': len(chosen)}), flush=True)
        source_config = yaml.safe_load(root_path(config['source_configs'][dataset]).read_text(encoding='utf-8'))
        source_root = root_path(source_config['data_root'])
        portable = ROOT/'raw_data'/dataset
        if source_config['kind'] == 'space' and portable.exists():
            source_root = portable
        wanted = defaultdict(list)
        for record in chosen:
            wanted[record['provenance']['source_cif_sha256']].append(record)
        found = set()
        for split in ('train', 'val', 'test'):
            path = source_root/f'{split}.csv'
            if not path.exists():
                continue
            for chunk in pd.read_csv(path, chunksize=1000):
                for _, row in chunk.iterrows():
                    checksum = hashlib.sha256(row['cif'].encode()).hexdigest()
                    if checksum not in wanted or checksum in found:
                        continue
                    found.add(checksum)
                    for old in wanted[checksum]:
                        row_config = {**source_config, 'symmetry_tolerance': old['provenance'].get(
                            'symmetry_tolerance_angstrom', source_config['symmetry_tolerance'])}
                        items.append((len(items), row.to_dict(), row_config, str(path)))
                        rows.append({'id': old['id'], 'dataset': dataset, 'saved': old, 'source_cif_sha256': checksum})
        for checksum in wanted.keys()-found:
            for old in wanted[checksum]:
                rows.append({'id': old['id'], 'dataset': dataset, 'source_reencode_passed': None,
                             'reason': 'Source CIF unavailable or hash mismatch', 'source_cif_sha256': checksum})
    # Conversion is isolated in existing preprocessing workers: a native failure cannot
    # silently disappear from the denominator. Journal earlier outcomes before aborting.
    work_rows = [row for row in rows if 'saved' in row]
    with (folder/'source_reencoding.jsonl').open('w', encoding='utf-8') as stream, \
         multiprocessing.get_context('spawn').Pool(workers, initializer=_prepare_worker, maxtasksperchild=128) as pool:
        # imap bounds worker state; get(timeout) turns a native abort/hang into a visible failure.
        pending = pool.imap(source_worker, items, chunksize=1)
        for processed, _ in enumerate(items, 1):
            result = pending.next(timeout=120)
            index = result['row']
            row = work_rows[index]
            old = row.pop('saved')
            try:
                if 'record' not in result:
                    raise ValueError(result['rejection']['reason'])
                row.update(compare_records(old, result['record']))
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                row.update(source_reencode_passed=False, reason=str(error))
            stream.write(json.dumps(row, allow_nan=False)+'\n')
            stream.flush()
            if processed % 100 == 0 or processed == len(items):
                print(json.dumps({'phase': 'source_reencoding', 'evaluated': processed,
                                  'requested': len(items)}), flush=True)
        for row in rows:
            if row.get('source_reencode_passed') is None:
                stream.write(json.dumps(row)+'\n')
    return rows


def plot_report(frame, catalog, destination):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    for dataset, rows in frame.groupby('dataset'):
        for axis, key in zip(axes[:2], ['inverse_state_max_error', 'path_operation_residual_angstrom']):
            values = np.sort(rows[key].dropna().clip(lower=1e-16))
            axis.plot(values, np.arange(1, len(values)+1)/len(values), label=f'{dataset} (n={len(rows)})')
            axis.set_xscale('log')
    axes[0].set_xlabel('Inverse state error (dimensionless)')
    axes[1].set_xlabel('Path operation residual (Å)')
    axes[0].set_ylabel('Fraction of evaluated records')
    axes[0].legend(fontsize=8)
    counts = catalog.groupby('kind')['passed'].agg(['sum', 'count'])
    axes[2].bar(counts.index, counts['count'], color='#d4dce5', label='All orbit types')
    axes[2].bar(counts.index, counts['sum'], color='#437da6', label='Passed')
    axes[2].set_ylabel('Catalog single-orbit types')
    axes[2].legend(fontsize=8)
    fig.savefig(destination/'representation.png', dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/joint_space_layer_large.yaml')
    parser.add_argument('--output', default='artifacts/evaluations/representation')
    parser.add_argument('--figures', default='assets/representation')
    parser.add_argument('--limit', type=int, help='Per dataset; marks the report as partial')
    parser.add_argument('--source-limit', type=int, default=0, help='0: all records; positive: reproducible subset per source')
    parser.add_argument('--no-source-check', action='store_true')
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if args.workers < 1 or args.source_limit < 0 or (args.limit is not None and args.limit < 1):
        parser.error('Invalid record limit or worker count')
    import torch
    torch.set_num_threads(1)
    config = yaml.safe_load(root_path(args.config).read_text(encoding='utf-8'))
    output, figures = artifact_path(args.output), artifact_path(args.figures)
    output.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    paths = [root_path(path) for split in ('train', 'val', 'test') for path in config[f'{split}_files']]
    rows, candidates, counts = [], defaultdict(list), defaultdict(int)
    def inputs():
        selected = defaultdict(int)
        for path in paths:
            for line in path.open(encoding='utf-8'):
                if not line.strip():
                    continue
                record = json.loads(line)
                dataset = record['dataset']
                if args.limit is not None and selected[dataset] >= args.limit:
                    continue
                selected[dataset] += 1
                yield record
    from unifiedgroupgen.data import _prepare_worker
    with (output/'records.jsonl').open('w', encoding='utf-8') as journal, \
         multiprocessing.get_context('spawn').Pool(args.workers, initializer=_prepare_worker, maxtasksperchild=256) as pool:
        for record, result in pool.imap(record_worker, inputs(), chunksize=8):
            dataset = record['dataset']
            counts[dataset] += 1
            rows.append(result)
            if not args.no_source_check:
                candidates[dataset].append(record)
            journal.write(json.dumps(result, allow_nan=False)+'\n')
            if len(rows) % 1000 == 0:
                journal.flush(); print(json.dumps({'phase': 'records', 'evaluated': len(rows)}), flush=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output/'records.csv', index=False)
    catalog = pd.DataFrame(audit_catalog())
    catalog.to_csv(output/'catalog.csv', index=False)
    for dataset, selected in candidates.items():
        selected.sort(key=lambda r: hashlib.sha256(f"{dataset}:{r['id']}".encode()).hexdigest())
        if args.source_limit:
            candidates[dataset] = selected[:args.source_limit]
    sources = check_sources(candidates, config, output, args.workers) if candidates else []
    pd.DataFrame(sources).to_csv(output/'source_reencoding.csv', index=False)
    report = {'record_scope': 'partial' if args.limit else 'all_prepared_train_val_test',
              'parameters': {'seed': 182, 'path_points': 3, 'dtype': 'float64', 'source_limit': args.source_limit,
                             'record_limit_per_source': args.limit, 'workers': args.workers},
              'code_sha256': {path.name: file_hash(path) for path in
                  (Path(__file__), ROOT/'src/unifiedgroupgen/representation_audit.py')},
              'input_sha256': {str(path.relative_to(ROOT)): file_hash(path) for path in paths},
              'records': {}, 'catalog': {}, 'source_reencoding': {},
              'scope': 'Catalog SG/LG/PG and prepared structural charts. Source re-encoding compares independently '
                       'standardized source CIFs to stored charts, not raw-coordinate identity or physical stability. '
                       'Finite orbit types do not exhaust unbounded repeated multi-orbit lists.'}
    for dataset, group in frame.groupby('dataset'):
        report['records'][dataset] = {'requested': len(group), 'passed': int(group['passed'].sum()),
            'statistics': {key: quantiles(group[key].tolist()) for key in frame.select_dtypes(include='number').columns}}
    for kind, group in catalog.groupby('kind'):
        report['catalog'][kind] = {'groups': int(group['group'].nunique()), 'orbit_types': len(group),
                                   'passed': int(group['passed'].sum())}
    for dataset in counts:
        selected = [r for r in sources if r['dataset'] == dataset]
        report['source_reencoding'][dataset] = {'selected': len(selected),
            'evaluated': sum(r.get('source_reencode_passed') is not None for r in selected),
            'passed': sum(r.get('source_reencode_passed') is True for r in selected),
            'scope': 'not_requested' if args.no_source_check else ('subset' if args.source_limit or args.limit else 'all'),
            'statistics': {key: quantiles([r.get(key) for r in selected]) for key in
                ('source_reencode_rms_angstrom', 'source_reencode_max_angstrom', 'source_reencode_metric_relative_error')}}
    report['passed'] = bool(frame['passed'].all() and catalog['passed'].all()
                            and all(r.get('source_reencode_passed') is True for r in sources))
    (output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    plot_report(frame, catalog, figures)
    print(json.dumps({'passed': report['passed'], 'records': len(frame), 'output': str(output)}), flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
