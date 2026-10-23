"""Audit 12 thread timings; distinguish data changes from unused scaler roundoff."""

import argparse
import copy
import csv
import hashlib
import json
import math
from pathlib import Path

MODELS = ('mp20_all', 'c2db51_all', 'joint_all')
THREADS = (1, 2, 4, 8)
IDENTITY = ('config', 'implementation', 'proposal_parameters', 'flow_parameters', 'total_parameters',
            'batch_size', 'edge_budget', 'batch_sizes', 'selection', 'atoms', 'record_ids',
            'checkpoint_layers', 'edge_chunk_size', 'composition_probability', 'observation_probability',
            'allowed_elements', 'data_revision', 'config_sha256', 'seed', 'warmup', 'device',
            'precision', 'requested_precision', 'torch', 'train_split_sha256', 'val_split_sha256',
            'timing_pool_sha256', 'synthetic_resource_test')


def compare(reference, candidate):
    errors = [key for key in IDENTITY if key not in reference or key not in candidate
              or reference[key] != candidate[key]]
    properties = copy.deepcopy(candidate['properties'])
    roundoff = []
    # PropertyEncoder._normalized uses domain_scalers when present, so these global
    # shift/scale values do not participate in joint-model condition encoding.
    for name, expected in reference['properties'].items():
        actual = properties.get(name, {})
        scalers = expected.get('domain_scalers')
        if not scalers or actual.get('domain_scalers') != scalers:
            continue
        for key in ('shift', 'scale'):
            a, b = expected.get(key), actual.get(key)
            if a == b:
                continue
            if (isinstance(a, (int, float)) and isinstance(b, (int, float))
                    and math.isfinite(a) and math.isfinite(b)
                    and math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)):
                roundoff.append({'path': f'properties.{name}.{key}', 'reference': a, 'candidate': b,
                                 'absolute_difference': abs(a-b), 'used_for_condition_encoding': False})
                actual[key] = a
    if properties != reference['properties']:
        errors.append('effective_properties_or_non_roundoff_metadata')
    if (reference['fitted_config_sha256'] != candidate['fitted_config_sha256'] and not roundoff):
        errors.append('unexplained_fitted_config_sha256')
    return {'compatible': not errors, 'errors': errors,
            'exact_fitted_config_hash': reference['fitted_config_sha256'] == candidate['fitted_config_sha256'],
            'unused_global_scaler_roundoff': roundoff}


def audit(directory, output):
    output.mkdir(parents=True, exist_ok=True)
    report = {'passed': True, 'models': {}, 'files_sha256': {},
              'scope': 'Short-step runtime candidates; not full-epoch or generation quality validation'}
    rows = []
    for model in MODELS:
        records = {}
        for threads in THREADS:
            path = directory/f'{model}_t{threads}.json'
            records[threads] = json.loads(path.read_text(encoding='utf-8'))
            report['files_sha256'][path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        comparisons = {str(t): compare(records[1], r) for t, r in records.items()}
        valid = all(c['compatible'] for c in comparisons.values())
        for t, r in records.items():
            valid = valid and r['threads'] == t and r['checkpoint_layers'] is False and r['edge_chunk_size'] == 32768
            valid = valid and r['precision'] == 'bf16' and r['batch_size'] == 64 and r['data_revision'] == 'source_integrity_v1'
            valid = valid and math.isfinite(r['records_per_second']) and r['records_per_second'] > 0
            rows.append({'model': model, 'threads': t, 'mean_seconds': r['seconds_per_step'],
                         'median_seconds': r['median_seconds_per_step'], 'samples_per_second': r['records_per_second'],
                         'allocated_GiB': r['peak_cuda_bytes']/2**30,
                         'reserved_GiB': r['peak_cuda_reserved_bytes']/2**30})
        winner = max(records, key=lambda t: records[t]['records_per_second']) if valid else None
        recommendation = ({'threads': winner, 'checkpoint_layers': False, 'edge_chunk_size': 32768,
                           'batch_size': 64, 'precision': 'bf16',
                           'samples_per_second': records[winner]['records_per_second']} if valid else None)
        report['models'][model] = {'passed': bool(valid), 'comparisons_with_t1': comparisons,
                                   'fastest_measured_candidate': recommendation}
        report['passed'] = report['passed'] and bool(valid)
        print(model, json.dumps(recommendation) if valid else 'INPUT AUDIT FAILED')
    with (output/'thread_comparison.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    (output/'thread_audit.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    if not report['passed']:
        raise ValueError('Benchmark identities or runtime settings differ; inspect thread_audit.json')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    audit(args.directory.resolve(), (args.output or args.directory/'audit').resolve())


if __name__ == '__main__':
    main()
