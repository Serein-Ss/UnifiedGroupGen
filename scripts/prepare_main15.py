"""Write/verify the 15 approved configs and audit the unchanged source split union."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import hashlib
import json
from pathlib import Path
import sys
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.main_experiments import configurations, paths
from unifiedgroupgen.preflight import audit_files, check_training_gate
from unifiedgroupgen.training import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--configs-only', action='store_true', help='No data audit; not a training authorization')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    if args.configs_only and args.verify_only:
        parser.error('Choose at most one mode')
    torch.set_num_threads(4)
    configs = configurations(ROOT)
    directory = ROOT/'configs/main15'
    directory.mkdir(parents=True, exist_ok=True)
    for name, config in configs.items():
        path = directory/f'{name}.yaml'
        if path.exists() and yaml.safe_load(path.read_text(encoding='utf-8')) != config:
            raise ValueError(f'Existing experiment config differs: {path}; do not silently change a run')
        if not path.exists():
            path.write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    if args.configs_only:
        print(json.dumps({'configurations': list(configs), 'audited': False}))
        return
    reports = {}
    for dataset in ('mp20', 'c2db51', 'joint'):
        name = dataset+'_all'
        config = configs[name]
        report_path = ROOT/config['data_audit']
        input_paths = paths(config, ROOT)
        if args.verify_only:
            check_training_gate(config, input_paths['train'], input_paths['val'], ROOT)
            report = json.loads(report_path.read_text(encoding='utf-8'))
        else:
            report = audit_files(input_paths, config, report_path)
        if not report['passed']:
            raise ValueError(f'{dataset} full integrity/split audit failed: {report["errors"][:5]}')
        # These variants differ only in property-conditioning/training metadata.
        # The most restrictive all-property audit covers all their input checks.
        for regime in ('u', 'ef', 'eg', 'eh'):
            variant = configs[f'{dataset}_{regime}']
            audit_path = ROOT/variant['data_audit']
            if not args.verify_only:
                derived = dict(report)
                derived['config_sha256'] = hashlib.sha256(json.dumps(variant, sort_keys=True).encode()).hexdigest()
                derived['integrity_audit_reused_from'] = {'path': str(report_path), 'sha256': file_hash(report_path),
                    'scope': 'Identical input files, source configs, budgets, integrity and split rules; fewer target properties'}
                audit_path.parent.mkdir(parents=True, exist_ok=True)
                audit_path.write_text(json.dumps(derived, indent=2), encoding='utf-8')
            check_training_gate(variant, input_paths['train'], input_paths['val'], ROOT)
        reports[dataset] = {'counts': {s: r['records'] for s, r in report['splits'].items()},
                            'all_property_audit_sha256': file_hash(report_path)}
        print(json.dumps({'dataset': dataset, 'passed': True, **reports[dataset]}), flush=True)
    output = ROOT/'artifacts/main15/preparation.json'
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({'passed': True, 'datasets': reports,
        'configs': {name: file_hash(directory/f'{name}.yaml') for name in configs},
        'scope': 'Source-reference splits and their MP20+C2DB union; cross-split source/hash candidates audited, '
                 'not exhaustive prototype separation; no training performed.'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
