"""Export current source-CIF audit tables without trained-model results."""

from unifiedgroupgen.common.configuration import project_root, resolve_path
import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path

ROOT = project_root()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', default='data/dataset/audits/source_integrity_v1')
    args = parser.parse_args()
    folder = resolve_path(ROOT/args.audit)
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if not summary['complete'] or not summary['passed']:
        raise ValueError('No complete accounted audit available')
    rows = [json.loads(s) for s in (folder/'repair.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    out = resolve_path(ROOT/'assets/source_integrity')
    out.mkdir(parents=True, exist_ok=True)
    public_rows = []
    for r in rows:
        old, new = r.get('old', {}), r.get('new', {})
        public_rows.append({'dataset': r['dataset'], 'id': r['id'], 'source_split': r['source_split'],
            'composition_conserved_before': old.get('composition_conserved'),
            'geometry_equivalent_before': old.get('geometry_equivalent'),
            'old_max_displacement_A': old.get('max_displacement_angstrom'),
            'old_measure_per_atom_relative_error': old.get('measure_per_atom_relative_error'),
            'new_max_displacement_A': new.get('max_displacement_angstrom'),
            'new_measure_per_atom_relative_error': new.get('measure_per_atom_relative_error'),
            'acceptance_tolerance_A': old.get('tolerance_angstrom'),
            'source_atoms': old.get('source_atoms'), 'old_encoded_atoms': old.get('encoded_atoms'),
            'new_encoded_atoms': new.get('encoded_atoms'),
            'source_composition': json.dumps(old.get('source_composition'), sort_keys=True),
            'old_encoded_composition': json.dumps(old.get('encoded_composition'), sort_keys=True),
            'outcome': 'quarantined' if 'error' in r else 'repaired' if r['affected'] else 'unchanged',
            'reason': r.get('error', old.get('failure', '')), 'source_cif_sha256': r['source_cif_sha256']})
    with gzip.open(out/'all_records.csv.gz', 'wt', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(public_rows[0])); writer.writeheader(); writer.writerows(public_rows)
    with (out/'affected_records.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(public_rows[0])); writer.writeheader()
        writer.writerows(r for r in public_rows if r['outcome'] != 'unchanged')
    summary['software'] = {p: importlib.metadata.version(p) for p in ('pymatgen', 'pyxtal', 'spglib', 'numpy')}
    summary['implementation_sha256'] = {p: hashlib.sha256((resolve_path(ROOT/p)).read_bytes()).hexdigest() for p in (
        'unifiedgroupgen/tests/source_integrity.py', 'unifiedgroupgen/pl_data/encoding.py', 'unifiedgroupgen/tests/workflows/audit_source_integrity.py')}
    for key in ('model_training_affected', 'models_require_clean_retraining',
                'clean_revision_models_trained', 'joint_historical_split_impact'):
        summary.pop(key, None)
    summary['scope'] += ' Numerical/tolerance-level agreement is not exact raw-coordinate identity. C2DB sources are verified recovered CIFs; the malformed original CSV is not treated as a valid fractional CIF.'
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'verified':summary['verified_records'],
                      'quarantined':summary['quarantined_records'],'output':str(out)}),flush=True)


if __name__ == '__main__':
    main()
