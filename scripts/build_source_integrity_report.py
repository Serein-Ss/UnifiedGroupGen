"""Publish the full source audit, model-impact tables and a PNG-only evidence figure."""

import argparse
import csv
import gzip
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', default='artifacts/source_integrity_20261022_v3')
    args = parser.parse_args()
    folder = ROOT/args.audit
    summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    if not summary['complete'] or not summary['passed']:
        raise ValueError('No complete accounted audit available')
    rows = [json.loads(s) for s in (folder/'repair.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
    out = ROOT/'assets/source_integrity'
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
    summary['implementation_sha256'] = {p: hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in (
        'src/unifiedgroupgen/source_integrity.py', 'src/unifiedgroupgen/data.py', 'scripts/audit_source_integrity.py')}
    summary['model_training_affected'] = {
        'M1': summary['datasets']['mp_20']['source_splits']['train']['affected'],
        'M2': summary['datasets']['mpts_52']['source_splits']['train']['affected'],
        'M3': sum(r['affected'] for r in summary['joint_historical_split_impact']['train'].values())}
    summary['models_require_clean_retraining'] = ['M1', 'M2', 'M3']
    summary['clean_revision_models_trained'] = 0
    summary['scope'] += ' Numerical/tolerance-level agreement is not exact raw-coordinate identity. C2DB sources are verified recovered CIFs; the malformed original CSV is not treated as a valid fractional CIF.'
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    names = ['mp_20', 'mpts_52', 'c2db_51']; labels = ['MP20', 'MPTS52', 'C2DB']
    values = [summary['datasets'][n] for n in names]
    plt.rcParams.update({'font.family': 'Times New Roman', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 3, figsize=(13.6, 4.4))
    x = np.arange(3)
    composition = np.array([r['composition_failed'] for r in values])
    geometry = np.array([r['geometry_failed_after_composition'] for r in values])
    axes[0].bar(x, composition, color='#c5784c', label='Composition changed')
    axes[0].bar(x, geometry, bottom=composition, color='#477da4', label='Geometry / measure mismatch')
    for i,r in enumerate(values):axes[0].text(i,r['affected']+1,f"{r['affected']}/{r['records']:,}",ha='center',fontsize=9)
    axes[0].set(title='a  Historical encodings requiring action', ylabel='Records', xticks=x, xticklabels=labels, ylim=(0,60))
    axes[0].legend(frameon=False, fontsize=8)
    repaired = np.array([r['affected']-r['errors'] for r in values]); quarantine = np.array([r['errors'] for r in values])
    axes[1].bar(x-.17,repaired,.34,color='#477da4',label='Repaired and verified')
    axes[1].bar(x+.17,quarantine,.34,color='#c5784c',label='Quarantined, retained separately')
    for i,(a,b) in enumerate(zip(repaired,quarantine)):
        axes[1].text(i-.17,a+.8,str(a),ha='center');axes[1].text(i+.17,b+.8,str(b),ha='center')
    axes[1].set(title='b  Repair disposition', ylabel='Records',xticks=x,xticklabels=labels,ylim=(0,60))
    axes[1].legend(frameon=False, fontsize=8)
    impact = summary['model_training_affected']
    axes[2].bar(list(impact),list(impact.values()),color=['#477da4','#c5784c','#55927d'])
    for i,v in enumerate(impact.values()):axes[2].text(i,v+1,str(v),ha='center')
    axes[2].set(title='c  Affected historical TRAIN records', ylabel='Records',ylim=(0,70))
    for ax in axes:ax.grid(axis='y',color='#e5e9ed',lw=.6);ax.set_axisbelow(True)
    fig.suptitle(f"Exhaustive source-CIF audit: {summary['verified_records']:,} verified outputs; {summary['quarantined_records']} quarantined inputs",fontsize=13)
    fig.subplots_adjust(left=.055,right=.99,top=.82,bottom=.15,wspace=.26)
    fig.savefig(out/'01_full_source_audit.png',dpi=260,facecolor='white');plt.close(fig)
    shutil.copy2(out/'01_full_source_audit.png',ROOT/'assets/paper/fig02_representation.png')
    print(json.dumps({'verified':summary['verified_records'],'quarantined':summary['quarantined_records'],
                      'affected_training':impact,'output':str(out)}),flush=True)


if __name__ == '__main__':
    main()
