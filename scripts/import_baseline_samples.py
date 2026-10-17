"""Import baseline CIFs with an explicit complete attempt manifest for the same metric code."""

import argparse
import json
from pathlib import Path
import shutil
import sys
import warnings

import pandas as pd
from pymatgen.io.cif import CifParser

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.cli import root_path, artifact_path
from unifiedgroupgen.standard_metrics import layer_structure
from unifiedgroupgen.training import file_hash


def convert(manifest, dataset, count, output, targets):
    frame = pd.read_csv(manifest)
    if len(frame) != count or sorted(frame['index']) != list(range(count)):
        raise ValueError('Manifest must include every attempted index, including failed outputs')
    if not frame['status'].isin(['generated', 'failed']).all():
        raise ValueError('Baseline status must be generated or failed')
    kind = 'layer' if dataset == 'c2db_51' else 'space'
    rows = []
    for item in frame.sort_values('index').to_dict('records'):
        target = item.get('target_group')
        target = int(target) if pd.notna(target) else None
        row = {'index': int(item['index']), 'status': item['status'], 'properties_requested': targets,
               'descriptor': {'kind': kind, 'number': target}, 'validation': {}, 'generator': 'external_baseline'}
        if item['status'] == 'generated':
            source = Path(item['cif'])
            source = source if source.is_absolute() else manifest.parent/source
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    structure = CifParser(source, frac_tolerance=0).parse_structures(primitive=False)[0]
                if kind == 'layer':
                    structure = layer_structure(structure)
                destination = output/f'sample_{int(item["index"]):05d}.cif'
                if kind == 'space':
                    shutil.copyfile(source, destination)
                else:
                    structure.to(filename=str(destination))
                row['input_cif_sha256'] = file_hash(source)
            except (ValueError, RuntimeError, OSError, KeyError) as error:
                row.update(status='failed', reason=f'Baseline CIF import: {error}')
        else:
            row['reason'] = 'Declared failed in original baseline attempt manifest'
        rows.append(row)
    with (output/'samples.jsonl').open('w', encoding='utf-8') as stream:
        for row in rows:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    (output/'import.json').write_text(json.dumps({'manifest_sha256': file_hash(manifest), 'dataset': dataset,
        'attempted': count, 'imported': sum(r['status'] == 'generated' for r in rows),
        'scope': 'Complete original attempt accounting is user-supplied. Import does not establish matched '
                 'training splits, baseline reproducibility or fairness. No target group means exact-target '
                 'symmetry is not applicable; --reidentify can still report the recovered group.'}, indent=2), encoding='utf-8')
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, help='CSV index,status,cif,optional target_group')
    parser.add_argument('--dataset', required=True, choices=['mp_20', 'mpts_52', 'c2db_51'])
    parser.add_argument('--count', required=True, type=int)
    parser.add_argument('--properties', default='{}', help='Requested targets only, never evaluated values')
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    output = artifact_path(args.output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty baseline import directory')
    output.mkdir(parents=True, exist_ok=True)
    rows = convert(root_path(args.manifest), args.dataset, args.count, output, json.loads(args.properties))
    print(json.dumps({'attempted': len(rows), 'imported': sum(r['status'] == 'generated' for r in rows)}))


if __name__ == '__main__':
    main()
