"""Build explicitly partial diagnostics from already downloaded and verified structures."""

import argparse
import hashlib
import json
import pandas as pd
from pymatgen.io.cif import CifWriter
from recover_c2db import verify_structure
from unifiedgroupgen.cli import artifact_path, root_path
from unifiedgroupgen.training import file_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', default='artifacts/source_recovery/c2db_official/official_json')
    parser.add_argument('--output', default='artifacts/pretrain_validation/c2db_cached')
    parser.add_argument('--limit', type=int, default=256)
    args = parser.parse_args()
    output = artifact_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    for split in ('train', 'val', 'test'):
        source = root_path(f'../data/c2db_51/{split}.csv')
        original_hash = file_hash(source)
        records = []
        for index, row in pd.read_csv(source).iterrows():
            path = root_path(args.cache)/(hashlib.sha256(str(row.uid).encode()).hexdigest()+'.json')
            if not path.exists():
                continue
            content = json.loads(path.read_text())
            entries = [v for v in content.values() if isinstance(v, dict) and 'positions' in v]
            if len(entries) != 1:
                raise ValueError(f'Unexpected official JSON for {row.uid}')
            restored, residual = verify_structure(row, entries[0])
            record = row.to_dict()
            record.update({'cif': str(CifWriter(restored)), 'official_structure_sha256': file_hash(path),
                           'original_csv_sha256': original_hash, 'original_source_row': int(index),
                           'official_structure_url': f'https://c2db.fysik.dtu.dk/material/{row.uid}/download/json',
                           'source_verification': 'UID + species + Cartesian positions + cell shape + energy',
                           'cartesian_match_residual_angstrom': residual})
            records.append(record)
            if len(records) == args.limit:
                break
        pd.DataFrame(records).to_csv(output/f'{split}.csv', index=False)
        print(json.dumps({'split': split, 'records': len(records), 'full_source': False}), flush=True)


if __name__ == '__main__':
    main()
