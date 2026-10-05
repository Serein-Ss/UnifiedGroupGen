"""Recover missing C2DB cell matrices by UID, verifying against every original CSV row."""

import argparse
import hashlib
import json
from pathlib import Path
import time
import urllib.request
from urllib.parse import quote
import numpy as np
import pandas as pd
from pymatgen.core import Structure
from pymatgen.io.cif import CifWriter
from scipy.optimize import linear_sum_assignment
from unifiedgroupgen.cli import artifact_path, root_path
from unifiedgroupgen.data import parse_cif
from unifiedgroupgen.training import file_hash


def atoms_from_database(database, row):
    """Use an exact UID/ASE unique_id match; never substitute a similar structure."""
    try:
        entry = database.get(uid=str(row['uid']))
    except KeyError:
        if not isinstance(row.get('unique_id'), str) or not row['unique_id']:
            raise ValueError(f"Original unique_id is missing for {row['uid']}")
        entry = database.get(unique_id=row['unique_id'])
    atoms = entry.toatoms()
    if 'energy' not in entry:
        raise ValueError(f"Original ASE entry has no energy for {row['uid']}")
    result = {'cell': atoms.cell.array.tolist(), 'numbers': atoms.numbers.tolist(),
              'positions': atoms.positions.tolist(), 'pbc': atoms.pbc.tolist(),
              'energy': float(entry.energy)}
    verify_structure(row, result)
    return result, int(entry.id)


def verify_structure(row, atoms, tolerance=1e-6):
    """Require unchanged species, absolute Cartesian positions, cell shape, PBC and energy."""
    original = parse_cif(row['cif'], 'cartesian_in_fractional_fields')
    restored = Structure(atoms['cell'], atoms['numbers'], atoms['positions'],
                         coords_are_cartesian=True, to_unit_cell=False)
    if atoms['pbc'] != [True, True, False]:
        raise ValueError('Official structure has unexpected PBC')
    if sorted(original.atomic_numbers) != sorted(restored.atomic_numbers):
        raise ValueError('Official species/counts differ from the CSV')
    if not np.allclose(original.lattice.abc, restored.lattice.abc, atol=tolerance, rtol=0) or \
       not np.allclose(original.lattice.angles, restored.lattice.angles, atol=tolerance, rtol=0):
        raise ValueError('Official cell shape differs from the CSV')
    if abs(float(row['energy'])-float(atoms['energy'])) > tolerance:
        raise ValueError('Official energy differs from the CSV')
    residual = 0.0
    for element in set(original.atomic_numbers):
        old = original.cart_coords[np.array(original.atomic_numbers) == element]
        new = restored.cart_coords[np.array(restored.atomic_numbers) == element]
        distance = np.linalg.norm(old[:, None]-new[None], axis=-1)
        first, second = linear_sum_assignment(distance)
        residual = max(residual, float(distance[first, second].max()))
    if residual > tolerance:
        raise ValueError(f'Official Cartesian coordinates differ from the CSV: {residual} Angstrom')
    return restored, residual


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='../data/c2db_51')
    parser.add_argument('--output', default='artifacts/source_recovery/c2db_51')
    parser.add_argument('--ase-db', help='Original ASE database; supplies exact missing records without changing source CSVs')
    parser.add_argument('--limit', type=int, help='Explicit per-split diagnostic limit; never marks full recovery')
    args = parser.parse_args()
    destination = artifact_path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    cache = destination/'official_json'
    cache.mkdir(exist_ok=True)
    database, database_source = None, None
    original_databases, checked_databases = {}, {}
    if args.ase_db:
        from ase.db import connect
        database_path = root_path(args.ase_db).resolve()
        if not database_path.is_file():
            raise ValueError(f'Original ASE database is missing: {database_path}')
        database_source = {'path': str(database_path), 'sha256': file_hash(database_path)}
        checked_databases[str(database_path)] = database_source['sha256']
        database = connect(database_path, use_lock_file=False)
    reports = {}
    for split in ('train', 'val', 'test'):
        source = root_path(args.source)/f'{split}.csv'
        frame = pd.read_csv(source)
        source_hash = file_hash(source)
        rows, rejected = [], []
        count = len(frame) if args.limit is None else min(len(frame), args.limit)
        for index, row in frame.iloc[:count].iterrows():
            uid = str(row['uid'])
            url = f'https://c2db.fysik.dtu.dk/material/{quote(uid, safe="")}/download/json'
            cached = cache/(hashlib.sha256(uid.encode()).hexdigest()+'.json')
            try:
                if not cached.exists() and database is not None:
                    try:
                        atoms, database_row = atoms_from_database(database, row)
                    except KeyError:
                        pass  # This UID is absent locally; try its public download next.
                    else:
                        content = {'1': atoms, '_source': {**database_source, 'kind': 'original_ase_database',
                                                          'row': database_row}}
                        temporary = cached.with_suffix('.json.tmp')
                        temporary.write_text(json.dumps(content), encoding='utf-8')
                        temporary.replace(cached)
                if not cached.exists():
                    for attempt in range(3):
                        try:
                            request = urllib.request.Request(url, headers={'User-Agent': 'UnifiedGroupGen research data verification'})
                            with urllib.request.urlopen(request, timeout=20) as response:
                                payload = response.read(1024*1024)
                            json.loads(payload)
                            temporary = cached.with_suffix('.json.tmp')
                            temporary.write_bytes(payload)
                            temporary.replace(cached)
                            break
                        except (OSError, ValueError):
                            if attempt == 2:
                                raise
                            time.sleep(2**attempt)
                    time.sleep(0.5)
                content = json.loads(cached.read_text(encoding='utf-8'))
                entries = [value for value in content.values() if isinstance(value, dict) and 'positions' in value]
                if len(entries) != 1:
                    raise ValueError('Download did not contain exactly one structure')
                restored, residual = verify_structure(row, entries[0])
                provenance = content.get('_source')
                if provenance:
                    if provenance['path'] not in checked_databases:
                        checked_databases[provenance['path']] = file_hash(provenance['path'])
                    if provenance['kind'] != 'original_ase_database' or checked_databases[provenance['path']] != provenance['sha256']:
                        raise ValueError('Cached original ASE source has changed')
                    original_databases[provenance['path']] = provenance['sha256']
                    url = f"ase-db:{provenance['path']}#row={provenance['row']}"
                record = row.to_dict()
                record.update({'cif': str(CifWriter(restored)), 'official_structure_url': url,
                               'official_structure_sha256': file_hash(cached),
                               'original_csv_sha256': source_hash, 'original_source_row': int(index),
                               'source_verification': 'UID + species + Cartesian positions + cell shape + energy',
                               'cartesian_match_residual_angstrom': residual})
                if provenance:
                    record['original_ase_source'] = json.dumps(provenance, sort_keys=True)
                rows.append(record)
            except (OSError, ValueError, KeyError, TypeError) as error:
                rejection = {'row': int(index), 'uid': uid, 'reason': str(error)}
                rejected.append(rejection)
                print(json.dumps({'event': 'source_rejection', 'split': split, **rejection}), flush=True)
            if (index+1) % 25 == 0 or index+1 == count:
                print(json.dumps({'split': split, 'processed': int(index+1), 'input': count,
                                  'accepted': len(rows), 'rejected': len(rejected)}), flush=True)
        report = {'source': str(source), 'source_sha256': source_hash, 'input': count,
                  'source_rows': len(frame), 'full_source': args.limit is None,
                  'accepted': len(rows), 'rejected': rejected, 'passed': not rejected and args.limit is None}
        temporary = destination/f'{split}.csv.tmp'
        pd.DataFrame(rows).to_csv(temporary, index=False)
        temporary.replace(destination/f'{split}.csv')
        report['recovered_sha256'] = file_hash(destination/f'{split}.csv')
        (destination/f'{split}.recovery.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        reports[split] = report
    passed = all(r['passed'] for r in reports.values())
    (destination/'recovery.json').write_text(json.dumps({'passed': passed, 'splits': reports,
                                                         'original_databases': original_databases}, indent=2), encoding='utf-8')
    print(json.dumps({'passed': passed, 'output': str(destination)}), flush=True)
    if any(r['rejected'] for r in reports.values()):
        raise SystemExit(2)


if __name__ == '__main__':
    main()
