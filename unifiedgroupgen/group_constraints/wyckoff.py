"""Standard display labels for operation-compiled orbit templates."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import csv
from functools import lru_cache
from hashlib import sha256
import json
from pathlib import Path


TABLE = Path(__file__).with_name('wyckoff_templates.csv')
FIELDS = ('kind', 'group', 'setting', 'wyckoff_letter', 'template_id',
          'multiplicity', 'coordinate_dof', 'compiled_spec_sha256')


@lru_cache(maxsize=512)
def _spec_digest(spec):
    return sha256(json.dumps(spec.to_dict(), sort_keys=True, separators=(',', ':')).encode()).hexdigest()


@lru_cache(maxsize=1)
def _table():
    matches = {}
    with TABLE.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            key = (row['kind'], int(row['group']), row['compiled_spec_sha256'])
            matches.setdefault(key, {})[row['template_id']] = row['wyckoff_letter']
    return matches


def orbit_labels(spec, letters, group=None):
    """Display the canonical requested setting; unspecified/custom groups keep template IDs."""
    number = spec.number if group is None else group
    if not letters or not number or not 1 <= number <= {'space': 230, 'layer': 80, 'rod': 75}.get(spec.kind, 0):
        return list(letters)
    matches = _table().get((spec.kind, number, _spec_digest(spec)))
    if matches is None and spec.number == number:
        from unifiedgroupgen.group_constraints.unified import catalog_spec
        if spec == catalog_spec(spec.kind, number):
            matches = {letter: letter for letter in letters}
    if matches is None:
        return list(letters)
    return [f'{spec.template(letter).multiplicity}{matches[letter]}' if letter in matches else letter
            for letter in letters]


def record_orbit_labels(record, state):
    """Cached descriptors supply standard provenance; explicit custom state keeps its IDs."""
    descriptor = record.get('descriptor') or {}
    group = descriptor.get('number') if 'public_state' not in record and 'symmetry' not in record else 0
    return orbit_labels(state.spec, state.letters, group)


def event_orbit_label(spec, event, group=None):
    letter = (event or {}).get('action', {}).get('letter')
    return orbit_labels(spec, (letter,), group)[0] if letter else None


def _by_geometry(key):
    # Compare complete affine orbit families modulo the periodic lattice.
    from collections import Counter
    import numpy as np
    from scipy.linalg import null_space
    from unifiedgroupgen.group_constraints.compiler import group_table
    from unifiedgroupgen.group_constraints.operations import operation_spec, _flat, _rod_rationalization
    from unifiedgroupgen.group_constraints.unified import catalog_spec

    kind, number = key
    catalog = catalog_spec(kind, number)
    compiled = operation_spec(kind, number)
    inverse = np.linalg.inv(_rod_rationalization(compiled.operations)[0]) if kind == 'rod' else None

    def signature(template):
        a, b = template.arrays()
        if inverse is not None:
            a, b = inverse @ a, b @ inverse.T
        flats = []
        for basis, offset in zip(a, b):
            equations = null_space(basis.T, rcond=1e-8).T
            flat = _flat(equations, equations @ offset, compiled.rank)
            if flat is None:
                raise ValueError(f'Invalid affine chart for {key}/{template.letter}')
            flats.append(flat)
        return template.dof, tuple(sorted(Counter(flats).items()))

    atlas = {}
    for item in compiled.templates:
        atlas.setdefault(signature(item), []).append(item.letter)
    setting = f'Hall {group_table(kind, number).hall_number}' if kind == 'space' else f'PyXtal standard {kind}'
    digest = _spec_digest(compiled)
    rows = []
    for wp in reversed(catalog.templates):
        matches = atlas.get(signature(wp), [])
        if len(matches) != 1:
            raise ValueError(f'Expected one compiled chart for {key}/{wp.letter}: {matches}')
        rows.append((kind, number, setting, wp.letter, matches[0], wp.multiplicity, wp.dof, digest))
    if len(set(row[4] for row in rows)) != len(rows) or len(rows) != len(compiled.templates):
        raise ValueError(f'Non-bijective atlas for {key}')
    return rows


def main():
    parser = argparse.ArgumentParser(description='Build the checked standard Wyckoff display table.')
    parser.add_argument('--output', type=Path, default=TABLE)
    parser.add_argument('--workers', type=int, default=1)
    args = parser.parse_args()
    groups = [(kind, group) for kind, count in (('space', 230), ('layer', 80), ('rod', 75)) for group in range(1, count+1)]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = []
        for index, group_rows in enumerate(pool.map(_by_geometry, groups), 1):
            rows.extend(group_rows)
            if index % 25 == 0:
                print(f'{index}/{len(groups)} groups checked', flush=True)
    with args.output.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(FIELDS)
        writer.writerows(rows)
    print(f'{len(rows)} mappings for {len(groups)} groups', flush=True)


if __name__ == '__main__':
    main()
