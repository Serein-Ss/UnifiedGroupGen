"""Model-independent structure ledgers and content-addressed evaluation inputs."""
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure


def sha256(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def read_records(path):
    records = []
    with Path(path).open(encoding='utf-8-sig') as stream:
        for line, text in enumerate(stream, 1):
            if text.strip():
                try:
                    records.append(json.loads(text))
                except ValueError as error:
                    raise ValueError(f'{path}:{line}: incomplete/invalid JSON; input was not modified') from error
    return records


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def counts_of(structure):
    return dict(sorted(Counter(int(z) for z in structure.atomic_numbers).items()))


def formula_key(counts):
    values = list(counts.values())
    divisor = int(np.gcd.reduce(values))
    return tuple((int(z), int(n)//divisor) for z, n in sorted(counts.items()))


def record_counts(record, base):
    if record.get('descriptor'):
        from .symmetry import group_table
        descriptor = record['descriptor']
        multiplicities = {w.letter: w.multiplicity for w in group_table(descriptor['kind'], descriptor['number'])}
        counts = Counter()
        for orbit in descriptor['orbits']:
            counts[int(orbit['atomic_number'])] += multiplicities[orbit['letter']]
        return dict(sorted(counts.items()))
    return counts_of(load_structure(record, base))


def inline_cif_bytes(record):
    """Validate and return the exact UTF-8 bytes of an embedded CIF."""
    text = record['cif']
    if not isinstance(text, str) or not text.strip():
        raise ValueError('Inline CIF must be a nonempty string')
    content = text.encode('utf-8')
    expected = record.get('cif_sha256')
    if expected and hashlib.sha256(content).hexdigest() != expected:
        raise ValueError('Inline CIF hash mismatch')
    return content


def load_structure(record, base):
    """Read embedded/exported CIFs, or pymatgen/array/UGG records."""
    base = Path(base)
    if 'cif' in record:
        from pymatgen.io.cif import CifParser
        return CifParser.from_str(inline_cif_bytes(record).decode('utf-8'), frac_tolerance=0).parse_structures(primitive=False)[0]
    name = record.get('cif_path')
    if name is None and 'index' in record and 'status' in record and not record.get('structure'):
        candidate = base/f"sample_{int(record['index']):05d}.cif"
        if candidate.exists():
            name = str(candidate)
        elif record.get('cif_sha256'):
            raise FileNotFoundError(f'Missing declared exported CIF: {candidate}')
    if name:
        path = Path(name)
        path = path if path.is_absolute() else base/path
        expected = record.get('cif_sha256')
        if expected and sha256(path) != expected:
            raise ValueError(f'CIF hash mismatch: {path.name}')
        from pymatgen.io.cif import CifParser
        return CifParser(path, frac_tolerance=0).parse_structures(primitive=False)[0]
    if record.get('structure'):
        return Structure.from_dict(record['structure'])
    if all(key in record for key in ('lattice', 'atomic_numbers', 'frac_coords')):
        return Structure(Lattice(record['lattice']), record['atomic_numbers'], record['frac_coords'])
    if record.get('descriptor') and 'state' in record:
        from .representation_audit import compiled_chart, expand_numpy
        from .symmetry import Descriptor
        compiled = compiled_chart(Descriptor.from_dict(record['descriptor']))
        lattice, coordinates, _ = expand_numpy(compiled, record['state'])
        return Structure(Lattice(lattice.T, pbc=tuple(compiled.periodic)), compiled.elements, coordinates)
    raise ValueError('A structure needs cif_path, a pymatgen dictionary, arrays, or descriptor/state')


class ReferenceBank:
    """Lazy training/reference structures indexed by reduced integer composition."""
    def __init__(self, paths, dataset=None):
        self.paths = [Path(p).resolve() for p in paths]
        self.rows, self.by_id, self.by_formula = [], {}, defaultdict(list)
        self._structures = {}
        for path in self.paths:
            for record in read_records(path):
                if dataset and record.get('dataset') is not None and record['dataset'] != dataset:
                    continue
                key = str(record.get('id', record.get('case_id', len(self.rows))))
                if key in self.by_id:
                    raise ValueError(f'Duplicate reference identifier: {key}')
                item = (record, path.parent)
                self.by_id[key] = item
                self.rows.append(item)
                try:
                    self.by_formula[formula_key(record_counts(record, path.parent))].append(key)
                except (ValueError, KeyError, OSError, RuntimeError):
                    # Invalid references remain in the requested denominator and are reported later.
                    pass

    def structure(self, key):
        if key not in self._structures:
            record, base = self.by_id[key]
            if len(self._structures) >= 4096:
                self._structures.pop(next(iter(self._structures)))
            self._structures[key] = load_structure(record, base)
        return self._structures[key]

    def identity(self):
        return {str(path): sha256(path) for path in self.paths}
