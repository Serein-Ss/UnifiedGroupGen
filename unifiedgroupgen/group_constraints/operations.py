"""Nonmagnetic affine-operation closure and fixed-locus orbit atlas.

Only general-position OPERATIONS are read by the optional catalog adapter.
Special orbit charts are computed from fixed loci, never copied from a table.
Coordinates are fractional in periodic directions and real otherwise.
"""
from fractions import Fraction
from functools import lru_cache
from itertools import product
from math import lcm
import numpy as np
from sympy import Matrix
from sympy.matrices.normalforms import hermite_normal_form
from unifiedgroupgen.group_constraints.compiler import GROUP_PERIODIC, ROD_HEX_FRAME, catalog_operations
from unifiedgroupgen.group_constraints.unified import Operation, OrbitTemplate, SymmetrySpec, _matrix


def _fraction(x, denominator=96):
    value = Fraction(float(x)).limit_denominator(denominator)
    if abs(float(value)-float(x)) > 1e-9:
        raise ValueError('Operation coefficients must be rational with denominator at most 96')
    return value


def _key(op, rank):
    r = tuple(_fraction(x) for row in op.rotation for x in row)
    t = tuple(_fraction(x) % 1 if i < rank else _fraction(x) for i, x in enumerate(op.translation))
    return r+t


def _compose(a, b, rank):
    r = tuple(sum(a[3*i+k]*b[3*k+j] for k in range(3)) for i in range(3) for j in range(3))
    t = [sum(a[3*i+k]*b[9+k] for k in range(3))+a[9+i] for i in range(3)]
    return r+tuple(x % 1 if i < rank else x for i, x in enumerate(t))


def _change_frame(operations, frame):
    inverse = np.linalg.inv(frame)
    return tuple(Operation(_matrix(frame @ np.asarray(op.rotation) @ inverse),
                           tuple(frame @ op.translation)) for op in operations)


def _rod_rationalization(operations):
    """Find a rational chart for the supported Cartesian rod operations."""
    for frame in (np.eye(3), np.asarray(ROD_HEX_FRAME)):
        converted = _change_frame(operations, np.linalg.inv(frame))
        try:
            for op in converted:
                _key(op, 1)
        except ValueError:
            continue
        return frame, converted
    raise ValueError('Rod operations must be rational in the Cartesian or standard 120-degree transverse frame')


def _closure_in_chart(operations, kind, max_operations):
    if kind not in GROUP_PERIODIC:
        raise ValueError('The automatic nonmagnetic compiler supports space, layer, plane and rod')
    rank = sum(GROUP_PERIODIC[kind])
    validated = []
    for value in operations:
        op = value if isinstance(value, Operation) else Operation(**value)
        if op.spin is not None or op.time_reversal:
            raise ValueError('Automatic spatial compilation currently requires nonmagnetic operations')
        r, t = np.asarray(op.rotation, float), np.asarray(op.translation, float)
        if r.shape != (3, 3) or t.shape != (3,) or not np.isfinite(r).all() or not np.isfinite(t).all():
            raise ValueError('Expected finite 3x3 rotations and length-three translations')
        if (not np.allclose(r[:rank, :rank], np.round(r[:rank, :rank]))
                or not np.isclose(abs(np.linalg.det(r[:rank, :rank])), 1)
                or not np.allclose(r[:rank, rank:], 0) or not np.allclose(r[rank:, :rank], 0)
                or not np.allclose(r[rank:, rank:].T@r[rank:, rank:], np.eye(3-rank))):
            raise ValueError('Operations must preserve the adapted periodic lattice and nonperiodic metric')
        if kind == 'plane' and (not np.allclose(t[2], 0) or not np.allclose(r[2], [0, 0, 1])):
            raise ValueError('Plane operations must preserve z=0 with the identity normal action')
        validated.append(op)
    frame = None
    if kind == 'rod':
        frame, validated = _rod_rationalization(validated)
    supplied = [_key(op, rank) for op in validated]
    identity = tuple(Fraction(int(i == j)) for i in range(3) for j in range(3))+(Fraction(0),)*3
    members, generators = {identity}, []
    for candidate in sorted(set(supplied)):
        if candidate in members:
            continue
        generators.append(candidate)
        queue = list(members)
        for current in queue:
            for generator in generators:
                new = _compose(current, generator, rank)
                if new not in members:
                    if len(members) >= max_operations:
                        raise ValueError('Operations do not close into a supported finite crystallographic quotient')
                    members.add(new); queue.append(new)
    closed = tuple(Operation(_matrix(np.array(k[:9], float).reshape(3, 3)), tuple(float(x) for x in k[9:]))
                   for k in sorted(members))
    return closed, frame


def operation_closure(operations, kind, max_operations=384):
    """Deterministic exact closure modulo the reference translation lattice."""
    closed, frame = _closure_in_chart(operations, kind, max_operations)
    return _change_frame(closed, frame) if frame is not None else closed


def _clean(values):
    return tuple(float(_fraction(x, 1000000)) for x in np.asarray(values).ravel())


@lru_cache(maxsize=2048)
def _translation_lattice(rows, rank):
    c = np.asarray(rows, float).reshape(-1, 3)
    scales = [lcm(*(_fraction(x, 1000000).denominator for x in row)) for row in c]
    integer = np.rint(c[:, :rank]*np.array(scales)[:, None]).astype(int)
    h = np.asarray(hermite_normal_form(Matrix(integer)), float).reshape(len(c), -1)
    lattice = h/np.array(scales)[:, None]
    indices = tuple(Matrix(h.T.astype(int)).rref()[1]) if h.shape[1] else ()
    return lattice, indices


def _flat(c, d, rank, canonical=True):
    """RREF affine flat; quotient offsets by lattice translations via HNF."""
    a = np.column_stack((np.asarray(c, float).reshape(-1, 3), np.asarray(d, float)))
    pivot = 0
    for column in range(3):
        candidates = np.flatnonzero(np.abs(a[pivot:, column]) > 1e-8)
        if not len(candidates):
            continue
        row = pivot+int(candidates[0])
        a[[pivot, row]] = a[[row, pivot]]
        a[pivot] /= a[pivot, column]
        for j in range(len(a)):
            if j != pivot:
                a[j] -= a[j, column]*a[pivot]
        pivot += 1
        if pivot == len(a):
            break
    if np.any(np.abs(a[pivot:, 3]) > 1e-7):
        return None
    c, d = a[:pivot, :3], a[:pivot, 3]
    if pivot and canonical:
        lattice, indices = _translation_lattice(_clean(c), rank)
        if indices:
            coefficients = np.linalg.solve(lattice[list(indices)], d[list(indices)])
            d = d-lattice@np.floor(coefficients+1e-8)
    return _clean(c), _clean(d)


@lru_cache(maxsize=8192)
def _branches(flat, rank):
    """All translated lifts that can meet the closed reference periodic cell."""
    c = np.asarray(flat[0], float).reshape(-1, 3)
    d = np.asarray(flat[1], float)
    if not len(c):
        return (flat,)
    lattice, indices = _translation_lattice(flat[0], rank)
    if not indices:
        return (flat,)
    inverse = np.linalg.inv(lattice[list(indices)])
    matrix = inverse@c[list(indices), :rank]
    shift = inverse@d[list(indices)]
    low = np.ceil(np.minimum(matrix, 0).sum(1)-shift-1e-8).astype(int)
    high = np.floor(np.maximum(matrix, 0).sum(1)-shift+1e-8).astype(int)
    return tuple((flat[0], _clean(d+lattice@np.asarray(n)))
                 for n in product(*(range(a, b+1) for a, b in zip(low, high))))


def _chart(flat):
    c = np.asarray(flat[0], float).reshape(-1, 3)
    d = np.asarray(flat[1], float)
    pivots = [int(np.flatnonzero(np.abs(row) > 1e-8)[0]) for row in c]
    free = [i for i in range(3) if i not in pivots]
    b, a = np.zeros(3), np.zeros((3, len(free)))
    b[pivots] = d
    a[free, np.arange(len(free))] = 1
    if pivots:
        a[pivots] = -c[:, free]
    return a, b


@lru_cache(maxsize=512)
def _atlas(operations, kind):
    rank = sum(GROUP_PERIODIC[kind])
    base_c = np.array([[0, 0, 1.]]) if kind == 'plane' else np.zeros((0, 3))
    base_d = np.zeros(len(base_c))
    base = _flat(base_c, base_d, rank)
    primitives = set()
    for op in operations:
        c = np.asarray(op.rotation)-np.eye(3)
        t = np.asarray(op.translation)
        low = np.ceil(np.minimum(c[:rank, :rank], 0).sum(1)+t[:rank]-1e-8).astype(int)
        high = np.floor(np.maximum(c[:rank, :rank], 0).sum(1)+t[:rank]+1e-8).astype(int)
        for n in product(*(range(a, b+1) for a, b in zip(low, high))):
            rhs = np.pad(np.asarray(n, float), (0, 3-rank))-t
            flat = _flat(np.vstack((base_c, c)), np.r_[base_d, rhs], rank)
            if flat is not None and len(flat[1]) > len(base[1]):
                primitives.add(flat)
    lifted = tuple(branch for flat in sorted(primitives) for branch in _branches(flat, rank))
    flats, queue = {base}, [base]
    for current in queue:
        if len(current[1]) == 3:
            continue
        for left in _branches(current, rank):
            for right in lifted:
                candidate = _flat(np.r_[left[0], right[0]].reshape(-1, 3), np.r_[left[1], right[1]], rank)
                if candidate is not None and candidate not in flats:
                    flats.add(candidate); queue.append(candidate)
    # Quotient connected flats by spatial group action, not by Wyckoff labels.
    representatives, seen = [], set()
    for flat in sorted(flats, key=lambda f: (-len(f[1]), f)):
        if flat in seen:
            continue
        c = np.asarray(flat[0]).reshape(-1, 3)
        for op in operations:
            transformed = c@np.linalg.inv(np.asarray(op.rotation))
            seen.add(_flat(transformed, np.asarray(flat[1])+transformed@op.translation, rank))
        representatives.append(flat)
    templates = []
    for flat in representatives:
        a, b = _chart(flat)
        maps = {}
        for op in operations:
            r, t = np.asarray(op.rotation), np.asarray(op.translation)
            u, v = r@a, r@b+t
            v[:rank] %= 1
            key = (_clean(u), _clean(v))
            maps[key] = (u, v)
        arrays = [maps[key] for key in sorted(maps)]
        templates.append(OrbitTemplate('o'+str(len(templates)).zfill(3),
            tuple(tuple(tuple(float(x) for x in row) for row in u) for u, _ in arrays),
            tuple(tuple(float(x) for x in v) for _, v in arrays)))
    return tuple(templates)


def compile_operations(operations, kind='space'):
    """Compile generators or redundant operation sets without any orbit table."""
    closed, frame = _closure_in_chart(operations, kind, 384)
    templates = _atlas(closed, kind)
    if frame is not None:
        closed = _change_frame(closed, frame)
        converted = []
        for template in templates:
            a, b = template.arrays()
            converted.append(OrbitTemplate(template.letter,
                             tuple(_matrix(block) for block in frame @ a), _matrix(b @ frame.T)))
        templates = tuple(converted)
    return SymmetrySpec(kind, 0, GROUP_PERIODIC[kind], 2 if kind == 'plane' else 3,
                        closed, templates)


def compile_request(data):
    """Required operations only; never silently ignore unsupported conditions."""
    if data.get('forbidden') or data.get('magnetic') or data.get('spin_generators'):
        raise ValueError('Automatic requests currently support required nonmagnetic spatial operations, not exclusions or spin groups')
    if 'operations' not in data and 'generators' not in data:
        raise ValueError('Supply an explicit operations or generators array (empty means identity)')
    if 'operations' in data and 'generators' in data:
        raise ValueError('Supply either operations or generators, not both')
    spec = compile_operations(data.get('generators', data.get('operations')), data['kind'])
    if 'periodic' in data and tuple(data['periodic']) != spec.periodic:
        raise ValueError('Periodic mask contradicts the requested group family')
    if 'dimension' in data and data['dimension'] != spec.dimension:
        raise ValueError('Dimension contradicts the requested group family')
    return spec


@lru_cache(maxsize=512)
def operation_spec(kind, number):
    """Number is an adapter only: read general operations, discard all orbit rows."""
    operations = (Operation(_matrix(r), tuple(t)) for r, t in catalog_operations(kind, number))
    return compile_operations(operations, kind)
