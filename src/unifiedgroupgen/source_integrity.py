"""Independent composition and periodic-geometry checks against the source CIF.

Cell multiplicities, origins and rigid rotations may differ. Coordinates and
periodic measure may not be freely rescaled; layer vacuum is never a tolerance.
"""

from collections import Counter
from functools import reduce
from math import gcd
import hashlib
import json
import numpy as np
from pymatgen.core import Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher
from .representation_audit import compiled_chart, expand_numpy
from .symmetry import Descriptor


PROTOCOL = 'source-cif-integrity-v1'


class _LayerMatcher(StructureMatcher):
    """Forbid auxiliary-vacuum shear/repetition during 2D lattice search."""

    def _get_lattices(self, target_lattice, structure, supercell_size=1):
        for lattice, transform in super()._get_lattices(target_lattice, structure, supercell_size):
            if (np.all(transform[:2, 2] == 0) and np.all(transform[2, :2] == 0)
                    and abs(transform[2, 2]) == 1):
                yield lattice, transform


def encoding_digest(record):
    payload = {'descriptor': record['descriptor'], 'state': record['state']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def reduced_counts(numbers):
    counts = Counter(int(z) for z in numbers)
    divisor = reduce(gcd, counts.values())
    return {str(z): n // divisor for z, n in sorted(counts.items())}


def expanded_structure(record):
    chart = compiled_chart(Descriptor.from_dict(record['descriptor']))
    lattice, coords, _ = expand_numpy(chart, record['state'])
    if record['descriptor']['kind'] == 'space':
        return Structure(Lattice(lattice.T), chart.elements, coords)
    rows = lattice.T.copy()
    height = max(40.0, float(np.ptp(coords[:, 2])) + 20)
    rows[2] = [0, 0, height]
    f = coords.copy()
    f[:, 2] = (f[:, 2] - f[:, 2].mean()) / height + .5
    return Structure(Lattice(rows, pbc=(True, True, False)), chart.elements, f)


def _isolated_layer(structure, height):
    ab = structure.lattice.matrix[:2]
    normal = np.cross(*ab)
    normal /= np.linalg.norm(normal)
    cart = structure.cart_coords
    z = cart @ normal
    vacuum_period = abs(structure.lattice.matrix[2] @ normal)
    if vacuum_period:
        wrapped = z % vacuum_period
        ordered = np.sort(wrapped)
        gaps = np.diff(np.r_[ordered, ordered[0] + vacuum_period])
        cut = ordered[(int(np.argmax(gaps)) + 1) % len(ordered)]
        z = (wrapped - cut) % vacuum_period
    z -= z.mean()
    xy = cart @ ab.T @ np.linalg.inv(ab @ ab.T)
    rows = np.zeros((3, 3))
    rows[:2, :2] = np.linalg.cholesky(ab @ ab.T)
    rows[2, 2] = height
    return Structure(Lattice(rows), structure.species,
                     np.column_stack([xy % 1, z / height + .5]))


def compare_source(source, encoded, kind, tolerance):
    """Return measured equivalence under declared absolute modelling tolerance.

    A stricter numerical-identity flag is separate from tolerance-level agreement.
    Lattice matching is bounded by tol/min(periodic length); scale=False.
    In 3D, the matcher displacement is normalized by volume-per-site^(1/3).
    In 2D, we explicitly inspect matched Cartesian displacements after mapping.
    """
    source_counts = reduced_counts(source.atomic_numbers)
    encoded_counts = reduced_counts(encoded.atomic_numbers)
    report = {'protocol': PROTOCOL, 'source_composition': source_counts,
              'encoded_composition': encoded_counts,
              'composition_conserved': source_counts == encoded_counts,
              'source_atoms': len(source), 'encoded_atoms': len(encoded),
              'tolerance_angstrom': float(tolerance), 'geometry_equivalent': False,
              'numerically_identical': False, 'max_displacement_angstrom': None,
              'measure_per_atom_relative_error': None}
    if not report['composition_conserved']:
        report['failure'] = 'composition_changed'
        return report
    dim = 3 if kind == 'space' else 2
    if dim == 2:
        height = 20 * max(*source.lattice.abc[:2], *encoded.lattice.abc[:2], 4)
        source, encoded = _isolated_layer(source, height), _isolated_layer(encoded, height)
        # Two centered conventional cells of equal area need not be connected
        # by an integer basis change. Reduce their translation lattices first.
        primitive_tolerance = max(1e-6, tolerance / 10)
        source = _isolated_layer(source.get_primitive_structure(tolerance=primitive_tolerance), height)
        encoded = _isolated_layer(encoded.get_primitive_structure(tolerance=primitive_tolerance), height)
    first_measure = abs(np.linalg.det(source.lattice.matrix)) if dim == 3 else np.linalg.norm(np.cross(*source.lattice.matrix[:2]))
    second_measure = abs(np.linalg.det(encoded.lattice.matrix)) if dim == 3 else np.linalg.norm(np.cross(*encoded.lattice.matrix[:2]))
    relative = abs(second_measure / len(encoded) / (first_measure / len(source)) - 1)
    length = min(*source.lattice.abc[:dim], *encoded.lattice.abc[:dim])
    lattice_tol = max(1e-5, tolerance / length)
    report.update(measure_per_atom_relative_error=float(relative),
                  lattice_relative_tolerance=float(lattice_tol),
                  scale_normalization=False, periodic_dimension=dim)
    # A strain of one lattice_tol along each periodic axis changes the measure
    # by at most this amount. This rejects density changes even if sites match.
    if relative > (1 + lattice_tol) ** dim - 1 + 1e-8:
        report['failure'] = 'periodic_measure_changed'
        return report
    angle_tol = max(.001, np.rad2deg(lattice_tol))
    if dim == 3:
        free_length = ((source.volume / len(source)) ** (1 / 3) +
                       (encoded.volume / len(encoded)) ** (1 / 3)) / 2
        # StructureMatcher's default primitive reduction uses 0.25 Angstrom,
        # which can distort nearly higher-symmetry cells before a strict check.
        # Reduce with our declared precision instead, then disable that default.
        primitive_tolerance = max(1e-6, tolerance / 10)
        first = source.get_primitive_structure(tolerance=primitive_tolerance)
        second = encoded.get_primitive_structure(tolerance=primitive_tolerance)
        matcher = StructureMatcher(ltol=lattice_tol, stol=max(1e-6, tolerance / free_length),
                                   angle_tol=angle_tol, scale=False, primitive_cell=False, attempt_supercell=True)
        distance = matcher.get_rms_dist(first, second)
        if distance is not None:
            report['max_displacement_angstrom'] = float(distance[1] * free_length)
            report['normalized_rms_displacement'] = float(distance[0])
    else:
        if len(source) < len(encoded):
            source, encoded = encoded, source
        matcher = _LayerMatcher(ltol=lattice_tol, stol=2, angle_tol=angle_tol,
                                   scale=False, primitive_cell=False, attempt_supercell=True)
        mapped = matcher.get_s2_like_s1(source, encoded)
        if mapped is not None and len(mapped) == len(source):
            # Disallow any exchange of a periodic axis with the artificial vacuum.
            normal_preserved = np.allclose(np.abs(mapped.lattice.matrix[2]),
                                          np.abs(source.lattice.matrix[2]), atol=1e-6)
            if normal_preserved:
                delta = mapped.frac_coords - source.frac_coords
                delta[:, :2] -= np.round(delta[:, :2])
                delta[:, 2] -= delta[:, 2].mean()
                cart_delta = delta @ source.lattice.matrix
                from pymatgen.util.coord import pbc_shortest_vectors
                # Use the auxiliary 3D lattice only for the oblique xy minimum
                # image. Restore the unwrapped z displacement explicitly.
                vec = pbc_shortest_vectors(Lattice(source.lattice.matrix),
                                           np.zeros((1, 3)), delta)[0]
                vec[:, 2] = cart_delta[:, 2]
                report['max_displacement_angstrom'] = float(np.linalg.norm(vec, axis=1).max())
    displacement = report['max_displacement_angstrom']
    report['geometry_equivalent'] = displacement is not None and displacement <= tolerance * (1 + 1e-5) + 1e-8
    report['numerically_identical'] = bool(report['geometry_equivalent'] and displacement < 1e-4 and relative < 1e-5)
    if not report['geometry_equivalent']:
        report['failure'] = 'periodic_geometry_changed'
    return report


def check_record_source(source, record, tolerance):
    return compare_source(source, expanded_structure(record), record['descriptor']['kind'], tolerance)


def require_source_integrity(source, descriptor, state, tolerance):
    record = {'descriptor': descriptor.to_dict(), 'state': np.asarray(state).tolist()}
    result = check_record_source(source, record, tolerance)
    if not result['composition_conserved'] or not result['geometry_equivalent']:
        raise ValueError('Source integrity failed: ' + result.get('failure', 'unknown'))
    return result
