"""Invert orbit expansion: full positions to independent Wyckoff coordinates."""
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment
from .compiler import Descriptor, OrbitSpec, group_table, wp_arrays

def _delta(a, b, periodic):
    diff = np.asarray(a)-np.asarray(b)
    diff[..., periodic] -= np.round(diff[..., periodic])
    return diff


def _norm(delta, physical_lattice=None):
    return np.linalg.norm(delta if physical_lattice is None else delta@physical_lattice.T, axis=-1)


def operation_residual(coords, elements, operations, periodic, physical_lattice=None):
    residual = 0.0
    for op in operations:
        transformed = coords @ op.rotation_matrix.T + op.translation_vector
        for element in np.unique(elements):
            ids = np.flatnonzero(elements == element)
            costs = _norm(_delta(transformed[ids, None], coords[None, ids], periodic), physical_lattice)
            rows, cols = linear_sum_assignment(costs)
            residual = max(residual, float(np.max(costs[rows, cols])))
    return residual


def _q_for_point(a, b, point, periodic, tolerance, physical_lattice=None):
    images = itertools.product(*[(-1, 0, 1) if flag else (0,) for flag in periodic])
    for image in images:
        rhs = point-b+np.array(image)
        q = np.linalg.lstsq(a, rhs, rcond=None)[0]
        if _norm(a@q-rhs, physical_lattice) < tolerance:
            yield q


def decompose_orbits(kind, number, coords, elements, tolerance=1e-4, physical_lattice=None):
    """Exact-to-tolerance finite orbit matching; failures are recorded, not assigned P1."""
    periodic = np.array([True, True, kind == "space"])
    remaining = set(range(len(coords)))
    rows = sorted(group_table(kind, number), key=lambda w: (w.multiplicity, w.letter))
    orbits, parameters = [], []
    while remaining:
        i = min(remaining)
        candidates = np.array(sorted(j for j in remaining if elements[j] == elements[i]))
        found = False
        for wp in rows:
            a, b, axes = wp_arrays(kind, number, wp.letter)
            if len(a) > len(candidates):
                continue
            for j in range(len(a)):
                for q in _q_for_point(a[j], b[j], coords[i], periodic, tolerance, physical_lattice):
                    generated = b+np.einsum('iaq,q->ia', a, q)
                    if len(a) > 1:
                        pair = _norm(_delta(generated[:, None], generated[None], periodic), physical_lattice)
                        np.fill_diagonal(pair, np.inf)
                        if np.min(pair) < tolerance:
                            continue
                    costs = _norm(_delta(generated[:, None], coords[None, candidates], periodic), physical_lattice)
                    matched, columns = linear_sum_assignment(costs)
                    if len(matched) == len(a) and np.max(costs[matched, columns]) < tolerance:
                        q[np.array(periodic[axes], dtype=bool)] %= 1
                        orbits.append(OrbitSpec(wp.letter, int(elements[i])))
                        parameters.extend(q.tolist())
                        remaining.difference_update(candidates[columns].tolist())
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            raise ValueError(f"Cannot match atom {i} to a {kind} {number} orbit in this setting/origin")
    return Descriptor(kind, number, tuple(orbits)), np.array(parameters)
