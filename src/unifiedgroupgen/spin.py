"""Optional collinear, unit-moment representation extension; no magnetic data training."""

import numpy as np
from scipy.linalg import null_space


def collinear_axis_basis(signs, operations, tolerance=1e-8):
    """operations = [(orthogonal spin representation D_g, atom permutation pi_g), ...].

    Magnetic time reversal can be included in D_g. Arbitrary spin representations
    must be supplied and checked by the caller; they are not inferred from an SG.
    """
    signs = np.asarray(signs)
    if not np.isin(signs, [-1, 1]).all():
        raise ValueError("Magnetic-site signs must be +/-1")
    rows = []
    for representation, permutation in operations:
        d = np.asarray(representation)
        permutation = np.asarray(permutation)
        if d.shape != (3, 3) or not np.allclose(d.T@d, np.eye(3), atol=tolerance):
            raise ValueError("Unit moments require an orthogonal spin representation")
        if sorted(permutation.tolist()) != list(range(len(signs))):
            raise ValueError("Invalid site permutation")
        rows.extend([signs[permutation[i]]*np.eye(3)-signs[i]*d for i in range(len(signs))])
    constraints = np.concatenate(rows) if rows else np.zeros((0, 3))
    if not constraints.size or np.max(np.abs(constraints)) < tolerance:
        return np.eye(3)
    return null_space(constraints, rcond=tolerance)


def sample_collinear_moments(signs, operations, seed=0, fixed_axis=None):
    basis = collinear_axis_basis(signs, operations)
    if basis.shape[1] == 0:
        raise ValueError("These signs and group actions admit no nonzero collinear axis")
    if fixed_axis is None:
        axis = basis @ np.random.default_rng(seed).normal(size=basis.shape[1])
    else:
        axis = np.asarray(fixed_axis, dtype=float)
        if not np.allclose(axis, basis@basis.T@axis, atol=1e-7):
            raise ValueError("Fixed spin axis violates the supplied group representation")
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        raise ValueError("A unit spin axis cannot be zero")
    return np.asarray(signs)[:, None] * (axis/norm)
