"""Real, site-local internal representations (not quantum spinor wavefunctions).

Explicit spin matrices are complete representations, including time reversal.
Otherwise MSG moments use epsilon * det(Q) * Q with Q in Cartesian coordinates.
Missing moment labels are represented by None, never fabricated zero targets.
"""
import torch


def spin_representation(operation, lattice):
    if operation.spin is not None:
        return lattice.new_tensor(operation.spin)
    r = lattice.new_tensor(operation.rotation)
    q = lattice @ r @ torch.linalg.inv(lattice)
    return (-1 if operation.time_reversal else 1) * torch.linalg.det(q) * q


def moment_constraints(spec, permutations, lattice):
    n = len(permutations[0])
    eye = torch.eye(3, device=lattice.device, dtype=lattice.dtype)
    rows = []
    for op, permutation in zip(spec.operations, permutations):
        d = spin_representation(op, lattice)
        for i, j in enumerate(permutation):
            block = lattice.new_zeros((3, 3*n))
            block[:, 3*j:3*j+3] += eye
            block[:, 3*i:3*i+3] -= d
            rows.append(block)
    for generator in spec.spin_generators:
        g = lattice.new_tensor(generator)
        for i in range(n):
            block = lattice.new_zeros((3, 3*n))
            block[:, 3*i:3*i+3] = g
            rows.append(block)
    return torch.cat(rows) if rows else lattice.new_zeros((0, 3*n))


def project_moments(moments, spec, permutations, lattice):
    if moments.shape != (len(permutations[0]), 3):
        raise ValueError('Moment array must follow the compiled atomic ordering')
    if not moments.numel():
        return moments.clone()
    c = moment_constraints(spec, permutations, lattice)
    if not c.numel() or float(c.detach().abs().max()) < 1e-8:
        return moments
    flat = moments.reshape(-1)
    return (flat - torch.linalg.pinv(c, rtol=1e-6) @ (c @ flat)).reshape_as(moments)


def moment_residual(moments, spec, permutations, lattice):
    c = moment_constraints(spec, permutations, lattice)
    return (c @ moments.reshape(-1)).abs().max() if c.numel() else moments.new_zeros(())


def allowed_moment_basis(spec, permutations, lattice, tolerance=1e-7):
    """Return the complete noncollinear allowed space, including spin-only generators."""
    c = moment_constraints(spec, permutations, lattice)
    dimension = c.shape[1]
    if not c.numel():
        return torch.eye(dimension, device=lattice.device, dtype=lattice.dtype)
    _, singular, vh = torch.linalg.svd(c, full_matrices=True)
    rank = int((singular > tolerance).sum())
    return vh[rank:].T
