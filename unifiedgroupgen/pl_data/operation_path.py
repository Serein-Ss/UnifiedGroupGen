"""Lossless in-memory conversion of existing endpoints to operation-derived charts."""
from dataclasses import replace
from itertools import product
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from unifiedgroupgen.group_constraints.operations import compile_operations
from unifiedgroupgen.group_constraints.state import JointState
from unifiedgroupgen.group_constraints.unified import compile_constraints


def match_orbit(spec, coordinates, tolerance=2e-5):
    """Find a computed chart and ordering; accept only a complete set match."""
    x = np.asarray(coordinates, float).copy()
    periodic = np.asarray(spec.periodic)
    x[:, periodic] %= 1
    shifts = np.array([np.pad(n, (0, 3-spec.rank)) for n in product((-1, 0, 1), repeat=spec.rank)])
    for template in spec.templates:
        if template.multiplicity != len(x):
            continue
        a, b = template.arrays()
        inverse = np.linalg.pinv(a[0])
        # One atom determines the at-most-three free coordinates of its orbit.
        for point in x:
            candidates = point+shifts-b[0]
            qs = candidates@inverse.T
            errors = np.linalg.norm(qs@a[0].T-candidates, axis=1)
            for q in qs[errors < tolerance]:
                candidate = b+np.einsum('niq,q->ni', a, q)
                delta = candidate[:, None]-x[None]
                delta[..., periodic] -= np.round(delta[..., periodic])
                cost = np.linalg.norm(delta, axis=-1)
                rows, columns = linear_sum_assignment(cost)
                if cost[rows, columns].max() < tolerance:
                    return template.letter, candidate, columns
    raise ValueError('No operation-derived orbit chart matches this endpoint without changing geometry')


def recompile_target(target):
    old = target.state
    spec = compile_operations(old.spec.operations, old.spec.kind)
    if old.spec.magnetic or old.spec.spin_generators:
        raise ValueError('Automatic endpoint conversion currently supports nonmagnetic targets')
    decoded = old.decoded(wrap=False)
    letters, coordinates, moments = [], [], []
    for section in old.compiled.slices:
        letter, x, order = match_orbit(spec, decoded['coordinates'][section].detach().cpu().numpy())
        letters.append(letter); coordinates.append(old.coordinates.new_tensor(x))
        if old.moments is not None:
            moments.append(old.moments[section][torch.as_tensor(order, device=old.moments.device)])
    compiled = compile_constraints(spec, tuple(letters))
    lattice = decoded['lattice'].detach().double().cpu().numpy()[:spec.rank, :spec.rank]
    inverse = np.linalg.inv(compiled.root)
    values, vectors = np.linalg.eigh(inverse@(lattice.T@lattice)@inverse)
    symmetric = (vectors*np.log(values))@vectors.T
    metric = np.einsum('kij,ij->k', compiled.standard, symmetric)
    state = JointState(spec, tuple(letters), old.elements, old.metric.new_tensor(metric),
                       torch.cat(coordinates), None if old.moments is None else torch.cat(moments),
                       old.step, old.budget, old.status, chemistry=old.chemistry)
    # Reject unsupported settings rather than silently distorting the target cell.
    if not torch.allclose(state.decoded()['lattice'], decoded['lattice'], atol=2e-5, rtol=2e-5):
        raise ValueError('Operation-derived metric does not reconstruct the target lattice')
    return replace(target, state=state)
