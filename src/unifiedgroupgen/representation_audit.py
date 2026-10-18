"""Numerical evidence for the existing catalog compiler, independent of learned weights."""

from functools import lru_cache
import numpy as np
from scipy.linalg import expm

from .symmetry import CompiledDescriptor, Descriptor, OrbitSpec, group_table, metric_chart


@lru_cache(maxsize=128)
def compiled_chart(descriptor):
    # Bound memory during a full-data audit; do not populate the training compiler cache.
    return CompiledDescriptor(descriptor)


def expand_numpy(compiled, state):
    state = np.asarray(state, dtype=float)
    if state.shape != (compiled.dof,) or not np.isfinite(state).all():
        raise ValueError('Nonfinite or wrong-dimensional state')
    h = compiled.root @ expm(np.einsum('k,kij->ij', state[:compiled.lattice_dof], compiled.metric_basis)) @ compiled.root
    lattice = np.eye(3)
    lattice[:len(h), :len(h)] = np.linalg.cholesky(h).T
    coords = compiled.b.copy()
    if compiled.coordinate_dof:
        coords += np.einsum('ijq,q->ij', compiled.coordinate_basis, state[compiled.lattice_dof:])
    return lattice, coords, h


def action_statistics(compiled, coords, lattice):
    """Check each orbit from its anchor under every operation; no dense 3N constraint matrix."""
    operations = group_table(compiled.descriptor.kind, compiled.descriptor.number)[0].ops
    rotations = np.stack([op.rotation_matrix.copy() for op in operations])
    translations = np.stack([op.translation_vector.copy() for op in operations])
    if compiled.descriptor.kind == 'plane':
        rotations[:, 2] = [0, 0, 1]
        translations[:, 2] = 0
    closure, tangent = 0.0, 0.0
    ranks, nullities = [], []
    for index, basis in enumerate(compiled.blocks):
        positions = coords[compiled.atom_orbit == index]
        mapped = np.einsum('gij,j->gi', rotations, positions[0])+translations
        delta = mapped[:, None]-positions[None]
        delta[..., compiled.periodic] -= np.round(delta[..., compiled.periodic])
        distances = np.linalg.norm(delta @ lattice.T, axis=-1)
        selected = distances.argmin(axis=1)
        closure = max(closure, float(distances.min(axis=1).max()))
        transformed_basis = np.einsum('gij,jq->giq', rotations, basis[0])
        tangent = max(tangent, float(np.max(np.abs(transformed_basis-basis[selected]), initial=0)))
        fixed = selected == 0
        stabilizer = (rotations[fixed]-np.eye(3)).reshape(-1, 3)
        if compiled.descriptor.kind == 'plane':
            stabilizer = np.vstack([stabilizer, [0, 0, 1]])
        ranks.append(int(np.linalg.matrix_rank(basis[0], tol=1e-8)))
        nullities.append(3-int(np.linalg.matrix_rank(stabilizer, tol=1e-8)))
    h = lattice.T @ lattice
    metric = rotations.transpose(0, 2, 1) @ h @ rotations-h
    return {'operation_residual_angstrom': closure, 'tangent_action_residual': tangent,
            'metric_relative_residual': float(np.max(np.abs(metric))/max(np.linalg.norm(h), 1e-30)),
            'anchor_basis_rank': sum(ranks), 'anchor_kernel_dimension': sum(nullities)}


def audit_record(record, seed=182, path_points=3):
    descriptor = Descriptor.from_dict(record['descriptor'])
    c = compiled_chart(descriptor)
    z = np.asarray(record['state'], dtype=float)
    lattice, coords, h = expand_numpy(c, z)
    recovered = []
    for i, basis in enumerate(c.blocks):
        rhs = coords[c.atom_orbit == i]-c.b[c.atom_orbit == i]
        recovered.extend(np.linalg.lstsq(basis.reshape(-1, basis.shape[-1]), rhs.ravel(), rcond=None)[0]
                         if basis.shape[-1] else [])
    k = c.encode_metric(h, tolerance=1e-7)
    inverse_error = float(np.max(np.abs(np.r_[k, recovered]-z), initial=0))
    stats = action_statistics(c, coords, lattice)
    # Regularity is checked on a reproducible random point, not a possibly special data point.
    rng = np.random.default_rng(seed)
    start = rng.normal(0, 0.2, c.dof)
    start[c.state_periodic] = rng.uniform(0.1, 0.9, c.state_periodic.sum())
    displacement = z-start
    displacement[c.state_periodic] = (displacement[c.state_periodic]+0.5) % 1-0.5
    path_residual = 0.0
    for t in np.linspace(0, 1, path_points):
        ll, ff, _ = expand_numpy(c, start+t*displacement)
        path_residual = max(path_residual, action_statistics(c, ff, ll)['operation_residual_angstrom'])
    result = {'id': record['id'], 'dataset': record['dataset'], 'kind': descriptor.kind,
              'group': descriptor.number, 'atoms': c.num_atoms, 'orbits': len(descriptor.orbits),
              'lattice_dof': c.lattice_dof, 'coordinate_dof': c.coordinate_dof, 'state_dof': c.dof,
              'inverse_state_max_error': inverse_error, 'metric_min_eigenvalue': float(np.linalg.eigvalsh(h).min()),
              'path_operation_residual_angstrom': path_residual, **stats}
    result['passed'] = (inverse_error < 1e-7 and stats['operation_residual_angstrom'] < 1e-7
                        and stats['metric_relative_residual'] < 1e-7 and path_residual < 1e-7
                        and stats['tangent_action_residual'] < 1e-7
                        and stats['anchor_basis_rank'] == stats['anchor_kernel_dimension'] == c.coordinate_dof
                        and result['metric_min_eigenvalue'] > 0)
    # Source idealization is distinct from the algebraic encode/expand round trip.
    provenance = record.get('provenance', {})
    for name in ('source_operation_residual_angstrom', 'std_position_rms_angstrom',
                 'std_position_max_angstrom', 'std_metric_relative_change', 'chart_fit_max_angstrom'):
        result[name] = provenance.get(name)
    return result


def audit_catalog(seed=182, kinds=('space', 'layer', 'plane')):
    """Exhaust all catalog groups and single-orbit types; multi-orbit lists are unbounded."""
    rng = np.random.default_rng(seed)
    for kind, limit in (('space', 230), ('layer', 80), ('plane', 17)):
        if kind not in kinds:
            continue
        for number in range(1, limit+1):
            root, _, bases = metric_chart(kind, number)
            gram = np.einsum('aij,bij->ab', bases, bases)
            for wp in group_table(kind, number):
                c = CompiledDescriptor(Descriptor(kind, number, (OrbitSpec(wp.letter, 14),)))
                z = rng.uniform(0.13, 0.87, c.dof)
                z[:c.lattice_dof] *= 0.2
                lattice, coords, _ = expand_numpy(c, z)
                result = action_statistics(c, coords, lattice)
                result.update(kind=kind, group=number, letter=wp.letter, multiplicity=wp.multiplicity,
                              coordinate_dof=c.coordinate_dof, lattice_dof=c.lattice_dof,
                              metric_basis_orthogonality_error=float(np.max(np.abs(gram-np.eye(len(bases))))))
                result['passed'] = (result['anchor_basis_rank'] == result['anchor_kernel_dimension'] == c.coordinate_dof
                    and result['operation_residual_angstrom'] < 1e-7
                    and result['tangent_action_residual'] < 1e-7
                    and result['metric_relative_residual'] < 1e-7
                    and result['metric_basis_orthogonality_error'] < 1e-7)
                yield result
