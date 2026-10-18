"""Fail before model execution if the installed layer-group catalog is inconsistent."""

from functools import lru_cache
import hashlib
from importlib.metadata import version
import json

import numpy as np

from .representation_audit import audit_catalog, compiled_chart, expand_numpy
from .symmetry import Descriptor, OrbitSpec, group_table


@lru_cache(maxsize=1)
def check_layer_catalog():
    """Check all 80 groups / 477 single orbits once per process; do not touch RNG state."""
    installed = version('pyxtal')
    if installed != '1.1.3':
        raise ValueError(f'PyXtal {installed} is not the verified catalog version. '
                         'Install with: python -m pip install --no-deps "pyxtal==1.1.3". '
                         'PyXtal 1.1.1 contains a malformed LG52 4c orbit.')
    failures, groups, count = [], set(), 0
    digest = hashlib.sha256()
    rng = np.random.default_rng(815)
    maxima = {key: 0.0 for key in ('operation_residual_angstrom', 'tangent_action_residual',
                                  'metric_relative_residual')}
    affine_residual = 0.0
    for row in audit_catalog(kinds=('layer',)):
        count += 1
        groups.add(row['group'])
        c = compiled_chart(Descriptor('layer', row['group'], (OrbitSpec(row['letter'], 14),)))
        # A bijection check also detects duplicated/missing members of a listed orbit.
        z = rng.uniform(0.11, 0.89, c.dof)
        z[:c.lattice_dof] *= 0.2
        _, coordinates, _ = expand_numpy(c, z)
        try:
            permutations = c.permutations(coordinates, tolerance=1e-7)
            for operation, (rotation, permutation) in zip(group_table('layer', row['group'])[0].ops, permutations):
                delta_b = c.b[permutation]-(c.b @ rotation.T+operation.translation_vector)
                delta_b[..., c.periodic] -= np.round(delta_b[..., c.periodic])
                residual = float(np.max(np.abs(delta_b)))
                if c.coordinate_dof:
                    delta_basis = c.coordinate_basis[permutation]-np.einsum('ab,ibq->iaq', rotation, c.coordinate_basis)
                    residual = max(residual, float(np.max(np.abs(delta_basis))))
                affine_residual = max(affine_residual, residual)
                if residual > 1e-7:
                    raise ValueError('Affine group action is not closed for all free parameters')
            delta = coordinates[:, None]-coordinates[None]
            delta[..., c.periodic] -= np.round(delta[..., c.periodic])
            distances = np.linalg.norm(delta, axis=-1)
            np.fill_diagonal(distances, np.inf)
            if np.any(distances < 1e-8):
                raise ValueError('Repeated members in a single Wyckoff orbit')
        except ValueError as error:
            row = {**row, 'passed': False, 'error': str(error)}
        if not row['passed']:
            failures.append(row)
        for key in maxima:
            maxima[key] = max(maxima[key], row[key])
        payload = {'group': row['group'], 'letter': row['letter'], 'b': c.b.tolist(),
                   'basis': c.coordinate_basis.tolist(), 'root': c.root.tolist(),
                   'metric_basis': c.metric_basis.tolist()}
        digest.update(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
    if count != 477 or groups != set(range(1, 81)) or failures:
        details = ', '.join(f"LG{r['group']} {r['multiplicity']}{r['letter']} "
                            f"({r.get('error', 'algebraic consistency failed')}; "
                            f"closure={r['operation_residual_angstrom']:.3g})" for r in failures)
        raise ValueError(f'Layer catalog preflight failed: {len(groups)}/80 groups, '
                         f'{count-len(failures)}/477 single orbits passed; {details}')
    maxima['affine_coefficient_residual'] = affine_residual
    return {'passed': True, 'pyxtal': installed, 'layer_groups': len(groups), 'single_orbit_types': count,
            'catalog_sha256': digest.hexdigest(), 'max_residuals': maxima,
            'scope': 'Single-orbit closure, multiplicity, tangent/kernel and metric consistency; '
                     'not exhaustive multi-orbit combinations or material stability.'}
