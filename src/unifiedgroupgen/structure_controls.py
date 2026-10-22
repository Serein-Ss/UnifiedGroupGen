"""Fixed-cell scaffold completion in an explicit standard group setting.

Complete observed orbits can form a descriptor prefix. Incomplete orbits require
an explicit full descriptor and source-to-expanded-atom mapping. No group/origin
inference is silently performed on a fragment.
"""

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from .constraints import AffineObservation
from .data import parse_cif, decompose_orbits, _q_for_point


class Scaffold:
    def __init__(self, structure, kind, number, atom_map=None, tolerance=1e-4):
        self.kind, self.number, self.tolerance = kind, int(number), tolerance
        self.lattice = np.asarray(structure.lattice.matrix).T.copy()
        self.elements = np.asarray(structure.atomic_numbers)
        self.coordinates = np.asarray(structure.frac_coords).copy()
        if kind != 'space':
            if not (np.allclose(self.lattice[:2, 2], 0, atol=tolerance)
                    and np.allclose(self.lattice[2, :2], 0, atol=tolerance)):
                raise ValueError('Layer scaffold requires xy periodic plane and an orthogonal vacuum axis')
            # Layer charts store absolute height in Angstrom, with z=0 at the cell centre.
            self.coordinates[:, 2] = (self.coordinates[:, 2]-.5)*self.lattice[2, 2]
            self.lattice[2, 2] = 1
        self.atom_map = atom_map
        self.prefix_descriptor, self.q = None, None
        if atom_map is None:
            self.prefix_descriptor, self.q = decompose_orbits(kind, self.number, self.coordinates,
                self.elements, tolerance, self.lattice)

    @classmethod
    def from_cif(cls, path, kind, number, atom_map=None, tolerance=1e-4):
        from pathlib import Path
        return cls(parse_cif(Path(path).read_text(encoding='utf-8'), preserve_fractional=True),
                   kind, number, atom_map, tolerance)

    def prefix(self, vocabulary):
        if self.prefix_descriptor is None:
            raise ValueError('Incomplete observed orbits require a fixed descriptor')
        return vocabulary.encode(self.prefix_descriptor)[:-1]

    def observation(self, compiled):
        if (compiled.descriptor.kind, compiled.descriptor.number) != (self.kind, self.number):
            raise ValueError('Scaffold group differs from generated descriptor')
        k = compiled.encode_metric(self.lattice.T@self.lattice, tolerance=self.tolerance)
        if self.atom_map is None:
            count = len(self.prefix_descriptor.orbits)
            if compiled.descriptor.orbits[:count] != self.prefix_descriptor.orbits:
                raise ValueError('Generated descriptor did not preserve the known orbit prefix')
            values = np.r_[k, self.q]
            observation = AffineObservation.fixed(compiled.dof, list(range(len(values))), values)
        else:
            mapping = {int(source): int(target) for source, target in self.atom_map.items()}
            if set(mapping) != set(range(len(self.elements))) or len(set(mapping.values())) != len(mapping):
                raise ValueError('Atom map must cover every observed atom once with distinct target indices')
            matrix, values = [], []
            for i, value in enumerate(k):
                row = np.zeros(compiled.dof); row[i] = 1
                matrix.append(row); values.append(value)
            fitted_parameters = {}
            for source, target in sorted(mapping.items()):
                if not 0 <= target < compiled.num_atoms or compiled.elements[target] != self.elements[source]:
                    raise ValueError('Atom map changes an observed element or uses an invalid target index')
                orbit = compiled.atom_orbit[target]
                offset = compiled.offsets[orbit]
                width = compiled.offsets[orbit+1]-offset
                block = compiled.coordinate_basis[target, :, offset:offset+width]
                fitted = None
                candidates = ([fitted_parameters[orbit]] if orbit in fitted_parameters else
                              _q_for_point(block, compiled.b[target], self.coordinates[source],
                                           compiled.periodic, self.tolerance, self.lattice))
                for q in candidates:
                    compatible = True
                    for i, j in mapping.items():
                        if compiled.atom_orbit[j] != orbit:
                            continue
                        delta = (compiled.b[j]+compiled.coordinate_basis[j, :, offset:offset+width]@q
                                 -self.coordinates[i])
                        delta[compiled.periodic] -= np.round(delta[compiled.periodic])
                        if np.linalg.norm(self.lattice@delta) > self.tolerance:
                            compatible = False
                            break
                    if compatible:
                        fitted = compiled.b[target]+block@q
                        fitted_parameters[orbit] = q
                        break
                if fitted is None:
                    raise ValueError('Observed coordinates conflict with the mapped symmetry orbit')
                for axis in range(3):
                    row = np.zeros(compiled.dof)
                    row[compiled.lattice_dof:] = compiled.coordinate_basis[target, axis]
                    matrix.append(row)
                    values.append(fitted[axis]-compiled.b[target, axis])
            observation = AffineObservation(np.asarray(matrix), np.asarray(values))
        # Detect incompatible observations before the ODE starts.
        observation.retract(torch.zeros(compiled.dof, dtype=torch.float64))
        return observation

    def report(self, compiled, state):
        expanded = compiled.expand(torch.as_tensor(state, dtype=torch.float64))
        coordinates = expanded['coordinates'].numpy()
        lattice = expanded['lattice'].numpy()
        errors = []
        if self.atom_map is not None:
            pairs = [(int(i), int(j)) for i, j in self.atom_map.items()]
        else:
            pairs = []
            known_atoms = sum(len(block) for block in compiled.blocks[:len(self.prefix_descriptor.orbits)])
            for element in set(self.elements):
                source = np.flatnonzero(self.elements == element)
                target = np.flatnonzero(compiled.elements[:known_atoms] == element)
                delta = coordinates[target][None]-self.coordinates[source][:, None]
                delta[..., compiled.periodic] -= np.round(delta[..., compiled.periodic])
                costs = np.linalg.norm(delta@lattice.T, axis=-1)
                rows, cols = linear_sum_assignment(costs)
                pairs.extend((int(source[i]), int(target[j])) for i, j in zip(rows, cols))
        for i, j in pairs:
            delta = coordinates[j]-self.coordinates[i]
            delta[compiled.periodic] -= np.round(delta[compiled.periodic])
            errors.append(float(np.linalg.norm(lattice@delta)))
        observed_metric = self.lattice.T@self.lattice
        metric_error = float(np.max(np.abs(lattice.T@lattice-observed_metric)))
        return {'observed_atoms': len(self.elements), 'matched_atoms': len(pairs),
                'maximum_observed_displacement_angstrom': max(errors, default=0),
                'fixed_metric_max_residual': metric_error,
                'scaffold_preserved': len(pairs) == len(self.elements)
                    and max(errors, default=0) <= self.tolerance and metric_error <= self.tolerance,
                'scope': 'Fixed cell metric and known atoms in the supplied group setting; not physical stability'}
