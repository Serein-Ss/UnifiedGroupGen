"""Operator-based constraints for the shared, variable-size atom-graph model.

The canonical data decoder and operation compiler share the coordinate convention:
fractional periodic components and real nonperiodic components (angstrom).
"""
from dataclasses import dataclass
from functools import lru_cache
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from unifiedgroupgen.group_constraints.compiler import (
    GROUP_PERIODIC, catalog_operations, group_table, wp_arrays, metric_chart, _symmetric_basis, _sqrt_spd,
)


def _matrix(value):
    return tuple(tuple(float(x) for x in row) for row in value)


@dataclass(frozen=True)
class Operation:
    rotation: tuple
    translation: tuple
    spin: tuple | None = None
    time_reversal: bool = False

    def to_dict(self):
        return dict(rotation=self.rotation, translation=self.translation,
                    spin=self.spin, time_reversal=self.time_reversal)


@dataclass(frozen=True)
class OrbitTemplate:
    letter: str
    basis: tuple
    offset: tuple

    @property
    def multiplicity(self):
        return len(self.offset)

    @property
    def dof(self):
        return len(self.basis[0][0])

    def arrays(self):
        return (np.asarray(self.basis).reshape(self.multiplicity, 3, self.dof),
                np.asarray(self.offset))

    def to_dict(self):
        return dict(letter=self.letter, basis=self.basis, offset=self.offset)


@dataclass(frozen=True)
class SymmetrySpec:
    kind: str
    number: int
    periodic: tuple
    dimension: int
    operations: tuple
    templates: tuple
    magnetic: bool = False
    spin_generators: tuple = ()

    def __post_init__(self):
        if len(self.periodic) != 3 or self.dimension not in (2, 3):
            raise ValueError('Expected a three-component periodic mask and dimension 2 or 3')
        rank = sum(self.periodic)
        if tuple(self.periodic) != tuple(i < rank for i in range(3)):
            raise ValueError('Use an adapted frame with periodic directions first')
        if self.kind == 'rod' and (rank != 1 or self.dimension != 3):
            raise ValueError('A rod group requires one periodic axis in three dimensions')
        if not self.operations or not self.templates:
            raise ValueError('Operations and orbit templates must be nonempty')
        if len({w.letter for w in self.templates}) != len(self.templates):
            raise ValueError('Orbit template names must be unique')
        for op in self.operations:
            r = np.asarray(op.rotation)
            if r.shape != (3, 3) or not np.isfinite(r).all() or abs(np.linalg.det(r)) < 1e-8:
                raise ValueError('Invalid spatial operation')
            if len(op.translation) != 3 or not np.isfinite(op.translation).all():
                raise ValueError('Invalid affine translation')
            if not np.allclose(r[rank:, :rank], 0, atol=1e-8):
                raise ValueError('Operation does not preserve the periodic subspace')
            if not np.allclose(r[:rank, rank:], 0, atol=1e-8):
                raise ValueError('Use orthogonal periodic/nonperiodic subspaces in the adapted frame')
            if rank and (not np.allclose(r[:rank, :rank], np.round(r[:rank, :rank]), atol=1e-8)
                         or not np.isclose(abs(np.linalg.det(r[:rank, :rank])), 1)):
                raise ValueError('Periodic operations must be unimodular lattice automorphisms')
            nonperiodic = r[rank:, rank:]
            if not np.allclose(nonperiodic.T@nonperiodic, np.eye(3-rank), atol=1e-7):
                raise ValueError('Nonperiodic operations must be Euclidean isometries')
            if op.spin is not None:
                d = np.asarray(op.spin)
                if d.shape != (3, 3) or not np.allclose(d.T @ d, np.eye(3), atol=1e-7):
                    raise ValueError('Classical moment representations must be orthogonal')
        for generator in self.spin_generators:
            g = np.asarray(generator)
            if g.shape != (3, 3) or not np.allclose(g.T, -g, atol=1e-7):
                raise ValueError('Continuous orthogonal spin generators must be skew symmetric')
        for template in self.templates:
            a, b = template.arrays()
            if not np.isfinite(a).all() or not np.isfinite(b).all():
                raise ValueError('Nonfinite orbit chart')
            if template.dof and np.linalg.matrix_rank(a.reshape(-1, template.dof)) != template.dof:
                raise ValueError('Orbit chart must have independent free coordinates')
            _template_permutations(self, template)

    @property
    def rank(self):
        return sum(self.periodic)

    def template(self, letter):
        try:
            return next(w for w in self.templates if w.letter == letter)
        except StopIteration as error:
            raise ValueError(f'Unknown orbit template {letter}') from error

    def to_dict(self):
        return dict(kind=self.kind, number=self.number, periodic=self.periodic,
                    dimension=self.dimension, magnetic=self.magnetic,
                    operations=[op.to_dict() for op in self.operations],
                    templates=[w.to_dict() for w in self.templates],
                    spin_generators=self.spin_generators)

    @classmethod
    def from_dict(cls, data, *, _compiled=False):
        if 'templates' not in data:
            from unifiedgroupgen.group_constraints.operations import compile_request
            return compile_request(data)
        values=(data['kind'], int(data.get('number', 0)), tuple(data['periodic']),
                   int(data.get('dimension', 3)),
                   tuple(Operation(_matrix(o['rotation']), tuple(o['translation']),
                                   None if o.get('spin') is None else _matrix(o['spin']),
                                   bool(o.get('time_reversal', False))) for o in data['operations']),
                   tuple(OrbitTemplate(w['letter'], tuple(tuple(tuple(v) for v in row) for row in w['basis']),
                                       _matrix(w['offset'])) for w in data['templates']),
                   bool(data.get('magnetic', False)),
                   tuple(_matrix(g) for g in data.get('spin_generators', ())))
        if not _compiled:return cls(*values)
        # Only the checksummed, already compiled Tensor cache uses this path.
        # Raw user-supplied operations still run the normal constructor checks.
        result=object.__new__(cls)
        for name,value in zip(cls.__dataclass_fields__,values):object.__setattr__(result,name,value)
        return result

    @classmethod
    def magnetic_database(cls, uni_number, templates=None):
        """Spglib MSG operators; supply special charts for their precise setting.

        Without charts, the spatially unique general orbit is supplied. This does
        not pretend to enumerate every special magnetic Wyckoff position.
        """
        import spglib
        result = spglib.get_magnetic_symmetry_from_database(int(uni_number))
        if result is None:
            raise ValueError('Unknown magnetic space-group UNI number')
        operations = tuple(Operation(_matrix(r), tuple(t), None, bool(tr)) for r, t, tr in zip(
            result['rotations'], result['translations'], result['time_reversals']))
        if templates is None:
            unique = {}
            for op in operations:
                key = (op.rotation, tuple(np.remainder(op.translation, 1).round(10)))
                unique[key] = op
            templates = (OrbitTemplate('general', tuple(op.rotation for op in unique.values()),
                                       tuple(op.translation for op in unique.values())),)
        return cls('magnetic', int(uni_number), (True, True, True), 3,
                   operations, tuple(templates), True)


@lru_cache(maxsize=512)
def catalog_spec(kind, number):
    table = group_table(kind, number)
    operations = tuple(Operation(_matrix(r), tuple(t)) for r, t in catalog_operations(kind, number))
    templates = []
    for wp in table:
        a, b, _ = wp_arrays(kind, number, wp.letter)
        templates.append(OrbitTemplate(wp.letter,
                         tuple(tuple(tuple(v) for v in row) for row in a), _matrix(b)))
    return SymmetrySpec(kind, number, GROUP_PERIODIC[kind],
                        2 if kind == 'plane' else 3, tuple(operations), tuple(templates))


@lru_cache(maxsize=512)
def _metric_data(spec):
    d = spec.rank
    if not d:
        return np.zeros((0, 0)), np.zeros((0, 0, 0)), np.zeros((0, 0, 0))
    standard = _symmetric_basis(d)
    if (spec.kind in GROUP_PERIODIC and spec.number > 0
            and {op.rotation for op in spec.operations} == {op.rotation for op in catalog_spec(spec.kind, spec.number).operations}):
        root, _, basis = metric_chart(spec.kind, spec.number)
        return root, basis, standard
    from scipy.linalg import null_space
    generators = [np.asarray(op.rotation)[:d, :d] for op in spec.operations]
    rotations = [np.eye(d)]
    keys = {tuple(np.eye(d).round(8).ravel())}
    for current in rotations:
        for generator in generators:
            candidate = current@generator
            key = tuple(candidate.round(8).ravel())
            if key not in keys:
                if len(rotations) >= 192:
                    raise ValueError('Periodic spatial rotations do not close into a finite crystallographic point group')
                keys.add(key); rotations.append(candidate)
    rotations = np.asarray(rotations)
    h0 = np.mean(rotations.transpose(0, 2, 1) @ rotations, axis=0)
    h0 = h0 / np.linalg.det(h0) ** (1 / d) * 9
    root, invroot = _sqrt_spd(h0), _sqrt_spd(h0, inverse=True)
    orthogonal = root[None] @ rotations @ invroot[None]
    rows = np.concatenate([np.stack([(r.T @ b @ r - b).ravel() for b in standard], axis=1)
                           for r in orthogonal])
    coefficients = null_space(rows, rcond=1e-8) if np.max(np.abs(rows)) > 1e-8 else np.eye(len(standard))
    return root, np.einsum('ik,ide->kde', coefficients, standard), standard


@lru_cache(maxsize=8192)
def compile_constraints(spec, letters):
    return CompiledConstraints(spec, tuple(letters))


class CompiledConstraints:
    def __init__(self, spec, letters):
        self.spec, self.letters = spec, letters
        self.templates = tuple(spec.template(letter) for letter in letters)
        self.root, self.metric_basis, self.standard = _metric_data(spec)
        self.metric_dimension = len(self.standard)
        self.slices, self.blocks, count = [], [], 0
        for template in self.templates:
            _template_permutations(spec, template)
            a, b = template.arrays()
            flat = a.reshape(3 * len(b), template.dof)
            inverse = np.linalg.pinv(flat) if template.dof else np.zeros((0, 3 * len(b)))
            self.blocks.append((a, b, inverse))
            self.slices.append(slice(count, count + len(b)))
            count += len(b)
        self.num_atoms = count
        self.atom_orbit = np.repeat(np.arange(len(letters)), [w.multiplicity for w in self.templates])

    def metric(self, latent):
        if latent.shape != (self.metric_dimension,):
            raise ValueError('Wrong public metric-state dimension')
        if not self.spec.rank:
            return latent.new_zeros((0, 0))
        standard = torch.as_tensor(self.standard, device=latent.device, dtype=latent.dtype)
        basis = torch.as_tensor(self.metric_basis, device=latent.device, dtype=latent.dtype)
        root = torch.as_tensor(self.root, device=latent.device, dtype=latent.dtype)
        raw = torch.einsum('k,kij->ij', latent, standard)
        projected = torch.einsum('k,kij->ij', torch.einsum('kij,ij->k', basis, raw), basis)
        return root @ torch.matrix_exp(projected) @ root

    def lattice(self, latent):
        if not self.spec.rank:
            return torch.eye(3, device=latent.device, dtype=latent.dtype)
        top = torch.linalg.cholesky(self.metric(latent)).T
        lattice = torch.eye(3, device=latent.device, dtype=latent.dtype)
        lattice[:self.spec.rank, :self.spec.rank] = top
        return lattice

    def project_coordinates(self, latent, wrap=True):
        if latent.shape != (self.num_atoms, 3):
            raise ValueError('Wrong public coordinate-state shape')
        values = []
        for block, section in zip(self.blocks, self.slices):
            a, b, inverse = [torch.as_tensor(v, device=latent.device, dtype=latent.dtype) for v in block]
            q = inverse @ (latent[section] - b).reshape(-1)
            values.append(b + torch.einsum('ijk,k->ij', a, q))
        result = torch.cat(values) if values else latent.clone()
        if wrap:
            periodic = torch.tensor(self.spec.periodic, device=latent.device)
            result = torch.where(periodic, torch.remainder(result, 1), result)
        return result

    def permutations(self):
        return _permutations(self.spec, self.letters)

    def project_moments(self, moments, lattice):
        if moments is None:
            return None
        from unifiedgroupgen.group_constraints.fields import project_moments
        return project_moments(moments, self.spec, self.permutations(), lattice)

    def residuals(self, coordinates, elements, lattice, moments=None):
        maximum = coordinates.new_zeros(())
        chemical = True
        periodic = torch.tensor(self.spec.periodic, device=coordinates.device)
        for op, permutation in zip(self.spec.operations, self.permutations()):
            r, t = coordinates.new_tensor(op.rotation), coordinates.new_tensor(op.translation)
            delta = coordinates @ r.T + t - coordinates[list(permutation)]
            delta = torch.where(periodic, delta - delta.round(), delta)
            if delta.numel():
                maximum = torch.maximum(maximum, (delta @ lattice.T).norm(dim=-1).max())
            chemical = chemical and bool(torch.equal(elements, elements[list(permutation)]))
        result = {'coordinates': maximum, 'elements': chemical}
        if moments is not None:
            from unifiedgroupgen.group_constraints.fields import moment_residual
            result['moments'] = moment_residual(moments, self.spec, self.permutations(), lattice)
        return result


@lru_cache(maxsize=8192)
def _template_permutations(spec, template):
    a, b = template.arrays()
    q = np.array([.137031, .293117, .419071])[:template.dof]
    coords = b + np.einsum('ijk,k->ij', a, q) if len(q) else b.copy()
    result = []
    for op in spec.operations:
        mapped = coords @ np.asarray(op.rotation).T + np.asarray(op.translation)
        delta = mapped[:, None] - coords[None, :]
        delta[..., spec.periodic] -= np.round(delta[..., spec.periodic])
        cost = np.linalg.norm(delta, axis=-1)
        rows, cols = linear_sum_assignment(cost)
        if cost[rows, cols].max() > 1e-6:
            raise ValueError('Orbit template is not invariant under the supplied operations')
        permutation = np.empty(template.multiplicity, dtype=np.int64)
        permutation[rows] = cols
        # Check the entire affine chart, not just a lucky representative point.
        linear = np.einsum('ij,njk->nik', np.asarray(op.rotation), a)-a[permutation]
        offset = b @ np.asarray(op.rotation).T+np.asarray(op.translation)-b[permutation]
        offset[:, spec.periodic] -= np.round(offset[:, spec.periodic])
        if (linear.size and np.max(np.abs(linear)) > 1e-7) or np.max(np.abs(offset)) > 1e-7:
            raise ValueError('Supplied orbit is not an invariant affine chart')
        result.append(tuple(permutation.tolist()))
    return tuple(result)


@lru_cache(maxsize=512)
def _permutations(spec, letters):
    compiled = compile_constraints(spec, letters)
    local = [_template_permutations(spec, template) for template in compiled.templates]
    return tuple(tuple(index+section.start for section, permutations in zip(compiled.slices, local)
                       for index in permutations[g]) for g in range(len(spec.operations)))
