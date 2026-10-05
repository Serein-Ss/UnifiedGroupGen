"""Compile group/Wyckoff metadata into affine orbit charts and invariant SPD metrics.

All lattices use COLUMN vectors: Cartesian r = L @ f; metric H = L.T @ L.
Layer coordinates are (fractional x, fractional y, height in angstrom), z is NOT wrapped.
PyXtal Wyckoff maps are often singular parameter maps, not point-group rotations.
"""

from dataclasses import asdict, dataclass
from functools import lru_cache
import numpy as np
from scipy.linalg import null_space, qr
import torch
from pyxtal.symmetry import Group


# Embed the 17 wallpaper groups into layer groups with unchanged normal coordinate.
PLANE_TO_LAYER = (1, 3, 11, 12, 13, 23, 24, 25, 26, 49, 55, 56, 65, 69, 70, 73, 77)


@dataclass(frozen=True)
class OrbitSpec:
    letter: str
    atomic_number: int


@dataclass(frozen=True)
class Descriptor:
    kind: str
    number: int
    orbits: tuple[OrbitSpec, ...]

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        return cls(data["kind"], int(data["number"]), tuple(OrbitSpec(**w) for w in data["orbits"]))


@lru_cache(maxsize=None)
def group_table(kind, number):
    limits = {"space": 230, "layer": 80, "plane": 17}
    if kind not in limits or not 1 <= number <= limits[kind]:
        raise ValueError(f"Invalid {kind} group number: {number}")
    actual = PLANE_TO_LAYER[number - 1] if kind == "plane" else number
    return Group(actual, dim=3 if kind == "space" else 2)


def wp_arrays(kind, number, letter):
    wp = next((w for w in group_table(kind, number) if w.letter == letter), None)
    if wp is None:
        raise ValueError(f"Unknown orbit {kind}:{number}:{letter}")
    maps = np.stack([s.affine_matrix for s in wp.ops])
    if kind == "plane":
        maps[:, 2, :] = 0
        maps[:, :, 2] = 0
    p = maps[0, :3, :3]
    rank = np.linalg.matrix_rank(p, tol=1e-8)
    _, _, pivots = qr(p, pivoting=True)
    axes = sorted(pivots[:rank].tolist())
    a = maps[:, :3, axes]
    b = maps[:, :3, 3]
    if rank and np.linalg.matrix_rank(a.reshape(-1, rank), tol=1e-8) != rank:
        raise ValueError("Orbit parameter map has inconsistent rank")
    return a, b, axes


def _symmetric_basis(d):
    result = []
    for i in range(d):
        for j in range(i, d):
            b = np.zeros((d, d))
            b[i, j] = b[j, i] = 1 if i == j else 1 / np.sqrt(2)
            result.append(b)
    return np.array(result)


def _sqrt_spd(h, inverse=False):
    values, vectors = np.linalg.eigh(h)
    if np.min(values) <= 0:
        raise ValueError("Expected positive definite metric")
    return (vectors * values ** (-0.5 if inverse else 0.5)) @ vectors.T


@lru_cache(maxsize=None)
def metric_chart(kind, number):
    d = 3 if kind == "space" else 2
    rotations = np.stack([op.rotation_matrix[:d, :d] for op in group_table(kind, number)[0].ops])
    # Reynolds averaging supplies a reference metric even in a nonorthogonal fractional basis.
    h0 = np.mean(rotations.transpose(0, 2, 1) @ rotations, axis=0)
    h0 = h0 / np.linalg.det(h0) ** (1 / d) * 9.0
    root, invroot = _sqrt_spd(h0), _sqrt_spd(h0, inverse=True)
    u = root[None] @ rotations @ invroot[None]
    standard = _symmetric_basis(d)
    constraints = np.concatenate([
        np.stack([(r.T @ b @ r - b).reshape(-1) for b in standard], axis=1)
        for r in u
    ], axis=0)
    coefficients = null_space(constraints, rcond=1e-8) if np.max(np.abs(constraints)) > 1e-8 else np.eye(len(standard))
    # Canonicalize the subspace basis: an arbitrary SVD basis can rotate/sign-flip
    # between BLAS implementations, which would change serialized k coordinates.
    projector = coefficients @ coefficients.T
    canonical = []
    for column in projector.T:
        vector = column.copy()
        for previous in canonical:
            vector -= previous * (previous @ vector)
        norm = np.linalg.norm(vector)
        if norm > 1e-7:
            vector /= norm
            pivot = np.flatnonzero(np.abs(vector)>1e-7)[0]
            canonical.append(vector if vector[pivot]>0 else -vector)
    coefficients = np.stack(canonical, axis=1)
    bases = np.einsum("ik,ide->kde", coefficients, standard)
    return root, invroot, bases


class CompiledDescriptor:
    def __init__(self, descriptor):
        if not descriptor.orbits:
            raise ValueError("Empty structures are not allowed")
        self.descriptor = descriptor
        self.periodic = np.array([True, True, descriptor.kind == "space"])
        self.root, self.invroot, self.metric_basis = metric_chart(descriptor.kind, descriptor.number)
        self.lattice_dof = len(self.metric_basis)
        self.blocks, self.offsets, self.q_periodic = [], [0], []
        atom_orbit, elements, translations = [], [], []
        zero_sites = set()
        for i, orbit in enumerate(descriptor.orbits):
            if not 1 <= orbit.atomic_number <= 118:
                raise ValueError("Atomic number must be between 1 and 118")
            a, b, axes = wp_arrays(descriptor.kind, descriptor.number, orbit.letter)
            if not axes:
                if orbit.letter in zero_sites:
                    raise ValueError("A zero-dimensional orbit cannot be occupied twice")
                zero_sites.add(orbit.letter)
            self.blocks.append(a)
            self.offsets.append(self.offsets[-1] + len(axes))
            self.q_periodic.extend(self.periodic[axes].tolist())
            atom_orbit.extend([i] * len(a))
            elements.extend([orbit.atomic_number] * len(a))
            translations.append(b)
        self.atom_orbit = np.array(atom_orbit)
        self.elements = np.array(elements)
        self.b = np.concatenate(translations)
        self.num_atoms = len(elements)
        self.coordinate_dof = self.offsets[-1]
        self.coordinate_basis = np.zeros((self.num_atoms, 3, self.coordinate_dof))
        start = 0
        for i, a in enumerate(self.blocks):
            self.coordinate_basis[start:start + len(a), :, self.offsets[i]:self.offsets[i + 1]] = a
            start += len(a)
        self.dof = self.lattice_dof + self.coordinate_dof
        self.state_periodic = np.array([False] * self.lattice_dof + self.q_periodic)

    def tensor(self, array, like):
        return torch.as_tensor(array, dtype=like.dtype, device=like.device)

    def metric(self, k):
        b, root = self.tensor(self.metric_basis, k), self.tensor(self.root, k)
        s = torch.einsum("...k,kij->...ij", k, b)
        return root @ torch.matrix_exp(s) @ root

    def encode_metric(self, h, tolerance=1e-5):
        d = self.root.shape[0]
        h = np.asarray(h)[:d, :d]
        normalized = self.invroot @ h @ self.invroot
        values, vectors = np.linalg.eigh(normalized)
        if np.min(values) <= 0:
            raise ValueError("Data lattice is not positive definite")
        s = (vectors * np.log(values)) @ vectors.T
        k = np.einsum("kij,ij->k", self.metric_basis, s)
        reconstructed = np.einsum("k,kij->ij", k, self.metric_basis)
        if np.linalg.norm(s - reconstructed) > tolerance:
            raise ValueError("Data lattice is incompatible with the declared group/setting")
        return k

    def expand(self, state, wrap=True):
        if state.shape != (self.dof,):
            raise ValueError(f"Expected state dimension {self.dof}, got {state.shape}")
        q = state[self.lattice_dof:]
        coords = self.tensor(self.b, state) + torch.einsum("ijq,q->ij", self.tensor(self.coordinate_basis, state), q)
        if wrap:
            periodic = torch.as_tensor(self.periodic, device=state.device)
            coords = torch.where(periodic[None], torch.remainder(coords, 1), coords)
        h = self.metric(state[:self.lattice_dof])
        # Cholesky gives a canonical Cartesian frame; L.T @ L = H.
        l = torch.linalg.cholesky(h).T
        if self.descriptor.kind != "space":
            top = torch.cat((l, l.new_zeros((2, 1))), dim=1)
            l = torch.cat((top, l.new_tensor([[0, 0, 1]])), dim=0)
        return {"coordinates": coords, "lattice": l, "atomic_numbers": torch.as_tensor(self.elements, device=state.device),
                "orbit_index": torch.as_tensor(self.atom_orbit, device=state.device)}

    def coordinate_velocity(self, q_velocity):
        return torch.einsum("ijq,q->ij", self.tensor(self.coordinate_basis, q_velocity), q_velocity)

    def reduce_velocity(self, candidate, lattice):
        if not self.coordinate_dof:
            return candidate.new_zeros(0)
        a = self.tensor(self.coordinate_basis, candidate)
        m = lattice.T @ lattice
        gram = torch.einsum("iaq,ab,ibr->qr", a, m, a)
        rhs = torch.einsum("iaq,ab,ib->q", a, m, candidate)
        return torch.linalg.solve(gram, rhs)

    def prior(self, device="cpu", dtype=torch.float32, generator=None):
        state = torch.randn(self.dof, device=device, dtype=dtype, generator=generator)
        indices = torch.as_tensor(np.flatnonzero(self.state_periodic), device=device)
        state[indices] = torch.rand(len(indices), device=device, dtype=dtype, generator=generator)
        return state

    def permutations(self, coords, tolerance=1e-6):
        """Diagnostic group action; at a regular point the orbit permutation is unique."""
        from scipy.optimize import linear_sum_assignment
        f = np.asarray(coords)
        result = []
        for op in group_table(self.descriptor.kind, self.descriptor.number)[0].ops:
            r, t = op.rotation_matrix.copy(), op.translation_vector.copy()
            if self.descriptor.kind == "plane":
                r[2] = [0, 0, 1]
                t[2] = 0
            mapped = f @ r.T + t
            permutation = np.empty(self.num_atoms, dtype=int)
            for orbit in range(len(self.blocks)):
                ids = np.flatnonzero(self.atom_orbit == orbit)
                delta = mapped[ids, None] - f[None, ids]
                delta[..., self.periodic] -= np.round(delta[..., self.periodic])
                cost = np.linalg.norm(delta, axis=-1)
                rows, cols = linear_sum_assignment(cost)
                if np.max(cost[rows, cols]) > tolerance:
                    raise ValueError("Orbit is not invariant under a requested group operation")
                permutation[ids[rows]] = ids[cols]
            result.append((r, permutation))
        return result

    def constraint_matrix(self, coords):
        """Stack u_pi(i) - R u_i = 0, including all site stabilizers.

        Used for small mathematical checks, not dense construction during training.
        """
        rows = []
        for r, permutation in self.permutations(coords):
            for i, j in enumerate(permutation):
                c = np.zeros((3, 3 * self.num_atoms))
                c[:, 3*j:3*j+3] += np.eye(3)
                c[:, 3*i:3*i+3] -= r
                rows.append(c)
        if self.descriptor.kind == "plane":
            c = np.zeros((self.num_atoms, 3*self.num_atoms))
            c[np.arange(self.num_atoms), 3*np.arange(self.num_atoms)+2] = 1
            rows.append(c)
        return np.concatenate(rows)


@lru_cache(maxsize=4096)
def compile_descriptor(descriptor):
    return CompiledDescriptor(descriptor)
