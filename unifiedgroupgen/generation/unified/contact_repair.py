"""Bounded terminal repair projected onto the current Wyckoff coordinates."""
from dataclasses import replace

import numpy as np
import torch
from ase.geometry import find_mic, minkowski_reduce

from unifiedgroupgen.group_constraints.flow_state import coordinate_tangent


def repair_short_contacts(state):
    """Keep the lattice, orbit templates and elements fixed throughout repair."""
    current = state.to("cpu", torch.float64)
    decoded = current.decoded(wrap=False)
    lattice = decoded["lattice"]
    cell = lattice.numpy().T
    periodic = np.asarray(state.spec.periodic)
    reduced, _ = minkowski_reduce(cell, pbc=periodic)
    self_distance = np.linalg.norm(reduced[periodic], axis=1).min()
    if self_distance < .6:
        return state, dict(applied=False, outcome="periodic_self_contact",
                           periodic_self_distance_angstrom=float(self_distance))
    pairs = np.triu_indices(current.num_atoms, 1)
    if not len(pairs[0]):
        return state, None

    def contacts(candidate):
        positions = candidate.decoded(wrap=False)["coordinates"].numpy() @ cell
        i, j = pairs
        return find_mic(positions[j]-positions[i], cell, pbc=periodic)

    baseline = decoded["coordinates"]
    vectors, distances = contacts(current)
    before = float(distances.min())
    if before >= .6:
        return state, None
    report = dict(applied=False, initial_pair_distance_angstrom=before)

    for step in range(32):
        k = int(distances.argmin())
        i, j = pairs[0][k], pairs[1][k]
        distance = distances[k]
        if distance < 1e-10:
            # At coincidence use directions tied to the lattice rather than a lab axis.
            directions = lattice.T / lattice.T.norm(dim=1, keepdim=True)
        else:
            directions = current.coordinates.new_tensor(vectors[k]/distance).unsqueeze(0)
        best = None
        for direction in directions:
            proposal = torch.zeros_like(current.coordinates)
            proposal[i], proposal[j] = -direction, direction
            tangent = coordinate_tangent(current, proposal, cartesian=True)
            response = float(((tangent[j]-tangent[i]) @ lattice.T) @ direction)
            if best is None or response > best[0]:
                best = response, tangent
        response, tangent = best
        if response < 1e-8:
            return state, dict(report, outcome="immovable_pair")
        current = replace(current, coordinates=current.coordinates+(.605-distance)/response*tangent)
        delta = (current.decoded(wrap=False)["coordinates"]-baseline) @ lattice.T
        displacement = float(delta.norm(dim=1).max())
        if displacement > .6:
            return state, dict(report, outcome="displacement_limit")
        vectors, distances = contacts(current)
        if distances.min() >= .6:
            repaired = replace(state, coordinates=current.coordinates.to(state.coordinates))
            return repaired, dict(report, applied=True, outcome="separated", iterations=step+1,
                                  final_pair_distance_angstrom=float(distances.min()),
                                  max_displacement_angstrom=displacement)
    return state, dict(report, outcome="iteration_limit")
