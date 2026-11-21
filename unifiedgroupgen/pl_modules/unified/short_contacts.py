"""Endpoint repulsion on the existing symmetry-constrained flow coordinates."""
from dataclasses import replace

import numpy as np
import torch
from ase.geometry import find_mic, minkowski_reduce


def endpoint_short_contacts(state, metric_velocity, coordinate_velocity, minimum=.6):
    """Sum squared short-contact gaps per atom, including periodic self images."""
    with torch.autocast(device_type=state.metric.device.type, enabled=False):
        endpoint = replace(state, metric=state.metric + metric_velocity.to(state.metric),
                           coordinates=state.coordinates + coordinate_velocity.to(state.coordinates))
        decoded = endpoint.decoded(wrap=False)
        return _periodic_short_contacts(decoded['coordinates'], decoded['lattice'].T,
                                        state.spec.periodic, minimum)


def _periodic_short_contacts(coordinates, cell, periodic, minimum):
    # The discrete MIC image choice is detached. The selected distances keep
    # derivatives through both the lattice and the representative coordinates.
    n = len(coordinates)
    i, j = torch.triu_indices(n, n, offset=1, device=coordinates.device)
    delta = coordinates[j] - coordinates[i]
    matrix = cell.detach().cpu().double().numpy()
    mask = np.asarray(periodic, dtype=bool)
    shift = None
    if len(delta) and mask.any():
        displacement = delta.detach().cpu().double().numpy() @ matrix
        nearest, _ = find_mic(displacement, matrix, pbc=mask)
        image = np.rint(np.linalg.solve(matrix.T, (nearest-displacement).T).T)
        image[:, ~mask] = 0
        shift = delta.new_tensor(image)
    vectors = (delta if shift is None else delta + shift) @ cell
    lengths = [torch.linalg.vector_norm(vectors, dim=1)]
    if mask.any():
        reduced, basis = minkowski_reduce(matrix, pbc=mask)
        shortest = np.flatnonzero(mask)[np.linalg.norm(reduced[mask], axis=1).argmin()]
        self_length = torch.linalg.vector_norm(cell.new_tensor(basis[shortest]) @ cell)
        lengths.append(self_length.expand(n))
    distance = torch.cat(lengths)
    return (1 - distance/minimum).clamp_min(0).square().sum() / n
