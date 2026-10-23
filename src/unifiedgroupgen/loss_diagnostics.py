"""Sample unscaled loss gradients without modifying .grad or consuming RNG."""

import math
import torch


def _gradients(loss, parameters):
    if not loss.requires_grad:
        return (None,)*len(parameters)
    return torch.autograd.grad(loss, parameters, retain_graph=True, allow_unused=True)


def _dot(first, second, indices):
    products = [(first[i].detach().float()*second[i].detach().float()).sum()
                for i in indices if first[i] is not None and second[i] is not None]
    return float(torch.stack(products).sum()) if products else 0.0


def gradient_diagnostics(proposal, flow, losses, weights, length_scale):
    """Report raw/effective norms and lattice-coordinate conflict on shared parameters.

    Norms are before GradScaler and global clipping. Coordinate gradients are
    obtained once from the raw physical loss; fixed-scale gradients follow by
    exact scalar multiplication. Heads are excluded from the shared comparison.
    """
    proposal_parameters = [p for p in proposal.parameters() if p.requires_grad]
    named = [(n, p) for n, p in flow.named_parameters() if p.requires_grad]
    parameters = [p for _, p in named]
    descriptor = _gradients(losses['descriptor'].mean(), proposal_parameters)
    lattice = _gradients(losses['lattice'].mean(), parameters)
    coordinates = _gradients(losses['coordinates'].mean(), parameters)
    all_flow = range(len(parameters))
    shared = [i for i, (n, _) in enumerate(named)
              if not n.startswith(('coordinate_head.', 'lattice_head.'))]
    descriptor_norm = math.sqrt(max(_dot(descriptor, descriptor, range(len(descriptor))), 0.0))
    lattice_norm = math.sqrt(max(_dot(lattice, lattice, all_flow), 0.0))
    coordinate_norm = math.sqrt(max(_dot(coordinates, coordinates, all_flow), 0.0))
    shared_lattice = math.sqrt(max(_dot(lattice, lattice, shared), 0.0))
    shared_coordinate = math.sqrt(max(_dot(coordinates, coordinates, shared), 0.0))
    inner = _dot(lattice, coordinates, all_flow)
    shared_inner = _dot(lattice, coordinates, shared)
    scale = length_scale**-2
    combined_squared = weights['flow']**2*(lattice_norm**2+scale**2*coordinate_norm**2+2*scale*inner)
    combined_squared += (weights['descriptor']*descriptor_norm)**2
    return {
        'coordinate_length_scale_angstrom': length_scale,
        'descriptor_raw_norm': descriptor_norm,
        'lattice_raw_norm': lattice_norm,
        'coordinates_raw_norm': coordinate_norm,
        'descriptor_weighted_norm': weights['descriptor']*descriptor_norm,
        'lattice_weighted_norm': weights['flow']*lattice_norm,
        'coordinates_weighted_norm': weights['flow']*scale*coordinate_norm,
        'shared_lattice_raw_norm': shared_lattice,
        'shared_coordinates_raw_norm': shared_coordinate,
        'shared_lattice_weighted_norm': weights['flow']*shared_lattice,
        'shared_coordinates_weighted_norm': weights['flow']*scale*shared_coordinate,
        'shared_lattice_coordinate_cosine': (max(-1.0, min(1.0, shared_inner/(shared_lattice*shared_coordinate)))
                                              if shared_lattice*shared_coordinate else None),
        'reconstructed_total_preclip_norm': math.sqrt(max(combined_squared, 0.0)),
    }
