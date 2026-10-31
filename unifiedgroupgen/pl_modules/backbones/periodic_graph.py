"""Periodic image graphs for crystals and isolated layers, with live edge gradients."""
import itertools
import torch


def geometry(compiled, states, lattices):
    with torch.autocast(device_type=states[0].device.type, enabled=False):
        return _geometry(compiled, states, lattices)


def _geometry(compiled, states, lattices):
    coords, numbers, periodic = [], [], []
    for chart, z in zip(compiled, states):
        f = chart.tensor(chart.b, z) + chart.coordinate_velocity(z[chart.lattice_dof:])
        mask = chart.tensor(chart.periodic, z, torch.bool)
        coords.append(torch.where(mask, f.remainder(1), f))
        numbers.append(chart.tensor(chart.elements, z, torch.long))
        periodic.append(mask)
    counts = torch.tensor([len(f) for f in coords], device=states[0].device)
    return torch.cat(coords), torch.cat(numbers), counts, torch.stack(periodic)


def radius_graph(coords, cells, counts, periodic, cutoff=6., max_neighbors=50, max_images=16):
    """Cells use row vectors. An edge j→i includes the image of j relative to i.

    Reciprocal-vector bounds include every image in the cutoff; no fixed 27-image
    approximation, and no images along a layer's nonperiodic normal.
    """
    with torch.autocast(device_type=coords.device.type, enabled=False):
        return _radius_graph(coords, cells, counts, periodic, cutoff, max_neighbors, max_images)


def _radius_graph(coords, cells, counts, periodic, cutoff, max_neighbors, max_images):
    all_edges, all_shifts, counts_edges = [], [], []
    offset = 0
    for g, count in enumerate(counts.tolist()):
        cell, f = cells[g], coords[offset:offset+count]
        bounds = torch.ceil(cutoff*torch.linalg.vector_norm(torch.linalg.inv(cell), dim=0)).long()+1
        bounds = torch.where(periodic[g], bounds, torch.zeros_like(bounds))
        if bounds.max() > max_images:
            raise ValueError('Periodic image bound exceeds max_images; increase the explicit graph limit')
        shifts = torch.tensor(list(itertools.product(*(range(-n, n+1) for n in bounds.tolist()))),
                              device=f.device, dtype=torch.long)
        source, target = torch.meshgrid(torch.arange(count, device=f.device),
                                        torch.arange(count, device=f.device), indexing='ij')
        source, target = source.flatten(), target.flatten()
        # Topology does not differentiate through neighbor selection.
        with torch.no_grad():
            distances = torch.linalg.vector_norm((f[source, None]-f[target, None]+shifts) @ cell, dim=-1)
            pairs, images = ((distances < cutoff) & (distances > 1e-6)).nonzero(as_tuple=True)
            selected = []
            for atom in range(count):
                ids = (target[pairs] == atom).nonzero().flatten()
                selected.append(ids[distances[pairs[ids], images[ids]].argsort(stable=True)[:max_neighbors]])
            ids = torch.cat(selected)
            pairs, images = pairs[ids], images[ids]
        all_edges.append(torch.stack((source[pairs]+offset, target[pairs]+offset)))
        all_shifts.append(shifts[images]); counts_edges.append(len(pairs))
        offset += count
    edges, shifts = torch.cat(all_edges, dim=1), torch.cat(all_shifts)
    graph = torch.repeat_interleave(torch.arange(len(counts), device=coords.device),
                                   torch.tensor(counts_edges, device=coords.device))
    vectors = torch.einsum('ei,eij->ej', coords[edges[0]]+shifts-coords[edges[1]], cells[graph])
    return edges, shifts, torch.tensor(counts_edges, device=coords.device), vectors


def fractional_vectors(cartesian, lattices, node_graph):
    with torch.autocast(device_type=cartesian.device.type, enabled=False):
        return torch.linalg.solve(lattices[node_graph], cartesian.to(lattices.dtype).unsqueeze(-1)).squeeze(-1)
