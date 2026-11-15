"""Periodic image graphs for crystals and isolated layers, with live edge gradients."""
import itertools
from functools import lru_cache
import torch


@lru_cache(maxsize=64)
def _image_shifts(bounds):
    return torch.tensor(list(itertools.product(*(range(-n,n+1) for n in bounds))),dtype=torch.long)


_IMAGE_DISTANCE_BUDGET = 1 << 20


def _shift_chunks(bounds, count, device):
    shifts = itertools.product(*(range(-n,n+1) for n in bounds))
    width = min(65536, max(1, _IMAGE_DISTANCE_BUDGET // max(count*count, 1)))
    while chunk := list(itertools.islice(shifts, width)):
        yield torch.tensor(chunk, dtype=torch.long, device=device)


def _large_image_neighbors(f, cell, bounds, source, target, cutoff, max_neighbors, keep_shells):
    """Exact two-pass selection without materializing the full image cube.

    First find each target's kth distance, then retain its complete boundary
    shell. Scratch space is chunked; time and retained shells still depend on
    the actual geometry. The caller disables gradients for topology selection.
    """
    count = len(f)
    best = f.new_full((count, max_neighbors), torch.inf)
    delta = f[source, None]-f[target, None]
    for shifts in _shift_chunks(bounds, count, f.device):
        distances = torch.linalg.vector_norm((delta+shifts) @ cell, dim=-1)
        valid = (distances < cutoff) & (distances > 1e-6)
        distances = distances.masked_fill(~valid, torch.inf)
        by_target = distances.reshape(count,count,-1).transpose(0,1).reshape(count,-1)
        best = torch.cat((best,by_target),dim=1).topk(max_neighbors,dim=1,largest=False).values
    boundary = best.max(dim=1).values
    pairs_all, shifts_all, distances_all = [], [], []
    for shifts in _shift_chunks(bounds, count, f.device):
        distances = torch.linalg.vector_norm((delta+shifts) @ cell, dim=-1)
        limit = boundary[target,None]+(1e-5 if keep_shells else 0.)
        pairs, images = ((distances < cutoff) & (distances > 1e-6) & (distances <= limit)).nonzero(as_tuple=True)
        pairs_all.append(pairs); shifts_all.append(shifts[images]); distances_all.append(distances[pairs,images])
    pairs, shifts, distances = torch.cat(pairs_all), torch.cat(shifts_all), torch.cat(distances_all)
    # Restore dense pair/image enumeration as the tie breaker, independently of
    # chunk boundaries: target, distance, pair index, lexicographic image.
    order = torch.arange(len(pairs),device=f.device)
    for key in (shifts[:,2],shifts[:,1],shifts[:,0],pairs,distances,target[pairs]):
        order = order[key[order].argsort(stable=True)]
    if not keep_shells:
        degree = torch.bincount(target[pairs],minlength=count)
        starts = degree.cumsum(0)-degree
        rank = torch.arange(len(order),device=f.device)-torch.repeat_interleave(starts,degree)
        order = order[rank < max_neighbors]
    return pairs[order], shifts[order]


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


def radius_graph(coords, cells, counts, periodic, cutoff=6., max_neighbors=50, max_images=16, keep_shells=False,
                 allow_large_images=False):
    """Cells use row vectors. An edge j→i includes the image of j relative to i.

    Reciprocal-vector bounds include every image in the cutoff; no fixed 27-image
    approximation, and no images along a layer's nonperiodic normal.
    Legacy callers retain the explicit image guard. Joint-flow callers can opt
    into exact chunked search when that guard would otherwise reject a state.
    """
    with torch.autocast(device_type=coords.device.type, enabled=False):
        return _radius_graph(coords, cells, counts, periodic, cutoff, max_neighbors, max_images, keep_shells,
                             allow_large_images)


def _radius_graph(coords, cells, counts, periodic, cutoff, max_neighbors, max_images, keep_shells=False,
                  allow_large_images=False):
    all_edges, all_shifts, counts_edges = [], [], []
    offset = 0
    for g, count in enumerate(counts.tolist()):
        cell, f = cells[g], coords[offset:offset+count]
        bounds = torch.ceil(cutoff*torch.linalg.vector_norm(torch.linalg.inv(cell), dim=0)).long()+1
        bounds = torch.where(periodic[g], bounds, torch.zeros_like(bounds))
        if bounds.max() > max_images:
            if not allow_large_images:
                raise ValueError('Periodic image bound exceeds max_images; increase the explicit graph limit')
            source, target = torch.meshgrid(torch.arange(count,device=f.device),
                                           torch.arange(count,device=f.device),indexing='ij')
            source, target = source.flatten(), target.flatten()
            with torch.no_grad():
                pairs, shifts = _large_image_neighbors(f,cell,tuple(bounds.tolist()),source,target,
                                                       cutoff,max_neighbors,keep_shells)
            all_edges.append(torch.stack((source[pairs]+offset,target[pairs]+offset)))
            all_shifts.append(shifts); counts_edges.append(len(pairs))
            offset += count
            continue
        shifts = _image_shifts(tuple(bounds.tolist())).to(device=f.device)
        source, target = torch.meshgrid(torch.arange(count, device=f.device),
                                        torch.arange(count, device=f.device), indexing='ij')
        source, target = source.flatten(), target.flatten()
        # Topology does not differentiate through neighbor selection.
        with torch.no_grad():
            distances = torch.linalg.vector_norm((f[source, None]-f[target, None]+shifts) @ cell, dim=-1)
            pairs, images = ((distances < cutoff) & (distances > 1e-6)).nonzero(as_tuple=True)
            # Stable two-key ordering exactly matches each target's old sort.
            order = distances[pairs,images].argsort(stable=True)
            order = order[target[pairs[order]].argsort(stable=True)]
            degree = torch.bincount(target[pairs],minlength=count)
            starts = degree.cumsum(0)-degree
            rank = torch.arange(len(order),device=f.device)-torch.repeat_interleave(starts,degree)
            if keep_shells and len(order):
                boundary_ids = (starts+degree.clamp(max=max_neighbors)-1).clamp_min(0)
                boundary = distances[pairs[order[boundary_ids]],images[order[boundary_ids]]]
                keep = distances[pairs[order],images[order]] <= boundary[target[pairs[order]]]+1e-5
            else:
                keep = rank < max_neighbors
            ids = order[keep]
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
