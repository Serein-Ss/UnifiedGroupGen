"""Packed crystal inputs for joint-flow backbones with direct velocity readouts."""
from dataclasses import dataclass

import torch
from torch import nn

from unifiedgroupgen.common.data_utils import radius_graph
from unifiedgroupgen.pl_modules.debug import debugging, observe, stage


@dataclass
class PackedAtoms:
    fractional: torch.Tensor
    positions: torch.Tensor
    cells: torch.Tensor
    numbers: torch.Tensor
    periodic: torch.Tensor
    counts: torch.Tensor
    batch: torch.Tensor
    context: torch.Tensor


def pack_atoms(states, context):
    live = [i for i, state in enumerate(states) if state.num_atoms]
    if not live:
        return None
    decoded = [states[i].decoded() for i in live]
    fractional = torch.cat([value['coordinates'] for value in decoded])
    counts = torch.tensor([states[i].num_atoms for i in live], device='cpu')
    batch = torch.repeat_interleave(
        torch.arange(len(live), device=fractional.device),
        counts.to(fractional.device), output_size=len(fractional))
    return PackedAtoms(
        fractional, torch.cat([value['cartesian_coordinates'] for value in decoded]),
        torch.stack([value['lattice'].T for value in decoded]),
        torch.cat([value['atomic_numbers'] for value in decoded]),
        torch.tensor([states[i].spec.periodic for i in live], device=fractional.device),
        counts, batch, context[live][batch])


class PeriodicGraphFeatures(nn.Module):
    """Shared radius graph and Cartesian vector/scalar readout contract."""
    def __init__(self, options):
        super().__init__()
        self.channels = options.get('channels', 128)
        self.cutoff = options.get('cutoff', 6.)
        self.max_neighbors = options.get('max_neighbors', 64)
        self.max_images = options.get('max_images', 16)
        self.context = nn.Linear(options.get('hidden', 128), self.channels, bias=False)
        self.velocity = nn.Linear(self.channels, 1, bias=False)

    def forward_states(self, states, context):
        # Keep equivariant geometry and the reference kernels in the model dtype.
        with torch.autocast(device_type=context.device.type, enabled=False):
            return self._forward_states(states, context.to(self.context.weight.dtype))

    def _forward_states(self, states, context):
        atoms = pack_atoms(states, context)
        if atoms is None:
            return context.new_zeros((0, 3)), context.new_zeros((0, self.channels))
        with stage('graph.radius_search'):
            edges, shifts, neighbors, vectors = radius_graph(
                atoms.fractional, atoms.cells, atoms.counts, atoms.periodic,
                self.cutoff, self.max_neighbors, self.max_images, keep_shells=True,
                allow_large_images=True)
        if debugging():
            observe('periodic_graph', structures=len(atoms.counts), nodes=len(atoms.numbers),
                    edges=edges.shape[1], edge_index=edges, edge_vectors=vectors,
                    cell_offsets=shifts, lattice=atoms.cells)
        condition = self.context(atoms.context)
        if not edges.shape[1]:
            return condition.new_zeros((len(condition), 3)), self.embed(atoms.numbers) + condition
        with stage(type(self).__name__ + '.forward'):
            vector, scalar = self.features(atoms, edges, vectors, condition)
        connected = (neighbors > 0)[atoms.batch, None]
        initial = self.embed(atoms.numbers) + condition
        return torch.where(connected, vector, 0.), torch.where(connected, scalar, initial)
