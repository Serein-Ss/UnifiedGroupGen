"""Official DiffCSP CSPNet core with explicit layer-periodicity adaptation."""
from types import MethodType
import torch
from torch import nn
from .periodic_graph import geometry


class CSPNetBackbone(nn.Module):
    def __init__(self, hidden=512, layers=6, frequencies=128, layer_norm=True):
        super().__init__()
        from .official.cspnet import CSPNet
        self.core = CSPNet(hidden_dim=hidden, latent_dim=hidden, num_layers=layers, max_atoms=119,
                           num_freqs=frequencies, ln=layer_norm, ip=True, edge_style='fc')
        self._node_context = None
        self._periodic = None
        self.core.node_embedding.register_forward_hook(self._condition_nodes)
        self.core.gen_edges = MethodType(self._edges, self.core)

    def _condition_nodes(self, module, inputs, result):
        return result + self._node_context if self._node_context is not None else result

    def _edges(self, core, num_atoms, frac_coords, lattices, node2graph):
        edges, offset = [], 0
        for count in num_atoms.tolist():
            source, target = torch.meshgrid(torch.arange(count, device=frac_coords.device),
                torch.arange(count, device=frac_coords.device), indexing='ij')
            edges.append(torch.stack((source.flatten()+offset, target.flatten()+offset))); offset += count
        index = torch.cat(edges, dim=1)
        difference = frac_coords[index[1]]-frac_coords[index[0]]
        difference = torch.where(self._periodic[node2graph[index[0]]], difference.remainder(1), difference)
        return index, difference

    def forward(self, compiled, states, lattices, node_context, global_context, node_graph):
        f, numbers, counts, self._periodic = geometry(compiled, states, lattices)
        self._node_context = node_context
        try:
            matrix, velocity = self.core(global_context, numbers, f, lattices.transpose(1, 2), counts, node_graph)
        finally:
            self._node_context = None
        return velocity, matrix
