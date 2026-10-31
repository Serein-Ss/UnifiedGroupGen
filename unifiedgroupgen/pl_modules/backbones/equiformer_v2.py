"""Official FAIRChem EquiformerV2 core, periodic graph and generative readout adapters."""
from types import MethodType
import torch
from torch import nn
from .periodic_graph import geometry, radius_graph, fractional_vectors


class EquiformerV2Backbone(nn.Module):
    def __init__(self, hidden=512, layers=12, channels=128, lmax=6, mmax=2,
                 cutoff=6., max_neighbors=50, max_images=16):
        super().__init__()
        try:
            from .official.fairchem.core.models.equiformer_v2.equiformer_v2_deprecated import EquiformerV2
        except ImportError as error:
            raise ImportError('Install this project environment; the pinned EqV2 core is bundled') from error
        self.core = EquiformerV2(use_pbc=True, otf_graph=False, regress_forces=True,
            max_num_elements=119, num_layers=layers, sphere_channels=channels, num_heads=8,
            attn_hidden_channels=channels, attn_alpha_channels=32, attn_value_channels=16,
            ffn_hidden_channels=channels*4, edge_channels=channels, lmax_list=[lmax], mmax_list=[mmax],
            max_radius=cutoff, max_neighbors=max_neighbors)
        self.context = nn.Linear(hidden, channels, bias=False)
        self.lattice_head = nn.Linear(channels, 9)
        self.cutoff, self.max_neighbors, self.max_images = cutoff, max_neighbors, max_images
        self._node_context, self._scalar_nodes = None, None
        self.core.sphere_embedding.register_forward_hook(self._condition)
        self.core.norm.register_forward_hook(self._capture)
        self.core.generate_graph = MethodType(self._graph, self.core)
        for module in self.core.modules():
            if module.__class__.__module__.endswith('.layer_norm'):
                module.register_forward_pre_hook(self._normalization_float32)

    @staticmethod
    def _normalization_float32(module, inputs):
        # Upstream disables autocast here but does not promote the BF16 input.
        # Cast only the sensitive normalization, leaving linear/attention work in AMP.
        return (inputs[0].float(), *inputs[1:])

    def _condition(self, module, inputs, result):
        return result+self._node_context

    def _capture(self, module, inputs, result):
        self._scalar_nodes = result[:, 0]

    def _graph(self, core, data, **kwargs):
        from .official.fairchem.core.models.base import GraphData
        return GraphData(data.edge_index, data.edge_vectors.norm(dim=-1), data.edge_vectors,
                         data.cell_offsets, data.offset_vectors, data.neighbors,
                         data.batch, data.atomic_numbers)

    def forward(self, compiled, states, lattices, node_context, global_context, node_graph):
        from torch_geometric.data import Data
        f, numbers, counts, periodic = geometry(compiled, states, lattices)
        cells = lattices.transpose(1, 2)
        edges, shifts, neighbors, vectors = radius_graph(f, cells, counts, periodic, self.cutoff,
                                                         self.max_neighbors, self.max_images)
        graph = torch.repeat_interleave(torch.arange(len(counts), device=f.device), neighbors)
        with torch.autocast(device_type=f.device.type, enabled=False):
            positions = torch.einsum('ni,nij->nj', f, cells[node_graph])
            offset_vectors = torch.einsum('ni,nij->nj', shifts.to(f), cells[graph])
        data = Data(pos=positions, atomic_numbers=numbers,
            natoms=counts, batch=node_graph, cell=cells, pbc=periodic, edge_index=edges,
            cell_offsets=shifts, neighbors=neighbors, edge_vectors=vectors,
            offset_vectors=offset_vectors)
        self._node_context = self.context(node_context)
        try:
            result = self.core(data)
            pooled = f.new_zeros((len(counts), self._scalar_nodes.shape[-1])).index_add(0, node_graph, self._scalar_nodes)
            matrix = self.lattice_head(pooled/counts[:, None]).reshape(-1, 3, 3)
        finally:
            self._node_context, self._scalar_nodes = None, None
        return fractional_vectors(result['forces'], lattices, node_graph), matrix
