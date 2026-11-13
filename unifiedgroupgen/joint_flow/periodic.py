"""Typed scalar/vector readout from the pinned, unchanged EquiformerV2 core."""
import torch
from unifiedgroupgen.pl_modules.backbones.equiformer_v2 import EquiformerV2Backbone
from unifiedgroupgen.pl_modules.backbones.periodic_graph import radius_graph

class EquiformerFeatures(EquiformerV2Backbone):
    def forward_states(self,states,context):
        counts_all=[s.num_atoms for s in states]
        live=[i for i,n in enumerate(counts_all) if n]
        h=self.core.sphere_channels
        if not live:return context.new_zeros((0,3)),context.new_zeros((0,h))
        from torch_geometric.data import Data
        decoded=[states[i].decoded() for i in live]
        f=torch.cat([d['coordinates'] for d in decoded])
        numbers=torch.cat([d['atomic_numbers'] for d in decoded])
        lattices=torch.stack([d['lattice'] for d in decoded])
        counts=torch.tensor([counts_all[i] for i in live],device=f.device)
        periodic=torch.tensor([states[i].spec.periodic for i in live],device=f.device)
        batch=torch.repeat_interleave(torch.arange(len(live),device=f.device),counts)
        node_context=torch.repeat_interleave(context[live],counts,dim=0)
        cells=lattices.transpose(1,2)
        edges,shifts,neighbors,vectors=radius_graph(f,cells,counts,periodic,self.cutoff,self.max_neighbors,self.max_images,keep_shells=True)
        if not edges.shape[1]:
            scalar=self.core.sphere_embedding(numbers)+self.context(node_context)
            return f.new_zeros((len(f),3)),scalar
        graph=torch.repeat_interleave(torch.arange(len(live),device=f.device),neighbors)
        data=Data(pos=torch.cat([d['cartesian_coordinates'] for d in decoded]),atomic_numbers=numbers,natoms=counts,
            batch=batch,cell=cells,pbc=periodic,edge_index=edges,cell_offsets=shifts,
            edge_vectors=vectors,offset_vectors=torch.einsum('ni,nij->nj',shifts.to(f),cells[graph]),
            neighbors=neighbors)
        def core_forward(context):
            self._node_context=self.context(context)
            try:
                output=self.core(data)
                return output['forces'],self._scalar_nodes
            finally:self._node_context=None
        if getattr(self,'checkpoint_layers',False) and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            return checkpoint(core_forward,node_context,use_reentrant=False)
        return core_forward(node_context)
