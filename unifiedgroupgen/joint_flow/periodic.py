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
        edges,shifts,neighbors,vectors=radius_graph(f,cells,counts,periodic,self.cutoff,self.max_neighbors,self.max_images,
                                                   keep_shells=True,allow_large_images=True)
        active=neighbors>0
        # An edgeless graph must have the same local readout alone or packed
        # beside other graphs. Its scalar embedding remains conditioned once;
        # an equivariant vector without any geometric direction is zero.
        isolated=~active[batch]
        isolated_scalar=None
        if isolated.any():
            self._node_context=self.context(node_context[isolated])
            try:isolated_scalar=self.core.sphere_embedding(numbers[isolated])
            finally:self._node_context,self._scalar_nodes=None,None
        if not active.any():
            return f.new_zeros((len(f),3)),isolated_scalar
        graph=torch.repeat_interleave(torch.arange(len(live),device=f.device),neighbors)
        positions=torch.cat([d['cartesian_coordinates'] for d in decoded])
        active_nodes=~isolated
        node_map=active_nodes.long().cumsum(0)-1
        graph_map=active.long().cumsum(0)-1
        data=Data(pos=positions[active_nodes],atomic_numbers=numbers[active_nodes],natoms=counts[active],
            batch=graph_map[batch[active_nodes]],cell=cells[active],pbc=periodic[active],edge_index=node_map[edges],cell_offsets=shifts,
            edge_vectors=vectors,offset_vectors=torch.einsum('ni,nij->nj',shifts.to(f),cells[graph]),
            neighbors=neighbors[active])
        def core_forward(context):
            self._node_context=self.context(context)
            try:
                output=self.core(data)
                return output['forces'],self._scalar_nodes
            finally:self._node_context,self._scalar_nodes=None,None
        if getattr(self,'checkpoint_layers',False) and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            vector,scalar=checkpoint(core_forward,node_context[active_nodes],use_reentrant=False)
        else:vector,scalar=core_forward(node_context[active_nodes])
        if isolated_scalar is None:return vector,scalar
        full_vector=vector.new_zeros((len(f),3));full_scalar=scalar.new_zeros((len(f),h))
        full_vector[active_nodes]=vector;full_scalar[active_nodes]=scalar;full_scalar[isolated]=isolated_scalar
        return full_vector,full_scalar
