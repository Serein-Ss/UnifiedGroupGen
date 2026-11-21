"""Typed scalar/vector readout from the pinned, unchanged EquiformerV2 core."""
import torch
from unifiedgroupgen.pl_modules.backbones.equiformer_v2 import EquiformerV2Backbone
from unifiedgroupgen.common.data_utils import radius_graph
from unifiedgroupgen.pl_modules.debug import debugging, observe, stage

class EquiformerFeatures(EquiformerV2Backbone):
    def forward_states(self,states,context):
        counts_all=[s.num_atoms for s in states]
        live=[i for i,n in enumerate(counts_all) if n]
        h=self.core.sphere_channels
        if not live:
            observe('periodic_graph', structures=len(states), nodes=0, edges=0)
            return context.new_zeros((0,3)),context.new_zeros((0,h))
        from torch_geometric.data import Data
        decoded=[states[i].decoded() for i in live]
        f=torch.cat([d['coordinates'] for d in decoded])
        numbers=torch.cat([d['atomic_numbers'] for d in decoded])
        lattices=torch.stack([d['lattice'] for d in decoded])
        sizes=torch.tensor([counts_all[i] for i in live])
        counts=sizes.to(f.device)
        periodic=torch.tensor([states[i].spec.periodic for i in live],device=f.device)
        batch=torch.repeat_interleave(torch.arange(len(live),device=f.device),counts,output_size=len(f))
        node_context=context[live][batch]
        cells=lattices.transpose(1,2)
        with stage('graph.radius_search'):
            edges,shifts,neighbors,vectors=radius_graph(f,cells,sizes,periodic,self.cutoff,self.max_neighbors,self.max_images,
                                                       keep_shells=True,allow_large_images=True)
        if debugging():
            observe('periodic_graph', structures=len(live), nodes=len(f), edges=edges.shape[1],
                    edge_index=edges, edge_vectors=vectors, cell_offsets=shifts, lattice=cells)
        graph_has_edges=[n>0 for n in neighbors.tolist()]
        isolated_scalar=None
        if not all(graph_has_edges):
            # An edgeless graph keeps the same local readout alone or in a batch.
            active=torch.tensor(graph_has_edges,device=f.device)
            isolated=~active[batch]
            self._node_context=self.context(node_context[isolated])
            try:isolated_scalar=self.core.sphere_embedding(numbers[isolated])
            finally:self._node_context,self._scalar_nodes=None,None
            if not any(graph_has_edges):
                return f.new_zeros((len(f),3)),isolated_scalar
        graph=batch[edges[1]]
        positions=torch.cat([d['cartesian_coordinates'] for d in decoded])
        offsets=torch.einsum('ni,nij->nj',shifts.to(f),cells[graph])
        if isolated_scalar is not None:
            active_nodes=~isolated
            node_map=active_nodes.long().cumsum(0)-1
            graph_map=active.long().cumsum(0)-1
            positions,numbers,core_context=positions[active_nodes],numbers[active_nodes],node_context[active_nodes]
            counts,cells,periodic,neighbors=counts[active],cells[active],periodic[active],neighbors[active]
            batch,edges=graph_map[batch[active_nodes]],node_map[edges]
        else:
            core_context=node_context
        data=Data(pos=positions,atomic_numbers=numbers,natoms=counts,batch=batch,cell=cells,pbc=periodic,
                  edge_index=edges,cell_offsets=shifts,edge_vectors=vectors,offset_vectors=offsets,
                  neighbors=neighbors)
        def core_forward(context):
            self._node_context=self.context(context)
            try:
                with stage('equiformer.forward'):
                    output=self.core(data)
                return output['forces'],self._scalar_nodes
            finally:self._node_context,self._scalar_nodes=None,None
        if getattr(self,'checkpoint_layers',False) and self.training and torch.is_grad_enabled():
            from torch.utils.checkpoint import checkpoint
            vector,scalar=checkpoint(core_forward,core_context,use_reentrant=False)
        else:vector,scalar=core_forward(core_context)
        if isolated_scalar is None:return vector,scalar
        full_vector=vector.new_zeros((len(f),3));full_scalar=scalar.new_zeros((len(f),h))
        full_vector[active_nodes]=vector;full_scalar[active_nodes]=scalar;full_scalar[isolated]=isolated_scalar
        return full_vector,full_scalar
