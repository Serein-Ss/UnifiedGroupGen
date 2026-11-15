import math
import torch
from torch import nn
from unifiedgroupgen.conditional.properties import PropertyEncoder

from unifiedgroupgen.pl_modules.backbones.message_passing import mlp, MessageLayer

class OrbitFlow(nn.Module):
    def __init__(self, properties, hidden=128, layers=4, frequencies=8, cond_dropout=0.1,
                 checkpoint_layers=False, edge_chunk_size=8192, backbone_factory=None, candidate_backbone_factory=None):
        super().__init__()
        self.frequencies = frequencies
        self.checkpoint_layers = checkpoint_layers
        self.edge_chunk_size = edge_chunk_size
        self.atom_embedding = nn.Embedding(119, hidden)
        self.orbit_embedding = nn.Embedding(64, hidden)
        self.group_embedding = nn.Embedding(328, hidden)  # 230 SG + 80 LG + 17 PG
        self.property_encoder = PropertyEncoder(properties, hidden, cond_dropout)
        self.time_encoder = mlp(16, hidden, hidden)
        self.lattice_encoder = mlp(9, hidden, hidden)
        # Observation masks and values are hard conditioning, shared in both CFG branches.
        self.observation_encoder = mlp(12, hidden, hidden)
        self.site_encoder = mlp(6, hidden, hidden)
        edge_dim = 6*frequencies+3+9
        if candidate_backbone_factory is None:
            self.layers = nn.ModuleList([(backbone_factory or MessageLayer)(hidden=hidden, edge_dim=edge_dim) for _ in range(layers)])
            self.coordinate_head = nn.Linear(hidden, 3)
            self.lattice_head = nn.Linear(hidden, 9)
        else:
            self.candidate_backbone = candidate_backbone_factory(hidden=hidden)
        self.register_buffer('_time_frequencies', torch.tensor([2**i for i in range(8)]), persistent=False)
        self.register_buffer('_coordinate_frequencies', torch.arange(1, frequencies+1), persistent=False)

    @staticmethod
    def group_id(descriptor):
        offset = {"space": 0, "layer": 230, "plane": 310}[descriptor.kind]
        return offset + descriptor.number

    def _features(self, compiled, state, lattice, global_h, observation, null_observation=None, include_edges=True):
        # Geometry and constraint solves stay FP32/FP64 even under neural-network AMP.
        with torch.autocast(device_type=state.device.type, enabled=False):
            coords = compiled.tensor(compiled.b, state) + compiled.coordinate_velocity(state[compiled.lattice_dof:])
            coords = torch.where(compiled.tensor(compiled.periodic, state, torch.bool), torch.remainder(coords, 1), coords)
            h_metric = lattice.T @ lattice
        letters = compiled.tensor(compiled.orbit_letters, state, torch.long)
        orbit_ids = compiled.tensor(compiled.atom_orbit, state, torch.long)
        h = self.atom_embedding(compiled.tensor(compiled.elements, state, torch.long)) + self.orbit_embedding(letters[orbit_ids]) + global_h
        if observation is None:
            # Biases remain trainable: zero input is not zero encoder output.
            if null_observation is None:
                null_observation = (self.observation_encoder(state.new_zeros(12)),
                                    self.site_encoder(state.new_zeros(6)))
            h = h + null_observation[0]
            h = h + null_observation[1]
        else:
            observed_base, observed_mask = observation.features(state)
            padded_k = state.new_zeros(6)
            padded_mask = state.new_zeros(6)
            padded_k[:compiled.lattice_dof] = observed_base[:compiled.lattice_dof]
            padded_mask[:compiled.lattice_dof] = observed_mask[:compiled.lattice_dof]
            h = h + self.observation_encoder(torch.cat((padded_k, padded_mask)))
            a = compiled.tensor(compiled.coordinate_basis, state)
            observed_pos = torch.einsum('iaq,q->ia', a, observed_base[compiled.lattice_dof:])
            observed_site_mask = torch.einsum('iaq,q->ia', a.abs(), observed_mask[compiled.lattice_dof:])
            h = h + self.site_encoder(torch.cat((observed_pos, observed_site_mask), dim=-1))
        if not include_edges:
            return h, None, None, lattice
        edge_index = compiled.tensor(compiled.edge_index, state, torch.long)
        source, target = edge_index
        diff = coords[target] - coords[source]
        periodic = compiled.tensor(compiled.periodic, state, torch.bool)
        periodic_diff = torch.where(periodic, diff, torch.zeros_like(diff))
        nonperiodic_diff = torch.where(periodic, torch.zeros_like(diff), diff)
        phase = periodic_diff[..., None] * self._coordinate_frequencies.to(state) * (2*math.pi)
        edge_features = torch.cat((phase.sin().flatten(-2), phase.cos().flatten(-2),
                                   nonperiodic_diff, h_metric.reshape(1, 9).expand(len(source), -1)), dim=-1)
        return h, edge_features, edge_index, lattice

    def _project_coordinates(self, compiled, candidates, lattices, states, node_graph):
        """The original metric least-squares projection, solved in orbit blocks."""
        if not any(c.coordinate_dof for c in compiled):
            return [z.new_zeros(0) for z in states], None
        orbit_offsets = [0]
        for c in compiled:
            orbit_offsets.append(orbit_offsets[-1]+len(c.blocks))
        basis = torch.cat([c.tensor(c.projection_basis, z) for c, z in zip(compiled, states)])
        orbit = torch.cat([c.tensor(c.atom_orbit, z, torch.long)+offset
                           for c, z, offset in zip(compiled, states, orbit_offsets)])
        metrics = lattices.transpose(1, 2) @ lattices
        weighted = basis.transpose(1, 2) @ metrics[node_graph]
        gram = states[0].new_zeros((orbit_offsets[-1], 3, 3)).index_add(0, orbit, weighted @ basis)
        rhs = states[0].new_zeros((orbit_offsets[-1], 3, 1)).index_add(
            0, orbit, weighted @ candidates.to(basis.dtype).unsqueeze(-1))
        unused = torch.cat([c.tensor(c.projection_unused, z, torch.bool) for c, z in zip(compiled, states)])
        gram = gram + torch.diag_embed(unused.to(gram.dtype))
        solution, info = torch.linalg.solve_ex(gram, rhs, check_errors=False)
        velocities = [solution[start:stop].reshape(-1)[c.tensor(c.projection_indices, z, torch.long)]
                      for c, z, start, stop in zip(compiled, states, orbit_offsets, orbit_offsets[1:])]
        return velocities, info

    def forward(self, compiled, state, time, properties=None, force_null=False, observation=None, composition=None):
        return self.forward_batch([compiled], [state], [time], [properties or {}], force_null,
                                  [observation], [composition])[0]

    def forward_batch(self, compiled, states, times, properties=None, force_null=False,
                      observations=None, compositions=None, return_lattices=False, property_visibility=None):
        """Packed disjoint crystal graphs; no edges or attention cross graph boundaries."""
        count = len(compiled)
        properties = properties or [{} for _ in compiled]
        observations = observations or [None]*count
        compositions = compositions or [None]*count
        conditions = self.property_encoder.forward_batch(properties, states[0], force_null, compositions,
                                                          visibility=property_visibility)
        with torch.autocast(device_type=states[0].device.type, enabled=False):
            # A padded 3D chart also represents a layer metric with an unchanged unit normal.
            flat = torch.cat([*states, states[0].new_zeros(1)])
            indices, offset = [], 0
            for c in compiled:
                indices.append([*range(offset, offset+c.lattice_dof), *([len(flat)-1]*(6-c.lattice_dof))])
                offset += c.dof
            k = flat[torch.tensor(indices, device=flat.device)]
            bases = torch.stack([c.tensor(c.padded_metric_basis, z) for c, z in zip(compiled, states)])
            roots = torch.stack([c.tensor(c.padded_root, z) for c, z in zip(compiled, states)])
            metrics = roots @ torch.matrix_exp(torch.einsum('bk,bkij->bij', k, bases)) @ roots
            factors, metric_info = torch.linalg.cholesky_ex(metrics, check_errors=False)
            lattices = factors.transpose(1, 2)
            lattice_metrics = lattices.transpose(1, 2) @ lattices
        times = torch.stack([torch.as_tensor(t, device=flat.device, dtype=flat.dtype) for t in times])
        phases = self._time_frequencies.to(flat)*math.pi*times[:, None]
        global_h = self.time_encoder(torch.cat((phases.sin(), phases.cos()), -1))
        global_h = global_h + self.lattice_encoder(lattice_metrics.reshape(count, 9))
        groups = torch.tensor([self.group_id(c.descriptor) for c in compiled], device=flat.device)
        global_h = global_h + self.group_embedding(groups) + conditions
        null_observation = ((self.observation_encoder(flat.new_zeros(12)), self.site_encoder(flat.new_zeros(6)))
                            if any(o is None for o in observations) else None)
        features = [self._features(c, z, lattice, context, o, null_observation,
                                  include_edges=not hasattr(self, 'candidate_backbone'))
                    for c, z, lattice, context, o in zip(compiled, states, lattices, global_h, observations)]
        h = torch.cat([f[0] for f in features])
        offsets = [0]
        for c in compiled:
            offsets.append(offsets[-1]+c.num_atoms)
        counts = torch.tensor([c.num_atoms for c in compiled], device=h.device)
        node_graph = torch.arange(count, device=h.device).repeat_interleave(counts, output_size=offsets[-1])
        degree = (counts-1).repeat_interleave(counts, output_size=offsets[-1]).to(h.dtype)
        if hasattr(self, 'candidate_backbone'):
            candidates, proposals = self.candidate_backbone(compiled, states, lattices, h, global_h, node_graph)
        else:
            edges = torch.cat([f[1] for f in features])
            edge_index = torch.cat([f[2]+offset for f, offset in zip(features, offsets)], dim=1)
            for layer in self.layers:
                if self.checkpoint_layers and self.training and torch.is_grad_enabled():
                    from torch.utils.checkpoint import checkpoint
                    h = checkpoint(layer, h, edges, edge_index, degree, self.edge_chunk_size, use_reentrant=False)
                else:
                    h = layer(h, edges, edge_index, degree, self.edge_chunk_size)
            candidates = self.coordinate_head(h)
            pooled = h.new_zeros(count, h.shape[-1]).index_add(0, node_graph, h)/counts[:, None]
            proposals = self.lattice_head(pooled).reshape(count, 3, 3)
        velocities = []
        with torch.autocast(device_type=h.device.type, enabled=False):
            q_velocities, projection_info = self._project_coordinates(compiled, candidates, lattices, states, node_graph)
            # Combine CUDA error checks into one synchronization, without hiding failures.
            infos = [metric_info.reshape(-1)]
            if projection_info is not None:
                infos.append(projection_info.reshape(-1))
            if torch.cat(infos).ne(0).any():
                raise torch.linalg.LinAlgError('Metric factorization or orbit projection failed')
            proposed = proposals.to(flat.dtype)
            k_velocities = torch.einsum('bkij,bij->bk', bases, (proposed+proposed.transpose(1, 2))/2)
            for i, (c, observation) in enumerate(zip(compiled, observations)):
                velocity = torch.cat((k_velocities[i, :c.lattice_dof], q_velocities[i]))
                velocities.append(observation.project(velocity) if observation is not None else velocity)
        return (velocities, [feature[3] for feature in features]) if return_lattices else velocities
