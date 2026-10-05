"""CSPNet-inspired dense message passing and flow matching in true orbit coordinates.

The network proposes a velocity. The compiled chart enforces sample-level group
constraints; the bare message network is not claimed to be SO(3)-equivariant.
"""

import math
import torch
from torch import nn
from .conditioning import PropertyEncoder


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class MessageLayer(nn.Module):
    def __init__(self, hidden, edge_dim):
        super().__init__()
        self.edge = mlp(2*hidden+edge_dim, hidden, hidden)
        self.node = mlp(2*hidden, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edges, edge_index, degree, chunk_size=8192):
        aggregate = torch.zeros_like(h)
        for start in range(0, edges.shape[0], chunk_size):
            source, target = edge_index[:, start:start+chunk_size]
            messages = self.edge(torch.cat((h[source], h[target], edges[start:start+chunk_size]), dim=-1))
            aggregate = aggregate.index_add(0, source, messages.to(aggregate.dtype))
        aggregate = aggregate / degree[:, None].clamp_min(1)
        return self.norm(h + self.node(torch.cat((h, aggregate), dim=-1)))


class OrbitFlow(nn.Module):
    def __init__(self, properties, hidden=128, layers=4, frequencies=8, cond_dropout=0.1,
                 checkpoint_layers=False, edge_chunk_size=8192):
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
        self.layers = nn.ModuleList([MessageLayer(hidden, edge_dim) for _ in range(layers)])
        self.coordinate_head = nn.Linear(hidden, 3)
        self.lattice_head = nn.Linear(hidden, 9)

    @staticmethod
    def group_id(descriptor):
        offset = {"space": 0, "layer": 230, "plane": 310}[descriptor.kind]
        return offset + descriptor.number

    def _features(self, compiled, state, time, properties, force_null, observation, composition):
        # Geometry and constraint solves stay FP32/FP64 even under neural-network AMP.
        with torch.autocast(device_type=state.device.type, enabled=False):
            structure = compiled.expand(state)
        coords, lattice = structure["coordinates"], structure["lattice"]
        h_metric = lattice.T @ lattice
        phases = state.new_tensor([2**i for i in range(8)]) * math.pi * time
        global_h = self.time_encoder(torch.cat((phases.sin(), phases.cos())))
        global_h = global_h + self.lattice_encoder(h_metric.reshape(-1))
        global_h = global_h + self.group_embedding(torch.tensor(self.group_id(compiled.descriptor), device=state.device))
        global_h = global_h + self.property_encoder(properties or {}, state, force_null, composition)
        letters = torch.tensor([ord(o.letter)-ord('a') if o.letter.islower() else 26+ord(o.letter)-ord('A')
                                for o in compiled.descriptor.orbits], device=state.device)
        orbit_ids = structure["orbit_index"]
        h = self.atom_embedding(structure["atomic_numbers"]) + self.orbit_embedding(letters[orbit_ids]) + global_h
        observed_base, observed_mask = (observation.features(state) if observation is not None
                                        else (torch.zeros_like(state), torch.zeros_like(state)))
        padded_k = state.new_zeros(6)
        padded_mask = state.new_zeros(6)
        padded_k[:compiled.lattice_dof] = observed_base[:compiled.lattice_dof]
        padded_mask[:compiled.lattice_dof] = observed_mask[:compiled.lattice_dof]
        h = h + self.observation_encoder(torch.cat((padded_k, padded_mask)))
        # Per-site physical observed position; mask amplitude also signals which orbit is constrained.
        a = compiled.tensor(compiled.coordinate_basis, state)
        observed_pos = torch.einsum('iaq,q->ia', a, observed_base[compiled.lattice_dof:])
        observed_site_mask = torch.einsum('iaq,q->ia', a.abs(), observed_mask[compiled.lattice_dof:])
        h = h + self.site_encoder(torch.cat((observed_pos, observed_site_mask), dim=-1))
        n = len(h)
        source = torch.arange(n, device=h.device).repeat_interleave(n-1)
        target = torch.arange(n-1, device=h.device).repeat(n)
        target = target + (target >= source).to(target.dtype)
        diff = coords[target] - coords[source]
        periodic = torch.as_tensor(compiled.periodic, device=state.device)
        periodic_diff = torch.where(periodic, diff, torch.zeros_like(diff))
        nonperiodic_diff = torch.where(periodic, torch.zeros_like(diff), diff)
        phase = periodic_diff[..., None] * state.new_tensor(range(1, self.frequencies+1)) * (2*math.pi)
        edge_features = torch.cat((phase.sin().flatten(-2), phase.cos().flatten(-2),
                                   nonperiodic_diff, h_metric.reshape(1, 9).expand(len(source), -1)), dim=-1)
        return h, edge_features, torch.stack((source, target)), lattice

    def forward(self, compiled, state, time, properties=None, force_null=False, observation=None, composition=None):
        return self.forward_batch([compiled], [state], [time], [properties or {}], force_null,
                                  [observation], [composition])[0]

    def forward_batch(self, compiled, states, times, properties=None, force_null=False,
                      observations=None, compositions=None, return_lattices=False):
        """Packed disjoint crystal graphs; no edges or attention cross graph boundaries."""
        count = len(compiled)
        properties = properties or [{} for _ in compiled]
        observations = observations or [None]*count
        compositions = compositions or [None]*count
        features = [self._features(c, z, t, p, force_null, o, comp)
                    for c, z, t, p, o, comp in zip(compiled, states, times, properties, observations, compositions)]
        h = torch.cat([f[0] for f in features])
        edges = torch.cat([f[1] for f in features])
        offsets = [0]
        for c in compiled:
            offsets.append(offsets[-1]+c.num_atoms)
        counts = torch.tensor([c.num_atoms for c in compiled], device=h.device)
        node_graph = torch.arange(count, device=h.device).repeat_interleave(counts, output_size=offsets[-1])
        edge_index = torch.cat([f[2]+offset for f, offset in zip(features, offsets)], dim=1)
        degree = (counts-1).repeat_interleave(counts, output_size=offsets[-1]).to(h.dtype)
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
            for i, (c, z, feature, observation) in enumerate(zip(compiled, states, features, observations)):
                candidate = candidates[offsets[i]:offsets[i+1]].to(z.dtype)
                q_velocity = c.reduce_velocity(candidate, feature[3])
                d = c.root.shape[0]
                proposed = proposals[i, :d, :d].to(z.dtype)
                proposed = (proposed+proposed.T)/2
                k_velocity = torch.einsum('kij,ij->k', c.tensor(c.metric_basis, z), proposed)
                velocity = torch.cat((k_velocity, q_velocity))
                velocities.append(observation.project(velocity) if observation is not None else velocity)
        return (velocities, [feature[3] for feature in features]) if return_lattices else velocities


def conditional_path(compiled, endpoint, time, observation=None, generator=None):
    start = compiled.prior(endpoint.device, endpoint.dtype, generator)
    if observation is not None:
        start = observation.retract(start)
        if not torch.allclose(observation.retract(endpoint), endpoint, atol=1e-5):
            raise ValueError("Training endpoint does not satisfy the observations")
        # General linear combinations need a consistent lift. Independent wrapping could leave Az=b.
        displacement = endpoint-start
    else:
        periodic = torch.as_tensor(compiled.state_periodic, device=endpoint.device)
        delta = endpoint-start
        displacement = torch.where(periodic, torch.remainder(delta+0.5, 1)-0.5, delta)
    return start + time*displacement, displacement


def flow_loss(model, compiled, endpoint, properties=None, observation=None, composition=None, time=None):
    t = torch.rand((), device=endpoint.device) if time is None else endpoint.new_tensor(time)
    state, target = conditional_path(compiled, endpoint, t, observation)
    prediction = model(compiled, state, t, properties, observation=observation, composition=composition)
    return velocity_loss(compiled, state, target, prediction)


def velocity_loss(compiled, state, target, prediction, lattice=None):
    error = prediction-target
    lattice_error = error[:compiled.lattice_dof].square().mean()
    if compiled.coordinate_dof:
        lattice = compiled.expand(state)["lattice"] if lattice is None else lattice
        physical_error = compiled.coordinate_velocity(error[compiled.lattice_dof:]) @ lattice.T
        coordinate_error = physical_error.square().sum()/max(compiled.coordinate_dof, 1)
    else:
        coordinate_error = lattice_error*0
    return {"loss": lattice_error+coordinate_error, "lattice": lattice_error, "coordinates": coordinate_error}


def flow_loss_batch(model, compiled, endpoints, properties, observations=None, compositions=None, times=None):
    observations = observations or [None]*len(compiled)
    times = times if times is not None else torch.rand(len(compiled), device=endpoints[0].device)
    with torch.autocast(device_type=endpoints[0].device.type, enabled=False):
        paths = [conditional_path(c, z, t, o) for c, z, t, o in zip(compiled, endpoints, times, observations)]
    states, targets = zip(*paths)
    predictions, lattices = model.forward_batch(compiled, states, times, properties, observations=observations,
                                                compositions=compositions, return_lattices=True)
    with torch.autocast(device_type=endpoints[0].device.type, enabled=False):
        losses = [velocity_loss(c, z, target, prediction, lattice) for c, z, target, prediction, lattice
                  in zip(compiled, states, targets, predictions, lattices)]
    return {key: torch.stack([loss[key] for loss in losses]) for key in ('loss', 'lattice', 'coordinates')}


@torch.no_grad()
def sample_flow(model, compiled, properties=None, steps=64, guidance=1.0, observation=None,
                composition=None, generator=None, return_trajectory=False):
    if model.training:
        raise RuntimeError("Call eval() before sampling")
    if steps < 1:
        raise ValueError("steps must be positive")
    parameter = next(model.parameters())
    state = compiled.prior(parameter.device, parameter.dtype, generator)
    if observation is not None:
        state = observation.retract(state)
    trajectory = [state.clone()] if return_trajectory else None
    def velocity(z, t):
        if guidance == 0:
            return model(compiled, z, t, properties, force_null=True, observation=observation, composition=composition)
        conditional = model(compiled, z, t, properties, observation=observation, composition=composition)
        if guidance == 1:
            return conditional
        unconditional = model(compiled, z, t, properties, force_null=True, observation=observation, composition=composition)
        return unconditional + guidance*(conditional-unconditional)
    dt = 1/steps
    for step in range(steps):
        t = step*dt
        v1 = velocity(state, t)
        trial = state + dt*v1
        v2 = velocity(trial, t+dt)
        state = state + dt*(v1+v2)/2
        if observation is not None:
            state = observation.retract(state)
        if not torch.isfinite(state).all():
            raise FloatingPointError("Non-finite flow state; sample was not repaired or silently clipped")
        if trajectory is not None:
            trajectory.append(state.clone())
    result = compiled.expand(state)
    result["state"] = state
    if trajectory is not None:
        result["trajectory"] = torch.stack(trajectory)
    return result
