"""DiffCSP/DiffCSP++ periodic CSPNet adapted to the joint generator.

Based on Rui Jiao et al., DiffCSP and DiffCSP-PP, MIT license.
Uses the authors' fully connected graph, Fourier displacement embedding,
lattice Gram features and residual mean-aggregated messages.
"""
import math

import torch
from torch import nn

from unifiedgroupgen.pl_modules.backbones.periodic_inputs import pack_atoms
from unifiedgroupgen.pl_modules.debug import stage


class CSPMessageLayer(nn.Module):
    def __init__(self, width, edge_dim):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.edge = nn.Sequential(nn.Linear(2*width + edge_dim, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU())
        self.node = nn.Sequential(nn.Linear(2*width, width), nn.SiLU(),
                                  nn.Linear(width, width), nn.SiLU())

    def forward(self, features, source, target, edge_features, degree):
        normalized = self.norm(features)
        message = self.edge(torch.cat((normalized[source], normalized[target], edge_features), dim=-1))
        aggregate = features.new_zeros(features.shape).index_add(0, source, message.to(features.dtype))
        update = self.node(torch.cat((normalized, aggregate/degree), dim=-1))
        return features + update


class CSPNetFeatures(nn.Module):
    def __init__(self, options):
        super().__init__()
        self.channels = options.get('channels', 128)
        width = self.channels
        frequencies = options.get('fourier_frequencies', 10)
        self.register_buffer('frequencies', 2*math.pi*torch.arange(frequencies))
        self.elements = nn.Embedding(119, width)
        self.context = nn.Linear(options.get('hidden', 128), width, bias=False)
        self.input = nn.Linear(2*width, width)
        # Upper triangle of the Gram matrix, Fourier differences, and real displacements on open axes.
        self.layers = nn.ModuleList(CSPMessageLayer(width, 6 + 6*frequencies + 3)
                                    for _ in range(options.get('layers', 4)))
        self.output = nn.LayerNorm(width)
        self.velocity = nn.Linear(width, 3, bias=False)
        self.register_buffer('gram_indices', torch.triu_indices(3, 3), persistent=False)

    def forward_states(self, states, context):
        atoms = pack_atoms(states, context)
        if atoms is None:
            return context.new_zeros((0, 3)), context.new_zeros((0, self.channels))
        with stage('CSPNetFeatures.forward'):
            pair_sources, pair_targets = [], []
            start = 0
            for count in atoms.counts.tolist():
                indices = torch.arange(start, start + count, device=atoms.numbers.device)
                pair_sources.append(indices.repeat_interleave(count))
                pair_targets.append(indices.repeat(count))
                start += count
            source, target = torch.cat(pair_sources), torch.cat(pair_targets)
            graphs = atoms.batch[source]
            pbc = atoms.periodic[graphs]
            delta = atoms.fractional[target] - atoms.fractional[source]
            angles = (torch.where(pbc, delta.remainder(1), 0.)[..., None]*self.frequencies).flatten(1)
            gram = atoms.cells @ atoms.cells.transpose(-1, -2)
            row, col = self.gram_indices
            features = torch.cat((gram[graphs][:, row, col], angles.sin(), angles.cos(),
                                  torch.where(pbc, 0., delta)), dim=-1)
            hidden = self.input(torch.cat((self.elements(atoms.numbers), self.context(atoms.context)), dim=-1))
            degree = atoms.counts.to(atoms.batch.device)[atoms.batch, None].to(hidden.dtype)
            for layer in self.layers:
                hidden = layer(hidden, source, target, features, degree)
            hidden = self.output(hidden)
            return self.velocity(hidden), hidden
