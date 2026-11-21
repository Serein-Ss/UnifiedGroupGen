"""MACE product-basis message passing with direct crystal generation readouts."""
import math

import torch
from e3nn import o3
from e3nn import nn as e3nn_nn
from torch import nn
from torch.nn import functional as F

from .official.mace.symmetric_contraction import SymmetricContraction
from .periodic_inputs import PeriodicGraphFeatures


class MACEInteraction(nn.Module):
    def __init__(self, nodes, spherical, interaction, width, basis, correlation, species):
        super().__init__()
        self.up = o3.Linear(nodes, nodes)
        irreps, instructions = [], []
        for i, (multiplicity, node_irrep) in enumerate(nodes):
            for j, (_, harmonic) in enumerate(spherical):
                for irrep in node_irrep * harmonic:
                    if irrep in interaction:
                        instructions.append((i, j, len(irreps), 'uvu', True))
                        irreps.append((multiplicity, irrep))
        intermediate = o3.Irreps(irreps)
        self.message = o3.TensorProduct(nodes, spherical, intermediate, instructions,
                                        internal_weights=False, shared_weights=False)
        self.radial = e3nn_nn.FullyConnectedNet(
            [basis, width, width, self.message.weight_numel], F.silu)
        self.after_scatter = o3.Linear(intermediate, interaction)
        self.product = SymmetricContraction(interaction, interaction, correlation=correlation,
                                            num_elements=species, shared_weights=True)
        self.mix = o3.Linear(interaction, interaction)
        self.skip = o3.FullyConnectedTensorProduct(nodes, f'{species}x0e', interaction)
        self.width = width
        self.sections = interaction.slices()

    def forward(self, hidden, attributes, source, target, harmonics, radial, normalization):
        message = self.message(self.up(hidden)[source], harmonics, self.radial(radial))
        pooled = hidden.new_zeros((len(hidden), message.shape[-1])).index_add(0, target, message)
        pooled = self.after_scatter(pooled / normalization)
        by_channel = torch.cat([pooled[:, section].reshape(len(hidden), self.width, -1)
                                for section in self.sections], dim=-1)
        return self.mix(self.product(by_channel, attributes)) + self.skip(hidden, attributes)


class MACEFeatures(PeriodicGraphFeatures):
    def __init__(self, options):
        super().__init__(options)
        width = self.channels
        self.lmax = options.get('lmax', 2)
        self.average_neighbors = options.get('average_neighbors', 32.)
        self.spherical = o3.Irreps.spherical_harmonics(self.lmax)
        irreps = o3.Irreps([(width, irrep) for _, irrep in self.spherical])
        radial_basis = options.get('radial_basis', 8)
        self.register_buffer('frequencies', math.pi*torch.arange(1, radial_basis+1))
        species = sorted({0, *options['allowed_elements']})
        lookup = torch.zeros(119, dtype=torch.long)
        lookup[species] = torch.arange(len(species))
        self.register_buffer('species_index', lookup)
        self.species = len(species)
        self.elements = nn.Embedding(self.species, width)
        self.layers = nn.ModuleList(
            MACEInteraction(o3.Irreps(f'{width}x0e') if i == 0 else irreps,
                            self.spherical, irreps, width, radial_basis,
                            options.get('correlation', 3), self.species)
            for i in range(options.get('layers', 4)))

    def embed(self, numbers):
        return self.elements(self.species_index[numbers])

    def features(self, atoms, edges, vectors, condition):
        attributes = F.one_hot(self.species_index[atoms.numbers], self.species).to(condition)
        hidden = self.embed(atoms.numbers) + condition
        # Polynomial-envelope Bessel basis and direction harmonics on periodic images.
        length = vectors.norm(dim=-1, keepdim=True)
        scaled = length/self.cutoff
        envelope = (1 - scaled).clamp_min(0).pow(3) * (1 + 3*scaled + 6*scaled.square())
        radial = math.sqrt(2/self.cutoff)*torch.sin(scaled*self.frequencies)/length*envelope
        harmonics = o3.spherical_harmonics(self.spherical, vectors, normalize=True,
                                             normalization='component')
        for layer in self.layers:
            hidden = layer(hidden, attributes, edges[0], edges[1], harmonics, radial,
                           self.average_neighbors)
        scalar = hidden[:, :self.channels]
        vector = hidden[:, self.channels:4*self.channels].reshape(len(hidden), self.channels, 3)
        return self.velocity(vector.transpose(1, 2)).squeeze(-1), scalar
