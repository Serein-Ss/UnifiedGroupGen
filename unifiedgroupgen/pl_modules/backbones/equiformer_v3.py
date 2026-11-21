"""EquiformerV3 author core with direct generative readouts on our periodic graph."""
from .official.equiformer_v3.equiformer_v3 import EquiformerV3_OC
from .periodic_inputs import PeriodicGraphFeatures


class EquiformerV3Features(PeriodicGraphFeatures):
    def __init__(self, options):
        super().__init__(options)
        width = self.channels
        lmax, mmax = options.get('lmax', 2), options.get('mmax', 2)
        self.core = EquiformerV3_OC(
            otf_graph=False, regress_forces=False, regress_stress=False,
            max_num_elements=119, max_radius=self.cutoff, max_neighbors=self.max_neighbors,
            num_layers=options.get('layers', 4), num_channels=width, lmax=lmax, mmax=mmax,
            attn_grid_resolution_list=[3*lmax+2, 3*mmax+2],
            ffn_grid_resolution_list=[3*lmax+2, 3*lmax+2],
            num_heads=options.get('heads', 8), attn_hidden_channels=max(16, width//2),
            num_radial_basis=options.get('radial_basis', 64), edge_channels=width,
            ffn_hidden_channels=2*width, attn_weights_drop=0., drop_path_rate=0.)
        # The paper's node energy head is replaced by the shared generative heads.
        del self.core.energy_block

    def embed(self, numbers):
        return self.core.sphere_embedding(numbers)

    def features(self, atoms, edges, vectors, condition):
        core = self.core
        core.dtype, core.device = atoms.positions.dtype, atoms.positions.device
        distance, envelope = core._forward_edge(vectors.norm(dim=-1), vectors)
        x = core._forward_embedding(atoms.numbers, distance, edges, envelope)
        x[:, 0, :] = x[:, 0, :] + condition
        scalar, embedding = core._forward_blocks(
            x, atoms.numbers[edges[0]], atoms.numbers[edges[1]], distance, edges,
            envelope, atoms.batch)
        return self.velocity(embedding[:, 1:4, :]).squeeze(-1), scalar
