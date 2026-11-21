"""FAIRChem eSEN core with direct velocity and scalar readouts."""
from .official.fairchem.core.models.esen.esen import eSEN_Backbone
from .periodic_inputs import PeriodicGraphFeatures


class _PrecomputedESEN(eSEN_Backbone):
    def generate_graph(self, data):
        return data['graph']


class ESENFeatures(PeriodicGraphFeatures):
    def __init__(self, options):
        super().__init__(options)
        self.core = _PrecomputedESEN(
            max_num_elements=119, sphere_channels=self.channels,
            lmax=options.get('lmax', 2), mmax=options.get('mmax', 2),
            num_layers=options.get('layers', 4), hidden_channels=self.channels,
            otf_graph=True, max_neighbors=self.max_neighbors, cutoff=self.cutoff,
            edge_channels=self.channels, distance_function='bessel',
            num_distance_basis=options.get('radial_basis', 64),
            direct_forces=True, regress_forces=False, regress_stress=False,
            act_type='gate', mlp_type='spectral', use_envelope=True)
        self._condition = None
        self.core.sphere_embedding.register_forward_hook(self._condition_atoms)

    def _condition_atoms(self, module, args, result):
        return result if self._condition is None else result + self._condition

    def embed(self, numbers):
        return self.core.sphere_embedding(numbers)

    def features(self, atoms, edges, vectors, condition):
        graph = dict(atomic_numbers_full=atoms.numbers, batch_full=atoms.batch,
                     edge_index=edges, edge_distance=vectors.norm(dim=-1),
                     edge_distance_vec=vectors, node_offset=0)
        self._condition = condition
        try:
            result = self.core(dict(atomic_numbers=atoms.numbers, pos=atoms.positions,
                                    graph=graph))['node_embedding']
        finally:
            self._condition = None
        return self.velocity(result[:, 1:4, :]).squeeze(-1), result[:, 0, :]
