"""Frozen public MEGNet property models with their native graph construction."""
import json
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from unifiedgroupgen.tests.legacy.evaluation_before_tasks.io import sha256
from .structures import predictor_structure, structure_from_graph

FIDELITIES = {'PBE': 0, 'GLLB-SC': 1, 'HSE': 2, 'SCAN': 3}


def model_registry():
    path = Path(__file__).parents[1]/'resources/public_megnet_models.json'
    return json.loads(path.read_text(encoding='utf-8'))


def fetch_model(specification, registry, cache):
    """Download only the fixed official revision; verify every byte on reuse."""
    import requests
    from filelock import FileLock
    directory = Path(cache)/specification['name']
    directory.mkdir(parents=True, exist_ok=True)
    with FileLock(str(directory/'.download.lock'), timeout=180):
        for name, expected in specification['files'].items():
            path = directory/name
            if not path.exists():
                url = registry['base_url']+specification['name']+'/'+name
                response = requests.get(url, timeout=(15, 120))
                response.raise_for_status()
                temporary = path.with_suffix(path.suffix+'.part')
                temporary.write_bytes(response.content)
                if sha256(temporary) != expected:
                    raise ValueError('Official pretrained download hash mismatch: '+url)
                temporary.replace(path)
            if sha256(path) != expected:
                raise ValueError('Cached pretrained file changed: '+str(path))
    return directory


class MatGLPredictor:
    """CPU MEGNet evaluator. The optional DGL wheel does not require a training-env change."""
    def __init__(self, cache, properties=('formation_energy', 'band_gap'), band_gap_fidelity='PBE'):
        if band_gap_fidelity not in FIDELITIES:
            raise ValueError('Unknown band-gap fidelity')
        self.registry = model_registry()
        if not properties or len(set(properties)) != len(properties) or not set(properties) <= self.registry['models'].keys():
            raise ValueError('Choose distinct supported property names')
        try:
            import matgl
            import dgl
            from matgl.data.transformer import Normalizer
        except ImportError as error:
            raise ImportError('Install the optional predictor packages: pip install -e "./unifiedgroupgen[pretrained]"') from error
        if version('matgl') != '0.9.2' or version('dgl') != '1.1.2':
            raise ValueError('This verified MEGNet adapter requires matgl==0.9.2 and dgl==1.1.2')
        self.models, self.fidelity = {}, band_gap_fidelity
        for name in properties:
            spec = self.registry['models'][name]
            directory = fetch_model(spec, self.registry, cache)
            # The published model.pt contains a Normalizer, absent from the Torch 2.7 default allowlist.
            with torch.serialization.safe_globals([Normalizer]):
                model = matgl.load_model(directory)
            # MatGL's convenience loader uses strict=False; require complete weight agreement here.
            model.load_state_dict(torch.load(directory/'state.pt', map_location='cpu', weights_only=True), strict=True)
            self.models[name] = model.cpu().eval().requires_grad_(False)
        self.identity = {'backend': 'public_MatGL_MEGNet', 'source': self.registry['source'],
            'revision': self.registry['revision'], 'models': {name: self.registry['models'][name] for name in self.models},
            'versions': {name: version(name) for name in ('matgl', 'dgl', 'torch', 'numpy', 'pymatgen')},
            'band_gap_fidelity': self.fidelity, 'device': 'cpu',
            'graph_policy': 'Official Structure2Graph, model-specific cutoff and element vocabulary; no UGG embeddings',
            'layer_policy': 'Normal gap = 2*cutoff+2 A; reject any normal-image edge; proxy transfer only'}

    def graph(self, structure, name, kind):
        from matgl.ext.pymatgen import Structure2Graph
        core = self.models[name].model
        structure = predictor_structure(structure, kind, core.cutoff)
        graph, lattice, _ = Structure2Graph(core.element_types, core.cutoff).get_graph(structure)
        if not graph.num_edges():
            raise ValueError('No neighbors inside the pretrained model cutoff')
        if kind == 'layer' and torch.any(graph.edata['pbc_offset'][:, 2] != 0):
            raise ValueError('A normal periodic image entered the layer predictor graph')
        graph.edata['pbc_offshift'] = graph.edata['pbc_offset'] @ lattice[0]
        graph.ndata['pos'] = graph.ndata['frac_coords'] @ lattice[0]
        return graph

    def _forward(self, graphs, name):
        import dgl
        from matgl.graph.compute import compute_pair_vector_and_distance
        model = self.models[name]
        graph = dgl.batch(graphs)
        _, distance = compute_pair_vector_and_distance(graph)
        edges = model.model.bond_expansion(distance)
        state = (torch.full((len(graphs),), FIDELITIES[self.fidelity], dtype=torch.long)
                 if name == 'band_gap' else torch.zeros((len(graphs), 2)))
        with torch.no_grad():
            # TransformedTargetModel applies the official inverse target normalization exactly once.
            values = model(graph, edges, graph.ndata['node_type'], state).reshape(-1).cpu().numpy()
        if values.shape != (len(graphs),) or not np.isfinite(values).all():
            raise ValueError('Nonfinite or incorrectly shaped pretrained prediction')
        return values

    def predict_structures(self, structures, *, kind, batch_size=8):
        if batch_size < 1:
            raise ValueError('batch_size must be positive')
        rows = [{name: None for name in self.models} for _ in structures]
        errors = [[] for _ in structures]
        for start in range(0, len(structures), batch_size):
            for name in self.models:
                graphs, indices = [], []
                for index in range(start, min(start+batch_size, len(structures))):
                    if structures[index] is None:
                        errors[index].append(name+': no valid input structure')
                        continue
                    try:
                        graphs.append(self.graph(structures[index], name, kind)); indices.append(index)
                    except (ValueError, RuntimeError, KeyError) as error:
                        errors[index].append(name+': '+str(error))
                if not graphs:
                    continue
                try:
                    values = self._forward(graphs, name)
                    for index, value in zip(indices, values):
                        rows[index][name] = float(value)
                except (ValueError, RuntimeError):
                    # Isolate a bad structure without dropping other batch members or their indices.
                    for index, graph in zip(indices, graphs):
                        try: rows[index][name] = float(self._forward([graph], name)[0])
                        except (ValueError, RuntimeError) as single: errors[index].append(name+': '+str(single))
        for row, error in zip(rows, errors):
            row['prediction_error'] = '; '.join(error)
        return rows

    def predict_graph(self, graph, *, kind, lattice_columns=True):
        structure = structure_from_graph(graph, kind=kind, lattice_columns=lattice_columns)
        return self.predict_structures([structure], kind=kind)[0]
