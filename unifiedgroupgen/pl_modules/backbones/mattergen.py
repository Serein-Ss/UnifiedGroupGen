"""Pinned official MatterGen GemNetT subset, with conditioning and chart adapters."""
import importlib
import math
import os
from functools import wraps
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from .periodic_graph import geometry, radius_graph, fractional_vectors


def official_gemnet():
    # Upstream globals mutates PROJECT_ROOT during import. Never leak it to this project.
    previous = os.environ.get('PROJECT_ROOT')
    try:
        prefix = 'unifiedgroupgen.pl_modules.backbones.official.mattergen'
        cls = importlib.import_module(prefix+'.common.gemnet.gemnet').GemNetT
        module = importlib.import_module(prefix+'.common.gemnet.gemnet')
        for name in ('frac_to_cart_coords_with_lattice', 'get_pbc_distances',
                     'edge_score_to_lattice_score_frac_symmetric'):
            original = getattr(module, name)
            if not getattr(original, '_ugg_fp32', False):
                @wraps(original)
                def geometry32(*args, _function=original, **kwargs):
                    tensor = next(value for value in (*args, *kwargs.values()) if isinstance(value, torch.Tensor))
                    with torch.autocast(device_type=tensor.device.type, enabled=False):
                        return _function(*args, **kwargs)
                geometry32._ugg_fp32 = True
                setattr(module, name, geometry32)
        basis = importlib.import_module(prefix+'.common.gemnet.layers.basis_utils')
        if not hasattr(basis.np, 'math'):
            # NumPy 1.x's removed np.math was Python's math module. Scope the
            # alias to this upstream module; do not monkey-patch global NumPy.
            basis.np = SimpleNamespace(**vars(np), math=math)
        return cls
    except ImportError as error:
        raise ImportError('Install this project environment; the pinned MatterGen core is bundled') from error
    finally:
        if previous is None: os.environ.pop('PROJECT_ROOT', None)
        else: os.environ['PROJECT_ROOT'] = previous


class AtomicEmbedding(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.emb_size = hidden
        self.embedding = nn.Embedding(119, hidden)
        self.context = None

    def forward(self, numbers):
        result = self.embedding(numbers)
        return result if self.context is None else result+self.context


class MatterGenBackbone(nn.Module):
    def __init__(self, hidden=512, blocks=3, cutoff=6., max_neighbors=50, max_images=16, scale_file=None):
        super().__init__()
        cls = official_gemnet()
        atomic = AtomicEmbedding(hidden)
        self.core = cls(num_targets=1, latent_dim=hidden, atom_embedding=atomic,
                        emb_size_atom=hidden, emb_size_edge=hidden, num_blocks=blocks,
                        cutoff=cutoff, max_neighbors=max_neighbors, regress_stress=True,
                        otf_graph=False, scale_file=scale_file)
        self.cutoff, self.max_neighbors, self.max_images = cutoff, max_neighbors, max_images

    def forward(self, compiled, states, lattices, node_context, global_context, node_graph):
        f, numbers, counts, periodic = geometry(compiled, states, lattices)
        cells = lattices.transpose(1, 2)
        edges, shifts, neighbors, _ = radius_graph(f, cells, counts, periodic, self.cutoff,
                                                   self.max_neighbors, self.max_images)
        self.core.atom_emb.context = node_context
        try:
            result = self.core(global_context, f, numbers, counts, node_graph, edge_index=edges,
                               to_jimages=shifts, num_bonds=neighbors, lattice=cells)
        finally:
            self.core.atom_emb.context = None
        return fractional_vectors(result.forces, lattices, node_graph), result.stress
