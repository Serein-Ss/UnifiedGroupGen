"""Official backbone integration, periodic graphs, gradients, and group constraints."""
import copy
import importlib.util
import os
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pytest
import torch

from unifiedgroupgen.group_constraints.compiler import Descriptor, OrbitSpec, compile_descriptor, group_table
from unifiedgroupgen.pl_modules.factory import build_models
from unifiedgroupgen.pl_modules.backbones.periodic_graph import geometry, radius_graph


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def compiled_charts():
    return [compile_descriptor(Descriptor('space', 136, (OrbitSpec('a', 22), OrbitSpec('f', 8)))),
            compile_descriptor(Descriptor('layer', 1, (OrbitSpec('a', 16), OrbitSpec('a', 16))))]


def test_periodic_graph_keeps_self_images_and_never_wraps_layer_normal():
    coords = torch.zeros(2, 3, dtype=torch.float64)
    cells = torch.eye(3, dtype=torch.float64).repeat(2, 1, 1)*3
    counts = torch.tensor([1, 1]); periodic = torch.tensor([[True, True, True], [True, True, False]])
    edges, shifts, neighbors, vectors = radius_graph(coords, cells, counts, periodic, cutoff=3.1)
    assert neighbors.tolist() == [6, 4]
    assert (edges[0] == edges[1]).all()  # The neighbors are periodic images of a one-atom cell.
    assert shifts[6:, 2].eq(0).all()
    torch.testing.assert_close(vectors.norm(dim=-1), torch.full((10,), 3., dtype=coords.dtype))


def test_periodic_image_geometry_rotates_with_the_cell():
    coords = torch.tensor([[.1, .2, .3]], dtype=torch.float64)
    cell = torch.tensor([[[3., 0., 0.], [.4, 3.4, 0.], [.2, .1, 3.9]]], dtype=torch.float64)
    rotation, _ = torch.linalg.qr(torch.tensor([[1., 2., 3.], [2., 1., -1.], [.5, 2., -2.]], dtype=coords.dtype))
    arguments = (torch.tensor([1]), torch.ones((1,3), dtype=torch.bool))
    a = radius_graph(coords, cell, *arguments, cutoff=4., max_neighbors=1000)
    b = radius_graph(coords, cell @ rotation.T, *arguments, cutoff=4., max_neighbors=1000)
    # Compare by lattice image rather than arbitrary ordering of equal-distance neighbors.
    for shift, vector in zip(a[1], a[3]):
        match = (b[1] == shift).all(dim=1)
        torch.testing.assert_close(b[3][match][0], vector @ rotation.T, rtol=1e-10, atol=1e-10)


def test_cspnet_raw_core_matches_official_forward_and_parameter_gradients():
    pytest.importorskip('einops')
    from unifiedgroupgen.pl_modules.backbones.cspnet import CSPNetBackbone
    from unifiedgroupgen.pl_modules.backbones.official.cspnet import CSPNet
    torch.manual_seed(12)
    adapter = CSPNetBackbone(hidden=16, layers=2, frequencies=2)
    reference = CSPNet(hidden_dim=16, latent_dim=16, num_layers=2, max_atoms=119, num_freqs=2, ln=True, ip=True)
    reference.load_state_dict(adapter.core.state_dict(), strict=True)
    cc = compiled_charts()[:1]; states = [cc[0].prior()*.1]
    lattices = torch.stack([c.expand(z)['lattice'] for c, z in zip(cc, states)])
    f, numbers, counts, _ = geometry(cc, states, lattices)
    node_graph = torch.zeros(len(f), dtype=torch.long)
    context = torch.randn(1, 16)
    velocity, matrix = adapter(cc, states, lattices, torch.zeros(len(f), 16), context, node_graph)
    expected_matrix, expected_velocity = reference(context, numbers, f, lattices.transpose(1,2), counts, node_graph)
    torch.testing.assert_close(matrix, expected_matrix, rtol=0, atol=0)
    torch.testing.assert_close(velocity, expected_velocity, rtol=0, atol=0)
    (matrix.square().sum()+velocity.square().sum()).backward()
    (expected_matrix.square().sum()+expected_velocity.square().sum()).backward()
    for a,b in zip(adapter.core.parameters(), reference.parameters()):
        if a.grad is not None: torch.testing.assert_close(a.grad,b.grad,rtol=0,atol=0)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('precision', ['fp32', 'bf16'])
@pytest.mark.parametrize('name,options', [
    ('cspnet', {'layers':1, 'frequencies':2}),
    ('mattergen', {'blocks':1, 'cutoff':4., 'max_neighbors':16}),
    ('equiformer_v2', {'layers':1, 'channels':16, 'lmax':2, 'mmax':1, 'cutoff':4., 'max_neighbors':16})])
def test_backbone_forward_backward_and_tangent_constraints(name, options, device, precision, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    if precision == 'bf16' and device == 'cpu': pytest.skip('Training uses CUDA mixed precision')
    dependency = {'cspnet':'einops','mattergen':'torch_sparse','equiformer_v2':'e3nn'}[name]
    if importlib.util.find_spec(dependency) is None: pytest.skip('Optional upstream package unavailable')
    monkeypatch.setenv('PROJECT_ROOT', str(Path(__file__).resolve().parents[3]))
    previous = os.environ['PROJECT_ROOT']
    target = {'cspnet':'CSPNetBackbone','mattergen':'MatterGenBackbone','equiformer_v2':'EquiformerV2Backbone'}[name]
    cfg = {'groups':[['space',136],['layer',1]],'properties':{},'max_atoms':16,'max_orbits':8,
           'model':{'hidden':32,'heads':4,'proposal_layers':1,'flow_layers':1,'frequencies':2,
                    'candidate_backbone':{'_target_':f'unifiedgroupgen.pl_modules.backbones.{name}.{target}',**options}}}
    _, model = build_models(cfg,device)
    assert os.environ['PROJECT_ROOT'] == previous
    assert (name == 'cspnet' or model.candidate_backbone.core.__class__.__module__.startswith(
            'unifiedgroupgen.pl_modules.backbones.official.mattergen.' if name == 'mattergen' else
            'unifiedgroupgen.pl_modules.backbones.official.fairchem.'))
    cc=compiled_charts(); states=[(c.prior(device)*.1).requires_grad_() for c in cc]
    with torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext():
        values = model.forward_batch(cc,states,[.3,.3])
        loss = sum(v.square().sum() for v in values)
    loss.backward()
    assert all(torch.isfinite(v).all() for v in values)
    assert all(z.grad is not None and torch.isfinite(z.grad).all() for z in states)
    assert any(p.grad is not None for p in model.candidate_backbone.parameters())
    for c,z,v in zip(cc,states,values):
        f=c.expand(z)['coordinates'].detach().cpu().numpy()
        constraint=c.constraint_matrix(f)
        expanded_velocity=c.coordinate_velocity(v[c.lattice_dof:]).detach().cpu().numpy().reshape(-1)
        assert np.linalg.norm(constraint@expanded_velocity) < 1e-4
    cfg['backbone_identity'] = model.backbone_identity
    _, restored = build_models(cfg, device)
    restored.load_state_dict(model.state_dict(), strict=True)
    cfg['backbone_identity'] = {'core': 'incorrect'}
    with pytest.raises(ValueError, match='source or dependency versions'):
        build_models(cfg, device)
