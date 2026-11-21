"""Periodic/edgeless repair checks: forward, autograd and load only, never fit."""
from dataclasses import dataclass, replace
from types import SimpleNamespace
import copy
import itertools
import math
from pathlib import Path

import pytest
import torch

from unifiedgroupgen.pl_modules.unified.periodic import EquiformerFeatures
from unifiedgroupgen.common import data_utils as graphs

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def no_training(monkeypatch):
    import pytorch_lightning as pl
    def forbidden(*args, **kwargs):
        raise AssertionError('Training/optimizer updates are forbidden in this suite')
    monkeypatch.setattr(pl.Trainer, 'fit', forbidden)
    monkeypatch.setattr(torch.optim.AdamW, 'step', forbidden)
    monkeypatch.setattr(torch.optim.Adam, 'step', forbidden)
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        yield
    torch.set_num_threads(old)


@dataclass
class State:
    """Already decoded state, so boundary geometries are controlled explicitly."""
    coordinates: torch.Tensor
    cell: torch.Tensor
    periodic: tuple = (True, True, True)

    @property
    def num_atoms(self):
        return len(self.coordinates)

    @property
    def spec(self):
        return SimpleNamespace(periodic=self.periodic)

    def decoded(self):
        return dict(coordinates=self.coordinates, lattice=self.cell.T,
                    cartesian_coordinates=self.coordinates @ self.cell,
                    atomic_numbers=torch.tensor([8, 22, 16][:self.num_atoms]))


def state(scale=3., n=1, periodic=(True, True, True)):
    f = torch.tensor([[.13, .27, .41], [.44, .18, .63], [.72, .66, .21]])[:n]
    return State(f, torch.eye(3)*scale, periodic)


def backbone():
    torch.manual_seed(11)
    return EquiformerFeatures(hidden=16, layers=1, channels=8, lmax=2, mmax=1,
                              cutoff=4., max_neighbors=16, use_gate_act=True).eval()


def test_overflow_search_preserves_legacy_guard_but_v2_can_opt_in():
    f = torch.zeros((1, 3), requires_grad=True)
    cell = torch.diag(torch.tensor([.3, 5., 5.]))[None].requires_grad_()
    args = (f, cell, torch.tensor([1]), torch.ones((1, 3), dtype=torch.bool))
    with pytest.raises(ValueError, match='max_images'):
        graphs.radius_graph(*args)
    actual = graphs.radius_graph(*args, allow_large_images=True, keep_shells=True)
    expected = graphs.radius_graph(*args, max_images=32, keep_shells=True)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    ga = torch.autograd.grad(actual[-1].square().sum(),(f,cell))
    gb = torch.autograd.grad(expected[-1].square().sum(),(f,cell))
    for a,b in zip(ga,gb):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_edgeless_context_once_and_cleanup():
    model = backbone()
    s = state(scale=20.)
    context = torch.randn(1, 16, requires_grad=True)
    vector, scalar = model.forward_states([s], context)
    expected = torch.nn.functional.embedding(s.decoded()['atomic_numbers'],
                                             model.core.sphere_embedding.weight) + model.context(context)
    torch.testing.assert_close(scalar, expected, rtol=0, atol=0)
    assert not vector.any()
    assert model._node_context is None and model._scalar_nodes is None
    gradient, = torch.autograd.grad(scalar.sum(), context)
    assert torch.isfinite(gradient).all() and gradient.abs().sum() > 0


def brute_graph(f, cell, periodic, cutoff, k, shells):
    """Independent small reference: enumerate, sort and select per target."""
    bounds = (torch.ceil(cutoff*torch.linalg.vector_norm(torch.linalg.inv(cell),dim=0)).long()+1).tolist()
    shifts = list(itertools.product(*(range(-b,b+1) if p else [0] for b,p in zip(bounds,periodic))))
    candidates = []
    for i in range(len(f)):
        for j in range(len(f)):
            for shift in shifts:
                v = (f[j]-f[i]+f.new_tensor(shift)) @ cell
                d = v.norm().item()
                if 1e-6 < d < cutoff:
                    candidates.append((i,d,j,shift))
    candidates.sort()
    chosen = []
    for i in range(len(f)):
        rows = [r for r in candidates if r[0] == i]
        threshold = rows[min(k,len(rows))-1][1]+1e-5 if rows else -1.
        chosen += [r for r in rows if r[1]<=threshold] if shells else rows[:k]
    edges = torch.tensor([[r[2] for r in chosen],[r[0] for r in chosen]],dtype=torch.long)
    images = torch.tensor([r[3] for r in chosen],dtype=torch.long).reshape(-1,3)
    vectors = (f[edges[0]]+images-f[edges[1]]) @ cell
    return edges, images, torch.tensor([len(chosen)]), vectors


@pytest.mark.parametrize('shells', [False, True])
@pytest.mark.parametrize('periodic', [(True,True,True),(True,True,False),(True,False,False)])
@pytest.mark.parametrize('dtype', [torch.float32,torch.float64])
def test_large_images_match_bruteforce_and_chunk_boundaries(shells, periodic, dtype, monkeypatch):
    f = torch.tensor([[.13,.27,.41],[.44,.18,.63]],dtype=dtype)
    cell = torch.tensor([[.3,0.,0.],[.11,5.,0.],[.07,.19,5.]],dtype=dtype)
    args = (f,cell[None],torch.tensor([2]),torch.tensor([periodic]))
    expected = brute_graph(f,cell,periodic,2.,7,shells)
    # Force fallback with a low legacy limit; compare three chunk partitions.
    for size in (28,116,1 << 20):
        monkeypatch.setattr(graphs,'_IMAGE_DISTANCE_BUDGET',size)
        actual = graphs.radius_graph(*args,cutoff=2.,max_neighbors=7,max_images=2,
                                     keep_shells=shells,allow_large_images=True)
        for a,b in zip(actual,expected):
            tolerance = 1e-6 if dtype==torch.float32 else 1e-12
            torch.testing.assert_close(a,b,rtol=tolerance,atol=tolerance)
        assert actual[1][:,torch.tensor(periodic).logical_not()].eq(0).all()


def test_dense_normal_path_and_gradients_are_unchanged(monkeypatch):
    f = torch.tensor([[.13,.27,.41],[.44,.18,.63]],requires_grad=True)
    cell = torch.tensor([[[3.,0.,0.],[.2,3.4,0.],[.1,.4,3.9]]],requires_grad=True)
    args = (f,cell,torch.tensor([2]),torch.ones((1,3),dtype=torch.bool))
    old = graphs.radius_graph(*args,cutoff=4.,keep_shells=True)
    def unexpected(*args, **kwargs):
        raise AssertionError('Normal path must not use fallback')
    monkeypatch.setattr(graphs,'_large_image_neighbors',unexpected)
    new = graphs.radius_graph(*args,cutoff=4.,keep_shells=True,allow_large_images=True)
    for a,b in zip(old,new):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    a = torch.autograd.grad(old[-1].square().sum(),(f,cell))
    b = torch.autograd.grad(new[-1].square().sum(),(f,cell))
    for x,y in zip(a,b):
        torch.testing.assert_close(x,y,rtol=0,atol=0)


def test_fallback_keeps_complete_tied_shell_and_live_gradients(monkeypatch):
    f = torch.zeros((1,3),dtype=torch.float64,requires_grad=True)
    cell = (torch.eye(3,dtype=torch.float64)*.3).unsqueeze(0).requires_grad_()
    args = (f,cell,torch.tensor([1]),torch.ones((1,3),dtype=torch.bool))
    monkeypatch.setattr(graphs,'_IMAGE_DISTANCE_BUDGET',31)
    monkeypatch.setattr(graphs,'_image_shifts',lambda bounds:pytest.fail('Must not allocate full image cube'))
    output = graphs.radius_graph(*args,cutoff=.61,max_neighbors=1,max_images=2,
                                 keep_shells=True,allow_large_images=True)
    assert output[2].tolist() == [6]
    assert (output[0][0] == output[0][1]).all()
    derivatives = torch.autograd.grad(output[-1].square().sum(),(f,cell))
    assert all(torch.isfinite(g).all() for g in derivatives)
    assert derivatives[1].abs().sum() > 0
    strict = graphs.radius_graph(*args,cutoff=.61,max_neighbors=1,max_images=2,
                                 keep_shells=False,allow_large_images=True)
    assert strict[2].tolist() == [1]


def legacy_forward(model, states, context):
    """Frozen pre-repair nonempty/normal path for output/gradient regression."""
    from torch_geometric.data import Data
    decoded = [s.decoded() for s in states]
    f = torch.cat([d['coordinates'] for d in decoded])
    numbers = torch.cat([d['atomic_numbers'] for d in decoded])
    cells = torch.stack([d['lattice'] for d in decoded]).transpose(1,2)
    counts = torch.tensor([s.num_atoms for s in states])
    periodic = torch.tensor([s.spec.periodic for s in states])
    batch = torch.repeat_interleave(torch.arange(len(states)),counts)
    node_context = torch.repeat_interleave(context,counts,dim=0)
    edges,shifts,neighbors,vectors = graphs.radius_graph(f,cells,counts,periodic,
        model.cutoff,model.max_neighbors,model.max_images,keep_shells=True)
    assert edges.shape[1] and (neighbors > 0).all()
    graph = torch.repeat_interleave(torch.arange(len(states)),neighbors)
    data = Data(pos=torch.cat([d['cartesian_coordinates'] for d in decoded]),atomic_numbers=numbers,
        natoms=counts,batch=batch,cell=cells,pbc=periodic,edge_index=edges,cell_offsets=shifts,
        edge_vectors=vectors,offset_vectors=torch.einsum('ni,nij->nj',shifts.to(f),cells[graph]),neighbors=neighbors)
    model._node_context = model.context(node_context)
    try:
        output = model.core(data)
        return output['forces'],model._scalar_nodes
    finally:
        model._node_context,model._scalar_nodes = None,None


def test_normal_network_outputs_and_parameter_gradients_match_old_path():
    model = backbone()
    reference = copy.deepcopy(model)
    states = [state(n=2),state(scale=3.7,n=3,periodic=(True,True,False))]
    context = torch.randn(2,16,requires_grad=True)
    rng = torch.get_rng_state()
    actual = model.forward_states(states,context)
    torch.set_rng_state(rng)
    expected = legacy_forward(reference,states,context)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    ga = torch.autograd.grad(sum(v.square().sum() for v in actual),tuple(model.parameters()),allow_unused=True)
    gb = torch.autograd.grad(sum(v.square().sum() for v in expected),tuple(reference.parameters()),allow_unused=True)
    for a,b in zip(ga,gb):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a,b,rtol=0,atol=0)


@pytest.mark.parametrize('order', [(0,1,2,3),(3,2,1,0),(1,0,3,2)])
def test_empty_edgeless_and_normal_graphs_match_single_and_gradients(order):
    model = backbone()
    states = [state(n=0),state(scale=20.),state(n=3),state(scale=3.7,n=2,periodic=(True,True,False))]
    states = [states[i] for i in order]
    context = torch.randn(4,16,requires_grad=True)
    rng = torch.get_rng_state()
    actual = model.forward_states(states,context)
    torch.set_rng_state(rng)
    single = [model.forward_states([s],context[i:i+1]) for i,s in enumerate(states)]
    expected = [torch.cat([s[k] for s in single]) for k in (0,1)]
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)
    parameters = tuple(model.parameters())+(context,)
    ga = torch.autograd.grad(sum(v.square().sum() for v in actual),parameters,allow_unused=True)
    gb = torch.autograd.grad(sum(v.square().sum() for v in expected),parameters,allow_unused=True)
    for a,b in zip(ga,gb):
        assert (a is None) == (b is None)
        if a is not None:
            assert torch.isfinite(a).all()
            torch.testing.assert_close(a,b,atol=3e-5,rtol=3e-4)
    assert model._node_context is None and model._scalar_nodes is None


@pytest.mark.parametrize('where', ['embedding','core'])
def test_context_cleared_after_exceptions(where, monkeypatch):
    model = backbone()
    module = model.core.sphere_embedding if where == 'embedding' else model.core
    def fail(*args, **kwargs):
        raise RuntimeError('injected failure')
    with monkeypatch.context() as patch:
        patch.setattr(module,'forward',fail)
        with pytest.raises(RuntimeError,match='injected failure'):
            model.forward_states([state(scale=20. if where=='embedding' else 3.)],torch.randn(1,16))
    assert model._node_context is None and model._scalar_nodes is None
    output = model.forward_states([state()],torch.randn(1,16))
    assert all(torch.isfinite(v).all() for v in output)


def test_activation_checkpoint_recomputation_matches_without_parameter_updates():
    model = backbone().train()
    other = copy.deepcopy(model)
    other.checkpoint_layers = True
    states = [state(scale=20.),state(n=2)]
    context = torch.randn(2,16,requires_grad=True)
    rng = torch.get_rng_state()
    first = model.forward_states(states,context)
    torch.set_rng_state(rng)
    second = other.forward_states(states,context)
    for a,b in zip(first,second):
        torch.testing.assert_close(a,b,rtol=0,atol=0)
    ga = torch.autograd.grad(sum(v.square().sum() for v in first),tuple(model.parameters()),allow_unused=True)
    gb = torch.autograd.grad(sum(v.square().sum() for v in second),tuple(other.parameters()),allow_unused=True)
    for a,b in zip(ga,gb):
        assert (a is None) == (b is None)
        if a is not None:
            torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert other._node_context is None and other._scalar_nodes is None


def test_isolated_atom_inside_active_graph_stays_on_native_path():
    model = backbone()
    s = State(torch.tensor([[0.,0.,0.],[.1,0.,0.],[.5,.5,.5]]),torch.eye(3)*20.)
    context = torch.randn(1,16)
    rng = torch.get_rng_state()
    actual = model.forward_states([s],context)
    torch.set_rng_state(rng)
    expected = legacy_forward(model,[s],context)
    for a,b in zip(actual,expected):
        torch.testing.assert_close(a,b,rtol=0,atol=0)


def test_all_edgeless_and_empty_graph_batch_retains_node_order():
    model = backbone()
    states = [state(scale=20.),state(n=0),state(scale=25.)]
    context = torch.randn(3,16)
    vector,scalar = model.forward_states(states,context)
    expected = torch.cat([model.forward_states([s],context[i:i+1])[1] for i,s in enumerate(states)])
    torch.testing.assert_close(scalar,expected,atol=1e-6,rtol=1e-6)
    assert vector.shape == (2,3) and not vector.any()
    empty = model.forward_states([state(n=0)],context[:1])
    assert [tuple(v.shape) for v in empty] == [(0,3),(0,8)]


def test_large_image_rotation_and_atom_permutation():
    f = torch.tensor([[.13,.27,.41],[.44,.18,.63]],dtype=torch.float64)
    cell = torch.tensor([[.3,0.,0.],[.11,5.,0.],[.07,.19,5.]],dtype=torch.float64)
    rotation,_ = torch.linalg.qr(torch.tensor([[1.,2.,3.],[2.,1.,-1.],[.5,2.,-2.]],dtype=f.dtype))
    kwargs = dict(cutoff=2.,max_neighbors=7,max_images=2,keep_shells=True,allow_large_images=True)
    args = (torch.tensor([2]),torch.tensor([[True,True,False]]))
    expected = graphs.radius_graph(f,cell[None],*args,**kwargs)
    rotated = graphs.radius_graph(f,(cell@rotation.T)[None],*args,**kwargs)
    permuted = graphs.radius_graph(f.flip(0),cell[None],*args,**kwargs)
    def keyed(result, permute=False):
        e,s,_,v = result
        if permute:e=1-e
        return {(int(e[0,i]),int(e[1,i]),*s[i].tolist()):v[i] for i in range(len(s))}
    reference = keyed(expected)
    for result,permutation,rotate in ((rotated,False,True),(permuted,True,False)):
        actual = keyed(result,permutation)
        assert set(actual) == set(reference)
        for key in reference:
            torch.testing.assert_close(actual[key],reference[key]@rotation.T if rotate else reference[key],
                                       atol=1e-12,rtol=1e-12)


@pytest.mark.parametrize('kind', ['space','layer','plane'])
@pytest.mark.parametrize('task', ['dng','csp'])
def test_full_joint_model_mixed_states_match_single(kind, task):
    from unifiedgroupgen.group_constraints.chemistry import ChemicalControl
    from unifiedgroupgen.group_constraints.unified import catalog_spec
    from unifiedgroupgen.group_constraints.flow_state import initial,apply,assign,JointAction
    from unifiedgroupgen.pl_modules.factory import build_joint_model
    control = ChemicalControl.from_inputs(counts={8:1}) if task=='csp' else ChemicalControl()
    empty = initial(catalog_spec(kind,1),generator=torch.Generator().manual_seed(2),budget=12,chemistry=control)
    born = apply(empty,JointAction('birth',letter='a',element=8 if task=='csp' else 0),torch.Generator().manual_seed(3))
    if task=='dng':born=assign(born,0,8)
    def scaled(scale):
        standard = born.metric.new_tensor(born.compiled.standard)
        value = torch.einsum('kij,ij->k',standard,torch.eye(born.spec.rank)*2*math.log(scale))
        return replace(born,metric=value)
    states = [scaled(20.),empty,scaled(3.)]
    cfg = dict(model=dict(family='joint_flow_v2',backbone='equiformer_v2',hidden=16,layers=1,channels=8,
        lmax=2,mmax=1,use_gate_act=True,task_controls=True,cutoff=4.,max_neighbors=16),
        allowed_elements=[8,16,22],properties={},generation_task=task,max_atoms=8,max_orbits=4,event_budget=12)
    torch.manual_seed(9)
    model = build_joint_model(cfg).eval()
    rng = torch.get_rng_state()
    with torch.no_grad():
        actual = model.forward_batch(states,[.5]*3)
        torch.set_rng_state(rng)
        expected = [model(s,.5) for s in states]
    for a,b in zip(actual,expected):
        for key in ('metric_velocity_scaled','coordinate_velocity_scaled','element_logits','remaining','action_logits'):
            torch.testing.assert_close(a[key],b[key],atol=3e-5,rtol=3e-4)
        assert a['actions']==b['actions']


def test_current_native_checkpoint_strictly_checks_resume_identity():
    from unifiedgroupgen.pl_modules.unified.module import FLOW_FORMAT, JointGeneratorModule
    from unifiedgroupgen.experiment_config import compose_experiment, training_definition
    config = training_definition(compose_experiment('flow_mp20_dng'))
    config['allowed_elements'] = [8, 22]
    config['model'].update(hidden=16, layers=1, channels=8, lmax=2, mmax=1)
    runtime = {'precision': 'fp32', 'batch_size': 2}
    hashes = {'train': {'train.pt': 'digest'}}
    module = JointGeneratorModule(config, runtime, hashes)
    rng = torch.Generator().manual_seed(7).get_state()
    state = dict(format=FLOW_FORMAT, config=config, runtime=runtime, split_hashes=hashes,
                 rng={'torch': rng}, pending_lr_metric=.25)
    module.on_load_checkpoint({'joint_state': state})
    assert module._pending_rng is state['rng']
    assert module._pending_lr_metric == .25
    for key in ('config', 'runtime', 'split_hashes'):
        with pytest.raises(ValueError, match='differs|differ'):
            module.on_load_checkpoint({'joint_state': {**state, key: {}}})


def test_full_recipe_backbone_parameter_schema_matches_unchanged_base():
    from unifiedgroupgen.pl_modules.backbones.equiformer_v2 import EquiformerV2Backbone
    from unifiedgroupgen.experiment_config import compose_experiment,training_definition
    experiment = compose_experiment('full_mp20_eqv2_dng')
    cfg = training_definition(experiment)
    a = cfg['model']
    options = dict(hidden=a['hidden'],layers=a['layers'],channels=a['channels'],lmax=a['lmax'],mmax=a['mmax'],
                   cutoff=a['cutoff'],max_neighbors=a['max_neighbors'],max_images=a['max_images'],use_gate_act=a['use_gate_act'])
    reference = EquiformerV2Backbone(**options)
    repaired = EquiformerFeatures(**options)
    repaired.load_state_dict(reference.state_dict(),strict=True)
    assert {k:tuple(v.shape) for k,v in reference.state_dict().items()} == {
        k:tuple(v.shape) for k,v in repaired.state_dict().items()}
    assert experiment.runtime.precision=='fp32' and cfg['epochs']==1000
