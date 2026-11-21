"""Acceleration must retain graph topology, FP32 geometry and shared gradients."""
import pytest
import torch
from unifiedgroupgen.tests.unit.test_joint_flow_v2 import config,record
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.common.data_utils import target_from_record
from unifiedgroupgen.pl_data.unified.tasks import task_target
from unifiedgroupgen.pl_data.unified.path import sample_path
from unifiedgroupgen.pl_modules.unified.objectives import loss
from unifiedgroupgen.group_constraints.flow_state import metric_tangent,coordinate_tangent
from unifiedgroupgen.common.data_utils import radius_graph


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('shells',[False,True])
@pytest.mark.parametrize('periodic',[(True,True,True),(True,True,False)])
def test_vectorized_neighbors_equal_individual_target_sort(shells,periodic):
    torch.manual_seed(8)
    coords=torch.rand((9,3));cell=torch.tensor([[3.,0,0],[.7,3.2,0],[0,0,5.]])
    e,shifts,_,vectors=radius_graph(coords,cell[None],torch.tensor([9]),torch.tensor([periodic]),4.,5,16,shells)
    import itertools
    bounds=torch.ceil(4.*torch.linalg.vector_norm(torch.linalg.inv(cell),dim=0)).long()+1
    bounds=torch.where(torch.tensor(periodic),bounds,torch.zeros_like(bounds))
    images=torch.tensor(list(itertools.product(*(range(-n,n+1) for n in bounds.tolist()))))
    source,target=torch.meshgrid(torch.arange(9),torch.arange(9),indexing='ij');source=source.flatten();target=target.flatten()
    distances=torch.linalg.vector_norm((coords[source,None]-coords[target,None]+images)@cell,dim=-1)
    pairs,im=((distances<4.)&(distances>1e-6)).nonzero(as_tuple=True)
    selected=[]
    for atom in range(9):
        ids=(target[pairs]==atom).nonzero().flatten()
        ordered=ids[distances[pairs[ids],im[ids]].argsort(stable=True)]
        if shells and len(ordered)>5:
            boundary=distances[pairs[ordered[4]],im[ordered[4]]]
            ordered=ordered[distances[pairs[ordered],im[ordered]]<=boundary+1e-5]
        else:ordered=ordered[:5]
        selected.append(ordered)
    ids=torch.cat(selected)
    assert torch.equal(e,torch.stack((source[pairs[ids]],target[pairs[ids]])))
    assert torch.equal(shifts,images[im[ids]])
    torch.testing.assert_close(vectors,(coords[e[0]]+shifts-coords[e[1]])@cell,atol=0,rtol=0)


@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
def test_checkpointed_core_matches_outputs_and_gradients(dataset):
    cfg=config();torch.manual_seed(9);plain=build_joint_model(cfg)
    cfg2={**cfg,'model':{**cfg['model'],'checkpoint_layers':True}}
    checked=build_joint_model(cfg2);checked.load_state_dict(plain.state_dict())
    target=task_target(target_from_record(record(dataset)),cfg,5)
    samples=[sample_path(target,5,time=.8,allowed_elements=cfg['allowed_elements'])]
    rng=torch.get_rng_state();a=loss(plain,samples);a['loss'].backward();torch.set_rng_state(rng)
    b=loss(checked,samples);b['loss'].backward()
    torch.testing.assert_close(a['loss'],b['loss'],atol=1e-5,rtol=1e-5)
    for (name,p),(other,q) in zip(plain.named_parameters(),checked.named_parameters()):
        assert name==other and (p.grad is None)==(q.grad is None)
        if p.grad is not None:torch.testing.assert_close(p.grad,q.grad,atol=2e-5,rtol=2e-4)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='CUDA AMP/fused execution')
@pytest.mark.parametrize('backend',['equiformer_v2','joint_atom_graph'])
@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
def test_cuda_bf16_geometry_remains_float32(backend,dataset):
    cfg=config(backend);m=build_joint_model(cfg,'cuda:0')
    target=task_target(target_from_record(record(dataset)),cfg,31)
    samples=[sample_path(target,31,time=.8,allowed_elements=cfg['allowed_elements']).to('cuda:0')]
    s=samples[0].state
    with torch.autocast('cuda',dtype=torch.bfloat16):
        decoded=s.decoded();value=loss(m,samples)
        assert decoded['lattice'].dtype==decoded['coordinates'].dtype==torch.float32
        assert metric_tangent(s,torch.zeros_like(s.metric)).dtype==torch.float32
        assert coordinate_tangent(s,torch.zeros_like(s.coordinates)).dtype==torch.float32
    value['loss'].backward()  # Precision/gradient verification, no parameter update.
    assert all(torch.isfinite(p).all() for p in m.parameters())
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)


def test_fast_profile_resolves_without_changing_model_definition():
    from unifiedgroupgen.experiment_config import compose_experiment,training_definition
    a=compose_experiment('flow_mp20_dng')
    b=compose_experiment('flow_mp20_dng',overrides=['runtime=joint_flow_fast'])
    assert training_definition(a)==training_definition(b)
    assert a.train.chemical_set_probability==0.
    assert b.runtime.precision=='bf16' and b.runtime.fused_adamw


@pytest.mark.parametrize('cartesian', [False, True])
def test_batched_relative_tangent_matches_two_original_projections(cartesian):
    from unifiedgroupgen.group_constraints.flow_state import initial, apply, JointAction
    from unifiedgroupgen.group_constraints.unified import catalog_spec

    spec = catalog_spec('space', 10)
    state = initial(spec, generator=torch.Generator().manual_seed(8))
    free = next(w for w in spec.templates if w.dof)
    fixed = next(w for w in spec.templates if not w.dof)
    for template in (free, fixed, free, spec.templates[0]):
        if not template.dof and template.letter in state.letters:
            continue
        state = apply(state, JointAction('birth', letter=template.letter))
    value = torch.randn_like(state.coordinates, requires_grad=True)
    reference = torch.randn_like(state.coordinates, requires_grad=True)

    def old_projection(vector, is_cartesian):
        lattice = state.decoded(wrap=False)['lattice']
        pieces = []
        for section, (basis, _, _) in zip(state.compiled.slices, state.compiled.blocks):
            a = vector.new_tensor(basis)
            if a.shape[-1] == 0:
                pieces.append(vector[section]*0)
                continue
            j = torch.einsum('ab,nbq->naq', lattice, a).reshape(-1, a.shape[-1])
            v = vector[section] if is_cartesian else vector[section] @ lattice.T
            q = torch.linalg.solve(j.T @ j, j.T @ v.reshape(-1))
            pieces.append(torch.einsum('nbq,q->nb', a, q))
        return torch.cat(pieces)

    old = old_projection(value, cartesian) - old_projection(reference, False)
    new = coordinate_tangent(state, value, cartesian=cartesian, reference=reference)
    torch.testing.assert_close(new, old, atol=2e-6, rtol=1e-5)
    weights = torch.randn_like(value)
    expected = torch.autograd.grad((old*weights).sum(), (value, reference))
    actual = torch.autograd.grad((new*weights).sum(), (value, reference))
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=1e-5)
