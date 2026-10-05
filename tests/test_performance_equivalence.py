"""Check optimized operations against the original mathematical constructions."""

import copy
import pytest
import torch
from unifiedgroupgen.conditioning import PropertyEncoder
from unifiedgroupgen.constraints import AffineObservation
from unifiedgroupgen.flow import OrbitFlow, velocity_loss
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor


@pytest.fixture(params=['cpu', 'cuda'])
def device(request):
    if request.param == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    return request.param


def test_fixed_observation_matches_general_affine_values_and_gradients(device):
    fixed = AffineObservation.fixed(7, [5, 1], [0.4, -0.8])
    general = AffineObservation(fixed.matrix, fixed.values)
    for operation in ('retract', 'project', 'features'):
        z = torch.randn(7, device=device, dtype=torch.float64, requires_grad=True)
        actual = getattr(fixed, operation)(z)
        expected = getattr(general, operation)(z)
        actual = actual if isinstance(actual, tuple) else (actual,)
        expected = expected if isinstance(expected, tuple) else (expected,)
        for a, b in zip(actual, expected):
            assert torch.allclose(a, b, atol=1e-12)
            if a.requires_grad:
                assert torch.allclose(torch.autograd.grad(a.sum(), z, retain_graph=True)[0],
                                      torch.autograd.grad(b.sum(), z, retain_graph=True)[0], atol=1e-12)


@pytest.mark.parametrize('force_null', [False, True])
def test_condition_batch_matches_independent_encoding_and_gradients(device, force_null):
    configs = {'gap': {'shift': 1, 'scale': 2},
               'domain': {'type': 'categorical', 'num_classes': 2, 'context': True}}
    model = PropertyEncoder(configs, hidden=8).to(device).eval()
    reference = copy.deepcopy(model)
    props = [{'gap': 2.0, 'domain': 0}, {'gap': None, 'domain': 1},
             {'gap': float('nan'), 'domain': 0}]
    compositions = [{8: 2, 14: 1}, None, {6: 3}]
    like = torch.zeros(1, device=device)
    actual = model.forward_batch(props, like, force_null, compositions)
    expected = torch.stack([reference(p, like, force_null, c) for p, c in zip(props, compositions)])
    assert torch.allclose(actual, expected, atol=1e-6)
    actual.square().sum().backward()
    expected.square().sum().backward()
    for a, b in zip(model.parameters(), reference.parameters()):
        # Batched masked branches may allocate an exactly zero gradient instead of None.
        if a.grad is not None or b.grad is not None:
            ga = torch.zeros_like(a) if a.grad is None else a.grad
            gb = torch.zeros_like(b) if b.grad is None else b.grad
            assert torch.allclose(ga, gb, atol=2e-5, rtol=1e-5)
    model.train(); model.dropout = 1
    assert torch.allclose(model.forward_batch(props, like, compositions=compositions),
                          model.forward_batch(props, like, True, compositions), atol=1e-6)
    with pytest.raises(ValueError, match='required context'):
        model.forward_batch([{'gap': 1}], like)


@pytest.mark.parametrize('fixed_only', [False, True])
def test_batched_metrics_and_projection_match_individual_charts(device, fixed_only):
    descriptors = [Descriptor('space', 225, (OrbitSpec('a', 14),))]
    if not fixed_only:
        descriptors += [Descriptor('space', 2, (OrbitSpec('i', 14),)),
                        Descriptor('layer', 37, (OrbitSpec('r', 8),)),
                        Descriptor('plane', 1, (OrbitSpec('a', 6),))]
    charts = [compile_descriptor(d) for d in descriptors]
    states = [c.prior(device, torch.float64).requires_grad_() for c in charts]
    model = OrbitFlow({}, hidden=8, layers=1, frequencies=2).to(device=device, dtype=torch.float64).eval()
    captured = {}
    handles = [model.coordinate_head.register_forward_hook(lambda m, a, v: captured.update(coordinates=v)),
               model.lattice_head.register_forward_hook(lambda m, a, v: captured.update(lattice=v))]
    actual, lattices = model.forward_batch(charts, states, [0.3]*len(charts), return_lattices=True)
    for handle in handles:
        handle.remove()
    expected, offset = [], 0
    for i, (c, z, lattice) in enumerate(zip(charts, states, lattices)):
        original = c.expand(z)['lattice']
        assert torch.allclose(lattice, original, atol=1e-8, rtol=1e-8)
        ga = torch.autograd.grad(lattice.square().sum(), z, retain_graph=True)[0]
        gb = torch.autograd.grad(original.square().sum(), z, retain_graph=True)[0]
        assert torch.allclose(ga, gb, atol=1e-7, rtol=1e-7)
        q = c.reduce_velocity(captured['coordinates'][offset:offset+c.num_atoms], lattice)
        d = c.root.shape[0]
        proposed = captured['lattice'][i].reshape(3, 3)[:d, :d]
        k = torch.einsum('kij,ij->k', c.tensor(c.metric_basis, z), (proposed+proposed.T)/2)
        expected.append(torch.cat((k, q)))
        offset += c.num_atoms
    for a, b in zip(actual, expected):
        assert torch.allclose(a, b, atol=1e-8, rtol=1e-8)
    parameters = list(model.coordinate_head.parameters())+list(model.lattice_head.parameters())
    ga = torch.autograd.grad(sum(v.square().sum() for v in actual), parameters, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(sum(v.square().sum() for v in expected), parameters, allow_unused=True)
    for a, b in zip(ga, gb):
        if a is not None and b is not None:
            assert torch.allclose(a, b, atol=1e-7, rtol=1e-7)


def test_compiled_constants_are_cached_by_device_and_dtype(device):
    c = compile_descriptor(Descriptor('space', 2, (OrbitSpec('i', 14),)))
    z = c.prior(device)
    first = c.tensor(c.coordinate_basis, z)
    assert first is c.tensor(c.coordinate_basis, z)
    assert c.tensor(c.coordinate_basis, z.double()).dtype == torch.float64
    assert c.tensor(c.elements, z, torch.long).dtype == torch.long
    assert not first.requires_grad


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
@pytest.mark.parametrize('kind,number,letter', [('space', 2, 'i'), ('layer', 37, 'r'), ('plane', 17, 'a'), ('space', 225, 'a')])
def test_prior_cpu_cardinality_preserves_exact_samples_and_rng(device, kind, number, letter):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    compiled = compile_descriptor(Descriptor(kind, number, (OrbitSpec(letter, 14),)))
    generator = torch.Generator(device=device).manual_seed(24)
    legacy = torch.randn(compiled.dof, device=device, generator=generator)
    mask = torch.as_tensor(compiled.state_periodic, device=device)
    legacy[mask] = torch.rand(int(mask.sum()), device=device, generator=generator)
    expected_rng = generator.get_state()
    generator.manual_seed(24)
    actual = compiled.prior(device=device, generator=generator)
    assert torch.equal(actual, legacy) and torch.equal(generator.get_state(), expected_rng)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_graph_edges_include_every_ordered_distinct_atom_pair(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    model = OrbitFlow({}, hidden=16, layers=1, frequencies=2).to(device).eval()
    for kind, number, letter in [('space', 1, 'a'), ('space', 2, 'i'), ('layer', 37, 'r')]:
        compiled = compile_descriptor(Descriptor(kind, number, (OrbitSpec(letter, 14),)))
        state = compiled.prior(device=device)
        _, features, edges, _ = model._features(compiled, state, compiled.expand(state)['lattice'], state.new_zeros(16), None)
        pairs = [(i, j) for i in range(compiled.num_atoms)
                 for j in range(compiled.num_atoms) if i != j]
        expected = torch.tensor(pairs, device=device, dtype=torch.long).reshape(-1, 2).T
        assert torch.equal(edges, expected)
        assert features.shape[0] == len(pairs)


@pytest.mark.parametrize('device', ['cpu', 'cuda'])
def test_reused_geometry_preserves_physical_losses_and_state_parameter_gradients(device):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    torch.manual_seed(9)
    compiled = [compile_descriptor(Descriptor('space', 2, (OrbitSpec('i', 14), OrbitSpec('i', 8)))),
                compile_descriptor(Descriptor('layer', 37, (OrbitSpec('r', 14),))),
                compile_descriptor(Descriptor('space', 225, (OrbitSpec('a', 14),)))]
    states = [c.prior(device=device, dtype=torch.float64).requires_grad_() for c in compiled]
    model = OrbitFlow({}, hidden=16, layers=1, frequencies=2).to(device).double().eval()
    predictions, lattices = model.forward_batch(compiled, states, [0.3]*len(compiled), return_lattices=True)
    targets = [torch.randn_like(state) for state in states]
    legacy = [velocity_loss(c, state, target, prediction) for c, state, target, prediction
              in zip(compiled, states, targets, predictions)]
    reused = [velocity_loss(c, state, target, prediction, lattice) for c, state, target, prediction, lattice
              in zip(compiled, states, targets, predictions, lattices)]
    for first, second in zip(legacy, reused):
        for name in first:
            torch.testing.assert_close(first[name], second[name], rtol=1e-10, atol=1e-10)
    variables = [*states, *model.parameters()]
    first_grads = torch.autograd.grad(sum(loss['loss'] for loss in legacy), variables, retain_graph=True, allow_unused=True)
    second_grads = torch.autograd.grad(sum(loss['loss'] for loss in reused), variables, allow_unused=True)
    for first, second in zip(first_grads, second_grads):
        if first is None:
            assert second is None
        else:
            torch.testing.assert_close(first, second, atol=1e-8, rtol=1e-8)

