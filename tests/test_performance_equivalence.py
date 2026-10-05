import pytest
import torch
from unifiedgroupgen.flow import OrbitFlow, velocity_loss
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor


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
        _, features, edges, _ = model._features(compiled, state, 0.3, {}, False, None, None)
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
            torch.testing.assert_close(first[name], second[name], rtol=0, atol=0)
    variables = [*states, *model.parameters()]
    first_grads = torch.autograd.grad(sum(loss['loss'] for loss in legacy), variables, retain_graph=True, allow_unused=True)
    second_grads = torch.autograd.grad(sum(loss['loss'] for loss in reused), variables, allow_unused=True)
    for first, second in zip(first_grads, second_grads):
        if first is None:
            assert second is None
        else:
            torch.testing.assert_close(first, second, atol=1e-8, rtol=1e-8)

