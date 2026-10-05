import numpy as np
import pytest
import torch
from scipy.linalg import null_space
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor, group_table, metric_chart, wp_arrays
from unifiedgroupgen.constraints import AffineObservation, tangent_project
from unifiedgroupgen.flow import conditional_path
from unifiedgroupgen.spin import sample_collinear_moments


def test_all_catalog_metrics_are_invariant_and_spd():
    rng = np.random.default_rng(182)
    for kind, count in (("space", 230), ("layer", 80), ("plane", 17)):
        for number in range(1, count+1):
            root, _, basis = metric_chart(kind, number)
            k = torch.tensor(rng.normal(0, 0.2, len(basis)), dtype=torch.float64)
            s = torch.einsum('k,kij->ij', k, torch.tensor(basis))
            h = (torch.tensor(root) @ torch.matrix_exp(s) @ torch.tensor(root)).numpy()
            assert np.linalg.eigvalsh(h).min()>0
            d = h.shape[0]
            for operation in group_table(kind, number)[0].ops:
                r = operation.rotation_matrix[:d, :d]
                assert np.allclose(r.T@h@r, h, atol=1e-7), (kind, number)


def test_all_catalog_orbit_ranks_and_periodic_parameters():
    for kind, count in (("space", 230), ("layer", 80), ("plane", 17)):
        for number in range(1, count+1):
            for wp in group_table(kind, number):
                a, b, axes = wp_arrays(kind, number, wp.letter)
                assert a.shape == (wp.multiplicity, 3, len(axes))
                assert len(axes) == (int(wp.get_dof()) if kind != 'plane' else int(wp.get_dof())-1)
                # Unit free-coordinate wraps expand to integer lattice translations.
                for i, axis in enumerate(axes):
                    if axis < 2 or kind == 'space':
                        assert np.allclose(a[:, :, i], np.round(a[:, :, i])), (kind, number, wp.letter)
                if kind == 'plane':
                    assert np.allclose(a[:, 2], 0) and np.allclose(b[:, 2], 0)


@pytest.mark.parametrize('kind,number,letter', [
    ('space', 1, 'a'), ('space', 2, 'i'), ('space', 2, 'a'),
    ('space', 14, 'e'), ('space', 14, 'a'), ('space', 75, 'd'),
    ('space', 191, 'm'), ('space', 225, 'a'),
    ('layer', 1, 'a'), ('layer', 37, 'r'), ('layer', 37, 'i'),
    ('layer', 80, 'a'), ('plane', 1, 'a'), ('plane', 17, 'a')])
def test_orbit_basis_equals_group_constraint_kernel(kind, number, letter):
    compiled = compile_descriptor(Descriptor(kind, number, (OrbitSpec(letter, 14),)))
    z = compiled.prior(dtype=torch.float64)
    coords = compiled.expand(z)['coordinates'].numpy()
    c = compiled.constraint_matrix(coords)
    a = compiled.coordinate_basis.reshape(3*compiled.num_atoms, compiled.coordinate_dof)
    assert np.linalg.norm(c@a)<1e-7
    kernel = null_space(c, rcond=1e-8) if np.max(np.abs(c))>1e-8 else np.eye(c.shape[1])
    assert kernel.shape[1] == compiled.coordinate_dof
    if compiled.coordinate_dof:
        assert np.allclose(a@np.linalg.pinv(a), kernel@kernel.T, atol=1e-7)


def test_repeated_general_orbits_and_single_special_occupancy():
    compiled = compile_descriptor(Descriptor('space', 2, (OrbitSpec('i', 8), OrbitSpec('i', 8), OrbitSpec('a', 14))))
    assert compiled.num_atoms == 5 and compiled.coordinate_dof == 6
    with pytest.raises(ValueError, match='twice'):
        compile_descriptor(Descriptor('space', 2, (OrbitSpec('a', 8), OrbitSpec('a', 14))))


def test_entire_periodic_training_path_including_affine_translation_is_legal():
    compiled = compile_descriptor(Descriptor('space', 14, (OrbitSpec('e', 8),)))
    endpoint = compiled.prior(dtype=torch.float64)
    torch.manual_seed(23)
    start, velocity = conditional_path(compiled, endpoint, 0.0)
    full_velocity = compiled.coordinate_velocity(velocity[compiled.lattice_dof:]).numpy()
    for t in np.linspace(0, 1, 13):
        z = start+t*velocity
        coords = compiled.expand(z)['coordinates'].numpy()
        for rotation, permutation in compiled.permutations(coords):
            assert np.allclose(full_velocity[permutation], full_velocity@rotation.T, atol=1e-7)
    final = compiled.expand(start+velocity)['coordinates'].numpy()
    target = compiled.expand(endpoint)['coordinates'].numpy()
    diff = final-target
    assert np.allclose(diff-np.round(diff), 0, atol=1e-7)


def test_weighted_projection_and_common_affine_fiber():
    compiled = compile_descriptor(Descriptor('space', 191, (OrbitSpec('m', 8),)))
    state = compiled.prior(dtype=torch.float64)
    lattice = compiled.expand(state)['lattice']
    candidate = torch.randn(compiled.num_atoms, 3, dtype=torch.float64)
    reduced = compiled.reduce_velocity(candidate, lattice)
    residual = candidate-compiled.coordinate_velocity(reduced)
    a = torch.tensor(compiled.coordinate_basis)
    assert torch.linalg.norm(torch.einsum('iaq,ab,ib->q', a, lattice.T@lattice, residual))<1e-7
    matrix = torch.randn(2, compiled.dof, dtype=torch.float64)
    observation = AffineObservation(matrix, matrix@state)
    projected = observation.retract(torch.randn_like(state))
    assert torch.allclose(matrix@projected, matrix@state, atol=1e-7)
    velocity = observation.project(torch.randn_like(state))
    assert torch.linalg.norm(matrix@velocity)<1e-7
    j = torch.tensor([[1., 1.]], dtype=torch.float64)
    v = torch.tensor([2., -0.4], dtype=torch.float64)
    metric = torch.diag(torch.tensor([2., 4.], dtype=torch.float64))
    assert torch.linalg.norm(j@tangent_project(v, j, metric))<1e-7


def test_layer_height_is_not_periodic_and_plane_height_is_zero():
    layer = compile_descriptor(Descriptor('layer', 1, (OrbitSpec('a', 8),)))
    z = torch.zeros(layer.dof, dtype=torch.float64)
    z[-1] = 2.3
    assert layer.expand(z)['coordinates'][0, 2] == 2.3
    plane = compile_descriptor(Descriptor('plane', 1, (OrbitSpec('a', 8),)))
    assert plane.coordinate_dof == 2
    assert plane.expand(plane.prior())['coordinates'][0, 2] == 0


def test_collinear_unit_spins_use_the_same_representation_constraint():
    permutation = [1, 0]
    operations = [(-np.eye(3), permutation)]
    moments = sample_collinear_moments([1, -1], operations)
    assert np.allclose(np.linalg.norm(moments, axis=1), 1)
    assert np.allclose(moments[permutation], moments@(-np.eye(3)).T)
    with pytest.raises(ValueError, match='nonzero'):
        sample_collinear_moments([1, 1], operations)
