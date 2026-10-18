import numpy as np
import pytest

from unifiedgroupgen.structure_rendering import display_geometry, display_bonds, element_color, render_structure, coordination_faces


def test_layer_display_keeps_height_and_only_tiles_periodic_axes():
    lattice = np.diag([3., 3., 1.])
    fractional = np.array([[0., 0., -1.5], [.5, .5, 1.5]])
    original = fractional.copy()
    positions, numbers, owners, corners, edges = display_geometry(
        lattice, fractional, [42, 16], periodic=(True, True, False), repeat=(2, 2, 1))
    assert np.array_equal(fractional, original)
    assert set(positions[:, 2]) == {-1.5, 1.5}
    assert len(corners) == 4 and len(edges) == 4
    assert np.all(corners[:, 2] == 0)
    assert np.array_equal(numbers, np.array([42, 16])[owners])
    with pytest.raises(ValueError, match='Nonperiodic'):
        display_geometry(lattice, fractional, [42, 16], periodic=(True, True, False), repeat=(1, 1, 2))


def test_bulk_closed_cell_has_eight_corner_images_of_one_independent_atom():
    positions, numbers, owners, corners, edges = display_geometry(np.eye(3)*3, [[0, 0, 0]], [22])
    assert len(positions) == len(corners) == 8
    assert len(edges) == 12 and set(owners) == {0}


def test_vesta_colors_and_compound_display_contacts():
    assert element_color(8) == (254, 3, 0)
    assert element_color(22) == (120, 202, 255)
    xyz = np.array([[0., 0, 0], [3., 0, 0], [1.5, 0, 0]])
    assert set(display_bonds(xyz, [22, 22, 8])) == {(0, 2), (1, 2)}


def test_render_is_deterministic_and_preserves_coordinates():
    fractional = np.array([[0., 0, 0], [.5, .5, .5]])
    before = fractional.copy()
    kwargs = dict(size=(320, 240))
    a = render_structure(np.eye(3)*3, fractional, [22, 8], **kwargs)
    b = render_structure(np.eye(3)*3, fractional, [22, 8], **kwargs)
    assert a.size == (320, 240) and a.tobytes() == b.tobytes()
    assert np.array_equal(fractional, before)
    assert np.asarray(a).min() < 100


def test_transparent_export_and_real_octahedral_faces():
    positions = np.array([[0, 0, 0], [1, 0, 0], [-1, 0, 0],
                          [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], dtype=float)
    elements = [22, 8, 8, 8, 8, 8, 8]
    faces = coordination_faces(positions, elements, display_bonds(positions, elements))
    assert len(faces) == 8 and all(center == 22 for _, center in faces)
    assert all(0 not in indices for indices, _ in faces)
    image = render_structure(np.eye(3)*5, (positions+2)/5, elements,
                             size=(320, 320), background=None)
    assert image.mode == 'RGBA' and image.getpixel((0, 0))[3] == 0
    assert np.array(image)[..., 3].max() == 255
    isolated = render_structure(np.eye(3)*3, [[.5, .5, .5]], [22],
                                size=(320, 240), background=None)
    assert np.all(np.asarray(isolated)[115:125, 155:165, 3] == 255)
