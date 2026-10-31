"""Deterministic Crystal Toolkit inspired views of actual Cartesian geometry.

Uses pymatgen's VESTA palette and Crystal Toolkit's uniform 0.5 A sphere
radius. Rendering is orthographic, with matte spheres, coordination hulls,
split-color bonds and an orientation compass. PNG backgrounds may be transparent.
Distance-cutoff contacts are display aids, not chemical-validity evidence.
In mixed metal/nonmetal compounds, metal-metal contacts are omitted.
Only the declared periodic axes are tiled; layer heights are never wrapped.
"""
from functools import lru_cache
from itertools import product
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


@lru_cache(maxsize=1)
def _palette():
    import pymatgen.vis
    import yaml
    path = Path(pymatgen.vis.__path__[0]) / 'ElementColorSchemes.yaml'
    return yaml.safe_load(path.read_text(encoding='utf-8'))['VESTA']


def element_color(number):
    from pymatgen.core import Element
    return tuple(_palette().get(Element.from_Z(int(number)).symbol, (180, 180, 180)))


def display_geometry(lattice, fractional, elements, periodic=(True, True, True), repeat=None):
    """Row-vector lattice convention; include periodic images on closed faces."""
    lattice, fractional = np.asarray(lattice), np.array(fractional, dtype=float, copy=True)
    periodic = np.asarray(periodic, dtype=bool)
    repeat = np.array(repeat if repeat is not None else (1, 1, 1), dtype=int)
    if np.any(repeat[~periodic] != 1):
        raise ValueError('Nonperiodic directions cannot be repeated')
    fractional[:, periodic] %= 1
    positions, numbers, owners = [], [], []
    for i, (f, z) in enumerate(zip(fractional, elements)):
        shifts = [range(repeat[j]+1) if periodic[j] and abs(f[j]) < 1e-8
                  else range(repeat[j]) if periodic[j] else (0,) for j in range(3)]
        for shift in product(*shifts):
            positions.append((f + shift) @ lattice)
            numbers.append(int(z)); owners.append(i)
    axes = np.flatnonzero(periodic)
    corners = np.array([sum((bit*repeat[j]*lattice[j] for bit, j in zip(bits, axes)), np.zeros(3))
                        for bits in product((0, 1), repeat=len(axes))])
    edges = [(i, j) for i in range(len(corners)) for j in range(i+1, len(corners))
             if (i ^ j).bit_count() == 1]
    return np.array(positions), np.array(numbers), np.array(owners), corners, edges


def display_bonds(positions, elements):
    """Current-frame covalent-radius contacts; omit metal-metal links in compounds."""
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
    from pymatgen.core import Element
    species = [Element.from_Z(int(z)) for z in elements]
    radii = [CovalentRadius.radius.get(el.symbol, 1.) for el in species]
    mixed = any(el.is_metal for el in species) and any(not el.is_metal for el in species)
    pairs = []
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if mixed and species[i].is_metal and species[j].is_metal:
                continue
            d = np.linalg.norm(positions[j] - positions[i])
            if .35 < d <= 1.15*(radii[i]+radii[j]):
                pairs.append((i, j))
    return pairs


@lru_cache(maxsize=256)
def _sphere(radius, color):
    radius = max(2, int(radius)); size = radius*2+3
    y, x = np.mgrid[:size, :size].astype(float)
    x = (x-radius-1)/radius; y = (y-radius-1)/radius
    mask = x*x+y*y <= 1
    z = np.sqrt(np.maximum(0, 1-x*x-y*y))
    light = np.maximum(0, -.36*x-.46*y+.81*z)
    rgb = np.asarray(color)[None, None, :]*(.35+.65*light[..., None])
    rgba = np.empty((size, size, 4), dtype=np.uint8)
    rgba[..., :3] = np.clip(rgb, 0, 255); rgba[..., 3] = mask*255
    return Image.fromarray(rgba)


def coordination_faces(positions, elements, bonds):
    """Actual neighbor convex hulls; do not invent faces for planar/low-CN sites."""
    from pymatgen.core import Element
    from scipy.spatial import ConvexHull, QhullError
    faces = []
    for i, number in enumerate(elements):
        if not Element.from_Z(int(number)).is_metal:
            continue
        neighbors = [b if a == i else a for a, b in bonds if a == i or b == i]
        if len(neighbors) < 4:
            continue
        try:
            hull = ConvexHull(positions[neighbors])
        except QhullError:
            continue
        if hull.volume > 1e-8:
            faces.extend((np.array(neighbors)[face], int(number)) for face in hull.simplices)
    return faces


def render_structure(lattice, fractional, elements, *, periodic=(True, True, True),
                     repeat=None, size=(900, 650), angle=.45, view='isometric',
                     vectors=None, vector_color='#d9483b', background='white'):
    """Return a PIL image without altering coordinates or the input cell."""
    xyz, numbers, owners, corners, edges = display_geometry(lattice, fractional, elements, periodic, repeat)
    ca, sa = np.cos(angle), np.sin(angle)
    rotation = np.array([[ca, -sa, 0], [sa*.42, ca*.42, -.907], [sa*.907, ca*.907, .42]])
    if view == 'top':
        rotation = np.array([[1., 0, 0], [0, -1., 0], [0, 0, 1.]])
    elif view == 'side':
        rotation = np.array([[1., 0, 0], [0, 0, -1.], [0, 1., 0]])
    points = np.concatenate([xyz, corners]) @ rotation.T
    low, high = points[:, :2].min(0)-.65, points[:, :2].max(0)+.65
    width, height = size
    scale = min((width-36)/max(high[0]-low[0], .1), (height-36)/max(high[1]-low[1], .1))
    points[:, :2] = (points[:, :2]-(low+high)/2)*scale + [width/2, height/2]
    image = Image.new('RGBA' if background is None else 'RGB', size,
                      (0, 0, 0, 0) if background is None else background)
    draw = ImageDraw.Draw(image)
    atom_points, cell_points = points[:len(xyz)], points[len(xyz):]
    for i, j in edges:
        draw.line([tuple(cell_points[i, :2]), tuple(cell_points[j, :2])], fill='#6e7781', width=max(1, width//550))
    bonds = display_bonds(xyz, numbers)
    primitives = [(float(atom_points[face, 2].mean()), 'face', (atom_points[face], element_color(number)))
                  for face, number in coordination_faces(xyz, numbers, bonds)]
    for i, j in bonds:
        a, b = atom_points[i], atom_points[j]
        # Stop the cylinder at the sphere surface in 3D, then split at midpoint.
        distance = np.linalg.norm(xyz[j]-xyz[i]); inset = min(.5/distance, .35)
        start, end = a+(b-a)*inset, b-(b-a)*inset
        middle = (a+b)/2
        primitives.extend([(float((start[2]+middle[2])/2), 'bond', (start, middle, element_color(numbers[i]))),
                           (float((middle[2]+end[2])/2), 'bond', (middle, end, element_color(numbers[j])))])
    for i, point in enumerate(atom_points):
        primitives.append((float(point[2]), 'atom', (point, element_color(numbers[i]))))
    radius = max(3, round(.5*scale))
    # Resolve sphere surfaces in depth before compositing transparent faces.
    # Faces meet ligand centers, so painter sorting alone would paint over ligands.
    opaque_depth = np.full((height, width), -np.inf)
    sphere_size = radius*2+3
    sy, sx = np.mgrid[:sphere_size, :sphere_size]
    radial = ((sx-radius-1)/radius)**2+((sy-radius-1)/radius)**2
    surface = .5*np.sqrt(np.maximum(0, 1-radial))
    surface[radial > 1] = -np.inf
    for point in atom_points:
        x, y = round(point[0])-radius-1, round(point[1])-radius-1
        tile = opaque_depth[y:y+sphere_size, x:x+sphere_size]
        np.maximum(tile, point[2]+surface, out=tile)
    for _, kind, data in sorted(primitives, key=lambda value: value[0]):
        if kind == 'face':
            vertices, color = data
            low = np.maximum(0, np.floor(vertices[:, :2].min(0)-1).astype(int))
            high = np.minimum(size, np.ceil(vertices[:, :2].max(0)+2).astype(int))
            patch = Image.new('RGBA', tuple(high-low))
            ImageDraw.Draw(patch).polygon([tuple(v[:2]-low) for v in vertices],
                                         fill=(*color, 68), outline=(*color, 115))
            matrix = np.column_stack([vertices[:, :2], np.ones(3)])
            if abs(np.linalg.det(matrix)) < 1e-8:
                continue
            a, b, c = np.linalg.solve(matrix, vertices[:, 2])
            yy, xx = np.mgrid[low[1]:high[1], low[0]:high[0]]
            pixels = np.array(patch)
            pixels[..., 3][a*xx+b*yy+c <= opaque_depth[low[1]:high[1], low[0]:high[0]]] = 0
            patch = Image.fromarray(pixels)
            if image.mode == 'RGBA':
                image.alpha_composite(patch, tuple(low))
            else:
                image.paste(patch, tuple(low), patch)
            draw = ImageDraw.Draw(image)
        elif kind == 'bond':
            start, end, color = data
            thickness = max(2, round(.13*scale))
            draw.line([tuple(start[:2]), tuple(end[:2])], fill=tuple(int(v*.68) for v in color), width=thickness)
            draw.line([tuple(start[:2]-[1, 1]), tuple(end[:2]-[1, 1])], fill=color, width=max(1, thickness//2))
        else:
            point, color = data; sphere = _sphere(radius, color)
            origin = (round(point[0])-radius-1, round(point[1])-radius-1)
            pixels = np.array(sphere)
            tile = opaque_depth[origin[1]:origin[1]+sphere_size, origin[0]:origin[0]+sphere_size]
            pixels[..., 3][point[2]+surface < tile] = 0
            sphere = Image.fromarray(pixels)
            if image.mode == 'RGBA':
                image.alpha_composite(sphere, origin)
            else:
                image.paste(sphere, origin, sphere)
    if vectors is not None:
        for point, owner in zip(atom_points, owners):
            delta = np.asarray(vectors[owner]) @ rotation.T
            start = point[:2]; end = start + delta[:2]*18
            norm = np.linalg.norm(end-start)
            if norm < 1:
                continue
            draw.line([tuple(start), tuple(end)], fill=vector_color, width=3)
            if norm > 7:
                u = (end-start)/norm; normal = np.array([-u[1], u[0]])
                draw.polygon([tuple(end), tuple(end-10*u+4*normal), tuple(end-10*u-4*normal)], fill=vector_color)
    origin = np.array([max(32, width*.07), height-max(32, height*.08)])
    for axis, color in enumerate(('#ff0000', '#009900', '#0000ff')):
        direction = rotation[:2, axis]
        end = origin + direction*max(23, min(size)*.09)
        norm = np.linalg.norm(end-origin)
        if norm < 3:
            draw.ellipse((*tuple(origin-3), *tuple(origin+3)), fill=color)
            continue
        unit = (end-origin)/norm; normal = np.array([-unit[1], unit[0]])
        draw.line([tuple(origin), tuple(end)], fill=color, width=max(2, width//220))
        draw.polygon([tuple(end), tuple(end-8*unit+4*normal), tuple(end-8*unit-4*normal)], fill=color)
    return image
