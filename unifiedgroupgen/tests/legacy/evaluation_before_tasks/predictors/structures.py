"""Convert physical graph geometry to predictor inputs; never reuse learned embeddings."""
import numpy as np
from pymatgen.core import Lattice, Structure


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def structure_from_graph(graph, *, kind, lattice_columns=True):
    """Read one expanded OrbitFlow graph (column lattice) or an explicit row-lattice graph."""
    if kind not in ('space', 'layer'):
        raise ValueError('Predictors currently support space and layer structures')
    lattice = _array(graph['lattice'])
    coords = _array(graph['coordinates'] if 'coordinates' in graph else graph['frac_coords'])
    numbers = _array(graph['atomic_numbers'])
    if lattice.shape != (3, 3) or coords.shape != (len(numbers), 3) or numbers.ndim != 1 or not len(numbers):
        raise ValueError('Expected lattice[3,3], coordinates[N,3] and atomic_numbers[N]')
    if not all(np.isfinite(v).all() for v in (lattice, coords, numbers)):
        raise ValueError('Nonfinite physical graph geometry')
    if not np.equal(numbers, numbers.astype(int)).all() or not ((numbers >= 1) & (numbers <= 118)).all():
        raise ValueError('Invalid atomic numbers')
    rows = lattice.T if lattice_columns else lattice
    if abs(np.linalg.det(rows)) < 1e-10:
        raise ValueError('Singular lattice')
    return Structure(Lattice(rows, pbc=(True, True, kind == 'space')), numbers.astype(int), coords)


def predictor_structure(structure, kind, cutoff):
    """Keep all 3D geometry; put a 2D layer in a cutoff-safe auxiliary vacuum cell."""
    if not structure.is_ordered or not len(structure):
        raise ValueError('Predictor input must be a nonempty, fully ordered structure')
    if not np.isfinite(structure.lattice.matrix).all() or not np.isfinite(structure.cart_coords).all():
        raise ValueError('Nonfinite structure')
    if kind == 'space':
        # UGG expands a column lattice. Its transpose can be Fortran-contiguous,
        # whereas the official pymatgen Cython neighbor builder requires C order.
        return Structure(Lattice(np.array(structure.lattice.matrix, dtype=float, order='C'),
                                 pbc=structure.lattice.pbc), structure.species,
                         np.array(structure.frac_coords, dtype=float, order='C'))
    if kind != 'layer':
        raise ValueError('kind must be space or layer')
    from unifiedgroupgen.tests.legacy.evaluation_before_tasks.standard import layer_structure
    normal = np.cross(*structure.lattice.matrix[:2])
    normal /= np.linalg.norm(normal)
    thickness = float(np.ptp(structure.cart_coords @ normal))
    return layer_structure(structure, height=thickness+2*cutoff+2.)
