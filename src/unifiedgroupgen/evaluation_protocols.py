"""Explicit matching profiles. Existing standard_metrics remains unchanged."""
import numpy as np
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure

from .standard_metrics import layer_structure, structure_match


PROFILES = {
    'legacy_ugg_v1': {'ltol': .2, 'stol': .3, 'angle_tol': 5, 'scale': False},
    'diffcsp_v1': {'ltol': .3, 'stol': .5, 'angle_tol': 10, 'scale': True},
}


def match_pair(prediction, reference, kind, profile='legacy_ugg_v1'):
    spec = PROFILES[profile]
    result = {'match': False, 'rmse_normalized': None, 'rmsd_angstrom': None,
              'length_relative_error': None, 'angle_mae_degrees': None,
              'measure_relative_error': None}
    if prediction.composition.reduced_formula != reference.composition.reduced_formula:
        return result
    if kind == 'layer':
        # Never normalize the open direction by an arbitrary vacuum length.
        matched = structure_match(prediction, reference, kind, spec['ltol'], spec['stol'], spec['angle_tol'])
        height = 10*max(*prediction.lattice.abc[:2], *reference.lattice.abc[:2],
                        np.ptp(prediction.cart_coords @ np.cross(*prediction.lattice.matrix[:2]) /
                               np.linalg.norm(np.cross(*prediction.lattice.matrix[:2]))), 4)
        a, b = layer_structure(reference, height), layer_structure(prediction, height)
        if len(a) < len(b):
            a, b = b, a
        a = Structure(Lattice(a.lattice.matrix), a.species, a.frac_coords)
        b = Structure(Lattice(b.lattice.matrix), b.species, b.frac_coords)
        mapper = StructureMatcher(ltol=spec['ltol'], stol=2, angle_tol=spec['angle_tol'],
                                  scale=False, primitive_cell=False, attempt_supercell=True)
        mapped = mapper.get_s2_like_s1(a, b) if matched else None
        result['match'] = bool(matched)
        if mapped is not None and len(mapped) == len(a):
            delta = mapped.frac_coords-a.frac_coords
            delta[:, :2] -= np.round(delta[:, :2])
            delta[:, 2] -= delta[:, 2].mean()
            rms = float(np.sqrt(np.mean(np.sum((delta @ a.lattice.matrix)**2, axis=1))))
            area = np.linalg.norm(np.cross(*a.lattice.matrix[:2]))
            result.update(rmsd_angstrom=rms, rmse_normalized=rms/np.sqrt(area/len(a)))
    else:
        matcher = StructureMatcher(**spec)
        rms = matcher.get_rms_dist(reference, prediction)
        result['match'] = rms is not None
        result['rmse_normalized'] = float(rms[0]) if rms is not None else None
        # Physical displacement uses an unscaled, explicitly aligned structure.
        mapper = StructureMatcher(ltol=spec['ltol'], stol=spec['stol'], angle_tol=spec['angle_tol'],
                                  scale=False, primitive_cell=False, attempt_supercell=True)
        a, b = reference, prediction
        if len(a) < len(b):
            a, b = b, a
        mapped = mapper.get_s2_like_s1(a, b) if result['match'] else None
        if mapped is not None and len(mapped) == len(a):
            distances = [a.lattice.get_distance_and_image(x, y)[0] for x, y in zip(a.frac_coords, mapped.frac_coords)]
            result['rmsd_angstrom'] = float(np.sqrt(np.mean(np.square(distances))))
    if mapped is not None and len(mapped) == len(a):
        dimensions = 2 if kind == 'layer' else 3
        # get_s2_like_s1 assigns the reference cell; do not report its zero cell error.
        transform = mapper.get_transformation(a, b)
        if transform is not None:
            aligned = Lattice(np.asarray(transform[0]) @ b.lattice.matrix)
            lengths = np.array(a.lattice.abc[:dimensions])
            result['length_relative_error'] = float(np.mean(np.abs(np.array(aligned.abc[:dimensions])/lengths-1)))
            angles = [2] if kind == 'layer' else [0, 1, 2]
            result['angle_mae_degrees'] = float(np.mean(np.abs(np.array(aligned.angles)[angles]-np.array(a.lattice.angles)[angles])))
        ma = np.linalg.norm(np.cross(*reference.lattice.matrix[:2]))/len(reference) if kind == 'layer' else reference.volume/len(reference)
        mb = np.linalg.norm(np.cross(*prediction.lattice.matrix[:2]))/len(prediction) if kind == 'layer' else prediction.volume/len(prediction)
        result['measure_relative_error'] = float(abs(mb/ma-1))
    return result
