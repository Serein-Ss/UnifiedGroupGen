"""Algebraic and geometric checks. These do not certify chemical stability or target properties."""

import numpy as np
from pymatgen.core import Lattice, Structure
import spglib


def summarize_attempts(records):
    """Use every requested attempt in operational validity rates, including failures."""
    total = len(records)
    generated = [r for r in records if r['status'] == 'generated']
    reports = [r['validation'] for r in generated]
    counts = {key: sum(bool(r.get(key)) for r in reports) for key in
              ('group_operation_closure', 'distance_valid', 'orbit_multiplicity_regular',
               'no_coincident_orbits', 'chemistry_validated', 'stability_validated', 'properties_validated')}
    identified = [r for r in reports if r.get('exact_target_group') is not None]
    return {'attempted': total, 'generated': len(generated), 'failed': total-len(generated),
            'counts': counts, 'rates_over_all_attempts': {key: count/total if total else None for key, count in counts.items()},
            'exact_group_identified': len(identified),
            'exact_group_success': sum(bool(r['exact_target_group']) for r in identified),
            'exact_group_unidentified_generated': len(reports)-len(identified),
            'scope': 'Distance threshold is a geometric diagnostic; chemistry, stability and property hit require external verification.'}


def to_structure(compiled, generated, vacuum=20.0):
    lattice = generated["lattice"].detach().cpu().numpy().copy()
    coordinates = generated["coordinates"].detach().cpu().numpy().copy()
    if compiled.descriptor.kind != "space":
        thickness = np.ptp(coordinates[:, 2])
        cell_height = max(vacuum, thickness+10)
        lattice[2, 2] = cell_height
        coordinates[:, 2] = (coordinates[:, 2] - coordinates[:, 2].mean())/cell_height + 0.5
    # pymatgen stores lattice vectors as rows.
    pmg_lattice = Lattice(lattice.T, pbc=tuple(compiled.periodic))
    return Structure(pmg_lattice, compiled.elements.tolist(), coordinates)


def validate(compiled, generated, minimum_distance=0.6, isolate_identifier=False):
    coords = generated["coordinates"].detach().cpu().numpy()
    lattice = generated["lattice"].detach().cpu().numpy()
    residual = 0.0
    for rotation, permutation in compiled.permutations(coords, tolerance=1e-4):
        # Closure checked including affine translations in permutations().
        h = lattice.T@lattice
        residual = max(residual, float(np.max(np.abs(rotation.T@h@rotation-h))))
    structure = to_structure(compiled, generated)
    distances = structure.distance_matrix
    np.fill_diagonal(distances, np.inf)
    closest = float(distances.min()) if len(structure)>1 else None
    orbit_regular = True
    for index in range(len(compiled.blocks)):
        ids = np.flatnonzero(compiled.atom_orbit == index)
        if len(ids)>1 and np.min(distances[np.ix_(ids, ids)])<1e-5:
            orbit_regular = False
    report = {"target_kind": compiled.descriptor.kind, "target_group": compiled.descriptor.number,
              "atoms": compiled.num_atoms, "coordinate_dof": compiled.coordinate_dof,
              "lattice_dof": compiled.lattice_dof, "group_operation_closure": True,
              "metric_residual": residual, "minimum_distance_angstrom": closest,
              "distance_valid": closest is None or closest >= minimum_distance,
              "orbit_multiplicity_regular": orbit_regular,
              "no_coincident_orbits": closest is None or closest>1e-5,
              "chemistry_validated": False, "stability_validated": False, "properties_validated": False}
    if isolate_identifier and compiled.descriptor.kind in ('space', 'layer'):
        from .native_identifier import identify_isolated
        identified = identify_isolated(structure, compiled.descriptor.kind)
        number = identified['number']
        report[f'recovered_{compiled.descriptor.kind}_group'] = number
        report['exact_target_group'] = number == compiled.descriptor.number if number is not None else None
        if compiled.descriptor.kind == 'layer':
            report['exact_layer_group'] = report['exact_target_group']
        report['group_identifier'] = 'isolated spglib, symprec=0.001 Angstrom; layer aperiodic_dir=2'
        report['group_identification_error'] = identified.get('error')
    elif compiled.descriptor.kind == "space":
        result = spglib.get_symmetry_dataset((structure.lattice.matrix, structure.frac_coords, compiled.elements), symprec=1e-3)
        report["recovered_space_group"] = int(result.number) if result is not None else None
        report["exact_target_group"] = result.number == compiled.descriptor.number if result is not None else None
    elif compiled.descriptor.kind == 'layer' and hasattr(spglib, 'get_layergroup'):
        result = spglib.get_layergroup((structure.lattice.matrix, structure.frac_coords, compiled.elements),
                                      aperiodic_dir=2, symprec=1e-3)
        report['recovered_layer_group'] = int(result.number) if result is not None else None
        report['exact_layer_group'] = result.number == compiled.descriptor.number if result is not None else None
        report['exact_target_group'] = report['exact_layer_group']
        report['group_identifier'] = 'spglib.get_layergroup, aperiodic_dir=2, symprec=0.001 Angstrom'
    else:
        report["exact_layer_group"] = None
    return report
