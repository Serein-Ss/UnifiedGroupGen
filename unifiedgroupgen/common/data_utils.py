"""Crystal parsing, information extraction, constraints and periodic graph utilities."""

from collections import Counter
from dataclasses import dataclass, replace
from functools import lru_cache, reduce
import hashlib
from itertools import islice, product
from math import gcd
import json
import re
import warnings

import numpy as np
import pandas as pd
from pymatgen.analysis.structure_matcher import StructureMatcher
from pymatgen.core import Lattice, Structure
from scipy.optimize import linear_sum_assignment
import spglib
import torch

from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor, group_table
from unifiedgroupgen.group_constraints.encoding import _delta, _norm, decompose_orbits, operation_residual
from unifiedgroupgen.group_constraints.operations import compile_operations
from unifiedgroupgen.group_constraints.representation import compiled_chart, expand_numpy
from unifiedgroupgen.group_constraints.state import JointState
from unifiedgroupgen.group_constraints.unified import SymmetrySpec, catalog_spec, compile_constraints


def parse_cif(text):
    """Parse ordered fractional CIF coordinates without fraction snapping."""
    structure = Structure.from_str(text, fmt='cif', frac_tolerance=0)
    if not structure.is_ordered:
        raise ValueError('Disordered/partially occupied structures are not supported')
    return structure


def _layer_origin_candidates(coords, elements, operations, tolerance=1e-3, physical_lattice=None):
    yield np.zeros(2)
    # Pick a rare species to reduce possible symmetry partners for an anchor.
    species = min(np.unique(elements), key=lambda z: int(np.sum(elements == z)))
    ids = np.flatnonzero(elements == species)
    anchor = coords[ids[0]]
    rank_two, rank_one = [], []
    for op in operations:
        c = np.eye(2)-op.rotation_matrix[:2, :2]
        rank = np.linalg.matrix_rank(c, tol=1e-8)
        if rank == 2:
            rank_two.append((op, c))
        elif rank == 1:
            rank_one.append((op, c))
    if rank_two:
        op, c = rank_two[0]
        for partner in coords[ids]:
            if abs(partner[2]-op.rotation_matrix[2, 2]*anchor[2]-op.translation_vector[2]) > tolerance:
                continue
            rhs = partner[:2]-op.rotation_matrix[:2, :2]@anchor[:2]-op.translation_vector[:2]
            for image in product((-1, 0, 1), repeat=2):
                yield np.linalg.solve(c, rhs+image) % 1
        return
    if not rank_one:
        return
    selected = [rank_one[0]]
    for candidate in rank_one[1:]:
        if np.linalg.matrix_rank(np.concatenate((selected[0][1], candidate[1]))) == 2:
            selected.append(candidate)
            break
    right_sides = []
    for op, c in selected:
        values = []
        for partner in coords[ids]:
            rhs = partner[:2]-op.rotation_matrix[:2, :2]@anchor[:2]-op.translation_vector[:2]
            for image in product((-1, 0, 1), repeat=2):
                b = rhs+image
                residual = np.r_[c@np.linalg.pinv(c)@b-b, 0.0]
                if _norm(residual, physical_lattice) <= tolerance:
                    values.append(b)
        right_sides.append(values)
    matrix = np.concatenate([c for _, c in selected])
    for values in product(*right_sides):
        yield (np.linalg.pinv(matrix) @ np.concatenate(values)) % 1


def _layer_record_in_setting(structure, number, tolerance=1e-4, orbit_labels=None):
    rows = structure.lattice.matrix
    e1 = rows[0]/np.linalg.norm(rows[0])
    normal = np.cross(rows[0], rows[1])
    normal /= np.linalg.norm(normal)
    e2 = np.cross(normal, e1)
    plane = np.array([[rows[0]@e1, rows[1]@e1], [rows[0]@e2, rows[1]@e2]])
    cart = structure.cart_coords
    xy = np.linalg.solve(plane, np.stack((cart@e1, cart@e2))).T
    heights = cart@normal
    # The single isolated layer may cross the vacuum boundary in a valid fractional CIF.
    cell_height = abs(rows[2]@normal)
    if cell_height > 0:
        wrapped = heights % cell_height
        ordered = np.sort(wrapped)
        gaps = np.diff(np.r_[ordered, ordered[0]+cell_height])
        cut = ordered[(int(np.argmax(gaps))+1) % len(ordered)]
        heights = (wrapped-cut) % cell_height
    heights -= heights.mean()
    f = np.column_stack((xy % 1, heights))
    elements = np.array(structure.atomic_numbers)
    operations = group_table("layer", number)[0].ops
    periodic = np.array([True, True, False])
    # Test signed axis permutations; this preserves the 2D translation lattice and does not mix vacuum into it.
    transforms = [np.eye(2), np.array([[0, 1], [1, 0]])]
    for base in transforms:
        for signs in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
            transform = np.diag(signs)@base
            candidate = f.copy()
            candidate[:, :2] = f[:, :2]@transform.T
            h2 = transform @ (plane.T@plane) @ transform.T
            physical_lattice = np.eye(3)
            physical_lattice[:2, :2] = np.linalg.cholesky(h2).T
            seen = set()
            for origin in _layer_origin_candidates(candidate, elements, operations, tolerance, physical_lattice):
                key = tuple(np.round(origin % 1, 7))
                if key in seen:
                    continue
                seen.add(key)
                adjusted = candidate.copy()
                adjusted[:, :2] -= origin
                residual = operation_residual(adjusted, elements, operations, periodic, physical_lattice)
                if residual > tolerance:
                    continue
                descriptor, q = decompose_orbits("layer", number, adjusted, elements, tolerance, physical_lattice,
                                                orbit_labels=orbit_labels)
                compiled = compile_descriptor(descriptor)
                k = compiled.encode_metric(h2, tolerance=max(tolerance*10, 1e-4))
                encoded = compiled.b + np.einsum('ijq,q->ij', compiled.coordinate_basis, q)
                fitted_distances = []
                for element in np.unique(elements):
                    old = adjusted[elements == element]
                    new = encoded[compiled.elements == element]
                    costs = _norm(_delta(old[:, None], new[None], periodic), physical_lattice)
                    first, second = linear_sum_assignment(costs)
                    fitted_distances.extend(costs[first, second].tolist())
                if max(fitted_distances) > tolerance:
                    continue
                return descriptor, np.r_[k, q], {"origin_xy": origin.tolist(), "axis_transform": transform.tolist(),
                                                "operation_residual_angstrom": residual,
                                                "coordinate_tolerance_angstrom": tolerance,
                                                "chart_fit_rms_angstrom": float(np.sqrt(np.mean(np.square(fitted_distances)))),
                                                "chart_fit_max_angstrom": float(max(fitted_distances))}
    raise ValueError(f"Declared layer group {number} could not be matched; check coordinates, setting, or tolerance")


SYMMETRY_PROTOCOL = 'single-pass-standard-orbits-v1'
STANDARD_INTEGRITY_PROTOCOL = 'source-standard-map-integrity-v1'


def _standard_orbit_labels(result):
    """Carry original equivalence classes through a possibly expanded standard cell."""
    labels = {}
    # Standard-cell orbits include symmetries absent from a nonstandard input
    # supercell. equivalent_atoms describes the latter and can split an orbit.
    for primitive, equivalent in zip(result.mapping_to_primitive, result.crystallographic_orbits):
        if int(primitive) in labels and labels[int(primitive)] != int(equivalent):
            raise ValueError('Inconsistent primitive-to-orbit mapping')
        labels[int(primitive)] = int(equivalent)
    return np.array([labels[int(i)] for i in result.std_mapping_to_primitive])


def _standard_space_integrity(structure, result, compiled, state, tolerance, angle_tolerance):
    """Verify the native correspondence, without another primitive/basis search."""
    source_counts = reduced_counts(structure.atomic_numbers)
    encoded_counts = reduced_counts(compiled.elements)
    fitted = compiled.b + np.einsum('ijq,q->ij', compiled.coordinate_basis, state[compiled.lattice_dof:])
    # First map native standard atoms bijectively onto the encoded standard cell.
    standard_to_encoded = np.empty(len(result.std_types), dtype=int)
    for element in np.unique(result.std_types):
        native = np.flatnonzero(result.std_types == element)
        encoded = np.flatnonzero(compiled.elements == element)
        cost = _norm(_delta(result.std_positions[native, None], fitted[None, encoded], compiled.periodic),
                     result.std_lattice.T)
        rows, columns = linear_sum_assignment(cost)
        standard_to_encoded[native[rows]] = encoded[columns]
    transformed = structure.frac_coords @ result.transformation_matrix.T + result.origin_shift
    residuals = []
    for primitive in np.unique(result.mapping_to_primitive):
        source_ids = np.flatnonzero(result.mapping_to_primitive == primitive)
        std_ids = np.flatnonzero(result.std_mapping_to_primitive == primitive)
        target_ids = standard_to_encoded[std_ids]
        if not len(target_ids) or not np.all(np.asarray(structure.atomic_numbers)[source_ids, None]
                                              == compiled.elements[target_ids][None]):
            raise ValueError('Native source-to-standard mapping changed species')
        delta = _delta(transformed[source_ids, None], fitted[None, target_ids], compiled.periodic)
        cost = _norm(delta, result.std_lattice.T)
        residuals.extend(delta[np.arange(len(source_ids)),cost.argmin(axis=1)].tolist())
    # Absolute origin is a gauge. Compare after one shared rigid translation,
    # never after independently moving atoms or changing the endpoint.
    residuals = np.asarray(residuals)
    alignment = residuals.mean(axis=0)
    distances = _norm(_delta(residuals, alignment, compiled.periodic), result.std_lattice.T)
    before = Lattice(np.linalg.inv(result.transformation_matrix).T @ structure.lattice.matrix)
    after = Lattice(result.std_lattice)
    lattice_tol = max(1e-5, tolerance/min(*before.abc, *after.abc))
    length_change = float(np.max(np.abs(np.asarray(after.abc)/before.abc-1)))
    angle_change = float(np.max(np.abs(np.asarray(after.angles)-before.angles)))
    measure_change = abs(after.volume/len(compiled.elements)/(structure.volume/len(structure))-1)
    maximum = float(max(distances))
    from scipy.linalg import eigvalsh
    strain = np.sqrt(eigvalsh(after.metric_tensor, before.metric_tensor))
    # symprec/angle_tolerance were applied by the native search in its input
    # basis. Do not impose a second length/angle cutoff in a different basis.
    valid = (source_counts == encoded_counts and maximum <= tolerance*(1+1e-5)+1e-8
             and measure_change <= (1+lattice_tol)**3-1+1e-8)
    check = dict(protocol=STANDARD_INTEGRITY_PROTOCOL, source_composition=source_counts,
        encoded_composition=encoded_counts, composition_conserved=source_counts==encoded_counts,
        source_atoms=len(structure),encoded_atoms=compiled.num_atoms,tolerance_angstrom=float(tolerance),
        angle_tolerance_degrees=float(angle_tolerance),geometry_equivalent=bool(valid),
        max_displacement_angstrom=maximum,measure_per_atom_relative_error=float(measure_change),
        lattice_relative_tolerance=float(lattice_tol),lattice_length_relative_change=length_change,
        lattice_angle_change_degrees=angle_change,max_principal_strain=float(np.max(np.abs(strain-1))),
        numerically_identical=bool(valid and maximum<1e-4 and np.max(np.abs(strain-1))<1e-5),
        scale_normalization=False,periodic_dimension=3,
        alignment_translation_fractional=alignment.tolist(),
        displacement_metric='Native correspondence with one shared translation at the ideal lattice; strain reported separately')
    if not valid:raise ValueError('Source integrity failed: native standard correspondence exceeds the declared budget')
    return check


def standard_record(structure, kind, tolerance, angle_tolerance=5., declared_number=None):
    """Identify from the source, standardize once, then encode fixed atom orbits.

    A Hall-setting request may query the SAME source again. It never identifies
    an already refined structure. Layer chart fitting keeps the native orbit
    partition; its finite residual is recorded rather than triggering a new group.
    """
    cell = (np.asarray(structure.lattice.matrix, dtype=np.float64),
            np.asarray(structure.frac_coords, dtype=np.float64), structure.atomic_numbers)
    if kind == 'space':
        result = spglib.get_symmetry_dataset(cell, symprec=tolerance, angle_tolerance=angle_tolerance)
        if result is not None:
            hall = group_table(kind, int(result.number)).hall_number
            if result.hall_number != hall:
                number = result.number
                result = spglib.get_symmetry_dataset(cell, symprec=tolerance,
                                                     angle_tolerance=angle_tolerance, hall_number=hall)
                if result is None or result.number != number or result.hall_number != hall:
                    raise ValueError('Could not retain the identified group in the compiler Hall setting')
    elif kind == 'layer':
        # Cyclic basis permutation, not reduction or rounding. spglib 2.6's
        # axis-2 route hangs for near-square 2HKO-2 at 0.01 A; its axis-0 route
        # agrees with 2.8 on the original cell. Standard output has normal axis 2.
        permutation = np.eye(3)[[2, 0, 1]]
        native_cell = (permutation @ cell[0], cell[1] @ permutation.T, cell[2])
        result = spglib.get_layergroup(native_cell, aperiodic_dir=0, symprec=tolerance)
    else:
        raise ValueError('Standard CSV protocol supports space and layer groups')
    if result is None:
        raise ValueError('Symmetry identification failed; no fallback was performed')
    labels = _standard_orbit_labels(result)
    elements = np.asarray(result.std_types)
    if reduced_counts(elements) != reduced_counts(structure.atomic_numbers):
        raise ValueError('Standardization changed source composition')
    if kind == 'layer':
        standardized = Structure(Lattice(result.std_lattice, pbc=(True, True, False)),
                                 elements, result.std_positions)
        descriptor, state, provenance = _layer_record_in_setting(standardized, int(result.number),
                                                                tolerance, orbit_labels=labels)
        provenance.update(native_aperiodic_dir=0, native_axis_order=[2,0,1])
    else:
        descriptor, q = decompose_orbits(kind, int(result.number), result.std_positions, elements,
                                         tolerance, result.std_lattice.T, orbit_labels=labels)
        compiled = compile_descriptor(descriptor)
        k = compiled.encode_metric(result.std_lattice @ result.std_lattice.T, tolerance=1e-5)
        state = np.r_[k, q]
        fitted = compiled.b + np.einsum('ijq,q->ij', compiled.coordinate_basis, q)
        # Quantify the one fixed-orbit coordinate fit separately from native
        # standardization. No second symmetry search or tolerance retry occurs.
        distances = []
        for element in np.unique(elements):
            costs = _norm(_delta(fitted[compiled.elements == element, None],
                                result.std_positions[None, elements == element], compiled.periodic),
                          result.std_lattice.T)
            first, second = linear_sum_assignment(costs)
            distances.extend(costs[first, second].tolist())
        provenance = {'hall_number': int(result.hall_number), 'chart_fit_max_angstrom': float(max(distances)),
                      'chart_fit_rms_angstrom': float(np.sqrt(np.mean(np.square(distances))))}
    compiled = compile_descriptor(descriptor)
    if len(descriptor.orbits) != len(np.unique(labels)) or compiled.num_atoms != len(elements):
        raise ValueError('Encoding changed the identified atom-orbit partition')
    if reduced_counts(compiled.elements) != reduced_counts(structure.atomic_numbers):
        raise ValueError('Orbit encoding changed source composition')
    encoded = compiled.b + np.einsum('ijq,q->ij', compiled.coordinate_basis, state[compiled.lattice_dof:])
    compiled.permutations(encoded, tolerance=1e-5)
    provenance.update({'symmetry_protocol': SYMMETRY_PROTOCOL, 'cell': 'spglib_'+kind+'_conventional',
        'source_atoms': len(structure), 'expanded_atoms': compiled.num_atoms,
        'identified_orbits': len(np.unique(labels)), 'symmetry_tolerance_angstrom': float(tolerance),
        'angle_tolerance_degrees': float(angle_tolerance) if kind == 'space' else None,
        'cif_fractional_rounding_disabled': True, 'niggli_reduction': False,
        'declared_layer_group': declared_number if kind == 'layer' else None,
        'identified_layer_group': int(result.number) if kind == 'layer' else None,
        'transformation_matrix': (result.transformation_matrix @ permutation if kind == 'layer'
                                  else result.transformation_matrix).tolist(),
        'origin_shift': result.origin_shift.tolist(),
        'source_idealization': 'native standardization; fixed-group, fixed-orbit chart encoding; no re-identification'})
    if kind == 'space':
        provenance['source_integrity'] = _standard_space_integrity(structure, result, compiled, state,
                                                                  tolerance, angle_tolerance)
    return descriptor, state, provenance


def _convert_row_once(item):
    """Worker returns explicit success/rejection; the original CSV is never modified."""
    index, row, config, source = item
    dataset = config["dataset"]
    kind = config["kind"]
    identifier = str(row.get('material_id', row.get('uid', index)))
    try:
        if (config.get('symmetry_protocol') != SYMMETRY_PROTOCOL
                or config.get('coordinate_mode', 'fractional') != 'fractional'
                or config.get('cif_fractional_tolerance', 0) != 0
                or config.get('niggli', False)):
            raise ValueError('Use the standard protocol with unsnapped fractional coordinates and no Niggli reduction')
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            structure = parse_cif(row['cif'])
        tolerance = config['symmetry_tolerance']
        if kind == 'layer':
            if 'pbc' in row and re.findall(r'True|False', str(row['pbc'])) != ['True', 'True', 'False']:
                raise ValueError('Expected periodic xy and nonperiodic normal for C2DB')
            if config.get('layer_group_policy') != 'identify' or not row.get('official_structure_sha256'):
                raise ValueError('Layer-group identification requires a verified fractional CIF')
            declared = int(row[config.get('group_column', 'lgnum')])
            descriptor, state, provenance = standard_record(structure, kind, tolerance, declared_number=declared)
        elif kind == 'space':
            descriptor, state, provenance = standard_record(structure, kind, tolerance,
                                                            config.get('angle_tolerance', 5.))
        else:
            raise ValueError('CSV preprocessing supports space and layer structures')
        # The sequence has one explicit serialization rule, independent of source atom order.
        original = compile_descriptor(descriptor)
        order = sorted(range(len(descriptor.orbits)), key=lambda i: (
            original.offsets[i+1]-original.offsets[i], descriptor.orbits[i].letter,
            descriptor.orbits[i].atomic_number, i))
        q = state[original.lattice_dof:]
        reordered_q = np.concatenate([q[original.offsets[i]:original.offsets[i+1]] for i in order])
        descriptor = Descriptor(kind, descriptor.number, tuple(descriptor.orbits[i] for i in order))
        state = np.r_[state[:original.lattice_dof], reordered_q]
        compiled = compile_descriptor(descriptor)
        if compiled.num_atoms > config.get("max_atoms", 192) or len(descriptor.orbits) > config.get("max_orbits", 32):
            raise ValueError("Canonical descriptor exceeds configured atom/orbit budget")
        properties = {}
        for name, column in config.get("property_columns", {}).items():
            value = row.get(column)
            properties[name] = float(value) if value is not None and pd.notna(value) and np.isfinite(float(value)) else None
        record = {"id": identifier, "dataset": dataset, "descriptor": descriptor.to_dict(),
                  "state": state.tolist(), "properties": properties}
        integrity = provenance.get('source_integrity')
        if integrity is None:
            integrity = require_source_integrity(structure, descriptor, state, tolerance,
                angle_tolerance=config.get('angle_tolerance', 5.) if kind == 'space' else None)
        if not config.get('_training_cache'):
            cif_hash = hashlib.sha256(row['cif'].encode()).hexdigest()
            record['provenance'] = {
                "source_csv": source, "source_row": int(index), "source_cif_sha256": cif_hash,
                **{key: row[key] for key in ('official_structure_url', 'official_structure_sha256',
                     'original_csv_sha256', 'source_verification') if key in row},
                "orbit_order": "coordinate_dof,letter,element,source_index",
                "coordinate_mode": "fractional", **provenance,
                "source_integrity": {**integrity, 'source_cif_sha256': cif_hash,
                                     'encoding_sha256': encoding_digest(record)},
            }
        return {'row': int(index), 'record': record}
    except (ValueError, RuntimeError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        return {'row': int(index), 'rejection': {'id': identifier, 'row': int(index), 'reason': str(error), 'type': type(error).__name__}}


def convert_row(item):
    """Accept one valid pass; bound rechecks of an invalid numeric native result."""
    result = _convert_row_once(item)
    failure = result.get('rejection', {})
    if (item[2].get('symmetry_protocol') != SYMMETRY_PROTOCOL
            or failure.get('reason') != 'matrix contains invalid numeric entries'):
        return result
    checks = [_convert_row_once(item), _convert_row_once(item)]
    if any('record' not in check for check in checks):
        return result
    first, second = (check['record'] for check in checks)
    if first['descriptor'] != second['descriptor'] or not np.allclose(first['state'], second['state'], atol=1e-10, rtol=0):
        return result
    if 'provenance' in second:
        second['provenance']['numerical_recheck'] = dict(attempts=3, valid_agreeing_checks=2,
            original_input=True, parameters_unchanged=True, initial_error=failure['reason'])
    return checks[1]


@lru_cache(maxsize=64)
def _image_shifts(bounds):
    return torch.tensor(list(product(*(range(-n,n+1) for n in bounds))),dtype=torch.long)


_IMAGE_DISTANCE_BUDGET = 1 << 20


def _shift_chunks(bounds, count, device):
    shifts = product(*(range(-n,n+1) for n in bounds))
    width = min(65536, max(1, _IMAGE_DISTANCE_BUDGET // max(count*count, 1)))
    while chunk := list(islice(shifts, width)):
        yield torch.tensor(chunk, dtype=torch.long, device=device)


def _large_image_neighbors(f, cell, bounds, source, target, cutoff, max_neighbors, keep_shells):
    """Exact two-pass selection without materializing the full image cube.

    First find each target's kth distance, then retain its complete boundary
    shell. Scratch space is chunked; time and retained shells still depend on
    the actual geometry. The caller disables gradients for topology selection.
    """
    count = len(f)
    best = f.new_full((count, max_neighbors), torch.inf)
    delta = f[source, None]-f[target, None]
    for shifts in _shift_chunks(bounds, count, f.device):
        distances = torch.linalg.vector_norm((delta+shifts) @ cell, dim=-1)
        valid = (distances < cutoff) & (distances > 1e-6)
        distances = distances.masked_fill(~valid, torch.inf)
        by_target = distances.reshape(count,count,-1).transpose(0,1).reshape(count,-1)
        best = torch.cat((best,by_target),dim=1).topk(max_neighbors,dim=1,largest=False).values
    boundary = best.max(dim=1).values
    pairs_all, shifts_all, distances_all = [], [], []
    for shifts in _shift_chunks(bounds, count, f.device):
        distances = torch.linalg.vector_norm((delta+shifts) @ cell, dim=-1)
        limit = boundary[target,None]+(1e-5 if keep_shells else 0.)
        pairs, images = ((distances < cutoff) & (distances > 1e-6) & (distances <= limit)).nonzero(as_tuple=True)
        pairs_all.append(pairs); shifts_all.append(shifts[images]); distances_all.append(distances[pairs,images])
    pairs, shifts, distances = torch.cat(pairs_all), torch.cat(shifts_all), torch.cat(distances_all)
    # Restore dense pair/image enumeration as the tie breaker, independently of
    # chunk boundaries: target, distance, pair index, lexicographic image.
    order = torch.arange(len(pairs),device=f.device)
    for key in (shifts[:,2],shifts[:,1],shifts[:,0],pairs,distances,target[pairs]):
        order = order[key[order].argsort(stable=True)]
    if not keep_shells:
        degree = torch.bincount(target[pairs],minlength=count)
        starts = degree.cumsum(0)-degree
        rank = torch.arange(len(order),device=f.device)-torch.repeat_interleave(starts,degree)
        order = order[rank < max_neighbors]
    return pairs[order], shifts[order]


def geometry(compiled, states, lattices):
    with torch.autocast(device_type=states[0].device.type, enabled=False):
        return _geometry(compiled, states, lattices)


def _geometry(compiled, states, lattices):
    coords, numbers, periodic = [], [], []
    for chart, z in zip(compiled, states):
        f = chart.tensor(chart.b, z) + chart.coordinate_velocity(z[chart.lattice_dof:])
        mask = chart.tensor(chart.periodic, z, torch.bool)
        coords.append(torch.where(mask, f.remainder(1), f))
        numbers.append(chart.tensor(chart.elements, z, torch.long))
        periodic.append(mask)
    counts = torch.tensor([len(f) for f in coords], device=states[0].device)
    return torch.cat(coords), torch.cat(numbers), counts, torch.stack(periodic)


def radius_graph(coords, cells, counts, periodic, cutoff=6., max_neighbors=50, max_images=16, keep_shells=False,
                 allow_large_images=False):
    """Cells use row vectors. An edge j→i includes the image of j relative to i.

    Reciprocal-vector bounds include every image in the cutoff; no fixed 27-image
    approximation, and no images along a layer's nonperiodic normal.
    Legacy callers retain the explicit image guard. Joint-flow callers can opt
    into exact chunked search when that guard would otherwise reject a state.
    """
    with torch.autocast(device_type=coords.device.type, enabled=False):
        return _radius_graph(coords, cells, counts, periodic, cutoff, max_neighbors, max_images, keep_shells,
                             allow_large_images)


def _radius_graph(coords, cells, counts, periodic, cutoff, max_neighbors, max_images, keep_shells=False,
                  allow_large_images=False):
    all_edges, all_shifts, counts_edges = [], [], []
    offset = 0
    for g, count in enumerate(counts.tolist()):
        cell, f = cells[g], coords[offset:offset+count]
        bounds = torch.ceil(cutoff*torch.linalg.vector_norm(torch.linalg.inv(cell), dim=0)).long()+1
        bounds = torch.where(periodic[g], bounds, torch.zeros_like(bounds))
        if bounds.max() > max_images:
            if not allow_large_images:
                raise ValueError('Periodic image bound exceeds max_images; increase the explicit graph limit')
            source, target = torch.meshgrid(torch.arange(count,device=f.device),
                                           torch.arange(count,device=f.device),indexing='ij')
            source, target = source.flatten(), target.flatten()
            with torch.no_grad():
                pairs, shifts = _large_image_neighbors(f,cell,tuple(bounds.tolist()),source,target,
                                                       cutoff,max_neighbors,keep_shells)
            all_edges.append(torch.stack((source[pairs]+offset,target[pairs]+offset)))
            all_shifts.append(shifts); counts_edges.append(len(pairs))
            offset += count
            continue
        shifts = _image_shifts(tuple(bounds.tolist())).to(device=f.device)
        source, target = torch.meshgrid(torch.arange(count, device=f.device),
                                        torch.arange(count, device=f.device), indexing='ij')
        source, target = source.flatten(), target.flatten()
        # Topology does not differentiate through neighbor selection.
        with torch.no_grad():
            distances = torch.linalg.vector_norm((f[source, None]-f[target, None]+shifts) @ cell, dim=-1)
            pairs, images = ((distances < cutoff) & (distances > 1e-6)).nonzero(as_tuple=True)
            # Stable two-key ordering exactly matches each target's old sort.
            order = distances[pairs,images].argsort(stable=True)
            order = order[target[pairs[order]].argsort(stable=True)]
            degree = torch.bincount(target[pairs],minlength=count)
            starts = degree.cumsum(0)-degree
            rank = torch.arange(len(order),device=f.device)-torch.repeat_interleave(starts,degree)
            if keep_shells and len(order):
                boundary_ids = (starts+degree.clamp(max=max_neighbors)-1).clamp_min(0)
                boundary = distances[pairs[order[boundary_ids]],images[order[boundary_ids]]]
                keep = distances[pairs[order],images[order]] <= boundary[target[pairs[order]]]+1e-5
            else:
                keep = rank < max_neighbors
            ids = order[keep]
            pairs, images = pairs[ids], images[ids]
        all_edges.append(torch.stack((source[pairs]+offset, target[pairs]+offset)))
        all_shifts.append(shifts[images]); counts_edges.append(len(pairs))
        offset += count
    edges, shifts = torch.cat(all_edges, dim=1), torch.cat(all_shifts)
    graph = torch.repeat_interleave(torch.arange(len(counts), device=coords.device),
                                   torch.tensor(counts_edges, device=coords.device))
    vectors = torch.einsum('ei,eij->ej', coords[edges[0]]+shifts-coords[edges[1]], cells[graph])
    return edges, shifts, torch.tensor(counts_edges, device=coords.device), vectors


def fractional_vectors(cartesian, lattices, node_graph):
    with torch.autocast(device_type=cartesian.device.type, enabled=False):
        return torch.linalg.solve(lattices[node_graph], cartesian.to(lattices.dtype).unsqueeze(-1)).squeeze(-1)


@dataclass
class JointTarget:
    state: JointState
    identity: str
    properties: dict


def target_from_record(record, dtype=torch.float32, operation_compiler=False):
    target = _target_from_record(record, dtype)
    if operation_compiler:
        try:
            return recompile_target(target)
        except ValueError as error:
            if dtype!=torch.float32 or str(error)!='Operation-derived metric does not reconstruct the target lattice':
                raise
            # Avoid a matrix-exp -> eigendecomposition round trip in FP32.
            refined=recompile_target(_target_from_record(record,torch.float64))
            refined=replace(refined,state=refined.state.to(dtype=dtype))
            if not torch.allclose(refined.state.decoded(wrap=False)['lattice'],target.state.decoded(wrap=False)['lattice'],
                                  atol=2e-5,rtol=2e-5):
                raise error
            return refined
    return target


def _target_from_record(record, dtype=torch.float32):
    """Decode verified existing records without rewriting or re-standardizing the data."""
    if 'public_state' in record:
        spec = SymmetrySpec.from_dict(record['symmetry'])
        value = record['public_state']
        moments = record.get('magnetic_moments')
        if moments is not None and record.get('moment_order') != 'compiled':
            raise ValueError('Moment labels require moment_order=compiled')
        state = JointState(spec, tuple(value['letters']), tuple(value['elements']),
                           torch.tensor(value['metric'], dtype=dtype),
                           torch.tensor(value['coordinates'], dtype=dtype).reshape(-1, 3),
                           None if moments is None else torch.tensor(moments, dtype=dtype), status='done')
        if not state.num_atoms or not all(state.elements):
            raise ValueError('Training endpoints must be nonempty and contain no MASK elements')
        if not all(torch.isfinite(v).all() for v in (state.metric, state.coordinates)):
            raise ValueError('Nonfinite explicit target')
        if moments is not None:
            if not torch.isfinite(state.moments).all() or not torch.allclose(state.decoded()['moments'], state.moments, atol=2e-5, rtol=2e-5):
                raise ValueError('Moment labels violate the supplied representation')
        return JointTarget(state, str(record.get('id', 'structure')), dict(record.get('properties', {})))
    descriptor = Descriptor.from_dict(record['descriptor'])
    old = compile_descriptor(descriptor)
    endpoint = torch.tensor(record['state'], dtype=dtype)
    if endpoint.shape != (old.dof,) or not torch.isfinite(endpoint).all():
        raise ValueError(f'Invalid endpoint: {record.get("id")}')
    spec = SymmetrySpec.from_dict(record['symmetry']) if 'symmetry' in record else catalog_spec(descriptor.kind, descriptor.number)
    letters = tuple(w.letter for w in descriptor.orbits)
    elements = tuple(w.atomic_number for w in descriptor.orbits)
    from unifiedgroupgen.group_constraints.unified import compile_constraints
    new = compile_constraints(spec, letters)
    symmetric = torch.einsum('k,kij->ij', endpoint[:old.lattice_dof], torch.tensor(old.metric_basis, dtype=dtype))
    metric = torch.einsum('kij,ij->k', torch.tensor(new.standard, dtype=dtype), symmetric)
    coordinates = old.expand(endpoint, wrap=False)['coordinates']
    moments = record.get('magnetic_moments')
    if moments is not None:
        if 'symmetry' not in record or record.get('moment_order') != 'compiled':
            raise ValueError('Moment supervision needs explicit symmetry and moment_order=compiled')
        moments = torch.tensor(moments, dtype=dtype)
        if moments.shape != coordinates.shape or not torch.isfinite(moments).all():
            raise ValueError('Invalid or unaligned moment labels')
    state = JointState(spec, letters, elements, metric, coordinates, moments, status='done')
    decoded = state.decoded(wrap=False)
    if not torch.allclose(decoded['coordinates'], coordinates, atol=2e-5, rtol=2e-5):
        raise ValueError('Record does not lie on the declared coordinate chart')
    if moments is not None and not torch.allclose(decoded['moments'], moments, atol=2e-5, rtol=2e-5):
        raise ValueError('Moment labels violate their declared symmetry')
    return JointTarget(state, str(record.get('id', 'structure')), dict(record.get('properties', {})))


def match_orbit(spec, coordinates, tolerance=2e-5):
    """Find a computed chart and ordering; accept only a complete set match."""
    x = np.asarray(coordinates, float).copy()
    periodic = np.asarray(spec.periodic)
    x[:, periodic] %= 1
    shifts = np.array([np.pad(n, (0, 3-spec.rank)) for n in product((-1, 0, 1), repeat=spec.rank)])
    for template in spec.templates:
        if template.multiplicity != len(x):
            continue
        a, b = template.arrays()
        inverse = np.linalg.pinv(a[0])
        # One atom determines the at-most-three free coordinates of its orbit.
        for point in x:
            candidates = point+shifts-b[0]
            qs = candidates@inverse.T
            errors = np.linalg.norm(qs@a[0].T-candidates, axis=1)
            for q in qs[errors < tolerance]:
                candidate = b+np.einsum('niq,q->ni', a, q)
                delta = candidate[:, None]-x[None]
                delta[..., periodic] -= np.round(delta[..., periodic])
                cost = np.linalg.norm(delta, axis=-1)
                rows, columns = linear_sum_assignment(cost)
                if cost[rows, columns].max() < tolerance:
                    return template.letter, candidate, columns
    raise ValueError('No operation-derived orbit chart matches this endpoint without changing geometry')


def recompile_target(target):
    old = target.state
    spec = compile_operations(old.spec.operations, old.spec.kind)
    if old.spec.magnetic or old.spec.spin_generators:
        raise ValueError('Automatic endpoint conversion currently supports nonmagnetic targets')
    decoded = old.decoded(wrap=False)
    letters, coordinates, moments = [], [], []
    for section in old.compiled.slices:
        letter, x, order = match_orbit(spec, decoded['coordinates'][section].detach().cpu().numpy())
        letters.append(letter); coordinates.append(old.coordinates.new_tensor(x))
        if old.moments is not None:
            moments.append(old.moments[section][torch.as_tensor(order, device=old.moments.device)])
    compiled = compile_constraints(spec, tuple(letters))
    lattice = decoded['lattice'].detach().double().cpu().numpy()[:spec.rank, :spec.rank]
    inverse = np.linalg.inv(compiled.root)
    values, vectors = np.linalg.eigh(inverse@(lattice.T@lattice)@inverse)
    symmetric = (vectors*np.log(values))@vectors.T
    metric = np.einsum('kij,ij->k', compiled.standard, symmetric)
    state = JointState(spec, tuple(letters), old.elements, old.metric.new_tensor(metric),
                       torch.cat(coordinates), None if old.moments is None else torch.cat(moments),
                       old.step, old.budget, old.status, chemistry=old.chemistry)
    # Reject unsupported settings rather than silently distorting the target cell.
    if not torch.allclose(state.decoded()['lattice'], decoded['lattice'], atol=2e-5, rtol=2e-5):
        raise ValueError('Operation-derived metric does not reconstruct the target lattice')
    return replace(target, state=state)


PROTOCOL = 'source-cif-integrity-v1'


class _LayerMatcher(StructureMatcher):
    """Forbid auxiliary-vacuum shear/repetition during 2D lattice search."""

    def _get_lattices(self, target_lattice, structure, supercell_size=1):
        for lattice, transform in super()._get_lattices(target_lattice, structure, supercell_size):
            if (np.all(transform[:2, 2] == 0) and np.all(transform[2, :2] == 0)
                    and abs(transform[2, 2]) == 1):
                yield lattice, transform


def encoding_digest(record):
    payload = {'descriptor': record['descriptor'], 'state': record['state']}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def reduced_counts(numbers):
    counts = Counter(int(z) for z in numbers)
    divisor = reduce(gcd, counts.values())
    return {str(z): n // divisor for z, n in sorted(counts.items())}


def expanded_structure(record):
    chart = compiled_chart(Descriptor.from_dict(record['descriptor']))
    lattice, coords, _ = expand_numpy(chart, record['state'])
    if record['descriptor']['kind'] == 'space':
        return Structure(Lattice(lattice.T), chart.elements, coords)
    rows = lattice.T.copy()
    height = max(40.0, float(np.ptp(coords[:, 2])) + 20)
    rows[2] = [0, 0, height]
    f = coords.copy()
    f[:, 2] = (f[:, 2] - f[:, 2].mean()) / height + .5
    return Structure(Lattice(rows, pbc=(True, True, False)), chart.elements, f)


def _isolated_layer(structure, height):
    ab = structure.lattice.matrix[:2]
    normal = np.cross(*ab)
    normal /= np.linalg.norm(normal)
    cart = structure.cart_coords
    z = cart @ normal
    vacuum_period = abs(structure.lattice.matrix[2] @ normal)
    if vacuum_period:
        wrapped = z % vacuum_period
        ordered = np.sort(wrapped)
        gaps = np.diff(np.r_[ordered, ordered[0] + vacuum_period])
        cut = ordered[(int(np.argmax(gaps)) + 1) % len(ordered)]
        z = (wrapped - cut) % vacuum_period
    z -= z.mean()
    xy = cart @ ab.T @ np.linalg.inv(ab @ ab.T)
    rows = np.zeros((3, 3))
    rows[:2, :2] = np.linalg.cholesky(ab @ ab.T)
    rows[2, 2] = height
    return Structure(Lattice(rows), structure.species,
                     np.column_stack([xy % 1, z / height + .5]))


def compare_source(source, encoded, kind, tolerance, angle_tolerance=None):
    """Return measured equivalence under declared absolute modelling tolerance.

    A stricter numerical-identity flag is separate from tolerance-level agreement.
    Lattice matching is bounded by tol/min(periodic length); scale=False.
    In 3D, the matcher displacement is normalized by volume-per-site^(1/3).
    In 2D, we explicitly inspect matched Cartesian displacements after mapping.
    """
    source_counts = reduced_counts(source.atomic_numbers)
    encoded_counts = reduced_counts(encoded.atomic_numbers)
    report = {'protocol': PROTOCOL, 'source_composition': source_counts,
              'encoded_composition': encoded_counts,
              'composition_conserved': source_counts == encoded_counts,
              'source_atoms': len(source), 'encoded_atoms': len(encoded),
              'tolerance_angstrom': float(tolerance), 'geometry_equivalent': False,
              'numerically_identical': False, 'max_displacement_angstrom': None,
              'measure_per_atom_relative_error': None}
    if not report['composition_conserved']:
        report['failure'] = 'composition_changed'
        return report
    dim = 3 if kind == 'space' else 2
    if dim == 2:
        height = 20 * max(*source.lattice.abc[:2], *encoded.lattice.abc[:2], 4)
        source, encoded = _isolated_layer(source, height), _isolated_layer(encoded, height)
        # Two centered conventional cells of equal area need not be connected
        # by an integer basis change. Reduce their translation lattices first.
        primitive_tolerance = max(1e-6, tolerance / 10)
        source = _isolated_layer(source.get_primitive_structure(tolerance=primitive_tolerance), height)
        encoded = _isolated_layer(encoded.get_primitive_structure(tolerance=primitive_tolerance), height)
    first_measure = abs(np.linalg.det(source.lattice.matrix)) if dim == 3 else np.linalg.norm(np.cross(*source.lattice.matrix[:2]))
    second_measure = abs(np.linalg.det(encoded.lattice.matrix)) if dim == 3 else np.linalg.norm(np.cross(*encoded.lattice.matrix[:2]))
    relative = abs(second_measure / len(encoded) / (first_measure / len(source)) - 1)
    length = min(*source.lattice.abc[:dim], *encoded.lattice.abc[:dim])
    lattice_tol = max(1e-5, tolerance / length)
    report.update(measure_per_atom_relative_error=float(relative),
                  lattice_relative_tolerance=float(lattice_tol),
                  scale_normalization=False, periodic_dimension=dim)
    # A strain of one lattice_tol along each periodic axis changes the measure
    # by at most this amount. This rejects density changes even if sites match.
    if relative > (1 + lattice_tol) ** dim - 1 + 1e-8:
        report['failure'] = 'periodic_measure_changed'
        return report
    angle_tol = max(.001, np.rad2deg(lattice_tol)) if angle_tolerance is None else angle_tolerance
    report['angle_tolerance_degrees'] = float(angle_tol)
    if dim == 3:
        free_length = ((source.volume / len(source)) ** (1 / 3) +
                       (encoded.volume / len(encoded)) ** (1 / 3)) / 2
        # StructureMatcher's default primitive reduction uses 0.25 Angstrom,
        # which can distort nearly higher-symmetry cells before a strict check.
        # Reduce with our declared precision instead, then disable that default.
        primitive_tolerance = max(1e-6, tolerance / 10)
        first = source.get_primitive_structure(tolerance=primitive_tolerance)
        second = encoded.get_primitive_structure(tolerance=primitive_tolerance)
        matcher = StructureMatcher(ltol=lattice_tol, stol=max(1e-6, tolerance / free_length),
                                   angle_tol=angle_tol, scale=False, primitive_cell=False, attempt_supercell=True)
        distance = matcher.get_rms_dist(first, second)
        if distance is not None:
            report['max_displacement_angstrom'] = float(distance[1] * free_length)
            report['normalized_rms_displacement'] = float(distance[0])
    else:
        if len(source) < len(encoded):
            source, encoded = encoded, source
        matcher = _LayerMatcher(ltol=lattice_tol, stol=2, angle_tol=angle_tol,
                                   scale=False, primitive_cell=False, attempt_supercell=True)
        mapped = matcher.get_s2_like_s1(source, encoded)
        if mapped is not None and len(mapped) == len(source):
            # Disallow any exchange of a periodic axis with the artificial vacuum.
            normal_preserved = np.allclose(np.abs(mapped.lattice.matrix[2]),
                                          np.abs(source.lattice.matrix[2]), atol=1e-6)
            if normal_preserved:
                delta = mapped.frac_coords - source.frac_coords
                delta[:, :2] -= np.round(delta[:, :2])
                delta[:, 2] -= delta[:, 2].mean()
                cart_delta = delta @ source.lattice.matrix
                from pymatgen.util.coord import pbc_shortest_vectors
                # Use the auxiliary 3D lattice only for the oblique xy minimum
                # image. Restore the unwrapped z displacement explicitly.
                vec = pbc_shortest_vectors(Lattice(source.lattice.matrix),
                                           np.zeros((1, 3)), delta)[0]
                vec[:, 2] = cart_delta[:, 2]
                report['max_displacement_angstrom'] = float(np.linalg.norm(vec, axis=1).max())
    displacement = report['max_displacement_angstrom']
    report['geometry_equivalent'] = displacement is not None and displacement <= tolerance * (1 + 1e-5) + 1e-8
    report['numerically_identical'] = bool(report['geometry_equivalent'] and displacement < 1e-4 and relative < 1e-5)
    if not report['geometry_equivalent']:
        report['failure'] = 'periodic_geometry_changed'
    return report


def check_record_source(source, record, tolerance, angle_tolerance=None):
    return compare_source(source, expanded_structure(record), record['descriptor']['kind'], tolerance, angle_tolerance)


def require_source_integrity(source, descriptor, state, tolerance, angle_tolerance=None):
    record = {'descriptor': descriptor.to_dict(), 'state': np.asarray(state).tolist()}
    result = check_record_source(source, record, tolerance, angle_tolerance)
    if not result['composition_conserved'] or not result['geometry_equivalent']:
        raise ValueError('Source integrity failed: ' + result.get('failure', 'unknown'))
    return result


def crystal_graph_arrays(target, cutoff=6., max_neighbors=64, max_images=16):
    """Physical endpoint graph; noisy-state neighbors are rebuilt during training."""
    state=target.state;decoded=state.decoded()
    coordinates=decoded['coordinates'];cell=decoded['lattice'].T[None]
    count=torch.tensor([state.num_atoms],dtype=torch.long)
    pbc=torch.tensor([state.spec.periodic],dtype=torch.bool)
    edges,shifts,neighbors,vectors=radius_graph(coordinates,cell,count,pbc,cutoff,max_neighbors,max_images,
                                             keep_shells=True,allow_large_images=True)
    sizes=torch.tensor([w.multiplicity for w in state.compiled.templates])
    return dict(pos=decoded.get('cartesian_coordinates',coordinates@cell[0]),frac_coords=coordinates,
        atomic_numbers=decoded['atomic_numbers'],cell=cell,pbc=pbc,natoms=count,
        edge_index=edges,cell_offsets=shifts,edge_vectors=vectors,neighbors=neighbors,
        orbit_index=torch.repeat_interleave(torch.arange(len(sizes)),sizes),
        num_orbits=len(sizes),num_nodes=state.num_atoms)
