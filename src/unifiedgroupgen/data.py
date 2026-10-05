"""Read existing CGDiT CSV splits into canonical group/orbit/state JSONL records.

No original CSV is rewritten. Layer labels are retained only after direct group
operation checks, never replaced by a 3D vacuum-supercell space group.
"""

import itertools
import json
import re
import warnings
import hashlib
from collections import deque
import multiprocessing
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from pymatgen.core import Element, Lattice, Structure
from pymatgen.io.cif import CifParser
from pyxtal import pyxtal
import spglib
from .symmetry import Descriptor, OrbitSpec, compile_descriptor, group_table, wp_arrays


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False)+"\n")


def _number(value):
    return float(re.sub(r"\([^)]*\)$", "", str(value)))


def parse_cif(text, coordinate_mode="fractional", preserve_fractional=False):
    if coordinate_mode == "fractional":
        structure = Structure.from_str(text, fmt="cif", **({'frac_tolerance': 0} if preserve_fractional else {}))
    elif coordinate_mode == "cartesian_in_fractional_fields":
        # Explicit correction for the inspected c2db_51 CSV export, not a guess for any CIF.
        # These fields contain Cartesian lengths; fraction snapping would alter the source coordinates.
        parser = CifParser.from_str(text, frac_tolerance=0)
        block = next(iter(parser.as_dict().values()))
        lattice = Lattice.from_parameters(*[_number(block[f"_cell_length_{axis}"]) for axis in 'abc'],
                                          *[_number(block[f"_cell_angle_{axis}"]) for axis in ('alpha', 'beta', 'gamma')])
        species = block["_atom_site_type_symbol"]
        coordinates = np.array([[_number(v) for v in block[f"_atom_site_fract_{axis}"]] for axis in 'xyz']).T
        occupancies = [_number(v) for v in block.get("_atom_site_occupancy", [1]*len(species))]
        if not np.allclose(occupancies, 1):
            raise ValueError("Partial occupancy is outside the fixed-orbit model")
        structure = Structure(lattice, species, coordinates, coords_are_cartesian=True, to_unit_cell=False)
    else:
        raise ValueError(f"Unsupported coordinate mode {coordinate_mode}")
    if not structure.is_ordered:
        raise ValueError("Disordered/partial occupancy structures are not supported")
    return structure


def _delta(a, b, periodic):
    diff = np.asarray(a)-np.asarray(b)
    diff[..., periodic] -= np.round(diff[..., periodic])
    return diff


def _norm(delta, physical_lattice=None):
    return np.linalg.norm(delta if physical_lattice is None else delta@physical_lattice.T, axis=-1)


def operation_residual(coords, elements, operations, periodic, physical_lattice=None):
    residual = 0.0
    for op in operations:
        transformed = coords @ op.rotation_matrix.T + op.translation_vector
        for element in np.unique(elements):
            ids = np.flatnonzero(elements == element)
            costs = _norm(_delta(transformed[ids, None], coords[None, ids], periodic), physical_lattice)
            rows, cols = linear_sum_assignment(costs)
            residual = max(residual, float(np.max(costs[rows, cols])))
    return residual


def _q_for_point(a, b, point, periodic, tolerance, physical_lattice=None):
    images = itertools.product(*[(-1, 0, 1) if flag else (0,) for flag in periodic])
    for image in images:
        rhs = point-b+np.array(image)
        q = np.linalg.lstsq(a, rhs, rcond=None)[0]
        if _norm(a@q-rhs, physical_lattice) < tolerance:
            yield q


def decompose_orbits(kind, number, coords, elements, tolerance=1e-4, physical_lattice=None):
    """Exact-to-tolerance finite orbit matching; failures are recorded, not assigned P1."""
    periodic = np.array([True, True, kind == "space"])
    remaining = set(range(len(coords)))
    rows = sorted(group_table(kind, number), key=lambda w: (w.multiplicity, w.letter))
    orbits, parameters = [], []
    while remaining:
        i = min(remaining)
        candidates = np.array(sorted(j for j in remaining if elements[j] == elements[i]))
        found = False
        for wp in rows:
            a, b, axes = wp_arrays(kind, number, wp.letter)
            if len(a) > len(candidates):
                continue
            for j in range(len(a)):
                for q in _q_for_point(a[j], b[j], coords[i], periodic, tolerance, physical_lattice):
                    generated = b+np.einsum('iaq,q->ia', a, q)
                    if len(a) > 1:
                        pair = _norm(_delta(generated[:, None], generated[None], periodic), physical_lattice)
                        np.fill_diagonal(pair, np.inf)
                        if np.min(pair) < tolerance:
                            continue
                    costs = _norm(_delta(generated[:, None], coords[None, candidates], periodic), physical_lattice)
                    matched, columns = linear_sum_assignment(costs)
                    if len(matched) == len(a) and np.max(costs[matched, columns]) < tolerance:
                        q[np.array(periodic[axes], dtype=bool)] %= 1
                        orbits.append(OrbitSpec(wp.letter, int(elements[i])))
                        parameters.extend(q.tolist())
                        remaining.difference_update(candidates[columns].tolist())
                        found = True
                        break
                if found:
                    break
            if found:
                break
        if not found:
            raise ValueError(f"Cannot match atom {i} to a {kind} {number} orbit in this setting/origin")
    return Descriptor(kind, number, tuple(orbits)), np.array(parameters)


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
            for image in itertools.product((-1, 0, 1), repeat=2):
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
            for image in itertools.product((-1, 0, 1), repeat=2):
                b = rhs+image
                residual = np.r_[c@np.linalg.pinv(c)@b-b, 0.0]
                if _norm(residual, physical_lattice) <= tolerance:
                    values.append(b)
        right_sides.append(values)
    matrix = np.concatenate([c for _, c in selected])
    for values in itertools.product(*right_sides):
        yield (np.linalg.pinv(matrix) @ np.concatenate(values)) % 1


def _layer_record_in_setting(structure, number, tolerance=1e-4):
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
                descriptor, q = decompose_orbits("layer", number, adjusted, elements, tolerance, physical_lattice)
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


def layer_record(structure, number, tolerance=1e-4, infer_cell_orientation=False):
    """Try explicit setting conversions, retaining the input label only after closure checks.

    Correct structures use coordinate-preserving basis changes. Guessing a missing
    cell orientation is allowed only for explicitly marked malformed exports.
    """
    try:
        return _layer_record_in_setting(structure, number, tolerance)
    except ValueError as initial_error:
        centered = number in {10, 13, 18, 22, 26, 35, 36, 47, 48}
        rows = structure.lattice.matrix
        gamma = np.deg2rad(structure.lattice.gamma)
        angles = [0.0] if centered else []
        if infer_cell_orientation:
            angles += [np.pi/2, -np.pi/2, np.pi]
            if centered:
                angles += [gamma/2, -gamma/2, np.pi/2+gamma/2, np.pi/2-gamma/2]
        for angle in angles:
            rotate = np.array([[np.cos(angle), -np.sin(angle), 0],
                               [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
            candidate = Structure(Lattice(rows@rotate.T), structure.species, structure.cart_coords,
                                  coords_are_cartesian=True, to_unit_cell=False)
            if centered:
                candidate.make_supercell([[1, 1, 0], [-1, 1, 0], [0, 0, 1]])
            try:
                descriptor, state, provenance = _layer_record_in_setting(candidate, number, tolerance)
                provenance.update({'cell_orientation_inferred': bool(angle), 'source_frame_angle_radians': angle,
                                   'conventional_cell_multiplier': 2 if centered else 1,
                                   'source_atoms': len(structure), 'expanded_atoms': len(candidate)})
                return descriptor, state, provenance
            except ValueError:
                continue
        raise initial_error


def identified_layer_record(structure, tolerance=0.05, declared_number=None):
    """Identify a real layer group and encode its standardized cell; never assign a fallback group."""
    result = spglib.get_layergroup((structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers),
                                  aperiodic_dir=2, symprec=tolerance)
    if result is None:
        raise ValueError('Layer-group identification failed; no fallback or label substitution was performed')
    standardized = Structure(Lattice(result.std_lattice, pbc=(True, True, False)),
                             result.std_types, result.std_positions)
    chart_idealized = False
    try:
        descriptor, state, provenance = _layer_record_in_setting(standardized, int(result.number), 1e-5)
    except ValueError:
        # Native identification can return standard positions that are still noisy.
        # Fit the same identified group within the configured physical tolerance.
        descriptor, state, provenance = _layer_record_in_setting(standardized, int(result.number), tolerance)
        compiled = compile_descriptor(descriptor)
        encoded = compiled.b + np.einsum('ijq,q->ij', compiled.coordinate_basis, state[compiled.lattice_dof:])
        compiled.permutations(encoded, tolerance=1e-5)
        chart_idealized = True
    from pymatgen.core.operations import SymmOp
    operations = [SymmOp.from_rotation_and_translation(r, t)
                  for r, t in zip(result.rotations, result.translations)]
    residual = operation_residual(structure.frac_coords, np.array(structure.atomic_numbers), operations,
                                  np.array([True, True, False]), structure.lattice.matrix.T)
    # Separate coordinate idealization at a fixed standard cell from lattice idealization.
    transformed = structure.frac_coords @ result.transformation_matrix.T + result.origin_shift
    distances = []
    for element in np.unique(structure.atomic_numbers):
        old = transformed[np.array(structure.atomic_numbers) == element]
        new = result.std_positions[result.std_types == element]
        delta = _delta(old[:, None], new[None], np.array([True, True, False]))
        costs = np.linalg.norm(delta @ result.std_lattice, axis=-1)
        distances.extend(costs.min(axis=1).tolist())
    before = np.linalg.inv(result.transformation_matrix).T @ structure.lattice.matrix
    raw_metric, ideal_metric = (before@before.T)[:2, :2], (result.std_lattice@result.std_lattice.T)[:2, :2]
    provenance.update({'cell': 'spglib_layer_conventional', 'source_atoms': len(structure),
                       'expanded_atoms': len(standardized), 'declared_layer_group': declared_number,
                       'identified_layer_group': int(result.number), 'layer_group_policy': 'identify',
                       'symmetry_tolerance_angstrom': tolerance, 'source_operation_residual_angstrom': residual,
                       'std_position_rms_angstrom': float(np.sqrt(np.mean(np.square(distances)))),
                       'std_position_max_angstrom': float(max(distances)),
                       'native_std_chart_idealized': chart_idealized,
                       'std_position_total_max_upper_bound_angstrom': float(max(distances))+provenance['chart_fit_max_angstrom'],
                       'std_metric_relative_change': float(np.linalg.norm(raw_metric-ideal_metric)/np.linalg.norm(raw_metric)),
                       'displacement_metric': 'Nearest same-species standard position at fixed ideal cell; excludes lattice strain',
                       'source_idealization': 'spglib layer standardization and bounded orbit fit; not raw-coordinate identity',
                       'transformation_matrix': result.transformation_matrix.tolist(),
                       'origin_shift': result.origin_shift.tolist(),
                       'std_rotation_matrix': result.std_rotation_matrix.tolist(), 'cell_orientation_inferred': False})
    return descriptor, state, provenance


def space_record(structure, tolerance=1e-3):
    crystal = pyxtal()
    crystal.from_seed(structure, tol=tolerance, style="pyxtal")
    number = crystal.group.number
    orbits, q_values = [], []
    for site in crystal.atom_sites:
        letter = site.wp.letter
        a, b, axes = wp_arrays("space", number, letter)
        # Explicitly reject mismatched Hall settings; a letter alone does not identify an affine chart.
        reference = next(w for w in group_table("space", number) if w.letter == letter)
        if not np.allclose(np.stack([op.affine_matrix for op in site.wp.ops]), np.stack([op.affine_matrix for op in reference.ops]), atol=1e-7):
            raise ValueError("Noncanonical Hall setting requires explicit conversion before training")
        q = np.linalg.lstsq(a[0], np.asarray(site.position)-b[0], rcond=None)[0]
        if np.linalg.norm(a[0]@q+b[0]-site.position) > 1e-5:
            raise ValueError("Cannot encode a representative in its canonical affine chart")
        q %= 1
        orbits.append(OrbitSpec(letter, Element(site.specie).Z))
        q_values.extend(q.tolist())
    descriptor = Descriptor("space", number, tuple(orbits))
    compiled = compile_descriptor(descriptor)
    rows = crystal.lattice.matrix
    k = compiled.encode_metric(rows@rows.T, tolerance=max(tolerance*10, 1e-4))
    return descriptor, np.r_[k, q_values], {"cell": "pyxtal_conventional", "source_atoms": len(structure),
        "expanded_atoms": compiled.num_atoms, "symmetry_tolerance_angstrom": tolerance,
        "source_idealization": "PyXtal/pymatgen symmetry standardization; not raw-coordinate identity"}


def convert_row(item):
    """Worker returns explicit success/rejection; the original CSV is never modified."""
    index, row, config, source = item
    dataset = config["dataset"]
    kind = config["kind"]
    coordinate_mode = config.get("coordinate_mode", "fractional")
    identifier = str(row.get("material_id", row.get("uid", index)))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            preserve_fractional = config['kind'] == 'layer' and config.get('layer_group_policy') == 'identify'
            structure = parse_cif(row["cif"], coordinate_mode, preserve_fractional=preserve_fractional)
        tolerance = config.get("symmetry_tolerance", 1e-3)
        if kind == "layer":
            if "pbc" in row and re.findall(r"True|False", str(row["pbc"])) != ["True", "True", "False"]:
                raise ValueError("Expected periodic xy and nonperiodic normal for C2DB")
            if "thickness" in row and pd.notna(row["thickness"]):
                normal = np.cross(structure.lattice.matrix[0], structure.lattice.matrix[1])
                normal /= np.linalg.norm(normal)
                raw_height = structure.cart_coords@normal
                if coordinate_mode == "cartesian_in_fractional_fields" and abs(np.ptp(raw_height)-float(row["thickness"])) > 0.05:
                    raise ValueError("Corrected C2DB heights disagree with recorded thickness")
            policy = config.get('layer_group_policy', 'declared')
            declared = int(row[config.get("group_column", "lgnum")])
            if policy == 'identify':
                if coordinate_mode != 'fractional' or not row.get('official_structure_sha256'):
                    raise ValueError('Layer-group re-identification requires a verified recovered structure')
                descriptor, state, provenance = identified_layer_record(structure, tolerance, declared)
            elif policy == 'declared':
                descriptor, state, provenance = layer_record(structure, declared, tolerance,
                                                            infer_cell_orientation=coordinate_mode == 'cartesian_in_fractional_fields')
            else:
                raise ValueError(f'Unknown layer group policy: {policy}')
        elif kind == "space":
            descriptor, state, provenance = space_record(structure, tolerance)
        else:
            raise ValueError("CSV preprocessing supports space and layer datasets; plane charts use explicit descriptors")
        if coordinate_mode == 'cartesian_in_fractional_fields' or preserve_fractional:
            provenance['cif_fractional_rounding_disabled'] = True
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
                  "state": state.tolist(), "properties": properties,
                  "provenance": {"source_csv": source, "source_row": int(index),
                                 "source_cif_sha256": hashlib.sha256(row['cif'].encode()).hexdigest(),
                                 **{key: row[key] for key in ('official_structure_url', 'official_structure_sha256',
                                      'original_csv_sha256', 'source_verification') if key in row},
                                 "orbit_order": "coordinate_dof,letter,element,source_index",
                                 "coordinate_mode": coordinate_mode, **provenance}}
        return {'row': int(index), 'record': record}
    except (ValueError, RuntimeError, TypeError, KeyError, np.linalg.LinAlgError) as error:
        return {'row': int(index), 'rejection': {'id': identifier, 'row': int(index), 'reason': str(error), 'type': type(error).__name__}}


def _prepare_worker():
    import torch
    torch.set_num_threads(1)


def _parallel_rows(items, workers, timeout):
    """Bound queued work and fail visibly if a native-library worker stops responding."""
    pool = multiprocessing.get_context('spawn').Pool(workers, initializer=_prepare_worker)
    pending = deque()
    source = iter(items)
    try:
        for _ in range(workers*2):
            item = next(source, None)
            if item is not None:
                pending.append((item[0], pool.apply_async(convert_row, (item,))))
        while pending:
            index, result = pending.popleft()
            try:
                yield result.get(timeout=timeout)
            except multiprocessing.TimeoutError as error:
                raise RuntimeError(f'Preparation worker timed out at source row {index}; journal is preserved. '
                                   'Inspect that row and resume with fewer workers.') from error
            item = next(source, None)
            if item is not None:
                pending.append((item[0], pool.apply_async(convert_row, (item,))))
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()


def _read_preparation_journal(path):
    """Recover only a torn final write; complete corrupt records still raise."""
    records, valid_bytes = [], 0
    with path.open('rb') as stream:
        for line in stream:
            try:
                records.append(json.loads(line))
                valid_bytes += len(line)
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line.endswith(b'\n') or stream.read(1):
                    raise
                Path(str(path)+'.incomplete_tail').write_bytes(line)
                with path.open('r+b') as recovery:
                    recovery.truncate(valid_bytes)
                break
    return records


def prepare_csv(csv_path, output, config, limit=None, workers=0, resume=False):
    from .training import file_hash
    frame = pd.read_csv(csv_path, nrows=limit)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    journal = Path(str(output)+'.attempts.jsonl')
    manifest_path = Path(str(output)+'.manifest.json')
    manifest = {'source_sha256': file_hash(csv_path), 'rows_requested': len(frame),
                'conversion_config': {key: config.get(key) for key in ('dataset', 'kind', 'coordinate_mode',
                    'symmetry_tolerance', 'group_column', 'property_columns', 'max_atoms', 'max_orbits',
                    *(['layer_group_policy'] if config['kind'] == 'layer' else []))},
                'converter_version': 7 if config['kind'] == 'layer' else 2}
    if resume:
        if not manifest_path.exists() or json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
            raise ValueError('Preparation resume source/configuration differs from the manifest')
        attempted = _read_preparation_journal(journal)
    else:
        if output.exists() or journal.exists():
            raise ValueError('Preparation output exists; use resume or a new output path')
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        attempted = []
    if [r['row'] for r in attempted] != list(range(len(attempted))):
        raise ValueError('Preparation journal is not a contiguous source-row prefix')
    items = [(int(index), row.to_dict(), config, str(Path(csv_path).resolve()))
             for index, row in frame.iloc[len(attempted):].iterrows()]
    results = _parallel_rows(items, workers, config.get('preparation_row_timeout', 120)) if workers else map(convert_row, items)
    with journal.open('a', encoding='utf-8') as stream:
        for result in results:
            attempted.append(result)
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False)+'\n')
            stream.flush()
            if len(attempted) % 100 == 0:
                print(json.dumps({'event': 'prepare_progress', 'attempted': len(attempted), 'total': len(frame)}), flush=True)
    records = [r['record'] for r in attempted if 'record' in r]
    rejected = [r['rejection'] for r in attempted if 'rejection' in r]
    write_jsonl(output, records)
    write_jsonl(str(output)+".rejected.jsonl", rejected)
    report = {"dataset": config['dataset'], "input": len(frame), "accepted": len(records), "rejected": len(rejected),
              "coordinate_mode": config.get('coordinate_mode', 'fractional'), "source_sha256": manifest['source_sha256'],
              "full_source": limit is None, "output": str(output.resolve())}
    Path(str(output)+".report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report
