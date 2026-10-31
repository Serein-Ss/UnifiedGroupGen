"""Geometry, independent group identification and declared-operation residuals."""
import json
import subprocess
import sys
import numpy as np
from scipy.optimize import linear_sum_assignment
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.standard import geometry_metrics, layer_structure
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.io import counts_of, formula_key
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.protocols import PROFILES, match_pair


def geometry(structure, kind):
    if not structure.is_ordered or not len(structure):
        raise ValueError('Evaluation requires a nonempty fully occupied ordered structure')
    if not np.isfinite(structure.lattice.matrix).all() or not np.isfinite(structure.frac_coords).all():
        raise ValueError('Nonfinite structure')
    if abs(np.linalg.det(structure.lattice.matrix)) < 1e-10:
        raise ValueError('Degenerate lattice')
    result = geometry_metrics(structure, kind)
    prepared = layer_structure(structure) if kind=='layer' else structure
    area = float(np.linalg.norm(np.cross(*prepared.lattice.matrix[:2])))
    result['atomic_density_angstrom3'] = len(structure)/structure.volume if kind=='space' else None
    result['areal_atomic_density_angstrom2'] = len(structure)/area if kind=='layer' else None
    # atomic mass unit per angstrom^2 -> microgram/cm^2
    result['areal_density_ug_cm2'] = float(structure.composition.weight)*.016605390666/area if kind=='layer' else None
    result['coordination_number'] = None
    result['rank2_atom_fraction'] = None
    try:
        from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
        radii = [CovalentRadius.radius.get(site.specie.symbol) for site in prepared]
        if all(v is not None for v in radii):
            centers, targets, images, distances = prepared.get_neighbor_list(2.4*max(radii))
            selected = distances <= 1.2*(np.array(radii)[centers]+np.array(radii)[targets])
            result['coordination_number'] = float(selected.sum()/len(prepared))
            # Rank is evaluated per connected component, not just the maximum rank.
            edges = [[] for _ in prepared]
            for i, j, shift in zip(centers[selected], targets[selected], images[selected]):
                edges[i].append((j, shift)); edges[j].append((i, -shift))
            visited, rank2 = set(), 0
            for initial in range(len(prepared)):
                if initial in visited:
                    continue
                offsets, pending, cycles = {initial: np.zeros(3)}, [initial], []
                while pending:
                    i = pending.pop(); visited.add(i)
                    for j, shift in edges[i]:
                        delta = offsets[i]+shift
                        if j in offsets:
                            cycles.append(delta-offsets[j])
                        else:
                            offsets[j] = delta; pending.append(j)
                rank = int(np.linalg.matrix_rank(cycles, tol=1e-7)) if cycles else 0
                rank2 += len(offsets) if rank==2 else 0
            result['rank2_atom_fraction'] = rank2/len(prepared)
    except (KeyError, ValueError, RuntimeError) as error:
        result['connectivity_error'] = str(error)
    return result


def identify(structure, kind, tolerances):
    cell = [structure.lattice.matrix.tolist(), structure.frac_coords.tolist(), list(structure.atomic_numbers)]
    # One fresh native process handles all tolerances. No fork after CUDA/OpenMP.
    worker = """import json,sys,spglib
r=json.load(sys.stdin); out={}
for t in r['tolerances']:
 try:
  d=(spglib.get_symmetry_dataset(tuple(r['cell']),symprec=t) if r['kind']=='space' else spglib.get_layergroup(tuple(r['cell']),aperiodic_dir=2,symprec=t))
  out[str(t)]={'number':int(d.number) if d is not None else None,'error':None if d is not None else 'No group'}
 except (ValueError,RuntimeError,AttributeError) as e: out[str(t)]={'number':None,'error':str(e)}
print(json.dumps(out))"""
    try:
        process = subprocess.run([sys.executable, '-c', worker], input=json.dumps({'cell': cell, 'kind': kind, 'tolerances': tolerances}),
            text=True, capture_output=True, timeout=30,
            creationflags=subprocess.CREATE_NO_WINDOW if sys.platform=='win32' else 0)
        if process.returncode:
            raise ValueError(f'Native identifier exited {process.returncode}')
        return json.loads(process.stdout)
    except (ValueError, subprocess.TimeoutExpired) as error:
        return {str(t): {'number': None, 'error': str(error)} for t in tolerances}


def operation_check(structure, kind, number, operations=None):
    """Fractional standard SG setting; centered mixed (fractional x,y; Cartesian z) LG setting."""
    if operations is None:
        from unifiedgroupgen.group_constraints.compiler import group_table
        operations = [{'R': op.rotation_matrix.tolist(), 't': op.translation_vector.tolist()}
                      for op in group_table(kind, number)[0].ops]
    if kind=='layer':
        structure = layer_structure(structure)
        cell = structure.lattice.matrix.copy(); cell[2] = [0, 0, 1]
        coords = structure.frac_coords.copy(); coords[:, 2] = structure.cart_coords[:, 2]-structure.cart_coords[:, 2].mean()
        from pymatgen.core import Lattice
        lattice = Lattice(cell, pbc=(True, True, False))
    else:
        lattice, coords = structure.lattice, structure.frac_coords
    numbers = np.array(structure.atomic_numbers)
    metric = lattice.matrix @ lattice.matrix.T
    residual, metric_error = 0., 0.
    for op in operations:
        rotation, translation = np.array(op['R']), np.array(op['t'])
        mapped = coords @ rotation.T+translation
        metric_error = max(metric_error, float(np.linalg.norm(rotation.T @ metric @ rotation-metric)/np.linalg.norm(metric)))
        for z in set(numbers):
            indices = np.flatnonzero(numbers==z)
            distances = lattice.get_all_distances(mapped[indices], coords[indices])
            i, j = linear_sum_assignment(distances)
            residual = max(residual, float(distances[i, j].max()))
    return residual, metric_error


def descriptor_metadata(record, kind):
    result = {}
    descriptor = record.get('descriptor')
    if descriptor:
        result['declared_group'] = descriptor['number']
        result['template'] = json.dumps([kind, descriptor['number'], sorted(o['letter'] for o in descriptor['orbits'])])
        try:
            from unifiedgroupgen.group_constraints.compiler import wp_arrays
            dimensions = [wp_arrays(kind, descriptor['number'], o['letter'])[0].shape[-1] for o in descriptor['orbits']]
            result.update(coordinate_dof=sum(dimensions), wyckoff_dimension_signature=','.join(map(str, sorted(dimensions))))
        except (ValueError, KeyError):
            pass
    return result


def check_declared_orbits(row, record, structure, samples, counts):
    if record.get('descriptor'):
        from unifiedgroupgen.tests.legacy.evaluation_before_tasks.io import record_counts
        declared=record_counts(record,samples.parent)
        row['descriptor_composition_satisfied']=counts==declared
        if record.get('state') is not None:
            from unifiedgroupgen.tests.representation import compiled_chart, expand_numpy
            from unifiedgroupgen.group_constraints.compiler import Descriptor
            chart=compiled_chart(Descriptor.from_dict(record['descriptor']))
            ll,ff,_=expand_numpy(chart,record['state'])
            from pymatgen.core import Lattice
            distances=Lattice(ll.T,pbc=tuple(chart.periodic)).get_all_distances(ff,ff)
            np.fill_diagonal(distances,np.inf)
            same=chart.atom_orbit[:,None]==chart.atom_orbit[None,:]
            row['orbit_multiplicity_regular']=bool(np.all(distances[same]>1e-4))
            row['no_coincident_orbits']=bool(np.all(distances[~same]>1e-4))
            row['orbit_check_source']='independently expanded declared state; exported CIF composition checked separately'


def check_requested_chemistry(row, record, request, refs, task, counts):
    expected = request.get('composition') or record.get('chemical_request', {}).get('composition')
    reference_record = refs.by_id.get(row['case_id']) if row['case_id'] is not None else None
    if task=='csp' and reference_record and expected is None:
        expected = counts_of(refs.structure(row['case_id']))
    if expected:
        expected = {int(z):int(n) for z,n in expected.items()}
        row['composition_satisfied'] = counts==expected
        row['formula_satisfied'] = formula_key(counts)==formula_key(expected)
        row['atom_count_satisfied'] = sum(counts.values())==sum(expected.values())
    elif task=='csp':
        raise ValueError('CSP needs requested counts or a named reference structure')
    allowed = request.get('allowed_elements', record.get('chemical_request', {}).get('allowed'))
    required = request.get('required_elements', record.get('chemical_request', {}).get('required'))
    if allowed is not None or required:
        row['element_set_satisfied'] = (allowed is None or set(counts)<=set(map(int,allowed))) and set(map(int,required or []))<=set(counts)
    return reference_record


def check_symmetry(row, record, request, structure, kind, reidentify):
    target_group = row['target_group']
    row['target_group'] = target_group
    working = layer_structure(structure) if kind=='layer' else structure
    if reidentify:
        identified = identify(working, kind, [1e-4, 1e-3, 1e-2])
        row['recovered_group'] = identified['0.001']['number']
        row['identifier_error'] = identified['0.001'].get('error')
        for tolerance, value in identified.items():
            row['exact_group_'+tolerance] = value['number']==target_group if value['number'] is not None and target_group is not None else None
        row['exact_target_group'] = row.get('exact_group_0.001')
    if target_group is not None and (record.get('descriptor') or request.get('operations')):
        residual, metric = operation_check(structure, kind, target_group, request.get('operations'))
        row.update(operation_residual_angstrom=residual, metric_relative_residual=metric,
            target_operation_containment=bool(residual<=1e-4 and metric<=1e-5),
            operation_check_frame='standard fractional SG / centered mixed-coordinate LG')


def check_reconstruction(row, reference_record, refs, structure, kind, task):
    if task=='csp' and reference_record:
        target = refs.structure(row['case_id'])
        row['reference_coordinate_dof'] = descriptor_metadata(reference_record[0], kind).get('coordinate_dof')
        row['reference_group_agreement'] = row.get('recovered_group')==reference_record[0].get('descriptor', {}).get('number') if row.get('recovered_group') is not None and reference_record[0].get('descriptor') else None
        for profile in PROFILES:
            try:
                match = match_pair(structure, target, kind, profile)
                for key, value in match.items():
                    row[key+'_'+profile] = value
            except (ValueError, TypeError, RuntimeError, np.linalg.LinAlgError) as error:
                row['matching_error_'+profile] = str(error)
