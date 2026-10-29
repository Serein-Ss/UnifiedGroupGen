"""Raw-structure evaluation shared by CSP and de novo generation (no calculators)."""
from collections import defaultdict
import csv
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import time
import warnings

import numpy as np
from scipy.optimize import linear_sum_assignment

from .evaluation_io import ReferenceBank, counts_of, formula_key, load_structure, read_records, sha256, write_json
from .evaluation_properties import evaluate_properties
from .evaluation_protocols import PROFILES, PROFILE_LABELS, match_pair
from .evaluation_statistics import numeric, population_metrics, rate, frechet_distance, logical_and
from .standard_metrics import composition_validity, fingerprints, geometry_metrics, layer_structure, structure_match


PROTOCOL = 'unified_raw_v2'
BOOL_FIELDS = ('generation_success', 'parse_success', 'finite', 'lattice_valid', 'legacy_structural_valid',
    'periodic_distance_valid', 'radius_distance_valid', 'chemical_valid', 'basic_valid', 'fingerprint_success',
    'legacy_joint_valid', 'composition_satisfied', 'formula_satisfied', 'atom_count_satisfied',
    'element_set_satisfied', 'descriptor_composition_satisfied', 'target_operation_containment', 'exact_target_group',
    'orbit_multiplicity_regular', 'no_coincident_orbits', 'observation_satisfied')


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
        from .symmetry import group_table
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
            from .symmetry import wp_arrays
            dimensions = [wp_arrays(kind, descriptor['number'], o['letter'])[0].shape[-1] for o in descriptor['orbits']]
            result.update(coordinate_dof=sum(dimensions), wyckoff_dimension_signature=','.join(map(str, sorted(dimensions))))
        except (ValueError, KeyError):
            pass
    return result


def reference_metrics(bank, kind, use_fingerprints, cache):
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pymatgen', 'matminer', 'smact', 'pyxtal', 'spglib')}
    identity = {'files': bank.identity(), 'kind': kind, 'fingerprints': use_fingerprints, 'versions': versions,
        'code': {Path(p).name: sha256(p) for p in (__file__, *[Path(__file__).with_name(n) for n in
            ('standard_metrics.py','symmetry.py','representation_audit.py','evaluation_io.py')])}}
    key = __import__('hashlib').sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache)/(key+'.json')
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['identity'] != identity:
            raise ValueError('Reference cache identity mismatch')
        return saved
    rows, features = [], []
    for index, (record, base) in enumerate(bank.rows):
        row = {'id': str(record.get('id', index)), **descriptor_metadata(record, kind)}
        row['recovered_group'] = row.get('declared_group')  # Audited reference label, explicitly identified in protocol.
        try:
            structure = load_structure(record, base)
            row.update(geometry(structure, kind))
            row['elements'] = sorted(counts_of(structure))
            row['chemical_valid'], row['chemical_error'] = composition_validity(structure)
            row['basic_valid'] = logical_and(row['periodic_distance_valid'], row['chemical_valid'])
            if use_fingerprints:
                pair = fingerprints(structure, kind)
                features.append([p.tolist() for p in pair])
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, ZeroDivisionError, np.linalg.LinAlgError) as error:
            row['evaluation_error'] = str(error)
        rows.append(row)
        if (index+1)%100==0:
            print(json.dumps({'phase': 'reference', 'completed': index+1, 'requested': len(bank.rows)}), flush=True)
    saved = {'identity': identity, 'rows': rows, 'fingerprints': features}
    write_json(path, saved)
    return saved


def csp_summary(rows, request):
    cases = defaultdict(list)
    for row in rows:
        if row.get('case_id') is None:
            raise ValueError('CSP requires case_id or reference_id for every attempt, including failures')
        cases[row['case_id']].append(row)
    expected = request.get('case_ids', list(cases))
    if set(expected) != set(cases):
        raise ValueError('CSP case ledger does not cover exactly the requested cases')
    minimum = min(map(len, cases.values())) if cases else 0
    if request.get('attempts_per_case') and any(len(v)!=request['attempts_per_case'] for v in cases.values()):
        raise ValueError('Incomplete CSP candidate budget; failed attempts must remain in the ledger')
    budgets = [k for k in sorted(set([1, 5, 20, minimum])) if k<=minimum and k>0]
    output = {'cases': len(cases), 'minimum_attempts_per_case': minimum, 'profiles': {}, 'case_rows': [],
              'sampling_protocol':'For each test case, evaluate the FIRST 1, 5, 20 candidates from one 20-candidate sequence; failures retain their slots',
              'matching_profile_labels':PROFILE_LABELS}
    for profile in PROFILES:
        result = {}
        for k in budgets:
            group = [sorted(v, key=lambda r:r['index'])[:k] for v in cases.values()]
            result[str(k)] = {'matched_cases': sum(any(r.get('match_'+profile) is True for r in v) for v in group),
                'match_rate': sum(any(r.get('match_'+profile) is True for r in v) for v in group)/len(group) if group else None,
                'valid_match_rate': sum(any(r.get('match_'+profile) is True and r.get('basic_valid') for r in v) for v in group)/len(group) if group else None,
                'match_and_property_hit_rate': sum(any(r.get('match_'+profile) is True and r.get('valid_joint_property_hit') for r in v) for v in group)/len(group)
                    if group and any(r.get('joint_property_hit') is not None for r in rows) else None}
            for error in ('rmse_normalized','rmsd_angstrom'):
                best=[min((r[error+'_'+profile] for r in case if r.get('match_'+profile) and r.get(error+'_'+profile) is not None),default=None) for case in group]
                result[str(k)]['best_'+error+'_on_matched_cases']=numeric(best)
        matched = [r for r in rows if r.get('match_'+profile)]
        result['matched_error'] = {name: numeric(r.get(name+'_'+profile) for r in matched) for name in
            ('rmse_normalized', 'rmsd_angstrom', 'length_relative_error', 'angle_mae_degrees', 'measure_relative_error')}
        output['profiles'][profile] = result
    for key, group in cases.items():
        ordered=sorted(group,key=lambda r:r['index'])
        output['case_rows'].append({'case_id': key, 'candidate_count': len(group),
            'basic_valid': sum(bool(r.get('basic_valid')) for r in group),
            'unique_valid_candidates': sum(bool(r.get('unique')) for r in group),
            **{p+'_match_at_'+str(k):any(r.get('match_'+p) is True for r in ordered[:k]) for p in PROFILES for k in budgets},
            **{p+'_matches': sum(bool(r.get('match_'+p)) for r in group) for p in PROFILES}})
    output['coordinate_dof_strata'] = {}
    for name, predicate in [('zero', lambda q:q==0), ('1-3', lambda q:0<q<=3), ('4-9',lambda q:3<q<=9), ('10+',lambda q:q>9)]:
        group = [v for v in cases.values() if v[0].get('reference_coordinate_dof') is not None and predicate(v[0]['reference_coordinate_dof'])]
        output['coordinate_dof_strata'][name] = {'cases':len(group), 'match_at_1':rate(v[0].get('match_legacy_ugg_v1') for v in group)}
    return output


def evaluate(samples, task, kind, output, reference_files=(), novelty_files=(), dataset=None, request=None,
             assessments=None, assessment_metadata=None, use_fingerprints=True, reidentify=True,
             cache=None, expected_attempts=None, embeddings=None, reference_assessments=None):
    if task not in ('csp', 'dng') or kind not in ('space', 'layer'):
        raise ValueError('task=csp/dng and kind=space/layer are required')
    samples, output = Path(samples).resolve(), Path(output).resolve()
    request = request or {}
    records = read_records(samples)
    if not records or len({r['index'] for r in records}) != len(records):
        raise ValueError('A nonempty ledger with unique attempt indices is required')
    if expected_attempts is not None and len(records)!=expected_attempts:
        raise ValueError('Attempt count differs from the requested budget')
    refs, train = ReferenceBank(reference_files, dataset), ReferenceBank(novelty_files, dataset)
    predictions, metadata = assessments or {}, assessment_metadata or {}
    if predictions and (not metadata.get('evaluator') or metadata.get('samples_sha256')!=sha256(samples)):
        raise ValueError('Independent assessments require evaluator identity and matching samples_sha256')
    if reference_assessments and metadata.get('reference_sha256')!=refs.identity():
        raise ValueError('Reference predictions do not bind the exact reference files')
    reference = reference_metrics(refs, kind, use_fingerprints, cache or output/'reference_cache') if refs.rows else {'rows': [], 'fingerprints': []}
    if task=='csp' and any(str(r.get('case_id', r.get('reference_id'))) not in refs.by_id for r in records):
        raise ValueError('Each CSP attempt must name an existing reference case, including failures')
    rows, features, unique = [], [], defaultdict(list)
    tolerances = request.get('property_tolerances', {'formation_energy': .1, 'band_gap': .2, 'energy_above_hull': .05})
    started = time.perf_counter()
    for record in records:
        index = record['index']
        row = {'index': index, 'case_id': str(record.get('case_id', record.get('reference_id'))) if record.get('case_id', record.get('reference_id')) is not None else None,
               'generation_success': record.get('status', 'generated')=='generated', 'parse_success': None,
               'task': task, 'dataset': dataset, 'kind': kind, 'basic_valid': None,
               'generation_seconds': record.get('wall_seconds'), **descriptor_metadata(record, kind)}
        reference_record = refs.by_id.get(row['case_id']) if row['case_id'] is not None else None
        if reference_record:
            row['reference_coordinate_dof'] = descriptor_metadata(reference_record[0], kind).get('coordinate_dof')
        row['target_group'] = request.get('group') or record.get('requested_group') or row.get('declared_group')
        row['property_targets'] = request.get('properties', {k:v for k,v in record.get('properties_requested', {}).items() if k!='source_domain' and v is not None})
        validation = record.get('validation', {})
        for name in ('orbit_multiplicity_regular','no_coincident_orbits'):
            row[name] = validation.get(name)
        if 'observation_max_residual' in validation:
            row['observation_max_residual'] = validation['observation_max_residual']
            row['observation_satisfied'] = validation['observation_max_residual']<=request.get('observation_tolerance', 1e-5)
        if not row['generation_success']:
            row.update(basic_valid=False,legacy_joint_valid=False,unique=False)
            if task=='csp': row.update(composition_satisfied=False,formula_satisfied=False,atom_count_satisfied=False)
            if row['target_group'] is not None: row.update(exact_target_group=False,target_operation_containment=False)
            if novelty_files: row['novel_against_reference']=False
            row['failure_reason'] = record.get('reason'); rows.append(row); continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                structure = load_structure(record, samples.parent)
            row['parse_success'] = True
            row['finite'] = bool(np.isfinite(structure.lattice.matrix).all() and np.isfinite(structure.frac_coords).all())
            row['lattice_valid'] = bool(row['finite'] and abs(np.linalg.det(structure.lattice.matrix))>1e-10)
            if not row['lattice_valid']:
                row.update(generation_success=False,basic_valid=False,failure_stage='numeric_validation')
            row.update(geometry(structure, kind))
            row['chemical_valid'], row['chemical_error'] = composition_validity(structure)
            row['basic_valid'] = logical_and(row['periodic_distance_valid'], row['chemical_valid'])
            row['legacy_joint_valid'] = (bool(row['legacy_structural_valid'] and row['chemical_valid']) if row['chemical_valid'] is not None else None)
            counts = counts_of(structure)
            row['elements'] = sorted(counts)
            if record.get('descriptor'):
                from .evaluation_io import record_counts
                declared=record_counts(record,samples.parent)
                row['descriptor_composition_satisfied']=counts==declared
                if record.get('state') is not None:
                    from .representation_audit import compiled_chart, expand_numpy
                    from .symmetry import Descriptor
                    chart=compiled_chart(Descriptor.from_dict(record['descriptor']))
                    ll,ff,_=expand_numpy(chart,record['state'])
                    from pymatgen.core import Lattice
                    distances=Lattice(ll.T,pbc=tuple(chart.periodic)).get_all_distances(ff,ff)
                    np.fill_diagonal(distances,np.inf)
                    same=chart.atom_orbit[:,None]==chart.atom_orbit[None,:]
                    row['orbit_multiplicity_regular']=bool(np.all(distances[same]>1e-4))
                    row['no_coincident_orbits']=bool(np.all(distances[~same]>1e-4))
                    row['orbit_check_source']='independently expanded declared state; exported CIF composition checked separately'
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
            if use_fingerprints:
                try:
                    pair = fingerprints(structure, kind)
                    row['fingerprint_success'] = True
                    if row.get('basic_valid'):
                        features.append([p.tolist() for p in pair])
                except (ValueError, TypeError, KeyError, RuntimeError, ZeroDivisionError, np.linalg.LinAlgError) as error:
                    row['fingerprint_success'] = False; row['fingerprint_error'] = str(error)
                row['legacy_joint_valid'] = bool(row.get('legacy_joint_valid') and row.get('fingerprint_success')) if row['chemical_valid'] is not None else None
            if row.get('basic_valid'):
                key = formula_key(counts)
                bucket = (row['case_id'], key) if task=='csp' else key
                row['unique'] = not any(structure_match(structure, old, kind) for old in unique[bucket])
                if row['unique']:
                    unique[bucket].append(structure)
                if novelty_files:
                    row['novel_against_reference'] = not any(structure_match(structure, train.structure(identifier), kind) for identifier in train.by_formula.get(key, []))
            elif row.get('basic_valid') is False:
                row['unique']=False
                if novelty_files: row['novel_against_reference']=False
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, ZeroDivisionError, np.linalg.LinAlgError) as error:
            row['evaluation_error'] = str(error)
            if row['parse_success'] is None:
                row['parse_success'] = False
                row.update(generation_success=False,basic_valid=False,legacy_joint_valid=False,unique=False,
                           failure_stage='structure_readback',error_type=type(error).__name__)
        rows.append(row)
        if len(rows)%25==0:
            print(json.dumps({'phase': 'unified_metrics', 'completed':len(rows), 'requested':len(records)}), flush=True)
    properties = evaluate_properties(rows, predictions, metadata, tolerances, reference_assessments)
    generated_rows=[r for r in rows if r['generation_success']]
    requested_fields=set(('generation_success','legacy_structural_valid','periodic_distance_valid','chemical_valid','basic_valid','legacy_joint_valid'))
    if use_fingerprints: requested_fields.add('fingerprint_success')
    if reidentify and any(r.get('target_group') is not None for r in rows): requested_fields.add('exact_target_group')
    if task=='csp': requested_fields.update(('composition_satisfied','formula_satisfied','atom_count_satisfied'))
    for name in BOOL_FIELDS:
        if any(r.get(name) is not None for r in generated_rows): requested_fields.add(name)
    if not use_fingerprints: requested_fields.discard('fingerprint_success')
    if not reidentify: requested_fields.discard('exact_target_group')
    technical=('parse_success','finite','lattice_valid','fingerprint_success','radius_distance_valid')
    common={name:rate((r.get(name) for r in (rows if name=='generation_success' else generated_rows)),name in requested_fields)
            for name in BOOL_FIELDS if name not in technical}
    diagnostics={name:rate((r.get(name) for r in generated_rows),name in requested_fields) for name in technical}
    diagnostics['radius_rule']={'status':'heuristic_only','coefficient':.6,
        'source':'Project-specific threshold without a verified paper source; excluded from main validity'}
    for row in rows:
        for name in requested_fields-set(technical)-{'generation_success'}:
            row[name+'_status']='passed' if row.get(name) else ('calculation_failed' if row.get(name) is None and row['generation_success'] else 'failed')
        if row['generation_success'] and row.get('fingerprint_success') is False:
            row['fingerprint_status']='calculation_failed'
    report = {'protocol_id': PROTOCOL, 'task': task, 'sampling_task':request.get('sampling_task',task), 'kind': kind, 'dataset': dataset, 'requested':len(rows),
        'generation':{'requested':len(rows),'generated':len(generated_rows),'failed':len(rows)-len(generated_rows),
                      'success_rate':len(generated_rows)/len(rows)},
        'common':common,'diagnostics':diagnostics,'properties':properties,
        'matching_profiles': PROFILES,'matching_profile_labels':{p:label.replace('scale=True (3D)','scale=False (2D adaptation)') if kind=='layer' else label
            for p,label in PROFILE_LABELS.items()}, 'protocol': {'basic_valid':'periodic-distance>=0.5 A AND SMACT; fingerprint independent',
            'denominators':'Generation success: requested count. Structure validity: generated structures. DNG yields and property hits: requested count. CSP match: test cases.',
            'legacy_joint_valid':'unchanged legacy pair-distance / volume AND SMACT AND fingerprint when enabled',
            'layer_matching':'physical 2D displacement / sqrt(area/atoms), unscaled; diffcsp_v1 label is a 2D adaptation',
            'reference_group_labels':'audited descriptor labels when supplied', 'fingerprints_enabled':use_fingerprints,
            'uniqueness_scope':'within each CSP case' if task=='csp' else 'whole DNG population',
            'novelty_reference':train.identity(), 'symmetry_tolerances_angstrom':[1e-4,1e-3,1e-2]},
        'identity':{'samples_sha256':sha256(samples),'references':refs.identity(),'novelty_references':train.identity(),
                    'request':request}, 'scope':'Raw structures and independent proxy quantities; no DFT, relaxation, stability or SUN claim.'}
    report['geometry'] = {key:numeric(r.get(key) for r in rows) for key in
        ('minimum_pair_distance_angstrom','volume_per_atom_angstrom3','area_per_atom_angstrom2',
         'layer_thickness_angstrom','periodic_connectivity_rank','rank2_atom_fraction','operation_residual_angstrom','metric_relative_residual')}
    if task=='csp':
        report['csp'] = csp_summary(rows, request)
        report['csp']['matching_profile_labels']=report['matching_profile_labels']
    else:
        report['dng'] = {'unique_valid':rate(r.get('unique') for r in rows),
            'novel_valid':rate((r.get('novel_against_reference') for r in rows),bool(novelty_files)),
            'vun_yield':rate([bool(r.get('basic_valid') and r.get('unique') and r.get('novel_against_reference'))
                             if r.get('novel_against_reference') is not None else None for r in rows],requested=bool(novelty_files))}
        if refs.rows:
            report['dng'].update(population_metrics(rows, reference['rows'], features, reference['fingerprints'], len(rows), (.4,10.)))
        for row in rows:
            row['template_unique'] = None
        observed = set()
        reference_templates = {r.get('template') for r in reference['rows'] if r.get('template')}
        train_templates = {json.dumps([kind, r['descriptor']['number'], sorted(o['letter'] for o in r['descriptor']['orbits'])])
                           for r, _ in train.rows if r.get('descriptor')}
        for row in rows:
            template = row.get('template')
            if template and row.get('basic_valid'):
                row['template_unique'] = template not in observed; observed.add(template)
                row['template_novel_against_test'] = template not in reference_templates if reference_templates else None
                row['template_novel_against_train'] = template not in train_templates if train_templates else None
        report['dng']['template_unique'] = rate((r.get('template_unique') for r in rows),any(r.get('template') for r in rows))
        report['dng']['template_novel_against_test'] = rate((r.get('template_novel_against_test') for r in rows),bool(reference_templates))
        report['dng']['template_novel_against_train'] = rate((r.get('template_novel_against_train') for r in rows),bool(train_templates))
    groups = defaultdict(list)
    for row in rows:
        if row.get('target_group') is not None:
            groups[row['target_group']].append(row)
    report['groups'] = [{'group':g,'requested':len(v),'exact':rate((r.get('exact_target_group') for r in v),reidentify),
        'containment':rate((r.get('target_operation_containment') for r in v),'target_operation_containment' in requested_fields),
        'basic_valid':rate(r.get('basic_valid') for r in v),
        'multi_tolerance':{str(t):rate((r.get('exact_group_'+str(t)) for r in v),reidentify) for t in (1e-4,1e-3,1e-2)}} for g,v in sorted(groups.items())]
    group_values = [g['exact']['rate'] for g in report['groups'] if g['exact']['rate'] is not None]
    report['group_macro_exact_rate'] = float(np.mean(group_values)) if group_values else None
    report['group_macro_denominator'] = {'requested_groups':len(report['groups']),
        'groups_with_evaluated_exact_rate':len(group_values),
        'scope':'Macro over requested groups with recorded outcomes; failures count as failures'}
    report['embedding_distance'] = {'status':'not_evaluated','reason':'Provide content-bound independent Wrenformer embeddings to compute FWD'}
    if embeddings:
        source = json.loads(Path(embeddings).read_text(encoding='utf-8'))
        if source.get('samples_sha256')!=sha256(samples) or source.get('reference_sha256')!=refs.identity() or not source.get('evaluator'):
            raise ValueError('Embedding identity must bind samples, reference files and evaluator')
        a = [source['generated'][str(r['index'])] for r in rows if r.get('basic_valid') and str(r['index']) in source['generated']]
        b = [source['reference'][r['id']] for r in reference['rows'] if r['id'] in source['reference']]
        report['embedding_distance'] = {'status':'provided_embeddings','evaluator':source['evaluator'],
            'frechet_distance':frechet_distance(a,b),'generated_count':len(a),'reference_count':len(b)}
    elapsed = time.perf_counter()-started
    generation_times = [r['generation_seconds'] for r in rows if r.get('generation_seconds') is not None]
    report['efficiency'] = {'metric_seconds_excluding_reference_cache':elapsed,'generation_seconds':numeric(generation_times),
        'generation_time_coverage':len(generation_times),'generated_structures_per_second':len(generated_rows)/sum(generation_times)
            if len(generation_times)==len(rows) and sum(generation_times)>0 else None}
    output.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for r in rows for key in r})
    with (output/'samples.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream,fieldnames=fields); writer.writeheader()
        for row in rows:
            writer.writerow({k:json.dumps(v,ensure_ascii=False,sort_keys=True) if isinstance(v,(dict,list)) else v for k,v in row.items()})
    if task=='csp':
        with (output/'cases.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=list(report['csp']['case_rows'][0]) if report['csp']['case_rows'] else ['case_id']);writer.writeheader();writer.writerows(report['csp']['case_rows'])
        with (output/'csp_metrics.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=['profile','profile_label','kind','ltol','stol','angle_degrees','scale',
                'candidates_per_case','test_cases','matched_cases','match_rate','valid_match_rate'])
            writer.writeheader()
            for profile,values in report['csp']['profiles'].items():
                for k,value in values.items():
                    if k.isdigit(): writer.writerow({'profile':profile,'profile_label':report['matching_profile_labels'][profile],
                        'kind':kind,'ltol':PROFILES[profile]['ltol'],'stol':PROFILES[profile]['stol'],
                        'angle_degrees':PROFILES[profile]['angle_tol'],'scale':PROFILES[profile]['scale'] if kind=='space' else False,
                        'candidates_per_case':int(k),'test_cases':report['csp']['cases'],
                        **{n:value[n] for n in ('matched_cases','match_rate','valid_match_rate')}})
    write_json(output/'summary.json',report)
    return rows,report,reference['rows']
