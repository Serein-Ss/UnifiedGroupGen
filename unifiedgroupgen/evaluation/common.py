"""Common raw-structure validity, group compliance and aggregate generation metrics."""
from collections import defaultdict
from dataclasses import dataclass
import json
import warnings
import subprocess
import sys
from pathlib import Path
import numpy as np
from scipy.optimize import linear_sum_assignment
from unifiedgroupgen.common.structures import ReferenceBank, counts_of, formula_key, load_structure, sha256
from .utils import (geometry_metrics, layer_structure, PROFILES, PROFILE_LABELS, numeric, rate, frechet_distance, logical_and, composition_validity, fingerprints, structure_match)
from .csp import csp_summary, check_reconstruction
from .dng import dng_summary

@dataclass(frozen=True)
class EvaluationContext:
    task: str
    kind: str
    dataset: str | None
    request: dict
    refs: ReferenceBank
    train: ReferenceBank
    use_fingerprints: bool
    reidentify: bool
    novelty_files: tuple | list
    samples: Path

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
        from unifiedgroupgen.common.structures import record_counts
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

def group_summary(rows, requested_fields, reidentify):
    report = {}
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
    return report


def embedding_summary(rows, reference, refs, samples, embeddings):
    report = {}
    report['embedding_distance'] = {'status':'not_evaluated','reason':'Provide content-bound independent Wrenformer embeddings to compute FWD'}
    if embeddings:
        source = embeddings
        if source.get('samples_sha256')!=sha256(samples) or source.get('reference_sha256')!=refs.identity() or not source.get('evaluator'):
            raise ValueError('Embedding identity must bind samples, reference files and evaluator')
        a = [source['generated'][str(r['index'])] for r in rows if r.get('basic_valid') and str(r['index']) in source['generated']]
        b = [source['reference'][r['id']] for r in reference['rows'] if r['id'] in source['reference']]
        report['embedding_distance'] = {'status':'provided_embeddings','evaluator':source['evaluator'],
            'frechet_distance':frechet_distance(a,b),'generated_count':len(a),'reference_count':len(b)}
    return report['embedding_distance']


def build_summary(rows, features, reference, properties, context, embeddings):
    task, kind, dataset, request = context.task, context.kind, context.dataset, context.request
    refs, train, use_fingerprints = context.refs, context.train, context.use_fingerprints
    reidentify, novelty_files, samples = context.reidentify, context.novelty_files, context.samples
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
        report['dng'] = dng_summary(rows, reference, features, refs, train, kind, novelty_files)
    report.update(group_summary(rows, requested_fields, reidentify))
    report['embedding_distance'] = embedding_summary(rows, reference, refs, samples, embeddings)
    return report

def evaluate_records(records, context):
    task, kind, dataset, request = context.task, context.kind, context.dataset, context.request
    refs, train, use_fingerprints = context.refs, context.train, context.use_fingerprints
    reidentify, novelty_files, samples = context.reidentify, context.novelty_files, context.samples
    rows, features, unique = [], [], defaultdict(list)
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
            check_declared_orbits(row, record, structure, samples, counts)
            reference_record = check_requested_chemistry(row, record, request, refs, task, counts)
            check_symmetry(row, record, request, structure, kind, reidentify)
            check_reconstruction(row, reference_record, refs, structure, kind, task)
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
    return rows, features

def summarize_attempts(records):
    """Use every requested attempt in operational validity rates, including failures."""
    total = len(records)
    generated = [r for r in records if r['status'] == 'generated']
    reports = [r['validation'] for r in generated]
    counts = {key: sum(bool(r.get(key)) for r in reports) for key in
              ('group_operation_closure', 'distance_valid', 'orbit_multiplicity_regular',
               'no_coincident_orbits', 'chemistry_validated', 'stability_validated', 'properties_validated')}
    identified = [r for r in reports if r.get('exact_target_group') is not None]
    geometry_keys = ('group_operation_closure', 'distance_valid', 'orbit_multiplicity_regular', 'no_coincident_orbits')
    geometry = [r for r in reports if all(r.get(key) for key in geometry_keys)
                and np.isfinite(r.get('metric_residual', np.nan))]
    exact_success = sum(bool(r['exact_target_group']) for r in identified)
    return {'attempted': total, 'generated': len(generated), 'failed': total-len(generated),
            'counts': counts, 'rates_over_all_attempts': {key: count/total if total else None for key, count in counts.items()},
            'exact_group_identified': len(identified),
            'exact_group_success': exact_success,
            'exact_group_rate_over_all_attempts': exact_success/total if total else None,
            'exact_group_unidentified_generated': len(reports)-len(identified),
            'geometry_screen_success': len(geometry),
            'geometry_and_exact_group_success': sum(bool(r.get('exact_target_group')) for r in geometry),
            'external_validation_performed_counts': {key: counts[key] for key in
                ('chemistry_validated', 'stability_validated', 'properties_validated')},
            'scope': 'Distance threshold is a geometric diagnostic; chemistry, stability and property hit require external verification.'}
