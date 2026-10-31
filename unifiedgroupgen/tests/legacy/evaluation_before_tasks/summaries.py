"""Protocol-preserving CSP, DNG and symmetry population summaries."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.io import sha256
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.protocols import PROFILES, PROFILE_LABELS
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.statistics import numeric, population_metrics, rate, frechet_distance


PROTOCOL = 'unified_raw_v2'
BOOL_FIELDS = ('generation_success', 'parse_success', 'finite', 'lattice_valid', 'legacy_structural_valid',
    'periodic_distance_valid', 'radius_distance_valid', 'chemical_valid', 'basic_valid', 'fingerprint_success',
    'legacy_joint_valid', 'composition_satisfied', 'formula_satisfied', 'atom_count_satisfied',
    'element_set_satisfied', 'descriptor_composition_satisfied', 'target_operation_containment', 'exact_target_group',
    'orbit_multiplicity_regular', 'no_coincident_orbits', 'observation_satisfied')


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


def dng_summary(rows, reference, features, refs, train, kind, novelty_files):
    report = {}
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
    return report['dng']


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
        source = json.loads(Path(embeddings).read_text(encoding='utf-8'))
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
