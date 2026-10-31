"""CSP reconstruction and first-1/5/20 candidate matching metrics."""
from collections import defaultdict
import numpy as np
from .utils import PROFILES, PROFILE_LABELS, match_pair, numeric, rate

def check_reconstruction(row, reference_record, refs, structure, kind, task):
    from .common import descriptor_metadata
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
