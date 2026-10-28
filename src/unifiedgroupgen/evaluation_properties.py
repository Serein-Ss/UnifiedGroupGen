"""Independent proxy quantities; no relaxation, stability or SUN evaluation."""
from collections import defaultdict
import math

from .evaluation_statistics import numeric, rate, logical_and


def property_spec(value, tolerance=None):
    spec = {'value': float(value), 'op': 'eq', 'tolerance': tolerance} if not isinstance(value, dict) else dict(value)
    spec.setdefault('op', 'eq')
    if spec['op'] not in ('eq', 'le', 'ge', 'between'):
        raise ValueError('Property operator must be eq, le, ge or between')
    if spec['op']=='between':
        if len(spec.get('bounds', [])) != 2 or not all(math.isfinite(float(x)) for x in spec['bounds']) or spec['bounds'][0]>spec['bounds'][1]:
            raise ValueError('Property bounds must be an ordered finite pair')
    elif not math.isfinite(float(spec['value'])):
        raise ValueError('Property target must be finite')
    if spec['op']=='eq' and (spec.get('tolerance') is None or not math.isfinite(float(spec['tolerance'])) or spec['tolerance']<0):
        raise ValueError('Equality targets require a finite nonnegative property tolerance')
    return spec


def assess(value, spec):
    if value is None or not math.isfinite(float(value)):
        return None, None
    value = float(value)
    if spec['op']=='eq':
        error = abs(value-spec['value']); return error, error<=spec['tolerance']
    if spec['op']=='le':
        return max(0., value-spec['value']), value<=spec['value']
    if spec['op']=='ge':
        return max(0., spec['value']-value), value>=spec['value']
    lo, hi = spec['bounds']; return max(lo-value, value-hi, 0.), lo<=value<=hi


def evaluate_properties(rows, predictions, metadata, tolerances, reference_predictions=None):
    names = set()
    for row in rows:
        names.update(row.get('property_targets', {}))
    for values in predictions.values():
        names.update(values)
    names -= {'source_domain', 'job', 'case_id', 'index', 'id'}
    # CSV may carry provenance columns: evaluate only numeric quantities or requested names.
    names = {n for n in names if any(n in r.get('property_targets', {}) for r in rows)
             or any(isinstance(p.get(n), (int, float)) for p in predictions.values())}
    for row in rows:
        values = predictions.get(str(row['index']), {})
        hits = []
        for name in names:
            value = values.get(name)
            value = float(value) if isinstance(value, (int, float)) and math.isfinite(value) else None
            row['predicted_'+name] = value
            if name in row.get('property_targets', {}):
                spec = property_spec(row['property_targets'][name], tolerances.get(name))
                error, hit = assess(value, spec)
                row['property_error_'+name], row['property_hit_'+name] = error, hit
                row['valid_property_hit_'+name] = logical_and(row.get('basic_valid'),hit) if hit is not None else None
                hits.append(hit)
        row['joint_property_hit'] = (all(hits) if hits and all(v is not None for v in hits) else None)
        row['valid_joint_property_hit'] = (logical_and(row.get('basic_valid'),row['joint_property_hit'])
                                           if row['joint_property_hit'] is not None else None)
    summary = {'status': 'provided_independent_proxy' if predictions else 'not_evaluated',
               'evaluator': metadata, 'quantities': {}, 'joint_hit': rate(r['joint_property_hit'] for r in rows),
               'valid_joint_hit': rate(r['valid_joint_property_hit'] for r in rows)}
    for name in sorted(names):
        targeted = [r for r in rows if name in r.get('property_targets', {})]
        summary['quantities'][name] = {'predictions': numeric(r.get('predicted_'+name) for r in rows),
            'targeted_attempts': len(targeted), 'target_error': numeric(r.get('property_error_'+name) for r in targeted),
            'hit': rate(r.get('property_hit_'+name) for r in targeted),
            'valid_hit': rate(r.get('valid_property_hit_'+name) for r in targeted)}
        from scipy.stats import wasserstein_distance
        generated=[r['predicted_'+name] for r in rows if r.get('basic_valid') and r.get('predicted_'+name) is not None]
        reference=[v[name] for v in (reference_predictions or {}).values() if v.get(name) is not None and math.isfinite(v[name])]
        summary['quantities'][name]['wasserstein_same_proxy']={
            'value':float(wasserstein_distance(generated,reference)) if generated and reference else None,
            'generated_valid_count':len(generated),'reference_count':len(reference)}
    groups = defaultdict(list)
    import json
    for row in rows:
        groups[json.dumps(row.get('property_targets', {}), sort_keys=True)].append(row)
    summary['target_groups'] = [{'targets': json.loads(key), 'attempted': len(group),
        'joint_hit': rate(r['joint_property_hit'] for r in group),
        'valid_joint_hit': rate(r['valid_joint_property_hit'] for r in group),
        'predictions': {n: numeric(r.get('predicted_'+n) for r in group) for n in sorted(names)}}
        for key, group in groups.items()]
    return summary
