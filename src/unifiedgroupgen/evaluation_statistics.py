"""Population metrics with named protocols, bounded memory and explicit denominators."""
from collections import Counter
import numpy as np
from scipy.linalg import sqrtm
from scipy.spatial.distance import cdist, jensenshannon, pdist
from scipy.stats import wasserstein_distance

from .standard_metrics import coverage


def logical_and(*values):
    if any(v is not None and not v for v in values):
        return False
    return None if any(v is None for v in values) else True


def rate(values):
    values = list(values)
    known = [x for x in values if x is not None]
    passed = sum(bool(x) for x in known)
    return {'attempted': len(values), 'evaluated': len(known), 'passed': passed,
            'unknown': len(values)-len(known),
            'rate_over_all_attempts': passed/len(values) if values and known else None,
            'rate_on_evaluated': passed/len(known) if known else None}


def numeric(values):
    a = np.array([float(x) for x in values if x is not None and np.isfinite(x)])
    return {'evaluated': len(a), 'mean': float(a.mean()) if len(a) else None,
            'median': float(np.median(a)) if len(a) else None,
            'p05': float(np.quantile(a, .05)) if len(a) else None,
            'p95': float(np.quantile(a, .95)) if len(a) else None}


def joint_coverage(generated, reference, attempted, reference_requested, structural_cutoff=.4, composition_cutoff=10.):
    if not generated or not reference:
        return {'status': 'unavailable', 'cov_precision': None, 'cov_recall': None}
    gs, gc = map(np.array, zip(*generated)); rs, rc = map(np.array, zip(*reference))
    recalled = np.zeros(len(rs), dtype=bool); precise = 0
    for start in range(0, len(gs), 128):
        # Both conditions must hold for the SAME generated/reference pair.
        close = (cdist(gs[start:start+128], rs) <= structural_cutoff) & (cdist(gc[start:start+128], rc) <= composition_cutoff)
        recalled |= close.any(axis=0)
        precise += int(close.any(axis=1).sum())
    return {'status': 'complete' if len(rs) == reference_requested else 'partial_reference',
            'cov_precision': precise/attempted, 'cov_recall': float(recalled.mean()),
            'cov_recall_over_all_reference': int(recalled.sum())/reference_requested,
            'generated_fingerprints': len(gs), 'reference_fingerprints': len(rs),
            'reference_requested': reference_requested, 'protocol': 'joint_neighbor'}


def js_distance(a, b):
    a, b = Counter(a), Counter(b)
    keys = sorted(set(a)|set(b), key=str)
    if not a or not b:
        return None
    # Divergence in bits, not Jensen-Shannon distance.
    return float(jensenshannon([a[k] for k in keys], [b[k] for k in keys], base=2)**2)


def feature_distances(a, b):
    if not a or not b:
        return {'status': 'unavailable'}
    a, b = np.array(a), np.array(b)
    # CMD with a shared bounded support from the union; definition is recorded.
    scale = np.maximum(np.ptp(np.concatenate([a, b]), axis=0), 1e-12)
    u, v = a/scale, b/scale
    value = np.linalg.norm(u.mean(0)-v.mean(0))
    uc, vc = u-u.mean(0), v-v.mean(0)
    for order in range(2, 51):
        value += np.linalg.norm((uc**order).mean(0)-(vc**order).mean(0))
    # Deterministic, bounded subset: pair counts are exported with the value.
    subset = a[np.linspace(0, len(a)-1, min(len(a), 1000), dtype=int)]
    return {'cmd_50_union_range_normalized': float(value),
            'pairwise_l2_mean': float(pdist(subset).mean()) if len(subset)>1 else None,
            'diversity_samples': len(subset), 'reference_samples': len(b)}


def frechet_distance(a, b):
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    if len(a)<2 or len(b)<2 or a.ndim!=2 or b.ndim!=2 or a.shape[1]!=b.shape[1]:
        raise ValueError('Frechet embeddings need >=2 finite vectors per population with equal dimension')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite embedding')
    ca, cb = np.atleast_2d(np.cov(a, rowvar=False)), np.atleast_2d(np.cov(b, rowvar=False))
    root = sqrtm(ca @ cb)
    if np.iscomplexobj(root) and np.max(np.abs(root.imag))>1e-5:
        raise ValueError('Unstable covariance square root')
    return float(max(0., np.sum((a.mean(0)-b.mean(0))**2)+np.trace(ca+cb-2*root.real)))


DISTRIBUTIONS = ('atoms', 'num_elements', 'density_g_cm3', 'atomic_density_angstrom3',
                 'volume_per_atom_angstrom3', 'area_per_atom_angstrom2', 'areal_density_ug_cm2',
                 'areal_atomic_density_angstrom2', 'layer_thickness_angstrom', 'coordination_number')


def population_metrics(rows, reference_rows, generated_fp, reference_fp, attempted, cutoffs):
    result = {'coverage': {
        'legacy_independent_nn': coverage(generated_fp, reference_fp, attempted, len(reference_rows), *cutoffs),
        'joint_neighbor': joint_coverage(generated_fp, reference_fp, attempted, len(reference_rows), *cutoffs)},
        'distributions': {}}
    for key in DISTRIBUTIONS:
        a = [r[key] for r in rows if r.get('basic_valid') and r.get(key) is not None]
        b = [r[key] for r in reference_rows if r.get(key) is not None]
        result['distributions'][key] = {'generated_count': len(a), 'reference_count': len(b),
            'wasserstein': float(wasserstein_distance(a, b)) if a and b else None}
    for key in ('recovered_group', 'atoms', 'num_elements', 'coordinate_dof', 'wyckoff_dimension_signature'):
        a = [r[key] for r in rows if r.get('basic_valid') and r.get(key) is not None]
        b = [r[key] for r in reference_rows if r.get(key) is not None]
        result['distributions'][key+'_jsd_bits'] = {'value': js_distance(a, b), 'generated_count': len(a), 'reference_count': len(b)}
    a = [z for r in rows if r.get('basic_valid') for z in r.get('elements', [])]
    b = [z for r in reference_rows for z in r.get('elements', [])]
    result['distributions']['element_prevalence_jsd_bits'] = {'value':js_distance(a,b),
        'definition':'Element presences per structure, normalized across all element presences'}
    for index, name in enumerate(('structural', 'compositional')):
        result[name+'_fingerprints'] = feature_distances([p[index] for p in generated_fp], [p[index] for p in reference_fp])
    return result
