"""Post-generation CSP matching: FlowMM-style 3D and explicit vacuum-free 2D."""
from collections import Counter
from itertools import product
from functools import lru_cache
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from unifiedgroupgen.pl_data.joint_path import target_from_record


def target_view(record):
    state = target_from_record(record).state
    decoded = state.decoded(wrap=False)
    return dict(kind=state.spec.kind, atomic_numbers=decoded['atomic_numbers'].tolist(),
                coordinates=decoded['coordinates'].tolist(), lattice=decoded['lattice'].tolist())


@lru_cache(maxsize=1)
def _bases():
    result = []
    for values in product((-1, 0, 1), repeat=4):
        matrix = np.asarray(values).reshape(2, 2)
        if abs(round(np.linalg.det(matrix))) == 1:
            result.append(matrix)
    return tuple(result)


def _layer_rms(reference, generated, stol, ltol, angle_tol):
    """Bounded GL(2,Z) basis search, periodic xy only; real z never wraps."""
    r, g = np.asarray(reference['coordinates']), np.asarray(generated['coordinates'])
    lr = np.asarray(reference['lattice'])[:2, :2]
    lg = np.asarray(generated['lattice'])[:2, :2]
    area = abs(np.linalg.det(lr))
    generated_area = abs(np.linalg.det(lg))
    if area < 1e-10 or generated_area < 1e-10:
        return None
    species_r, species_g = np.asarray(reference['atomic_numbers']), np.asarray(generated['atomic_numbers'])
    counts = Counter(species_r)
    rare = min(counts, key=counts.get)
    anchor = int(np.flatnonzero(species_r == rare)[0])
    scale = np.sqrt(area/generated_area)
    length = np.sqrt(area/len(r))
    zr, zg = r[:, 2]-r[:, 2].mean(), g[:, 2]-g[:, 2].mean()
    shifts = np.asarray(list(product((-1, 0, 1), repeat=2)))

    def cell_values(lattice):
        norms = np.linalg.norm(lattice, axis=0)
        cosine = np.dot(lattice[:, 0], lattice[:, 1])/np.prod(norms)
        return norms, np.degrees(np.arccos(np.clip(cosine, -1, 1)))

    nr, ar = cell_values(lr)
    best = None
    for basis in _bases():
        ng, ag = cell_values(scale*lg@basis)
        if np.any(abs(ng-nr) > ltol*nr) or abs(ag-ar) > angle_tol:
            continue
        xy = g[:, :2]@np.linalg.inv(basis).T
        for j in np.flatnonzero(species_g == rare):
            shift = r[anchor, :2]-xy[j]
            for sign in ((1,) if np.allclose(zg, 0) else (-1, 1)):
                distances = []
                for z in counts:
                    ri, gi = np.flatnonzero(species_r == z), np.flatnonzero(species_g == z)
                    delta = r[ri, None, :2]-(xy[None, gi]+shift)
                    delta -= np.round(delta)
                    cartesian = (delta[:, :, None, :]+shifts)@lr.T
                    cost = np.min(np.sum(cartesian**2, axis=-1), axis=-1)
                    cost += (zr[ri, None]-sign*zg[None, gi])**2
                    rows, columns = linear_sum_assignment(cost)
                    distances.extend(cost[rows, columns].tolist())
                rms = np.sqrt(np.mean(distances))/length
                maximum = np.sqrt(max(distances))/length
                if maximum <= stol and (best is None or rms < best):
                    best = float(rms)
    return best


def match_csp(reference, generated, stol=.5, ltol=.3, angle_tol=10):
    expected = Counter(reference['atomic_numbers'])
    actual = Counter(generated.get('atomic_numbers', []))
    result = dict(composition_match=actual == expected, matched=False, normalized_rmse=None,
                  protocol='flowmm_3d' if reference['kind']=='space' else 'layer_xy_real_z_v1')
    if actual != expected or not generated.get('coordinates'):
        return result
    if generated.get('kind') != reference['kind']:
        return result
    try:
        if reference['kind'] == 'space':
            from pymatgen.core import Structure, Lattice
            from pymatgen.analysis.structure_matcher import StructureMatcher
            structures = [Structure(Lattice(np.asarray(x['lattice']).T), x['atomic_numbers'], x['coordinates'])
                          for x in (reference, generated)]
            distances = StructureMatcher(stol=stol, ltol=ltol, angle_tol=angle_tol).get_rms_dist(*structures)
            # pymatgen already normalizes get_rms_dist by (V/N)^(1/3).
            value = None if distances is None else float(distances[0])
        else:
            value = _layer_rms(reference, generated, stol, ltol, angle_tol)
        result.update(matched=value is not None, normalized_rmse=value)
    except (ValueError, np.linalg.LinAlgError, FloatingPointError) as error:
        result['matching_error'] = str(error)
    return result


def evaluate_csp(samples, references, ks=(1,)):
    """References are consumed only here, after generation; missing cases fail."""
    if not ks or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError('Candidate counts must be positive integers')
    keys = [(r.get('dataset', ''), r['id']) for r in references]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('CSP references must have unique dataset/id pairs')
    results = {}; key_set = set(keys)
    for sample in samples:
        key = (sample.get('dataset', ''), sample.get('case_id'))
        if key not in key_set:
            raise ValueError('Every CSP sample needs a dataset/case_id present in references')
        results.setdefault(key, []).append(sample)
    cases = []
    for key, reference in zip(keys, references):
        view = target_view(reference)
        candidates = sorted(results.get(key, []), key=lambda x: x.get('candidate', x.get('index', 0)))
        candidates = [dict(candidate=i, **match_csp(view, row)) for i, row in enumerate(candidates)]
        cases.append(dict(dataset=key[0], case_id=key[1], protocol='flowmm_3d' if view['kind']=='space'
                          else 'layer_xy_real_z_v1', candidates=candidates))
    summary = {}
    for k in sorted(set(ks)):
        matches, errors, exact = 0, [], 0
        for case in cases:
            rows = case['candidates'][:k]
            exact += any(x['composition_match'] for x in rows)
            rms = [x['normalized_rmse'] for x in rows if x['matched']]
            matches += bool(rms)
            if rms:
                errors.append(min(rms))
        summary[str(k)] = dict(cases=len(cases), matched=matches, match_rate=matches/len(cases),
                              exact_composition_cases=exact,
                              matched_only_normalized_rmse=float(np.mean(errors)) if errors else None)
    return dict(summary=summary, cases=cases, protocol=dict(stol=.5,ltol=.3,angle_tol=10,
                basic_flowmm_candidates=1, layer_length_scale='sqrt(area/N)', layer_normal='real Angstrom, no wrap',
                layer_basis_entries=[-1,0,1], missing_and_failed_cases_in_denominator=True,
                scope='Reconstruction matching; not ground-state or stability certification'))
