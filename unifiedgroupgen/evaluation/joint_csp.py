"""Post-generation CSP matching: FlowMM-style 3D and explicit vacuum-free 2D."""
from collections import Counter
from itertools import product
from functools import lru_cache
import json
import numpy as np
from scipy.optimize import linear_sum_assignment
from unifiedgroupgen.common.csp_cases import parse_csp_manifest
from unifiedgroupgen.common.data_utils import target_from_record
from unifiedgroupgen.common.structures import record_counts


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


def _protocol(kind):
    if kind == 'rod':
        raise ValueError('Rod CSP requires the axial/transverse 1D matching protocol')
    return 'flowmm_3d' if kind == 'space' else 'layer_xy_real_z_v1'


def match_csp(reference, generated, stol=.5, ltol=.3, angle_tol=10):
    protocol = _protocol(reference['kind'])
    expected = Counter(reference['atomic_numbers'])
    actual = Counter(generated.get('atomic_numbers', []))
    result = dict(composition_match=actual == expected, matched=False, normalized_rmse=None,
                  protocol=protocol)
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


def prepare_csp_manifest(references, candidates_per_case):
    """Extract only the conventional-cell inventory and identity from reference records."""
    rows = []
    for record in references:
        rows.append(dict(dataset=record.get('dataset'), case_id=record.get('id'),
                         kind=record['descriptor']['kind'], counts=record_counts(record, '.')))
    return parse_csp_manifest(dict(schema='ugg_csp_cases_v1',
                                   candidates_per_case=candidates_per_case, cases=rows))


def _top_k(cases, ks):
    summary = {}
    for k in ks:
        matches, errors, exact, attempted = 0, [], 0, 0
        outcomes = Counter()
        for case in cases:
            rows = [row for row in case['candidates'] if row['candidate'] < k]
            attempted += len(rows)
            outcomes.update(row['status'] for row in rows)
            exact += any(row['composition_match'] for row in rows)
            rms = [row['normalized_rmse'] for row in rows if row['matched']]
            matches += bool(rms)
            if rms:
                errors.append(min(rms))
        planned = len(cases) * k
        summary[str(k)] = dict(cases=len(cases), matched=matches, match_rate=matches/len(cases),
                              exact_composition_cases=exact, planned_candidates=planned,
                              attempted_candidates=attempted, unattempted_candidates=planned-attempted,
                              candidate_outcomes=dict(outcomes),
                              matched_only_normalized_rmse=float(np.mean(errors)) if errors else None)
    return summary


def evaluate_csp(samples, references, ks=(1,), manifest=None):
    """Match after generation, retaining the planned case and candidate denominators."""
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError('Candidate counts must be positive integers')
    ks = sorted(set(ks))
    keys = [(row.get('dataset', ''), row['id']) for row in references]
    if not keys or len(keys) != len(set(keys)):
        raise ValueError('CSP references must have unique dataset/id pairs')
    reference_by_key = dict(zip(keys, references))
    planned = {case.key: case for case in manifest.cases} if manifest else None
    if manifest:
        if ks[-1] > manifest.candidates_per_case:
            raise ValueError('Requested top-K exceeds the manifest candidate count')
        missing = planned.keys() - reference_by_key.keys()
        if missing:
            raise ValueError(f'CSP manifest cases are missing from references: {sorted(missing)[:5]}')
        keys = list(planned)
    results = {key: {} for key in keys}
    manifest_digest = manifest.digest if manifest else None
    match_fields = ('kind', 'atomic_numbers', 'coordinates', 'lattice', 'status', 'reason')
    for sample in samples:
        key = (sample.get('dataset', ''), sample.get('case_id'))
        if key not in results:
            raise ValueError(f'Unexpected CSP sample dataset/case_id: {key}')
        candidate = sample.get('candidate', sample.get('index'))
        if type(candidate) is not int or candidate < 0:
            raise ValueError('Every CSP sample needs a nonnegative integer candidate index')
        if candidate in results[key]:
            raise ValueError(f'Duplicate CSP candidate {candidate} for {key}')
        if manifest and (candidate >= manifest.candidates_per_case
                         or sample.get('case_manifest_sha256') != manifest_digest):
            raise ValueError(f'CSP sample candidate or manifest identity does not match {key}')
        results[key][candidate] = {field: sample[field] for field in match_fields if field in sample}

    cases = []
    by_dataset = {}
    for key in keys:
        reference = reference_by_key[key]
        view = target_view(reference)
        counts = Counter(view['atomic_numbers'])
        if manifest and (planned[key].kind != view['kind'] or dict(planned[key].chemistry.counts) != counts):
            raise ValueError(f'CSP reference kind or per-cell counts changed for {key}')
        protocol = _protocol(view['kind'])
        candidates = []
        for candidate, row in sorted(results[key].items()):
            status = row.get('status', 'generated')
            if status == 'generated':
                metrics = match_csp(view, row)
            else:
                metrics = dict(composition_match=Counter(row.get('atomic_numbers') or []) == counts,
                               matched=False, normalized_rmse=None, protocol=protocol)
            candidates.append(dict(candidate=candidate, status=status, reason=row.get('reason'), **metrics))
        case = dict(dataset=key[0], case_id=key[1], protocol=protocol, candidates=candidates)
        cases.append(case)
        by_dataset.setdefault(key[0], []).append(case)
    return dict(summary=_top_k(cases, ks), cases=cases,
                by_dataset={name: _top_k(rows, ks) for name, rows in sorted(by_dataset.items())},
                case_manifest_sha256=manifest_digest,
                protocol=dict(stol=.5, ltol=.3, angle_tol=10, basic_flowmm_candidates=1,
                              layer_length_scale='sqrt(area/N)', layer_normal='real Angstrom, no wrap',
                              layer_basis_entries=[-1, 0, 1], missing_and_failed_cases_in_denominator=True,
                              scope='Reconstruction matching; not ground-state or stability certification'))
