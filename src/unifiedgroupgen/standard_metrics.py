"""CDVAE/CGDiT-style metrics, with explicit 2D geometry and missing-evidence accounting."""

from collections import Counter, defaultdict
import itertools
import json
from pathlib import Path
from functools import lru_cache
import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance
from pymatgen.core import Element, Lattice, Structure
from pymatgen.analysis.structure_matcher import StructureMatcher


def layer_structure(structure, height=None):
    """Remove arbitrary vacuum and rigid rotation; retain physical out-of-plane distances."""
    ab = structure.lattice.matrix[:2]
    normal = np.cross(*ab)
    normal /= np.linalg.norm(normal)
    cart = structure.cart_coords
    z = cart @ normal
    z -= z.mean()
    xy = cart @ ab.T @ np.linalg.inv(ab @ ab.T)
    height = height or max(40.0, float(np.ptp(z))+20)
    rows = np.zeros((3, 3))
    rows[:2, :2] = np.linalg.cholesky(ab @ ab.T)
    rows[2, 2] = height
    return Structure(Lattice(rows, pbc=(True, True, False)), structure.species,
                     np.column_stack([xy % 1, z/height+0.5]))


def composition_validity(structure, maximum_combinations=10_000_000):
    """Same unary/alloy exceptions, charge neutrality and Pauling test as prior CGDiT."""
    import smact
    from smact.screening import pauling_test
    counts = Counter(structure.atomic_numbers)
    numbers = sorted(counts)
    symbols = [Element.from_Z(z).symbol for z in numbers]
    if len(symbols) == 1 or all(symbol in smact.metals for symbol in symbols):
        return True, None
    elements = smact.element_dictionary(symbols)
    oxidation = [elements[symbol].oxidation_states for symbol in symbols]
    if not all(oxidation):
        return None, 'Missing oxidation-state data'
    if np.prod([len(states) for states in oxidation], dtype=float) > maximum_combinations:
        return None, 'Oxidation-state search budget exceeded'
    amounts = np.array([counts[z] for z in numbers])
    amounts //= np.gcd.reduce(amounts)
    electronegativities = [elements[s].pauling_eneg for s in symbols]
    for states in itertools.product(*oxidation):
        balanced, _ = smact.neutral_ratios(states, stoichs=[(int(n),) for n in amounts], threshold=int(amounts.max()))
        if balanced:
            try:
                allowed = pauling_test(states, electronegativities)
            except TypeError:
                # Preserve prior benchmark treatment of missing Pauling electronegativity.
                allowed = True
            if allowed:
                return True, None
    return False, None


def geometry_metrics(structure, kind, cutoff=0.5, radius_ratio=0.6):
    structure = layer_structure(structure) if kind != 'space' else structure
    if kind == 'space':
        structure = Structure(Lattice(structure.lattice.matrix, pbc=(True, True, True)),
                              structure.species, structure.frac_coords)
    distances = structure.distance_matrix.copy()
    np.fill_diagonal(distances, np.inf)
    pair_min = float(distances.min()) if len(structure)>1 else None
    standard = (pair_min is None or pair_min >= cutoff) and structure.volume >= 0.1
    # Unlike the legacy matrix check, this includes an atom's own periodic images.
    _, _, _, short_distances = structure.get_neighbor_list(cutoff)
    periodic_valid = not np.any(short_distances < cutoff-1e-10)
    radii = []
    from pymatgen.analysis.molecule_structure_comparator import CovalentRadius
    for site in structure:
        radii.append(CovalentRadius.radius.get(site.specie.symbol))
    radius_valid, connectivity_rank = None, None
    if all(r is not None for r in radii):
        centers, neighbors, images, dd = structure.get_neighbor_list(2*max(radii)*1.2)
        sums = np.array(radii)[centers]+np.array(radii)[neighbors]
        radius_valid = bool(np.all(dd >= radius_ratio*sums))
        selected = dd <= 1.2*sums
        connectivity_rank = periodic_connectivity(len(structure), centers[selected], neighbors[selected], images[selected])
    result = {'legacy_structural_valid': bool(standard), 'periodic_distance_valid': bool(periodic_valid),
              'radius_distance_valid': radius_valid, 'minimum_pair_distance_angstrom': pair_min,
              'periodic_connectivity_rank': connectivity_rank,
              'num_elements': len(set(structure.atomic_numbers)), 'atoms': len(structure),
              'density_g_cm3': float(structure.density) if kind == 'space' else None,
              'volume_per_atom_angstrom3': structure.volume/len(structure) if kind == 'space' else None,
              'area_per_atom_angstrom2': None, 'layer_thickness_angstrom': None}
    if kind != 'space':
        area = np.linalg.norm(np.cross(*structure.lattice.matrix[:2]))
        result.update(area_per_atom_angstrom2=float(area/len(structure)),
                      layer_thickness_angstrom=float(np.ptp(structure.cart_coords[:, 2])))
    return result


def periodic_connectivity(n, centers, neighbors, images):
    """Rank of translation cycles in the largest periodic bond-connected component."""
    edges = [[] for _ in range(n)]
    for i, j, shift in zip(centers, neighbors, images):
        shift = np.asarray(shift)
        edges[int(i)].append((int(j), shift))
        edges[int(j)].append((int(i), -shift))
    visited, rank = set(), 0
    for initial in range(n):
        if initial in visited:
            continue
        offsets, pending, cycles = {initial: np.zeros(3)}, [initial], []
        while pending:
            i = pending.pop()
            visited.add(i)
            for j, shift in edges[i]:
                candidate = offsets[i]+shift
                if j in offsets:
                    cycles.append(candidate-offsets[j])
                else:
                    offsets[j] = candidate
                    pending.append(j)
        if cycles:
            rank = max(rank, int(np.linalg.matrix_rank(cycles, tol=1e-7)))
    return rank


def structure_match(first, second, kind, ltol=0.2, stol=0.3, angle_tol=5):
    if first.composition.reduced_formula != second.composition.reduced_formula:
        return False
    if kind == 'space':
        return bool(StructureMatcher(ltol=ltol, stol=stol, angle_tol=angle_tol, scale=False).fit(first, second))
    # A common tall cell fixes the nonperiodic axis during lattice matching. The final
    # displacement threshold uses sqrt(A/N), so changing vacuum cannot improve the score.
    height = 10*max(*first.lattice.abc[:2], *second.lattice.abc[:2],
                    np.ptp(first.cart_coords[:, 2]), np.ptp(second.cart_coords[:, 2]), 4)
    a, b = layer_structure(first, height), layer_structure(second, height)
    # StructureMatcher's lattice mapping is implemented for full PBC. Use a tall
    # auxiliary cell only to find a mapping, then enforce physical 2D distances below.
    a = Structure(Lattice(a.lattice.matrix), a.species, a.frac_coords)
    b = Structure(Lattice(b.lattice.matrix), b.species, b.frac_coords)
    if len(a) < len(b):
        a, b = b, a
    matcher = StructureMatcher(ltol=ltol, stol=2, angle_tol=angle_tol, scale=False,
                               primitive_cell=False, attempt_supercell=True)
    mapped = matcher.get_s2_like_s1(a, b)
    if mapped is None or len(mapped) != len(a):
        return False
    if not np.allclose(np.abs(mapped.lattice.matrix[2]), np.abs(a.lattice.matrix[2]), atol=1e-6):
        return False
    delta = mapped.frac_coords-a.frac_coords
    delta[:, :2] -= np.round(delta[:, :2])
    delta[:, 2] -= delta[:, 2].mean()  # One rigid normal translation is allowed, not per-atom wrapping.
    # Fractional coordinates have already been mapped to the reference basis;
    # the physical frame may differ by a rigid orthogonal transformation.
    physical = delta @ a.lattice.matrix
    area = np.linalg.norm(np.cross(*a.lattice.matrix[:2]))
    return bool(np.max(np.linalg.norm(physical, axis=1)) <= stol*np.sqrt(area/len(a)))


@lru_cache(maxsize=1)
def _fingerprint_tools():
    from matminer.featurizers.composition import ElementProperty
    from matminer.featurizers.site import CrystalNNFingerprint
    scaler = json.loads((Path(__file__).parent/'benchmark_comp_scaler.json').read_text())
    return (ElementProperty.from_preset('magpie'), CrystalNNFingerprint.from_preset('ops'),
            np.array(scaler['means']), np.array(scaler['stds']))


@lru_cache(maxsize=8)
def _layer_fingerprint(height):
    from matminer.featurizers.site import CrystalNNFingerprint
    # Periodic auxiliary images close the Voronoi polyhedron. A >=12 Å gap
    # exceeds CrystalNN's chemical distance cutoff; normal-image bonds get zero weight.
    feature = CrystalNNFingerprint.from_preset('ops', search_cutoff=height+1)
    original = feature.cnn.get_nn_data
    def planar_data(structure, index):
        result = original(structure, index)
        if any(info['weight'] > 0 and abs(info['image'][2]) > 1e-8 for info in result.all_nninfo):
            raise ValueError('Auxiliary normal image entered the 2D chemical neighborhood')
        return result
    feature.cnn.get_nn_data = planar_data
    return feature


def fingerprints(structure, kind):
    composition, site, means, stds = _fingerprint_tools()
    if kind != 'space':
        structure = layer_structure(structure)
        height = float(max(24, np.ptp(structure.cart_coords[:, 2])+12))
        structure = layer_structure(structure, height)
        structure = Structure(Lattice(structure.lattice.matrix), structure.species, structure.frac_coords)
        site = _layer_fingerprint(height)
    comp = np.array(composition.featurize(structure.composition), dtype=float)
    local = np.array([site.featurize(structure, i) for i in range(len(structure))])
    structural = local.mean(axis=0)
    scaled = (comp-means)/stds
    scaled = np.nan_to_num(scaled, nan=0, posinf=0, neginf=0)
    if not np.isfinite(structural).all():
        raise ValueError('Nonfinite CrystalNN fingerprint')
    return structural, scaled


def coverage(generated, reference, attempted, reference_requested, structural_cutoff=0.4, composition_cutoff=10.0):
    """Preserve legacy independent nearest-distance coverage; bound distance-matrix memory."""
    if not generated or not reference:
        return {'status': 'unavailable', 'cov_recall': None, 'cov_precision': None}
    gs, gc = map(np.array, zip(*generated))
    rs, rc = map(np.array, zip(*reference))
    sr, cr = np.full(len(rs), np.inf), np.full(len(rs), np.inf)
    sp, cp = [], []
    for start in range(0, len(gs), 256):
        sd, cd = cdist(gs[start:start+256], rs), cdist(gc[start:start+256], rc)
        sr, cr = np.minimum(sr, sd.min(axis=0)), np.minimum(cr, cd.min(axis=0))
        sp.extend(sd.min(axis=1)); cp.extend(cd.min(axis=1))
    sp, cp = np.array(sp), np.array(cp)
    recalled = (sr <= structural_cutoff) & (cr <= composition_cutoff)
    precise = (sp <= structural_cutoff) & (cp <= composition_cutoff)
    return {'status': 'complete' if len(reference) == reference_requested else 'partial_reference_fingerprints',
            'cov_recall': float(recalled.mean()), 'cov_precision': float(precise.sum()/attempted),
            'cov_recall_lower_bound_over_all_reference': float(recalled.sum()/reference_requested),
            'amsd_recall': float(sr.mean()), 'amsd_precision': float(sp.mean()),
            'amcd_recall': float(cr.mean()), 'amcd_precision': float(cp.mean()),
            'generated_fingerprints': len(generated), 'reference_fingerprints': len(reference),
            'reference_requested': reference_requested, 'structural_cutoff': structural_cutoff,
            'composition_cutoff': composition_cutoff,
            'protocol': 'Independent nearest structure/composition distances, matching prior CGDiT/CDVAE; '
                        'COV-P uses all attempts. 2D uses auxiliary-Voronoi closure with zero normal-image '
                        'chemical weights, and is a layer-adapted metric.'}


def quantiles(values):
    values = np.array([x for x in values if x is not None and np.isfinite(x)], dtype=float)
    return {'count': len(values), **dict(zip(('min', 'p50', 'p95', 'p99', 'max'),
        [float(x) for x in np.quantile(values, [0, .5, .95, .99, 1])]))} if len(values) else {'count': 0}


def validity_summary(rows):
    total = len(rows)
    result = {'attempted': total, 'generated': sum(r.get('generated', False) for r in rows)}
    for name in ('legacy_structural_valid', 'periodic_distance_valid', 'radius_distance_valid',
                 'composition_valid', 'joint_valid', 'exact_target_group', 'unique', 'novel_against_reference'):
        known = [r[name] for r in rows if r.get(name) is not None]
        success = sum(bool(x) for x in known)
        result[name] = {'evaluated': len(known), 'passed': success, 'unknown': total-len(known),
                        'rate_over_all_attempts': success/total if total else None,
                        'rate_on_evaluated': success/len(known) if known else None}
    return result


def distribution_distances(generated, reference):
    result = {}
    for name in ('density_g_cm3', 'volume_per_atom_angstrom3', 'area_per_atom_angstrom2',
                 'layer_thickness_angstrom', 'num_elements'):
        a = [r[name] for r in generated if r.get('joint_valid') and r.get(name) is not None]
        b = [r[name] for r in reference if r.get(name) is not None]
        result[name] = {'wasserstein': float(wasserstein_distance(a, b)) if a and b else None,
                        'generated_count': len(a), 'reference_count': len(b)}
    return result


def property_distances(rows, assessments, reference, reference_requested):
    """Prior wdist_prop, using the same independent evaluator on both populations."""
    result = {}
    for name in ('formation_energy', 'band_gap', 'energy_above_hull'):
        generated = [assessments[r['index']][name] for r in rows if r.get('joint_valid')
                     and name in assessments.get(r['index'], {})]
        values = [r[name] for r in reference.values() if name in r]
        result[name] = {'wasserstein': float(wasserstein_distance(generated, values)) if generated and values else None,
                        'generated_count': len(generated), 'reference_count': len(values),
                        'reference_requested': reference_requested,
                        'status': 'complete' if generated and len(values) == reference_requested else
                                  ('partial' if generated and values else 'not_evaluated')}
    return result


def external_metrics(rows, requested, assessments, tolerances, hull_threshold=0.1):
    """Consume independently evaluated quantities, never requested targets as predictions."""
    known, hits, stable, sun, energy_evaluated, sun_evaluated = Counter(), Counter(), 0, 0, 0, 0
    errors, combined = defaultdict(list), 0
    for row in rows:
        assessment = assessments.get(row['index'], {})
        all_hit = bool(requested)
        for name, target in requested.items():
            value = assessment.get(name)
            if value is None or not np.isfinite(value):
                all_hit = False
                continue
            known[name] += 1
            error = abs(value-target)
            errors[name].append(error)
            hit = error <= tolerances[name]
            hits[name] += bool(hit and row.get('joint_valid'))
            all_hit &= hit
        combined += bool(all_hit and row.get('joint_valid'))
        hull = assessment.get('energy_above_hull')
        energy_known = (row.get('generated', True) and hull is not None and np.isfinite(hull)
                        and assessment.get('relaxation_success') is True)
        energy_evaluated += bool(energy_known)
        is_stable = energy_known and hull <= hull_threshold
        stable += bool(is_stable)
        relaxed_known = row.get('relaxed_unique') is not None and row.get('relaxed_novel_against_reference') is not None
        sun_evaluated += bool(energy_known and relaxed_known)
        sun += bool(is_stable and row.get('relaxed_joint_valid') and row.get('relaxed_unique')
                    and row.get('relaxed_novel_against_reference'))
    total = len(rows)
    return {'status': 'provided_independent_assessments' if assessments else 'not_evaluated',
            'properties': {name: {'evaluated': known[name], 'absolute_error': quantiles(errors[name]),
                'valid_hit_count': hits[name], 'valid_hit_rate_over_all_attempts': hits[name]/total if known[name] and total else None,
                'tolerance': tolerances[name]} for name in requested},
            'joint_valid_property_hits': combined,
            'joint_valid_property_hit_rate_over_all_attempts': combined/total
                if total and requested and all(known[name] for name in requested) else None,
            'stable_count': stable if energy_evaluated else None, 'sun_count': sun if sun_evaluated else None,
            'relaxed_energy_evaluated': energy_evaluated, 'sun_evaluated': sun_evaluated,
            'stable_rate_over_all_attempts': stable/total if energy_evaluated and total else None,
            'sun_rate_over_all_attempts': sun/total if sun_evaluated and total else None,
            'hull_threshold_ev_per_atom': hull_threshold}
