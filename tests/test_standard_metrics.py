import json
import importlib.util
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from unifiedgroupgen.representation_audit import audit_record, audit_catalog
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, CompiledDescriptor
from unifiedgroupgen.standard_metrics import (coverage, external_metrics, fingerprints, geometry_metrics,
    periodic_connectivity, structure_match, composition_validity)

ROOT = Path(__file__).resolve().parents[1]


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_representation_catalog_and_real_degrees_of_freedom():
    rows = list(audit_catalog())
    failures = [row for row in rows if not row['passed']]
    assert not failures, failures[:10]
    assert {kind: len({r['group'] for r in rows if r['kind'] == kind}) for kind in ('space', 'layer', 'plane')} == {
        'space': 230, 'layer': 80, 'plane': 17}
    for kind, number, letter in [('space', 136, 'f'), ('layer', 1, 'a'), ('plane', 17, 'a')]:
        descriptor = Descriptor(kind, number, (OrbitSpec(letter, 8),))
        c = CompiledDescriptor(descriptor)
        state = np.arange(c.dof)*0.137+0.173
        record = {'id': kind, 'dataset': kind, 'descriptor': descriptor.to_dict(), 'state': state.tolist()}
        assert audit_record(record)['passed']
        record['state'] = record['state'][:-1]
        with pytest.raises(ValueError, match='wrong-dimensional'):
            audit_record(record)


def test_periodic_self_image_and_vacuum_independent_layer_metrics():
    bulk = Structure(Lattice.orthorhombic(.4, 3, 3), ['Si'], [[0, 0, 0]])
    report = geometry_metrics(bulk, 'space')
    assert report['legacy_structural_valid']
    assert not report['periodic_distance_valid']
    a = Structure(Lattice.orthorhombic(3, 3, 20), ['S', 'S'], [[0, 0, .45], [.5, .5, .55]])
    b = Structure(Lattice.orthorhombic(3, 3, 60), ['S', 'S'], [[0, 0, .25], [.5, .5, .25+2/60]])
    ga, gb = geometry_metrics(a, 'layer'), geometry_metrics(b, 'layer')
    for key in ('area_per_atom_angstrom2', 'layer_thickness_angstrom', 'minimum_pair_distance_angstrom'):
        assert ga[key] == pytest.approx(gb[key])
    assert ga['density_g_cm3'] is None
    assert structure_match(a, b, 'layer')
    c = b.copy(); c.translate_sites([1], [0, 0, .1], frac_coords=True, to_unit_cell=False)
    assert not structure_match(a, c, 'layer')
    supercell = a.copy(); supercell.make_supercell([2, 1, 1])
    assert structure_match(a, supercell, 'layer')
    first_fp, first_comp = fingerprints(a, 'layer')
    second_fp, second_comp = fingerprints(b, 'layer')
    assert np.allclose(first_fp, second_fp, atol=1e-6)
    assert np.allclose(first_comp, second_comp, atol=1e-6)


def test_connectivity_rank_uses_translation_cycles():
    assert periodic_connectivity(1, [0, 0], [0, 0], [[1, 0, 0], [0, 1, 0]]) == 2
    assert periodic_connectivity(2, [0], [1], [[0, 0, 0]]) == 0
    assert periodic_connectivity(1, [0], [0], [[1, 0, 0]]) == 1


def test_legacy_coverage_keeps_failures_in_denominator_and_separate_minima():
    generated = [(np.array([0.]), np.array([0.])), (np.array([1.]), np.array([1.]))]
    reference = [(np.array([0.]), np.array([1.])), (np.array([1.]), np.array([0.]))]
    report = coverage(generated, reference, attempted=4, reference_requested=3, structural_cutoff=.01, composition_cutoff=.01)
    assert report['cov_precision'] == .5
    assert report['cov_recall'] == 1
    assert report['cov_recall_lower_bound_over_all_reference'] == pytest.approx(2/3)
    assert report['status'] == 'partial_reference_fingerprints'


def test_real_smact_and_benchmark_fingerprint_dimensions():
    structure = Structure(Lattice.cubic(5.64), ['Na', 'Cl'], [[0, 0, 0], [.5, .5, .5]])
    assert composition_validity(structure)[0] is True
    structural, composition = fingerprints(structure, 'space')
    assert structural.shape == (61,) and composition.shape == (132,)
    assert np.isfinite(structural).all() and np.isfinite(composition).all()


def test_offline_evaluation_counts_failure_duplicate_and_missing_cif(tmp_path):
    script = load_script('evaluate_generation')
    structure = Structure(Lattice.cubic(5.64), ['Na', 'Cl'], [[0, 0, 0], [.5, .5, .5]])
    records = []
    for index in (0, 1, 2):
        records.append({'index': index, 'status': 'generated', 'validation': {'exact_target_group': True}})
        if index != 2:
            structure.to(filename=str(tmp_path/f'sample_{index:05d}.cif'))
    records.append({'index': 3, 'status': 'failed', 'reason': 'recorded failure'})
    samples = tmp_path/'samples.jsonl'
    samples.write_text('\n'.join(json.dumps(r) for r in records))
    reference = {'rows': [geometry_metrics(structure, 'space')], 'fingerprints': [], 'requested': 1}
    training = {structure.composition.reduced_formula: [structure]}
    rows, report = script.evaluate_samples(samples, reference, training, 'space', use_fingerprints=False)
    assert report['attempted'] == 4 and report['generated'] == 3
    assert rows[0]['unique'] and not rows[1]['unique']
    assert rows[0]['novel_against_reference'] is False
    assert rows[2]['joint_valid'] is False and rows[2]['evaluation_error']
    assert report['joint_valid']['rate_over_all_attempts'] == .5
    assert report['external']['status'] == 'not_evaluated'


def test_source_reencoding_compares_geometry_not_orbit_serialization():
    script = load_script('verify_representation')
    first = Descriptor('space', 1, (OrbitSpec('a', 14), OrbitSpec('a', 8)))
    second = Descriptor('space', 1, (OrbitSpec('a', 8), OrbitSpec('a', 14)))
    lattice = [0.1]*6
    a = {'descriptor': first.to_dict(), 'state': lattice+[.13, .27, .39, .56, .67, .81]}
    b = {'descriptor': second.to_dict(), 'state': lattice+[.56, .67, .81, .13, .27, .39]}
    assert script.compare_records(a, b)['source_reencode_passed']
    b['state'][-1] += .1
    assert not script.compare_records(a, b)['source_reencode_passed']


def test_external_property_hits_require_independent_values():
    rows = [{'index': 0, 'joint_valid': True, 'unique': True, 'novel_against_reference': True},
            {'index': 1, 'joint_valid': False}]
    missing = external_metrics(rows, {'band_gap': 1}, {}, {'band_gap': .2})
    assert missing['properties']['band_gap']['evaluated'] == 0
    assert missing['stable_rate_over_all_attempts'] is None
    measured = external_metrics(rows, {'band_gap': 1}, {0: {'band_gap': 1.1}}, {'band_gap': .2})
    assert measured['properties']['band_gap']['valid_hit_rate_over_all_attempts'] == .5
    assert measured['sun_rate_over_all_attempts'] is None
    from unifiedgroupgen.standard_metrics import property_distances
    distances = property_distances(rows, {0: {'band_gap': 1.2}}, {'ref': {'band_gap': 2.2}}, 1)
    assert distances['band_gap']['wasserstein'] == pytest.approx(1)
    assert distances['band_gap']['status'] == 'complete'
    assert distances['formation_energy']['wasserstein'] is None


def test_cli_exports_png_and_standard_metrics_with_real_fingerprints(tmp_path):
    import yaml
    script = load_script('evaluate_generation')
    descriptor = Descriptor('space', 1, (OrbitSpec('a', 11), OrbitSpec('a', 17)))
    record = {'id': 'reference', 'dataset': 'mp_20', 'descriptor': descriptor.to_dict(),
              'state': [0.0]*9+[.5, .5, .5], 'properties': {}}
    structure = script.record_structure(record)
    train, test = tmp_path/'train.jsonl', tmp_path/'test.jsonl'
    for path in (train, test):
        path.write_text(json.dumps(record)+'\n')
    config = tmp_path/'config.yaml'
    config.write_text(yaml.safe_dump({'train_files': [str(train)], 'test_files': [str(test)],
                                    'dataset_domains': {'mp_20': 0}}))
    structure.to(filename=str(tmp_path/'sample_00000.cif'))
    samples = tmp_path/'samples.jsonl'
    samples.write_text(json.dumps({'index': 0, 'status': 'generated', 'descriptor': descriptor.to_dict(),
                                  'validation': {'exact_target_group': False}})+'\n')
    artifacts = ROOT/'artifacts'
    artifacts.mkdir(exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='test_standard_metrics_', dir=artifacts) as folder:
        output, figures = Path(folder)/'metrics', Path(folder)/'figures'
        command = [sys.executable, str(ROOT/'scripts/evaluate_generation.py'), '--samples', str(samples),
                   '--dataset', 'mp_20', '--reference-config', str(config), '--output', str(output),
                   '--figures', str(figures)]
        result = subprocess.run(command, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stdout+'\n'+result.stderr
        report = json.loads((output/'summary.json').read_text())
        job = report['jobs']['mp_20_explicit']
        assert job['coverage']['cov_recall'] == 1 and job['coverage']['cov_precision'] == 1
        assert job['joint_valid']['passed'] == 1
        assert job['novel_against_reference']['passed'] == 0
        assert report['reference_identity'] == 'user_selected_references'
        assert [path.suffix for path in figures.iterdir()] == ['.png']
        # Exercise the server's manifest/checkpoint-bound entry and reject changed training references.
        import torch
        checkpoint = tmp_path/'checkpoint.pt'
        torch.save({'train_split_sha256': [script.file_hash(train)]}, checkpoint)
        run = tmp_path/'run'; run.mkdir()
        manifest = {'state': 'complete', 'identity': {'checkpoint_sha256': script.file_hash(checkpoint),
                    'test_sha256': [script.file_hash(test)]}, 'jobs': {'mp_20_unconditional': {
                    'directory': '..', 'output_sha256': {'samples.jsonl': script.file_hash(samples),
                    'sample_00000.cif': script.file_hash(tmp_path/'sample_00000.cif')}}}}
        (run/'pipeline.json').write_text(json.dumps(manifest))
        bound_command = [sys.executable, str(ROOT/'scripts/evaluate_generation.py'), '--run', str(run),
                         '--checkpoint', str(checkpoint), '--reference-config', str(config),
                         '--output', str(output.parent/'bound'), '--figures', str(figures.parent/'bound_figures')]
        bound = subprocess.run(bound_command, capture_output=True, text=True, timeout=120)
        assert bound.returncode == 0, bound.stdout+'\n'+bound.stderr
        assert json.loads((output.parent/'bound/summary.json').read_text())['reference_identity'] == 'checkpoint_and_test_bound'
        train.write_text(train.read_text()+'\n')
        changed = subprocess.run(bound_command, capture_output=True, text=True, timeout=60)
        assert changed.returncode != 0
        assert 'Novelty training references differ' in changed.stderr
        # Relaxed-file assessment follows its own uniqueness/novelty, and missing files
        # do not retroactively change the as-generated validity result.
        measured, external = script.evaluate_samples(samples,
            {'rows': [geometry_metrics(structure, 'space')], 'fingerprints': [], 'requested': 1}, {}, 'space',
            assessments={0: {'relaxed_cif': str(tmp_path/'sample_00000.cif'), 'relaxation_success': True,
                             'energy_above_hull': .05}}, use_fingerprints=False)
        assert external['external']['sun_count'] == 1
        assert measured[0]['relaxed_novel_against_reference']
