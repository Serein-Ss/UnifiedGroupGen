import json
from pathlib import Path
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from unifiedgroupgen.data import parse_cif, space_record, convert_row
from unifiedgroupgen.source_integrity import compare_source, expanded_structure, reduced_counts


@pytest.mark.parametrize('identifier', ['mp-1281111', 'mp-997160', 'mp-570459'])
def test_raw_cif_composition_survives_orbit_encoding(identifier):
    rows = json.loads((Path(__file__).parent/'fixtures/source_integrity_regressions.json').read_text(encoding='utf-8'))
    row = rows[identifier]
    source = parse_cif(row['cif'], preserve_fractional=True)
    descriptor, state, _ = space_record(source, .001)
    encoded = expanded_structure({'descriptor': descriptor.to_dict(), 'state': state.tolist()})
    check = compare_source(source, encoded, 'space', .001)
    assert check['composition_conserved'] and check['geometry_equivalent'], check
    config = {'dataset': 'mp_20', 'kind': 'space', 'symmetry_tolerance': .001}
    converted = convert_row((0, row, config, 'fixture.csv'))
    assert 'record' in converted, converted
    assert converted['record']['provenance']['source_integrity']['geometry_equivalent']


def test_cell_size_change_is_allowed_but_species_or_density_changes_are_not():
    primitive = Structure(Lattice.cubic(4), ['Na', 'Cl'], [[0, 0, 0], [.5, .5, .5]])
    supercell = primitive.copy()
    supercell.make_supercell([2, 1, 1])
    assert compare_source(primitive, supercell, 'space', .001)['geometry_equivalent']
    wrong = supercell.copy()
    wrong.replace(0, 'Li')
    assert not compare_source(primitive, wrong, 'space', .001)['composition_conserved']
    wrong = primitive.copy()
    wrong.scale_lattice(wrong.volume * 1.02)
    assert not compare_source(primitive, wrong, 'space', .001)['geometry_equivalent']


def test_layer_equivalence_ignores_vacuum_but_not_normal_displacements():
    first = Structure(Lattice.hexagonal(3.2, 30), ['Mo', 'S', 'S'],
                      [[0, 0, .5], [1/3, 2/3, .45], [1/3, 2/3, .55]])
    second = Structure(Lattice.hexagonal(3.2, 80), first.species,
                       np.column_stack([first.frac_coords[:, :2], (first.cart_coords[:, 2] - 15)/80 + .5]))
    check = compare_source(first, second, 'layer', .001)
    assert check['geometry_equivalent'] and check['max_displacement_angstrom'] < 1e-6, check
    second.translate_sites([1], [0, 0, .2], frac_coords=False)
    assert not compare_source(first, second, 'layer', .001)['geometry_equivalent']


def test_old_corrupt_wyckoff_assignment_is_rejected():
    rows = json.loads((Path(__file__).parent/'fixtures/source_integrity_regressions.json').read_text(encoding='utf-8'))
    source = parse_cif(rows['mp-997160']['cif'], preserve_fractional=True)
    # Historical cached description had Ag on 6c, doubling the Ag/Mn ratio.
    assert reduced_counts(source.atomic_numbers) != reduced_counts([25]*3 + [47]*6 + [8]*6)


def test_training_gate_detects_edits_after_source_certification():
    from copy import deepcopy
    from unifiedgroupgen.preflight import inspect_records
    rows = json.loads((Path(__file__).parent/'fixtures/source_integrity_regressions.json').read_text(encoding='utf-8'))
    config = {'dataset': 'mp_20', 'kind': 'space', 'symmetry_tolerance': .001}
    first = convert_row((0, rows['mp-1281111'], config, 'fixture.csv'))['record']
    second = convert_row((1, rows['mp-997160'], config, 'fixture.csv'))['record']
    gate = {'require_source_integrity': True, 'allowed_elements': [3, 8, 24, 25, 47]}
    report = inspect_records({'train': [first], 'val': [second]}, gate)
    assert not any('source-CIF' in r.get('reason', '') for r in report['errors'])
    tampered = deepcopy(first)
    tampered['state'][0] += .1
    report = inspect_records({'train': [tampered], 'val': [second]}, gate)
    assert any('source-CIF' in r.get('reason', '') for r in report['errors'])


@pytest.mark.parametrize('identifier', ['1AlP-1', '4ClIn-1'])
def test_layer_centering_and_auxiliary_vacuum_do_not_create_false_failures(identifier):
    data = json.loads((Path(__file__).parent/'fixtures/layer_source_integrity_regressions.json').read_text(encoding='utf-8'))[identifier]
    source = parse_cif(data['cif'], preserve_fractional=True)
    result = compare_source(source, expanded_structure(data), 'layer', .1)
    assert result['composition_conserved'] and result['geometry_equivalent'], result


@pytest.mark.parametrize('identifier', ['mp-621925', 'mp-1078353'])
def test_idealization_outside_declared_integrity_budget_is_quarantined(identifier):
    data = json.loads((Path(__file__).parent/'fixtures/source_integrity_regressions.json').read_text(encoding='utf-8'))[identifier]
    result = convert_row((0, data, {'dataset': 'mpts_52', 'kind': 'space',
                                  'symmetry_tolerance': .01}, 'fixture.csv'))
    assert 'rejection' in result and 'Source integrity failed' in result['rejection']['reason']
