from pathlib import Path
import numpy as np
import pandas as pd
import torch
from unifiedgroupgen.data import parse_cif, space_record, layer_record, convert_row, identified_layer_record
from unifiedgroupgen.symmetry import compile_descriptor
from unifiedgroupgen.evaluation import validate


DATA = Path(__file__).resolve().parents[2]/'data'
if not DATA.exists():
    DATA = Path(__file__).resolve().parents[1]/'raw_data'


def test_actual_mp20_and_mpts52_records_reconstruct_with_declared_group():
    for dataset in ('mp_20', 'mpts_52'):
        row = pd.read_csv(DATA/dataset/'train.csv', nrows=1).iloc[0]
        structure = parse_cif(row.cif)
        descriptor, state, _ = space_record(structure)
        compiled = compile_descriptor(descriptor)
        report = validate(compiled, compiled.expand(torch.tensor(state, dtype=torch.float64)))
        assert report['group_operation_closure'] and report['metric_residual']<1e-5


def test_actual_c2db_cartesian_export_and_layer_group():
    row = pd.read_csv(DATA/'c2db_51'/'train.csv', nrows=1).iloc[0]
    structure = parse_cif(row.cif, 'cartesian_in_fractional_fields')
    assert abs(np.ptp(structure.cart_coords[:, 2])-row.thickness)<1e-6
    descriptor, state, _ = layer_record(structure, int(row.lgnum), tolerance=0.002)
    assert descriptor.kind == 'layer' and descriptor.number == 37
    compiled = compile_descriptor(descriptor)
    report = validate(compiled, compiled.expand(torch.tensor(state, dtype=torch.float64)))
    assert report['group_operation_closure']
    assert report['recovered_layer_group'] == 37 and report['exact_layer_group']


def test_layer_residual_uses_physical_cell_lengths():
    from pymatgen.core.operations import SymmOp
    from unifiedgroupgen.data import operation_residual
    coords = np.array([[0.006, 0, 0]])
    elements = np.array([14])
    operations = [SymmOp.from_rotation_and_translation(-np.eye(3))]
    periodic = [True, True, False]
    fractional = operation_residual(coords, elements, operations, periodic)
    physical = operation_residual(coords, elements, operations, periodic, np.diag([100, 1, 1]))
    assert np.isclose(fractional, 0.012)
    assert np.isclose(physical, 1.2)


def test_rank_one_layer_origin_accepts_perturbations_within_physical_tolerance():
    from pymatgen.core.operations import SymmOp
    from unifiedgroupgen.data import _layer_origin_candidates, operation_residual
    coords = np.array([[0.2, 0.1, 0.2], [0.201, 0.4, -0.2]])
    elements = np.array([14, 14])
    operations = [SymmOp.from_rotation_and_translation(np.diag([1, -1, -1]))]
    metric = np.diag([5.0, 5.0, 1.0])
    origins = list(_layer_origin_candidates(coords, elements, operations, 0.02, metric))
    residuals = []
    for origin in origins:
        adjusted = coords.copy(); adjusted[:, :2] -= origin
        residuals.append(operation_residual(adjusted, elements, operations, [True, True, False], metric))
    assert residuals and min(residuals) <= 0.02


def test_mpts52_sensitive_record_at_declared_formal_tolerance():
    # At 0.001 this input terminates the installed spglib native process.
    # The formal MPTS52 configuration uses a uniform 0.01, never per-row P1 fallback.
    row = pd.read_csv(DATA/'mpts_52'/'train.csv', nrows=3112).iloc[3111]
    assert row.material_id == 'mp-568324'
    descriptor, state, provenance = space_record(parse_cif(row.cif), tolerance=0.01)
    assert descriptor.number == 140
    compiled = compile_descriptor(descriptor)
    assert validate(compiled, compiled.expand(torch.tensor(state, dtype=torch.float64)))['group_operation_closure']
    assert provenance['symmetry_tolerance_angstrom'] == 0.01


def test_verified_unknown_layer_label_is_identified_without_dropping_structure():
    from pymatgen.io.cif import CifWriter
    row = pd.read_csv(DATA/'c2db_51'/'train.csv', nrows=1).iloc[0].to_dict()
    structure = parse_cif(row['cif'], 'cartesian_in_fractional_fields')
    row.update({'cif': str(CifWriter(structure)), 'lgnum': -1, 'official_structure_sha256': 'unit-test-fixture'})
    config = {'dataset': 'c2db_51', 'kind': 'layer', 'coordinate_mode': 'fractional',
              'layer_group_policy': 'identify', 'symmetry_tolerance': 0.05}
    converted = convert_row((0, row, config, 'synthetic_recovered.csv'))
    assert 'record' in converted
    record = converted['record']
    assert record['descriptor']['number'] == 37
    assert record['provenance']['declared_layer_group'] == -1
    assert not record['provenance']['cell_orientation_inferred']
    del row['official_structure_sha256']
    rejected = convert_row((0, row, config, 'unverified.csv'))
    assert 'verified recovered structure' in rejected['rejection']['reason']


def test_layer_identification_failure_never_assigns_a_fallback(monkeypatch):
    import pytest
    import unifiedgroupgen.data as data
    row = pd.read_csv(DATA/'c2db_51'/'train.csv', nrows=1).iloc[0]
    structure = parse_cif(row.cif, 'cartesian_in_fractional_fields')
    monkeypatch.setattr(data.spglib, 'get_layergroup', lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match='no fallback'):
        identified_layer_record(structure)


def test_cartesian_export_parser_preserves_near_rational_lengths():
    import pytest
    row = pd.read_csv(DATA/'c2db_51'/'train.csv', nrows=3968).iloc[3967]
    assert row.uid == '2IIrS2-1'
    # CifParser's default fraction snapping changed this Cartesian length by 1.3013e-5 Angstrom.
    structure = parse_cif(row.cif, 'cartesian_in_fractional_fields')
    assert structure.cart_coords[1, 1] == pytest.approx(0.66667968, abs=1e-10)


def test_recovered_fractional_coordinates_can_be_preserved_before_layer_identification():
    import pytest
    from pymatgen.core import Lattice, Structure
    from pymatgen.io.cif import CifWriter
    original = Structure(Lattice.cubic(4), ['Si'], [[0.33332, 0.17, 0.48]])
    cif = str(CifWriter(original))
    preserved = parse_cif(cif, preserve_fractional=True)
    assert preserved.frac_coords[0, 0] == pytest.approx(0.33332, abs=1e-12)
    with pytest.warns(UserWarning, match='rounded to ideal'):
        snapped = parse_cif(cif)
    assert abs(snapped.frac_coords[0, 0]-preserved.frac_coords[0, 0]) > 1e-5


def test_actual_layer72_nonideal_standard_positions_fit_the_identified_group():
    import json
    row = json.loads((Path(__file__).parent/'fixtures/c2db_layer72_unidealized.json').read_text(encoding='utf-8'))
    config = {'dataset': 'c2db_51', 'kind': 'layer', 'coordinate_mode': 'fractional',
              'layer_group_policy': 'identify', 'symmetry_tolerance': 0.1}
    result = convert_row((3268, row, config, 'verified_recovered_train.csv'))
    assert 'record' in result
    record = result['record']
    from unifiedgroupgen.symmetry import Descriptor
    compiled = compile_descriptor(Descriptor.from_dict(record['descriptor']))
    report = validate(compiled, compiled.expand(torch.tensor(record['state'], dtype=torch.float64)))
    assert record['descriptor']['number'] == 72
    assert report['group_operation_closure'] and report['exact_layer_group']
    assert record['provenance']['native_std_chart_idealized']
    assert 0 < record['provenance']['chart_fit_max_angstrom'] < 0.02
    assert record['provenance']['std_position_total_max_upper_bound_angstrom'] < 0.1
    assert record['provenance']['source_atoms'] == 10 and compiled.num_atoms == 5


def test_layer_orbit_fit_rejects_positions_outside_the_configured_tolerance(monkeypatch):
    import json
    import pytest
    from types import SimpleNamespace
    import unifiedgroupgen.data as data
    row = json.loads((Path(__file__).parent/'fixtures/c2db_layer72_unidealized.json').read_text(encoding='utf-8'))
    structure = parse_cif(row['cif'], preserve_fractional=True)
    native = data.spglib.get_layergroup((structure.lattice.matrix, structure.frac_coords, structure.atomic_numbers),
                                       aperiodic_dir=2, symprec=0.1)
    altered = SimpleNamespace(**{key: getattr(native, key) for key in ('std_lattice', 'std_types', 'std_positions',
                               'number', 'rotations', 'translations', 'transformation_matrix', 'origin_shift',
                               'std_rotation_matrix')})
    altered.std_positions = altered.std_positions.copy()
    altered.std_positions[0, :2] += [0.12, 0.19]
    monkeypatch.setattr(data.spglib, 'get_layergroup', lambda *args, **kwargs: altered)
    with pytest.raises(ValueError, match='could not be matched'):
        identified_layer_record(structure, tolerance=0.1, declared_number=72)
