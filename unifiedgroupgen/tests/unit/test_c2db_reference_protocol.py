"""C2DB reference tolerance changes labels, not sources or the single-pass rules."""
import hashlib
from pathlib import Path

import pandas as pd
import pytest
import torch

from unifiedgroupgen.common.configuration import load_yaml
from unifiedgroupgen.common import data_utils as data
from unifiedgroupgen.pl_data.cache import ensure_tensor_cache
from unifiedgroupgen.group_constraints.compiler import compile_descriptor
from unifiedgroupgen.group_constraints.encoding import operation_residual


ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(scope='module')
def examples():
    wanted = {'2HKO-2', '2HKTe-1', '1CoCuNiTe2-1', '1BrISSc-1'}
    frame = pd.read_csv(ROOT / 'data/src/c2db_51/train.csv', nrows=4025)
    return {row['uid']: row.to_dict() for _, row in frame.iterrows() if row['uid'] in wanted}


def test_c2db_source_uses_reference_tolerance():
    config = load_yaml(ROOT / 'configs/sources/c2db_51.yaml')
    assert config['symmetry_tolerance'] == .1
    assert config['symmetry_protocol'] == data.SYMMETRY_PROTOCOL
    assert config['cif_fractional_tolerance'] == 0 and config['niggli'] is False
    assert config['layer_group_policy'] == 'identify'


@pytest.mark.parametrize('identifier,number', [('2HKO-2', 64), ('2HKTe-1', 64),
                                              ('1CoCuNiTe2-1', 69), ('1BrISSc-1', 13)])
def test_reference_examples_keep_species_and_close_under_group(examples, identifier, number):
    config = load_yaml(ROOT / 'configs/sources/c2db_51.yaml')
    record = data.convert_row((0, examples[identifier], config, 'fixture.csv'))
    assert 'record' in record, record
    record = record['record']
    assert record['descriptor']['number'] == number
    provenance = record['provenance']
    assert provenance['source_integrity']['composition_conserved']
    assert provenance['source_integrity']['max_displacement_angstrom'] <= .1 + 1e-8
    assert provenance['niggli_reduction'] is False and provenance['cif_fractional_rounding_disabled']
    compiled = compile_descriptor(data.Descriptor.from_dict(record['descriptor']))
    expanded = compiled.expand(torch.tensor(record['state'], dtype=torch.float64))
    assert operation_residual(expanded['coordinates'].numpy(), compiled.elements,
        data.group_table('layer', number)[0].ops, compiled.periodic, expanded['lattice'].numpy()) < 1e-5


def test_reference_identification_uses_original_input_once(examples, monkeypatch):
    from pyxtal import pyxtal
    config = load_yaml(ROOT / 'configs/sources/c2db_51.yaml')
    native = data.spglib.get_layergroup
    calls = []
    def identify(cell, **kwargs):
        calls.append((cell, kwargs))
        return native(cell, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail('Reference protocol must not invoke PyXtal refinement')
    monkeypatch.setattr(data.spglib, 'get_layergroup', identify)
    monkeypatch.setattr(pyxtal, 'from_seed', forbidden)
    result = data.convert_row((0, examples['2HKO-2'], config, 'fixture.csv'))
    assert 'record' in result, result
    assert len(calls) == 1 and calls[0][1] == dict(aperiodic_dir=0, symprec=.1)


def test_reference_cache_does_not_select_or_modify_strict_cache(tmp_path, examples):
    torch.set_num_threads(1)
    folder = tmp_path / 'csv'
    folder.mkdir()
    source = folder / 'train.csv'
    pd.DataFrame([examples['2HKO-2']]).to_csv(source, index=False)
    original_source = hashlib.sha256(source.read_bytes()).hexdigest()
    config = {**load_yaml(ROOT / 'configs/sources/c2db_51.yaml'), 'data_root': str(folder)}
    strict = {**config, 'symmetry_tolerance': .01}
    old, first, _ = ensure_tensor_cache(strict, 'train', tmp_path)
    original_cache = hashlib.sha256(old.read_bytes()).hexdigest()
    new, second, hit = ensure_tensor_cache(config, 'train', tmp_path)
    assert not hit and new != old
    assert first['records'][0]['descriptor']['number'] == 9
    assert second['records'][0]['descriptor']['number'] == 64
    assert hashlib.sha256(old.read_bytes()).hexdigest() == original_cache
    assert hashlib.sha256(source.read_bytes()).hexdigest() == original_source
    assert ensure_tensor_cache(config, 'train', tmp_path)[2]
