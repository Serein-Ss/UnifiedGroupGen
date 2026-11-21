"""Source-directory rename and current Tensor input contracts; never train."""
import json
from pathlib import Path

import pytest
import torch
import yaml

from unifiedgroupgen.common.configuration import resolve_path
from unifiedgroupgen.common.data_utils import target_from_record
from unifiedgroupgen.pl_data.unified.path import sample_path

ROOT=Path(__file__).resolve().parents[3]


def test_regression_fixtures_are_under_tests_instead_of_training_cache():
    folder=ROOT/'unifiedgroupgen/tests/fixtures'
    assert (folder/'current_training_records.json').is_file()
    assert resolve_path('unifiedgroupgen/tests/fixtures') == folder
    assert not (ROOT/'data/dataset/test_fixtures').exists()


@pytest.mark.parametrize('name',['mp_20','c2db_51'])
def test_source_profiles_point_directly_to_original_csvs(name):
    config = yaml.safe_load((ROOT/'configs/sources'/f'{name}.yaml').read_text(encoding='utf-8'))
    path = resolve_path(config['data_root'])
    assert path == ROOT/'data/src'/name
    assert all((path/f'{split}.csv').is_file() for split in ('train', 'val', 'test'))
    assert resolve_path(f'data/scr/{name}/train.csv') == ROOT/'data/scr'/name/'train.csv'


def test_external_property_predictor_cache_uses_src():
    config=yaml.safe_load((ROOT/'configs/property_predictor/default.yaml').read_text(encoding='utf-8'))
    cache=resolve_path(config['model_cache'])
    assert cache==ROOT/'pretrained/property_predictors/matgl'
    assert not (ROOT/'data/src/pretrained').exists()
    for name in ('MEGNet-MP-2018.6.1-Eform','MEGNet-MP-2019.4.1-BandGap-mfi'):
        assert (cache/name/'model.json').is_file() and (cache/name/'state.pt').is_file()


@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
def test_record_decodes_to_tensors_and_path_keeps_supervision_out_of_model_state(dataset):
    from unifiedgroupgen.pl_data.dataset import CrystalDataset
    from torch_geometric.data import Data
    path=ROOT/'unifiedgroupgen/tests/fixtures/current_training_records.json'
    record=json.loads(path.read_text(encoding='utf-8'))[dataset]['train']
    data=CrystalDataset([record],operation_compiler=True)
    assert isinstance(data[0],Data)
    target=data[0].record['_joint_target']
    assert isinstance(target.state.metric,torch.Tensor)
    assert target.state.coordinates.shape==(target.state.num_atoms,3)
    assert target.state.coordinates.dtype==torch.float32
    sample=sample_path(target,31,time=.7,allowed_elements=list(range(1,119)))
    assert torch.is_tensor(sample.state.metric) and torch.is_tensor(sample.state.coordinates)
    assert not hasattr(sample.state,'coordinate_anchor') and not hasattr(sample.state,'next_action')
    assert torch.is_tensor(sample.coordinate_anchor) and sample.coordinate_anchor.shape==sample.state.coordinates.shape
