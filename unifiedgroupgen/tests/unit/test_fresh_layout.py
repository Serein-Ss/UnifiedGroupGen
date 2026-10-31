"""Configuration-driven reproduction without archived command dependencies."""
import json
from pathlib import Path
import zipfile

from omegaconf import OmegaConf
import pytest

from unifiedgroupgen.common.configuration import load_experiments
from unifiedgroupgen.common.configuration import resolve_path
from scripts.data_tools.bundle import pack, unpack
from unifiedgroupgen import run


def test_arbitrary_experiment_and_runtime_are_yaml_driven(tmp_path):
    (tmp_path/'model.yaml').write_text('dataset: arbitrary\nproperties: {}\n', encoding='utf-8')
    (tmp_path/'runtime.yaml').write_text('batch_size: 13\nedge_budget: 900\n', encoding='utf-8')
    catalog = tmp_path/'experiments.yaml'
    catalog.write_text('experiments:\n  new_chemistry_ablation:\n    config: model.yaml\n    runtime: runtime.yaml\n', encoding='utf-8')
    entries = load_experiments(catalog, tmp_path)
    assert list(entries) == ['new_chemistry_ablation']
    assert entries['new_chemistry_ablation']['runtime']['batch_size'] == 13
    assert entries['new_chemistry_ablation']['config']['dataset'] == 'arbitrary'
    assert sorted(p.name for p in tmp_path.iterdir()) == ['experiments.yaml', 'model.yaml', 'runtime.yaml']


def test_training_does_not_infer_runtime_from_experiment_name(capsys):
    cfg = OmegaConf.create(dict(task='plan', experiment_name='completely_new', device='cpu', seed=42,
        threads=1, epochs=None, output=None, resume=None, initialize_from=None,
        wandb=dict(mode='disabled', project='fixture'),
        experiment=dict(dataset='new', experiment_id='not_a_known_dataset', properties={},
                        runtime_profile=dict(batch_size=13, edge_budget=900, coordinate_length_scale=3.)),
        runtime=dict(batch_size=None, edge_budget=None, checkpoint=None, edge_chunk_size=None,
                     validation_interval=None, coordinate_length_scale=None)))
    run.execute(cfg)
    assert cfg.runtime.batch_size == 13 and cfg.runtime.edge_budget == 900
    assert cfg.runtime.coordinate_length_scale == 3.
    assert 'PLAN ONLY' in capsys.readouterr().out


def test_location_mapping_is_explicit_not_a_builtin_dataset_alias(tmp_path):
    assert resolve_path('artifacts/full_verified_v1/train.jsonl', tmp_path) == tmp_path/'artifacts/full_verified_v1/train.jsonl'
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/paths.yaml').write_text('mounts:\n  logical_inputs: data/prepared\n', encoding='utf-8')
    assert resolve_path('logical_inputs/train.jsonl', tmp_path) == tmp_path/'data/prepared/train.jsonl'


def test_both_workflow_directories_and_fixed_experiment_code_are_absent():
    package = Path(__file__).resolve().parents[2]
    assert not (package/'pl_data/workflows').exists()
    assert not (package/'evaluation/workflows').exists()
    assert not (package/'common/experiments.py').exists()


def test_new_layer_dataset_is_not_inferred_from_a_special_dataset_name(tmp_path):
    from scripts.evaluate_models import build_jobs
    records=tmp_path/'records.jsonl'
    records.write_text(json.dumps({'id': 'a', 'dataset': 'new_layers', 'properties': {},
        'descriptor': {'kind': 'layer'}})+'\n', encoding='utf-8')
    config={'dataset': 'new_layers', 'properties': {}, 'test_files': [str(records)], 'train_files': [str(records)]}
    jobs=build_jobs(config, 1, 1, 1, 1, 142)
    assert all(job['kind']=='layer' for job in jobs if job['task']!='components')


def test_bundle_round_trip_and_changed_input_are_not_overwritten(tmp_path):
    source = tmp_path/'source';source.mkdir()
    (source/'record.jsonl').write_text('{"id":"a"}\n', encoding='utf-8')
    archive=tmp_path/'data.zip'
    manifest=pack(source, ['record.jsonl'], archive)
    target=tmp_path/'target'
    assert unpack(archive, target) == manifest
    assert (target/'record.jsonl').read_bytes() == (source/'record.jsonl').read_bytes()
    assert unpack(archive, target) == manifest
    (target/'record.jsonl').write_text('changed', encoding='utf-8')
    with pytest.raises(ValueError, match='different existing input'):
        unpack(archive, target)
    assert (target/'record.jsonl').read_text() == 'changed'


@pytest.mark.parametrize('name', ['../escape', '/absolute', 'C:/drive', 'folder\\escape'])
def test_bundle_rejects_unsafe_members_before_writing(tmp_path, name):
    archive=tmp_path/'unsafe.zip'
    with zipfile.ZipFile(archive, 'w') as source:
        source.writestr(name, b'x')
        source.writestr('bundle.manifest.json', json.dumps({name: {'sha256': 'invalid', 'bytes': 1}}))
    target=tmp_path/'target'
    with pytest.raises(ValueError, match='Unsafe bundle path|contents differ'):
        unpack(archive, target)
    assert not target.exists()
