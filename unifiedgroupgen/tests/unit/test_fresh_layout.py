"""Configuration-driven reproduction without archived command dependencies."""
import json
from pathlib import Path
import zipfile

import pytest

from unifiedgroupgen.experiment_config import compose_experiment, training_definition
from unifiedgroupgen.common.configuration import resolve_path
from scripts.data_tools.bundle import pack, unpack
from unifiedgroupgen import run


def test_fresh_plan_comes_from_current_yaml_only(capsys):
    cfg = compose_experiment('full_mp20_eqv2_dng', overrides=[
        'task=plan', 'runtime=local', 'runtime.batch_size=13',
    ])
    definition = training_definition(cfg)
    run.execute(cfg)
    assert cfg.runtime.batch_size == 13
    assert definition['source_configs'] == {'mp_20': 'configs/sources/mp_20.yaml'}
    assert cfg.resume is None and cfg.initialize_from is None and definition['data_audit'] is None
    assert 'PLAN ONLY' in capsys.readouterr().out


def test_old_paths_are_never_rewritten(tmp_path):
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/paths.yaml').write_text('mounts: {logical_inputs: data/prepared}', encoding='utf-8')
    assert resolve_path('logical_inputs/train.jsonl', tmp_path) == tmp_path/'logical_inputs/train.jsonl'


def test_configuration_uses_active_source_with_workspace_data(monkeypatch, tmp_path):
    from unifiedgroupgen.common import configuration
    source, workspace = tmp_path/'source', tmp_path/'workspace'
    monkeypatch.setattr(configuration, 'package_parent', lambda: source)
    monkeypatch.setattr(configuration, 'project_root', lambda: workspace)
    assert configuration.configuration_root() == workspace
    (source/'configs').mkdir(parents=True)
    (source/'configs/run.yaml').touch()
    assert configuration.configuration_root() == source
    assert configuration.resolve_path('data/src') == workspace/'data/src'


def test_both_workflow_directories_and_fixed_experiment_code_are_absent():
    package = Path(__file__).resolve().parents[2]
    assert not (package/'pl_data/workflows').exists()
    assert not (package/'evaluation/workflows').exists()
    assert not (package/'common/experiments.py').exists()


def test_new_layer_dataset_is_not_inferred_from_a_special_dataset_name():
    import torch
    from unifiedgroupgen.group_constraints.group_prior import fit_group_prior, sample_group
    records=[dict(id='a',dataset='mp_20',descriptor=dict(kind='layer',number=78))]
    prior=fit_group_prior(records)
    renamed=fit_group_prior([{**row,'dataset':'arbitrary_new_layers'} for row in records])
    assert prior==renamed
    spec=sample_group(prior,torch.Generator().manual_seed(7))
    assert spec.kind=='layer' and spec.number==78


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
