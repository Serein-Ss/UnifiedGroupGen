"""Hydra composition preserves audited definitions and actual execution settings."""
import hashlib
import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import yaml

from unifiedgroupgen import run
from unifiedgroupgen.common.configuration import compose_experiment, load_experiments, training_definition

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT/'unifiedgroupgen/tests/legacy/configs'
CATALOG = yaml.safe_load((ROOT/'configs/experiments.yaml').read_text())['experiments']


@pytest.mark.parametrize('name', CATALOG)
def test_all_experiments_preserve_definition_audit_and_effective_runtime(name):
    before = yaml.safe_load((FROZEN/'main15'/f'{name}.yaml').read_text())
    cfg = compose_experiment(name)
    actual = training_definition(cfg)
    assert actual == before
    audit = json.loads((ROOT/'data/dataset/audits/main15'/f'{name}.json').read_text())
    assert audit['passed']
    assert hashlib.sha256(json.dumps(actual, sort_keys=True).encode()).hexdigest() == audit['config_sha256']
    dataset = name.rsplit('_', 1)[0]
    runtime = yaml.safe_load((FROZEN/'runtime'/f'{dataset}.yaml').read_text())
    assert OmegaConf.to_container(cfg.runtime, resolve=True) == runtime


def test_catalog_matches_training_composition_and_preserves_outer_hydra_context():
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
        before = compose(config_name='generate')
        entries = load_experiments('configs/experiments.yaml')
        after = compose(config_name='generate')
    assert before == after
    assert list(entries) == list(CATALOG)
    for name, entry in entries.items():
        cfg = compose_experiment(name)
        assert entry['config'] == training_definition(cfg)
        assert entry['runtime'] == OmegaConf.to_container(cfg.runtime, resolve=True)


def test_functional_overrides_reach_model_loss_runtime_and_training(monkeypatch):
    cfg = compose_experiment('joint_all', overrides=[
        'conditioning=band_gap', 'model.proposal_layers=6', 'optim.learning_rate=0.0001',
        'loss.loss_weights.descriptor=0.5', 'loss.coordinate_length_scale=4',
        'train.validation_interval=10', 'runtime.batch_size=96', 'logging.mode=disabled'])
    assert set(training_definition(cfg)['properties']) == {'source_domain', 'band_gap'}
    assert training_definition(cfg)['properties']['band_gap']['per_domain'] is True
    captured = {}
    monkeypatch.setattr(run, 'train_args', lambda args, config, resolved: captured.update(args=args, config=config))
    monkeypatch.setattr(run.pl, 'seed_everything', lambda *a, **k: None)
    run.execute(cfg)
    assert captured['config']['learning_rate'] == 0.0001
    assert captured['config']['model']['proposal_layers'] == 6
    assert captured['config']['loss_weights']['descriptor'] == .5
    assert captured['args'].batch_size == 96
    assert captured['args'].validation_interval == 10
    assert captured['args'].coordinate_length_scale == 4
    assert captured['args'].wandb_mode == 'disabled'


def test_no_property_recipe_leaks_targets_into_unconditional_model():
    cfg = compose_experiment('joint_all', overrides=['conditioning=none'])
    config = training_definition(cfg)
    assert list(config['properties']) == ['source_domain']
    assert config['cond_dropout'] == 0 and not config['property_subset_training']
    assert config['composition_probability'] == config['observation_probability'] == 0


def test_default_runtime_and_nested_network_groups_compose():
    cfg = compose_experiment('mp20_u', overrides=['runtime=default', 'train.precision=fp32'])
    assert cfg.runtime.batch_size == 64
    assert cfg.runtime.checkpoint is True and cfg.runtime.edge_chunk_size == 4096
    assert training_definition(cfg)['precision'] == 'fp32'
    assert cfg.model.proposal_hidden == 256 and cfg.model.flow_hidden == 512
    assert not (ROOT/'configs/main15').exists()


def test_new_selector_and_catalog_override_need_no_python_dataset_registry(tmp_path):
    import shutil
    shutil.copytree(ROOT/'configs', tmp_path/'configs')
    (tmp_path/'configs/experiment/custom.yaml').write_text(
        '# @package _global_\ndefaults:\n  - override /data: c2db51\n'
        '  - override /conditioning: none\n  - override /runtime: rtx4090_layer\n'
        'experiment_name: custom\n', encoding='utf-8')
    (tmp_path/'configs/catalog.yaml').write_text(
        'experiments:\n  custom:\n    experiment: custom\n'
        '    overrides: [optim.learning_rate=0.0001, runtime.batch_size=96]\n', encoding='utf-8')
    entry = load_experiments('configs/catalog.yaml', tmp_path)['custom']
    assert entry['config']['experiment_id'] == 'custom'
    assert entry['config']['dataset'] == 'c2db_51'
    assert entry['config']['learning_rate'] == .0001
    assert entry['runtime']['batch_size'] == 96


def test_descriptor_config_can_be_replaced_independently_of_orbitflow(tmp_path):
    import shutil
    shutil.copytree(ROOT/'configs', tmp_path/'configs')
    (tmp_path/'configs/model/descriptor/small.yaml').write_text(
        'proposal_hidden: 64\nproposal_layers: 2\nheads: 4\n', encoding='utf-8')
    cfg = compose_experiment('mp20_u', tmp_path, ['model/descriptor@model=small'])
    assert cfg.model.proposal_hidden == 64 and cfg.model.proposal_layers == 2
    assert cfg.model.flow_hidden == 512 and cfg.model.flow_layers == 6


def test_native_resume_leaves_unspecified_runtime_in_checkpoint(capsys):
    cfg = compose_experiment('mp20_u', overrides=['task=plan', 'resume=last.ckpt'])
    run.execute(cfg)
    assert all(value is None for value in cfg.runtime.values())
    assert 'PLAN ONLY' in capsys.readouterr().out


def test_native_resume_passes_only_explicit_runtime_overrides(capsys):
    from hydra.core.hydra_config import HydraConfig
    state = HydraConfig.instance().cfg
    try:
        with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
            full = compose(config_name='run', return_hydra_config=True, overrides=[
                'task=plan', 'resume=last.ckpt', 'runtime.batch_size=96',
                'loss.coordinate_length_scale=4', 'train.validation_interval=10'])
        HydraConfig.instance().set_config(full)
        cfg = OmegaConf.masked_copy(full, [key for key in full if key != 'hydra'])
        run.execute(cfg)
        assert cfg.runtime.batch_size == 96
        assert cfg.runtime.coordinate_length_scale == 4
        assert cfg.runtime.validation_interval == 10
        assert cfg.runtime.checkpoint is None and cfg.runtime.edge_budget is None
    finally:
        HydraConfig.instance().cfg = state
    assert 'PLAN ONLY' in capsys.readouterr().out
