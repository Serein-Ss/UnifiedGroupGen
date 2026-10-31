"""Contracts for the layered package, Hydra entry and interchangeable backbones."""
import copy
import json
from pathlib import Path
import random
import numpy as np

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
import pytest
import torch

from unifiedgroupgen.run import load_pretrained_models
from unifiedgroupgen.tests.legacy.paths import resolve_path
from unifiedgroupgen.pl_modules.factory import build_models

from unifiedgroupgen import run

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def preserve_process_rng_and_threads():
    state = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.get_num_threads())
    yield
    random.setstate(state[0]); np.random.set_state(state[1]); torch.set_rng_state(state[2])
    torch.set_num_threads(state[3])


def configuration(name='mp20_u'):
    with initialize_config_dir(config_dir=str(ROOT/'configs'), version_base='1.3'):
        return compose(config_name='run', overrides=['experiment='+name, 'task=plan', 'device=cpu'])


@pytest.mark.parametrize('name,batch,edges', [('mp20_u',256,400000),('c2db51_all',768,200000),('joint_all',256,400000)])
def test_hydra_plan_preserves_runtime_and_experiment(name, batch, edges, capsys):
    cfg = configuration(name)
    run.execute(cfg)
    assert cfg.experiment.experiment_id == name
    assert cfg.runtime.batch_size == batch and cfg.runtime.edge_budget == edges
    assert cfg.runtime.coordinate_length_scale == 3.
    assert cfg.runtime.checkpoint is False
    assert 'PLAN ONLY' in capsys.readouterr().out


def test_hydra_passes_composed_yaml_without_launching_training(monkeypatch):
    cfg=configuration('joint_u');cfg.task='train'
    received={}
    monkeypatch.setattr(run,'train_args',lambda args, config, resolved: received.update(args=args,config=config))
    run.execute(cfg)
    assert received['args'].trainer == 'lightning'
    assert received['config']['properties']['source_domain']['context']
    assert received['config']['composition_probability'] == 0


def test_resume_and_pretraining_are_mutually_exclusive():
    cfg=configuration();cfg.resume='last.ckpt';cfg.initialize_from='model.pt'
    with pytest.raises(ValueError,match='Choose one'):run.execute(cfg)


def small_config():
    return {'groups':[['space',2]],'properties':{},'max_atoms':8,'max_orbits':4,
            'model':{'hidden':16,'heads':4,'proposal_layers':1,'flow_layers':1,'frequencies':2}}


def test_explicit_default_backbones_preserve_weight_names_and_initial_values():
    config=small_config();torch.manual_seed(142);before=build_models(config,'cpu')
    config=copy.deepcopy(config)
    config['model'].update(
        descriptor_backbone={'_target_':'unifiedgroupgen.pl_modules.backbones.transformer.build_transformer'},
        flow_backbone={'_target_':'unifiedgroupgen.pl_modules.backbones.message_passing.MessageLayer'})
    torch.manual_seed(142);after=build_models(config,'cpu')
    for a,b in zip(before,after):
        assert a.state_dict().keys() == b.state_dict().keys()
        for key,value in a.state_dict().items():torch.testing.assert_close(value,b.state_dict()[key],rtol=0,atol=0)


@pytest.mark.parametrize('native',[False,True])
def test_initialize_supports_inference_and_native_weights_only(tmp_path,native):
    torch.manual_seed(142);before=build_models(small_config(),'cpu')
    if native:
        content={'ugg_state':{},'state_dict':{prefix+'.'+k:v for prefix,m in zip(('proposal','flow'),before)
                                            for k,v in m.state_dict().items()}}
    else:content={'proposal':before[0].state_dict(),'flow':before[1].state_dict()}
    path=tmp_path/'weights.pt';torch.save(content,path)
    after=build_models(small_config(),'cpu');load_pretrained_models(*after,path)
    for a,b in zip(before,after):
        for k,v in a.state_dict().items():torch.testing.assert_close(v,b.state_dict()[k],rtol=0,atol=0)


def test_source_aliases_resolve_to_actual_layered_implementation():
    assert resolve_path('src/unifiedgroupgen/symmetry.py') == ROOT/'unifiedgroupgen/group_constraints/compiler.py'
    assert resolve_path('unifiedgroupgen/benchmark_comp_scaler.json').is_file()
    assert resolve_path('scripts') == ROOT/'scripts'
    assert resolve_path('scripts/evaluate_structures.py') == ROOT/'unifiedgroupgen/tests/legacy/evaluation_before_tasks/cli.py'


def test_worker_pythonpath_uses_imported_package_and_keeps_explicit_paths(monkeypatch):
    import os
    from unifiedgroupgen.tests.legacy.paths import package_parent, runtime_pythonpath
    monkeypatch.setenv('PYTHONPATH', 'existing_package_path')
    paths=runtime_pythonpath().split(os.pathsep)
    assert paths[0] == str(package_parent())
    assert 'existing_package_path' in paths


def test_custom_output_name_does_not_change_dataset_runtime(capsys):
    cfg=configuration('c2db51_all');cfg.experiment_name='c2db_backbone_ablation'
    run.execute(cfg)
    assert cfg.runtime.batch_size == 768 and cfg.runtime.edge_budget == 200000
    assert 'PLAN ONLY' in capsys.readouterr().out
