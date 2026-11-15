"""Core ownership, legacy API identity and read-only historical checkpoint loads."""
import importlib
import json
from pathlib import Path
import pickle

import pytest
import torch

PACKAGE=Path(__file__).resolve().parents[2]
ROOT=PACKAGE.parent
LAYOUT=json.loads((PACKAGE/'core_layout.json').read_text(encoding='utf-8'))


@pytest.fixture(autouse=True)
def no_updates(monkeypatch):
    import pytorch_lightning as pl
    def forbidden(*args,**kwargs):raise AssertionError('No training/optimizer updates in layout verification')
    monkeypatch.setattr(pl.Trainer,'fit',forbidden)
    monkeypatch.setattr(torch.optim.AdamW,'step',forbidden)
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.mark.parametrize('old,new',list(LAYOUT['aliases'].items()))
def test_old_and_new_imports_are_one_module_including_private_api(old,new):
    legacy=importlib.import_module('unifiedgroupgen.'+old[:-3].replace('/','.'))
    canonical=importlib.import_module(new)
    assert legacy is canonical
    for name,value in vars(canonical).items():
        assert getattr(legacy,name) is value


def test_monkeypatch_through_old_path_changes_actual_sampler_dependency(monkeypatch):
    import unifiedgroupgen.joint_flow.path as old
    import unifiedgroupgen.pl_data.unified.path as new
    def replacement(*args,**kwargs):return 'fixture'
    monkeypatch.setattr(old,'_program',replacement)
    assert new.sample_path.__globals__['_program'] is replacement


@pytest.mark.parametrize('old,names',list(LAYOUT['legacy_exports'].items()))
def test_all_pre_refactor_exported_names_remain_available(old,names):
    module=importlib.import_module('unifiedgroupgen.'+old[:-3].replace('/','.'))
    assert not [name for name in names if not hasattr(module,name)]


def test_split_shared_classes_and_functions_have_single_implementations():
    from unifiedgroupgen.pl_data import joint_path,task_path,targets,programs
    from unifiedgroupgen.pl_data.unified import tasks
    from unifiedgroupgen.conditional import tasks as rules
    from unifiedgroupgen.pl_modules import joint_graph,layers
    from unifiedgroupgen.generation import joint,types
    assert joint_path.JointTarget is targets.JointTarget
    assert joint_path._program is programs._program
    assert joint_path.PathStep is programs.PathStep
    assert task_path.inventory_program is programs.inventory_program
    assert tasks.validate_task_config is rules.validate_task_config
    assert task_path.task_target is tasks.task_target
    assert joint_graph.AtomGraphBlock is layers.AtomGraphBlock
    assert joint.JointSample is types.JointSample


@pytest.mark.parametrize('old,name,new',[
    ('unifiedgroupgen.joint_flow.state','FlowState','unifiedgroupgen.group_constraints.flow_state'),
    ('unifiedgroupgen.generation.joint','JointSample','unifiedgroupgen.generation.types'),
    ('unifiedgroupgen.pl_modules.descriptor','DescriptorTransformer','unifiedgroupgen.pl_modules.two_stage.descriptor'),
    ('unifiedgroupgen.pl_modules.joint_training','JointGeneratorModule','unifiedgroupgen.pl_modules.unified.module'),
    ('unifiedgroupgen.run','UnifiedGroupGenModule','unifiedgroupgen.pl_modules.two_stage.module')])
def test_historical_pickle_global_names_still_resolve(old,name,new):
    serialized=('c'+old+'\n'+name+'\n.').encode('ascii')
    assert pickle.loads(serialized) is getattr(importlib.import_module(new),name)


def test_periodic_graph_keeps_coordinate_and_cell_gradients_through_old_import():
    from unifiedgroupgen.pl_modules.backbones.periodic_graph import radius_graph
    from unifiedgroupgen.pl_data.periodic_graph import radius_graph as actual
    assert radius_graph is actual
    f=torch.tensor([[.1,.2,.3],[.4,.5,.6]],requires_grad=True)
    cell=(torch.eye(3)*3).unsqueeze(0).requires_grad_()
    result=radius_graph(f,cell,torch.tensor([2]),torch.ones((1,3),dtype=torch.bool))
    derivatives=torch.autograd.grad(result[-1].square().sum(),(f,cell))
    assert all(torch.isfinite(v).all() for v in derivatives)


@pytest.mark.parametrize('filename',['best.ckpt','last.ckpt','model.pt'])
@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
@pytest.mark.parametrize('task',['dng','csp','property'])
def test_existing_joint_flow_weights_strictly_load_without_rewriting(dataset,task,filename):
    from unifiedgroupgen.pl_modules.factory import build_joint_model
    from unifiedgroupgen.pl_modules.joint_training import load_joint_checkpoint
    path=ROOT/'intermediate/checks/joint_flow_v2_20261113/runs'/dataset/task/'equiformer_v2'/filename
    if not path.is_file():pytest.skip('Historical local short-run weights not supplied')
    before=(path.stat().st_size,path.stat().st_mtime_ns)
    saved=load_joint_checkpoint(path)
    model=build_joint_model(saved['config'])
    model.load_state_dict(saved['model'],strict=True)
    assert all(torch.isfinite(v).all() for v in model.state_dict().values() if torch.is_tensor(v))
    assert (path.stat().st_size,path.stat().st_mtime_ns)==before


HISTORICAL=['mp20_'+x for x in ['u','ef','eg','eh','all']]+['c2db51_'+x for x in ['u','ef','eg','eh','all']]+[
    'joint_'+x for x in ['u','ef','eg','eh','all']]+['mp20_u_comp_v2','c2db51_u_comp_v2','joint_u_comp_v2','joint_u_comp_source_control_v2']


@pytest.mark.parametrize('experiment',HISTORICAL)
def test_existing_two_stage_checkpoint_strictly_loads(experiment):
    from unifiedgroupgen.pl_modules.factory import load_model_checkpoint,build_models
    run='hpc5090_comp_v1' if '_comp_' in experiment else '26-10-30-12-50-31'
    path=ROOT/'outputs'/experiment/run/'model.best.pt'
    if not path.is_file():pytest.skip('Historical local weights not supplied')
    before=(path.stat().st_size,path.stat().st_mtime_ns)
    saved=load_model_checkpoint(path)
    proposal,flow=build_models(saved['config'],'cpu')
    proposal.load_state_dict(saved['proposal'],strict=True)
    flow.load_state_dict(saved['flow'],strict=True)
    assert (path.stat().st_size,path.stat().st_mtime_ns)==before


def test_predictor_registry_available_at_new_and_old_resource_paths():
    old=PACKAGE/'common/predictors/public_megnet_models.json'
    new=PACKAGE/'evaluation/predictors/public_megnet_models.json'
    assert old.read_bytes()==new.read_bytes()
    from unifiedgroupgen.common.predictors.matgl import model_registry as legacy
    from unifiedgroupgen.evaluation.predictors.matgl import model_registry
    assert legacy is model_registry


def test_two_stage_save_hook_keeps_rng_and_pending_schedule_without_fit():
    from types import SimpleNamespace
    from unifiedgroupgen.pl_modules.two_stage.module import UnifiedGroupGenModule
    stub=SimpleNamespace(config={'fixture':True},runtime={'validation_interval':1},hashes={},best=2.,
                         logger=False,val_loss=2.,current_epoch=2,lr_schedulers=lambda:SimpleNamespace(last_epoch=2))
    saved={}
    UnifiedGroupGenModule.on_save_checkpoint(stub,saved)
    assert saved['ugg_state']['pending_plateau']==2.
    assert saved['ugg_state']['wandb_id'] is None
    assert set(saved['ugg_state']['rng']) >= {'torch','numpy','python'}


def test_two_stage_artifact_callback_writes_only_fixture_files(tmp_path):
    from types import SimpleNamespace
    from unifiedgroupgen.pl_modules.two_stage.module import TrainingArtifacts
    keys=('descriptor','flow','lattice','coordinates')
    module=SimpleNamespace(device=torch.device('cpu'),training_seconds=1.,epoch_started=0.,keys=keys,
        train_totals=torch.ones(4),train_seen=1,validated=dict(zip(keys,[1.]*4)),
        runtime={'coordinate_length_scale':1.},norm_totals=torch.tensor([1.,0.]),train_batches=1,
        weights={'descriptor':1.,'flow':1.},val_loss=2.,best=2.,validation_seconds=.1,
        output=tmp_path/'model.pt',log_dict=lambda *a,**k:None)
    trainer=SimpleNamespace(current_epoch=0,max_epochs=1000,optimizers=[SimpleNamespace(param_groups=[{'lr':.001}])],
                            datamodule=SimpleNamespace(records=[{}],validation=[{}]))
    TrainingArtifacts().on_train_epoch_end(trainer,module)
    assert json.loads((tmp_path/'model.status.json').read_text())['completed_epochs']==1
    assert (tmp_path/'train.log').is_file() and (tmp_path/'model.history.jsonl').is_file()
    assert not (tmp_path/'model.pt').exists()
