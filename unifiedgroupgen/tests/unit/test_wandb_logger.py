"""Current training logger and checkpoint identity without training or network use."""
from types import SimpleNamespace

import pytest
import torch


def test_joint_logger_identity_is_saved_without_rng_or_parameter_changes(tmp_path,monkeypatch):
    from pytorch_lightning.loggers import WandbLogger
    from unifiedgroupgen.pl_modules.unified import module as training
    from unifiedgroupgen.tests.unit.test_joint_training_controls import checkpoint
    logger=WandbLogger(project='fixture',id='fixed-id',offline=True,save_dir=tmp_path)
    module=SimpleNamespace(config=dict(model={'family':'joint_flow_v2'}),runtime={},split_hashes={},
        logger=logger,trainer=SimpleNamespace(num_training_batches=4,default_root_dir=str(tmp_path)))
    monkeypatch.setattr(training,'_collect_rng_states',lambda:{'fixture':True})
    rng=torch.get_rng_state().clone();saved=checkpoint()
    training.JointGeneratorModule.on_save_checkpoint(module,saved)
    assert saved['joint_state']['wandb_id']=='fixed-id'
    assert saved['joint_state']['wandb_run_directory']==str(tmp_path.resolve())
    assert torch.equal(torch.get_rng_state(),rng)


@pytest.mark.parametrize('mode,same_directory,reuse',[
    ('online',True,True),('online',False,False),('offline',True,False)])
def test_joint_logger_reuses_only_same_directory_online_identity(tmp_path,monkeypatch,mode,same_directory,reuse):
    from unifiedgroupgen import run
    saved=dict(joint_state=dict(wandb_id='old-id',wandb_run_directory=str((tmp_path if same_directory else tmp_path/'old').resolve())))
    monkeypatch.setattr(run.torch,'load',lambda *a,**k:saved)
    monkeypatch.setattr(run,'WandbLogger',lambda **options:options)
    settings=SimpleNamespace(wandb_mode=mode,wandb_dir=tmp_path/'logs')
    result=run.build_loggers(SimpleNamespace(experiment='fixture'),tmp_path/'model.pt',settings,'unused')[0]
    assert (result['id']=='old-id')==reuse
    expected=('allow' if reuse else 'never') if mode=='online' else None
    assert result['resume']==expected
    assert result['log_model'] is False


def test_disabled_logger_creates_no_run_or_directory(tmp_path):
    from unifiedgroupgen import run
    settings=SimpleNamespace(wandb_mode='disabled',wandb_dir=tmp_path/'logs')
    assert run.build_loggers(SimpleNamespace(),tmp_path/'model.pt',settings,None)==[]
    assert not settings.wandb_dir.exists()
