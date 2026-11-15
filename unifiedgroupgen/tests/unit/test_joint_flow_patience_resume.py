"""Detached policy migration and chained recovery; no real training or submission."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning.callbacks import EarlyStopping
from submit_files import joint_flow5090_resume as resume
from submit_files.hpc5090 import read,write
from unifiedgroupgen.tests.unit.test_joint_flow_repaired_resume import original,prepared,forbid_training


def hashes(root):
    return {str(p):resume.digest(p) for p in root.rglob('*') if p.is_file()}


def same(left,right):
    if torch.is_tensor(left):return torch.equal(left,right)
    if isinstance(left,dict):return left.keys()==right.keys() and all(same(v,right[k]) for k,v in left.items())
    if isinstance(left,(list,tuple)):return len(left)==len(right) and all(same(a,b) for a,b in zip(left,right))
    return left==right


@pytest.fixture
def policy_original(original,monkeypatch):
    manifest,campaign=original
    raw=dict(dataset='mp_20',model={'family':'joint_flow_v2'},epochs=1000,
             early_stopping=dict(patience=60,min_delta=.0001),lr_schedule=dict(factor=.5,patience=10,threshold=.0001,min_lr=1e-6))
    resolved=SimpleNamespace(runtime=dict(fused_adamw=True,log_every_n_steps=25,strict_fp32=True))
    monkeypatch.setattr(resume,'original_definition',lambda *args:(resolved,copy.deepcopy(raw)))
    for model in campaign['models']:
        audit_path=manifest.parent/'audits'/(model+'.json');audit=read(audit_path)
        audit['config_sha256']=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest();write(audit_path,audit)
        profile_path=manifest.parent/'profiles'/(model+'.json');profile=read(profile_path)
        profile['identity']=resume.identity(campaign,model,audit);write(profile_path,profile)
        config={**raw,'allowed_elements':[8]};write(Path(profile['trial_directory'])/'inputs.json',dict(config=config))
        directory=resume.original_output(campaign,model)
        for filename in ('last.ckpt','best.ckpt'):
            saved=torch.load(directory/filename,weights_only=False)
            saved['joint_state']['config']=copy.deepcopy(config)
            saved['hyper_parameters']={'config':copy.deepcopy(config)}
            saved['callbacks'].pop('EarlyStopping')
            saved['callbacks'][EarlyStopping('val/loss').state_key]=dict(wait_count=12 if model.endswith('dng') else 1,
                patience=60,best_score=torch.tensor(8.334162712097168 if model.endswith('dng') else 4.145624160766602),stopped_epoch=0)
            torch.save(saved,directory/filename)
    return manifest,campaign


@pytest.fixture
def chain(policy_original,tmp_path,monkeypatch):
    manifest=prepared(policy_original,tmp_path,monkeypatch);campaign=read(manifest)
    for model in campaign['models']:
        path=resume.original_output(campaign,model)/'last.ckpt'
        saved=torch.load(path,weights_only=False)
        saved.update(epoch=105 if model.endswith('dng') else 68,global_step=356160 if model.endswith('dng') else 461312)
        torch.save(saved,path)
    return manifest,campaign


def test_chained_check_reads_current_weights_but_original_calibration(chain,policy_original):
    manifest,campaign=chain;before=hashes(Path(campaign['root']))
    report=resume.check(manifest,early_stopping_patience=30)
    assert report['original_revision']=='new' and report['audit_origin']==str(policy_original[0])
    for model,result in report['models'].items():
        assert result['checkpoint']==str(resume.original_output(campaign,model)/'last.ckpt')
        assert result['epoch']==(105 if model.endswith('dng') else 68)
        assert result['early_stopping']['wait_count']==(12 if model.endswith('dng') else 1)
        assert result['early_stopping']['further_no_improvement_checks']==(18 if model.endswith('dng') else 29)
        assert result['early_stopping']['source_patience']==60 and result['early_stopping']['target_patience']==30
    assert before==hashes(Path(campaign['root']))


def test_prepare_migrates_copies_only_and_chain_can_be_checked_again(chain,policy_original,tmp_path,monkeypatch):
    manifest,source=chain;before=hashes(Path(source['root']));original_before=hashes(Path(policy_original[1]['root']))
    target=tmp_path/'offline30';target.mkdir();monkeypatch.setattr(resume,'ROOT',target)
    resume.prepare(manifest,'offline30',early_stopping_patience=30)
    next_manifest=target/'intermediate/hpc_runs/offline30/campaign.json';campaign=read(next_manifest)
    assert campaign['logging']['mode']=='offline' and campaign['early_stopping_patience']==30
    for model in campaign['models']:
        for name in ('last.ckpt','best.ckpt'):
            old=torch.load(resume.original_output(source,model)/name,weights_only=False)
            new=torch.load(resume.original_output(campaign,model)/name,weights_only=False)
            for key in ('state_dict','optimizer_states','lr_schedulers','loops'):
                assert same(old[key],new[key])
            assert same(old['joint_state']['rng'],new['joint_state']['rng'])
            assert old['epoch']==new['epoch'] and old['global_step']==new['global_step']
            assert new['joint_state']['config']['early_stopping']==dict(patience=30,min_delta=.0001)
            assert new['hyper_parameters']['config']==new['joint_state']['config']
            assert new['joint_state']['config']['lr_schedule']==old['joint_state']['config']['lr_schedule']
            key=EarlyStopping('val/loss').state_key
            assert new['callbacks'][key]=={**old['callbacks'][key],'patience':30}
    assert before==hashes(Path(source['root'])) and original_before==hashes(Path(policy_original[1]['root']))
    resume.verify_recovery(campaign)
    assert all(r['early_stopping']['source_patience']==30 for r in resume.check(next_manifest,early_stopping_patience=30)['models'].values())


@pytest.mark.parametrize('patience',[0,-1,True])
def test_invalid_patience_refused_without_changes(policy_original,patience):
    before=hashes(Path(policy_original[1]['root']))
    with pytest.raises(ValueError,match='positive integer'):resume.check(policy_original[0],early_stopping_patience=patience)
    assert before==hashes(Path(policy_original[1]['root']))


def test_unsigned_training_change_and_ancestor_change_are_rejected(chain,policy_original):
    manifest,campaign=chain;model=campaign['models'][0]
    path=resume.original_output(campaign,model)/'last.ckpt';saved=torch.load(path,weights_only=False)
    saved['joint_state']['config']['lr_schedule']['factor']=.1;torch.save(saved,path)
    with pytest.raises(ValueError,match='fitted configuration'):resume.check(manifest,early_stopping_patience=30)
    ancestor=read(policy_original[0]);ancestor['revision']='tampered';write(policy_original[0],ancestor)
    with pytest.raises(ValueError,match='Ancestor manifest'):resume.check(manifest)


@pytest.mark.parametrize('count',[12,1])
def test_lightning_restores_migrated_patience_and_keeps_exact_remaining_checks(count):
    config=dict(early_stopping=dict(patience=60,min_delta=.0001))
    callback=EarlyStopping('val/loss',patience=60,min_delta=.0001,check_on_train_epoch_end=False)
    state=dict(patience=60,wait_count=count,best_score=torch.tensor(2.),stopped_epoch=0)
    saved=dict(joint_state=dict(config=copy.deepcopy(config)),hyper_parameters=dict(config=copy.deepcopy(config)),
        callbacks={callback.state_key:state})
    old=copy.deepcopy(saved);new=resume.migrate_patience(copy.deepcopy(saved),30)
    callback.load_state_dict(new['callbacks'][callback.state_key])
    assert callback.patience==30 and callback.wait_count==count and callback.best_score==2.
    assert same(saved,old)
    for _ in range(30-count-1):assert not callback._evaluate_stopping_criteria(torch.tensor(2.))[0]
    assert callback._evaluate_stopping_criteria(torch.tensor(2.))[0]
    assert callback.wait_count==30


def test_native_state_load_after_policy_migration_without_training(tmp_path):
    from unifiedgroupgen.pl_modules.joint_training import JointGeneratorModule,joint_callbacks
    import random
    import numpy as np
    config=dict(model=dict(family='joint_flow_v2',backbone='joint_atom_graph',hidden=16,layers=1,heads=4,task_controls=True),
        allowed_elements=[8],properties={},generation_task='dng',early_stopping=dict(patience=60,min_delta=.0001),
        lr_schedule=dict(factor=.5,patience=10))
    module=JointGeneratorModule(config,{},{})
    options=module.configure_optimizers();optimizer=options['optimizer'];scheduler=options['lr_scheduler']['scheduler'].state_dict()
    scheduler['last_epoch']=8
    callbacks={c.state_key:c.state_dict() for c in joint_callbacks(config,tmp_path/'model.pt')}
    saved=dict(state_dict=module.state_dict(),optimizer_states=[optimizer.state_dict()],
        callbacks=callbacks,hyper_parameters=dict(config=copy.deepcopy(config)),epoch=9,global_step=30528,
        lr_schedulers=[scheduler],loops={'fit_loop':{'epoch_progress':{'total':{'completed':9}},
            'epoch_loop.batch_progress':{'current':{'completed':0},'total':{'completed':30528}},
            'epoch_loop.automatic_optimization.optim_progress':{'optimizer':{'step':{'total':{'completed':30528}}}}}},
        joint_state=dict(format='ugg_joint_flow_v2',config=config,runtime={},split_hashes={},
            epoch_boundary=True,pending_lr_metric=2.,
            rng=dict(torch=torch.get_rng_state(),python=random.getstate(),numpy=np.random.get_state())))
    rng=torch.get_rng_state().clone()
    assert resume.validate_checkpoint(saved,config,{},{})['strict_load']
    new=resume.migrate_patience(copy.deepcopy(saved),30)
    assert resume.validate_checkpoint(new,new['joint_state']['config'],{},{})['strict_load']
    assert torch.equal(torch.get_rng_state(),rng)
    loaded=JointGeneratorModule(new['joint_state']['config'],{},{})
    loaded.load_state_dict(new['state_dict'],strict=True);loaded.on_load_checkpoint(new)
    assert same(saved['state_dict'],loaded.state_dict())
    assert new['joint_state']['config']['early_stopping']['patience']==30


@pytest.mark.parametrize('index',[0,1])
def test_chained_worker_uses_original_data_gate_migrated_config_and_offline_logger(chain,policy_original,tmp_path,monkeypatch,index):
    from unifiedgroupgen.pl_data import dataset
    from unifiedgroupgen.pl_modules import joint_training
    from unifiedgroupgen.common.configuration import RunSettings
    from unifiedgroupgen import run
    target=tmp_path/'worker30';target.mkdir();monkeypatch.setattr(resume,'ROOT',target)
    resume.prepare(chain[0],'offline30',early_stopping_patience=30)
    manifest=target/'intermediate/hpc_runs/offline30/campaign.json';campaign=read(manifest);model=campaign['models'][index]
    monkeypatch.setattr(resume,'ROOT',Path(campaign['source']))
    monkeypatch.setattr(resume,'device_info',lambda:dict(gpu='mock'))
    called=[];finished=[];logger=SimpleNamespace(finalize=finished.append)
    def settings(root,mode):
        assert root==target and mode=='offline'
        return SimpleNamespace(wandb_mode='offline')
    monkeypatch.setattr(RunSettings,'load',settings)
    monkeypatch.setattr(run,'build_loggers',lambda *args:[logger])
    def load(config,root):
        assert root==Path(policy_original[1]['source']) and config['early_stopping']['patience']==60
        return [{'id':'train'}],[{'id':'val'}],{}
    monkeypatch.setattr(dataset,'load_training_splits',load)
    def fit(config,records,validation,output,**kwargs):
        assert config['early_stopping']==dict(patience=30,min_delta=.0001)
        assert kwargs['logger']==[logger] and kwargs['runtime']['precision']=='fp32'
        assert kwargs['runtime']['batch_size']==(8 if model.endswith('dng') else 4)
        assert kwargs['resume']==resume.original_output(campaign,model)/'last.ckpt'
        called.append(True)
    monkeypatch.setattr(joint_training,'fit_joint',fit)
    resume.run(manifest,model)
    assert called==[True] and finished==['success']
