"""Recovery checks/copies and mocked scheduler only: no fit or optimizer update."""
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from submit_files import joint_flow5090_resume as resume
from submit_files.hpc5090 import read,write


@pytest.fixture(autouse=True)
def forbid_training(monkeypatch):
    import pytorch_lightning as pl
    def forbidden(*args,**kwargs):raise AssertionError('No training/optimizer updates allowed')
    monkeypatch.setattr(pl.Trainer,'fit',forbidden)
    monkeypatch.setattr(torch.optim.AdamW,'step',forbidden)
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def original(tmp_path,monkeypatch):
    root=tmp_path/'original';folder=root/'intermediate/hpc_runs/original'
    models=['full_mp20_eqv2_dng','full_mp20_eqv2_csp']
    campaign=dict(schema='ugg_joint_flow_full_5090_v2',name='original',root=str(root),source=str(root/'source'),revision='old',
        models=models,entries={m:dict(experiment=m) for m in models},calibration=dict(candidates=[[8,40000],[4,40000]]),
        resources=dict(partition='5090',gpus=1,cpus=8,memory='64G',hours=72))
    inputs={}
    for split in ['train','val','test']:
        p=root/(split+'.jsonl');p.parent.mkdir(exist_ok=True,parents=True);p.write_text('{}\n')
        inputs[split]=[dict(path=str(p),sha256=resume.digest(p))]
    raw=dict(dataset='mp_20',model={'family':'joint_flow_v2'},epochs=1000)
    resolved=SimpleNamespace(runtime=dict(fused_adamw=True,log_every_n_steps=25,strict_fp32=True))
    monkeypatch.setattr(resume,'original_definition',lambda *a:(resolved,copy.deepcopy(raw)))
    def validate(saved,config,runtime,hashes):
        assert saved['joint_state']['config']==config
        return dict(epoch=saved['epoch'],global_step=saved['global_step'],runtime=runtime,strict_load=True)
    monkeypatch.setattr(resume,'validate_checkpoint',validate)
    for model in models:
        batch=8 if model.endswith('dng') else 4
        runtime=dict(batch_size=batch,edge_budget=40000,precision='fp32',device='cuda:0',train_records=1,val_records=1,**resolved.runtime)
        audit=dict(passed=True,inputs=inputs,config_sha256=hashlib.sha256(json.dumps(raw,sort_keys=True).encode()).hexdigest())
        trial=folder/'profiles'/model/'trial';write(trial/'inputs.json',dict(config={**raw,'allowed_elements':[8]}))
        profile=dict(state='complete',trial_directory=str(trial),identity=resume.identity(campaign,model,audit),
            selected=dict(state='pass',batch_size=batch,edge_budget=40000,precision='fp32',records_per_second=25.,
                          peak_reserved_bytes=100,memory_limit_bytes=200))
        write(folder/'audits'/(model+'.json'),audit);write(folder/'profiles'/(model+'.json'),profile)
        output=resume.original_output(campaign,model);output.mkdir(parents=True)
        callback=dict(dirpath=str(output),best_model_path=str(output/'best.ckpt'),last_model_path=str(output/'last.ckpt'),
            kth_best_model_path=str(output/'best.ckpt'),best_k_models={str(output/'best.ckpt'):torch.tensor(2.)},
            best_model_score=torch.tensor(2.),kth_value=torch.tensor(2.))
        state=dict(epoch=9,global_step=30528,state_dict={'model.weight':torch.ones(2)},
            optimizer_states=[{'state':{0:{'exp_avg':torch.zeros(2)}}}],lr_schedulers=[{'last_epoch':8}],loops={'fit_loop':{'counter':9}},
            callbacks={"ModelCheckpoint{'monitor': 'val/loss'}":callback,'EarlyStopping':{'wait_count':3}},
            joint_state=dict(format='ugg_joint_flow_v2',config={**raw,'allowed_elements':[8]},runtime=runtime,
                split_hashes={k:{r['path']:r['sha256'] for r in inputs[k]} for k in ('train','val')},
                epoch_boundary=True,pending_lr_metric=2.,rng={'torch':torch.get_rng_state()}))
        torch.save(state,output/'last.ckpt');torch.save(state,output/'best.ckpt')
        (output/'model.history.jsonl').write_text(json.dumps(dict(global_step=30000))+'\n'+json.dumps(dict(global_step=40000))+'\n')
    manifest=folder/'campaign.json';write(manifest,campaign)
    return manifest,campaign


def test_check_is_read_only_and_keeps_original_calibration_identity(original):
    path,campaign=original
    before={str(p):resume.digest(p) for p in Path(campaign['root']).rglob('*') if p.is_file()}
    result=resume.check(path)
    assert result['passed'] and len(result['models'])==2 and not result['training_started'] and not result['submitted']
    assert result['original_revision']=='old'
    assert [m['runtime']['batch_size'] for m in result['models'].values()]==[8,4]
    assert before=={str(p):resume.digest(p) for p in Path(campaign['root']).rglob('*') if p.is_file()}


@pytest.mark.parametrize('corruption',['data','profile','raw_config','fitted_config','runtime','hashes'])
def test_check_refuses_identity_mismatch(original,corruption):
    path,campaign=original;m=campaign['models'][0];base=path.parent
    if corruption=='data':Path(campaign['root'],'train.jsonl').write_text('changed')
    elif corruption=='profile':
        p=base/'profiles'/(m+'.json');v=read(p);v['identity']['revision']='new';write(p,v)
    elif corruption=='raw_config':
        p=base/'audits'/(m+'.json');v=read(p);v['config_sha256']='wrong';write(p,v)
    else:
        p=resume.original_output(campaign,m)/'last.ckpt';v=torch.load(p,weights_only=False)
        if corruption=='fitted_config':v['joint_state']['config']['epochs']=2
        elif corruption=='runtime':v['joint_state']['runtime']['batch_size']=128
        else:v['joint_state']['split_hashes']={}
        torch.save(v,p)
    with pytest.raises(ValueError):resume.check(path)


def test_relocation_changes_only_checkpoint_callback_paths(original,tmp_path):
    _,campaign=original;old=resume.original_output(campaign,campaign['models'][0])
    saved=torch.load(old/'last.ckpt',weights_only=False);original_bytes=resume.digest(old/'last.ckpt')
    new=tmp_path/'independent'
    converted=resume.relocate_checkpoint(saved,old,new)
    for key in ('joint_state','loops','lr_schedulers'):
        if key=='joint_state':
            assert torch.equal(converted[key]['rng']['torch'],saved[key]['rng']['torch'])
            assert converted[key]['config']==saved[key]['config'] and converted[key]['pending_lr_metric']==2.
        else:assert converted[key]==saved[key]
    assert converted['callbacks']['EarlyStopping']==saved['callbacks']['EarlyStopping']
    cb=next(v for k,v in converted['callbacks'].items() if k.startswith('ModelCheckpoint'))
    assert cb['dirpath']==str(new.resolve()) and cb['best_model_path']==str(new/'best.ckpt')
    assert cb['best_model_score']==2. and set(cb['best_k_models'])=={str(new/'best.ckpt')}
    assert resume.digest(old/'last.ckpt')==original_bytes
    bad=copy.deepcopy(saved);next(v for k,v in bad['callbacks'].items() if k.startswith('ModelCheckpoint'))['best_model_path']=str(tmp_path/'outside.ckpt')
    with pytest.raises(ValueError,match='outside original'):resume.relocate_checkpoint(bad,old,new)


def prepared(original,tmp_path,monkeypatch):
    manifest,campaign=original;root=tmp_path/'new';root.mkdir()
    monkeypatch.setattr(resume,'ROOT',root)
    monkeypatch.setattr(resume.subprocess,'run',lambda *a,**k:None)
    monkeypatch.setattr(resume.subprocess,'check_output',lambda args,**k:'new' if 'rev-parse' in args else '')
    def snapshot(root,folder,revision):
        source=folder/'source';source.mkdir();return source
    monkeypatch.setattr(resume,'snapshot',snapshot)
    resume.prepare(manifest,'repaired')
    return root/'intermediate/hpc_runs/repaired/campaign.json'


def test_prepare_copies_to_new_outputs_preserves_originals_and_trims_history(original,tmp_path,monkeypatch):
    old=Path(original[1]['root']);before={str(p):resume.digest(p) for p in old.rglob('*') if p.is_file()}
    manifest=prepared(original,tmp_path,monkeypatch);r=read(manifest)
    assert r['revision']=='new' and r['compatibility']['original_revision']=='old'
    for m in r['models']:
        out=Path(r['root'])/'outputs'/m/'repaired'
        assert (out/'best.ckpt').is_file() and (out/'last.ckpt').is_file()
        assert len((out/'model.history.jsonl').read_text().splitlines())==1
    assert before=={str(p):resume.digest(p) for p in old.rglob('*') if p.is_file()}
    with pytest.raises(FileExistsError):resume.prepare(original[0],'repaired')


def test_submit_is_mocked_idempotent_and_refuses_changed_copies(original,tmp_path,monkeypatch):
    manifest=prepared(original,tmp_path,monkeypatch);calls=[]
    def queue(args,**kwargs):calls.append(args);return str(100+len(calls))
    monkeypatch.setattr(resume.subprocess,'check_output',queue)
    resume.submit(manifest);resume.submit(manifest)
    assert len(calls)==2
    assert all(args[0]=='yhbatch' and 'joint_flow5090_resume.sbatch' in ' '.join(args) for args in calls)
    r=read(manifest);out=Path(r['root'])/'outputs'/r['models'][0]/'repaired/last.ckpt';out.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint changed'):resume.submit(manifest)
    assert len(calls)==2


def test_native_checkpoint_validation_without_fit_or_rng_mutation(tmp_path):
    from unifiedgroupgen.pl_modules.joint_training import JointGeneratorModule,joint_callbacks
    config=dict(model=dict(family='joint_flow_v2',backbone='joint_atom_graph',hidden=16,layers=1,heads=4,task_controls=True),
        allowed_elements=[8],properties={},lr_schedule=dict(patience=10,factor=.5),generation_task='dng')
    runtime=dict(batch_size=8,precision='fp32',fused_adamw=True)
    module=JointGeneratorModule(config,runtime,{})
    options=module.configure_optimizers()
    scheduler=options['lr_scheduler']['scheduler'].state_dict();scheduler['last_epoch']=8
    callbacks={c.state_key:c.state_dict() for c in joint_callbacks(config,tmp_path/'model.pt')}
    saved=dict(state_dict=module.state_dict(),optimizer_states=[options['optimizer'].state_dict()],
        lr_schedulers=[scheduler],loops={'fit_loop':{'epoch_progress':{'total':{'completed':9}},
            'epoch_loop.batch_progress':{'current':{'completed':0},'total':{'completed':30528}},
            'epoch_loop.automatic_optimization.optim_progress':{'optimizer':{'step':{'total':{'completed':30528}}}}}},callbacks=callbacks,
        epoch=9,global_step=30528,joint_state=dict(format='ugg_joint_flow_v2',config=config,runtime=runtime,split_hashes={},
            epoch_boundary=True,pending_lr_metric=2.,rng=dict(torch=torch.get_rng_state(),python=__import__('random').getstate(),numpy=__import__('numpy').random.get_state())))
    rng=torch.get_rng_state().clone()
    report=resume.validate_checkpoint(saved,config,runtime,{})
    assert torch.equal(torch.get_rng_state(),rng)
    assert report['strict_load'] and report['pending_lr_metric']==2. and report['global_step']==30528
    saved['state_dict'][next(iter(saved['state_dict']))].fill_(float('nan'))
    with pytest.raises(ValueError,match='Nonfinite'):resume.validate_checkpoint(saved,config,runtime,{})


def test_worker_uses_raw_data_gate_fitted_training_config_and_new_outputs(original,tmp_path,monkeypatch):
    from unifiedgroupgen.pl_data import dataset
    from unifiedgroupgen.pl_modules import joint_training
    manifest=prepared(original,tmp_path,monkeypatch);recovery=read(manifest);model=recovery['models'][0]
    monkeypatch.setattr(resume,'ROOT',Path(recovery['source']))
    monkeypatch.setattr(resume,'device_info',lambda:dict(gpu='mocked 5090'))
    calls=[]
    def load(config,root):
        assert 'allowed_elements' not in config  # Audit signs the pre-fitting definition.
        assert root==Path(original[1]['source'])
        return [{'id':'train'}],[{'id':'val'}],{}
    def fit(config,records,validation,output,**kwargs):
        assert config['allowed_elements']==[8]
        assert output==Path(recovery['root'])/'outputs'/model/'repaired/model.pt'
        assert kwargs['resume']==output.parent/'last.ckpt'
        assert kwargs['runtime']['batch_size']==8 and kwargs['epochs']==1000
        calls.append(kwargs)
    monkeypatch.setattr(dataset,'load_training_splits',load)
    monkeypatch.setattr(joint_training,'fit_joint',fit)
    resume.run(manifest,model)
    assert len(calls)==1
