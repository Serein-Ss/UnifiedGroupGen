"""History snapshots/backfill are independent of live training and original files."""
import builtins
import copy
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from scripts import wandb_history as history


def dump(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def row(epoch,step):
    return dict(epoch=epoch,global_step=step,metrics={'train/loss':99.,'train/loss_epoch':8.-epoch,
        'train/loss_step':123.,'val/loss':7.-epoch,'train/continuous_epoch':.2,'val/continuous':.3,
        'train/element_epoch':0.,'val/element':0.,'train/jump_epoch':4.,'val/jump':4.1},
        learning_rates_used=[.0003],early_stopping=dict(wait_count=epoch,patience=60,best_score=7.-epoch))


def lines(path,rows,tail=b''):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_bytes(b''.join((json.dumps(r)+'\n').encode() for r in rows)+tail)


@pytest.fixture
def campaign(tmp_path):
    original=tmp_path/'old';resumed=tmp_path/'active'
    models=['full_mp20_eqv2_dng','full_mp20_eqv2_csp']
    path=original/'intermediate/hpc_runs/old/campaign.json'
    dump(path,dict(schema='ugg_joint_flow_full_5090_v2',root=str(original),name='old',models=models,revision='old-revision'))
    manifest=resumed/'intermediate/hpc_runs/resumed/campaign.json'
    dump(manifest,dict(schema='ugg_joint_flow_repaired_resume_v1',root=str(resumed),name='resumed',
        original=str(path),original_manifest_sha256=history.sha(path.read_bytes()),revision='repaired-revision',
        models=models,compatibility=dict(models={m:{'global_step':20} for m in models})))
    for model in models:
        for root,name,rows in [(original,'old',[row(0,10),row(1,20),row(2,30)]),
                               (resumed,'resumed',[row(0,10),row(1,20),row(2,30),row(3,40)])]:
            directory=root/'outputs'/model/name
            lines(directory/'model.history.jsonl',rows,tail=b'{"epoch":4' if name=='resumed' else b'')
            (directory/'last.ckpt').write_bytes(b'Never read or rewrite this weight')
    return manifest


def input_hashes(root):
    return {str(p):(hashlib.sha256(p.read_bytes()).hexdigest(),p.stat().st_mtime_ns)
            for p in root.rglob('*') if p.is_file()}


def test_plan_is_read_only_deduplicates_and_uses_epoch_means(campaign,monkeypatch):
    before=input_hashes(campaign.parents[4])
    imported=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='wandb':raise AssertionError('Planning must not import W&B')
        return imported(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    report,files,_=history.collect(campaign)
    for model,info in report['models'].items():
        assert info['rows']==4 and info['duplicates_removed']==2 and info['missing_epochs']==[]
        assert info['sources'][0]['excluded_after_checkpoint']==1
        assert info['sources'][1]['ignored_tail_bytes']>0
        rows=[json.loads(line) for line in files[info['history']].splitlines()]
        assert [r['global_step'] for r in rows]==[10,20,30,40]
        assert rows[0]['train/loss']==8. and rows[0]['epoch_number']==1
        assert rows[-1]['segment']=='resume' and rows[0]['segment']=='original'
        assert not any(key.endswith('_step') and key!='global_step' for r in rows for key in r)
        assert rows[0]['val/element']==0.
    assert before==input_hashes(campaign.parents[4])


def test_snapshot_is_separate_immutable_and_survives_source_append(campaign,tmp_path):
    before=input_hashes(tmp_path)
    out=tmp_path/'backfill/snapshot';report=history.prepare(campaign,out)
    assert all((out/name).is_file() for name in report['files'])
    assert before=={p:input_hashes(tmp_path)[p] for p in before}
    live=Path(history.read(campaign)['root'])/'outputs'/next(iter(report['models']))/'resumed/model.history.jsonl'
    live.write_bytes(live.read_bytes()+b',"global_step":50,"metrics":{"train/loss":4.0,"val/loss":3.0}}\n')
    assert history.verify(out)[1]['models']==report['models']
    with pytest.raises(FileExistsError):history.prepare(campaign,out)
    for directory in (campaign.parent,campaign.parents[3]/'outputs'/next(iter(report['models']))/'resumed'):
        with pytest.raises(ValueError,match='separate'):history.prepare(campaign,directory/'backfill')
    target=out/next(iter(report['files']));target.write_bytes(b'changed')
    with pytest.raises(ValueError,match='hash mismatch'):history.verify(out)


@pytest.mark.parametrize('kind',['malformed','conflict','nonfinite','nonmonotonic'])
def test_invalid_complete_records_fail_before_publication(campaign,kind):
    recovery=history.read(campaign);model=recovery['models'][0]
    path=Path(recovery['root'])/'outputs'/model/'resumed/model.history.jsonl'
    rows=[row(0,10),row(1,20),row(2,30)]
    if kind=='malformed':path.write_bytes(b'{broken}\n')
    else:
        if kind=='conflict':rows[0]['metrics']['val/loss']=100.
        if kind=='nonfinite':rows[-1]['metrics']['val/loss']=float('nan')
        if kind=='nonmonotonic':rows[-1]['global_step']=19
        lines(path,rows)
    with pytest.raises(ValueError):history.collect(campaign)


def fake_wandb(monkeypatch,fail=False):
    calls=[]
    class Run:
        url='https://wandb.example/backfill'
        def __init__(self,options):self.options=options;self.metrics=[];self.rows=[];self.finished=[]
        def define_metric(self,*args,**kwargs):self.metrics.append((args,kwargs))
        def log(self,row,step):
            if fail:raise RuntimeError('Mock upload failure')
            self.rows.append((copy.deepcopy(row),step))
        def finish(self,**kwargs):self.finished.append(kwargs)
    def init(**options):
        run=Run(options);calls.append(run);return run
    monkeypatch.setitem(sys.modules,'wandb',SimpleNamespace(run=None,Settings=lambda **kwargs:kwargs,init=init))
    return calls


def test_online_publish_uses_new_runs_explicit_target_monotonic_steps_and_receipts(campaign,tmp_path,monkeypatch):
    out=tmp_path/'backfill';report=history.prepare(campaign,out)
    before=input_hashes(campaign.parents[3]);calls=fake_wandb(monkeypatch)
    with pytest.raises(ValueError,match='explicit --entity'):history.publish(out,'online')
    assert not calls
    result=history.publish(out,'online','my-account')
    assert len(calls)==2 and len(result)==2
    for run in calls:
        assert run.options['resume']=='never' and run.options['entity']=='my-account'
        assert run.options['id'] in report['runs'].values()
        assert run.options['job_type']=='history-backfill' and run.options['config']['epoch_only']
        assert run.options['settings']==dict(disable_git=True,disable_code=True,x_disable_stats=True)
        assert [step for _,step in run.rows]==[10,20,30,40]
        assert (('train/*',),{'step_metric':'epoch_number'}) in run.metrics
    history.publish(out,'online','my-account');assert len(calls)==2
    with pytest.raises(ValueError):history.publish(out,'online','another-account')
    assert before==input_hashes(campaign.parents[3])


def test_publish_failure_preserves_evidence_and_never_retries_into_original_run(campaign,tmp_path,monkeypatch):
    out=tmp_path/'backfill';history.prepare(campaign,out);calls=fake_wandb(monkeypatch,fail=True)
    with pytest.raises(RuntimeError):history.publish(out,'online','my-account')
    model=history.read(campaign)['models'][0]
    assert history.read(out/(model+'.receipt.json'))['state']=='failed'
    assert calls[0].finished==[{'exit_code':1}]
    with pytest.raises(ValueError,match='new snapshot'):history.publish(out,'online','my-account')
    assert len(calls)==1


def test_offline_publish_does_not_claim_online_upload(campaign,tmp_path,monkeypatch):
    out=tmp_path/'backfill';history.prepare(campaign,out);calls=fake_wandb(monkeypatch)
    results=history.publish(out)
    assert len(calls)==2 and all(v['url'] is None and v['mode']=='offline' for v in results.values())
    assert all('resume' not in run.options for run in calls)


def test_missing_epochs_are_reported_without_interpolation(campaign):
    recovery=history.read(campaign);model=recovery['models'][0]
    path=Path(recovery['root'])/'outputs'/model/'resumed/model.history.jsonl'
    lines(path,[row(0,10),row(1,20),row(4,50)])
    report,files,_=history.collect(campaign)
    assert report['models'][model]['missing_epochs']==[2,3]
    assert [json.loads(line)['epoch'] for line in files[model+'.jsonl'].splitlines()]==[0,1,4]


def test_repeated_recovery_histories_merge_all_segments_without_duplicate_epochs(campaign,tmp_path):
    previous=history.read(campaign);root=tmp_path/'next'
    manifest=root/'intermediate/hpc_runs/offline30/campaign.json'
    dump(manifest,dict(schema='ugg_joint_flow_repaired_resume_v1',root=str(root),name='offline30',
        original=str(campaign),original_manifest_sha256=history.sha(campaign.read_bytes()),revision='policy-30',
        models=previous['models'],compatibility={'models':{m:{'global_step':40} for m in previous['models']}}))
    for model in previous['models']:
        lines(root/'outputs'/model/'offline30/model.history.jsonl',[row(i,(i+1)*10) for i in range(5)])
    report,files,_=history.collect(manifest)
    for model,info in report['models'].items():
        rows=[json.loads(line) for line in files[info['history']].splitlines()]
        assert len(rows)==5 and info['duplicates_removed']==6
        assert [r['segment'] for r in rows]==['original','original','resume1','resume1','resume']
        assert len(info['sources'])==3


def test_unterminated_valid_json_is_deferred_and_complete_corruption_rejected(tmp_path):
    path=tmp_path/'history.jsonl';lines(path,[row(0,10)],tail=json.dumps(row(1,20)).encode())
    data,rows,source=history.history(path)
    assert len(rows)==1 and source['ignored_tail_bytes']>0
    path.write_bytes(data+b'{broken}\n')
    with pytest.raises(ValueError):history.history(path)


def test_corrupt_snapshot_is_rejected_before_sdk_import(campaign,tmp_path,monkeypatch):
    out=tmp_path/'backfill';report=history.prepare(campaign,out)
    (out/next(iter(report['files']))).write_bytes(b'changed')
    imported=builtins.__import__
    def guarded(name,*args,**kwargs):
        if name=='wandb':raise AssertionError('Must verify before SDK import')
        return imported(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',guarded)
    with pytest.raises(ValueError,match='hash mismatch'):history.publish(out,'online','my-account')


def test_joint_logger_identity_is_saved_without_rng_or_parameter_changes(tmp_path,monkeypatch):
    from pytorch_lightning.loggers import WandbLogger
    from unifiedgroupgen.pl_modules import joint_training as training
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


def test_two_stage_logger_identity_and_disabled_mode_remain_compatible(tmp_path,monkeypatch):
    from unifiedgroupgen import run
    monkeypatch.setattr(run.torch,'load',lambda *a,**k:{'ugg_state':{'wandb_id':'two-stage-id'}})
    monkeypatch.setattr(run,'WandbLogger',lambda **options:options)
    settings=SimpleNamespace(wandb_mode='offline',wandb_dir=tmp_path/'logs')
    result=run.build_loggers(SimpleNamespace(),tmp_path/'model.pt',settings,'unused')[0]
    assert result['id']=='two-stage-id' and 'resume' not in result
    settings.wandb_mode='disabled'
    assert run.build_loggers(SimpleNamespace(),tmp_path/'model.pt',settings,'unused')==[]
