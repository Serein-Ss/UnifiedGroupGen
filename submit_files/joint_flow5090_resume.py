"""Check and isolate repaired full-flow resumes; check/prepare never train or submit."""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from submit_files.hpc5090 import read,write,snapshot
from submit_files.joint_flow5090 import selection,catalog,device_info
from submit_files.joint_flow5090_calibrate import identity,validate_profile

SCHEMA='ugg_joint_flow_repaired_resume_v1'


def digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):result.update(chunk)
    return result.hexdigest()


def original_path(campaign,value):
    path=Path(value)
    return path if path.is_absolute() else Path(campaign['root'])/path


def original_output(campaign,model):
    return Path(campaign['root'])/'outputs'/model/campaign['name']


def original_definition(campaign,model):
    from unifiedgroupgen.common.configuration import compose_experiment,training_definition
    override='data.data_audit=intermediate/hpc_runs/'+campaign['name']+'/audits/'+model+'.json'
    resolved=compose_experiment(campaign['entries'][model]['experiment'],root=Path(campaign['source']),overrides=[override])
    return resolved,training_definition(resolved)


def validate_checkpoint(saved,config,runtime,hashes):
    import torch
    from unifiedgroupgen.pl_modules.joint_training import JointGeneratorModule
    if saved.get('joint_state',{}).get('format')!='ugg_joint_flow_v2':
        raise ValueError('A native joint_flow_v2 checkpoint is required')
    for name in ('state_dict','optimizer_states','lr_schedulers','callbacks','loops'):
        if not saved.get(name):raise ValueError('Missing native training state: '+name)
    if not saved['joint_state'].get('epoch_boundary'):
        raise ValueError('This recovery requires a completed epoch boundary')
    loop=saved['loops']['fit_loop']
    batch=loop['epoch_loop.batch_progress']
    progress=loop['epoch_loop.automatic_optimization.optim_progress']['optimizer']['step']
    if (loop['epoch_progress']['total']['completed']!=saved['epoch'] or batch['current']['completed']!=0
            or batch['total']['completed']!=saved['global_step'] or progress['total']['completed']!=saved['global_step']):
        raise ValueError('Checkpoint epoch/batch/optimizer counters disagree')
    pending=saved['joint_state'].get('pending_lr_metric')
    if config.get('lr_schedule') and saved['lr_schedulers'][0]['last_epoch']!=saved['epoch']-(pending is not None):
        raise ValueError('Pending scheduler metric disagrees with epoch boundary')
    def finite(value):
        if torch.is_tensor(value):return bool(torch.isfinite(value).all())
        if isinstance(value,dict):return all(finite(v) for v in value.values())
        if isinstance(value,(tuple,list)):return all(finite(v) for v in value)
        return True
    if not finite(saved['state_dict']) or not finite(saved['optimizer_states']):
        raise ValueError('Nonfinite model/optimizer tensors')
    with torch.random.fork_rng(devices=[]):
        module=JointGeneratorModule(config,runtime,hashes)
        module.load_state_dict(saved['state_dict'],strict=True)
        module.on_load_checkpoint(saved)
        options=module.configure_optimizers()
        optimizer=options['optimizer'] if isinstance(options,dict) else options
        optimizer.load_state_dict(saved['optimizer_states'][0])
        if isinstance(options,dict):
            options['lr_scheduler']['scheduler'].load_state_dict(saved['lr_schedulers'][0])
    rng=saved['joint_state']['rng']
    import random
    import numpy as np
    torch.Generator().set_state(rng['torch'])
    random.Random().setstate(rng['python'])
    np.random.RandomState().set_state(rng['numpy'])
    return dict(epoch=saved['epoch'],global_step=saved['global_step'],runtime=runtime,
                pending_lr_metric=saved['joint_state'].get('pending_lr_metric'),strict_load=True)


def check(original,models=None):
    import torch
    torch.set_num_threads(2)
    campaign=read(original)
    if campaign.get('schema')!='ugg_joint_flow_full_5090_v2':
        raise ValueError('Expected an original FP32 calibrated campaign')
    chosen=selection(catalog(),models)
    if set(chosen)-set(campaign['models']):raise ValueError('Model missing from original campaign')
    results={}
    for model in chosen:
        base=Path(original).resolve().parent
        audit=read(base/'audits'/(model+'.json'));profile=read(base/'profiles'/(model+'.json'))
        if not audit.get('passed'):raise ValueError('Original audit did not pass')
        selected=validate_profile(profile,identity(campaign,model,audit))
        resolved,definition=original_definition(campaign,model)
        if hashlib.sha256(json.dumps(definition,sort_keys=True).encode()).hexdigest()!=audit['config_sha256']:
            raise ValueError('Original frozen configuration hash differs')
        for rows in audit['inputs'].values():
            for row in rows:
                if digest(original_path(campaign,row['path']))!=row['sha256']:
                    raise ValueError('Dataset hash differs: '+row['path'])
        path=original_output(campaign,model)/'last.ckpt';before=digest(path)
        saved=torch.load(path,map_location='cpu',weights_only=False);state=saved['joint_state']
        trial=read(original_path(campaign,profile['trial_directory'])/'inputs.json')
        if json.dumps(state['config'],sort_keys=True)!=json.dumps(trial['config'],sort_keys=True):
            raise ValueError('Checkpoint differs from original fitted configuration')
        expected={k:{r['path']:r['sha256'] for r in audit['inputs'][k]} for k in ('train','val')}
        if state['split_hashes']!=expected:raise ValueError('Checkpoint split hashes differ')
        runtime=state['runtime']
        if any(runtime.get(k)!=selected[k] for k in ('batch_size','edge_budget','precision')):
            raise ValueError('Checkpoint runtime differs from original calibration')
        if any(runtime.get(k)!=resolved.runtime[k] for k in ('fused_adamw','log_every_n_steps','strict_fp32')):
            raise ValueError('Checkpoint execution policy differs')
        result=validate_checkpoint(saved,state['config'],runtime,expected)
        if digest(path)!=before:raise ValueError('Checkpoint changed during read-only check')
        results[model]={**result,'checkpoint':str(path),'sha256':before,
            'best_sha256':digest(path.parent/'best.ckpt'),
            'audit_sha256':digest(base/'audits'/(model+'.json')),
            'profile_sha256':digest(base/'profiles'/(model+'.json'))}
    return dict(passed=True,original=str(Path(original).resolve()),original_revision=campaign['revision'],
                models=results,training_started=False,submitted=False,
                calibration_scope='Original revision calibration inherited; repaired GPU behavior not yet measured')


def relocate_checkpoint(saved,old_directory,new_directory):
    """Only ModelCheckpoint paths change, on a detached copy; no score resets."""
    result=copy.deepcopy(saved)
    old=Path(old_directory).resolve();new=Path(new_directory).resolve()
    def relocate(value):
        if not value:return value
        path=Path(value)
        if not path.is_absolute():path=old/path
        if path.resolve().parent!=old:raise ValueError('Callback path outside original run: '+str(path))
        return str(new/path.name)
    found=False
    for key,state in result['callbacks'].items():
        if not key.startswith('ModelCheckpoint'):continue
        found=True
        if Path(state['dirpath']).resolve()!=old:raise ValueError('Callback dirpath differs from original run')
        state['dirpath']=str(new)
        for field in ('best_model_path','kth_best_model_path','last_model_path'):
            if field in state:state[field]=relocate(state[field])
        state['best_k_models']={relocate(p):score for p,score in state.get('best_k_models',{}).items()}
    if not found:raise ValueError('No ModelCheckpoint callback state')
    return result


def prepare(original,name,wandb_mode='offline',wandb_project='UnifiedGroupGen'):
    import torch
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name):raise ValueError('Use a single campaign name')
    old=read(original)
    if name==old['name']:raise ValueError('Use a new recovery campaign name')
    folder=ROOT/'intermediate/hpc_runs'/name
    outputs={m:ROOT/'outputs'/m/name for m in selection(catalog())}
    if folder.exists() or any(p.exists() or p.is_symlink() for p in outputs.values()):
        raise FileExistsError('Recovery paths exist; inspect them or choose another name')
    subprocess.run(['git','diff','--quiet','HEAD','--','unifiedgroupgen','configs','scripts','submit_files'],cwd=ROOT,check=True)
    extra=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--','unifiedgroupgen','configs','scripts','submit_files'],cwd=ROOT,text=True)
    if extra.strip():raise ValueError('Commit runtime source before preparing recovery')
    report=check(original)
    revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    folder.mkdir(parents=True)
    source=snapshot(ROOT,folder,revision)
    for model,target in outputs.items():
        target.mkdir(parents=True,exist_ok=False)
        origin=original_output(old,model)
        for filename in ('best.ckpt','last.ckpt'):
            path=origin/filename
            saved=torch.load(path,map_location='cpu',weights_only=False)
            converted=relocate_checkpoint(saved,origin,target)
            torch.save(converted,target/filename)
        history=origin/'model.history.jsonl'
        if history.is_file():
            limit=report['models'][model]['global_step']
            with history.open(encoding='utf-8') as src,(target/history.name).open('x',encoding='utf-8') as dst:
                for line in src:
                    if json.loads(line)['global_step']<=limit:dst.write(line)
        if digest(origin/'last.ckpt')!=report['models'][model]['sha256']:
            raise ValueError('Original checkpoint changed while preparing recovery')
        report['models'][model]['recovery_checkpoint_sha256']=digest(target/'last.ckpt')
        report['models'][model]['recovery_best_checkpoint_sha256']=digest(target/'best.ckpt')
        if digest(origin/'best.ckpt')!=report['models'][model]['best_sha256']:
            raise ValueError('Original best checkpoint changed while preparing recovery')
    recovery=dict(schema=SCHEMA,name=name,root=str(ROOT),source=str(source),revision=revision,
        original=str(Path(original).resolve()),original_manifest_sha256=digest(original),
        compatibility=report,resources=old['resources'],models=list(outputs),
        logging=dict(mode=wandb_mode,project=wandb_project))
    write(folder/'campaign.json',recovery)
    print(json.dumps(recovery,ensure_ascii=False,indent=2))


def submit(manifest):
    from filelock import FileLock
    recovery=read(manifest)
    if recovery['schema']!=SCHEMA:raise ValueError('Not a repaired recovery campaign')
    folder=Path(manifest).resolve().parent
    with FileLock(str(folder/'submit.lock'),timeout=0):
        verify_recovery(recovery)
        (folder/'logs').mkdir(exist_ok=True)
        jobs_path=folder/'submitted.json';jobs=read(jobs_path) if jobs_path.exists() else {}
        for model in recovery['models']:
            if model in jobs:print('ALREADY SUBMITTED',model,jobs[model]);continue
            r=recovery['resources']
            command=['yhbatch','--parsable','--export=ALL','--job-name=ugg_resume_'+model,
                '--partition='+str(r['partition']),'--gres=gpu:'+str(r['gpus']),
                '--cpus-per-task='+str(r['cpus']),'--mem='+r['memory'],'--time='+str(r['hours'])+':00:00',
                '--chdir='+recovery['source'],'--output='+str(folder/'logs/%x-%j.out'),
                str(Path(recovery['source'])/'submit_files/joint_flow5090_resume.sbatch'),
                '--campaign',str(Path(manifest).resolve()),'--model',model]
            job=subprocess.check_output(command,cwd=ROOT,env={**os.environ,'UGG_GIT_COMMIT':recovery['revision']},text=True).strip().split(';')[0]
            if not job.isdigit():raise RuntimeError('Unexpected scheduler response: '+job)
            jobs[model]=job;write(jobs_path,jobs);print('QUEUED',model,job,flush=True)


def verify_recovery(recovery):
    if digest(recovery['original'])!=recovery['original_manifest_sha256']:raise ValueError('Original manifest changed')
    report=check(recovery['original'],recovery['models'])
    for model in recovery['models']:
        expected=recovery['compatibility']['models'][model]
        if any(report['models'][model][k]!=expected[k] for k in ('sha256','best_sha256','audit_sha256','profile_sha256')):
            raise ValueError('Original checkpoint/audit/profile changed since preparation')
        path=Path(recovery['root'])/'outputs'/model/recovery['name']/'last.ckpt'
        if digest(path)!=expected['recovery_checkpoint_sha256']:
            raise ValueError('Recovery checkpoint changed; implicit retries are disabled')
        if digest(path.parent/'best.ckpt')!=expected['recovery_best_checkpoint_sha256']:
            raise ValueError('Recovery best checkpoint changed')


def run(manifest,model):
    import torch
    from filelock import FileLock
    from unifiedgroupgen.pl_data.dataset import load_training_splits
    from unifiedgroupgen.pl_modules.joint_training import fit_joint
    recovery=read(manifest)
    if recovery['schema']!=SCHEMA or model not in recovery['models']:raise ValueError('Unexpected recovery model')
    if ROOT.resolve()!=Path(recovery['source']).resolve():raise ValueError('Worker must use repaired frozen source')
    folder=Path(manifest).resolve().parent
    with FileLock(str(folder/(model+'.run.lock')),timeout=0):
        verify_recovery({**recovery,'models':[model]})
        old=read(recovery['original']);output=Path(recovery['root'])/'outputs'/model/recovery['name']/'model.pt'
        resume=output.parent/'last.ckpt'
        saved=torch.load(resume,map_location='cpu',weights_only=False);j=saved['joint_state']
        validate_checkpoint(saved,j['config'],j['runtime'],j['split_hashes'])
        print(json.dumps(dict(event='recovery_verified',model=model,revision=recovery['revision'],
            original_revision=old['revision'],hardware=device_info(),epoch=saved['epoch'],global_step=saved['global_step']),ensure_ascii=False),flush=True)
        # The audit signs the raw frozen definition, before TRAIN vocabulary and
        # group-prior fitting. Keep that gate; fit_joint restores the fitted one.
        _,definition=original_definition(old,model)
        records,validation,_=load_training_splits(definition,Path(old['source']))
        if (len(records),len(validation))!=(j['runtime']['train_records'],j['runtime']['val_records']):
            raise ValueError('Full training split counts differ')
        torch.set_num_threads(4)
        from unifiedgroupgen.common.configuration import RunSettings
        from unifiedgroupgen.run import build_loggers
        logging=recovery.get('logging',{})
        settings=RunSettings.load(Path(recovery['root']),logging.get('mode','offline'))
        loggers=build_loggers(SimpleNamespace(experiment=model,wandb_project=logging.get('project','UnifiedGroupGen')),
                              output,settings,resume)
        success=False
        try:
            fit_joint(j['config'],records,validation,output,device=j['runtime']['device'],epochs=j['config']['epochs'],
                runtime=j['runtime'],split_hashes=j['split_hashes'],resume=resume,
                max_steps=j['config'].get('joint_max_steps',-1),logger=loggers or False)
            success=True
        finally:
            for logger in loggers:logger.finalize('success' if success else 'failed')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='action',required=True)
    p=sub.add_parser('check');p.add_argument('--original',required=True)
    p=sub.add_parser('prepare');p.add_argument('--original',required=True);p.add_argument('--name',required=True)
    p.add_argument('--wandb-mode',choices=['offline','online','disabled'],default='offline')
    p.add_argument('--wandb-project',default='UnifiedGroupGen')
    p=sub.add_parser('submit');p.add_argument('--campaign',required=True)
    p=sub.add_parser('run');p.add_argument('--campaign',required=True);p.add_argument('--model',required=True)
    args=parser.parse_args()
    if args.action=='check':print(json.dumps(check(args.original),ensure_ascii=False,indent=2))
    elif args.action=='prepare':prepare(args.original,args.name,args.wandb_mode,args.wandb_project)
    elif args.action=='submit':submit(args.campaign)
    else:run(args.campaign,args.model)


if __name__=='__main__':main()
