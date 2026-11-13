"""Plan or submit full joint-flow runs; only the explicit worker starts training."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from submit_files.hpc5090 import snapshot, read, write

CATALOG='configs/experiments_joint_flow_full.yaml'


def catalog():
    import yaml
    return yaml.safe_load((ROOT/CATALOG).read_text(encoding='utf-8'))


def selection(data,models=None):
    models=list(data['default_models'] if models is None else models)
    if not models or len(models)!=len(set(models)) or set(models)-set(data['experiments']):
        raise ValueError('Select distinct model names from the full-flow catalog')
    return models


def plan(models=None):
    data=catalog();chosen=selection(data,models)
    return dict(selected=chosen,jobs=len(chosen),phase='audit_calibrate_train',calibration=data['calibration'],
                resources=data['resources'],available=data['experiments'],
                local_training_started=False)


def output(campaign,model):
    return Path(campaign['root'])/'outputs'/model/campaign['name']/'model.pt'


def complete(campaign,model):
    path=output(campaign,model);status=path.with_suffix('.status.json')
    if not status.exists() or read(status).get('state')!='complete':return False
    if not all((path.parent/name).is_file() for name in ('model.pt','last.ckpt','best.ckpt')):
        raise ValueError('Completed run lacks required checkpoints: '+str(path.parent))
    return True


def audit_command(campaign,model):
    entry=campaign['entries'][model];selector=entry['experiment']
    audit='data.data_audit=intermediate/hpc_runs/'+campaign['name']+'/audits/'+model+'.json'
    return [sys.executable,'-u','-m','unifiedgroupgen.joint','audit','--experiment',selector,'--override',audit]


def commands(campaign,model):
    audit=audit_command(campaign,model)
    selector=campaign['entries'][model]['experiment']
    path=output(campaign,model);last=path.parent/'last.ckpt'
    train_command=[sys.executable,'-u','-m','unifiedgroupgen.run','experiment='+selector,
                   audit[-1],'device=cuda:0','seed=42','threads=4','logging.mode=offline',
                   'output='+str(path)]
    profile=Path(campaign['root'])/'intermediate/hpc_runs'/campaign['name']/'profiles'/(model+'.json')
    if profile.is_file():
        from submit_files.joint_flow5090_calibrate import identity,validate_profile
        audit_report=read(profile.parent.parent/'audits'/(model+'.json'))
        chosen=validate_profile(read(profile),identity(campaign,model,audit_report))
        train_command+=['runtime.batch_size='+str(chosen['batch_size']),'runtime.edge_budget='+str(chosen['edge_budget'])]
    elif last.is_file():raise ValueError('Resume requires the original batch calibration profile')
    if last.is_file():train_command.append('resume='+str(last))
    elif any((path.parent/name).exists() for name in ('model.pt','best.ckpt','model.history.jsonl','model.status.json')):
        raise ValueError('Existing run has no resumable last.ckpt; use a new campaign: '+str(path.parent))
    return audit,train_command


def submit(name,models=None,retry=False):
    from filelock import FileLock
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name):
        raise ValueError('Use a single alphanumeric campaign name')
    base=ROOT/'intermediate/hpc_runs';base.mkdir(parents=True,exist_ok=True)
    folder=base/name;manifest=folder/'campaign.json'
    with FileLock(str(base/(name+'.submit.lock')),timeout=0):
        if manifest.exists():
            campaign=read(manifest)
            if campaign.get('schema')!='ugg_joint_flow_full_5090_v2':
                raise ValueError('Use a new campaign name for FP32/calibrated training; old frozen runs are preserved')
            if models is not None and list(models)!=campaign['models']:
                raise ValueError('Selection is frozen; use a new campaign name')
        else:
            if folder.exists():raise FileExistsError('Partial campaign exists; inspect it or use a new name')
            data=catalog();chosen=selection(data,models)
            tracked=['unifiedgroupgen','configs','scripts','submit_files']
            subprocess.run(['git','diff','--quiet','HEAD','--',*tracked],cwd=ROOT,check=True)
            untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*tracked],cwd=ROOT,text=True)
            if untracked.strip():raise ValueError('Commit runtime source before freezing: '+untracked.strip())
            revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            folder.mkdir();source=snapshot(ROOT,folder,revision)
            campaign=dict(schema='ugg_joint_flow_full_5090_v2',name=name,root=str(ROOT),
                          source=str(source),revision=revision,models=chosen,
                          entries={m:data['experiments'][m] for m in chosen},resources=data['resources'],calibration=data['calibration'])
            write(manifest,campaign)
        (folder/'logs').mkdir(exist_ok=True)
        jobs_path=folder/'submitted.json'
        jobs=read(jobs_path) if jobs_path.exists() else {}
        for model in campaign['models']:
            if model in jobs and not retry:
                print('ALREADY SUBMITTED',model,jobs[model],flush=True);continue
            if complete(campaign,model):
                print('ALREADY COMPLETE',model,flush=True);continue
            resource=campaign['resources']
            command=['yhbatch','--parsable','--export=ALL','--job-name=ugg_'+model,
                     '--partition='+str(resource['partition']),'--gres=gpu:'+str(resource['gpus']),
                     '--cpus-per-task='+str(resource['cpus']),'--mem='+resource['memory'],
                     '--chdir='+campaign['source'],'--time='+str(resource['hours'])+':00:00',
                     '--output='+str(folder/'logs/%x-%j.out'),
                     str(Path(campaign['source'])/'submit_files/joint_flow5090.sbatch'),
                     '--campaign',str(manifest),'--model',model]
            job=subprocess.check_output(command,cwd=ROOT,
                env={**os.environ,'UGG_GIT_COMMIT':campaign['revision']},text=True).strip().split(';')[0]
            if not job.isdigit():raise RuntimeError('Unexpected yhbatch job ID: '+job)
            jobs[model]=job;write(jobs_path,jobs)
            with (folder/'submission_history.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(dict(model=model,job=job,retry=retry,
                    utc=datetime.now(timezone.utc).isoformat()))+'\n')
            print('QUEUED',model,job,flush=True)
        print('Frozen revision:',campaign['revision'])
        print('Campaign:',manifest)


def device_info():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Full-flow 5090 profile requires the allocated CUDA GPU')
    return dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),
                gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)


def run_worker(campaign_path,model):
    from filelock import FileLock
    campaign=read(campaign_path)
    if campaign.get('schema')!='ugg_joint_flow_full_5090_v2' or model not in campaign['models']:
        raise ValueError('Model is not selected in this full-flow campaign')
    if ROOT.resolve()!=Path(campaign['source']).resolve():
        raise ValueError('Run workers from the frozen source directory')
    folder=Path(campaign_path).parent;(folder/'locks').mkdir(exist_ok=True)
    with FileLock(str(folder/'locks'/(model+'.lock')),timeout=0):
        if complete(campaign,model):
            print('Training already complete:',model);return
        audit=audit_command(campaign,model)
        print(json.dumps(dict(model=model,revision=campaign['revision'],hardware=device_info()),ensure_ascii=False),flush=True)
        print('AUDIT COMMAND',json.dumps(audit),flush=True)
        subprocess.run(audit,cwd=ROOT,check=True)
        certificate=folder/'audits'/(model+'.json')
        if not certificate.is_file() or not read(certificate).get('passed'):
            raise RuntimeError('No successful campaign data audit: '+str(certificate))
        profile=folder/'profiles'/(model+'.json')
        if not profile.is_file() or read(profile).get('state')!='complete':
            if (output(campaign,model).parent/'last.ckpt').exists():
                raise ValueError('Cannot recalibrate an existing training run; restore its original profile')
            subprocess.run([sys.executable,'-u',str(ROOT/'submit_files/joint_flow5090_calibrate.py'),
                            'calibrate','--campaign',str(campaign_path),'--model',model],cwd=ROOT,check=True)
        if not profile.is_file():raise RuntimeError('Calibration produced no profile; training is blocked')
        _,train=commands(campaign,model)
        print('TRAIN COMMAND',json.dumps(train),flush=True)
        subprocess.run(train,cwd=ROOT,check=True)
        if not complete(campaign,model):
            raise RuntimeError('Training ended without complete status and checkpoints')


def main():
    parser=argparse.ArgumentParser(description=__doc__);commands_parser=parser.add_subparsers(dest='action',required=True)
    p=commands_parser.add_parser('plan');p.add_argument('--models',nargs='+')
    p=commands_parser.add_parser('submit');p.add_argument('--name',required=True);p.add_argument('--models',nargs='+')
    p.add_argument('--retry',action='store_true',help='After old jobs end, resubmit incomplete models using frozen source and last.ckpt')
    p=commands_parser.add_parser('run');p.add_argument('--campaign',required=True);p.add_argument('--model',required=True)
    args=parser.parse_args()
    if args.action=='plan':print(json.dumps(plan(args.models),ensure_ascii=False,indent=2))
    elif args.action=='submit':submit(args.name,args.models,args.retry)
    else:run_worker(args.campaign,args.model)


if __name__=='__main__':main()
