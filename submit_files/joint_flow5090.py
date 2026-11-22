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
from submit_files.common import snapshot, read, write

CATALOG='configs/experiments_joint_flow_full.yaml'
CAMPAIGN_SCHEMA='ugg_joint_flow_full_5090_v3'


def catalog():
    import yaml
    return yaml.safe_load((ROOT/CATALOG).read_text(encoding='utf-8'))


def selection(data,models=None):
    models=list(data['default_models'] if models is None else models)
    if not models or len(models)!=len(set(models)) or set(models)-set(data['experiments']):
        raise ValueError('Select distinct model names from the full-flow catalog')
    return models


def main_selection(data, models=None):
    allowed = data['paper']['main_models']
    chosen = selection(data, allowed if models is None else models)
    if set(chosen) - set(allowed):
        raise ValueError('Main experiments are full EquiformerV2 DNG/CSP on MP-20, C2DB-51 or both')
    return chosen


def planned_entries(data, models=None, paper=None):
    if paper is None:
        chosen = selection(data, models)
        comparison = data['backbone_comparison']
        paired = set(chosen) & set(comparison['models'])
        if paired and paired != set(chosen):
            raise ValueError('Select the backbone comparison and the other models in separate campaigns')
        entries = {name: data['experiments'][name] for name in chosen}
        if not paired:
            return entries, data['calibration']
        sizes = comparison['batch_sizes']
        entries = {name: {**entry, 'batch_size': sizes[entry['backbone']]} for name, entry in entries.items()}
        candidates = sorted({entry['batch_size'] for entry in entries.values()})
        return entries, {**data['calibration'], 'candidates': candidates}
    settings = data['paper']
    chosen = selection(data, settings['models'] if models is None else models)
    if set(chosen) - set(settings['models']):
        raise ValueError('The paired matrix supports the four MP-20/C2DB-51 EquiformerV2 tasks')
    seeds = paper.get('seeds') or settings['seeds']
    if len(set(seeds)) != len(seeds) or not all(0 <= seed < 2**32 for seed in seeds):
        raise ValueError('Select distinct unsigned 32-bit training seeds')
    max_steps, batch_size = (paper[k] for k in ('max_steps','batch_size'))
    if min(max_steps, batch_size) < 1:
        raise ValueError('Paired runs need positive max steps and batch size')
    entries = {}
    for mode in settings['ablations']:
        for model in chosen:
            for seed in seeds:
                name = f'{model}_{mode}_s{seed}'
                entries[name] = {**data['experiments'][model], 'seed': seed, 'ablation': mode, 'max_steps': max_steps}
    # Each arm must pass the same candidate. The worker may never pick a different runtime.
    return entries, {**data['calibration'], 'candidates': [batch_size]}


def entry_overrides(entry):
    if 'ablation' not in entry:
        return []
    return [f"ablation={entry['ablation']}", f"train.joint_max_steps={entry['max_steps']}"]


def run_overrides(campaign, model):
    certificate = f"intermediate/hpc_runs/{campaign['name']}/audits/{model}.json"
    return [*entry_overrides(campaign['entries'][model]), 'data.data_audit='+certificate]


def plan(models=None, paper=None):
    data=catalog();entries,policy=planned_entries(data,models,paper)
    result = dict(selected=list(entries),jobs=len(entries),phase='audit_calibrate_train',calibration=policy,
                  resources=data['resources'],available=data['experiments'],local_training_started=False)
    if paper is not None or any('batch_size' in entry for entry in entries.values()):
        result['entries'] = entries
    return result


def output(campaign,model):
    return Path(campaign['root'])/'outputs'/model/campaign['name']/'model.pt'


def complete(campaign,model):
    path=output(campaign,model);status=path.with_suffix('.status.json')
    state = read(status) if status.exists() else {}
    if state.get('state')!='complete':return False
    budget = campaign['entries'][model].get('max_steps')
    if budget is not None and state.get('global_step') != budget:
        raise ValueError('Paired run finished outside the required optimization budget: '+str(path.parent))
    if not all((path.parent/name).is_file() for name in ('model.pt','last.ckpt','best.ckpt')):
        raise ValueError('Completed run lacks required checkpoints: '+str(path.parent))
    return True


def audit_command(campaign,model):
    entry = campaign['entries'][model]
    overrides = [argument for item in run_overrides(campaign, model) for argument in ('--override', item)]
    return [sys.executable,'-u','-m','unifiedgroupgen.joint','audit','--experiment',entry['experiment'],*overrides]


def commands(campaign,model):
    audit=audit_command(campaign,model)
    entry=campaign['entries'][model]
    path=output(campaign,model);last=path.parent/'last.ckpt'
    train_command=[sys.executable,'-u','-m','unifiedgroupgen.run','experiment='+entry['experiment'],
                   *run_overrides(campaign,model),'device=cuda:0',f"seed={entry.get('seed',42)}",'threads=4',
                   'logging.mode=offline','output='+str(path)]
    profile=Path(campaign['root'])/'intermediate/hpc_runs'/campaign['name']/'profiles'/(model+'.json')
    if profile.is_file():
        from submit_files.joint_flow5090_calibrate import identity,validate_profile
        audit_report=read(profile.parent.parent/'audits'/(model+'.json'))
        chosen=validate_profile(read(profile),identity(campaign,model,audit_report))
        train_command+=['runtime.batch_size='+str(chosen['batch_size'])]
    elif last.is_file():raise ValueError('Resume requires the original batch calibration profile')
    if last.is_file():train_command.append('resume='+str(last))
    elif any((path.parent/name).exists() for name in ('model.pt','best.ckpt','model.history.jsonl','model.status.json')):
        raise ValueError('Existing run has no resumable last.ckpt; use a new campaign: '+str(path.parent))
    return audit,train_command


def submit(name,models=None,retry=False,paper=None):
    from filelock import FileLock
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*',name):
        raise ValueError('Use a single alphanumeric campaign name')
    base=ROOT/'intermediate/hpc_runs';base.mkdir(parents=True,exist_ok=True)
    folder=base/name;manifest=folder/'campaign.json'
    with FileLock(str(base/(name+'.submit.lock')),timeout=0):
        if manifest.exists():
            campaign=read(manifest)
            if campaign.get('schema')!=CAMPAIGN_SCHEMA:
                raise ValueError('Use a new campaign name for fixed-size batches or retry with the original frozen submitter')
            if paper is not None:
                entries, policy = planned_entries(catalog(), models, paper)
                if list(entries) != campaign['models'] or entries != campaign['entries'] or policy != campaign['calibration']:
                    raise ValueError('Paired settings are frozen; use a new campaign name')
            elif models is not None and list(models)!=campaign['models']:
                raise ValueError('Selection is frozen; use a new campaign name')
        else:
            if folder.exists():raise FileExistsError('Partial campaign exists; inspect it or use a new name')
            data=catalog();entries,policy=planned_entries(data,models,paper)
            tracked=['unifiedgroupgen','configs','scripts','submit_files']
            subprocess.run(['git','diff','--quiet','HEAD','--',*tracked],cwd=ROOT,check=True)
            untracked=subprocess.check_output(['git','ls-files','--others','--exclude-standard','--',*tracked],cwd=ROOT,text=True)
            if untracked.strip():raise ValueError('Commit runtime source before freezing: '+untracked.strip())
            revision=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
            folder.mkdir();source=snapshot(ROOT,folder,revision)
            campaign=dict(schema=CAMPAIGN_SCHEMA,name=name,root=str(ROOT),
                          source=str(source),revision=revision,models=list(entries),
                          entries=entries,resources=data['resources'],calibration=policy)
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
                env={**os.environ,'UGG_GIT_COMMIT':campaign['revision'],'UGG_PROJECT_ROOT':campaign['root']},text=True).strip().split(';')[0]
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
    if campaign.get('schema')!=CAMPAIGN_SCHEMA or model not in campaign['models']:
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
    for action in ('plan', 'submit'):
        p=commands_parser.add_parser(action)
        p.add_argument('--models',nargs='+')
        mode=p.add_mutually_exclusive_group()
        mode.add_argument('--main',action='store_true',help='Select the six full-model runs at seed 42')
        mode.add_argument('--backbones',action='store_true',help='Select the four matched MP-20 backbone/task runs')
        mode.add_argument('--paper',action='store_true',help='Use the matched EquiformerV2 ablation matrix')
        p.add_argument('--seeds',nargs='+',type=int)
        p.add_argument('--max-steps',type=int)
        p.add_argument('--batch-size',type=int)
        if action == 'submit':
            p.add_argument('--name',required=True)
            p.add_argument('--retry',action='store_true',help='Reuse the frozen campaign after its old jobs end')
    p=commands_parser.add_parser('run');p.add_argument('--campaign',required=True);p.add_argument('--model',required=True)
    args=parser.parse_args()
    if args.action=='run':
        return run_worker(args.campaign,args.model)
    paper=None
    if args.paper:
        if None in (args.max_steps,args.batch_size):
            parser.error('--paper requires --max-steps and --batch-size')
        paper=dict(seeds=args.seeds,max_steps=args.max_steps,batch_size=args.batch_size)
    elif any(value is not None for value in (args.seeds,args.max_steps,args.batch_size)):
        parser.error('Budget and seed overrides require --paper')
    models=main_selection(catalog(),args.models) if args.main else args.models
    if args.backbones:
        if args.models is not None:
            parser.error('--backbones selects the four matching configurations without --models')
        models=catalog()['backbone_comparison']['models']
    if args.action=='plan':
        print(json.dumps(plan(models,paper),ensure_ascii=False,indent=2))
    else:
        submit(args.name,models,args.retry,paper)


if __name__=='__main__':main()
