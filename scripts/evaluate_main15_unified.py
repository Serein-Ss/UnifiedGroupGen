"""Sequential, resumable evaluation of the approved 15 checkpoints. Default: PLAN ONLY."""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
import numpy as np
import yaml

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from unifiedgroupgen.main_experiments import TRAINING_ORDER
from unifiedgroupgen.evaluation_io import read_records, sha256, write_json


TASKS=('components','dng','csp','fixed_descriptor','group')


def build_jobs(config, count, cases, k, conditional_count, seed, group_count=0,
               tasks=None, conditional_tasks=None, target_properties=None):
    jobs=[{'name':'components','task':'components'}]
    tests=[r for p in config['test_files'] for r in read_records(ROOT/p)]
    jobs[0]['test_records']=len(tests)
    train=[r for p in config['train_files'] for r in read_records(ROOT/p)]
    domains=config.get('dataset_domains') or {config['dataset']:None}
    properties=[p for p,spec in config.get('properties',{}).items() if not spec.get('context')]
    if target_properties is not None:
        properties=[p for p in properties if p in target_properties]
    for dataset,domain in domains.items():
        kind='layer' if dataset=='c2db_51' else 'space'
        context={'source_domain':domain} if domain is not None else {}
        selected=sorted((r for r in tests if r['dataset']==dataset),key=lambda r:hashlib.sha256(f"{seed}:{r['id']}".encode()).hexdigest())
        if cases: selected=selected[:cases]
        base={'dataset':dataset,'kind':kind,'properties':context,'count':count,'case_ids':[str(r['id']) for r in selected],'k':k}
        def add(suffix,task,targets=None,**extra):
            job={**base,'name':dataset+'_'+suffix,'task':task,'properties':{**context,**(targets or {})},**extra}
            if task!='csp':
                job.pop('case_ids');job.pop('k')
            jobs.append(job)
        add('dng','dng');add('csp','csp');add('fixed_descriptor','csp',fixed_descriptor=True)
        if group_count:
            groups=list(range(1,81 if kind=='layer' else 231))
            add('group_sweep','dng',groups=groups,per_group=group_count,count=len(groups)*group_count)
        domain_train=[r for r in train if r['dataset']==dataset]
        for p in properties:
            values=[r['properties'][p] for r in domain_train if r['properties'].get(p) is not None and np.isfinite(r['properties'][p])]
            for i,value in enumerate(sorted(set(map(float,np.quantile(values,[.1,.5,.9])))) if values else []):
                target={p:value}
                add(f'dng_{p}_q{i}','dng',target,count=conditional_count,conditional=True)
                add(f'csp_{p}_q{i}','csp',target,conditional=True)
        if len(properties)>1:
            complete=sorted([r for r in domain_train if all(r['properties'].get(p) is not None and np.isfinite(r['properties'][p]) for p in properties)],
                            key=lambda r:(r['properties'][properties[0]],str(r['id'])))
            for i,q in enumerate((.1,.5,.9)) if complete else []:
                r=complete[round(q*(len(complete)-1))];target={p:r['properties'][p] for p in properties}
                add(f'dng_joint_q{i}','dng',target,count=conditional_count,conditional=True)
                add(f'csp_joint_q{i}','csp',target,conditional=True)
    selected=set(tasks or TASKS)
    conditional_tasks=set(conditional_tasks if conditional_tasks is not None else ('dng','csp'))
    def selected_job(job):
        category=('components' if job['task']=='components' else 'group' if job.get('group') or job.get('groups') else
                  'fixed_descriptor' if job.get('fixed_descriptor') else job['task'])
        return category in selected and (not job.get('conditional') or job['task'] in conditional_tasks)
    return [j for j in jobs if selected_job(j)]


def run_command(command,log,env):
    with log.open('a',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u',*map(str,command)],cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def verify_files(root,outputs):
    for name,expected in outputs.items():
        path=root/name
        if not path.is_file() or sha256(path)!=expected:
            raise ValueError(f'Saved output missing or modified: {path}')


def output_hashes(root):
    return {str(p.relative_to(root)):sha256(p) for p in root.rglob('*')
            if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.png','.cif')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=TRAINING_ORDER,default=list(TRAINING_ORDER))
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS))
    parser.add_argument('--conditional-tasks',nargs='*',choices=['dng','csp'],default=['dng','csp'])
    parser.add_argument('--target-properties',nargs='+',choices=['formation_energy','band_gap','energy_above_hull'])
    parser.add_argument('--stage',choices=['all','generate','metrics'],default='all')
    parser.add_argument('--verify-generated',action='store_true')
    parser.add_argument('--checkpoint-root',type=Path,default=ROOT/'artifacts/runs/main15_seed42')
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/evaluations/main15_unified_v2_seed142')
    parser.add_argument('--count',type=int,default=1024)
    parser.add_argument('--csp-cases',type=int,default=0,help='0: complete test set (default); positive: explicitly limited diagnostic subset')
    parser.add_argument('--csp-k',type=int,default=20)
    parser.add_argument('--conditional-count',type=int,default=1024)
    parser.add_argument('--per-group',type=int,default=0)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--seed',type=int,default=142)
    parser.add_argument('--guidance',type=float,default=2.)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--test-batch-size',type=int,default=8)
    parser.add_argument('--cgdit-root',type=Path)
    parser.add_argument('--predictor-root',type=Path)
    parser.add_argument('--predictor-seed',type=int,default=123)
    parser.add_argument('--allow-proxy-transfer',action='store_true')
    parser.add_argument('--no-fingerprints',action='store_true')
    parser.add_argument('--plan',action='store_true')
    parser.add_argument('--run',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--status',action='store_true')
    args=parser.parse_args()
    if args.status:
        state=json.loads((args.output/'pipeline.json').read_text(encoding='utf-8'))
        result={'state':state['state'],'pid':state['pid'],'current':state.get('current'),
            'stage':state.get('stage'),'phase':state.get('phase'),
            'complete':sum(r['state']=='complete' for r in state['jobs']),'total':len(state['identity']['jobs']),
            'generated_jobs':sum(r.get('generation',{}).get('state')=='complete' for r in state['jobs']),
            'generation_jobs_total':sum(j['task']!='components' for j in state['identity']['jobs']),
            'last_completed':[{'name':r['name'],'seconds':r.get('seconds')} for r in state['jobs'] if r['state']=='complete'][-5:],
            'error':state.get('error')}
        status=args.output/state.get('current','')/'sampling.status.json'
        if status.exists(): result['sampling']=json.loads(status.read_text(encoding='utf-8'))
        print(json.dumps(result,ensure_ascii=False,indent=2));return
    if min(args.count,args.csp_k,args.conditional_count,args.steps,args.threads,args.test_batch_size)<1 or min(args.per_group,args.csp_cases)<0:
        parser.error('Budgets must be positive; per-group and csp-cases may be zero')
    if bool(args.cgdit_root)!=bool(args.predictor_root): parser.error('Provide both predictor paths')
    plan=[]
    for name in args.models:
        checkpoint=args.checkpoint_root/name/'model.best.pt'
        if checkpoint.exists():
            saved=torch.load(checkpoint,map_location='cpu',weights_only=False)
            config=saved['config']
            if config.get('experiment_id')!=name: raise ValueError('Checkpoint/model identity mismatch')
            checkpoint_hash=sha256(checkpoint)
            if args.run and not args.plan:
                from unified_eval_inference import checked_config
                checked_config(saved)
        else:
            config=yaml.safe_load((ROOT/f'configs/main15/{name}.yaml').read_text(encoding='utf-8'))
            checkpoint_hash=None
        data_hashes={p:sha256(ROOT/p) for split in ('train','val','test') for p in config[split+'_files']}
        for job in build_jobs(config,args.count,args.csp_cases,args.csp_k,args.conditional_count,args.seed,args.per_group,
                              args.tasks,args.conditional_tasks,args.target_properties):
            plan.append({'model':name,'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_hash,'data_sha256':data_hashes,**job})
    total=sum((len(j['case_ids'])*j['k'] if j['task']=='csp' else j.get('count',0)) for j in plan if j['task']!='components')
    options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('run','resume','plan','status','stage','verify_generated')}
    identity={'options':options,'jobs':plan,'code':{str(p.relative_to(ROOT)):sha256(p) for p in
        [Path(__file__),ROOT/'scripts/unified_eval_inference.py',ROOT/'scripts/evaluate_structures.py',ROOT/'scripts/unified_eval_properties.py',
         *sorted((ROOT/'src/unifiedgroupgen').glob('evaluation_*.py')),ROOT/'src/unifiedgroupgen/unified_evaluation.py',
         ROOT/'src/unifiedgroupgen/standard_metrics.py',ROOT/'scripts/legacy_eval/properties.py',
         *[ROOT/'src/unifiedgroupgen'/n for n in ('flow.py','proposal.py','symmetry.py','constraints.py','conditioning.py','chemical_controls.py','representation_audit.py','evaluation.py')]]}}
    if args.verify_generated:
        saved=json.loads((args.output/'pipeline.json').read_text(encoding='utf-8'))
        if saved['identity']!=identity: raise ValueError('Generation verification requires identical model/data/options/code')
        generated=failed=cifs=0
        for job in plan:
            if job['task']=='components': continue
            key=job['model']+'/'+job['name'];target=args.output/key
            entry=next((r for r in saved['jobs'] if r['name']==key),{})
            receipt=entry.get('generation',{})
            if receipt.get('state')!='complete': raise ValueError(f'Generation incomplete: {key}')
            verify_files(target,receipt['outputs'])
            records=read_records(target/'samples.jsonl')
            generated+=sum(r['status']=='generated' for r in records)
            failed+=sum(r['status']!='generated' for r in records)
            cifs+=sum(p.endswith('.cif') for p in receipt['outputs'])
        print(json.dumps({'state':'verified','requested':generated+failed,'generated':generated,'failed':failed,
            'cif_files_checked':cifs,'regeneration_started':False},indent=2));return
    if not args.run or args.plan:
        args.output.mkdir(parents=True,exist_ok=True)
        write_json(args.output/'plan.json',{'identity':identity,'generation_attempts':total,'formal_tasks_started':False})
        with (args.output/'plan.csv').open('w',newline='',encoding='utf-8-sig') as stream:
            fields=['model','job','task','dataset','kind','test_records','candidates_per_case','requested_structures','guidance','properties']
            writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
            for j in plan:
                writer.writerow({'model':j['model'],'job':j['name'],'task':j['task'],'dataset':j.get('dataset'),
                    'kind':j.get('kind'),'test_records':len(j.get('case_ids',[])) if j['task']=='csp' else j.get('test_records'),
                    'candidates_per_case':j.get('k') if j['task']=='csp' else None,
                    'requested_structures':len(j['case_ids'])*j['k'] if j['task']=='csp' else j.get('count',0),
                    'guidance':args.guidance if j.get('conditional') else 0.,'properties':json.dumps(j.get('properties',{}),sort_keys=True)})
        print(json.dumps({'models':len(args.models),'jobs':len(plan),'generation_attempts':total,'plan':str(args.output/'plan.json'),
                          'checkpoints_missing':sorted({j['model'] for j in plan if j['checkpoint_sha256'] is None}),
                          'formal_tasks_started':False},ensure_ascii=False,indent=2));return
    if any(j['checkpoint_sha256'] is None for j in plan): raise ValueError('Missing checkpoints; only a plan was possible')
    if args.device.startswith('cuda'):
        # Refuse competing GPU work; never stop or alter another task.
        check=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        if check.stdout.strip(): raise RuntimeError('GPU has active compute processes; evaluate later or use --device cpu')
    args.output.mkdir(parents=True,exist_ok=True)
    pipeline=args.output/'pipeline.json'
    state={'state':'running','pid':os.getpid(),'identity':identity,'jobs':[],'generation_attempts':total,'stage':args.stage}
    if pipeline.exists():
        old=json.loads(pipeline.read_text(encoding='utf-8'))
        if not args.resume or old['identity']!=identity: raise ValueError('Use --resume with identical inputs/options or a new output directory')
        state['jobs']=old['jobs']
    if args.stage=='metrics':
        for job in plan:
            if job['task']=='components': continue
            key=job['model']+'/'+job['name']
            entry=next((r for r in state['jobs'] if r['name']==key),{})
            if entry.get('generation',{}).get('state')!='complete':
                raise ValueError(f'Generate all requested structures first: {key}; metrics stage never samples')
            verify_files(args.output/key,entry['generation']['outputs'])
    lock=args.output/'evaluation.lock'
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'MKL_THREADING_LAYER':'SEQUENTIAL',
         'OMP_NUM_THREADS':str(args.threads),'MKL_NUM_THREADS':str(args.threads),'PYTHONIOENCODING':'utf-8'}
    try:
        for number,job in enumerate(plan):
            key=job['model']+'/'+job['name'];target=args.output/key
            if args.stage=='generate' and job['task']=='components': continue
            old=next((r for r in state['jobs'] if r['name']==key),None)
            if old and old['state']=='complete':
                verify_files(target,old['outputs'])
                continue
            target.mkdir(parents=True,exist_ok=True)
            entry=old or {'name':key}
            if old is None: state['jobs'].append(entry)
            entry.update(state='running',started=time.time());state.update(current=key,updated=time.time());write_json(pipeline,state)
            write_json(target/'job.json',job)
            print('[START] '+key,flush=True)
            scale=args.guidance if job.get('conditional') else 0.
            inference=['scripts/unified_eval_inference.py','--checkpoint',job['checkpoint'],'--job',target/'job.json','--output',target,
                '--device',args.device,'--seed',args.seed,'--steps',args.steps,'--guidance',scale,'--threads',args.threads,'--batch-size',args.test_batch_size]
            if job['task']=='components':
                state['phase']='components';write_json(pipeline,state)
                run_command(inference,target/'inference.log',env)
            elif args.stage in ('all','generate'):
                receipt=entry.get('generation',{})
                if receipt.get('state')=='complete':
                    verify_files(target,receipt['outputs'])
                else:
                    state['phase']='generate';write_json(pipeline,state)
                    generation_started=time.perf_counter()
                    run_command(inference,target/'inference.log',env)
                    entry['generation']={'state':'complete','seconds':time.perf_counter()-generation_started,
                                         'outputs':output_hashes(target)}
                entry['state']='generated';state['updated']=time.time();write_json(pipeline,state)
            if args.stage=='generate':
                print('[GENERATED] '+key,flush=True);continue
            evaluation_started=time.perf_counter()
            if job['task']!='components':
                config=torch.load(job['checkpoint'],map_location='cpu',weights_only=False)['config']
                property_args=[]
                if args.predictor_root and (job['dataset']=='mp_20' or args.allow_proxy_transfer):
                    state['phase']='properties';write_json(pipeline,state)
                    command=['scripts/unified_eval_properties.py','--samples',target/'samples.jsonl','--dataset',job['dataset'],
                        '--cgdit-root',args.cgdit_root,'--predictor-root',args.predictor_root,'--predictor-seed',args.predictor_seed,'--output',target/'proxy']
                    command+=['--reference-files',*config['test_files'],'--reference-cache',args.output/'proxy_reference_cache']
                    if args.allow_proxy_transfer: command+=['--allow-proxy-transfer']
                    run_command(command,target/'properties.log',env)
                    property_args=['--assessments',target/'proxy/predictions.csv','--assessment-metadata',target/'proxy/evaluator.json',
                                   '--reference-assessments',target/'proxy/reference_predictions.csv']
                request={'properties':{p:v for p,v in job['properties'].items() if p!='source_domain'},'group':job.get('group')}
                if job.get('groups'): request['sampling_task']='balanced_group_sweep'
                if job['task']=='csp': request.update(case_ids=job['case_ids'],attempts_per_case=job['k'])
                write_json(target/'request.json',request)
                expected=len(job['case_ids'])*job['k'] if job['task']=='csp' else job['count']
                state['phase']='metrics';write_json(pipeline,state)
                run_command(['scripts/evaluate_structures.py','--samples',target/'samples.jsonl','--task',job['task'],'--kind',job['kind'],
                    '--dataset',job['dataset'],'--reference-files',*config['test_files'],'--novelty-files',*config['train_files'],
                    '--request',target/'request.json','--output',target/'metrics','--cache',args.output/'reference_cache',
                    '--expected-attempts',expected,'--resume',*property_args,*(['--no-fingerprints'] if args.no_fingerprints else [])],target/'metrics.log',env)
            evaluation_seconds=(time.time()-entry['started'] if job['task']=='components' else time.perf_counter()-evaluation_started)
            entry.update(state='complete',evaluation_seconds=evaluation_seconds,
                seconds=entry.get('generation',{}).get('seconds',0)+evaluation_seconds,outputs=output_hashes(target))
            state['updated']=time.time();write_json(pipeline,state);print('[DONE] '+key,flush=True)
        if args.stage=='generate':
            state['state']='complete' if len(state['jobs'])==len(plan) and all(j['state']=='complete' for j in state['jobs']) else 'generation_complete'
        else:
            state['state']='complete'
            from unifiedgroupgen.evaluation_reporting import campaign_summary
            campaign_summary(args.output,state)
    except BaseException as error:
        state.update(state='failed',error=str(error));raise
    finally:
        state['updated']=time.time();write_json(pipeline,state);lock.unlink()


if __name__=='__main__': main()
