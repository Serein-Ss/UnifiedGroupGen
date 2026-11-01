"""Coordinate generation, component checks and metrics via separate entry points."""
from pathlib import Path
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
from unifiedgroupgen.common.configuration import project_root, resolve_path, source_key, load_experiments, RunSettings, runtime_pythonpath, task_options, model_result_directory
from unifiedgroupgen.common.source import package_sources
from unifiedgroupgen.common.structures import inline_cif_bytes, read_records, sha256, write_json
from unifiedgroupgen.common.reports import campaign_summary
from unifiedgroupgen.pl_modules.factory import load_model_checkpoint
ROOT=project_root()
TASKS=('components','dng','csp','fixed_descriptor','group')

def build_jobs(config, count, cases, k, conditional_count, seed, group_count=0,
               tasks=None, conditional_tasks=None, target_properties=None, root=None, split_domain_budget=False,
               fixed_descriptor_k=1):
    root = project_root() if root is None else Path(root)
    jobs=[{'name':'components','task':'components'}]
    tests=[r for p in config['test_files'] for r in read_records(resolve_path(root/p,root))]
    jobs[0]['test_records']=len(tests)
    train=[r for p in config['train_files'] for r in read_records(resolve_path(root/p,root))]
    domains=config.get('dataset_domains') or {name:None for name in config.get('evaluation_domains',[config['dataset']])}
    properties=[p for p,spec in config.get('properties',{}).items() if not spec.get('context')]
    if target_properties is not None:
        properties=[p for p in properties if p in target_properties]
    for dataset,domain in domains.items():
        kinds={r['descriptor']['kind'] for r in tests if r['dataset']==dataset and r.get('descriptor')}
        if len(kinds)>1:
            raise ValueError(f'A dataset must have one declared structure kind: {dataset}')
        kind=config.get('dataset_kinds', {}).get(dataset) or next(iter(kinds), config.get('kind'))
        if kind not in ('space', 'layer'):
            raise ValueError(f'Missing space/layer metadata for dataset: {dataset}')
        context={'source_domain':domain} if domain is not None else {}
        divisor=len(domains) if split_domain_budget else 1
        selected=sorted((r for r in tests if r['dataset']==dataset),key=lambda r:hashlib.sha256(f"{seed}/{r['id']}".encode()).hexdigest())
        available_cases=len(selected)
        if cases: selected=selected[:max(1,cases//divisor)]
        base={'dataset':dataset,'kind':kind,'properties':context,'count':max(1,count//divisor),'case_ids':[str(r['id']) for r in selected],'k':k}
        def add(suffix,task,targets=None,**extra):
            job={**base,'name':dataset+'_'+suffix,'task':task,'properties':{**context,**(targets or {})},**extra}
            if task!='csp':
                job.pop('case_ids');job.pop('k')
            else:
                job.update(available_test_cases=available_cases,
                           test_case_scope='full_test_set' if len(selected)==available_cases else 'diagnostic_subset')
            jobs.append(job)
        add('dng','dng');add('csp','csp');add('fixed_descriptor','csp',fixed_descriptor=True,k=fixed_descriptor_k)
        if group_count:
            groups=list(range(1,81 if kind=='layer' else 231))
            add('group_sweep','dng',groups=groups,per_group=group_count,count=len(groups)*group_count)
        domain_train=[r for r in train if r['dataset']==dataset]
        for p in properties:
            values=[r['properties'][p] for r in domain_train if r['properties'].get(p) is not None and np.isfinite(r['properties'][p])]
            for i,value in enumerate(sorted(set(map(float,np.quantile(values,[.1,.5,.9])))) if values else []):
                target={p:value}
                add(f'dng_{p}_q{i}','dng',target,count=max(1,conditional_count//divisor),conditional=True)
                add(f'csp_{p}_q{i}','csp',target,conditional=True)
        if len(properties)>1:
            complete=sorted([r for r in domain_train if all(r['properties'].get(p) is not None and np.isfinite(r['properties'][p]) for p in properties)],
                            key=lambda r:(r['properties'][properties[0]],str(r['id'])))
            for i,q in enumerate((.1,.5,.9)) if complete else []:
                r=complete[round(q*(len(complete)-1))];target={p:r['properties'][p] for p in properties}
                add(f'dng_joint_q{i}','dng',target,count=max(1,conditional_count//divisor),conditional=True)
                add(f'csp_joint_q{i}','csp',target,conditional=True)
    selected=set(tasks or TASKS)
    conditional_tasks=set(conditional_tasks if conditional_tasks is not None else ('dng','csp'))
    def selected_job(job):
        category=('components' if job['task']=='components' else 'group' if job.get('group') or job.get('groups') else
                  'fixed_descriptor' if job.get('fixed_descriptor') else job['task'])
        return category in selected and (not job.get('conditional') or job['task'] in conditional_tasks)
    return [j for j in jobs if selected_job(j)]


def build_plan(args, root):
    catalog = load_experiments(getattr(args, 'experiments', root/'configs/experiments.yaml'), root)
    plan=[]
    for name in args.models:
        entry = catalog[name]
        if args.checkpoint_root is not None:
            directory=args.checkpoint_root/name
            checkpoint=directory/'best.ckpt' if (directory/'best.ckpt').exists() else directory/'model.best.pt'
        elif entry.get('checkpoint'):
            checkpoint=resolve_path(entry['checkpoint'], root)
        else:
            runs=RunSettings.load(root).jobs/name
            candidates=sorted([*runs.glob('*/model.best.pt'), *runs.glob('*/best.ckpt')], key=lambda p:(p.parent.name,p.name=='best.ckpt'))
            checkpoint=candidates[-1] if candidates else RunSettings.load(root).jobs/name/'model.best.pt'
        if checkpoint.exists():
            saved=load_model_checkpoint(checkpoint,map_location='cpu',weights_only=False)
            config=saved['config']
            if config.get('experiment_id')!=name: raise ValueError('Checkpoint/model identity mismatch')
            checkpoint_hash=sha256(checkpoint)
            if args.run and not args.plan:
                from unifiedgroupgen.tests.components import checked_config
                checked_config(saved)
        else:
            config=entry['config']
            checkpoint_hash=None
        data_hashes={p:sha256(resolve_path(root/p,root)) for split in ('train','val','test') for p in config[split+'_files']}
        for job in build_jobs(config,args.count,args.csp_cases,args.csp_k,args.conditional_count,args.seed,args.per_group,
                              args.tasks,args.conditional_tasks,args.target_properties,root=root,
                              split_domain_budget=getattr(args,'split_domain_budget',False),
                              fixed_descriptor_k=getattr(args,'fixed_descriptor_k',1)):
            job['name'] = getattr(args, 'task_prefix', '') + job['name']
            plan.append({'model':name,'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_hash,'data_sha256':data_hashes,**job})
    if getattr(args,'reuse_manifest',None):
        prior=json.loads(args.reuse_manifest.read_text(encoding='utf-8'))
        for job in plan:
            key=job['model']+'/'+job['name']
            source=prior['jobs'].get(key)
            if source and source['checkpoint_sha256']==job['checkpoint_sha256']:
                if job['task']=='components': job['prior_components']=source
                else:
                    spec=source['prior_samples']
                    if job.get('groups'):
                        rows=read_records(spec['samples'])
                        if len(rows)>job['count'] or any(r.get('requested_group')!=job['groups'][i//job['per_group']]
                                                         for i,r in enumerate(rows)): continue
                    job['prior_samples']=spec
                    if job['task']=='csp':
                        slots={(str(r.get('case_id',r.get('reference_id'))),int(r.get('candidate',0)))
                               for r in read_records(spec['samples'])}
                        job['planned_reused']=sum((case,k) in slots for case in job['case_ids'] for k in range(job['k']))
                    else: job['planned_reused']=min(job['count'],spec['records'])
    total=sum((len(j['case_ids'])*j['k'] if j['task']=='csp' else j.get('count',0)) for j in plan if j['task']!='components')
    options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('action','run','skip_completed','plan','status','stage','verify_generated')}
    identity={'options':options,'jobs':plan,'code':{source_key(p,root):sha256(p) for p in
        [*package_sources(),Path(__file__) ]}}
    return plan, total, identity


def run_command(command,log,env):
    with log.open('a',encoding='utf-8') as stream:
        subprocess.run([sys.executable,'-u',*map(str,command)],cwd=project_root(),env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)


def verify_files(root,outputs):
    for name,expected in outputs.items():
        path=resolve_path(root/name,root)
        if not path.is_file() or sha256(path)!=expected:
            raise ValueError(f'Saved output missing or modified: {path}')


def output_hashes(root):
    return {source_key(p,root):sha256(p) for p in root.rglob('*')
            if p.is_file() and p.suffix in ('.json','.jsonl','.csv','.png','.cif')}


def show_status(args):
    state=json.loads((args.output/'pipeline.json').read_text(encoding='utf-8'))
    result={'state':state['state'],'pid':state['pid'],'current':state.get('current'),
        'stage':state.get('stage'),'phase':state.get('phase'),
        'complete':sum(r['state']=='complete' for r in state['jobs']),'total':len(state['identity']['jobs']),
        'generated_jobs':sum(r.get('generation',{}).get('state')=='complete' for r in state['jobs']),
        'generation_jobs_total':sum(j['task']!='components' for j in state['identity']['jobs']),
        'last_completed':[{'name':r['name'],'seconds':r.get('seconds')} for r in state['jobs'] if r['state']=='complete'][-5:],
        'error':state.get('error')}
    current=next((r for r in state['jobs'] if r['name']==state.get('current')), {})
    status=Path(current.get('directory', args.output))/'sampling.status.json'
    if status.exists(): result['sampling']=json.loads(status.read_text(encoding='utf-8'))
    print(json.dumps(result,ensure_ascii=False,indent=2));return


def verify_generated(args, plan, identity):
    saved=json.loads((args.output/'pipeline.json').read_text(encoding='utf-8'))
    if saved['identity']!=identity: raise ValueError('Generation verification requires identical model/data/options/code')
    generated=failed=cifs=inline_cifs=0
    for job in plan:
        if job['task']=='components': continue
        key=job['model']+'/'+job['name'];target=job_directory(job)
        entry=next((r for r in saved['jobs'] if r['name']==key),{})
        receipt=entry.get('generation',{})
        if receipt.get('state')!='complete': raise ValueError(f'Generation incomplete: {key}')
        verify_files(target,receipt['outputs'])
        records=read_records(target/'samples.jsonl')
        generated+=sum(r['status']=='generated' for r in records)
        failed+=sum(r['status']!='generated' for r in records)
        cifs+=sum(p.endswith('.cif') for p in receipt['outputs'])
        for record in records:
            if 'cif' in record:
                inline_cif_bytes(record);inline_cifs+=1
    print(json.dumps({'state':'verified','requested':generated+failed,'generated':generated,'failed':failed,
        'cif_files_checked':cifs,'cif_strings_checked':inline_cifs,'regeneration_started':False},indent=2));return


def write_plan(args, plan, total, identity):
    args.output.mkdir(parents=True,exist_ok=True)
    write_json(args.output/'plan.json',{'identity':identity,'generation_attempts':total,'formal_tasks_started':False})
    with (args.output/'plan.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        fields=['model','job','task','dataset','kind','test_records','test_case_scope','available_test_cases','candidates_per_case','requested_structures','existing_candidates','guidance','properties']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for j in plan:
            writer.writerow({'model':j['model'],'job':j['name'],'task':j['task'],'dataset':j.get('dataset'),
                'kind':j.get('kind'),'test_records':len(j.get('case_ids',[])) if j['task']=='csp' else j.get('test_records'),
                'test_case_scope':j.get('test_case_scope'),'available_test_cases':j.get('available_test_cases'),
                'candidates_per_case':j.get('k') if j['task']=='csp' else None,
                'requested_structures':len(j['case_ids'])*j['k'] if j['task']=='csp' else j.get('count',0),
                'existing_candidates':j.get('planned_reused',0),
                'guidance':args.guidance if j.get('conditional') else 0.,'properties':json.dumps(j.get('properties',{}),sort_keys=True)})
    print(json.dumps({'models':len(args.models),'jobs':len(plan),'generation_attempts':total,'plan':str(args.output/'plan.json'),
                      'checkpoints_missing':sorted({j['model'] for j in plan if j['checkpoint_sha256'] is None}),
                      'formal_tasks_started':False},ensure_ascii=False,indent=2));return


def evaluate_suite(args):
    if args.status:
        show_status(args); return
    if min(args.count,args.csp_k,args.conditional_count,args.steps,args.threads,args.test_batch_size)<1 or min(args.per_group,args.csp_cases)<0:
        raise ValueError('Budgets must be positive; per-group and csp-cases may be zero')
    if bool(args.cgdit_root)!=bool(args.predictor_root): raise ValueError('Provide both predictor paths')
    plan, total, identity = build_plan(args, ROOT)
    if args.verify_generated:
        verify_generated(args, plan, identity); return
    if not args.run or args.plan:
        write_plan(args, plan, total, identity); return
    if args.stage=='all':
        args.stage='generate'; evaluate_suite(args)
        args.stage='metrics'; return evaluate_suite(args)
    if any(j['checkpoint_sha256'] is None for j in plan): raise ValueError('Missing checkpoints; only a plan was possible')
    if args.device.startswith('cuda') and args.stage=='generate':
        # Refuse competing GPU work; never stop or alter another task.
        check=subprocess.run(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True)
        active=check.stdout.strip().splitlines()
        if sys.platform=='win32': active=[r for r in active if any(n in r.lower() for n in ('python','julia','matlab','blender','cuda'))]
        if active: raise RuntimeError('GPU has active compute processes; leave them running and use CPU or wait')
    args.output.mkdir(parents=True,exist_ok=True)
    pipeline=args.output/'pipeline.json'
    state={'state':'running','pid':os.getpid(),'identity':identity,'jobs':[],'generation_attempts':total,'stage':args.stage,
           'started_at':time.time()}
    if pipeline.exists():
        old=json.loads(pipeline.read_text(encoding='utf-8'))
        if not args.skip_completed or old['identity']!=identity:
            raise ValueError('Existing campaign requires unchanged inputs/options and skip_completed=true, or a new output directory')
        state['jobs']=old['jobs']
        state['started_at']=old.get('started_at',state['started_at'])
    if args.stage=='metrics':
        for job in plan:
            if job['task']=='components': continue
            key=job['model']+'/'+job['name']
            entry=next((r for r in state['jobs'] if r['name']==key),{})
            if entry.get('generation',{}).get('state')!='complete':
                raise ValueError(f'Generate all requested structures first: {key}; metrics stage never samples')
            verify_files(job_directory(job),entry['generation']['outputs'])
    lock=args.output/'evaluation.lock'
    if lock.exists():
        import psutil
        owner=int(lock.read_text())
        if psutil.pid_exists(owner): raise RuntimeError(f'Campaign lock still has a live owner PID {owner}; no process was stopped')
        lock.unlink()
    with lock.open('x') as stream: stream.write(str(os.getpid()))
    env={**os.environ,'PYTHONPATH':runtime_pythonpath(),'MKL_THREADING_LAYER':'SEQUENTIAL',
         'OMP_NUM_THREADS':str(args.threads),'MKL_NUM_THREADS':str(args.threads),'PYTHONIOENCODING':'utf-8'}
    try:
        for job in plan:
            key=job['model']+'/'+job['name'];target=job_directory(job)
            old=next((r for r in state['jobs'] if r['name']==key),None)
            if old and old['state']=='complete':
                verify_files(target,old['outputs'])
                continue
            target.mkdir(parents=True,exist_ok=True)
            entry=old or {'name':key}
            entry['directory']=str(target)
            if old is None: state['jobs'].append(entry)
            entry.update(state='running',started=time.time());state.update(current=key,updated=time.time());write_json(pipeline,state)
            write_json(target/'job.json',job)
            print('[START] '+key,flush=True)
            run_job(job, args, target, entry, state, pipeline, env)
        if args.stage=='generate':
            state['state']='complete' if len(state['jobs'])==len(plan) and all(j['state']=='complete' for j in state['jobs']) else 'generation_complete'
        else:
            state['state']='complete'
            campaign_summary(args.output,state)
            state.update(current=None,phase='complete')
    except BaseException as error:
        state.update(state='failed',error=str(error));raise
    finally:
        state['updated']=time.time();write_json(pipeline,state);lock.unlink()


def job_directory(job):
    return model_result_directory(job['checkpoint'], 'evaluate' if job['task']=='components' else 'generate', job['name'])


def run_worker(options, task, values, target, log, env):
    """Each worker invokes the entry responsible for its own operation."""
    config_path = target/(log.stem+'.yaml')
    write = {'defaults':['_self_',{'override hydra/job_logging':'disabled'}],
             'hydra':{'job':{'chdir':False},'run':{'dir':str(target)},'output_subdir':None},
             'task':task, 'device':options.device, 'seed':options.seed, 'threads':options.threads, **values}
    OmegaConf.save(OmegaConf.create(write), config_path)
    run_command(['-m', {'generate':'unifiedgroupgen.generate','check_components':'unifiedgroupgen.tests.components',
                             'evaluate':'unifiedgroupgen.evaluate','predict_properties':'unifiedgroupgen.evaluate'}[task],
                     '--config-path',target,'--config-name',config_path.stem], log, env)


def evaluate_metrics(job, args, target, state, pipeline, env):
    config = load_model_checkpoint(job['checkpoint'], map_location='cpu', weights_only=False)['config']
    generated = target
    target = model_result_directory(job['checkpoint'], 'evaluate', job['name'])
    target.mkdir(parents=True, exist_ok=True)
    property_args = {}
    if args.conditional_enabled and args.predictor_root and (job['dataset']=='mp_20' or args.allow_proxy_transfer):
        state['phase']='properties'; write_json(pipeline,state)
        spec = OmegaConf.to_container(OmegaConf.load(ROOT/'configs/property_predictor/default.yaml'), resolve=True)
        spec.update(backend='cgdit', samples=str(generated/'samples.jsonl'), dataset=job['dataset'], kind=job['kind'],
                    cgdit_root=str(args.cgdit_root), predictor_root=str(args.predictor_root),
                    predictor_seed=args.predictor_seed, output=str(target/'proxy'),
                    reference_files=config['test_files'], reference_cache=str(args.output/'proxy_reference_cache'),
                    allow_proxy_transfer=args.allow_proxy_transfer)
        if not (target/'proxy/evaluator.json').exists():
            run_worker(args,'predict_properties',{'property_predictor':spec},target,target/'properties.log',env)
        property_args = {'assessments':str(target/'proxy/predictions.csv'),
                         'assessment_metadata':str(target/'proxy/evaluator.json'),
                         'reference_assessments':str(target/'proxy/reference_predictions.csv')}
    request = {'properties':{p:v for p,v in job['properties'].items() if p!='source_domain'}, 'group':job.get('group')}
    if job.get('fixed_descriptor'): request.update(sampling_task='fixed_descriptor',information_level='true_group_orbits_and_elements')
    elif job['task']=='csp': request.update(sampling_task='composition_csp',information_level='integer_composition')
    if job.get('groups'): request['sampling_task']='balanced_group_sweep'
    if job['task']=='csp': request.update(case_ids=job['case_ids'], attempts_per_case=job['k'],
        test_case_scope=job['test_case_scope'],available_test_cases=job['available_test_cases'])
    write_json(target/'request.json',request)
    spec = OmegaConf.to_container(OmegaConf.load(ROOT/'configs/evaluation/default.yaml'),resolve=True)
    spec.update(samples=str(generated/'samples.jsonl'), task=job['task'], kind=job['kind'], dataset=job['dataset'],
                reference_files=config['test_files'], novelty_files=config['train_files'],
                request=str(target/'request.json'), output=str(target),
                cache=str(getattr(args,'reference_cache',args.output/'reference_cache')),
                expected_attempts=len(job['case_ids'])*job['k'] if job['task']=='csp' else job['count'],
                no_fingerprints=args.no_fingerprints, **property_args)
    spec['conditional']['enabled']=args.conditional_enabled
    state['phase']='metrics'; write_json(pipeline,state)
    manifest=target/'manifest.json'
    if manifest.exists():
        from unifiedgroupgen.evaluate import metric_versions
        receipt=json.loads(manifest.read_text(encoding='utf-8'))
        expected=receipt['identity']
        if expected.get('versions')!=metric_versions(): raise ValueError('Metric environment changed; choose a new campaign output')
        for name,digest in {**expected['inputs'],**{str(resolve_path(p)):h for p,h in expected['code'].items()}}.items():
            if sha256(name)!=digest: raise ValueError('Completed metric input or implementation changed: '+name)
        verify_files(manifest.parent,receipt['outputs'])
        print('[EXISTING METRICS VERIFIED] '+job['model']+'/'+job['name'],flush=True)
        return
    predictor=OmegaConf.to_container(OmegaConf.load(ROOT/'configs/property_predictor/default.yaml'),resolve=True)
    predictor.update(enabled=args.conditional_enabled and getattr(args,'predictor_backend',None)=='matgl',
                     backend=getattr(args,'predictor_backend',None) or 'matgl',
                     allow_proxy_transfer=args.allow_proxy_transfer,batch_size=8,
                     reference_cache=str(getattr(args,'property_reference_cache',args.output/'proxy_reference_cache')))
    run_worker(args,'evaluate',{'evaluation':spec,'property_predictor':predictor},target,target/'metrics.log',env)


def run_job(job, args, target, entry, state, pipeline, env):
    spec = {'checkpoint':job['checkpoint'], 'job':job, 'output':str(target),
            'steps':args.steps, 'guidance':args.guidance if job.get('conditional') else 0.,
            'batch_size':getattr(args,'generation_batch_size',args.test_batch_size),
            'precision':getattr(args,'generation_precision','fp32'),
            'solver':getattr(args,'generation_solver','heun')}
    if job['task']=='components':
        spec['batch_size'] = args.test_batch_size
        state['phase']='components'; write_json(pipeline,state)
        if job.get('prior_components'):
            import shutil
            previous=job['prior_components']
            for name,item in previous['files'].items():
                if sha256(item['path'])!=item['sha256']: raise ValueError('Previous component report changed')
                shutil.copyfile(item['path'],target/name)
            write_json(target/'reused_component_report.json',previous)
            entry['reused_component_report']=True
        else:
            run_worker(args,'check_components',{'component_check':spec},target,target/'components.log',env)
        seconds=time.time()-entry['started']
        entry.update(state='complete',evaluation_seconds=seconds,seconds=seconds,outputs=output_hashes(target))
        state['updated']=time.time(); write_json(pipeline,state)
        print('[COMPONENT CHECK DONE] '+entry['name'],flush=True)
        return
    elif args.stage in ('all','generate'):
        receipt=entry.get('generation',{})
        if receipt.get('state')=='complete':
            verify_files(target,receipt['outputs'])
        else:
            state['phase']='generate'; write_json(pipeline,state)
            started=time.perf_counter()
            run_worker(args,'generate',{'generation':spec},target,target/'inference.log',env)
            entry['generation']={'state':'complete','seconds':time.perf_counter()-started,'outputs':output_hashes(target)}
        entry['state']='generated'; state['updated']=time.time(); write_json(pipeline,state)
    if args.stage=='generate':
        print('[GENERATED] '+entry['name'],flush=True); return
    started=time.perf_counter()
    if job['task']!='components': evaluate_metrics(job,args,target,state,pipeline,env)
    seconds=time.time()-entry['started'] if job['task']=='components' else time.perf_counter()-started
    metrics=model_result_directory(job['checkpoint'], 'evaluate', job['name'])
    outputs={**output_hashes(target), **{str(metrics/name):value for name,value in output_hashes(metrics).items()}}
    entry.update(state='complete',evaluation_seconds=seconds,metric_directory=str(metrics),
                 seconds=entry.get('generation',{}).get('seconds',0)+seconds, outputs=outputs)
    state['updated']=time.time(); write_json(pipeline,state)
    print('[DONE] '+entry['name'],flush=True)


def execute(cfg):
    args=task_options(cfg,'evaluation_suite')
    if args.device=='auto': args.device='cuda' if torch.cuda.is_available() else 'cpu'
    args.run=args.action=='run'; args.plan=args.action=='plan'
    args.status=args.action=='status'; args.verify_generated=args.action=='verify'
    if args.action not in ('plan','run','status','verify'): raise ValueError('Choose action=plan/run/status/verify')
    args.models=args.models or list(load_experiments(args.experiments,ROOT))
    if set(args.models)-set(load_experiments(args.experiments,ROOT)): raise ValueError('Unknown model in experiment catalog')
    return evaluate_suite(args)


@hydra.main(version_base='1.3',config_path=str(ROOT/'configs'),config_name='evaluation_campaign')
def main(cfg):
    execute(cfg)


if __name__=='__main__': main()
