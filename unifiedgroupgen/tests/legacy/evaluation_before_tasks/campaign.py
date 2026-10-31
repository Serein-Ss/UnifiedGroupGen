"""Run one evaluation job and verify immutable outputs; no experiment planning."""
from unifiedgroupgen.pl_modules.factory import load_model_checkpoint

import subprocess
import sys
import time
import torch
from unifiedgroupgen.common.configuration import source_key, project_root, resolve_path
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.io import sha256, write_json


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




def evaluate_metrics(job, args, target, state, pipeline, env):
    config=load_model_checkpoint(job['checkpoint'],map_location='cpu',weights_only=False)['config']
    property_args=[]
    if args.predictor_root and (job['dataset']=='mp_20' or args.allow_proxy_transfer):
        state['phase']='properties';write_json(pipeline,state)
        command=['-m', 'unifiedgroupgen.tests.legacy.evaluation_before_tasks.property_inference','--samples',target/'samples.jsonl','--dataset',job['dataset'],
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
    run_command(['-m', 'unifiedgroupgen.tests.legacy.evaluation_before_tasks.cli','--samples',target/'samples.jsonl','--task',job['task'],'--kind',job['kind'],
        '--dataset',job['dataset'],'--reference-files',*config['test_files'],'--novelty-files',*config['train_files'],
        '--request',target/'request.json','--output',target/'metrics','--cache',args.output/'reference_cache',
        '--expected-attempts',expected,'--resume',*property_args,*(['--no-fingerprints'] if args.no_fingerprints else [])],target/'metrics.log',env)


def run_job(job, args, target, entry, state, pipeline, env):
    key = job['model']+'/'+job['name']
    scale=args.guidance if job.get('conditional') else 0.
    inference=['-m', 'unifiedgroupgen.tests.legacy.evaluation_before_tasks.inference','--checkpoint',job['checkpoint'],'--job',target/'job.json','--output',target,
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
        print('[GENERATED] '+key,flush=True);return
    evaluation_started=time.perf_counter()
    if job['task']!='components':
        evaluate_metrics(job, args, target, state, pipeline, env)
    evaluation_seconds=(time.time()-entry['started'] if job['task']=='components' else time.perf_counter()-evaluation_started)
    entry.update(state='complete',evaluation_seconds=evaluation_seconds,
        seconds=entry.get('generation',{}).get('seconds',0)+evaluation_seconds,outputs=output_hashes(target))
    state['updated']=time.time();write_json(pipeline,state);print('[DONE] '+key,flush=True)
