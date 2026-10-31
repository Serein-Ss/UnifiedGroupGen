"""Sequential, resumable evaluation of a YAML experiment catalog. Default: PLAN ONLY."""
from unifiedgroupgen.common.configuration import runtime_pythonpath
from unifiedgroupgen.common.configuration import project_root, resolve_path, source_key
import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch  # Initialize the scientific runtime before IO/NumPy imports.

ROOT=project_root()
from unifiedgroupgen.common.configuration import load_experiments
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.io import inline_cif_bytes, read_records, sha256, write_json


from unifiedgroupgen.tests.legacy.evaluation_before_tasks.planning import TASKS, build_jobs, build_plan
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.campaign import run_command, verify_files, output_hashes, run_job


def parse_args():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', type=Path, default=ROOT/'configs/experiments.yaml')
    parser.add_argument('--models', nargs='+', help='Names from the selected YAML catalog; default: all')
    parser.add_argument('--tasks',nargs='+',choices=TASKS,default=list(TASKS))
    parser.add_argument('--conditional-tasks',nargs='*',choices=['dng','csp'],default=['dng','csp'])
    parser.add_argument('--target-properties', nargs='+')
    parser.add_argument('--stage',choices=['all','generate','metrics'],default='all')
    parser.add_argument('--verify-generated',action='store_true')
    parser.add_argument('--checkpoint-root', type=Path, help='Optional explicit name/model.best.pt layout')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/evaluations/unified_seed142')
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
    catalog = load_experiments(args.experiments, ROOT)
    args.models = args.models or list(catalog)
    if set(args.models) - set(catalog):
        parser.error('Unknown experiments: '+str(sorted(set(args.models)-set(catalog))))
    return parser, args


def show_status(args):
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


def verify_generated(args, plan, identity):
    saved=json.loads((args.output/'pipeline.json').read_text(encoding='utf-8'))
    if saved['identity']!=identity: raise ValueError('Generation verification requires identical model/data/options/code')
    generated=failed=cifs=inline_cifs=0
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
        for record in records:
            if 'cif' in record:
                inline_cif_bytes(record);inline_cifs+=1
    print(json.dumps({'state':'verified','requested':generated+failed,'generated':generated,'failed':failed,
        'cif_files_checked':cifs,'cif_strings_checked':inline_cifs,'regeneration_started':False},indent=2));return


def write_plan(args, plan, total, identity):
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


def main():
    parser, args = parse_args()
    if args.status:
        show_status(args); return
    if min(args.count,args.csp_k,args.conditional_count,args.steps,args.threads,args.test_batch_size)<1 or min(args.per_group,args.csp_cases)<0:
        parser.error('Budgets must be positive; per-group and csp-cases may be zero')
    if bool(args.cgdit_root)!=bool(args.predictor_root): parser.error('Provide both predictor paths')
    plan, total, identity = build_plan(args, ROOT)
    if args.verify_generated:
        verify_generated(args, plan, identity); return
    if not args.run or args.plan:
        write_plan(args, plan, total, identity); return
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
    env={**os.environ,'PYTHONPATH':runtime_pythonpath(),'MKL_THREADING_LAYER':'SEQUENTIAL',
         'OMP_NUM_THREADS':str(args.threads),'MKL_NUM_THREADS':str(args.threads),'PYTHONIOENCODING':'utf-8'}
    try:
        for job in plan:
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
            run_job(job, args, target, entry, state, pipeline, env)
        if args.stage=='generate':
            state['state']='complete' if len(state['jobs'])==len(plan) and all(j['state']=='complete' for j in state['jobs']) else 'generation_complete'
        else:
            state['state']='complete'
            from unifiedgroupgen.tests.legacy.evaluation_before_tasks.reporting import campaign_summary
            campaign_summary(args.output,state)
    except BaseException as error:
        state.update(state='failed',error=str(error));raise
    finally:
        state['updated']=time.time();write_json(pipeline,state);lock.unlink()


if __name__=='__main__': main()
