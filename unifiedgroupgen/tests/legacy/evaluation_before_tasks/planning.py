"""Build explicit evaluation jobs and identities without starting subprocesses."""
from unifiedgroupgen.pl_modules.factory import load_model_checkpoint

from pathlib import Path
import hashlib
import torch
import numpy as np
from unifiedgroupgen.common.configuration import load_experiments
from unifiedgroupgen.common.configuration import RunSettings
from unifiedgroupgen.common.configuration import project_root, resolve_path, source_key
from unifiedgroupgen.common.source import package_sources
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.io import read_records, sha256

TASKS = ('components', 'dng', 'csp', 'fixed_descriptor', 'group')


def build_jobs(config, count, cases, k, conditional_count, seed, group_count=0,
               tasks=None, conditional_tasks=None, target_properties=None, root=None):
    root = project_root() if root is None else Path(root)
    jobs=[{'name':'components','task':'components'}]
    tests=[r for p in config['test_files'] for r in read_records(resolve_path(root/p,root))]
    jobs[0]['test_records']=len(tests)
    train=[r for p in config['train_files'] for r in read_records(resolve_path(root/p,root))]
    domains=config.get('dataset_domains') or {config['dataset']:None}
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
                from unifiedgroupgen.tests.legacy.evaluation_before_tasks.inference import checked_config
                checked_config(saved)
        else:
            config=entry['config']
            checkpoint_hash=None
        data_hashes={p:sha256(resolve_path(root/p,root)) for split in ('train','val','test') for p in config[split+'_files']}
        for job in build_jobs(config,args.count,args.csp_cases,args.csp_k,args.conditional_count,args.seed,args.per_group,
                              args.tasks,args.conditional_tasks,args.target_properties, root=root):
            plan.append({'model':name,'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_hash,'data_sha256':data_hashes,**job})
    total=sum((len(j['case_ids'])*j['k'] if j['task']=='csp' else j.get('count',0)) for j in plan if j['task']!='components')
    options={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k not in ('run','resume','plan','status','stage','verify_generated')}
    identity={'options':options,'jobs':plan,'code':{source_key(p,root):sha256(p) for p in
        package_sources()}}
    return plan, total, identity
