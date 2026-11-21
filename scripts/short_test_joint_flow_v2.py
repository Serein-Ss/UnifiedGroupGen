"""Reproducible short runs and synchronized RTX/CPU timings, never full training."""
import argparse,copy,json,time,statistics,hashlib
from pathlib import Path
import torch
import pytorch_lightning as pl
from unifiedgroupgen.common.configuration import project_root
from unifiedgroupgen.experiment_config import compose_experiment, training_definition
from unifiedgroupgen.common.data_utils import target_from_record
from unifiedgroupgen.pl_data.cache import load_tensor_cache, training_split_hashes
from unifiedgroupgen.pl_data.dataset import load_training_splits
from unifiedgroupgen.pl_data.unified.tasks import task_target
from unifiedgroupgen.pl_data.unified.path import sample_path
from unifiedgroupgen.pl_modules.unified.objectives import loss
from unifiedgroupgen.generation.unified.sampling import sample_batch
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_modules.unified.module import fit_joint,load_joint_checkpoint
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.group_constraints.group_prior import sample_group
from unifiedgroupgen.conditional.properties import fit_property_scalers

def sync(device):
    if str(device).startswith('cuda'):torch.cuda.synchronize()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def small_rows(records,n):
    result=[]
    for row in records:
        pointer=row['_tensor_cache']
        graph=load_tensor_cache(pointer['path'])['graphs'][pointer['index']]
        if graph['num_nodes']<=24 and row.get('properties',{}).get('band_gap') is not None:
            result.append(row)
        if len(result)==n:break
    return result
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',required=True)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--steps',type=int,default=4)
    parser.add_argument('--records',type=int,default=4)
    parser.add_argument('--tasks',nargs='+',choices=['dng','csp'],default=['dng','csp'])
    parser.add_argument('--conditioning',nargs='+',choices=['none','band_gap'],default=['none','band_gap'])
    parser.add_argument('--backbones',nargs='+',default=['equiformer_v2','joint_atom_graph'])
    args=parser.parse_args();torch.set_num_threads(1);out=project_root()/args.output
    out.mkdir(parents=True,exist_ok=False)
    report=dict(scope='Short integration and throughput only; no convergence/material-quality claim',
        hardware=dict(torch=torch.__version__,device=args.device,gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None),
        protocol=dict(train_records=args.records,val_records=2,optimizer_steps=args.steps,hidden=32,layers=2,
                      channels=32,lmax=2,mmax=1,precision='fp32',compile_constraints=True,
                      sampling_steps=8,sampling_epsilon=.01,training_timing='2 warmups + 5 synchronized steps; excludes CPU teacher construction'),
        runs=[],throughput=[])
    tracked=[p for folder in ('pl_modules/unified','pl_data/unified','generation/unified','group_constraints') for p in (project_root()/'unifiedgroupgen'/folder).glob('*.py')]+[Path(__file__)]
    report['source_sha256']={str(p.relative_to(project_root())):hashlib.sha256(p.read_bytes()).hexdigest() for p in tracked}
    for dataset,short in [('mp_20','mp20'),('c2db_51','c2db51')]:
        from unifiedgroupgen.group_constraints.group_prior import fit_group_prior
        definition=training_definition(compose_experiment(f'flow_{short}_dng'))
        full_train,full_val,paths=load_training_splits(definition,project_root())
        train,val=small_rows(full_train,args.records),small_rows(full_val,2)
        vocabulary={w['atomic_number'] for row in full_train for w in row['descriptor']['orbits']}
        prior=fit_group_prior(full_train)
        hashes=training_split_hashes(paths)
        for task,conditioning in ((task,option) for task in args.tasks for option in args.conditioning):
            for backend in args.backbones:
                pl.seed_everything(47,workers=True)
                cfg=training_definition(compose_experiment(f'flow_{short}_{task}',overrides=[f'conditioning={conditioning}']))
                cfg['model'].update(backbone=backend,hidden=32,layers=2,channels=32,lmax=2,mmax=1)
                cfg.update(allowed_elements=sorted(vocabulary),group_prior=prior,max_atoms=64,max_orbits=16,event_budget=32,chemical_set_probability=0.,learning_rate=1e-4)
                if cfg['properties']:
                    scalers=fit_property_scalers(train, cfg['properties'])
                    if scalers is not None:cfg['properties']=scalers
                path=out/'runs'/dataset/f'{task}_{conditioning}'/backend/'model.pt'
                sync(args.device);started=time.perf_counter()
                trainer=fit_joint(cfg,train,val,path,args.device,epochs=args.steps,max_steps=args.steps,
                                  runtime=dict(batch_size=4),split_hashes=hashes)
                sync(args.device);elapsed=time.perf_counter()-started
                checkpoint=load_joint_checkpoint(path)
                model=build_joint_model(checkpoint['config'],args.device)
                model.load_state_dict(checkpoint['model'],strict=True)
                entry=dict(dataset=dataset,task=task,conditioning=conditioning,backbone=backend,steps=trainer.global_step,
                    train_end_to_end_seconds=elapsed,train_ids=[r['id'] for r in train],val_ids=[r['id'] for r in val],
                    split_sha256=hashes,parameters=sum(p.numel() for p in model.parameters()),
                    validation_loss=float(trainer.callback_metrics['val/loss']),checkpoint=str(path))
                report['runs'].append(entry);write(out/'short_report.json',report)
                # Property adapter is exercised with real labels; exact formula with a held-out inventory.
                request=task_target(target_from_record(val[0]),checkpoint['config'],13)
                spec=request.state.spec
                prop=request.properties
                chemistry=request.state.chemistry
                g=torch.Generator(device=args.device).manual_seed(52)
                if task!='csp':
                    spec=sample_group(checkpoint['config']['group_prior'],g,kind=spec.kind,
                        max_atoms=model.max_atoms,max_orbits=model.max_orbits,budget=32)
                from unifiedgroupgen.group_constraints.operations import compile_operations
                spec=compile_operations(spec.operations,spec.kind)
                sync(args.device);started=time.perf_counter()
                samples=sample_batch(model,[spec]*2,steps=8,epsilon=.01,budget=32,
                                     seeds=[52,53],properties=[prop]*2,chemistries=[chemistry]*2)
                sync(args.device)
                entry['sampling_seconds']=time.perf_counter()-started
                entry['sampling_attempts']=len(samples);entry['completed']=sum(s.state.status=='done' for s in samples)
                entry['reasons']=[s.reason for s in samples]
                entry['initial_empty']=all(s.events and s.events[0]['atoms_before']==0 for s in samples)
                entry['generated_atoms']=[s.state.num_atoms for s in samples]
                from unifiedgroupgen.joint import sample_to_dict
                write(path.parent/'samples.json',[sample_to_dict(s) for s in samples])
                write(out/'short_report.json',report)
                if task=='dng' and conditioning=='none':
                    target=task_target(target_from_record(train[0]),checkpoint['config'],31)
                    sample=[sample_path(target,31+i,time=.65,allowed_elements=cfg['allowed_elements'],budget=32).to(args.device) for i in range(4)]
                    timing_snapshot=copy.deepcopy(model.state_dict())
                    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-5)
                    for batch in [1,4]:
                        durations=[]
                        for step in range(7):
                            optimizer.zero_grad(set_to_none=True);sync(args.device);started=time.perf_counter()
                            loss(model,sample[:batch])['loss'].backward();optimizer.step()
                            sync(args.device)
                            if step>=2:durations.append(time.perf_counter()-started)
                        median=statistics.median(durations)
                        row=dict(dataset=dataset,backbone=backend,phase='train',batch=batch,
                            median_step_seconds=median,samples_per_second=batch/median,measured_steps=5)
                        report['throughput'].append(row);write(out/'short_report.json',report)
                    # Same checkpoint parameters, group and seeds; compare batch-1 scheduling to packed batch-4.
                    model.load_state_dict(timing_snapshot)
                    snapshot=copy.deepcopy(model.state_dict());model.eval()
                    timing_spec=catalog_spec(spec.kind,1)
                    for batch in [1,4]:
                        sync(args.device);started=time.perf_counter();results=[]
                        for start in range(0,4,batch):
                            results.extend(sample_batch(model,[timing_spec]*min(batch,4-start),steps=8,epsilon=.01,budget=16,
                                seeds=list(range(71+start,71+min(start+batch,4)))))
                        sync(args.device);duration=time.perf_counter()-started
                        report['throughput'].append(dict(dataset=dataset,backbone=backend,phase='generate',batch=batch,
                            attempts=4,total_seconds=duration,attempts_per_second=4/duration,
                            completed=sum(r.state.status=='done' for r in results),
                            mean_events=statistics.mean(len(r.events) for r in results)))
                        write(out/'short_report.json',report)
                    model.load_state_dict(snapshot)
    report['source_unchanged_during_run']=all(hashlib.sha256((project_root()/p).read_bytes()).hexdigest()==h for p,h in report['source_sha256'].items())
    write(out/'short_report.json',report)
    print(json.dumps(report,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
