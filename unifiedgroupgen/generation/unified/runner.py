"""Chunked generation ledger; every request and every failed attempt is retained."""
import json
import time
from contextlib import nullcontext
from pathlib import Path
import torch
from unifiedgroupgen.common.configuration import sampling_files
from unifiedgroupgen.common.structures import write_json
from unifiedgroupgen.common.progress import Progress
from unifiedgroupgen.group_constraints.group_prior import sample_group,NoFeasibleGroup
from unifiedgroupgen.group_constraints.operations import compile_operations
from unifiedgroupgen.joint import evaluation_record
from .sampling import sample_batch

def generate(args,checkpoint,model,provided_spec,chemistry,properties,output):
    config=checkpoint['config'];outcomes={}
    if args.count<1 or args.batch_size<1:raise ValueError('count/batch-size must be positive')
    output=Path(output)
    summary,status=sampling_files(output)
    progress_log=summary.with_name(summary.name.replace('summary.json','sampling.progress.jsonl'))
    for path in (output,summary,status,progress_log):
        if path.exists() or path.is_symlink():raise FileExistsError(str(path))
    recorder=None
    directory=getattr(args,'trajectory_dir',None)
    if directory:
        from unifiedgroupgen.common.configuration import artifact_path
        from unifiedgroupgen.generation.trajectory import TrajectoryWriter
        recorder=TrajectoryWriter(artifact_path(directory),args.count,
                    getattr(args,'trajectory_indices',None),
                    metadata=dict(checkpoint=str(getattr(args,'checkpoint','')),
                        config=config,sampling_backend='joint_flow_v2/'+model.backend,
                        solver='packed_event_heun',steps=args.steps,epsilon=1e-4,
                        event_budget=config.get('event_budget',96),birth_scale=config.get('birth_scale',.2)))
    started=time.time()
    result=dict(schema='ugg_joint_flow_sampling_v1',samples=str(output),summary=str(summary),
        status_file=str(status),progress_log=str(progress_log),attempts=args.count,requested=args.count,batch_size=args.batch_size,
        checkpoint=str(getattr(args,'checkpoint','')),seed=args.seed,steps=args.steps,
        generation_task=config.get('generation_task','dng'),dataset=config.get('dataset',''),
        sampling_backend='joint_flow_v2/'+model.backend,solver='packed_event_heun',
        request=dict(kind=args.kind,group=args.group,properties=properties,
                     chemical_control=chemistry.to_dict(),exact_group=args.exact_group),
        started=started,outcomes=outcomes)
    def report(state,final=False,**extra):
        result.update(state=state,attempted=sum(outcomes.values()),
            generated=outcomes.get('generated',0),rejected=outcomes.get('rejected',0),
            failed=outcomes.get('failed',0),updated=time.time(),seconds=time.time()-started,**extra)
        result['accepted']=result['generated']+outcomes.get('generated_state',0)
        result['unattempted']=args.count-result['attempted']
        if recorder is not None:
            result.update(trajectory_dir=str(recorder.directory),trajectories=recorder.completed)
        if final:write_json(summary,result)
        write_json(status,result)
    output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('x',encoding='utf-8') as stream:
        # Exclusive claims preserve existing sidecars as well as the ledger.
        for path in (summary,status):
            with path.open('x',encoding='utf-8') as sidecar:
                json.dump(dict(result,state='running',attempted=0),sidecar,allow_nan=False)
        report('running',final=True)
        try:
            with Progress('Generate',args.count,log=progress_log,unit='structure') as display, (recorder if recorder is not None else nullcontext()):
                _generate_batches(args,config,model,provided_spec,chemistry,properties,recorder,stream,outcomes,report,display)
                if sum(outcomes.values())!=args.count:
                    raise RuntimeError('Sampler did not return every requested attempt')
        except BaseException as error:
            stream.flush()
            report('interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed',
                   final=True,error_type=type(error).__name__,error=str(error))
            raise
        report('complete',final=True)
    return result


def _generate_batches(args,config,model,provided_spec,chemistry,properties,recorder,stream,outcomes,report,display=None):
    for start in range(0,args.count,args.batch_size):
        requests=[];rows={}
        for i in range(start,min(args.count,start+args.batch_size)):
            generator=torch.Generator(device=args.device).manual_seed(args.seed+i)
            try:
                spec=provided_spec or sample_group(config['group_prior'],generator,kind=args.kind,chemistry=chemistry,
                    budget=config.get('event_budget',96),max_atoms=model.max_atoms,max_orbits=model.max_orbits)
                requested=args.group if spec.number==0 else spec.number
                if config['model'].get('operation_compiler'):
                    spec=compile_operations(spec.operations,spec.kind)
                requests.append((i,spec,requested,generator))
                if recorder is not None:
                    recorder.start(i,args.seed+i,spec,requested,chemistry,properties)
            except NoFeasibleGroup as error:
                rows[i]=dict(index=i,seed=args.seed+i,status='failed',accepted=False,reason=str(error))
                if recorder is not None:
                    recorder.start(i,args.seed+i,None,args.group,chemistry,properties)
        if requests:
            callback=None
            if recorder is not None or display is not None:
                def callback(local_index,stage,state,clock,step,event,reason):
                    index=requests[local_index][0]
                    if recorder is not None:
                        recorder.record(index,stage,state,clock,step,event,reason)
                    if display is not None and stage in ('continuous','event'):
                        display.observe(event='sampling_step',sample_index=index,stage=stage,
                            sampling_time=clock,integration_step=step,t=f'{clock:.0%}',atoms=state.num_atoms)
            samples=sample_batch(model,[r[1] for r in requests],steps=args.steps,
                budget=config.get('event_budget',96),birth_scale=config.get('birth_scale',.2),
                generators=[r[3] for r in requests],properties=[properties]*len(requests),
                chemistries=[chemistry]*len(requests),trajectory_callback=callback)
            for request,sample in zip(requests,samples):
                i,_,requested,_=request
                row=evaluation_record(sample,i,args.seed+i,args.exact_group,properties,requested)
                row['sampling_backend']='joint_flow_v2/'+model.backend
                row['solver']='packed_event_heun';rows[i]=row
        for i in sorted(rows):
            row=rows[i];row.update(generation_task=config.get('generation_task','dng'),
                dataset=config.get('dataset',''),case_id=args.case_id,candidate=i)
            if recorder is not None:recorder.finish(i,row)
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            outcomes[row['status']]=outcomes.get(row['status'],0)+1
            if display is not None:
                display.update(accepted=outcomes.get('generated',0)+outcomes.get('generated_state',0),
                               failed=outcomes.get('failed',0),rejected=outcomes.get('rejected',0))
        stream.flush()
        report('running')
