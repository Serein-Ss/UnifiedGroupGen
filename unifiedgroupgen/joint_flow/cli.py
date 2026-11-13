"""Chunked generation ledger; every request and every failed attempt is retained."""
import json
from contextlib import nullcontext
import torch
from unifiedgroupgen.group_constraints.group_prior import sample_group,NoFeasibleGroup
from unifiedgroupgen.group_constraints.operations import compile_operations
from unifiedgroupgen.joint import evaluation_record
from .sampling import sample_batch

def generate(args,checkpoint,model,provided_spec,chemistry,properties,output):
    config=checkpoint['config'];outcomes={}
    recorder=None
    directory=getattr(args,'trajectory_dir',None)
    if directory:
        from unifiedgroupgen.common.configuration import artifact_path
        from .trajectory import TrajectoryWriter
        recorder=TrajectoryWriter(artifact_path(directory),args.count,
                    getattr(args,'trajectory_indices',None),
                    metadata=dict(checkpoint=str(getattr(args,'checkpoint','')),
                        config=config,sampling_backend='joint_flow_v2/'+model.backend,
                        solver='packed_event_heun',steps=args.steps,epsilon=1e-4,
                        event_budget=config.get('event_budget',96),birth_scale=config.get('birth_scale',.2)))
    with (recorder if recorder is not None else nullcontext()), output.open('x',encoding='utf-8') as stream:
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
                if recorder is not None:
                    def callback(local_index,*frame):
                        recorder.record(requests[local_index][0],*frame)
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
                outcomes[row['status']]=outcomes.get(row['status'],0)+1
                stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
    result=dict(samples=str(output),attempts=args.count,batch_size=args.batch_size,outcomes=outcomes)
    if recorder is not None:
        result.update(trajectory_dir=str(recorder.directory),trajectories=recorder.completed)
    return result
