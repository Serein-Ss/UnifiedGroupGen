"""Chunked generation ledger; every request and every failed attempt is retained."""
import json
import time
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import torch
from unifiedgroupgen.common.configuration import sampling_files
from unifiedgroupgen.common.structures import write_json
from unifiedgroupgen.common.progress import Progress
from unifiedgroupgen.conditional.tasks import generation_mode
from unifiedgroupgen.group_constraints.group_prior import sample_group,NoFeasibleGroup
from unifiedgroupgen.group_constraints.operations import compile_operations
from unifiedgroupgen.joint import evaluation_record
from .sampling import sample_batch

def generate(args,checkpoint,model,provided_spec,chemistry,properties,output):
    config=checkpoint['config'];outcomes={}
    guidance_scale=getattr(args,'guidance_scale',1.)
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
                        solver='packed_event_heun',steps=args.steps,epsilon=1e-4,guidance_scale=guidance_scale,
                        event_budget=config.get('event_budget',96),birth_scale=config.get('birth_scale',.2)))
    manifest = getattr(args, 'case_manifest', None)
    policy = getattr(args, 'csp_group_policy', None) or config.get('csp_group_policy', 'prior')
    if provided_spec is None and policy == 'composition_ranked' and not config.get('group_prior', {}).get('composition_counts'):
        raise ValueError('Composition ranking requires fitted TRAIN composition counts')
    started=time.time()
    symmetry_input = ('operations' if getattr(args, 'symmetry', None) else
                      'number' if args.group is not None else 'prior')
    result=dict(schema='ugg_joint_flow_sampling_v1',samples=str(output),summary=str(summary),
        status_file=str(status),progress_log=str(progress_log),attempts=args.count,requested=args.count,batch_size=args.batch_size,
        checkpoint=str(getattr(args,'checkpoint','')),seed=args.seed,steps=args.steps,
        generation_task=generation_mode(config) or 'dng',dataset=config.get('dataset',''),
        orbit_ablation=config.get('orbit_ablation','joint'),
        sampling_backend='joint_flow_v2/'+model.backend,solver='packed_event_heun',
        request=dict(kind=provided_spec.kind if provided_spec is not None else args.kind,
                     group=args.group,symmetry_input=symmetry_input,group_policy=policy if provided_spec is None else 'specified',
                     properties=properties,guidance_scale=guidance_scale,
                     chemical_control=chemistry.to_dict(),exact_group=args.exact_group),
        started=started,outcomes=outcomes)
    if getattr(args, 'repair_short_contacts', False):
        result['request']['repair_short_contacts'] = True
    if manifest:
        result['request'].update(case_manifest_sha256=manifest.digest, cases=len(manifest.cases),
                                 candidates_per_case=manifest.candidates_per_case)
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
    labels={'label_group':0} if getattr(args,'symmetry',None) else {}
    symmetry_input = ('operations' if labels else 'number' if args.group is not None else 'prior')
    acceptance = {**labels, 'verify_cif_operations': True} if symmetry_input != 'prior' else labels
    manifest = getattr(args, 'case_manifest', None)
    training_dataset = config.get('dataset', '')
    policy = getattr(args, 'csp_group_policy', None) or config.get('csp_group_policy', 'prior')
    manifest_digest = manifest.digest if manifest else None
    for start in range(0,args.count,args.batch_size):
        requests, rows, identities = [], {}, {}
        for i in range(start,min(args.count,start+args.batch_size)):
            kind, control = args.kind, chemistry
            identity = dict(dataset=getattr(args, 'case_dataset', None) or training_dataset,
                            case_id=args.case_id, candidate=i, training_dataset=training_dataset, symmetry_input=symmetry_input)
            if manifest:
                case_index, candidate = divmod(i, manifest.candidates_per_case)
                case = manifest.cases[case_index]
                kind, control = case.kind, case.chemistry
                identity.update(dataset=case.dataset, case_id=case.case_id, candidate=candidate,
                                case_manifest_sha256=manifest_digest, requested_counts=dict(control.counts),
                                kind=kind)
            identity['group_policy'] = policy if provided_spec is None else 'specified'
            identities[i] = identity
            generator=torch.Generator(device=args.device).manual_seed(args.seed+i)
            try:
                if manifest and set(dict(control.counts)) - set(model.allowed_elements):
                    raise NoFeasibleGroup('CSP case contains elements outside the TRAIN vocabulary')
                spec=provided_spec or sample_group(config['group_prior'],generator,kind=kind,chemistry=control,
                    budget=config.get('event_budget',96),max_atoms=model.max_atoms,max_orbits=model.max_orbits,
                    **({'composition_rank': identity['candidate']} if policy == 'composition_ranked' else {}))
                requested=args.group if spec.number==0 else spec.number
                if config['model'].get('operation_compiler'):
                    spec=compile_operations(spec.operations,spec.kind)
                requests.append((i,spec,requested,generator,control))
                if recorder is not None:
                    recorder.start(i,args.seed+i,spec,requested,control,properties,**labels)
            except NoFeasibleGroup as error:
                rows[i]=dict(index=i,seed=args.seed+i,status='failed',accepted=False,reason=str(error))
                if recorder is not None:
                    recorder.start(i,args.seed+i,None,args.group,control,properties)
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
                chemistries=[r[4] for r in requests],trajectory_callback=callback,
                **({'guidance_scale':args.guidance_scale} if hasattr(args,'guidance_scale') else {}))
            for request,sample in zip(requests,samples):
                i,_,requested,_,_=request
                repair = None
                if (getattr(args, 'repair_short_contacts', False) and sample.state.status == 'done'
                        and sample.state.spec.kind in ('space', 'layer')):
                    from .contact_repair import repair_short_contacts
                    state, repair = repair_short_contacts(sample.state)
                    if state is not sample.state:
                        sample = replace(sample, state=state)
                        if recorder is not None:
                            recorder.record(i, 'contact_repair', state, sample.time, sample.integration_steps)
                row=evaluation_record(sample,i,args.seed+i,args.exact_group,properties,requested,**acceptance)
                if repair is not None:
                    row['contact_repair'] = repair
                row['sampling_backend']='joint_flow_v2/'+model.backend
                row['solver']='packed_event_heun';rows[i]=row
        for i in sorted(rows):
            row=rows[i];row.update(generation_task=generation_mode(config) or 'dng',
                                   orbit_ablation=config.get('orbit_ablation','joint'), **identities[i])
            if recorder is not None:recorder.finish(i,row)
            stream.write(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n')
            outcomes[row['status']]=outcomes.get(row['status'],0)+1
            if display is not None:
                display.update(accepted=outcomes.get('generated',0)+outcomes.get('generated_state',0),
                               failed=outcomes.get('failed',0),rejected=outcomes.get('rejected',0))
        stream.flush()
        report('running')
