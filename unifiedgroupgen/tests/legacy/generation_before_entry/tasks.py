from unifiedgroupgen.common.configuration import project_root, resolve_path
import hashlib
import json
import random
import time
import torch
import numpy as np
from unifiedgroupgen.common.structures import inline_cif_bytes, read_records, record_counts, sha256, write_json, verified_prior_records
ROOT=project_root()

def generate(proposal,flow,config,job,output,device,seed,steps,guidance):
    from unifiedgroupgen.conditional.chemistry import ChemicalRules
    from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
    from unifiedgroupgen.common.structures import composition_of
    from unifiedgroupgen.generation.flow import sample_flow
    from unifiedgroupgen.group_constraints.validation import to_structure
    refs=({str(r['id']):r for p in config['test_files'] for r in read_records(resolve_path(ROOT/p,ROOT)) if r['dataset']==job['dataset']}
          if job['task']=='csp' else {})
    if job['task']=='csp':
        attempts=[(key,candidate) for key in job['case_ids'] for candidate in range(job['k'])]
    else:
        attempts=[(None,i) for i in range(job['count'])]
    ledger=output/'samples.jsonl'
    prior=verified_prior_records(job['prior_samples']) if job.get('prior_samples') else []
    by_case={str(r.get('case_id',r.get('reference_id'))):r for r in prior} if job['task']=='csp' else {}
    previous=read_records(ledger) if ledger.exists() else []
    sampling_seconds=sum(r.get('wall_seconds',0.) or 0. for r in previous)
    reused_count=sum(bool(r.get('reused_from')) for r in previous)
    generated_count=sum(r['status']=='generated' for r in previous)
    if [r['index'] for r in previous]!=list(range(len(previous))) or len(previous)>len(attempts):
        raise ValueError('Attempt ledger is incomplete/corrupt; it was preserved')
    for r in previous:
        if r['status']=='generated':
            if 'cif' in r:
                inline_cif_bytes(r)
            elif sha256(output/r['cif_path'])!=r['cif_sha256']:
                raise ValueError('Existing CIF no longer matches its receipt')
    with torch.inference_mode():
        for index,(case,candidate) in enumerate(attempts):
            if index<len(previous): continue
            started=time.perf_counter()
            # Per-attempt seeds make interrupted/resumed generation identical.
            attempt_seed=seed+index;random.seed(attempt_seed);np.random.seed(attempt_seed);torch.manual_seed(attempt_seed)
            generator=torch.Generator(device=device).manual_seed(attempt_seed)
            requested_group=(job['groups'][index//job['per_group']] if job.get('groups') else job.get('group'))
            row={'index':index,'case_id':case,'candidate':candidate,'properties_requested':job['properties'],
                 'requested_group':requested_group,'seed':attempt_seed}
            reused=(by_case.get(str(case)) if candidate==0 else None) if case is not None else (prior[index] if index<len(prior) else None)
            if reused is not None:
                reused_count+=1
                generated_count+=reused['status']=='generated'
                row={**reused,'index':index,'case_id':case,'candidate':candidate,
                     'requested_group':requested_group,'properties_requested':job['properties']}
                with ledger.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                write_json(output/'sampling.status.json',{'state':'running','attempted':index+1,'requested':len(attempts),
                    'sampling_seconds':sampling_seconds,'reused':reused_count,'new_attempts':index+1-reused_count,
                    'generated':generated_count,'failed':index+1-generated_count,'last_action':'import_verified_previous_candidate'})
                continue
            phase='descriptor_sampling'
            try:
                reference=refs[case] if case is not None else None
                counts=record_counts(reference,ROOT) if reference else None
                row['chemical_request']={'composition':counts}
                fixed=Descriptor.from_dict(reference['descriptor']) if reference and job.get('fixed_descriptor') else None
                prefix=[proposal.vocab.index['BOS'],proposal.vocab.index[f"G:{job['kind']}:{requested_group}"]] if requested_group else None
                trained_composition=config.get('composition_probability',.5)>0
                descriptor=fixed or proposal.sample(job['properties'],job['kind'],counts,guidance,1.,generator,
                    chemical=ChemicalRules(),prefix=prefix,condition_composition=trained_composition)
                compiled=compile_descriptor(descriptor)
                if counts is not None and composition_of(compiled)!=counts: raise ValueError('Hard composition violation')
                phase='flow_integration'
                generated=sample_flow(flow,compiled,job['properties'],steps,guidance,None,
                    counts if trained_composition else None,generator,
                    condition_observation=config.get('observation_probability',0)>0)
                phase='numeric_validation'
                if not torch.isfinite(generated['state']).all():
                    raise FloatingPointError('Nonfinite continuous state rejected before CIF export')
                structure=to_structure(compiled,generated)
                phase='cif_export'
                cif=structure.to(fmt='cif')
                row.update(cif=cif,cif_sha256=hashlib.sha256(cif.encode('utf-8')).hexdigest())
                # Native output is only successful after in-memory CIF round-trip verification.
                from unifiedgroupgen.common.structures import load_structure, counts_of
                phase='cif_readback'
                restored=load_structure(row,output)
                if counts_of(restored)!=counts_of(structure):
                    raise RuntimeError('CIF export/readback changed integer composition')
                if not np.isfinite(restored.lattice.matrix).all() or not np.isfinite(restored.frac_coords).all():
                    raise FloatingPointError('Nonfinite CIF readback')
                row.update(status='generated',descriptor=descriptor.to_dict(),state=generated['state'].cpu().tolist())
            except (ValueError,RuntimeError,FloatingPointError,KeyError) as error:
                if isinstance(error,torch.cuda.OutOfMemoryError): raise
                row.update(status='failed',failure_stage=phase,error_type=type(error).__name__,reason=str(error))
            row['wall_seconds']=time.perf_counter()-started
            sampling_seconds+=row['wall_seconds']
            generated_count+=row['status']=='generated'
            with ledger.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            write_json(output/'sampling.status.json',{'state':'running','attempted':index+1,'requested':len(attempts),
                'last_attempt_seconds':row['wall_seconds'],'sampling_seconds':sampling_seconds,
                'reused':reused_count,'new_attempts':index+1-reused_count,
                'generated':generated_count,'failed':index+1-generated_count})
            if (index+1)%10==0 or index+1==len(attempts):
                print(json.dumps({'phase':'sample','attempted':index+1,'requested':len(attempts),'sampling_seconds':sampling_seconds}),flush=True)
    write_json(output/'sampling.status.json',{'state':'complete','attempted':len(attempts),'requested':len(attempts),
        'reused':reused_count,'new_attempts':len(attempts)-reused_count,'sampling_seconds':sampling_seconds,
        'generated':generated_count,'failed':len(attempts)-generated_count})
