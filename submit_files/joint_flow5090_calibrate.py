"""Server-only FP32 batch calibration; each candidate has a fresh CUDA process."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import uuid
from types import SimpleNamespace

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from submit_files.hpc5090 import read,write


def identity(campaign,model,audit):
    return dict(revision=campaign['revision'],experiment=campaign['entries'][model]['experiment'],
                config_sha256=audit['config_sha256'],train_inputs=audit['inputs']['train'],
                policy=campaign['calibration'],precision='fp32')


def validate_profile(profile,expected):
    if profile.get('identity')!=expected or profile.get('state')!='complete':
        raise ValueError('Calibration does not match frozen config, TRAIN inputs and policy')
    chosen=profile['selected']
    if chosen.get('state')!='pass' or [chosen['batch_size'],chosen['edge_budget']] not in expected['policy']['candidates']:
        raise ValueError('Calibration has no valid selected candidate')
    if (chosen.get('precision')!='fp32' or not math.isfinite(chosen.get('records_per_second',0))
            or chosen['records_per_second']<=0 or chosen['peak_reserved_bytes']>chosen['memory_limit_bytes']):
        raise ValueError('Calibration did not meet FP32 throughput/memory criteria')
    return chosen


def select_trial(trials):
    valid=[r for r in trials if r.get('state')=='pass' and
           math.isfinite(r.get('records_per_second',0)) and r['records_per_second']>0]
    if not valid:raise RuntimeError('No batch candidate passed memory and finite-gradient checks')
    return max(valid,key=lambda r:(r['records_per_second'],-r['batch_size']))


def workloads(sizes,batch_size,edge_budget,policy):
    from unifiedgroupgen.pl_data.joint_datamodule import JointBatchSampler
    def packed(order,shuffle):
        dataset=SimpleNamespace(records=[{'_joint_target':SimpleNamespace(state=SimpleNamespace(num_atoms=sizes[i]))} for i in order])
        batches=JointBatchSampler(dataset,batch_size,edge_budget,lambda:0,shuffle)
        return [[order[i] for i in b] for b in batches]
    cost=lambda b:sum(sizes[i]*(sizes[i]-1) for i in b)
    normal=sorted(packed(list(range(len(sizes))),True),key=cost)
    measured=[normal[int((len(normal)-1)*q)] for q in (.1,.3,.5,.7,.9)]
    large=packed(sorted(range(len(sizes)),key=lambda i:-sizes[i]),False)
    result=[dict(indices=list(measured[0]),time=None,timed=False) for _ in range(policy['warmup_steps'])]
    result += [dict(indices=b,time=None,timed=True) for b in measured]
    result += [dict(indices=large[0],time=t,timed=False) for t in policy['stress_times']]
    result.append(dict(indices=max(large,key=cost),time=.95,timed=False))
    return result


def prepare_inputs(campaign,model,folder):
    from unifiedgroupgen.common.configuration import compose_experiment,training_definition
    from unifiedgroupgen.pl_data.dataset import load_training_splits
    from unifiedgroupgen.run import fit_training_config
    from unifiedgroupgen.group_constraints.group_prior import fit_group_prior
    from unifiedgroupgen.group_constraints.unified import catalog_spec
    from unifiedgroupgen.pl_data.task_path import has_property_target
    override='data.data_audit=intermediate/hpc_runs/'+campaign['name']+'/audits/'+model+'.json'
    config=training_definition(compose_experiment(campaign['entries'][model]['experiment'],overrides=[override]))
    records,_,_=load_training_splits(config,ROOT)
    config=fit_training_config(config,records)
    config={**config,'group_prior':fit_group_prior(records),'magnetic_training':False}
    if config.get('generation_task')=='property':records=[r for r in records if has_property_target(r,config['properties'])]
    if not records:raise ValueError('No eligible TRAIN records for calibration')
    sizes=[sum(catalog_spec(r['descriptor']['kind'],r['descriptor']['number']).template(w['letter']).multiplicity
               for w in r['descriptor']['orbits']) for r in records]
    candidates=[];needed=set()
    for batch,edges in campaign['calibration']['candidates']:
        cases=workloads(sizes,batch,edges,campaign['calibration'])
        needed.update(i for c in cases for i in c['indices'])
        candidates.append(dict(batch_size=batch,edge_budget=edges,cases=cases))
    selected=sorted(needed);mapping={i:j for j,i in enumerate(selected)}
    for candidate in candidates:
        for case in candidate['cases']:case['indices']=[mapping[i] for i in case['indices']]
    value=dict(config=config,records=[records[i] for i in selected],candidates=candidates,
               expected_atoms=[sizes[i] for i in selected],
               policy=campaign['calibration'],source_split='train',eligible_train_records=len(records))
    write(folder/'inputs.json',value)


def calibrate(campaign_path,model):
    campaign=read(campaign_path);base=Path(campaign_path).parent
    audit=read(base/'audits'/(model+'.json'));expected=identity(campaign,model,audit)
    target=base/'profiles'/(model+'.json')
    saved=read(target) if target.exists() else None
    if saved and saved.get('state')=='complete':validate_profile(saved,expected);return
    if saved and saved.get('identity')!=expected:raise ValueError('Incomplete calibration identity changed')
    folder=base/'profiles'/model/('attempt_'+uuid.uuid4().hex[:12]);folder.mkdir(parents=True,exist_ok=True)
    if saved:write(folder/'previous_attempt.json',saved)
    prepare_inputs(campaign,model,folder)
    result=dict(state='running',identity=expected,trials=[],trial_directory=str(folder),temporary_weights_discarded=True,
                scope='Finite TRAIN forward/backward/optimizer probes, not convergence or quality validation')
    for i,_ in enumerate(campaign['calibration']['candidates']):
        output=folder/('trial_'+str(i)+'.json')
        with (folder/('trial_'+str(i)+'.log')).open('w',encoding='utf-8') as log:
            subprocess.run([sys.executable,'-u',str(ROOT/'submit_files/joint_flow5090_calibrate.py'),
                            'trial','--input',str(folder/'inputs.json'),'--index',str(i),'--output',str(output)],
                           cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        row=read(output);result['trials'].append(row);write(target,result)
        print('CALIBRATION',json.dumps(row),flush=True)
        if row['state'] not in ('pass','oom','memory_margin_failed'):
            raise RuntimeError('Non-memory calibration failure; inspect trial log')
    result.update(state='complete',selected=select_trial(result['trials']))
    write(target,result)


def trial(input_path,index,output):
    import time
    import torch
    from unifiedgroupgen.pl_modules.factory import build_joint_model
    from unifiedgroupgen.pl_modules.joint_training import strict_fp32
    from unifiedgroupgen.pl_data.joint_path import target_from_record
    from unifiedgroupgen.pl_data.task_path import task_target
    from unifiedgroupgen.joint_flow.path import sample_path
    from unifiedgroupgen.joint_flow.objective import loss
    torch.set_num_threads(4);torch.manual_seed(42)
    strict_fp32(dict(precision='fp32',strict_fp32=True))
    value=read(input_path);config=value['config'];candidate=value['candidates'][index]
    row={k:candidate[k] for k in ('batch_size','edge_budget')}
    row.update(precision='fp32',state='running')
    total=torch.cuda.get_device_properties(0).total_memory;free,_=torch.cuda.mem_get_info()
    cap=min(total*value['policy']['memory_fraction'],free-value['policy']['reserve_gib']*2**30)
    row.update(memory_limit_bytes=cap,gpu=torch.cuda.get_device_name(0))
    try:
        needed={i for c in candidate['cases'] for i in c['indices']}
        targets={i:target_from_record(value['records'][i],operation_compiler=config['model'].get('operation_compiler',False)) for i in needed}
        if any(targets[i].state.num_atoms!=value['expected_atoms'][i] for i in needed):
            raise ValueError('Calibration size estimate differs from compiled target')
        model=build_joint_model(config,'cuda:0').train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=config['learning_rate'],fused=True)
        durations=[];seen=0;atom_counts=[];torch.cuda.reset_peak_memory_stats()
        for step,case in enumerate(candidate['cases']):
            optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter();samples=[]
            for i in case['indices']:
                record=value['records'][i]
                seed=int.from_bytes(hashlib.sha256(repr((record['dataset'],record['id'])).encode()).digest()[:4],'little')+step*100003
                endpoint=task_target(targets[i],config,seed)
                sample=sample_path(endpoint,seed,time=case['time'],allowed_elements=config['allowed_elements'],
                    max_atoms=config['max_atoms'],max_orbits=config['max_orbits'],budget=config['event_budget'],
                    augment=True,birth_scale=config['birth_scale'])
                samples.append(sample.to('cuda:0'))
            with torch.autocast('cuda',enabled=False):parts=loss(model,samples,config['jump_weight'])
            parts['loss'].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),config['gradient_clip'],error_if_nonfinite=True)
            optimizer.step();torch.cuda.synchronize();elapsed=time.perf_counter()-start
            atom_counts.append(sum(s.state.num_atoms for s in samples))
            if case['timed']:durations.append(elapsed);seen+=len(samples)
        if not all(torch.isfinite(p).all() for p in model.parameters()):raise FloatingPointError('Nonfinite calibrated weights')
        peak=torch.cuda.max_memory_reserved()
        row.update(state='pass' if peak<=cap else 'memory_margin_failed',peak_reserved_bytes=peak,
                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),records_per_second=seen/sum(durations),
                   timed_records=seen,timed_seconds=sum(durations),actual_batch_sizes=[len(c['indices']) for c in candidate['cases']],
                   intermediate_atoms_per_batch=atom_counts,optimizer_steps=len(candidate['cases']),
                   timing_scope='CPU path construction/transfer + FP32 forward/backward/clip/AdamW; warmups/stress excluded from throughput')
    except torch.cuda.OutOfMemoryError:
        row.update(state='oom')
    except Exception as error:
        row.update(state='error',error=type(error).__name__+': '+str(error));write(output,row);raise
    write(output,row)


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='mode',required=True)
    p=sub.add_parser('calibrate');p.add_argument('--campaign',required=True);p.add_argument('--model',required=True)
    p=sub.add_parser('trial');p.add_argument('--input',required=True);p.add_argument('--index',required=True,type=int);p.add_argument('--output',required=True)
    args=parser.parse_args()
    if not (os.environ.get('SLURM_JOB_ID') or os.environ.get('YH_JOB_ID')):
        raise RuntimeError('Calibration may run only inside an allocated server GPU job')
    if args.mode=='calibrate':calibrate(args.campaign,args.model)
    else:trial(args.input,args.index,args.output)


if __name__=='__main__':main()
