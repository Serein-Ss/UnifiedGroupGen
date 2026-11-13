"""Bounded CUDA compute benchmark; no full training or material-quality claim."""
import argparse
import gc
import json
import statistics
import time
from pathlib import Path
import torch
from unifiedgroupgen.common.configuration import project_root,compose_experiment,training_definition
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_data.joint_path import target_from_record
from unifiedgroupgen.pl_data.task_path import task_target
from unifiedgroupgen.joint_flow.path import sample_path
from unifiedgroupgen.joint_flow.objective import loss


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=5)
    p.add_argument('--warmup',type=int,default=2);p.add_argument('--records',type=int,default=4)
    p.add_argument('--reference-model',action='store_true')
    p.add_argument('--output',required=True)
    args=p.parse_args()
    if args.steps<1 or args.warmup<0 or args.records<1:p.error('Invalid bounded timing counts')
    if not args.device.startswith('cuda') or not torch.cuda.is_available():p.error('CUDA is required')
    root=project_root();out=Path(args.output)
    if not out.is_absolute():out=root/out
    if out.exists():raise FileExistsError(out)
    out.parent.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1)
    report=dict(device=torch.cuda.get_device_name(torch.device(args.device)),scope='Synchronized compute only; excludes CPU teacher construction, validation and I/O; no full training',rows=[])
    for dataset,tag in [('mp_20','mp20'),('c2db_51','c2db51')]:
        cfg=training_definition(compose_experiment('flow_'+tag+'_dng'));cfg['chemical_set_probability']=0.
        records=[]
        with (root/'data/dataset/full_verified_v1'/dataset/'train.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                row=json.loads(line);target=target_from_record(row,operation_compiler=True)
                if target.state.num_atoms<=24:records.append((row,target))
                if len(records)==args.records:break
        cfg['allowed_elements']=sorted({z for _,t in records for z in t.state.elements})
        if not args.reference_model:cfg['model'].update(hidden=32,channels=32,layers=2,lmax=2,mmax=1)
        samples=[sample_path(task_target(t,cfg,31+i),31+i,time=.8,allowed_elements=cfg['allowed_elements'],augment=False).to(args.device) for i,(_,t) in enumerate(records)]
        for name,precision,fused,checkpoint in [('fp32',torch.float32,False,False),('bf16_fused',torch.bfloat16,True,False),('bf16_fused_checkpoint',torch.bfloat16,True,True)]:
            cfg['model']['checkpoint_layers']=checkpoint;torch.manual_seed(31);m=build_joint_model(cfg,args.device)
            optimizer=torch.optim.AdamW(m.parameters(),lr=1e-5,fused=fused)
            durations=[];torch.cuda.reset_peak_memory_stats()
            for step in range(args.warmup+args.steps):
                optimizer.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter()
                with torch.autocast('cuda',dtype=precision,enabled=precision!=torch.float32):v=loss(m,samples)
                v['loss'].backward();torch.nn.utils.clip_grad_norm_(m.parameters(),cfg.get('gradient_clip',10.));optimizer.step();torch.cuda.synchronize()
                if step>=args.warmup:durations.append(time.perf_counter()-start)
            report['rows'].append(dict(dataset=dataset,mode=name,architecture=cfg['model'].copy(),effective_native_heads=8,batch=len(samples),record_ids=[row['id'] for row,_ in records],current_atoms=[s.state.num_atoms for s in samples],median_seconds=statistics.median(durations),samples_per_second=len(samples)/statistics.median(durations),peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20,loss_finite=bool(torch.isfinite(v['loss']))))
            del m,optimizer,v;gc.collect();torch.cuda.empty_cache()
    out.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
