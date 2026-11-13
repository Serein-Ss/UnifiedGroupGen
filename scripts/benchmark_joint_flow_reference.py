"""Reference-config timing and maximum-expanded-size forward/backward audit."""
import csv,gzip,json,time,statistics,gc
from pathlib import Path
import torch
from unifiedgroupgen.common.configuration import project_root,compose_experiment,training_definition
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_data.joint_path import target_from_record
from unifiedgroupgen.pl_data.task_path import task_target
from unifiedgroupgen.joint_flow.path import sample_path
from unifiedgroupgen.joint_flow.objective import loss

def main():
    root=project_root();out=root/'intermediate/checks/joint_flow_v2_20261113'
    short=json.loads((out/'final/short_report.json').read_text(encoding='utf-8'))
    with gzip.open(root/'web/symmetry-lab/dist/tables/dataset_record_statistics.csv.gz','rt',encoding='utf-8-sig') as f:stats=list(csv.DictReader(f))
    torch.set_num_threads(1);device='cuda:0';report=dict(device=torch.cuda.get_device_name(0),precision='fp32',scope='Reference architecture compute only; no full training; max-size test is a single real-data backward',rows=[])
    def sync():torch.cuda.synchronize()
    for dataset,tag,label in [('mp_20','mp20','MP20'),('c2db_51','c2db51','C2DB')]:
        saved=next(r for r in short['runs'] if r['dataset']==dataset and r['task']=='dng' and r['backbone']=='equiformer_v2')
        checkpoint=torch.load(saved['checkpoint'],weights_only=False,map_location='cpu')
        cfg=training_definition(compose_experiment(f'flow_{tag}_dng'))
        cfg.update(allowed_elements=checkpoint['config']['allowed_elements'],chemical_set_probability=0.)
        biggest=max((r for r in stats if r['dataset']==label and r['split']=='train'),key=lambda r:int(r['expanded_atoms']))
        wanted=set(saved['train_ids'])|{biggest['id']};records={}
        with (root/'data/dataset/full_verified_v1'/dataset/'train.jsonl').open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r['id'] in wanted:records[r['id']]=r
        for backend in ['equiformer_v2','joint_atom_graph']:
            cfg['model']['backbone']=backend
            torch.manual_seed(47);m=build_joint_model(cfg,device)
            samples=[]
            for i,identity in enumerate(saved['train_ids']):
                target=task_target(target_from_record(records[identity],operation_compiler=True),cfg,31+i)
                samples.append(sample_path(target,31+i,time=.75,allowed_elements=cfg['allowed_elements'],budget=96).to(device))
            optimizer=torch.optim.AdamW(m.parameters(),lr=1e-5)
            for batch in [1,4]:
                durations=[];torch.cuda.reset_peak_memory_stats()
                for step in range(5):
                    optimizer.zero_grad(set_to_none=True);sync();start=time.perf_counter()
                    v=loss(m,samples[:batch]);v['loss'].backward();optimizer.step();sync()
                    if step>=2:durations.append(time.perf_counter()-start)
                row=dict(dataset=dataset,backend=backend,phase='reference_train',batch=batch,
                    hidden=128,layers=4,channels=128 if backend=='equiformer_v2' else None,lmax=4 if backend=='equiformer_v2' else None,
                    parameters=sum(p.numel() for p in m.parameters()),current_atoms=[s.state.num_atoms for s in samples[:batch]],
                    median_step_seconds=statistics.median(durations),samples_per_second=batch/statistics.median(durations),
                    peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20)
                report['rows'].append(row)
                (out/'reference_timing.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            target=task_target(target_from_record(records[biggest['id']],operation_compiler=True),cfg,64)
            s=sample_path(target,64,time=1.,allowed_elements=cfg['allowed_elements'],budget=96,augment=False).to(device)
            optimizer.zero_grad(set_to_none=True);torch.cuda.reset_peak_memory_stats();sync();start=time.perf_counter()
            v=loss(m,[s]);v['loss'].backward();sync()
            report['rows'].append(dict(dataset=dataset,backend=backend,phase='maximum_size_backward',
                id=biggest['id'],source_atoms=int(biggest['source_atoms']),expanded_atoms=s.state.num_atoms,
                seconds=time.perf_counter()-start,finite_loss=bool(torch.isfinite(v['loss'])),
                peak_allocated_MiB=torch.cuda.max_memory_allocated()/2**20))
            (out/'reference_timing.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            del m,optimizer,v,s,samples;gc.collect();torch.cuda.empty_cache()
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
