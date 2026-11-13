"""Final CUDA equivariance, native resume and actual batched CLI contracts."""
import json,sys,subprocess,hashlib
from pathlib import Path
from dataclasses import replace
import torch
from unifiedgroupgen.common.configuration import project_root
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_modules.joint_training import load_joint_checkpoint,fit_joint,FLOW_FORMAT
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.joint_flow.state import initial,apply,assign,JointAction
from unifiedgroupgen.pl_data.joint_path import target_from_record
from unifiedgroupgen.group_constraints.chemistry import orbit_counts

def main():
    torch.set_num_threads(1);root=project_root();out=root/'intermediate/checks/joint_flow_v2_20261113'
    report=json.loads((out/'final/short_report.json').read_text(encoding='utf-8'))
    entry=next(r for r in report['runs'] if r['dataset']=='mp_20' and r['task']=='dng' and r['backbone']=='equiformer_v2')
    ck=load_joint_checkpoint(entry['checkpoint']);assert ck['format']==FLOW_FORMAT
    model=build_joint_model(ck['config'],'cuda:0').eval();model.load_state_dict(ck['model'],strict=True)
    torch.manual_seed(51);q,_=torch.linalg.qr(torch.randn(3,3,device='cuda:0'))
    if torch.det(q)<0:q[:,0]*=-1
    states=[]
    for kind,group in [('space',1),('layer',78),('plane',17)]:
        s=initial(catalog_spec(kind,group),generator=torch.Generator().manual_seed(12),budget=32)
        w=next(w for w in s.spec.templates if w.dof)
        for _ in range(3 if kind=='space' else 1):
            s=apply(s,JointAction('birth',letter=w.letter),torch.Generator().manual_seed(13+len(s.letters)))
        states.append(s.to('cuda:0'))
    with torch.no_grad():
        a=model.forward_batch(states,[.4]*3)
        b=model.forward_batch([replace(s,frame=q,origin=q.new_tensor([2.,-3.,5.])) for s in states],[.4]*3)
    equiv=[]
    for s,u,v in zip(states,a,b):
        errors={}
        for name in ['metric_velocity_scaled','coordinate_velocity_scaled','element_logits','remaining','action_logits']:
            finite=torch.isfinite(u[name])&torch.isfinite(v[name])
            errors[name]=float((u[name][finite]-v[name][finite]).abs().max()) if finite.any() else 0.
            torch.testing.assert_close(u[name],v[name],atol=5e-5,rtol=5e-4)
        equiv.append(dict(kind=s.spec.kind,group=s.spec.number,atoms=s.num_atoms,max_abs_errors=errors))
    def selected(split,ids,dataset='mp_20'):
        rows={}
        with (root/'data/dataset/full_verified_v1'/dataset/(split+'.jsonl')).open(encoding='utf-8') as f:
            for line in f:
                r=json.loads(line)
                if r['id'] in ids:rows[r['id']]=r
        return [rows[i] for i in ids]
    train=selected('train',entry['train_ids']);val=selected('val',entry['val_ids'])
    resumed=out/'resume_equiformer/model.pt'
    trainer=fit_joint(ck['config'],train,val,resumed,'cuda:0',epochs=5,max_steps=5,
                      runtime=dict(batch_size=4,edge_budget=200000),split_hashes=entry['split_sha256'],
                      resume=Path(entry['checkpoint']).parent/'last.ckpt')
    assert trainer.global_step==5
    cli=[]
    for task in ['dng','csp']:
        item=next(r for r in report['runs'] if r['dataset']=='mp_20' and r['task']==task and r['backbone']=='equiformer_v2')
        output=out/('cli_'+task+'.jsonl')
        cmd=[sys.executable,'-m','unifiedgroupgen.joint','sample','--checkpoint',item['checkpoint'],
             '--kind','space','--count','2','--batch-size','2','--steps','8','--seed','73','--device','cuda:0','--output',str(output)]
        if task=='csp':
            reference=target_from_record(val[0]).state
            counts=orbit_counts(reference.spec,reference.letters,reference.elements)
            cmd+=['--composition',json.dumps(counts)]
        result=subprocess.run(cmd,cwd=root,text=True,encoding='utf-8',capture_output=True)
        (out/('cli_'+task+'.log')).write_text(result.stdout+result.stderr,encoding='utf-8')
        assert result.returncode==0,result.stderr
        rows=[json.loads(x) for x in output.read_text(encoding='utf-8').splitlines()]
        assert len(rows)==2
        for row in rows:
            assert row['sampling_backend']=='joint_flow_v2/equiformer_v2'
            if row['generation_status']=='done':
                assert row['constraint_residuals']['elements']
                if task=='csp':assert {int(z):n for z,n in row['current_counts'].items()}==counts
        cli.append(dict(task=task,attempts=len(rows),statuses=[r['status'] for r in rows],
                        all_started_empty=all(r['events'] and r['events'][0]['atoms_before']==0 for r in rows)))
    protected=json.loads((out/'protected_before.json').read_text())
    hashes={p:hashlib.sha256((root/p).read_bytes()).hexdigest()==h for p,h in protected.items()}
    assert all(hashes.values())
    result=dict(cuda_equivariance=equiv,native_equiformer_resume_step=trainer.global_step,
                cli=cli,legacy_sources_unchanged=hashes)
    (out/'runtime_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
if __name__=='__main__':main()
