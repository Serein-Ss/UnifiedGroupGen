"""Boundary and retained-candidate checks for independent metric calculation."""
import ast
import hashlib
import json
from pathlib import Path

import pytest
import torch

from unifiedgroupgen.common.structures import verified_prior_records,sha256
from unifiedgroupgen.group_constraints.compiler import Descriptor,OrbitSpec,compile_descriptor
from unifiedgroupgen.group_constraints.validation import to_structure
from unifiedgroupgen.generate import generate_records as generate

ROOT=Path(__file__).resolve().parents[3]


def test_training_entry_does_not_schedule_metrics_or_generation():
    code=(ROOT/'unifiedgroupgen/run.py').read_text(encoding='utf-8')
    definitions={n.name for n in ast.parse(code).body if isinstance(n,ast.FunctionDef)}
    assert not {'evaluate','evaluate_task','build_jobs','run_job','inference_task','execute_analysis'} & definitions
    assert len(code.splitlines())<550
    assert not any(n.module and n.module.startswith('unifiedgroupgen.evaluation')
                   for n in ast.walk(ast.parse(code)) if isinstance(n,ast.ImportFrom))


def test_metric_entry_has_no_sampler_or_training_dependency():
    tree=ast.parse((ROOT/'unifiedgroupgen/evaluate.py').read_text(encoding='utf-8'))
    imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
    assert not any(n.startswith(('unifiedgroupgen.generation','unifiedgroupgen.pl_modules','unifiedgroupgen.run')) for n in imports)
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert 'sample' not in calls and 'manual_seed' not in calls


def prior(tmp_path):
    descriptor=Descriptor('space',2,(OrbitSpec('i',6),))
    compiled=compile_descriptor(descriptor)
    state=torch.tensor([0.]*6+[.2,.3,.4])
    cif=to_structure(compiled,{**compiled.expand(state),'state':state}).to(fmt='cif')
    original=tmp_path/'prior';original.mkdir()
    (original/'sample_00000.cif').write_bytes(cif.encode('utf-8'))
    rows=[{'index':0,'reference_id':'a','seed':777,'status':'generated','descriptor':descriptor.to_dict(),
           'state':state.tolist(),'cif_sha256':hashlib.sha256(cif.encode('utf-8')).hexdigest(),'wall_seconds':123.},
          {'index':1,'reference_id':'b','seed':888,'status':'failed','reason':'preserved failure','wall_seconds':12.}]
    path=original/'samples.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    spec={'samples':str(path),'samples_sha256':sha256(path),'records':2}
    return spec,descriptor,cif


def test_prior_batch_hash_and_failure_slots_are_preserved(tmp_path):
    spec,descriptor,cif=prior(tmp_path)
    records=verified_prior_records(spec)
    assert records[0]['cif']==cif and records[0]['seed']==777
    assert records[0]['source_wall_seconds']==123. and 'wall_seconds' not in records[0]
    assert records[1]['status']=='failed' and records[1]['seed']==888
    assert sha256(spec['samples'])==spec['samples_sha256']
    (Path(spec['samples']).parent/'sample_00000.cif').write_text('changed',encoding='utf-8')
    with pytest.raises(ValueError,match='Previous CIF changed'): verified_prior_records(spec)


def test_csp_reuses_first_candidate_and_generates_remaining_candidates(tmp_path,monkeypatch):
    spec,descriptor,cif=prior(tmp_path)
    records=[{'id':key,'dataset':'fixture','descriptor':descriptor.to_dict()} for key in ('a','b')]
    refs=tmp_path/'refs.jsonl'
    refs.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    calls=[]
    def sample(model,compiled,properties,steps,guidance,observation,composition,generator,**kwargs):
        calls.append(1)
        state=compiled.prior('cpu',torch.float32,generator)
        return {**compiled.expand(state),'state':state}
    monkeypatch.setattr('unifiedgroupgen.generation.flow.sample_flow',sample)
    output=tmp_path/'result';output.mkdir()
    job={'task':'csp','case_ids':['a','b'],'k':2,'count':4,'dataset':'fixture',
         'kind':'space','properties':{},'fixed_descriptor':True,'prior_samples':spec}
    generate(None,None,{'test_files':[str(refs)],'composition_probability':0.,'observation_probability':0.},
             job,output,'cpu',142,1,0.)
    rows=[json.loads(s) for s in (output/'samples.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows)==4 and len(calls)==2
    assert [r['candidate'] for r in rows]==[0,1,0,1]
    assert [r['case_id'] for r in rows]==['a','a','b','b']
    assert rows[0]['cif']==cif and rows[0]['seed']==777
    assert rows[2]['status']=='failed' and rows[2]['reason']=='preserved failure'
    status=json.loads((output/'sampling.status.json').read_text())
    assert status['reused']==2 and status['new_attempts']==2 and status['attempted']==4
    assert status['generated']==3 and status['failed']==1
    assert status['sampling_seconds']<123.
    assert not list(output.glob('*.cif'))


def test_joint_budgets_are_split_without_mixing_csp_and_fixed_description(tmp_path):
    from scripts.evaluate_models import build_jobs
    rows=[{'id':str(i),'dataset':d,'descriptor':{'kind':k},'properties':{}}
          for d,k in [('crystals','space'),('layers','layer')] for i in range(6)]
    path=tmp_path/'data.jsonl';path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    config={'dataset_domains':{'crystals':0,'layers':1},'test_files':[str(path)],'train_files':[str(path)],'properties':{}}
    jobs=build_jobs(config,8,4,20,8,142,split_domain_budget=True,fixed_descriptor_k=1)
    for job in jobs:
        if job['task']=='components': continue
        if job['task']=='dng': assert job['count']==4
        if job['task']=='csp':
            assert len(job['case_ids'])==2
            assert job['k']==(1 if job.get('fixed_descriptor') else 20)
            assert job['test_case_scope']=='diagnostic_subset' and job['available_test_cases']==6
