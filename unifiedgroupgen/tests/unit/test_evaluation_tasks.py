"""New Hydra entry points and metric parity against the preserved implementation."""
import ast
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf
from pymatgen.core import Lattice, Structure
import pytest
import torch

from unifiedgroupgen import evaluate as metric_entry
from unifiedgroupgen.common.structures import read_records, sha256
from unifiedgroupgen.evaluation import common, utils
from unifiedgroupgen.evaluation.conditional import evaluate_properties
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.unified import evaluate as reference_evaluate

ROOT=Path(__file__).resolve().parents[3]


def ledger(path, records):
    path.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf-8')
    return path


def structure(kind):
    if kind=='space':
        return Structure(Lattice.cubic(5.6),['Na','Cl'],[[0,0,0],[.5,.5,.5]])
    return Structure(Lattice.hexagonal(3.16,20),['Mo','S','S'],
                     [[0,0,.5],[1/3,2/3,.578],[1/3,2/3,.422]])


def test_evaluation_contains_only_five_metric_modules_and_no_entrypoint():
    folder=ROOT/'unifiedgroupgen/evaluation'
    assert {p.name for p in folder.glob('*.py')}=={'__init__.py','common.py','csp.py','dng.py','conditional.py','utils.py'}
    assert not (folder/'predictors').exists()
    for path in folder.glob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        assert not any(isinstance(n,ast.FunctionDef) and n.name=='main' for n in ast.walk(tree))
        assert 'argparse' not in path.read_text(encoding='utf-8')


def test_metric_formula_function_bodies_are_preserved():
    folder=ROOT/'unifiedgroupgen'
    old=folder/'tests/legacy/evaluation_before_tasks'
    def bodies(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n,ast.FunctionDef)}
    expected={name:body for file in ('standard.py','statistics.py','protocols.py') for name,body in bodies(old/file).items()}
    actual=bodies(folder/'evaluation/utils.py')
    assert actual==expected
    assert bodies(folder/'evaluation/dng.py')['dng_summary']==bodies(old/'summaries.py')['dng_summary']
    assert bodies(folder/'evaluation/csp.py')['csp_summary']==bodies(old/'summaries.py')['csp_summary']


@pytest.mark.parametrize('kind',['space','layer'])
@pytest.mark.parametrize('task',['csp','dng'])
def test_metric_outputs_identical_to_preserved_protocol(tmp_path,kind,task):
    s=structure(kind)
    ref=ledger(tmp_path/'reference.jsonl',[{'id':'r','structure':s.as_dict()}])
    records=([{'index':i,'case_id':'r','candidate':i,'status':'failed','reason':'failure keeps its slot'}
              for i in range(20)] if task=='csp' else
             [{'index':i,'status':'generated','structure':s.as_dict()} for i in range(2)])
    if task=='csp': records[4].update(status='generated',structure=s.as_dict())
    else: records.append({'index':2,'status':'failed','reason':'retained'})
    samples=ledger(tmp_path/'samples.jsonl',records)
    request={'case_ids':['r'],'attempts_per_case':20} if task=='csp' else {}
    kwargs=dict(reference_files=[ref],novelty_files=[ref],request=request,use_fingerprints=False,reidentify=False)
    _,old,_=reference_evaluate(samples,task,kind,tmp_path/'old',**kwargs)
    _,new,_=metric_entry.evaluate(samples,task,kind,tmp_path/'new',**kwargs)
    for key in ('efficiency','properties'):
        old.pop(key);new.pop(key)
    assert new==old
    if task=='csp':
        for profile in utils.PROFILES:
            assert new['csp']['profiles'][profile]['1']['match_rate']==0
            assert new['csp']['profiles'][profile]['5']['match_rate']==1
            assert new['csp']['profiles'][profile]['20']['match_rate']==1


def test_conditional_metrics_opt_in_and_preserve_numbers(tmp_path):
    from unifiedgroupgen.tests.legacy.evaluation_before_tasks.properties import evaluate_properties as old_properties
    rows=[{'index':i,'basic_valid':True,'property_targets':{'band_gap':1.}} for i in range(3)]
    predictions={'0':{'band_gap':1.1},'1':{'band_gap':1.5},'2':{'prediction_error':'retained failure'}}
    disabled=copy.deepcopy(rows)
    assert evaluate_properties(disabled,predictions,{}, {'band_gap':.2})['status']=='disabled'
    assert disabled==rows
    before=old_properties(copy.deepcopy(rows),predictions,{}, {'band_gap':.2})
    after=evaluate_properties(copy.deepcopy(rows),predictions,{}, {'band_gap':.2},enabled=True)
    assert after==before
    assert after['quantities']['band_gap']['hit']['rate']==1/3


def launch(config,folder):
    folder.mkdir(exist_ok=True)
    name='worker'
    config={'defaults':['_self_',{'override hydra/job_logging':'disabled'}],
            'hydra':{'job':{'chdir':False},'run':{'dir':str(folder)},'output_subdir':None},
            'device':'cpu','threads':1,'seed':142,**config}
    OmegaConf.save(OmegaConf.create(config),folder/(name+'.yaml'))
    return subprocess.run([sys.executable,'-m',{'evaluate':'unifiedgroupgen.evaluate','evaluate_suite':'scripts.evaluate_models'}[config['task']],'--config-path',str(folder),
                           '--config-name',name],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')


def test_hydra_evaluate_resume_hashes_and_conditional_switch(tmp_path):
    samples=ledger(tmp_path/'samples.jsonl',[{'index':0,'structure':structure('space').as_dict()}])
    options=OmegaConf.to_container(OmegaConf.load(ROOT/'configs/evaluation/default.yaml'))
    options.update(samples=str(samples),output=str(tmp_path/'metrics'),no_fingerprints=True,no_reidentify=True)
    conf={'task':'evaluate','evaluation':options}
    result=launch(conf,tmp_path/'launch')
    assert result.returncode==0,result.stderr
    report=json.loads((tmp_path/'metrics/summary.json').read_text(encoding='utf-8'))
    assert report['properties']['status']=='disabled'
    assert (tmp_path/'metrics/samples.csv').exists() and (tmp_path/'metrics/metrics.png').exists()
    result=launch(conf,tmp_path/'launch')
    assert result.returncode!=0 and 'Completed metrics already exist' in result.stderr
    (tmp_path/'metrics/metrics.png').write_bytes(b'modified')
    result=launch(conf,tmp_path/'launch')
    assert result.returncode!=0 and 'Completed metrics already exist' in result.stderr
    options.update(output=str(tmp_path/'not_allowed'),assessments='predictions.csv')
    result=launch(conf,tmp_path/'launch')
    assert result.returncode!=0 and 'conditional.enabled' in result.stderr


def test_new_suite_end_to_end_components_generation_metrics_and_resume(tmp_path):
    from unifiedgroupgen.pl_modules.factory import build_models
    config={'experiment_id':'fixture','dataset':'fixture','kind':'space','groups':[['space',2]],
            'properties':{},'composition_probability':0.,'observation_probability':0.,
            'allowed_elements':[6],'max_atoms':4,'max_orbits':2,'require_data_audit':False,
            'model':{'proposal_hidden':16,'proposal_layers':1,'heads':2,'flow_hidden':16,'flow_layers':1,'frequencies':2}}
    record={'id':'c','dataset':'fixture','descriptor':{'kind':'space','number':2,'orbits':[{'letter':'i','atomic_number':6}]},
            'state':[0.]*6+[.2,.3,.4],'properties':{}}
    data=ledger(tmp_path/'data.jsonl',[record])
    for split in ('train','val','test'): config[split+'_files']=[str(data)]
    proposal,flow=build_models(config,'cpu')
    checkpoint=tmp_path/'model.pt'
    torch.save({'config':config,'proposal':proposal.state_dict(),'flow':flow.state_dict(),
                'epoch':0,'runtime_options':{'coordinate_length_scale':3.}},checkpoint)
    training=tmp_path/'experiment.yaml';OmegaConf.save(OmegaConf.create(config),training)
    catalog=tmp_path/'experiments.yaml'
    OmegaConf.save(OmegaConf.create({'experiments':{'fixture':{'config':str(training),'checkpoint':str(checkpoint)}}}),catalog)
    options=OmegaConf.to_container(OmegaConf.load(ROOT/'configs/evaluation_suite/default.yaml'))
    options.update(experiments=str(catalog),models=['fixture'],output=str(tmp_path/'suite'),count=1,
                   csp_cases=1,csp_k=1,conditional_count=1,steps=2,test_batch_size=1,no_fingerprints=True)
    conf={'task':'evaluate_suite','evaluation_suite':options}
    result=launch(conf,tmp_path/'launch')
    assert result.returncode==0,result.stderr
    assert not (tmp_path/'suite/pipeline.json').exists()
    options.update(action='run',stage='generate',skip_completed=True)
    result=launch(conf,tmp_path/'launch')
    assert result.returncode==0,result.stderr+'\n'+result.stdout
    ledgers={str(p):sha256(p) for p in tmp_path.rglob('generate/*/samples.jsonl')}
    assert len(ledgers)==3 and not list((tmp_path/'suite').rglob('*.cif'))
    assert not list((tmp_path/'suite').rglob('metrics/summary.json'))
    options['stage']='metrics'
    result=launch(conf,tmp_path/'launch')
    assert result.returncode==0,result.stderr+'\n'+result.stdout
    pipeline=json.loads((tmp_path/'suite/pipeline.json').read_text(encoding='utf-8'))
    assert pipeline['state']=='complete' and len(pipeline['jobs'])==4
    assert all(j['state']=='complete' for j in pipeline['jobs'])
    component=json.loads((tmp_path/'evaluate/components/summary.json').read_text(encoding='utf-8'))
    assert component['evaluation_type']=='model_component_check'
    assert (tmp_path/'suite/model_comparison.png').exists()
    assert {str(p):sha256(p) for p in tmp_path.rglob('generate/*/samples.jsonl')}==ledgers
    result=launch(conf,tmp_path/'launch')
    assert result.returncode==0,result.stderr
    assert {str(p):sha256(p) for p in tmp_path.rglob('generate/*/samples.jsonl')}==ledgers
    options['action']='verify'
    result=launch(conf,tmp_path/'launch')
    assert result.returncode==0 and 'regeneration_started' in result.stdout,result.stderr


def test_selective_cif_export_preserves_bytes_and_ledger(tmp_path):
    from unifiedgroupgen.common.structures import export_cif
    cif=structure('space').to(fmt='cif')
    path=ledger(tmp_path/'samples.jsonl',[{'index':3,'status':'generated','cif':cif}])
    before=path.read_bytes()
    output=tmp_path/'one.cif'
    result=export_cif(path,3,output)
    assert output.read_bytes()==cif.encode('utf-8') and path.read_bytes()==before
    assert result['cif_sha256']==hashlib.sha256(cif.encode('utf-8')).hexdigest()
    with pytest.raises(FileExistsError): export_cif(path,3,output)
