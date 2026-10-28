import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pytest
from pymatgen.core import Lattice, Structure

from unifiedgroupgen.evaluation_io import ReferenceBank, load_structure, read_records, sha256, write_json
from unifiedgroupgen.evaluation_properties import evaluate_properties, property_spec, assess
from unifiedgroupgen.evaluation_protocols import match_pair
from unifiedgroupgen.evaluation_statistics import rate, joint_coverage, frechet_distance
from unifiedgroupgen.unified_evaluation import evaluate, geometry, operation_check

ROOT=Path(__file__).resolve().parents[1]


def ledger(path,rows):
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8');return path


def nacl():
    return Structure(Lattice.cubic(5.6),['Na','Cl'],[[0,0,0],[.5,.5,.5]])


def mos2(height=20):
    return Structure(Lattice.hexagonal(3.16,height),['Mo','S','S'],[[0,0,.5],[1/3,2/3,.5+1.56/height],[1/3,2/3,.5-1.56/height]])


def test_rates_preserve_unknown_and_failed_denominators():
    assert rate([True,None,False])=={'attempted':3,'evaluated':2,'passed':1,'unknown':1,'rate_over_all_attempts':1/3,'rate_on_evaluated':.5}
    assert rate([None])['rate_over_all_attempts'] is None


def test_failed_group_attempts_remain_in_macro_denominator(tmp_path):
    samples=ledger(tmp_path/'failed.jsonl',[{'index':0,'status':'failed','requested_group':230,'reason':'retained'}])
    _,report,_=evaluate(samples,'dng','space',tmp_path/'out',use_fingerprints=False,reidentify=False)
    assert report['groups'][0]['group']==230 and report['groups'][0]['attempted']==1
    assert report['groups'][0]['exact']['passed']==0
    assert report['group_macro_exact_rate']==0.


def test_layer_vacuum_independence_and_normal_displacements():
    a,b=mos2(20),mos2(60)
    x,y=geometry(a,'layer'),geometry(b,'layer')
    for key in ('area_per_atom_angstrom2','layer_thickness_angstrom','areal_density_ug_cm2'):
        assert x[key]==pytest.approx(y[key])
    assert x['density_g_cm3'] is None
    assert match_pair(a,b,'layer')['match']
    b.translate_sites([1],[0,0,.15],frac_coords=True,to_unit_cell=False)
    assert not match_pair(a,b,'layer')['match']


def test_self_image_collision_and_legacy_preserved():
    s=Structure(Lattice.cubic(.3),['C'],[[0,0,0]])
    result=geometry(s,'space')
    assert not result['periodic_distance_valid']
    # Original pair-only criterion also includes volume >= .1.
    s=Structure(Lattice.orthorhombic(.3,3,3),['C'],[[0,0,0]])
    result=geometry(s,'space')
    assert result['legacy_structural_valid'] and not result['periodic_distance_valid']


def test_matching_cell_error_does_not_use_reference_assigned_lattice():
    a=nacl();b=a.copy();b.scale_lattice(a.volume*1.03)
    result=match_pair(b,a,'space','diffcsp_v1')
    assert result['match'] and result['measure_relative_error']==pytest.approx(.03)
    assert result['length_relative_error']>0


def test_joint_coverage_requires_same_neighbor():
    generated=[[[0.],[0.]]]
    refs=[[[0.],[100.]],[[100.],[0.]]]
    assert joint_coverage(generated,refs,2,2,1,1)['cov_precision']==0
    assert frechet_distance([[0.],[1.]],[[0.],[1.]])==pytest.approx(0,abs=1e-10)


def test_property_targets_independent_and_unknown():
    rows=[{'index':0,'basic_valid':True,'property_targets':{'band_gap':1.}},
          {'index':1,'basic_valid':False,'property_targets':{'band_gap':1.}},
          {'index':2,'basic_valid':None,'property_targets':{'band_gap':1.}}]
    result=evaluate_properties(rows,{'0':{'band_gap':1.1},'1':{'band_gap':1.}}, {'evaluator':'independent'},{'band_gap':.2})
    assert result['joint_hit']['passed']==2
    assert result['valid_joint_hit']['passed']==1
    assert result['valid_joint_hit']['unknown']==1
    assert result['valid_joint_hit']['rate_over_all_attempts']==pytest.approx(1/3)
    assert assess(.5,property_spec({'op':'le','value':1}))[1]
    assert assess(3,property_spec({'op':'between','bounds':[0,2]}))==(1,False)
    with pytest.raises(ValueError): property_spec(1)


def test_csp_cases_hard_counts_and_failed_candidates(tmp_path,monkeypatch):
    monkeypatch.setattr('unifiedgroupgen.unified_evaluation.composition_validity',lambda s:(True,None))
    s=nacl();reference=ledger(tmp_path/'reference.jsonl',[{'id':'a','structure':s.as_dict()},{'id':'b','structure':s.as_dict()}])
    samples=[]
    for case in ('a','b'):
        for candidate in range(5):
            index=len(samples)
            if candidate==4 and case=='b': samples.append({'index':index,'case_id':case,'status':'generated','structure':s.as_dict()})
            else: samples.append({'index':index,'case_id':case,'status':'failed','reason':'retained'})
    path=ledger(tmp_path/'samples.jsonl',samples)
    rows,report,_=evaluate(path,'csp','space',tmp_path/'out',[reference],request={'attempts_per_case':5},use_fingerprints=False,reidentify=False)
    assert report['attempted']==10 and report['common']['generation_success']['passed']==1
    assert report['csp']['profiles']['legacy_ugg_v1']['1']['match_rate']==0
    assert report['csp']['profiles']['legacy_ugg_v1']['5']['match_rate']==.5
    assert rows[-1]['composition_satisfied']
    assert rows[0]['failure_reason']=='retained'
    assert (tmp_path/'out/cases.csv').exists()


def test_dng_fingerprint_failure_does_not_change_basic_validity(tmp_path,monkeypatch):
    monkeypatch.setattr('unifiedgroupgen.unified_evaluation.composition_validity',lambda s:(True,None))
    def bad(*args): raise ValueError('fingerprint failed')
    monkeypatch.setattr('unifiedgroupgen.unified_evaluation.fingerprints',bad)
    s=nacl();samples=ledger(tmp_path/'samples.jsonl',[{'index':0,'structure':s.as_dict()},{'index':1,'structure':s.as_dict()},{'index':2,'status':'failed'}])
    rows,report,_=evaluate(samples,'dng','space',tmp_path/'out',use_fingerprints=True,reidentify=False)
    assert report['common']['basic_valid']['passed']==2
    assert report['common']['fingerprint_success']['passed']==0
    assert report['common']['legacy_joint_valid']['passed']==0
    assert report['dng']['unique_valid']['passed']==1
    assert report['dng']['vun_yield']['rate_over_all_attempts'] is None


def test_identity_guards_cif_predictions_cases_and_partial_json(tmp_path):
    p=tmp_path/'a.cif';nacl().to(filename=str(p))
    record={'cif_path':'a.cif','cif_sha256':sha256(p)}
    assert len(load_structure(record,tmp_path))==2
    record['cif_sha256']='wrong'
    with pytest.raises(ValueError,match='CIF hash'): load_structure(record,tmp_path)
    samples=ledger(tmp_path/'samples.jsonl',[{'index':0,'case_id':'missing','status':'failed'}])
    with pytest.raises(ValueError,match='reference case'): evaluate(samples,'csp','space',tmp_path/'out',use_fingerprints=False)
    with pytest.raises(ValueError,match='Independent'): evaluate(samples,'dng','space',tmp_path/'out',assessments={'0':{'band_gap':1.}})
    samples.write_text('{"index":0}\n{',encoding='utf-8')
    with pytest.raises(ValueError,match='not modified'): read_records(samples)


def test_operation_check_identity_and_inversion():
    s=nacl()
    residual,metric=operation_check(s,'space',2,[{'R':(-np.eye(3)).tolist(),'t':[0,0,0]}])
    assert residual<1e-10 and metric<1e-10


def test_components_match_existing_teacher_forced_objective():
    import torch
    from unifiedgroupgen.proposal import DescriptorTransformer
    from unifiedgroupgen.symmetry import Descriptor,OrbitSpec
    from unifiedgroupgen.evaluation_components import descriptor_batch
    model=DescriptorTransformer([('space',2)],{},hidden=16,layers=1,heads=2,max_atoms=8,max_orbits=4,allowed_elements=[8,22]).eval()
    descriptor=Descriptor('space',2,(OrbitSpec('i',22),OrbitSpec('i',8)))
    with torch.inference_mode():
        value=descriptor_batch(model,[descriptor],[{}])[0]
        assert value['descriptor_loss']==pytest.approx(float(model.loss(descriptor,{})),abs=1e-6)
    assert value['parts']['G']['tokens']==1 and value['parts']['STOP']['tokens']==1
    assert value['parts']['W']['tokens']==2 and value['parts']['Z']['tokens']==2


def test_native_adapter_preserves_resume_attempts(tmp_path,monkeypatch):
    import torch
    spec=importlib.util.spec_from_file_location('unified_native',ROOT/'scripts/unified_eval_inference.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from unifiedgroupgen.proposal import DescriptorTransformer
    from unifiedgroupgen.flow import OrbitFlow
    model=DescriptorTransformer([('space',2)],{},hidden=16,layers=1,heads=2,max_atoms=4,max_orbits=2,allowed_elements=[6]).eval()
    flow=OrbitFlow({},hidden=16,layers=1,frequencies=2).eval()
    test=ledger(tmp_path/'test.jsonl',[{'id':'ref','dataset':'mp_20','descriptor':{'kind':'space','number':2,'orbits':[{'letter':'i','atomic_number':6}]},'state':[0]*9}])
    config={'test_files':[str(test)],'composition_probability':0.,'observation_probability':0.}
    seen=[]
    def prior(model,compiled,properties,steps,guidance,observation,composition,generator,**kwargs):
        seen.append(composition)
        state=compiled.prior('cpu',torch.float32,generator)*.05
        result=compiled.expand(state);result['state']=state;return result
    monkeypatch.setattr('unifiedgroupgen.flow.sample_flow',prior)
    job={'task':'csp','dataset':'mp_20','kind':'space','properties':{},'case_ids':['ref'],'k':2}
    module.generate(model,flow,config,job,tmp_path,'cpu',142,2,0.)
    digest=sha256(tmp_path/'samples.jsonl')
    assert len(read_records(tmp_path/'samples.jsonl'))==2
    assert seen==[None,None]
    for r in read_records(tmp_path/'samples.jsonl'):
        assert r['status']=='generated' and r['chemical_request']['composition']=={'6':2}
    module.generate(model,flow,config,job,tmp_path,'cpu',142,2,0.)
    assert sha256(tmp_path/'samples.jsonl')==digest and len(seen)==2


def test_csp_property_and_reference_proxy_distance(tmp_path):
    rows=[{'index':0,'basic_valid':True,'property_targets':{'band_gap':1.}},
          {'index':1,'basic_valid':True,'property_targets':{'band_gap':1.,'energy_above_hull':0.}}]
    summary=evaluate_properties(rows,{'0':{'band_gap':1.},'1':{'band_gap':1.}}, {},{'band_gap':.2,'energy_above_hull':.05}, {'r':{'band_gap':2.}})
    assert summary['quantities']['band_gap']['wasserstein_same_proxy']['value']==pytest.approx(1.)
    assert rows[1]['joint_property_hit'] is None
    assert summary['quantities']['energy_above_hull']['hit']['rate_over_all_attempts'] is None


def test_cli_png_manifest_resume(tmp_path):
    s=nacl();samples=ledger(tmp_path/'samples.jsonl',[{'index':0,'status':'generated','structure':s.as_dict()}])
    command=[sys.executable,str(ROOT/'scripts/evaluate_structures.py'),'--samples',str(samples),'--task','dng','--kind','space',
             '--output',str(tmp_path/'out'),'--no-fingerprints','--no-reidentify']
    first=subprocess.run(command,capture_output=True,text=True)
    assert first.returncode==0,first.stderr
    outputs=list((tmp_path/'out').glob('*.png'))
    assert len(outputs)==2 and not list((tmp_path/'out').glob('*.svg'))
    second=subprocess.run(command+['--resume'],capture_output=True,text=True)
    assert second.returncode==0 and 'Verified' in second.stdout
    (tmp_path/'out/metrics.png').write_bytes(b'changed')
    third=subprocess.run(command+['--resume'],capture_output=True,text=True)
    assert third.returncode!=0 and 'modified' in third.stderr


def test_cli_independent_property_csv_and_reference_identity(tmp_path):
    s=nacl();samples=ledger(tmp_path/'samples.jsonl',[{'index':0,'structure':s.as_dict(),'properties_requested':{'band_gap':1.}}])
    reference=ledger(tmp_path/'reference.jsonl',[{'id':'r','structure':s.as_dict()}])
    prediction=tmp_path/'predictions.csv';prediction.write_text('index,band_gap\n0,1.0\n',encoding='utf-8')
    ref_prediction=tmp_path/'reference_predictions.csv';ref_prediction.write_text('id,band_gap\nr,2.0\n',encoding='utf-8')
    metadata=tmp_path/'evaluator.json';write_json(metadata,{'evaluator':'independent_test_fixture','samples_sha256':sha256(samples),
        'predictions_sha256':sha256(prediction),'reference_sha256':{str(reference.resolve()):sha256(reference)},
        'reference_predictions_sha256':sha256(ref_prediction)})
    command=[sys.executable,str(ROOT/'scripts/evaluate_structures.py'),'--samples',str(samples),'--task','dng','--kind','space',
        '--reference-files',str(reference),'--assessments',str(prediction),'--assessment-metadata',str(metadata),
        '--reference-assessments',str(ref_prediction),'--output',str(tmp_path/'out'),'--no-fingerprints','--no-reidentify']
    result=subprocess.run(command,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    summary=json.loads((tmp_path/'out/summary.json').read_text(encoding='utf-8'))
    assert summary['properties']['quantities']['band_gap']['hit']['passed']==1
    assert summary['properties']['quantities']['band_gap']['wasserstein_same_proxy']['value']==pytest.approx(1.)
    assert (tmp_path/'out/properties.png').exists()


def test_main15_plan_is_read_only_and_conditions_train_only(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('unified_scheduler',ROOT/'scripts/evaluate_main15_unified.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    train=ledger(tmp_path/'train.jsonl',[{'id':'t','dataset':'c2db_51','properties':{'band_gap':2.}}])
    test=ledger(tmp_path/'test.jsonl',[{'id':'x','dataset':'c2db_51','properties':{'band_gap':99.}}])
    config={'dataset':'c2db_51','test_files':[str(test)],'train_files':[str(train)],'properties':{'band_gap':{'type':'scalar'}}}
    jobs=module.build_jobs(config,4,1,5,2,142)
    assert [j['task'] for j in jobs[:4]]==['components','dng','csp','csp']
    assert all(j.get('properties',{}).get('band_gap',2.)==2. for j in jobs)
    assert all('source_domain' not in j.get('properties',{}) for j in jobs)
    config['dataset_domains']={'c2db_51':1};config['properties']['source_domain']={'context':True}
    jobs=module.build_jobs(config,4,1,5,2,142)
    assert all(j['properties']['source_domain']==1 for j in jobs[1:])


def test_campaign_integration_resume_and_checkpoint_identity(tmp_path):
    import torch
    from unifiedgroupgen.cli import build_models
    config={'experiment_id':'mp20_u','dataset':'mp_20','kind':'space','groups':[['space',2]],
            'properties':{},'composition_probability':0.,'observation_probability':0.,
            'allowed_elements':[6],'max_atoms':4,'max_orbits':2,'require_data_audit':False,
            'model':{'proposal_hidden':16,'proposal_layers':1,'heads':2,'flow_hidden':16,'flow_layers':1,'frequencies':2}}
    record={'id':'c','dataset':'mp_20','descriptor':{'kind':'space','number':2,'orbits':[{'letter':'i','atomic_number':6}]},
            'state':[0.]*6+[.2,.3,.4],'properties':{}}
    reference=ledger(tmp_path/'reference.jsonl',[record])
    for split in ('train','val','test'): config[split+'_files']=[str(reference)]
    proposal,flow=build_models(config,'cpu')
    checkpoint=tmp_path/'checkpoints/mp20_u/model.best.pt';checkpoint.parent.mkdir(parents=True)
    saved={'config':config,'proposal':proposal.state_dict(),'flow':flow.state_dict(),'epoch':0,'runtime_options':{'coordinate_length_scale':3.}}
    torch.save(saved,checkpoint)
    output=tmp_path/'campaign'
    command=[sys.executable,str(ROOT/'scripts/evaluate_main15_unified.py'),'--models','mp20_u','--checkpoint-root',str(checkpoint.parents[1]),
        '--output',str(output),'--count','1','--csp-cases','1','--csp-k','1','--conditional-count','1','--steps','2','--device','cpu','--threads','1','--no-fingerprints']
    plan=subprocess.run(command+['--plan'],capture_output=True,text=True)
    assert plan.returncode==0,plan.stderr
    assert not (output/'pipeline.json').exists()
    run=subprocess.run(command+['--run','--resume'],capture_output=True,text=True)
    assert run.returncode==0,run.stderr+'\n'+run.stdout
    pipeline=json.loads((output/'pipeline.json').read_text(encoding='utf-8'))
    assert pipeline['state']=='complete' and len(pipeline['jobs'])==4
    assert all(j['state']=='complete' for j in pipeline['jobs'])
    assert (output/'summary.csv').exists() and (output/'model_comparison.png').exists()
    digest=sha256(output/'mp20_u/mp_20_csp/samples.jsonl')
    resumed=subprocess.run(command+['--run','--resume'],capture_output=True,text=True)
    assert resumed.returncode==0,resumed.stderr
    assert sha256(output/'mp20_u/mp_20_csp/samples.jsonl')==digest
    saved['epoch']=1;torch.save(saved,checkpoint)
    rejected=subprocess.run(command+['--run','--resume'],capture_output=True,text=True)
    assert rejected.returncode!=0 and 'identical inputs' in rejected.stderr
