"""Pure sampler boundary and fixed-seed parity with preserved generation paths."""
import ast
import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from unifiedgroupgen import generate as entry
from unifiedgroupgen.common.structures import read_records
from unifiedgroupgen.group_constraints.compiler import Descriptor,OrbitSpec,compile_descriptor
from unifiedgroupgen.group_constraints.validation import validate,to_structure
from unifiedgroupgen.pl_modules.descriptor import DescriptorTransformer
from unifiedgroupgen.pl_modules.orbitflow import OrbitFlow
from unifiedgroupgen.tests.legacy.generation_before_entry import pipeline as reference_pipeline
from unifiedgroupgen.tests.legacy.generation_before_entry.tasks import generate as reference_batch

ROOT=Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True,scope='module')
def single_thread():
    previous=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def models(kind):
    torch.manual_seed(917)
    groups=[(kind,1),(kind,2)]
    properties={'gap':{'type':'scalar','shift':0.,'scale':1.}}
    proposal=DescriptorTransformer(groups,properties,hidden=16,layers=1,heads=2,
        max_atoms=4,max_orbits=2,allowed_elements=[6]).eval()
    flow=OrbitFlow(properties,hidden=16,layers=1,frequencies=2).eval()
    descriptor=Descriptor(kind,2,(OrbitSpec('i',6),)) if kind=='space' else Descriptor(kind,1,(OrbitSpec('a',6),))
    return proposal,flow,descriptor


def test_generation_directory_is_only_in_memory_sampling():
    folder=ROOT/'unifiedgroupgen/generation'
    assert {p.name for p in folder.glob('*.py')}=={'__init__.py','descriptor.py','flow.py','rules.py','structure.py','prior.py'}
    for path in folder.glob('*.py'):
        tree=ast.parse(path.read_text(encoding='utf-8'))
        imports=[n.module or '' for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        assert not any(x.startswith(('hydra','argparse','pathlib','unifiedgroupgen.pl_modules.factory',
            'unifiedgroupgen.common.configuration','unifiedgroupgen.evaluation')) for x in imports)
        calls={n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)}
        assert not calls & {'read_text','write_text','mkdir','open','load_state_dict','manual_seed','to_csv'}
        assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ('open','print') for n in ast.walk(tree))
    assert {'prepare_request','generate_one','sample','generate_records','generate_job','execute'} <= {
        n.name for n in ast.parse((ROOT/'unifiedgroupgen/generate.py').read_text(encoding='utf-8')).body if isinstance(n,ast.FunctionDef)}


def test_descriptor_flow_and_legality_algorithms_are_unchanged():
    expected=json.loads((ROOT/'unifiedgroupgen/tests/legacy/generation_before_entry/math_sources.json').read_text())
    for name,digest in expected.items():
        if name == 'flow.py': continue  # Extended priors/solvers have a frozen numerical parity test.
        tree=ast.parse((ROOT/'unifiedgroupgen/generation'/name).read_text(encoding='utf-8'))
        assert hashlib.sha256(ast.dump(tree,include_attributes=False).encode()).hexdigest()==digest


def test_unchanged_request_validation_helpers_match_reference():
    def nodes(path):
        return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    folder=ROOT/'unifiedgroupgen/tests/legacy/generation_before_entry'
    current=nodes(ROOT/'unifiedgroupgen/generate.py')
    for name,body in nodes(folder/'requests.py').items():
        if name == 'prepare_request': continue  # Extended with learned-completion checkpoint validation.
        assert current[name]==body
    # The standalone loop now records precision; scientific parity is checked
    # below on generated descriptors/states/CIFs, rather than by identical AST.


@pytest.mark.parametrize('kind',['space','layer'])
@pytest.mark.parametrize('mode',['dng','csp','fixed_descriptor','group'])
@pytest.mark.parametrize('guidance',[0.,2.])
def test_batch_outputs_match_reference_at_same_seed(tmp_path,kind,mode,guidance):
    proposal,flow,descriptor=models(kind)
    reference=tmp_path/'reference.jsonl'
    reference.write_text(json.dumps({'id':'r','dataset':'fixture','descriptor':descriptor.to_dict()})+'\n')
    config={'test_files':[str(reference)],'composition_probability':0.,'observation_probability':0.}
    job={'task':'csp' if mode in ('csp','fixed_descriptor') else 'dng','case_ids':['r'],
         'k':2,'count':2,'dataset':'fixture','kind':kind,'properties':{'gap':1.1}}
    if mode=='fixed_descriptor':job['fixed_descriptor']=True
    if mode=='group':job.update(groups=[1,2],per_group=1)
    old,new=tmp_path/'old',tmp_path/'new';old.mkdir();new.mkdir()
    reference_batch(proposal,flow,config,job,old,'cpu',142,3,guidance)
    entry.generate_records(proposal,flow,config,job,new,'cpu',142,3,guidance)
    def stable(path):
        rows=read_records(path/'samples.jsonl')
        for row in rows:
            row.pop('wall_seconds')
            row.pop('precision',None)  # New execution metadata, not a structure change.
        return rows
    assert stable(old)==stable(new)
    status=json.loads((new/'sampling.status.json').read_text())
    assert status['attempted']==2 and status['generated']+status['failed']==2
    assert not list(new.glob('*.cif'))


@pytest.mark.parametrize('kind',['space','layer'])
@pytest.mark.parametrize('prior',[False,True])
@pytest.mark.parametrize('observed',['none','chart','scaffold'])
def test_single_structure_cfg_prior_and_observation_match_reference(tmp_path,monkeypatch,kind,prior,observed):
    proposal,flow,descriptor=models(kind)
    compiled=compile_descriptor(descriptor)
    file=tmp_path/'descriptor.json';file.write_text(json.dumps(descriptor.to_dict()))
    observation=tmp_path/'observations.json'
    observation.write_text(json.dumps({'indices':[0],'values':[0.]}))
    args=SimpleNamespace(properties='{"gap":1.1}',composition=None,formula=None,formula_units=[1],
        chemical_system=None,allowed_elements=None,required_elements=None,kind=kind,group=None,
        descriptor=str(file),scaffold=None,atom_map=None,scaffold_tolerance=1e-4,
        observations=str(observation) if observed=='chart' else None,device='cpu',prior=prior,steps=3,
        guidance=2.,temperature=1.)
    if observed=='scaffold':
        state=compiled.prior(generator=torch.Generator().manual_seed(19))*.1
        full=to_structure(compiled,compiled.expand(state))
        fragment=full.copy()
        if len(fragment)>1:fragment.remove_sites(list(range(1,len(fragment))))
        scaffold=tmp_path/'scaffold.cif';fragment.to(filename=str(scaffold))
        mapping=tmp_path/'map.json';mapping.write_text('{"0":0}')
        args.scaffold=str(scaffold);args.atom_map=str(mapping)
    config={'properties':{'gap':{}},'composition_probability':0.,'observation_probability':0.}
    request=entry.prepare_request(args,config,proposal,Path)
    # Compare real geometry/rules while avoiding a fresh identifier process per pair.
    check=lambda c,g,**kw:validate(c,g,isolate_identifier=False)
    monkeypatch.setattr(entry,'validate',check);monkeypatch.setattr(reference_pipeline,'validate',check)
    old,new=tmp_path/'old',tmp_path/'new';old.mkdir();new.mkdir()
    def draw(function,folder):
        random.seed(71);np.random.seed(71);torch.manual_seed(71)
        return function(args,proposal,flow,config,request,torch.Generator().manual_seed(71),0,folder,Path)
    before, after = draw(reference_pipeline.generate_one,old), draw(entry.generate_one,new)
    assert before == {k:v for k,v in after.items() if k not in ('cif','cif_sha256')}
    assert (old/'sample_00000.cif').read_text(encoding='utf-8') == after['cif']
    assert not list(new.glob('*.cif'))


def test_invalid_request_does_not_start_sampling(tmp_path):
    proposal,_,_=models('space')
    args=SimpleNamespace(properties='{}',descriptor=None,kind='space',composition='C2',formula='C',
        chemical_system=None,allowed_elements=None,required_elements=None)
    with pytest.raises(ValueError,match='Choose exact'):
        entry.prepare_request(args,{'properties':{}},proposal,Path)


def test_changed_campaign_code_is_rejected_before_result_writes(tmp_path,monkeypatch):
    checkpoint=tmp_path/'model.pt';checkpoint.write_bytes(b'fixture-checkpoint')
    study=tmp_path/'study';study.mkdir()
    pipeline=study/'pipeline.json'
    pipeline.write_text(json.dumps({'identity':{'code':{'unifiedgroupgen/generate.py':'old-source'}}}))
    before=pipeline.read_bytes()
    monkeypatch.setattr(entry,'load_model_checkpoint',lambda *a,**kw:{'config':{'test_files':[]}})
    monkeypatch.setattr(entry,'checked_config',lambda saved:saved['config'])
    args=SimpleNamespace(batch_size=1,steps=2,checkpoint=checkpoint,job={},seed=142,guidance=0.,
                         output=study/'fixture/job',device='cpu')
    with pytest.raises(ValueError,match='different implementation'):
        entry.generate_job(args)
    assert not args.output.exists() and pipeline.read_bytes()==before
