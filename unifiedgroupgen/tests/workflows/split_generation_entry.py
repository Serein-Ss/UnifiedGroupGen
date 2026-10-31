"""Move generation orchestration to its entry without changing sampler mathematics."""
import ast
from datetime import datetime
import hashlib
import json
from pathlib import Path
import shutil

import psutil

ROOT=Path(__file__).resolve().parents[3]
PACKAGE=ROOT/'unifiedgroupgen'


def functions(source):
    return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


def replace_once(source,before,after):
    if source.count(before)!=1: raise ValueError('Unexpected source layout: '+before[:80])
    return source.replace(before,after)


def main():
    entry=(PACKAGE/'generate.py').read_text(encoding='utf-8')
    pipeline=(PACKAGE/'generation/pipeline.py').read_text(encoding='utf-8')
    tasks=(PACKAGE/'generation/tasks.py').read_text(encoding='utf-8')
    requests=(PACKAGE/'generation/requests.py').read_text(encoding='utf-8')
    pause_path=ROOT/'outputs/main15_evaluation/26-10-31-03-05-11/pause_receipt.json'
    pause=json.loads(pause_path.read_text(encoding='utf-8'))
    for row in pause['processes']:
        process=psutil.Process(row['pid'])
        if process.create_time()!=row['created'] or process.status()!=psutil.STATUS_STOPPED:
            raise ValueError('Evaluation must remain paused before source reorganization')
    backup=ROOT/'old_results/code_refactor'/datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    for directory in ('unifiedgroupgen','configs','scripts','docs'):
        shutil.copytree(ROOT/directory,backup/directory,
            ignore=shutil.ignore_patterns('__pycache__','*.egg-info','build','node_modules'))
    state=json.loads((pause_path.parent/'pipeline.json').read_text(encoding='utf-8'))
    # The new core was staged before this workflow; it is not part of the paused runtime.
    staged=backup/'unifiedgroupgen/generation/structure.py'
    if 'unifiedgroupgen/generation/structure.py' not in state['identity']['code']:
        staged.unlink()
    for name,expected in state['identity']['code'].items():
        original=ROOT/name
        archived=backup/name
        if hashlib.sha256(original.read_bytes()).hexdigest()!=expected or hashlib.sha256(archived.read_bytes()).hexdigest()!=expected:
            raise ValueError('Paused campaign implementation does not match its snapshot: '+name)
    location=ROOT/'intermediate/checks/generation_entry';location.mkdir(parents=True,exist_ok=True)
    receipt={'backup':str(backup),'paused_campaign':str(pause_path.parent),
             'paused_code_sha256':state['identity']['code'],
             'scope':'Old paused campaign retained; resume requires its original implementation, not the new working tree.'}
    (location/'migration.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    (backup/'paused_campaign.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')

    old=PACKAGE/'tests/legacy/generation_before_entry';old.mkdir(parents=True,exist_ok=False)
    (old/'__init__.py').write_text('"""Preserved generation orchestration for regression tests only."""\n',encoding='utf-8')
    for name,text in [('pipeline.py',pipeline),('tasks.py',tasks),('requests.py',requests)]:
        text=text.replace('from unifiedgroupgen.generation.requests import prepare_request',
                          'from unifiedgroupgen.tests.legacy.generation_before_entry.requests import prepare_request')
        (old/name).write_text(text,encoding='utf-8')

    e,p,t,r=map(functions,(entry,pipeline,tasks,requests))
    generic=p['generate_one']
    start=generic.index('    # Pure-property models')
    stop=generic.index('    observation = scaffold.observation')
    generic=generic[:start]+'''    learned_composition = counts if config.get('composition_probability', .5) > 0 else None
    compiled = sample_orbits(proposal, kind=kind, properties=properties, composition=counts,
        fixed=fixed, chemical=chemical, prefix=prefix, guidance=args.guidance,
        temperature=args.temperature, generator=generator,
        condition_composition=config.get('composition_probability', .5) > 0)
    descriptor = compiled.descriptor
'''+generic[stop:]
    start=generic.index('    if args.prior:')
    stop=generic.index('    report = validate')
    generic=generic[:start]+'''    generated = sample_geometry(flow, compiled, properties=properties, steps=args.steps,
        guidance=args.guidance, observation=observation, composition=learned_composition,
        generator=generator, prior=args.prior,
        condition_observation=config.get('observation_probability', .25) > 0)
'''+generic[stop:]

    batch=t['generate'].replace('def generate(', 'def generate_records(',1)
    for line in ["    from unifiedgroupgen.conditional.chemistry import ChemicalRules\n",
                 "    from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor\n",
                 "    from unifiedgroupgen.common.structures import composition_of\n",
                 "    from unifiedgroupgen.generation.flow import sample_flow\n",
                 "    from unifiedgroupgen.group_constraints.validation import to_structure\n"]:
        batch=replace_once(batch,line,'')
    start=batch.index('                descriptor=fixed or proposal.sample(')
    stop=batch.index("                phase='numeric_validation'")
    batch=batch[:start]+'''                compiled=sample_orbits(proposal,kind=job['kind'],properties=job['properties'],
                    composition=counts,fixed=fixed,chemical=ChemicalRules(),prefix=prefix,
                    guidance=guidance,temperature=1.,generator=generator,condition_composition=trained_composition)
                descriptor=compiled.descriptor
                phase='flow_integration'
                generated=sample_geometry(flow,compiled,properties=job['properties'],steps=steps,
                    guidance=guidance,composition=counts if trained_composition else None,generator=generator,
                    condition_observation=config.get('observation_probability',0)>0)
'''+batch[stop:]
    batch=replace_once(batch,'                from unifiedgroupgen.common.structures import load_structure, counts_of\n','')
    job=replace_once(e['generate_job'],'    from unifiedgroupgen.generation.tasks import generate\n','')
    job=replace_once(job,'    return generate(proposal, flow, config, job, args.output, args.device, args.seed, args.steps, args.guidance)',
                     '    return generate_records(proposal, flow, config, job, args.output, args.device, args.seed, args.steps, args.guidance)')
    execute=replace_once(e['execute'],'    from unifiedgroupgen.generation.pipeline import sample\n','')
    header='''"""Hydra generation entry: requests, checkpoints, task loops, storage and progress."""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path[0]=str(Path(__file__).resolve().parent.parent)
from dataclasses import dataclass
import hashlib
import json
import random
import time
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
from unifiedgroupgen.common.configuration import project_root, resolve_path, source_key, task_options
from unifiedgroupgen.common.structures import (sha256, write_json, export_cif, inline_cif_bytes,
    read_records, record_counts, verified_prior_records, composition_of, load_structure, counts_of)
from unifiedgroupgen.common.source import package_sources
from unifiedgroupgen.conditional.chemistry import ChemicalRules, integer_composition, feasible_formula_counts
from unifiedgroupgen.conditional.structure import Scaffold
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
from unifiedgroupgen.group_constraints.projections import AffineObservation
from unifiedgroupgen.group_constraints.validation import to_structure, validate, summarize_attempts
from unifiedgroupgen.generation.structure import sample_orbits, sample_geometry
from unifiedgroupgen.pl_modules.factory import build_models, load_model_checkpoint
from unifiedgroupgen.tests.components import checked_config
ROOT=project_root()
'''
    sections=[header,'@dataclass(frozen=True)\n'+r['GenerationRequest'],r['check_context'],r['scaffold_conditions'],
              r['prepare_request'],generic,p['sample'],batch,job,execute,
              "@hydra.main(version_base='1.3',config_path=str(project_root()/'configs'),config_name='generate')\n"+e['main'],
              "if __name__=='__main__': main()"]
    text='\n\n\n'.join(sections)+'\n';ast.parse(text)
    (PACKAGE/'generate.py').write_text(text,encoding='utf-8')
    for name in ('pipeline.py','tasks.py','requests.py'): (PACKAGE/'generation'/name).unlink()
    for relative in ('tests/legacy/cli.py','tests/legacy/evaluation_before_tasks/generation_cli.py',
                     'tests/legacy/training_and_evaluation_before_split.py'):
        path=PACKAGE/relative;text=path.read_text(encoding='utf-8')
        text=text.replace('unifiedgroupgen.generation.pipeline','unifiedgroupgen.tests.legacy.generation_before_entry.pipeline')
        text=text.replace('unifiedgroupgen.generation.tasks','unifiedgroupgen.tests.legacy.generation_before_entry.tasks')
        path.write_text(text,encoding='utf-8')
    for relative,before,after in [
        ('tests/unit/test_evaluate_entry.py','from unifiedgroupgen.generation.tasks import generate',
         'from unifiedgroupgen.generate import generate_records as generate'),
        ('tests/unit/test_lightning_training.py','from unifiedgroupgen.generation.pipeline import sample',
         'from unifiedgroupgen.generate import sample')]:
        path=PACKAGE/relative;text=path.read_text(encoding='utf-8')
        path.write_text(replace_once(text,before,after),encoding='utf-8')
    print(json.dumps({'backup':str(backup),'entry_lines':len((PACKAGE/'generate.py').read_text(encoding='utf-8').splitlines()),'sampling_resumed':False}))


if __name__=='__main__': main()
