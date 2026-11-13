"""Bounded MP-20/C2DB three-task integration tests; never a production training run."""
import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
import numpy as np
import torch
import pytorch_lightning as pl

from unifiedgroupgen.common.configuration import project_root, artifact_path, compose_experiment, training_definition
from unifiedgroupgen.common.source import file_hash
from unifiedgroupgen.conditional.properties import fit_property_scalers
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl, orbit_counts
from unifiedgroupgen.group_constraints.state import legal_actions
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.pl_data.joint_path import target_from_record, sample_training_path
from unifiedgroupgen.pl_data.task_path import task_target
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_modules.joint_training import fit_joint, load_joint_checkpoint, JointGeneratorModule
from unifiedgroupgen.group_constraints.group_prior import sample_group, NoFeasibleGroup
from unifiedgroupgen.generation.joint import sample_joint
from unifiedgroupgen.evaluation.joint_csp import evaluate_csp
from unifiedgroupgen.joint import evaluation_record
from unifiedgroupgen.tests.source_integrity import encoding_digest, PROTOCOL


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def scan(root, dataset, limits):
    selected, vocabulary, groups, fingerprints, unsupported = {}, set(), Counter(), {}, {}
    for split in ('train', 'val', 'test'):
        path = root/'data/dataset/full_verified_v1'/dataset/(split+'.jsonl')
        fingerprints[split] = file_hash(path)
        chosen, fallback, seen, outside = [], [], set(), 0
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                numbers = {w['atomic_number'] for w in row['descriptor']['orbits']}
                group = (row['descriptor']['kind'], row['descriptor']['number'])
                if split == 'train':
                    vocabulary.update(numbers); groups[group] += 1
                elif numbers-vocabulary:
                    outside += 1
                    continue
                if len(chosen) >= limits[split] or len(row['descriptor']['orbits']) > 16:
                    continue
                spec = catalog_spec(*group)
                n = sum(spec.template(w['letter']).multiplicity for w in row['descriptor']['orbits'])
                if n > 64:
                    continue
                if group not in seen:
                    chosen.append(row); seen.add(group)
                elif len(fallback) < limits[split]:
                    fallback.append(row)
        selected[split] = (chosen+fallback)[:limits[split]]
        unsupported[split] = outside
        if len(selected[split]) < limits[split]:
            raise ValueError('Too few supported bounded smoke records')
    prior = dict(source_split='train', train_records=sum(groups.values()),
                 entries=[dict(kind=k, number=n, count=count) for (k, n), count in sorted(groups.items())])
    return selected, sorted(vocabulary), prior, fingerprints, unsupported


def certify(rows):
    for row in rows:
        certification = row.get('provenance', {}).get('source_integrity', {})
        if (certification.get('protocol') != PROTOCOL or not certification.get('composition_conserved')
                or not certification.get('geometry_equivalent') or certification.get('encoding_sha256') != encoding_digest(row)):
            raise ValueError('Smoke record has no valid source-integrity certification')


def loss_on(module, rows):
    previous = module.training; module.eval()
    with torch.no_grad():
        losses = module.objective(rows, validation=True)
    module.train(previous)
    return {name: float(value) for name, value in losses.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='outputs/independent_tasks_short_'+datetime.now().strftime('%Y%m%d_%H%M%S'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--resume-completed', action='store_true', help='Verify and retain completed short tests; finish remaining tasks')
    parser.add_argument('--steps', type=int, default=12)
    parser.add_argument('--train-records', type=int, default=24)
    parser.add_argument('--eval-records', type=int, default=8)
    args = parser.parse_args()
    if not 1 <= args.steps <= 64 or not 2 <= args.train_records <= 64 or not 1 <= args.eval_records <= 16:
        parser.error('This runner is bounded to 1..64 steps, 2..64 train records and 1..16 evaluation records')
    torch.set_num_threads(1)
    root = project_root(); output = artifact_path(args.output)
    output.mkdir(parents=True, exist_ok=args.resume_completed)
    reports = []
    started = time.monotonic()
    for dataset, selector in [('mp_20', 'mp20'), ('c2db_51', 'c2db51')]:
        rows, vocabulary, prior, hashes, unsupported = scan(root, dataset,
            dict(train=args.train_records, val=args.eval_records, test=args.eval_records))
        certify(rows['train']+rows['val']+rows['test'])
        for split, records in rows.items():
            path = output/dataset/(split+'_subset.jsonl'); path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in records),encoding='utf-8')
        write(output/dataset/'selection.json',dict(source_hashes=hashes, train_prior=prior,
            unsupported_species_records=unsupported, selected={key:[r['id'] for r in values] for key,values in rows.items()},
            selection='first per observed group, <=64 atoms and <=16 orbits; bounded supported subset',
            scalers='selected TRAIN subset only', element_vocabulary='full TRAIN only'))
        for task in ('dng', 'property', 'csp'):
            run = output/dataset/task; run.mkdir(parents=True, exist_ok=args.resume_completed)
            cfg = training_definition(compose_experiment(f'task_{selector}_{task}'))
            cfg.update(allowed_elements=vocabulary, group_prior=prior, event_budget=96)
            cfg['model'].update(hidden=32, layers=2, heads=4)
            cfg['properties'] = fit_property_scalers(rows['train'], cfg.get('properties',{}))
            if args.resume_completed and (run/'short_test.json').exists():
                previous=json.loads((run/'short_test.json').read_text(encoding='utf-8'))
                if previous['steps']!=args.steps or json.loads((run/'config.json').read_text(encoding='utf-8'))!=cfg:
                    raise ValueError('Completed short-test configuration differs; use a new output')
                reports.append(previous)
                print(json.dumps(dict(event='retained_complete',dataset=dataset,task=task)),flush=True)
                continue
            if (run/'model.pt').exists():
                raise FileExistsError('An incomplete report already has weights; preserve it and use a new output')
            write(run/'config.json',cfg)
            print(json.dumps(dict(event='start',dataset=dataset,task=task,steps=args.steps)),flush=True)
            pl.seed_everything(142,workers=True)
            baseline = JointGeneratorModule(cfg).to(args.device)
            initial = loss_on(baseline,rows['val']); del baseline
            torch.cuda.empty_cache() if str(args.device).startswith('cuda') else None
            pl.seed_everything(142,workers=True)
            before = time.monotonic()
            trainer = fit_joint(cfg,rows['train'],rows['val'],run/'model.pt',device=args.device,
                epochs=max(2,args.steps),runtime=dict(batch_size=2,edge_budget=200000),
                max_steps=args.steps,split_hashes={dataset:hashes})
            assert trainer.global_step == args.steps
            validation = loss_on(trainer.lightning_module,rows['val'])
            test_rows = rows['test']
            if task=='property':
                from unifiedgroupgen.pl_data.task_path import has_property_target
                test_rows=[row for row in test_rows if has_property_target(row,cfg['properties'])]
            test = loss_on(trainer.lightning_module,test_rows)
            saved=load_joint_checkpoint(run/'model.pt',args.device)
            rebuilt=build_joint_model(saved['config'],args.device);rebuilt.load_state_dict(saved['model'],strict=True);rebuilt.eval()
            native=load_joint_checkpoint(run/'last.ckpt',args.device)
            assert saved['model'].keys()==native['model'].keys()
            for key in saved['model']:
                torch.testing.assert_close(saved['model'][key],native['model'][key],rtol=0,atol=0)
            max_coordinate_residual=0.; endpoint_checks=0
            for record in rows['train'][:8]+rows['val']+test_rows:
                target=target_from_record(record)
                if task=='property' and not any(target.properties.get(k) is not None for k in cfg['properties']):
                    continue
                prepared=task_target(target,cfg,31)
                current=sample_training_path(prepared,31,time=.55,allowed_elements=vocabulary,budget=96)
                if current.next_action is not None:
                    assert current.next_action in legal_actions(current.state,vocabulary)
                final=sample_training_path(prepared,31,time=1.,allowed_elements=vocabulary,budget=96)
                assert final.state.status=='done'
                assert orbit_counts(final.state.spec,final.state.letters,final.state.elements)==orbit_counts(target.state.spec,target.state.letters,target.state.elements)
                decoded=final.state.decoded();residual=final.state.compiled.residuals(decoded['coordinates'],decoded['atomic_numbers'],decoded['lattice'])
                assert residual['elements']
                max_coordinate_residual=max(max_coordinate_residual,float(residual['coordinates']))
                endpoint_checks+=1
            # Learned sampling uses only TRAIN group frequencies, property requests,
            # and (for CSP) the held-out composition. No true group or orbit is fed in.
            samples=[]
            cases=test_rows if task=='csp' else test_rows[:2]
            for index, record in enumerate(cases):
                target=target_from_record(record)
                chemistry=(ChemicalControl.from_inputs(counts=orbit_counts(target.state.spec,target.state.letters,target.state.elements))
                           if task=='csp' else ChemicalControl.from_inputs(allowed=vocabulary) if index==1 else ChemicalControl())
                properties={name:record['properties'][name] for name in cfg['properties'] if record['properties'].get(name) is not None}
                seed=500+index; generator=torch.Generator(device=args.device).manual_seed(seed)
                try:
                    spec=sample_group(prior,generator,kind=cfg['kind'],chemistry=chemistry,budget=96,
                                      max_atoms=cfg['max_atoms'],max_orbits=cfg['max_orbits'])
                    generated=sample_joint(rebuilt,spec,steps=12,epsilon=1e-3,budget=96,
                                           chemistry=chemistry,properties=properties,generator=generator)
                    row=evaluation_record(generated,index,seed,properties=properties)
                except NoFeasibleGroup as error:
                    row=dict(index=index,seed=seed,status='failed',generation_status='failed',accepted=False,reason=str(error))
                row.update(dataset=dataset,case_id=record['id'],candidate=0,generation_task=task,
                           properties_requested=properties,chemical_control=chemistry.to_dict())
                samples.append(row)
            path=run/'generate/smoke/samples.jsonl';path.parent.mkdir(parents=True)
            path.write_text(''.join(json.dumps(row,ensure_ascii=False,allow_nan=False)+'\n' for row in samples),encoding='utf-8')
            if task=='csp':
                matching=evaluate_csp(samples,cases)
                write(run/'evaluate/smoke/csp_metrics.json',matching)
            # Exercise the public CLI with a restored checkpoint and no group input.
            command=[sys.executable,'-m','unifiedgroupgen.joint','sample','--checkpoint',str(run/'model.pt'),
                     '--device',args.device,'--kind',cfg['kind'],'--steps','4','--output',str(run/'generate/cli/samples.jsonl')]
            if task=='property':command+=['--properties',json.dumps({name:test_rows[0]['properties'][name] for name in cfg['properties'] if test_rows[0]['properties'].get(name) is not None})]
            if task=='csp':
                t=target_from_record(cases[0]);command+=['--composition',json.dumps(orbit_counts(t.state.spec,t.state.letters,t.state.elements)),'--case-id',cases[0]['id']]
            cli=subprocess.run(command,cwd=root,capture_output=True,text=True,encoding='utf-8',timeout=300)
            (run/'cli.log').write_text(cli.stdout+cli.stderr,encoding='utf-8')
            if cli.returncode:raise RuntimeError(cli.stderr)
            report=dict(dataset=dataset,task=task,steps=trainer.global_step,initial_val=initial,
                final_val=validation,test=test,endpoint_checks=endpoint_checks,
                max_teacher_coordinate_residual_angstrom=max_coordinate_residual,
                strict_inference_and_native_checkpoint_reload=True,public_cli_without_group=True,
                learned_sampling=dict(attempts=len(samples),outcomes=dict(Counter(row['status'] for row in samples)),
                    reasons=dict(Counter(row.get('reason','') for row in samples))),
                seconds=time.monotonic()-before,checkpoint=str(run/'model.pt'),
                scope='Bounded execution, not converged generation quality; no property-hit or stability claim')
            write(run/'short_test.json',report);reports.append(report)
            write(output/'summary.json',dict(runs=reports,completed=len(reports),expected=6,
                 steps_per_task=args.steps,device=args.device,seconds=time.monotonic()-started,
                 joint_task_training=False,full_training=False,magnetic_testing=False))
            print(json.dumps(dict(event='complete',dataset=dataset,task=task,seconds=report['seconds'],
                                 sampling=report['learned_sampling'])),flush=True)
            del trainer,rebuilt,saved,native
            torch.cuda.empty_cache() if str(args.device).startswith('cuda') else None
    print(json.dumps(dict(event='all_complete',output=str(output),runs=len(reports))),flush=True)


if __name__=='__main__':
    main()
