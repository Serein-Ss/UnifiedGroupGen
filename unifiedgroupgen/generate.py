"""Hydra generation entry: requests, checkpoints, task loops, storage and progress."""
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
from unifiedgroupgen.common.configuration import project_root, resolve_path, source_key, task_options, model_result_directory
from unifiedgroupgen.common.structures import (sha256, write_json, export_cif, inline_cif_bytes,
    read_records, record_counts, verified_prior_records, composition_of, load_structure, counts_of)
from unifiedgroupgen.common.source import package_sources
from unifiedgroupgen.common.precision import resolve_precision
from unifiedgroupgen.conditional.chemistry import ChemicalRules, integer_composition, feasible_formula_counts
from unifiedgroupgen.conditional.structure import Scaffold
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
from unifiedgroupgen.group_constraints.projections import AffineObservation
from unifiedgroupgen.group_constraints.validation import to_structure, validate, summarize_attempts
from unifiedgroupgen.generation.structure import sample_orbits, sample_geometry
from unifiedgroupgen.pl_modules.factory import build_models, load_model_checkpoint
from unifiedgroupgen.tests.components import checked_config
ROOT=project_root()



@dataclass(frozen=True)
class GenerationRequest:
    properties: dict
    fixed: Descriptor | None
    composition: dict | None
    chemical: ChemicalRules
    kind: str
    prefix: list | None
    scaffold: Scaffold | None
    formula_counts: list | None


def check_context(config, properties, fixed, kind):
    unknown = set(properties)-set(config.get('properties', {}))
    if unknown:
        raise ValueError(f"Checkpoint has no property encoder for {sorted(unknown)}")
    missing = [name for name, specification in config.get('properties', {}).items()
               if specification.get('context') and properties.get(name) is None]
    if missing:
        raise ValueError(f'Sampling requires context values for {missing}')
    if config.get('domain_label_support'):
        domain = properties['source_domain']
        if int(domain) != domain or str(int(domain)) not in config['domain_label_support']:
            raise ValueError(f'No trained calculation domain: {domain}')
        support = config['domain_label_support'][str(int(domain))]
        if (fixed.kind if fixed is not None else kind) not in support['kinds']:
            raise ValueError('Requested material kind was not trained in this calculation domain')
        unavailable = [name for name in properties if name != 'source_domain'
                       and properties[name] is not None and not support['property_counts'].get(name)]
        if unavailable:
            raise ValueError(f'No training property labels in this calculation domain for {unavailable}')


def scaffold_conditions(args, fixed, group, kind, proposal, root_path, prefix):
    scaffold = None
    if args.scaffold:
        if args.scaffold_tolerance <= 0 or not np.isfinite(args.scaffold_tolerance):
            raise ValueError('Scaffold tolerance must be positive and finite')
        if group is None:
            raise ValueError('Scaffold completion requires --group or --descriptor in the supplied setting')
        if args.observations:
            raise ValueError('Use scaffold observations or chart observations, not both')
        atom_map = json.loads(root_path(args.atom_map).read_text(encoding='utf-8')) if args.atom_map else None
        if atom_map is not None and fixed is None:
            raise ValueError('Incomplete mapped orbits require --descriptor')
        scaffold = Scaffold.from_cif(root_path(args.scaffold), kind, group, atom_map, args.scaffold_tolerance)
        if fixed is None:
            prefix = scaffold.prefix(proposal.vocab)
    elif args.atom_map:
        raise ValueError('--atom-map requires --scaffold')
    return scaffold, prefix


def prepare_request(args, config, proposal, root_path):
    completion_method = getattr(args, 'completion_method', 'auto')
    if completion_method not in ('auto','projection_only','masked_flow'):
        raise ValueError('Unknown structure completion method')
    if completion_method == 'masked_flow' and (not config.get('observation_policy')
                                               or config.get('observation_probability',0) <= 0):
        raise ValueError('Learned masked-flow completion requires a checkpoint trained on completion masks')
    properties = json.loads(args.properties)
    fixed = Descriptor.from_dict(json.loads(root_path(args.descriptor).read_text(encoding='utf-8'))) if args.descriptor else None
    check_context(config, properties, fixed, args.kind)
    if args.composition and args.formula:
        raise ValueError('Choose exact --composition or empirical --formula, not both')
    composition = integer_composition(args.composition) if args.composition else None
    chemical = ChemicalRules(args.chemical_system or args.allowed_elements,
                             args.chemical_system or args.required_elements)
    if args.chemical_system and (args.allowed_elements or args.required_elements):
        raise ValueError('--chemical-system means exactly this element set; do not combine set options')
    chemical.validate_domain(proposal.legality, composition)
    kind = fixed.kind if fixed else args.kind
    group = fixed.number if fixed else args.group
    if args.group is not None and fixed is not None and args.group != fixed.number:
        raise ValueError('Fixed descriptor and requested group conflict')
    prefix = ([proposal.vocab.index['BOS'], proposal.vocab.index[f'G:{kind}:{group}']]
              if group is not None else None)
    scaffold, prefix = scaffold_conditions(args, fixed, group, kind, proposal, root_path, prefix)
    formula_counts = None
    if args.formula:
        formula_counts = feasible_formula_counts(args.formula, args.formula_units, proposal.legality,
                                                kind, chemical, prefix)
        if fixed is not None:
            exact = composition_of(compile_descriptor(fixed))
            formula_counts = [counts for counts in formula_counts if counts == exact]
            if not formula_counts:
                raise ValueError('Fixed descriptor is incompatible with the requested formula units')
    if args.observations and fixed is None:
        raise ValueError("Chart-index observations require an explicit fixed descriptor")
    return GenerationRequest(properties, fixed, composition, chemical, kind, prefix, scaffold, formula_counts)


def generate_one(args, proposal, flow, config, request, generator, index, output, root_path):
    """Generate and validate one candidate; callers own failure recording and progress."""
    properties, fixed, composition, chemical = request.properties, request.fixed, request.composition, request.chemical
    kind, prefix, scaffold, formula_counts = request.kind, request.prefix, request.scaffold, request.formula_counts
    counts = composition
    if formula_counts:
        choice = int(torch.randint(len(formula_counts), (), generator=generator, device=args.device))
        counts = formula_counts[choice]
    learned_composition = counts if config.get('composition_probability', .5) > 0 else None
    compiled = sample_orbits(proposal, kind=kind, properties=properties, composition=counts,
        fixed=fixed, chemical=chemical, prefix=prefix, guidance=args.guidance,
        temperature=args.temperature, generator=generator,
        condition_composition=config.get('composition_probability', .5) > 0)
    descriptor = compiled.descriptor
    observation = scaffold.observation(compiled) if scaffold is not None else None
    if args.observations:
        data = json.loads(root_path(args.observations).read_text(encoding='utf-8'))
        observation = (AffineObservation(data['matrix'], data['values']) if 'matrix' in data
                       else AffineObservation.fixed(compiled.dof, data['indices'], data['values']))
    generated = sample_geometry(flow, compiled, properties=properties, steps=args.steps,
        guidance=args.guidance, observation=observation, composition=learned_composition,
        generator=generator, prior=args.prior,
        condition_observation=config.get('observation_probability', .25) > 0 and
                              getattr(args,'completion_method','auto')!='projection_only',
        solver=getattr(args,'solver','heun'))
    report = validate(compiled, generated, isolate_identifier=True)
    if observation is not None:
        a, b = observation._tensors(generated['state'])
        report['observation_max_residual'] = float((a @ generated['state'] - b).abs().max())
    if scaffold is not None:
        report['scaffold'] = scaffold.report(compiled, generated['state'].cpu())
    row = {'index': index, 'descriptor': descriptor.to_dict(), 'state': generated['state'].cpu().tolist(),
                    'properties_requested': properties, 'validation': report, 'status': 'generated',
                    'chemical_request': {'allowed': sorted(chemical.allowed) if chemical.allowed else None,
                                         'required': sorted(chemical.required), 'composition': counts},
                    'scaffold_requested': bool(scaffold),
                    'generator': 'legal_prior' if args.prior else 'learned_flow'}
    finish_inline(row, compiled, generated, output)
    return row


def sample(args, root=None):
    # Explicit workspace context avoids coupling the generator to a CLI global.
    from functools import partial
    root = Path(root).resolve() if root is not None else project_root()
    root_path = partial(resolve_path, root=root)
    from unifiedgroupgen.common.configuration import artifact_path as checked_artifact_path
    artifact_path = partial(checked_artifact_path, root=root)
    checkpoint = load_model_checkpoint(root_path(args.checkpoint), map_location=args.device, weights_only=False)
    config = checkpoint['config']
    proposal, flow = build_models(config, args.device)
    proposal.load_state_dict(checkpoint['proposal'])
    flow.load_state_dict(checkpoint['flow'])
    proposal.eval()
    flow.eval()
    proposal.inference_precision = flow.inference_precision = resolve_precision(args.device, getattr(args,'precision','fp32'))
    request = prepare_request(args, config, proposal, root_path)
    if args.count < 1 or args.steps < 1:
        raise ValueError('Sampling count and steps must be positive')
    output = artifact_path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    started = time.monotonic()
    attempts_path = output/'samples.jsonl'
    if attempts_path.exists():
        raise ValueError('Sampling attempts already exist; use a new output directory')
    attempts_path.write_text('', encoding='utf-8')
    def progress(state):
        status = {'state': state, 'attempted': len(records), 'requested': args.count,
                  'generated': sum(r['status'] == 'generated' for r in records),
                  'seconds': time.monotonic()-started, 'precision':flow.inference_precision}
        temporary = output/'sampling.status.json.tmp'
        temporary.write_text(json.dumps(status), encoding='utf-8')
        temporary.replace(output/'sampling.status.json')
    progress('running')
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    for index in range(args.count):
        try:
            records.append(generate_one(args, proposal, flow, config, request, generator, index, output, root_path))
            records[-1]['precision'] = flow.inference_precision
        except (ValueError, RuntimeError, FloatingPointError) as error:
            records.append({'index': index, 'status': 'failed', 'reason': str(error)})
        # Persist every attempt, including failures, before advancing to the next one.
        with attempts_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(records[-1], allow_nan=False)+'\n')
        progress('running')
        if (index+1) % 10 == 0 or index+1 == args.count:
            print(json.dumps({'event': 'sampling_progress', 'attempted': index+1,
                              'requested': args.count, 'seconds': time.monotonic()-started}), flush=True)
    (output/'summary.json').write_text(json.dumps(summarize_attempts(records), indent=2), encoding='utf-8')
    progress('complete')
    print(json.dumps({'requested': args.count, 'generated': sum(r['status']=='generated' for r in records),
                      'output': str(output), 'all_attempts_recorded': True}))


def check_record_precision(records, flow):
    expected = getattr(flow,'inference_precision','fp32')
    if any(record.get('precision','fp32') != expected for record in records):
        raise ValueError('Existing samples use a different inference precision; preserve them and choose a new output')


def generate_records(proposal,flow,config,job,output,device,seed,steps,guidance,batch_size=1,solver='heun'):
    if batch_size > 1 and not job.get('prior_samples'):
        return generate_records_batched(proposal,flow,config,job,output,device,seed,steps,guidance,batch_size,solver)
    refs=({str(r['id']):r for p in config['test_files'] for r in read_records(resolve_path(ROOT/p,ROOT)) if r['dataset']==job['dataset']}
          if job['task']=='csp' else {})
    if job['task']=='csp':
        attempts=[(key,candidate) for key in job['case_ids'] for candidate in range(job['k'])]
    else:
        attempts=[(None,i) for i in range(job['count'])]
    ledger=output/'samples.jsonl'
    prior=verified_prior_records(job['prior_samples']) if job.get('prior_samples') else []
    by_case={str(r.get('case_id',r.get('reference_id'))):r for r in prior} if job['task']=='csp' else {}
    previous=read_records(ledger) if ledger.exists() else []
    check_record_precision(prior, flow)
    check_record_precision(previous, flow)
    sampling_seconds=sum(r.get('wall_seconds',0.) or 0. for r in previous)
    reused_count=sum(bool(r.get('reused_from')) for r in previous)
    generated_count=sum(r['status']=='generated' for r in previous)
    if [r['index'] for r in previous]!=list(range(len(previous))) or len(previous)>len(attempts):
        raise ValueError('Attempt ledger is incomplete/corrupt; it was preserved')
    for r in previous:
        if r['status']=='generated':
            if 'cif' in r:
                inline_cif_bytes(r)
            elif sha256(output/r['cif_path'])!=r['cif_sha256']:
                raise ValueError('Existing CIF no longer matches its receipt')
    with torch.inference_mode():
        for index,(case,candidate) in enumerate(attempts):
            if index<len(previous): continue
            started=time.perf_counter()
            # Per-attempt seeds make interrupted/resumed generation identical.
            attempt_seed=seed+index;random.seed(attempt_seed);np.random.seed(attempt_seed);torch.manual_seed(attempt_seed)
            generator=torch.Generator(device=device).manual_seed(attempt_seed)
            requested_group=(job['groups'][index//job['per_group']] if job.get('groups') else job.get('group'))
            row={'index':index,'case_id':case,'candidate':candidate,'properties_requested':job['properties'],
                 'requested_group':requested_group,'seed':attempt_seed,'precision':getattr(flow,'inference_precision','fp32')}
            reused=(by_case.get(str(case)) if candidate==0 else None) if case is not None else (prior[index] if index<len(prior) else None)
            if reused is not None:
                reused_count+=1
                generated_count+=reused['status']=='generated'
                row={**reused,'index':index,'case_id':case,'candidate':candidate,
                     'requested_group':requested_group,'properties_requested':job['properties']}
                with ledger.open('a',encoding='utf-8') as stream:
                    stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
                write_json(output/'sampling.status.json',{'state':'running','attempted':index+1,'requested':len(attempts),
                    'sampling_seconds':sampling_seconds,'reused':reused_count,'new_attempts':index+1-reused_count,
                    'generated':generated_count,'failed':index+1-generated_count,'last_action':'import_verified_previous_candidate'})
                continue
            phase='descriptor_sampling'
            try:
                reference=refs[case] if case is not None else None
                counts=record_counts(reference,ROOT) if reference else None
                row['chemical_request']={'composition':counts}
                fixed=Descriptor.from_dict(reference['descriptor']) if reference and job.get('fixed_descriptor') else None
                prefix=[proposal.vocab.index['BOS'],proposal.vocab.index[f"G:{job['kind']}:{requested_group}"]] if requested_group else None
                trained_composition=config.get('composition_probability',.5)>0
                compiled=sample_orbits(proposal,kind=job['kind'],properties=job['properties'],
                    composition=counts,fixed=fixed,chemical=ChemicalRules(),prefix=prefix,
                    guidance=guidance,temperature=1.,generator=generator,condition_composition=trained_composition)
                descriptor=compiled.descriptor
                phase='flow_integration'
                generated=sample_geometry(flow,compiled,properties=job['properties'],steps=steps,
                    guidance=guidance,composition=counts if trained_composition else None,generator=generator,
                    condition_observation=config.get('observation_probability',0)>0,solver=solver)
                phase='numeric_validation'
                if not torch.isfinite(generated['state']).all():
                    raise FloatingPointError('Nonfinite continuous state rejected before CIF export')
                structure=to_structure(compiled,generated)
                phase='cif_export'
                cif=structure.to(fmt='cif')
                row.update(cif=cif,cif_sha256=hashlib.sha256(cif.encode('utf-8')).hexdigest())
                # Native output is only successful after in-memory CIF round-trip verification.
                phase='cif_readback'
                restored=load_structure(row,output)
                if counts_of(restored)!=counts_of(structure):
                    raise RuntimeError('CIF export/readback changed integer composition')
                if not np.isfinite(restored.lattice.matrix).all() or not np.isfinite(restored.frac_coords).all():
                    raise FloatingPointError('Nonfinite CIF readback')
                row.update(status='generated',descriptor=descriptor.to_dict(),state=generated['state'].cpu().tolist())
            except (ValueError,RuntimeError,FloatingPointError,KeyError) as error:
                if isinstance(error,torch.cuda.OutOfMemoryError): raise
                row.update(status='failed',failure_stage=phase,error_type=type(error).__name__,reason=str(error))
            row['wall_seconds']=time.perf_counter()-started
            sampling_seconds+=row['wall_seconds']
            generated_count+=row['status']=='generated'
            with ledger.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            write_json(output/'sampling.status.json',{'state':'running','attempted':index+1,'requested':len(attempts),
                'last_attempt_seconds':row['wall_seconds'],'sampling_seconds':sampling_seconds,
                'reused':reused_count,'new_attempts':index+1-reused_count,
                'generated':generated_count,'failed':index+1-generated_count})
            if (index+1)%10==0 or index+1==len(attempts):
                print(json.dumps({'phase':'sample','attempted':index+1,'requested':len(attempts),'sampling_seconds':sampling_seconds}),flush=True)
    write_json(output/'sampling.status.json',{'state':'complete','attempted':len(attempts),'requested':len(attempts),
        'reused':reused_count,'new_attempts':len(attempts)-reused_count,'sampling_seconds':sampling_seconds,
        'generated':generated_count,'failed':len(attempts)-generated_count})


def integrate_candidates(flow, candidates, steps, guidance, solver='heun', observed_inputs=True, prior=False):
    """Batch the neural integration; isolate numerical failures with the SAME initial states."""
    from unifiedgroupgen.generation.flow import sample_flow_batch
    from unifiedgroupgen.generation.prior import sample_prior
    if not candidates: return []
    parameter = next(flow.parameters())
    states = [sample_prior(c['chart'],parameter.device,parameter.dtype,c['generator'],getattr(flow,'prior',None))
              for c in candidates]
    def integrate(ids):
        charts = [candidates[i]['chart'] for i in ids]
        observations = [candidates[i].get('observation') for i in ids]
        if prior:
            values = [o.retract(states[i]) if o is not None else states[i] for i,o in zip(ids,observations)]
            return [{**c.expand(z),'state':z} for c,z in zip(charts,values)]
        return sample_flow_batch(flow,charts,[candidates[i]['properties'] for i in ids],steps,guidance,
            observations,[candidates[i].get('composition') for i in ids],solver=solver,
            condition_observation=observed_inputs,initial_states=[states[i] for i in ids])
    try:
        return integrate(list(range(len(candidates))))
    except torch.cuda.OutOfMemoryError:
        raise
    except (ValueError,RuntimeError,FloatingPointError):
        results = []
        for i in range(len(candidates)):
            try: results.append(integrate([i])[0])
            except torch.cuda.OutOfMemoryError: raise
            except (ValueError,RuntimeError,FloatingPointError) as error: results.append(error)
        return results


def finish_inline(row, chart, generated, output):
    if isinstance(generated,Exception): raise generated
    structure = to_structure(chart,generated)
    cif = structure.to(fmt='cif')
    row.update(cif=cif,cif_sha256=hashlib.sha256(cif.encode('utf-8')).hexdigest())
    restored = load_structure(row,output)
    if counts_of(restored)!=counts_of(structure): raise RuntimeError('CIF round-trip changed composition')
    if not np.isfinite(restored.lattice.matrix).all() or not np.isfinite(restored.frac_coords).all():
        raise FloatingPointError('Nonfinite exported structure')
    row.update(status='generated',descriptor=chart.descriptor.to_dict(),state=generated['state'].cpu().tolist())


def generate_records_batched(proposal,flow,config,job,output,device,seed,steps,guidance,batch_size,solver):
    refs = ({str(r['id']):r for p in config['test_files'] for r in read_records(resolve_path(ROOT/p,ROOT))
             if r['dataset']==job['dataset']} if job['task']=='csp' else {})
    attempts = ([(key,k) for key in job['case_ids'] for k in range(job['k'])] if refs else
                [(None,i) for i in range(job['count'])])
    ledger = output/'samples.jsonl'
    previous = read_records(ledger) if ledger.exists() else []
    check_record_precision(previous, flow)
    if [r['index'] for r in previous]!=list(range(len(previous))) or len(previous)>len(attempts):
        raise ValueError('Incomplete/corrupt ledger; existing results preserved')
    for row in previous:
        if row['status']=='generated': inline_cif_bytes(row)
    elapsed = sum(r.get('wall_seconds',0.) for r in previous)
    success = sum(r['status']=='generated' for r in previous)
    for start in range(len(previous),len(attempts),batch_size):
        started = time.perf_counter(); rows,candidates,positions = [],[],[]
        for index in range(start,min(start+batch_size,len(attempts))):
            case,k = attempts[index];attempt_seed=seed+index
            random.seed(attempt_seed);np.random.seed(attempt_seed);torch.manual_seed(attempt_seed)
            generator=torch.Generator(device=device).manual_seed(attempt_seed)
            group=job['groups'][index//job['per_group']] if job.get('groups') else job.get('group')
            row={'index':index,'case_id':case,'candidate':k,'properties_requested':job['properties'],
                 'requested_group':group,'seed':attempt_seed,'sampling_backend':'packed','solver':solver,
                 'precision':getattr(flow,'inference_precision','fp32')}
            rows.append(row)
            try:
                reference=refs[case] if case is not None else None
                counts=record_counts(reference,ROOT) if reference else None
                row['chemical_request']={'composition':counts}
                fixed=Descriptor.from_dict(reference['descriptor']) if reference and job.get('fixed_descriptor') else None
                prefix=[proposal.vocab.index['BOS'],proposal.vocab.index[f"G:{job['kind']}:{group}"]] if group else None
                trained=config.get('composition_probability',.5)>0
                chart=sample_orbits(proposal,kind=job['kind'],properties=job['properties'],composition=counts,
                    fixed=fixed,chemical=ChemicalRules(),prefix=prefix,guidance=guidance,generator=generator,
                    condition_composition=trained)
                positions.append(len(rows)-1)
                candidates.append({'chart':chart,'generator':generator,'properties':job['properties'],
                                   'composition':counts if trained else None})
            except (ValueError,RuntimeError,KeyError) as error:
                if isinstance(error,torch.cuda.OutOfMemoryError): raise
                row.update(status='failed',failure_stage='descriptor_sampling',error_type=type(error).__name__,reason=str(error))
        with torch.inference_mode():
            generated=integrate_candidates(flow,candidates,steps,guidance,solver,
                                            config.get('observation_probability',0)>0)
            for position,candidate,result in zip(positions,candidates,generated):
                try: finish_inline(rows[position],candidate['chart'],result,output)
                except (ValueError,RuntimeError,FloatingPointError) as error:
                    if isinstance(error,torch.cuda.OutOfMemoryError): raise
                    rows[position].update(status='failed',failure_stage='flow_or_cif',error_type=type(error).__name__,reason=str(error))
        duration=time.perf_counter()-started;elapsed+=duration
        with ledger.open('a',encoding='utf-8') as stream:
            for row in rows:
                row.update(wall_seconds=duration/len(rows),batch_wall_seconds=duration,
                           timing_definition='Batch elapsed divided by attempts; not individual latency')
                success+=row['status']=='generated';stream.write(json.dumps(row,allow_nan=False)+'\n')
        completed=start+len(rows)
        write_json(output/'sampling.status.json',{'state':'complete' if completed==len(attempts) else 'running',
            'attempted':completed,'requested':len(attempts),'generated':success,'failed':completed-success,
            'sampling_seconds':elapsed,'sampling_backend':'packed','batch_size':batch_size})
        print(json.dumps({'event':'sampling_progress','attempted':completed,'requested':len(attempts),
                          'sampling_seconds':elapsed,'batch_size':batch_size}),flush=True)


def sample_batched(args):
    """Configured interactive tasks share one ledger and a packed flow backend."""
    saved=load_model_checkpoint(resolve_path(args.checkpoint),map_location=args.device,weights_only=False)
    config=saved['config'];proposal,flow=build_models(config,args.device)
    proposal.load_state_dict(saved['proposal'],strict=True);flow.load_state_dict(saved['flow'],strict=True)
    proposal.eval();flow.eval()
    proposal.inference_precision = flow.inference_precision = resolve_precision(args.device, getattr(args,'precision','fp32'))
    request=prepare_request(args,config,proposal,resolve_path)
    output=resolve_path(args.output);output.mkdir(parents=True,exist_ok=True)
    ledger=output/'samples.jsonl'
    if ledger.exists(): raise FileExistsError('Use a new output; existing samples preserved')
    elapsed=0.;success=0
    for start in range(0,args.count,args.batch_size):
        started=time.perf_counter();rows,candidates,positions=[],[],[]
        for index in range(start,min(start+args.batch_size,args.count)):
            attempt_seed=args.seed+index;generator=torch.Generator(device=args.device).manual_seed(attempt_seed)
            random.seed(attempt_seed);np.random.seed(attempt_seed);torch.manual_seed(attempt_seed)
            row={'index':index,'seed':attempt_seed,'properties_requested':request.properties,'sampling_backend':'packed',
                 'precision':flow.inference_precision,
                 'generator':'legal_prior' if args.prior else 'learned_flow',
                 'completion_method':('masked_flow' if config.get('observation_policy') and
                    getattr(args,'completion_method','auto')!='projection_only' else 'projection_only')
                    if request.scaffold or args.observations else None};rows.append(row)
            try:
                counts=request.composition
                if request.formula_counts:
                    counts=request.formula_counts[int(torch.randint(len(request.formula_counts),(),device=args.device,generator=generator))]
                chart=sample_orbits(proposal,kind=request.kind,properties=request.properties,composition=counts,
                    fixed=request.fixed,chemical=request.chemical,prefix=request.prefix,guidance=args.guidance,
                    temperature=args.temperature,generator=generator,condition_composition=config.get('composition_probability',.5)>0)
                observation=request.scaffold.observation(chart) if request.scaffold else None
                if args.observations:
                    data=json.loads(resolve_path(args.observations).read_text(encoding='utf-8'))
                    observation=(AffineObservation(data['matrix'],data['values']) if 'matrix' in data else
                                 AffineObservation.fixed(chart.dof,data['indices'],data['values']))
                positions.append(len(rows)-1);candidates.append({'chart':chart,'properties':request.properties,
                    'observation':observation,'composition':counts if config.get('composition_probability',.5)>0 else None,
                    'generator':generator})
                row['chemical_request']={'composition':counts,
                    'allowed':sorted(request.chemical.allowed) if request.chemical.allowed else None,
                    'required':sorted(request.chemical.required)}
            except (ValueError,RuntimeError,KeyError) as error:
                if isinstance(error,torch.cuda.OutOfMemoryError): raise
                row.update(status='failed',failure_stage='request',reason=str(error))
        with torch.inference_mode():
            results=integrate_candidates(flow,candidates,args.steps,args.guidance,getattr(args,'solver','heun'),
                config.get('observation_probability',.25)>0 and
                    getattr(args,'completion_method','auto')!='projection_only',args.prior)
            for position,candidate,result in zip(positions,candidates,results):
                try:
                    finish_inline(rows[position],candidate['chart'],result,output)
                    rows[position]['validation']=validate(candidate['chart'],result,isolate_identifier=True)
                    if candidate.get('observation') is not None:
                        a,b=candidate['observation']._tensors(result['state'])
                        rows[position]['validation']['observation_max_residual']=float((a@result['state']-b).abs().max())
                    if request.scaffold:
                        rows[position]['scaffold_requested']=True
                        rows[position]['validation']['scaffold']=request.scaffold.report(candidate['chart'],result['state'].cpu())
                except (ValueError,RuntimeError,FloatingPointError) as error:
                    if isinstance(error,torch.cuda.OutOfMemoryError): raise
                    rows[position].update(status='failed',failure_stage='flow_or_validation',reason=str(error))
        duration=time.perf_counter()-started;elapsed+=duration
        with ledger.open('a',encoding='utf-8') as stream:
            for row in rows:
                row.update(wall_seconds=duration/len(rows),batch_wall_seconds=duration)
                success+=row['status']=='generated';stream.write(json.dumps(row,allow_nan=False)+'\n')
        completed=start+len(rows)
        write_json(output/'sampling.status.json',{'state':'complete' if completed==args.count else 'running',
            'attempted':completed,'requested':args.count,'generated':success,'failed':completed-success,
            'sampling_seconds':elapsed,'batch_size':args.batch_size,'sampling_backend':'packed'})
    write_json(output/'summary.json',{'requested':args.count,'generated':success,'failed':args.count-success,
                                    'sampling_seconds':elapsed,'structure_storage':'inline_cif_utf8_v1'})


def generate_job(args):
    """Load one checkpoint and continue a verified per-attempt generation ledger."""
    if args.batch_size < 1 or args.steps < 1:
        raise ValueError('Batch size and flow steps must be positive')
    saved = load_model_checkpoint(args.checkpoint, map_location='cpu', weights_only=False)
    config = checked_config(saved)
    job = args.job
    if job.get('checkpoint_sha256') and job['checkpoint_sha256'] != sha256(args.checkpoint):
        raise ValueError('Checkpoint changed after the evaluation plan was created')
    identity = {'checkpoint_sha256':sha256(args.checkpoint), 'job':job, 'seed':args.seed,
                'steps':args.steps, 'guidance':args.guidance,
                'test':{p:sha256(resolve_path(p,ROOT)) for p in config['test_files']},
                'structure_storage':'inline_cif_utf8_v1', 'batch_size':args.batch_size, 'solver':getattr(args,'solver','heun'),
                'precision':resolve_precision(args.device,getattr(args,'precision','fp32')),
                'code':{source_key(p,ROOT):sha256(p) for p in package_sources()}}
    for parent in args.output.parents:
        pipeline=parent/'pipeline.json'
        if not pipeline.is_file(): continue
        recorded=json.loads(pipeline.read_text(encoding='utf-8')).get('identity',{}).get('code',{})
        expected={name:digest for name,digest in recorded.items() if name.startswith('unifiedgroupgen/')}
        if expected and expected!=identity['code']:
            raise ValueError('Campaign uses a different implementation; resume its saved source snapshot '
                             'or choose a new output directory. Existing results were preserved.')
        if expected: break
    args.output.mkdir(parents=True, exist_ok=True)
    receipt = args.output/'identity.json'
    if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8')) != identity:
        raise ValueError('Inference identity changed; preserve results and choose a new output directory')
    write_json(receipt, identity)
    proposal, flow = build_models(config, args.device)
    proposal.load_state_dict(saved['proposal'], strict=True)
    flow.load_state_dict(saved['flow'], strict=True)
    proposal.eval(); flow.eval()
    proposal.inference_precision = flow.inference_precision = identity['precision']
    return generate_records(proposal, flow, config, job, args.output, args.device, args.seed, args.steps, args.guidance,
                            args.batch_size,getattr(args,'solver','heun'))


def execute(cfg):
    torch.set_num_threads(cfg.get('threads',2))
    seed=cfg.get('seed',142)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if cfg.task=='export_cif':
        args=task_options(cfg,'export')
        print(json.dumps(export_cif(args.samples,args.index,args.output)))
        return
    if cfg.task!='generate': raise ValueError('generate.py accepts generate or export_cif')
    args=task_options(cfg,'generation')
    if args.device=='auto': args.device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.checkpoint is None: raise ValueError('generation.checkpoint is required')
    if args.output is None:
        task = getattr(args, 'task_name', None) or (args.job['name'] if args.job else
            f"{args.kind}_{'fixed_descriptor' if args.descriptor else 'csp' if args.composition or args.formula else 'group_'+str(args.group) if args.group else 'dng'}_seed{args.seed}")
        args.output = model_result_directory(args.checkpoint, 'generate', task)
    if getattr(args,'job',None) is not None: return generate_job(args)
    if args.batch_size < 1 or args.count < 1 or args.steps < 1:
        raise ValueError('Batch size, count and integration steps must be positive')
    if args.solver not in ('heun','euler'): raise ValueError('solver must be heun or euler')
    args.properties=json.dumps(args.properties)
    if isinstance(args.composition,dict): args.composition=json.dumps(args.composition)
    return sample_batched(args) if args.batch_size > 1 else sample(args)


@hydra.main(version_base='1.3',config_path=str(project_root()/'configs'),config_name='generate')
def main(cfg):
    execute(cfg)


if __name__=='__main__': main()
