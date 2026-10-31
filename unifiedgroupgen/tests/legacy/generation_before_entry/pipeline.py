from unifiedgroupgen.pl_modules.factory import load_model_checkpoint

from unifiedgroupgen.common.configuration import project_root, resolve_path
from pathlib import Path
from unifiedgroupgen.common.structures import composition_of
import json
import time
import torch
from unifiedgroupgen.group_constraints.projections import AffineObservation
from unifiedgroupgen.group_constraints.validation import to_structure, validate, summarize_attempts
from unifiedgroupgen.generation.flow import sample_flow
from unifiedgroupgen.group_constraints.compiler import compile_descriptor

from unifiedgroupgen.pl_modules.factory import build_models

from unifiedgroupgen.tests.legacy.generation_before_entry.requests import prepare_request


def generate_one(args, proposal, flow, config, request, generator, index, output, root_path):
    """Generate and validate one candidate; callers own failure recording and progress."""
    properties, fixed, composition, chemical = request.properties, request.fixed, request.composition, request.chemical
    kind, prefix, scaffold, formula_counts = request.kind, request.prefix, request.scaffold, request.formula_counts
    counts = composition
    if formula_counts:
        choice = int(torch.randint(len(formula_counts), (), generator=generator, device=args.device))
        counts = formula_counts[choice]
    # Pure-property models did not train the composition input. Keep it null;
    # enforce counts in the discrete decoder without an unseen embedding.
    learned_composition = counts if config.get('composition_probability', .5) > 0 else None
    descriptor = fixed or proposal.sample(properties, kind, counts, args.guidance, args.temperature, generator,
                                           chemical=chemical, prefix=prefix,
                                           condition_composition=config.get('composition_probability', .5) > 0)
    compiled = compile_descriptor(descriptor)
    if counts is not None and composition_of(compiled) != counts:
        raise ValueError("Fixed descriptor conflicts with requested composition")
    if not chemical.matches(composition_of(compiled)):
        raise ValueError('Generated descriptor violates requested element sets')
    observation = scaffold.observation(compiled) if scaffold is not None else None
    if args.observations:
        data = json.loads(root_path(args.observations).read_text(encoding='utf-8'))
        observation = (AffineObservation(data['matrix'], data['values']) if 'matrix' in data
                       else AffineObservation.fixed(compiled.dof, data['indices'], data['values']))
    if args.prior:
        state = compiled.prior(args.device, next(flow.parameters()).dtype, generator)
        if observation is not None:
            state = observation.retract(state)
        generated = compiled.expand(state)
        generated['state'] = state
    else:
        generated = sample_flow(flow, compiled, properties, args.steps, args.guidance, observation,
                                learned_composition, generator,
                                condition_observation=config.get('observation_probability', .25) > 0)
    report = validate(compiled, generated, isolate_identifier=True)
    if observation is not None:
        a, b = observation._tensors(generated['state'])
        report['observation_max_residual'] = float((a @ generated['state'] - b).abs().max())
    if scaffold is not None:
        report['scaffold'] = scaffold.report(compiled, generated['state'].cpu())
    to_structure(compiled, generated).to(filename=str(output/f'sample_{index:05d}.cif'))
    return {'index': index, 'descriptor': descriptor.to_dict(), 'state': generated['state'].cpu().tolist(),
                    'properties_requested': properties, 'validation': report, 'status': 'generated',
                    'chemical_request': {'allowed': sorted(chemical.allowed) if chemical.allowed else None,
                                         'required': sorted(chemical.required), 'composition': counts},
                    'scaffold_requested': bool(scaffold),
                    'generator': 'legal_prior' if args.prior else 'learned_flow'}


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
                  'seconds': time.monotonic()-started}
        temporary = output/'sampling.status.json.tmp'
        temporary.write_text(json.dumps(status), encoding='utf-8')
        temporary.replace(output/'sampling.status.json')
    progress('running')
    generator = torch.Generator(device=args.device).manual_seed(args.seed)
    for index in range(args.count):
        try:
            records.append(generate_one(args, proposal, flow, config, request, generator, index, output, root_path))
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
