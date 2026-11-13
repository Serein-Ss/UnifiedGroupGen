"""Opt-in command adapter: audited training, bounded smoke checks and joint sampling."""
import argparse
import itertools
import json
from pathlib import Path
import torch
from unifiedgroupgen.common.configuration import (
    project_root, resolve_path, artifact_path, compose_experiment, training_definition,
)


def _head(path, count):
    with resolve_path(path).open(encoding='utf-8') as stream:
        return [json.loads(line) for line in itertools.islice((s for s in stream if s.strip()), count)]


def sample_to_dict(sample):
    state = sample.state
    result = dict(status=state.status, reason=sample.reason, time=sample.time,
                  events=sample.events, integration_steps=sample.integration_steps,
                  epsilon=sample.epsilon, kind=state.spec.kind, group=state.spec.number,
                  letters=list(state.letters), orbit_elements=list(state.elements),
                  moment_field_enabled=state.moments is not None,
                  metric_state=state.metric.detach().cpu().tolist() if torch.isfinite(state.metric).all() else None,
                  coordinate_state=state.coordinates.detach().cpu().tolist() if torch.isfinite(state.coordinates).all() else None)
    from unifiedgroupgen.group_constraints.unified import catalog_spec
    builtin = (state.spec.kind in ('space','layer','plane') and state.spec.number > 0
               and state.spec == catalog_spec(state.spec.kind,state.spec.number))
    if not builtin:
        result['symmetry'] = state.spec.to_dict()
    from unifiedgroupgen.group_constraints.chemistry import orbit_counts
    result['chemical_control'] = state.chemistry.to_dict()
    result['current_counts'] = orbit_counts(state.spec, state.letters, state.elements)
    result['remaining_inventory'] = dict(state.chemistry.remaining(state.spec, state.letters, state.elements))
    if state.status == 'done':
        decoded = state.decoded()
        result.update({name: decoded[name].detach().cpu().tolist() for name in
                       ('coordinates', 'lattice', 'atomic_numbers', 'orbit_index')})
        result['periodic'] = list(state.spec.periodic)
        result['moments'] = None if decoded['moments'] is None else decoded['moments'].detach().cpu().tolist()
        residuals = state.compiled.residuals(decoded['coordinates'], decoded['atomic_numbers'], decoded['lattice'], decoded['moments'])
        result['constraint_residuals'] = {name: value.item() if torch.is_tensor(value) else value for name, value in residuals.items()}
    return result


def evaluation_record(sample, index, seed, exact=False, properties=None, requested_group=None):
    """Retain every attempt and export ordinary structures in the existing CIF ledger contract."""
    import hashlib
    row = sample_to_dict(sample)
    row.update(index=index, case_id=None, candidate=index, seed=seed,
               generation_status=row['status'], status='failed', accepted=False,
               requested_group=sample.state.spec.number, properties_requested=properties or {},
               precision='fp32', sampling_backend='joint_atom_graph', solver='event_heun')
    if requested_group is not None:
        row['requested_group'] = requested_group
    if sample.state.status != 'done':
        return row
    from unifiedgroupgen.group_constraints.identification import validate_joint_state
    row['validation'] = validate_joint_state(sample.state, exact=exact)
    if requested_group is not None:
        row['requested_group'] = requested_group
        if exact:
            detected = row['validation'].get('number')
            row['validation']['exact_target_group'] = None if detected is None else detected == requested_group
    row['accepted'] = bool(row['validation']['satisfies_requested_operations'] and
                           row['validation']['no_coincident_orbits'] and
                           row['validation']['distance_valid'] and
                           (not exact or row['validation']['exact_target_group'] is True))
    if sample.state.spec.kind not in ('space', 'layer', 'plane'):
        row['status'] = 'generated_state' if row['accepted'] else 'rejected'
        return row
    from types import SimpleNamespace
    import numpy as np
    from unifiedgroupgen.group_constraints.compiler import Descriptor, OrbitSpec
    from unifiedgroupgen.group_constraints.validation import to_structure
    from unifiedgroupgen.common.structures import load_structure, counts_of
    try:
        decoded = sample.state.decoded()
        compiled = SimpleNamespace(descriptor=SimpleNamespace(kind=sample.state.spec.kind),
                    periodic=np.asarray(sample.state.spec.periodic),elements=decoded['atomic_numbers'].detach().cpu().numpy())
        structure = to_structure(compiled,decoded)
        text = structure.to(fmt='cif')
        row.update(cif=text,cif_sha256=hashlib.sha256(text.encode('utf-8')).hexdigest())
        if 'symmetry' not in row:
            descriptor = Descriptor(sample.state.spec.kind, sample.state.spec.number,
                                    tuple(OrbitSpec(w,z) for w,z in zip(sample.state.letters,sample.state.elements)))
            row['descriptor'] = descriptor.to_dict()
        restored = load_structure(row, '.')
        if counts_of(restored) != counts_of(structure):
            raise ValueError('CIF readback changed composition')
        row['status'] = 'generated' if row['accepted'] else 'rejected'
    except (ValueError, FloatingPointError) as error:
        row.update(status='failed',accepted=False,reason='cif_export: '+str(error))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    compile_command = commands.add_parser('compile', help='Compute orbit constraints from operations, without supplied templates')
    compile_command.add_argument('--symmetry', help='JSON with kind and generators or operations')
    compile_command.add_argument('--kind', choices=['space', 'layer', 'plane'])
    compile_command.add_argument('--group', type=int, help='Optional catalog adapter; reads operations only')
    compile_command.add_argument('--output', required=True)
    audit = commands.add_parser('audit', help='Create a new data audit for a joint-model Hydra selector')
    audit.add_argument('--experiment', default='joint_symmetry_joint')
    audit.add_argument('--override', action='append', default=[], help='Matching Hydra definition override, repeatable')
    smoke = commands.add_parser('smoke', help='Bounded real-data training; never rewrites data or old checkpoints')
    smoke.add_argument('--datasets', nargs='+', choices=['mp_20', 'c2db_51'], default=['mp_20', 'c2db_51'])
    smoke.add_argument('--records', type=int, default=2)
    smoke.add_argument('--steps', type=int, default=4)
    smoke.add_argument('--device', default='cpu')
    smoke.add_argument('--operation-compiler', action='store_true', help='Enable computed orbit charts and network constraint layers')
    smoke.add_argument('--output', default='intermediate/checks/joint_symmetry_v1/smoke/model.pt')
    sample = commands.add_parser('sample', help='Sample with a joint checkpoint and specified operators/group')
    sample.add_argument('--checkpoint', required=True)
    sample.add_argument('--kind', choices=['space', 'layer', 'plane'])
    sample.add_argument('--group', type=int)
    sample.add_argument('--symmetry', help='Explicit operator/template JSON for other families, MSGs and SSGs')
    sample.add_argument('--magnetic', action='store_true', help='Requires weights trained on aligned moment labels')
    sample.add_argument('--properties', default='{}', help='JSON conditions matching the checkpoint schema')
    sample.add_argument('--chemical-system', nargs='+', help='Allowed symbols/numbers; any nonempty subset may occur')
    sample.add_argument('--composition', help='JSON exact per-cell counts, e.g. {"Ti":2,"O":4}')
    sample.add_argument('--formula', help='Integer formula; use --formula-units to choose cell size')
    sample.add_argument('--formula-units', type=int, default=1)
    sample.add_argument('--count', type=int, default=1)
    sample.add_argument('--batch-size', type=int, default=4, help='Packed joint_flow_v2 requests per forward')
    sample.add_argument('--trajectory-dir', help='Opt-in joint_flow_v2 JSONL trajectories, one file per selected attempt')
    sample.add_argument('--trajectory-indices', nargs='+', type=int,
                        help='Zero-based global sample indices to record; omit to record all attempts')
    sample.add_argument('--steps', type=int, default=128)
    sample.add_argument('--seed', type=int, default=42)
    sample.add_argument('--device', default='cpu')
    sample.add_argument('--output', required=True)
    sample.add_argument('--exact-group', action='store_true', help='Require maximal group recovery, retaining rejected attempts in the ledger')
    sample.add_argument('--case-id', help='CSP evaluation identifier only; no reference geometry is read')
    assess = commands.add_parser('evaluate-csp', help='Match a completed CSP ledger to held-out references')
    assess.add_argument('--samples', required=True)
    assess.add_argument('--references', required=True)
    assess.add_argument('--output', required=True)
    assess.add_argument('--ks', nargs='+', type=int, default=[1])
    args = parser.parse_args(argv)
    if args.command == 'sample' and args.trajectory_indices is not None:
        if not args.trajectory_dir:
            parser.error('--trajectory-indices requires --trajectory-dir')
        if any(not 0 <= i < args.count for i in args.trajectory_indices):
            parser.error('--trajectory-indices must lie in [0, --count)')
        args.trajectory_indices=sorted(set(args.trajectory_indices))
    torch.set_num_threads(1)
    root = project_root()
    if args.command == 'evaluate-csp':
        from unifiedgroupgen.evaluation.joint_csp import evaluate_csp
        report = evaluate_csp(_head(args.samples, 10**9), _head(args.references, 10**9), tuple(args.ks))
        output = artifact_path(args.output)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(report['summary']))
        return
    if args.command == 'compile':
        from unifiedgroupgen.group_constraints.operations import operation_spec, compile_request
        if args.symmetry:
            data = json.loads(resolve_path(args.symmetry).read_text(encoding='utf-8'))
            spec = compile_request(data)
        elif args.kind is not None and args.group is not None:
            spec = operation_spec(args.kind, args.group)
        else:
            parser.error('Supply --symmetry or --kind/--group')
        output = artifact_path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(spec.to_dict(), stream, indent=2, allow_nan=False)
        print(json.dumps(dict(output=str(output), operations=len(spec.operations),
                              templates=len(spec.templates), source='computed_fixed_loci')))
        return
    if args.command == 'audit':
        from unifiedgroupgen.tests.preflight import audit_files
        config = training_definition(compose_experiment(args.experiment, root, args.override))
        paths = {split: [resolve_path(p, root) for p in config[split+'_files']] for split in ('train', 'val', 'test')}
        report = audit_files(paths, config, artifact_path(config['data_audit'], root))
        print(json.dumps({'passed': report['passed'], 'audit': config['data_audit'],
                          'errors': report['errors']}, ensure_ascii=False))
        if not report['passed']:
            raise SystemExit(1)
        return
    if args.command == 'smoke':
        if args.records < 1 or args.steps < 1:
            parser.error('records and steps must be positive')
        from unifiedgroupgen.pl_modules.joint_training import fit_joint
        from unifiedgroupgen.tests.source_integrity import encoding_digest, PROTOCOL
        from unifiedgroupgen.common.source import file_hash
        train, val, vocabulary, hashes = [], [], set(), {}
        for dataset in args.datasets:
            base = root/'data/dataset/full_verified_v1'/dataset
            train.extend(_head(base/'train.jsonl', args.records))
            val.extend(_head(base/'val.jsonl', args.records))
            hashes[dataset] = {split: file_hash(base/(split+'.jsonl')) for split in ('train','val')}
            with (base/'train.jsonl').open(encoding='utf-8') as stream:
                for line in stream:
                    if line.strip():
                        vocabulary.update(w['atomic_number'] for w in json.loads(line)['descriptor']['orbits'])
        for record in train+val:
            certification = record.get('provenance', {}).get('source_integrity', {})
            if (certification.get('protocol') != PROTOCOL or not certification.get('composition_conserved')
                    or not certification.get('geometry_equivalent') or certification.get('encoding_sha256') != encoding_digest(record)):
                raise ValueError('Selected smoke record lacks valid source-integrity certification')
        config = dict(model={'family': 'joint_symmetry_v1', 'hidden': 32, 'layers': 2, 'heads': 4},
                      allowed_elements=sorted(vocabulary),
                      properties={}, max_atoms=192, max_orbits=52, event_budget=256,
                      birth_scale=.2, learning_rate=1e-3, cond_dropout=0.,
                      validation_scope='bounded_smoke_not_full_data_audit')
        if args.operation_compiler:
            config['model'].update(operation_compiler=True, constraint_layers=True)
        # Held-out unseen species remain visible failures, not silently filtered.
        unseen = {w['atomic_number'] for r in val for w in r['descriptor']['orbits']}-set(config['allowed_elements'])
        if unseen:
            raise ValueError(f'Validation smoke subset has species outside selected TRAIN vocabulary: {sorted(unseen)}; increase --records')
        pl = __import__('pytorch_lightning')
        pl.seed_everything(42, workers=True)
        trainer = fit_joint(config, train, val, artifact_path(args.output), args.device,
                            epochs=max(1, args.steps), runtime={'batch_size': 2, 'edge_budget': 200000},
                            split_hashes=hashes,max_steps=args.steps)
        print(json.dumps({'output': str(artifact_path(args.output)), 'global_step': trainer.global_step,
                          'datasets': args.datasets, 'scope': 'bounded real-data smoke'}))
        return
    from unifiedgroupgen.pl_modules.joint_training import load_joint_checkpoint
    from unifiedgroupgen.pl_modules.factory import build_joint_model
    from unifiedgroupgen.group_constraints.unified import catalog_spec, SymmetrySpec
    from unifiedgroupgen.generation.joint import sample_joint
    checkpoint = load_joint_checkpoint(resolve_path(args.checkpoint), args.device)
    model = build_joint_model(checkpoint['config'], args.device)
    model.load_state_dict(checkpoint['model'], strict=True)
    if args.trajectory_dir and getattr(model,'flow_version',None) != 'joint_flow_v2':
        parser.error('--trajectory-dir requires a joint_flow_v2 checkpoint; legacy sampling is unchanged')
    from unifiedgroupgen.group_constraints.chemistry import ChemicalControl
    from unifiedgroupgen.group_constraints.group_prior import sample_group, NoFeasibleGroup
    if args.composition and args.formula:
        parser.error('Choose exact --composition or --formula, not both')
    if args.formula_units < 1 or (not args.formula and args.formula_units != 1):
        parser.error('--formula-units is positive and only applies to --formula')
    counts = json.loads(args.composition) if args.composition else None
    if args.formula:
        from pymatgen.core import Composition
        counts = {z: n*args.formula_units for z, n in Composition(args.formula).get_el_amt_dict().items()}
    chemistry = ChemicalControl.from_inputs(args.chemical_system, counts)
    if chemistry.mode != 'free' and not checkpoint['config']['model'].get('task_controls'):
        parser.error('This checkpoint does not have trained hard-task controls')
    chemistry.domain(model.allowed_elements)
    task = checkpoint['config'].get('generation_task')
    if task == 'csp' and not chemistry.counts:
        parser.error('A CSP checkpoint requires --composition or --formula')
    if chemistry.counts and task != 'csp':
        parser.error('Exact inventory generation requires an independently trained CSP checkpoint')
    provided_spec = None
    if args.symmetry:
        symmetry_data = json.loads(resolve_path(args.symmetry).read_text(encoding='utf-8'))
        if checkpoint['config']['model'].get('operation_compiler', False):
            from unifiedgroupgen.group_constraints.operations import compile_request
            provided_spec = compile_request(symmetry_data)
        else:
            provided_spec = SymmetrySpec.from_dict(symmetry_data)
    elif args.group is not None:
        if args.kind is None:
            parser.error('A numbered --group needs --kind')
        provided_spec = catalog_spec(args.kind, args.group)
    elif 'group_prior' not in checkpoint['config']:
        parser.error('This checkpoint needs --kind/--group or --symmetry; it has no TRAIN group prior')
    if args.magnetic and not checkpoint['config'].get('magnetic_training', False):
        raise ValueError('This checkpoint has no certified magnetic training; missing labels are not zero labels')
    if args.count < 1:
        parser.error('count must be positive')
    properties = json.loads(args.properties)
    if not isinstance(properties,dict) or set(properties)-set(checkpoint['config'].get('properties',{})):
        raise ValueError('Property requests must match the checkpoint conditioning schema')
    if task == 'property' and not properties:
        parser.error('A property checkpoint needs at least one target property')
    import math
    if any(value is None or not math.isfinite(float(value)) for value in properties.values()):
        parser.error('Requested target properties must be finite numbers')
    output = artifact_path(args.output)
    if getattr(model, 'flow_version', None) == 'joint_flow_v2':
        if args.batch_size < 1 or args.count < 1: parser.error('count/batch-size must be positive')
        from unifiedgroupgen.joint_flow.cli import generate
        print(json.dumps(generate(args,checkpoint,model,provided_spec,chemistry,properties,output)))
        return
    outcomes = {}
    with output.open('x', encoding='utf-8') as stream:
        for index in range(args.count):
            generator = torch.Generator(device=args.device).manual_seed(args.seed+index)
            try:
                spec = provided_spec or sample_group(checkpoint['config']['group_prior'], generator,
                    kind=args.kind, chemistry=chemistry, budget=checkpoint['config'].get('event_budget', 256),
                    max_atoms=model.max_atoms, max_orbits=model.max_orbits)
                requested_number = args.group if spec.number == 0 else spec.number
                if checkpoint['config']['model'].get('operation_compiler', False):
                    from unifiedgroupgen.group_constraints.operations import compile_operations
                    spec = compile_operations(spec.operations, spec.kind)
                if args.exact_group and spec.number == 0 and requested_number is None:
                    parser.error('Exact numbered-group comparison needs a numbered group')
                result = sample_joint(model, spec, steps=args.steps, generator=generator,
                         budget=checkpoint['config'].get('event_budget', 256),
                         birth_scale=checkpoint['config'].get('birth_scale', .2), magnetic=args.magnetic,
                         properties=properties, chemistry=chemistry)
                row = evaluation_record(result,index,args.seed+index,args.exact_group,properties,
                                        requested_group=requested_number)
            except NoFeasibleGroup as error:
                row = dict(index=index, seed=args.seed+index, status='failed', accepted=False,
                           generation_status='failed', reason=str(error), chemical_control=chemistry.to_dict())
            row['generation_task'] = task or 'legacy_joint'
            row['dataset'] = checkpoint['config'].get('dataset', '')
            row['case_id'] = args.case_id
            row['candidate'] = index
            outcomes[row['status']] = outcomes.get(row['status'],0)+1
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    print(json.dumps({'samples': str(output), 'attempts': args.count, 'outcomes': outcomes}))


if __name__ == '__main__':
    main()
