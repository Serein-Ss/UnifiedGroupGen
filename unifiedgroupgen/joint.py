"""Opt-in command adapter: audited training, bounded smoke checks and joint sampling."""
import argparse
import json
from pathlib import Path
import torch
from unifiedgroupgen.common.progress import track
from unifiedgroupgen.common.configuration import (
    project_root, resolve_path, artifact_path,
)

from unifiedgroupgen.experiment_config import compose_experiment, training_definition


def sample_to_dict(sample, requested_group=None):
    from unifiedgroupgen.group_constraints.wyckoff import orbit_labels
    state = sample.state
    result = dict(status=state.status, reason=sample.reason, time=sample.time,
                  orbit_labels=orbit_labels(state.spec, state.letters, requested_group),
                  events=sample.events, integration_steps=sample.integration_steps,
                  epsilon=sample.epsilon, kind=state.spec.kind, group=state.spec.number,
                  letters=list(state.letters), orbit_elements=list(state.elements),
                  moment_field_enabled=state.moments is not None,
                  metric_state=state.metric.detach().cpu().tolist() if torch.isfinite(state.metric).all() else None,
                  coordinate_state=state.coordinates.detach().cpu().tolist() if torch.isfinite(state.coordinates).all() else None)
    from unifiedgroupgen.group_constraints.unified import catalog_spec
    builtin = (state.spec.kind in ('space','layer','plane','rod') and state.spec.number > 0
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


def evaluation_record(sample, index, seed, exact=False, properties=None, requested_group=None, *,
                      label_group=None, verify_cif_operations=False):
    """Retain every attempt and export ordinary structures in the existing CIF ledger contract."""
    import hashlib
    row = sample_to_dict(sample, requested_group if label_group is None else label_group)
    row.update(index=index, case_id=None, candidate=index, seed=seed,
               generation_status=row['status'], status='failed', accepted=False,
               requested_group=sample.state.spec.number, properties_requested=properties or {},
               precision='fp32', sampling_backend='joint_atom_graph', solver='event_heun')
    if requested_group is not None:
        row['requested_group'] = requested_group
    state = sample.state
    verify_cif_operations = verify_cif_operations and state.spec.kind in ('space', 'layer')
    if state.status != 'done':
        if verify_cif_operations:
            row['validation'] = {'structure_valid': False}
        return row
    from unifiedgroupgen.group_constraints.identification import validate_joint_state
    row['validation'] = validate_joint_state(state, exact=exact)
    if requested_group is not None:
        row['requested_group'] = requested_group
        if exact:
            detected = row['validation'].get('number')
            row['validation']['exact_target_group'] = None if detected is None else detected == requested_group
    row['accepted'] = bool(row['validation']['satisfies_requested_operations'] and
                           row['validation']['no_coincident_orbits'] and
                           row['validation']['distance_valid'] and
                           (not exact or row['validation']['exact_target_group'] is True))
    if sample.state.spec.kind not in ('space', 'layer', 'plane', 'rod'):
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
        if verify_cif_operations:
            from unifiedgroupgen.evaluation.common import exported_operation_check
            validation = row['validation']
            operations = [dict(R=op.rotation, t=op.translation) for op in state.spec.operations]
            origin = float(decoded['coordinates'][:, 2].double().mean()) if state.spec.kind == 'layer' else 0.
            if (not restored.is_ordered or not np.isfinite(restored.frac_coords).all()
                    or not np.isfinite(restored.lattice.matrix).all()
                    or not np.isfinite(restored.volume) or restored.volume <= 1e-10):
                raise ValueError('Invalid lattice or sites after CIF readback')
            validation['structure_valid'] = bool(validation['no_coincident_orbits'] and validation['distance_valid'])
            check = exported_operation_check(restored, state.spec.kind, row['requested_group'], operations,
                                             layer_origin=origin)
            validation['cif_operation_check'] = check
            reasons = []
            if not validation['satisfies_requested_operations'] or not check['target_operation_containment']:
                reasons.append('operation_violation')
            if not validation['structure_valid']:
                reasons.append('structure_invalid')
            if exact and validation['exact_target_group'] is not True:
                reasons.append('exact_group_mismatch')
            row['rejection_reasons'] = reasons
            row['accepted'] = not reasons
        row['status'] = 'generated' if row['accepted'] else 'rejected'
    except (ValueError, FloatingPointError) as error:
        if verify_cif_operations:
            row['validation']['structure_valid'] = False
        row.update(status='failed',accepted=False,reason='cif_export: '+str(error))
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    compile_command = commands.add_parser('compile', help='Compute orbit constraints from operations, without supplied templates')
    compile_command.add_argument('--symmetry', help='JSON with kind and generators or operations')
    compile_command.add_argument('--kind', choices=['space', 'layer', 'plane', 'rod'])
    compile_command.add_argument('--group', type=int, help='Optional catalog adapter; reads operations only')
    compile_command.add_argument('--output', required=True)
    audit = commands.add_parser('audit', help='Create a new data audit for a joint-model Hydra selector')
    audit.add_argument('--experiment', default='full_mp20_eqv2_dng')
    audit.add_argument('--output', help='Audit output JSON path')
    audit.add_argument('--override', action='append', default=[], help='Matching Hydra definition override, repeatable')
    sample = commands.add_parser('sample', help='Sample with a joint checkpoint and specified operators/group')
    sample.add_argument('--checkpoint', required=True)
    sample.add_argument('--kind', choices=['space', 'layer', 'plane', 'rod'])
    sample.add_argument('--group', type=int)
    sample.add_argument('--symmetry', help='Explicit operator/template JSON for other families, MSGs and SSGs')
    sample.add_argument('--magnetic', action='store_true', help='Requires weights trained on aligned moment labels')
    sample.add_argument('--properties', default='{}', help='JSON conditions matching the checkpoint schema')
    sample.add_argument('--guidance-scale', type=float,
                        help='Property CFG: 0=null, 1=conditional, >1=extrapolate; default 1.5 for CFG-ready property requests')
    sample.add_argument('--chemical-system', nargs='+', help='Allowed symbols/numbers; any nonempty subset may occur')
    sample.add_argument('--composition', help='JSON exact per-cell counts, e.g. {"Ti":2,"O":4}')
    sample.add_argument('--formula', help='Integer formula; use --formula-units to choose cell size')
    sample.add_argument('--formula-units', type=int, default=1)
    sample.add_argument('--count', type=int, help='Attempts for a single request; default 1')
    sample.add_argument('--batch-size', type=int, default=4, help='Packed joint_flow_v2 requests per forward')
    sample.add_argument('--trajectory-dir', help='Opt-in joint_flow_v2 JSONL trajectories, one file per selected attempt')
    sample.add_argument('--trajectory-indices', nargs='+', type=int,
                        help='Zero-based global sample indices to record; omit to record all attempts')
    sample.add_argument('--repair-short-contacts', action=argparse.BooleanOptionalAction,
                        help='Try bounded terminal contact repair in the existing Wyckoff chart')
    sample.add_argument('--steps', type=int, default=128)
    sample.add_argument('--seed', type=int, default=42)
    sample.add_argument('--device', default='cpu')
    sample.add_argument('--output', help='Explicit ledger path; v2 defaults beside the checkpoint')
    sample.add_argument('--task-name', help='v2 generate/<task-name>; default: <kind>_<task>_seed<seed>')
    sample.add_argument('--exact-group', action='store_true', help='Require maximal group recovery, retaining rejected attempts in the ledger')
    sample.add_argument('--case-id', help='CSP evaluation identifier only; no reference geometry is read')
    sample.add_argument('--case-dataset', help='Source dataset of a single test case, separate from the training dataset')
    sample.add_argument('--case-manifest', help='Geometry-free CSP case manifest produced by prepare-csp')
    from unifiedgroupgen.group_constraints.group_prior import GROUP_POLICIES
    sample.add_argument('--csp-group-policy', choices=GROUP_POLICIES,
                        help='Override free group selection for a CSP checkpoint')
    prepare = commands.add_parser('prepare-csp', help='Extract CSP case identities and per-cell inventories')
    prepare.add_argument('--references', required=True, action='append')
    prepare.add_argument('--candidates', required=True, type=int, help='Planned candidates per case')
    prepare.add_argument('--output', required=True)
    assess = commands.add_parser('evaluate-csp', help='Match a CSP ledger to held-out references')
    assess.add_argument('--samples', required=True)
    assess.add_argument('--references', required=True, action='append')
    assess.add_argument('--case-manifest', help='Use the same case and candidate denominator as generation')
    assess.add_argument('--output', required=True)
    assess.add_argument('--ks', nargs='+', type=int, default=[1])
    args = parser.parse_args(argv)
    if args.command == 'sample':
        if args.case_dataset is not None and (not args.case_id or not args.case_dataset.strip()):
            parser.error('--case-dataset requires --case-id and a nonempty source dataset')
        if args.case_manifest:
            if (any(value is not None for value in (args.kind, args.group, args.symmetry, args.case_id,
                    args.case_dataset, args.composition, args.formula, args.chemical_system, args.count))
                    or args.formula_units != 1 or args.magnetic):
                parser.error('--case-manifest supplies the kind, chemistry, count and case identity for each request')
            from unifiedgroupgen.common.csp_cases import read_csp_manifest
            args.case_manifest = read_csp_manifest(resolve_path(args.case_manifest))
            args.count = len(args.case_manifest.cases) * args.case_manifest.candidates_per_case
        elif args.count is None:
            args.count = 1
    if args.command == 'sample' and args.trajectory_indices is not None:
        if not args.trajectory_dir:
            parser.error('--trajectory-indices requires --trajectory-dir')
        if any(not 0 <= i < args.count for i in args.trajectory_indices):
            parser.error('--trajectory-indices must lie in [0, --count)')
        args.trajectory_indices=sorted(set(args.trajectory_indices))
    torch.set_num_threads(1)
    root = project_root()
    if args.command in ('prepare-csp', 'evaluate-csp'):
        from unifiedgroupgen.common.structures import iter_records
        from unifiedgroupgen.common.csp_cases import read_csp_manifest
        from unifiedgroupgen.evaluation.joint_csp import evaluate_csp, prepare_csp_manifest
        references = [row for path in args.references for row in iter_records(path)]
        if args.command == 'prepare-csp':
            manifest = prepare_csp_manifest(references, args.candidates)
            report = manifest.to_dict()
            status = dict(cases=len(manifest.cases), candidates_per_case=manifest.candidates_per_case,
                          case_manifest_sha256=manifest.digest)
        else:
            manifest = read_csp_manifest(resolve_path(args.case_manifest)) if args.case_manifest else None
            report = evaluate_csp(iter_records(args.samples), references, tuple(args.ks), manifest)
            status = report['summary']
        output = artifact_path(args.output)
        with output.open('x', encoding='utf-8') as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(status, ensure_ascii=False))
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
        from unifiedgroupgen.pl_data.validation import audit_files
        config = training_definition(compose_experiment(args.experiment, overrides=args.override))
        from unifiedgroupgen.pl_data.cache import cache_paths
        paths = {split: cache_paths(config, split, root) for split in ('train', 'val', 'test')}
        output = args.output or config.get('data_audit') or f'intermediate/data_audits/{args.experiment}.json'
        report = audit_files(paths, config, artifact_path(output, root))
        print(json.dumps({'passed': report['passed'], 'audit': output,
                          'errors': report['errors']}, ensure_ascii=False))
        if not report['passed']:
            raise SystemExit(1)
        return
    from unifiedgroupgen.pl_modules.unified.module import load_joint_checkpoint
    from unifiedgroupgen.pl_modules.factory import build_joint_model
    from unifiedgroupgen.group_constraints.unified import catalog_spec, SymmetrySpec
    checkpoint = load_joint_checkpoint(resolve_path(args.checkpoint), args.device)
    model = build_joint_model(checkpoint['config'], args.device)
    model.load_state_dict(checkpoint['model'], strict=True)
    from unifiedgroupgen.group_constraints.chemistry import ChemicalControl
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
    from unifiedgroupgen.conditional.tasks import generation_mode
    task = generation_mode(checkpoint['config'])
    if args.case_manifest and (task != 'csp' or not checkpoint['config']['model'].get('task_controls')):
        parser.error('--case-manifest requires a CSP checkpoint with hard-task controls')
    if task == 'csp' and not (chemistry.counts or args.case_manifest):
        parser.error('A CSP checkpoint requires --composition, --formula or --case-manifest')
    if (args.case_id and task == 'csp' and checkpoint['config'].get('dataset') == 'mp20_c2db51'
            and not args.case_dataset):
        parser.error('A single case from a mixed CSP checkpoint requires --case-dataset')
    if chemistry.counts and task != 'csp':
        parser.error('Exact inventory generation requires an independently trained CSP checkpoint')
    if args.csp_group_policy and task != 'csp':
        parser.error('--csp-group-policy requires a CSP checkpoint')
    if (args.csp_group_policy in ('composition_ranked', 'composition_neighbors')
            and not (args.symmetry or args.group is not None)
            and not checkpoint['config'].get('group_prior', {}).get('composition_counts')):
        parser.error('This checkpoint has no TRAIN composition counts. Select --csp-group-policy prior.')
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
    schema = checkpoint['config'].get('properties', {})
    if schema and not properties and not checkpoint['config'].get('cond_dropout', 0):
        parser.error('This checkpoint was trained without property dropout; provide a target via --properties')
    import math
    if any(value is None or not math.isfinite(float(value)) for value in properties.values()):
        parser.error('Requested target properties must be finite numbers')
    from unifiedgroupgen.conditional.guidance import validate_guidance
    dropout = checkpoint['config'].get('cond_dropout', 0.)
    if args.guidance_scale is None:
        args.guidance_scale = 1.5 if properties and 0 < dropout < 1 else 1.
    try:
        validate_guidance(args.guidance_scale, bool(properties), dropout)
    except ValueError as error:
        parser.error(str(error))
    if not args.output:
        from unifiedgroupgen.common.configuration import model_result_directory
        name = args.task_name or f'{args.kind or "mixed"}_{task or "dng"}_seed{args.seed}'
        args.output = model_result_directory(args.checkpoint,'generate',name)/'samples.jsonl'
    output = artifact_path(args.output)
    if args.batch_size < 1 or args.count < 1:
        parser.error('count/batch-size must be positive')
    from unifiedgroupgen.generation.unified.runner import generate
    generate(args, checkpoint, model, provided_spec, chemistry, properties, output)
    print(f'Output: {output}')


if __name__ == '__main__':
    main()
