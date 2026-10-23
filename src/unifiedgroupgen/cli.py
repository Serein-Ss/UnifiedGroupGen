"""Small research CLI: preprocessing, joint training, inverse sampling, and validation."""

import argparse
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from .conditioning import fit_property_scalers, attach_dataset_domains
from .constraints import AffineObservation
from .data import prepare_csv, read_jsonl, write_jsonl
from .evaluation import to_structure, validate, summarize_attempts
from .flow import OrbitFlow, flow_loss, sample_flow
from .proposal import DescriptorTransformer
from .symmetry import Descriptor, compile_descriptor


ROOT = Path(__file__).resolve().parents[2]


def root_path(path):
    candidate = Path(path)
    return candidate.resolve() if candidate.is_absolute() else (ROOT/candidate).resolve()


def artifact_path(path):
    candidate = root_path(path)
    if not candidate.is_relative_to(ROOT):
        raise ValueError("Generated artifacts must be inside UnifiedGroupGen")
    candidate.parent.mkdir(parents=True, exist_ok=True)
    return candidate


def composition_of(compiled):
    values, counts = np.unique(compiled.elements, return_counts=True)
    return {int(z): int(n) for z, n in zip(values, counts)}


def fit_training_config(config, records):
    """Use the same TRAIN-only element domain and property scalers in training and timing."""
    config = dict(config)
    if config.get('element_policy', 'training') == 'training' and not config.get('allowed_elements'):
        config['allowed_elements'] = sorted({o['atomic_number'] for r in records for o in r['descriptor']['orbits']})
    config['properties'] = fit_property_scalers(records, config.get('properties', {}))
    if config.get('dataset_domains'):
        config['domain_label_support'] = {}
        for dataset, domain in config['dataset_domains'].items():
            rows = [r for r in records if r['dataset'] == dataset]
            config['domain_label_support'][str(domain)] = {
                'kinds': sorted({r['descriptor']['kind'] for r in rows}),
                'property_counts': {name: sum(r['properties'].get(name) is not None for r in rows)
                                    for name, specification in config['properties'].items() if not specification.get('context')}}
    return config


def build_models(config, device):
    from .symmetry_preflight import check_layer_catalog
    check_layer_catalog()
    groups = config.get("groups")
    if groups is None:
        limits = {"space": 230, "layer": 80, "plane": 17}
        groups = [(kind, number) for kind in config.get("kinds", [config["kind"]]) for number in range(1, limits[kind]+1)]
    properties = config.get("properties", {})
    architecture = config.get("model", {})
    proposal = DescriptorTransformer(groups, properties, hidden=architecture.get("proposal_hidden", architecture.get("hidden", 128)),
                                     layers=architecture.get("proposal_layers", 3), heads=architecture.get("heads", 4),
                                     max_atoms=config.get("max_atoms", 192), max_orbits=config.get("max_orbits", 32),
                                     cond_dropout=config.get("cond_dropout", 0.1), allowed_elements=config.get('allowed_elements')).to(device)
    flow = OrbitFlow(properties, hidden=architecture.get("flow_hidden", architecture.get("hidden", 128)), layers=architecture.get("flow_layers", 4),
                     frequencies=architecture.get("frequencies", 8), cond_dropout=config.get("cond_dropout", 0.1),
                     checkpoint_layers=architecture.get('checkpoint_layers', False),
                     edge_chunk_size=architecture.get('edge_chunk_size', 8192)).to(device)
    proposal.condition.subset_training = config.get('property_subset_training', False)
    flow.property_encoder.subset_training = config.get('property_subset_training', False)
    return proposal, flow


def _record_loss(proposal, flow, record, observation_probability=0, composition_probability=0.5):
    descriptor = Descriptor.from_dict(record["descriptor"])
    compiled = compile_descriptor(descriptor)
    endpoint = next(flow.parameters()).new_tensor(record["state"])
    if len(endpoint) != compiled.dof:
        raise ValueError(f"State dimension mismatch in {record['id']}")
    composition = composition_of(compiled) if random.random() < composition_probability else None
    observation = None
    if random.random() < observation_probability:
        indices = (list(range(compiled.lattice_dof)) if random.random() < 0.5
                   else random.sample(range(compiled.dof), max(1, compiled.dof//3)))
        observation = AffineObservation.fixed(compiled.dof, indices, endpoint[indices].detach().cpu())
    discrete = proposal.loss(descriptor, record["properties"], composition)
    continuous = flow_loss(flow, compiled, endpoint, record["properties"], observation, composition)
    return discrete, continuous


def train(args):
    from .preflight import check_training_gate
    from .training import run_training
    config = yaml.safe_load(root_path(args.config).read_text(encoding="utf-8"))
    prepared = config.get('prepared_root', f"artifacts/{config['dataset']}")
    train_paths = [root_path(p) for p in (args.train or config.get('train_files') or [f"{prepared}/train.jsonl"])]
    val_paths = [root_path(p) for p in (args.val or config.get('val_files') or [f"{prepared}/val.jsonl"])]
    check_training_gate(config, train_paths, val_paths, ROOT)
    records = [r for path in train_paths for r in read_jsonl(path)]
    validation = [r for path in val_paths for r in read_jsonl(path)]
    records = attach_dataset_domains(records, config)
    validation = attach_dataset_domains(validation, config)
    if not records or not validation:
        raise ValueError("Training and validation records must both be nonempty")
    train_keys = {(r['dataset'], r['id']) for r in records}
    if train_keys & {(r['dataset'], r['id']) for r in validation}:
        raise ValueError("Training/validation identifier overlap")
    config = fit_training_config(config, records)
    proposal, flow = build_models(config, args.device)
    output = artifact_path(args.output or f"artifacts/{config['dataset']}/model.pt")
    if args.resume:
        args.resume = root_path(args.resume)
    run_training(args, config, records, validation, proposal, flow, train_paths, val_paths, output)


def sample(args):
    from .chemical_controls import ChemicalRules, integer_composition, feasible_formula_counts
    from .structure_controls import Scaffold
    checkpoint = torch.load(root_path(args.checkpoint), map_location=args.device, weights_only=False)
    config = checkpoint['config']
    proposal, flow = build_models(config, args.device)
    proposal.load_state_dict(checkpoint['proposal'])
    flow.load_state_dict(checkpoint['flow'])
    proposal.eval()
    flow.eval()
    properties = json.loads(args.properties)
    unknown = set(properties)-set(config.get('properties', {}))
    if unknown:
        raise ValueError(f"Checkpoint has no property encoder for {sorted(unknown)}")
    missing = [name for name, specification in config.get('properties', {}).items()
               if specification.get('context') and properties.get(name) is None]
    if missing:
        raise ValueError(f'Sampling requires context values for {missing}')
    fixed = Descriptor.from_dict(json.loads(root_path(args.descriptor).read_text(encoding='utf-8'))) if args.descriptor else None
    if config.get('domain_label_support'):
        domain = properties['source_domain']
        if int(domain) != domain or str(int(domain)) not in config['domain_label_support']:
            raise ValueError(f'No trained calculation domain: {domain}')
        support = config['domain_label_support'][str(int(domain))]
        if (fixed.kind if fixed is not None else args.kind) not in support['kinds']:
            raise ValueError('Requested material kind was not trained in this calculation domain')
        unavailable = [name for name in properties if name != 'source_domain'
                       and properties[name] is not None and not support['property_counts'].get(name)]
        if unavailable:
            raise ValueError(f'No training property labels in this calculation domain for {unavailable}')
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
            records.append({'index': index, 'descriptor': descriptor.to_dict(), 'state': generated['state'].cpu().tolist(),
                            'properties_requested': properties, 'validation': report, 'status': 'generated',
                            'chemical_request': {'allowed': sorted(chemical.allowed) if chemical.allowed else None,
                                                 'required': sorted(chemical.required), 'composition': counts},
                            'scaffold_requested': bool(scaffold),
                            'generator': 'legal_prior' if args.prior else 'learned_flow'})
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


def audit(args):
    records = read_jsonl(root_path(args.records))
    reports = []
    for record in records:
        compiled = compile_descriptor(Descriptor.from_dict(record['descriptor']))
        generated = compiled.expand(torch.tensor(record['state'], dtype=torch.float64))
        reports.append({'id': record['id'], **validate(compiled, generated, isolate_identifier=True)})
    write_jsonl(artifact_path(args.output), reports)
    print(json.dumps({'records': len(reports), 'distance_valid': sum(r['distance_valid'] for r in reports), 'output': args.output}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare = subparsers.add_parser('prepare')
    prepare.add_argument('--config', required=True)
    prepare.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    prepare.add_argument('--limit', type=int)
    prepare.add_argument('--output')
    prepare.add_argument('--workers', type=int, default=0)
    prepare.add_argument('--resume', action='store_true')
    training = subparsers.add_parser('train')
    training.add_argument('--config', required=True)
    training.add_argument('--train', nargs='+')
    training.add_argument('--val', nargs='+')
    training.add_argument('--device', default='cpu')
    training.add_argument('--epochs', type=int)
    training.add_argument('--batch-size', type=int)
    training.add_argument('--output')
    training.add_argument('--resume')
    training.add_argument('--checkpoint', action=argparse.BooleanOptionalAction, default=None,
                          help='Override activation checkpointing without changing model weights or the data configuration')
    training.add_argument('--edge-chunk-size', type=int)
    training.add_argument('--validation-interval', type=int, help='Validate every N epochs (default: 5); also validate the final epoch')
    sampling = subparsers.add_parser('sample')
    sampling.add_argument('--checkpoint', required=True)
    sampling.add_argument('--device', default='cpu')
    sampling.add_argument('--kind', choices=['space', 'layer', 'plane'], default='space')
    sampling.add_argument('--properties', default='{}')
    sampling.add_argument('--composition')
    sampling.add_argument('--formula', help='Reduced empirical formula; choose feasible formula-unit counts')
    sampling.add_argument('--formula-units', nargs='+', type=int, default=[1, 2, 3, 4])
    sampling.add_argument('--chemical-system', help='Exact element set, e.g. Li-Fe-O; all must occur')
    sampling.add_argument('--allowed-elements', help='Only these elements may occur')
    sampling.add_argument('--required-elements', help='All these elements must occur; extras allowed')
    sampling.add_argument('--group', type=int, help='Fix group token while generating remaining orbits')
    sampling.add_argument('--scaffold', help='Fixed-cell partial CIF in the supplied group setting')
    sampling.add_argument('--atom-map', help='JSON observed-CIF index to full-descriptor atom index')
    sampling.add_argument('--scaffold-tolerance', type=float, default=1e-4)
    sampling.add_argument('--descriptor')
    sampling.add_argument('--observations')
    sampling.add_argument('--prior', action='store_true', help='Legal random chart prior; bypass the trained continuous flow')
    sampling.add_argument('--guidance', type=float, default=1.0)
    sampling.add_argument('--temperature', type=float, default=1.0)
    sampling.add_argument('--steps', type=int, default=64)
    sampling.add_argument('--count', type=int, default=10)
    sampling.add_argument('--output', default='artifacts/samples')
    checking = subparsers.add_parser('audit')
    checking.add_argument('--records', required=True)
    checking.add_argument('--output', default='artifacts/audit.jsonl')
    preflight = subparsers.add_parser('preflight')
    preflight.add_argument('--config', required=True)
    preflight.add_argument('--train', nargs='+')
    preflight.add_argument('--val', nargs='+')
    preflight.add_argument('--test', nargs='+')
    preflight.add_argument('--output', required=True)
    parameters = subparsers.add_parser('parameters')
    parameters.add_argument('--config', required=True)
    summary = subparsers.add_parser('summarize')
    summary.add_argument('--samples', required=True)
    summary.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command in ('prepare', 'audit', 'preflight'):
        from .symmetry_preflight import check_layer_catalog
        check_layer_catalog()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.command == 'prepare':
        config = yaml.safe_load(root_path(args.config).read_text(encoding='utf-8'))
        source = root_path(config['data_root'])/f'{args.split}.csv'
        prepared = config.get('prepared_root', f"artifacts/{config['dataset']}")
        output = artifact_path(args.output or f"{prepared}/{args.split}.jsonl")
        print(json.dumps(prepare_csv(source, output, config, args.limit, args.workers, args.resume)))
    elif args.command == 'train':
        train(args)
    elif args.command == 'sample':
        sample(args)
    elif args.command == 'audit':
        audit(args)
    elif args.command == 'preflight':
        from .preflight import audit_files
        config = yaml.safe_load(root_path(args.config).read_text(encoding='utf-8'))
        prepared = config.get('prepared_root', f"artifacts/{config['dataset']}")
        paths = {'train': [root_path(p) for p in (args.train or config.get('train_files') or [f"{prepared}/train.jsonl"])],
                 'val': [root_path(p) for p in (args.val or config.get('val_files') or [f"{prepared}/val.jsonl"])]}
        if args.test or config.get('test_files'):
            paths['test'] = [root_path(p) for p in (args.test or config['test_files'])]
        report = audit_files(paths, config, artifact_path(args.output))
        print(json.dumps({'passed': report['passed'], 'errors': len(report['errors']), 'output': args.output}))
        if not report['passed']:
            raise SystemExit(2)
    elif args.command == 'summarize':
        report = summarize_attempts(read_jsonl(root_path(args.samples)))
        artifact_path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report))
    else:
        config = yaml.safe_load(root_path(args.config).read_text(encoding='utf-8'))
        proposal, flow = build_models(config, 'cpu')
        print(json.dumps({'proposal': sum(p.numel() for p in proposal.parameters()),
                          'flow': sum(p.numel() for p in flow.parameters()),
                          'total': sum(p.numel() for model in (proposal, flow) for p in model.parameters())}))


if __name__ == '__main__':
    main()
