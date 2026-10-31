"""Resolve user conditions into one explicit, reusable hard-constraint request."""
from dataclasses import dataclass
import json
import numpy as np
from unifiedgroupgen.conditional.chemistry import ChemicalRules, integer_composition, feasible_formula_counts
from unifiedgroupgen.conditional.structure import Scaffold
from unifiedgroupgen.common.structures import composition_of
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor


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
