"""Training-data gates; these certify inputs, not generated-material stability."""

import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
import torch
import yaml
from .data import read_jsonl
from .symmetry import Descriptor, compile_descriptor
from .training import file_hash
from .conditioning import attach_dataset_domains


ROOT = Path(__file__).resolve().parents[2]


def check_recovered_source(config, root=ROOT):
    """A partial recovered CSV cannot be promoted to a full dataset by changing a flag."""
    if not config.get('recovery_report'):
        return
    path = root/config['recovery_report']
    if not path.exists():
        raise ValueError(f'Full source recovery report is missing: {path}')
    report = json.loads(path.read_text(encoding='utf-8'))
    if not report.get('passed'):
        raise ValueError('Full source recovery has not passed; partial recovered data cannot train formally')
    for database, digest in report.get('original_databases', {}).items():
        if file_hash(database) != digest:
            raise ValueError('Original ASE database differs from the recovery report')
    for split in ('train', 'val', 'test'):
        recovered = report['splits'][split]
        if (not recovered['full_source'] or recovered['accepted'] != recovered['source_rows']
                or recovered['input'] != recovered['source_rows'] or recovered['rejected']):
            raise ValueError(f'Incomplete source recovery for {split}')
        source = root/config['data_root']/f'{split}.csv'
        if file_hash(source) != recovered['recovered_sha256'] or file_hash(recovered['source']) != recovered['source_sha256']:
            raise ValueError(f'Recovered or original {split} CSV differs from the recovery report')


def chart_fingerprint(compiled, state):
    """Strict chart fingerprint, NOT an exhaustive crystal-equivalence classifier."""
    structure = compiled.expand(torch.as_tensor(state, dtype=torch.float64))
    coords = structure['coordinates'].numpy()
    lattice = structure['lattice'].numpy()
    diff = coords[None]-coords[:, None]
    diff[..., compiled.periodic] -= np.round(diff[..., compiled.periodic])
    distances = np.linalg.norm(diff@lattice.T, axis=-1)
    pairs = []
    for i in range(compiled.num_atoms):
        for j in range(i):
            pairs.append((min(int(compiled.elements[i]), int(compiled.elements[j])),
                          max(int(compiled.elements[i]), int(compiled.elements[j])), round(float(distances[i, j]), 6)))
    data = {'kind': compiled.descriptor.kind, 'group': compiled.descriptor.number,
            'elements': sorted(compiled.elements.tolist()), 'metric': np.round(lattice.T@lattice, 6).tolist(),
            'pairs': sorted(pairs)}
    return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()


def inspect_records(splits, config):
    errors, warnings, summary = [], [], {}
    seen_ids, seen_cifs, seen_charts = {}, {}, {}
    train_elements = {o['atomic_number'] for r in splits.get('train', []) for o in r['descriptor']['orbits']}
    allowed = set(config.get('allowed_elements') or train_elements)
    for split, records in splits.items():
        groups, elements, labels, atoms, orbits = Counter(), Counter(), Counter(), [], []
        for record in records:
            identity = (record['dataset'], record['id'])
            if identity in seen_ids:
                errors.append({'type': 'duplicate_id', 'id': list(identity), 'splits': [seen_ids[identity], split]})
            seen_ids[identity] = split
            try:
                descriptor = Descriptor.from_dict(record['descriptor'])
                compiled = compile_descriptor(descriptor)
                state = np.asarray(record['state'], dtype=float)
                if state.shape != (compiled.dof,) or not np.isfinite(state).all():
                    raise ValueError('Nonfinite or wrong-dimensional state')
                if compiled.num_atoms > config.get('max_atoms', 192) or len(descriptor.orbits) > config.get('max_orbits', 32):
                    raise ValueError('Atom/orbit budget exceeded')
                unknown = set(compiled.elements.tolist())-allowed
                if unknown:
                    if split == 'test':
                        warnings.append({'type': 'test_out_of_domain_elements', 'id': record['id'],
                                         'elements': sorted(unknown), 'keep_in_evaluation_denominator': True})
                    else:
                        raise ValueError(f'Elements outside training domain: {sorted(unknown)}')
                generated = compiled.expand(torch.as_tensor(state, dtype=torch.float64))
                if not torch.isfinite(generated['lattice']).all() or float(torch.linalg.det(generated['lattice'])) <= 0:
                    raise ValueError('Invalid expanded lattice')
                cif_hash = record.get('provenance', {}).get('source_cif_sha256')
                if cif_hash and cif_hash in seen_cifs and seen_cifs[cif_hash][0] != split:
                    errors.append({'type': 'cross_split_identical_source_cif', 'id': record['id'],
                                   'previous': seen_cifs[cif_hash]})
                if cif_hash:
                    seen_cifs[cif_hash] = (split, record['id'])
                fingerprint = chart_fingerprint(compiled, state)
                if fingerprint in seen_charts:
                    previous_split, previous = seen_charts[fingerprint]
                    previous_id = previous['id']
                    # Pair-distance hashes can collide for homometric structures. Verify candidates.
                    from pymatgen.analysis.structure_matcher import StructureMatcher
                    from .evaluation import to_structure
                    pc = compile_descriptor(Descriptor.from_dict(previous['descriptor']))
                    pgen = pc.expand(torch.tensor(previous['state'], dtype=torch.float64))
                    matcher = StructureMatcher(ltol=1e-4, stol=1e-4, angle_tol=1e-3, scale=False)
                    if matcher.fit(to_structure(pc, pgen), to_structure(compiled, generated)):
                        entry = {'type': 'verified_duplicate_structure', 'id': record['id'],
                                 'split': split, 'previous': [previous_split, previous_id]}
                        (errors if previous_split != split else warnings).append(entry)
                else:
                    seen_charts[fingerprint] = (split, record)
                for name, prop in config.get('properties', {}).items():
                    value = record['properties'].get(name)
                    if value is not None:
                        if not np.isfinite(value):
                            raise ValueError(f'Nonfinite property {name}')
                        if prop.get('type', 'scalar') == 'categorical' and (int(value) != value or not 0 <= value < prop['num_classes']):
                            raise ValueError(f'Invalid categorical property {name}')
                        labels[name] += 1
                groups[f'{descriptor.kind}:{descriptor.number}'] += 1
                elements.update(compiled.elements.tolist())
                atoms.append(compiled.num_atoms); orbits.append(len(descriptor.orbits))
            except (ValueError, RuntimeError, KeyError, np.linalg.LinAlgError) as error:
                errors.append({'type': 'invalid_record', 'split': split, 'id': record['id'], 'reason': str(error)})
        summary[split] = {'records': len(records), 'groups': dict(groups), 'element_counts': dict(elements),
                          'known_property_labels': dict(labels), 'max_atoms': max(atoms, default=0),
                          'max_orbits': max(orbits, default=0)}
    if not splits.get('train') or not splits.get('val'):
        errors.append({'type': 'empty_train_or_val'})
    for name in config.get('properties', {}):
        if not summary.get('train', {}).get('known_property_labels', {}).get(name):
            errors.append({'type': 'missing_training_property', 'property': name})
    warnings.append({'type': 'scope', 'message': 'Chart-hash candidate verification is not exhaustive near-duplicate or prototype-disjoint splitting.'})
    return {'passed': not errors, 'errors': errors, 'warnings': warnings, 'splits': summary,
            'allowed_elements': sorted(allowed), 'stability_certified': False}


def audit_files(paths, config, output):
    records_by_path = {str(Path(path).resolve()): read_jsonl(path) for files in paths.values() for path in files}
    splits = {split: [r for path in files for r in records_by_path[str(Path(path).resolve())]] for split, files in paths.items()}
    splits = {split: attach_dataset_domains(records, config) for split, records in splits.items()}
    report = inspect_records(splits, config)
    if config.get('source_verified') is False:
        report['errors'].append({'type': 'unverified_coordinate_source',
                                 'message': 'Confirm original structures and coordinate/cell convention before formal training.'})
    report['inputs'] = {split: [{'path': str(Path(p).resolve()), 'sha256': file_hash(p)} for p in files]
                        for split, files in paths.items()}
    report['config_sha256'] = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    report['conversion'] = []
    if config.get('joint_split_manifest'):
        from .joint import check_joint_lineage
        try:
            report['joint_lineage'] = check_joint_lineage(config, paths)
        except (ValueError, OSError, KeyError) as error:
            report['errors'].append({'type': 'joint_lineage_not_verified', 'reason': str(error)})
    source_configs = {name: (ROOT/path, yaml.safe_load((ROOT/path).read_text(encoding='utf-8')))
                      for name, path in config.get('source_configs', {}).items()}
    report['source_configs'] = {name: {'path': str(path.resolve()), 'sha256': file_hash(path)}
                                for name, (path, _) in source_configs.items()}
    try:
        check_recovered_source(config)
        for _, source_config in source_configs.values():
            check_recovered_source(source_config)
    except (ValueError, OSError, KeyError) as error:
        report['errors'].append({'type': 'source_recovery_not_verified', 'reason': str(error)})
    for files in paths.values():
        for path in files:
            if config.get('joint_split_manifest'):
                continue  # Parent conversions and exact conservation are checked by joint lineage.
            metadata = Path(str(path)+'.report.json')
            if not metadata.exists():
                if config.get('require_full_preparation'):
                    report['errors'].append({'type': 'missing_conversion_report', 'path': str(path)})
                continue
            conversion = json.loads(metadata.read_text(encoding='utf-8'))
            report['conversion'].append(conversion)
            conversion_config = config
            if source_configs:
                datasets = {r['dataset'] for r in records_by_path[str(Path(path).resolve())]}
                if len(datasets) != 1 or conversion['dataset'] not in datasets or conversion['dataset'] not in source_configs:
                    report['errors'].append({'type': 'unknown_or_mixed_conversion_source', 'path': str(path)})
                    continue
                conversion_config = source_configs[conversion['dataset']][1]
                if conversion_config.get('source_verified') is False:
                    report['errors'].append({'type': 'unverified_coordinate_source', 'path': str(path)})
            manifest_path = Path(str(path)+'.manifest.json')
            if config.get('require_full_preparation'):
                if not manifest_path.exists():
                    report['errors'].append({'type': 'missing_conversion_manifest', 'path': str(path)})
                else:
                    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
                    expected = {key: conversion_config.get(key) for key in manifest['conversion_config']}
                    if manifest['conversion_config'] != expected:
                        report['errors'].append({'type': 'conversion_config_mismatch', 'path': str(path)})
            if config.get('require_full_preparation') and not conversion.get('full_source'):
                report['errors'].append({'type': 'partial_preparation', 'path': str(path)})
            coverage = conversion['accepted']/max(conversion['input'], 1)
            if coverage < config.get('minimum_conversion_coverage', 0):
                report['errors'].append({'type': 'insufficient_conversion_coverage', 'path': str(path), 'coverage': coverage})
    report['passed'] = not report['errors']
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    return report


def check_training_gate(config, train_paths, val_paths, root):
    if not config.get('require_data_audit'):
        return
    if config.get('source_verified') is False:
        raise ValueError('Original coordinate/cell source is unverified; confirm it before formal training')
    check_recovered_source(config, root)
    path = root/config['data_audit']
    if not path.exists():
        raise ValueError(f'Data audit is missing: {path}; run scripts/prepare_training.py first')
    report = json.loads(path.read_text(encoding='utf-8'))
    if not report['passed']:
        raise ValueError(f'Data audit failed: {path}; resolve its errors before formal training')
    expected = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if report['config_sha256'] != expected:
        raise ValueError('Data audit was made for a different configuration')
    for name, source_path in config.get('source_configs', {}).items():
        path = root/source_path
        if report.get('source_configs', {}).get(name) != {'path': str(path.resolve()), 'sha256': file_hash(path)}:
            raise ValueError('Source configuration differs from the audited conversion inputs')
        check_recovered_source(yaml.safe_load(path.read_text(encoding='utf-8')), root)
    for split, paths in [('train', train_paths), ('val', val_paths)]:
        actual = [{'path': str(Path(p).resolve()), 'sha256': file_hash(p)} for p in paths]
        if report['inputs'][split] != actual:
            raise ValueError('Training split contents differ from the audited inputs')
    if config.get('joint_split_manifest'):
        from .joint import check_joint_lineage
        check_joint_lineage(config, {'train': train_paths, 'val': val_paths})
