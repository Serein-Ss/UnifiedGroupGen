"""Whole-source joint splits with shared material IDs and verified duplicate components kept together."""

import hashlib
import json
from pathlib import Path
import yaml
from .data import read_jsonl, write_jsonl
from .symmetry import Descriptor, compile_descriptor
from .training import file_hash


ROOT = Path(__file__).resolve().parents[2]


def inventory_hash(records):
    rows = sorted(hashlib.sha256(json.dumps(record, sort_keys=True, allow_nan=False).encode()).hexdigest()
                  for record in records)
    return hashlib.sha256('\n'.join(rows).encode()).hexdigest()


def grouped_splits(records, families, seed=42):
    from .preflight import chart_fingerprint
    from .evaluation import to_structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    import torch
    parents, owners, charts = list(range(len(records))), {}, {}
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    def union(first, second):
        parents[root(second)] = root(first)
    matcher = StructureMatcher(ltol=1e-4, stol=1e-4, angle_tol=1e-3, scale=False)
    for index, record in enumerate(records):
        family = families[record['dataset']]
        keys = [('material', family, record['id'])]
        provenance = record.get('provenance', {})
        if provenance.get('source_cif_sha256'):
            keys.append(('cif', record['descriptor']['kind'], provenance['source_cif_sha256']))
        for key in keys:
            if key in owners:
                union(index, owners[key])
            else:
                owners[key] = index
        compiled = compile_descriptor(Descriptor.from_dict(record['descriptor']))
        fingerprint = chart_fingerprint(compiled, record['state'])
        if fingerprint in charts:
            previous = charts[fingerprint]
            if root(previous) != root(index):
                prior = records[previous]
                pc = compile_descriptor(Descriptor.from_dict(prior['descriptor']))
                first = to_structure(compiled, compiled.expand(torch.tensor(record['state'], dtype=torch.float64)))
                second = to_structure(pc, pc.expand(torch.tensor(prior['state'], dtype=torch.float64)))
                if matcher.fit(first, second):
                    union(index, previous)
        else:
            charts[fingerprint] = index
        if (index+1) % 1000 == 0:
            print(json.dumps({'event': 'joint_grouping', 'processed': index+1, 'total': len(records)}), flush=True)
    components = {}
    for index, record in enumerate(records):
        components.setdefault(root(index), []).append(record)
    assignments = {}
    for key, component in components.items():
        identity = min((families[r['dataset']], r['id']) for r in component)
        value = int(hashlib.sha256(json.dumps([seed, identity]).encode()).hexdigest()[:16], 16)/2**64
        assignments[key] = 'train' if value < 0.8 else 'val' if value < 0.9 else 'test'
    elements = {key: {orbit['atomic_number'] for record in rows for orbit in record['descriptor']['orbits']}
                for key, rows in components.items()}
    trained = set().union(*(elements[key] for key in components if assignments[key] == 'train'))
    missing = set().union(*elements.values())-trained
    moves = []
    for element in sorted(missing):
        if element in trained:
            continue
        key = min((key for key in components if element in elements[key]),
                  key=lambda key: min((families[r['dataset']], r['id']) for r in components[key]))
        moves.append({'element': element, 'from': assignments[key],
                      'material': min((families[r['dataset']], r['id']) for r in components[key])})
        assignments[key] = 'train'
        trained.update(elements[key])
    splits = {name: [] for name in ('train', 'val', 'test')}
    for key, component in components.items():
        splits[assignments[key]].extend(component)
    return splits, {'components': len(components), 'input_records': len(records),
                    'counts': {name: len(rows) for name, rows in splits.items()}, 'element_coverage_moves': moves,
                    'scope': 'Shared material IDs, identical CIFs and verified chart-hash candidates; not exhaustive near-duplicate/prototype separation'}


def prepare_joint(config, resume=False):
    from .preflight import check_training_gate
    manifest_path = ROOT/config['joint_split_manifest']
    if manifest_path.exists():
        raise ValueError('Joint manifest already exists; verify/re-audit it rather than rewriting its splits')
    records, sources = [], {}
    for dataset, path in config['source_configs'].items():
        source_config = yaml.safe_load((ROOT/path).read_text(encoding='utf-8'))
        if not source_config.get('require_data_audit') or not source_config.get('require_full_preparation'):
            raise ValueError(f'Joint source {dataset} must require a full-data audit')
        prepared = ROOT/source_config['prepared_root']
        check_training_gate(source_config, [prepared/'train.jsonl'], [prepared/'val.jsonl'], ROOT)
        paths = [prepared/f'{split}.jsonl' for split in ('train', 'val', 'test')]
        rows = [record for path in paths for record in read_jsonl(path)]
        if any(record['dataset'] != dataset for record in rows):
            raise ValueError(f'Unexpected dataset in prepared source {dataset}')
        sources[dataset] = {'config_path': str((ROOT/path).resolve()), 'config_sha256': file_hash(ROOT/path),
                            'files': [{'path': str(path), 'sha256': file_hash(path)} for path in paths], 'records': len(rows)}
        records.extend(rows)
    splits, report = grouped_splits(records, config['dataset_families'], config.get('split_seed', 42))
    manifest = {'sources': sources, 'inventory_sha256': inventory_hash(records), 'report': report,
                'split_seed': config.get('split_seed', 42), 'dataset_families': config['dataset_families'], 'outputs': []}
    for dataset in sources:
        for split, rows in splits.items():
            selected = [r for r in rows if r['dataset'] == dataset]
            path = ROOT/config['prepared_root']/dataset/f'{split}.jsonl'
            if path.exists() and not resume:
                raise ValueError(f'Joint split already exists: {path}; use a new prepared_root')
            temporary = path.with_suffix('.jsonl.tmp')
            write_jsonl(temporary, selected)
            temporary.replace(path)
            manifest['outputs'].append({'path': str(path.resolve()), 'split': split, 'dataset': dataset,
                                        'sha256': file_hash(path), 'records': len(selected)})
    path = ROOT/config['joint_split_manifest']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


def check_joint_lineage(config, paths=None):
    """Verify full parent datasets, output hashes and exact record conservation."""
    from .preflight import check_training_gate
    manifest_path = ROOT/config['joint_split_manifest']
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    if (manifest['split_seed'] != config.get('split_seed', 42)
            or manifest['dataset_families'] != config['dataset_families']
            or set(manifest['sources']) != set(config['source_configs'])):
        raise ValueError('Joint split policy/sources differ from the manifest')
    originals = []
    for dataset, source in manifest['sources'].items():
        path = ROOT/config['source_configs'][dataset]
        if file_hash(path) != source['config_sha256']:
            raise ValueError('Joint parent configuration changed')
        parent = yaml.safe_load(path.read_text(encoding='utf-8'))
        if not parent.get('require_data_audit') or not parent.get('require_full_preparation'):
            raise ValueError('Joint parents must require full-data audits')
        prepared = ROOT/parent['prepared_root']
        check_training_gate(parent, [prepared/'train.jsonl'], [prepared/'val.jsonl'], ROOT)
        for item in source['files']:
            if file_hash(item['path']) != item['sha256']:
                raise ValueError('Joint parent data changed')
            originals.extend(read_jsonl(item['path']))
    outputs, actual_paths, material_splits = [], {name: [] for name in ('train', 'val', 'test')}, {}
    for item in manifest['outputs']:
        if file_hash(item['path']) != item['sha256']:
            raise ValueError('Joint split changed after preparation')
        rows = read_jsonl(item['path'])
        for row in rows:
            if row['dataset'] != item['dataset']:
                raise ValueError('Joint output contains an unexpected source dataset')
            identity = (config['dataset_families'][row['dataset']], row['id'])
            if identity in material_splits and material_splits[identity] != item['split']:
                raise ValueError('Joint material ID crosses split boundaries')
            material_splits[identity] = item['split']
        outputs.extend(rows)
        actual_paths[item['split']].append(str(Path(item['path']).resolve()))
    if inventory_hash(originals) != manifest['inventory_sha256'] or inventory_hash(outputs) != manifest['inventory_sha256']:
        raise ValueError('Joint data do not preserve every source record exactly once')
    if paths and any(sorted(str(Path(p).resolve()) for p in files) != sorted(actual_paths[split])
                     for split, files in paths.items()):
        raise ValueError('Requested joint splits differ from the verified manifest')
    return manifest['report']
