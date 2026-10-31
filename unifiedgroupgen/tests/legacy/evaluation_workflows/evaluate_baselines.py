"""Resumable evaluation of the three repaired-data baselines; never starts training."""
from unifiedgroupgen.tests.legacy.paths import package_parent, runtime_pythonpath

from unifiedgroupgen.tests.legacy.paths import project_root, resolve_path, module_command, source_key
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile

ROOT = project_root()
MODELS = {'mp20_u': ['mp_20'], 'c2db51_u': ['c2db_51'], 'joint_u': ['mp_20', 'c2db_51']}
MODES = ('de_novo', 'composition', 'fixed_description', 'fixed_group')


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest() if hasattr(hashlib, 'file_digest') else _digest(stream)


def _digest(stream):
    value = hashlib.sha256()
    for block in iter(lambda: stream.read(1024*1024), b''):
        value.update(block)
    return value.hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def load(path):
    return json.loads(path.read_text(encoding='utf-8'))


def group_limit(dataset):
    return 80 if dataset == 'c2db_51' else 230


def domain_count(model, count):
    return count//2 if model == 'joint_u' else count


def plan(count=256, per_group=5):
    return [{'model': model, 'dataset': dataset,
             'de_novo': domain_count(model, count), 'composition': domain_count(model, count),
             'fixed_description': domain_count(model, count), 'fixed_group': group_limit(dataset)*per_group,
             'joint_context_retained': model == 'joint_u'}
            for model, datasets in MODELS.items() for dataset in datasets]


def resolve_input(path):
    """Relocate only the known repository prefix; file hashes remain mandatory."""
    value = str(path).replace('\\', '/')
    prefix = '/root/private_data/rszhong/workspace/UnifiedGroupGen/'
    if value.startswith(prefix):
        return resolve_path(ROOT/value[len(prefix):],ROOT)
    candidate = Path(path)
    return candidate if candidate.is_absolute() else resolve_path(ROOT/candidate,ROOT)


def verify_inputs(saved):
    from unifiedgroupgen.pl_data.encoding import read_jsonl
    config = saved['config']
    if config.get('data_revision') != 'source_integrity_v1' or not config.get('require_data_audit'):
        raise ValueError('This campaign requires the repaired, audited source_integrity_v1 data')
    if any(not specification.get('context') for specification in config.get('properties', {}).values()):
        raise ValueError('This campaign is for the three baseline models, not property-CFG models')
    audit_path = resolve_input(config['data_audit'])
    audit = load(audit_path)
    if not audit.get('passed'):
        raise ValueError('Data audit did not pass')
    files = {}
    for split in ('train', 'val', 'test'):
        paths = [resolve_input(p) for p in config[split+'_files']]
        actual = [digest(p) for p in paths]
        expected = [item['sha256'] for item in audit['inputs'][split]]
        if actual != expected:
            raise ValueError(f'{split} bytes differ from the audited data')
        if split != 'test' and actual != saved.get(split+'_split_sha256'):
            raise ValueError(f'{split} bytes differ from checkpoint training inputs')
        if paths != [resolve_input(item['path']) for item in audit['inputs'][split]]:
            raise ValueError(f'{split} paths differ from audited inputs')
        files[split] = [{'path': source_key(p,ROOT), 'sha256': h} for p, h in zip(paths, actual)]
    tests = [r for item in files['test'] for r in read_jsonl(resolve_path(ROOT/item['path'],ROOT))]
    if len({(r['dataset'], str(r['id'])) for r in tests}) != len(tests):
        raise ValueError('Duplicate test identifiers')
    return tests, {'audit_sha256': digest(audit_path), 'files': files}


def seed_for(seed, model, dataset, mode, index):
    return int(hashlib.sha256(f'{seed}/{model}/{dataset}/{mode}/{index}'.encode()).hexdigest()[:8], 16)


def read_attempts(folder):
    path = folder/'samples.jsonl'
    if not path.exists():
        return []
    # A crash may leave one partial final line. Preserve the evidence, then truncate that line only.
    raw = path.read_bytes()
    lines = raw.splitlines(keepends=True)
    rows, valid = [], 0
    for i, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if i != len(lines)-1:
                raise ValueError(f'Corrupt nonfinal attempt line: {path}')
            path.with_name('samples.interrupted-tail.bin').write_bytes(line)
            path.write_bytes(raw[:valid])
            break
        if row['index'] != len(rows):
            raise ValueError('Noncontinuous attempt indices')
        if row['status'] == 'generated' and not (folder/f"sample_{row['index']:05d}.cif").exists():
            raise ValueError('Completed attempt is missing its CIF')
        if row['status'] == 'generated' and digest(folder/f"sample_{row['index']:05d}.cif") != row['cif_sha256']:
            raise ValueError('Completed CIF changed since sampling')
        rows.append(row)
        valid += len(line)
    if rows and path.read_bytes()[-1:] != b'\n':
        with path.open('ab') as stream:
            stream.write(b'\n')
    return rows


def contexts(config, dataset):
    return {'source_domain': config['dataset_domains'][dataset]} if config.get('dataset_domains') else {}


def checkpoint_path(args, model):
    return resolve_path(ROOT/args.checkpoint_root/model/'model.best.pt',ROOT)


def gpu_guard(device):
    if not device.startswith('cuda'):
        return
    result = subprocess.run(module_command(['nvidia-smi', '--query-compute-apps=pid,process_name', '--format=csv,noheader,nounits']),
                            capture_output=True, text=True, check=True)
    active = result.stdout.strip().splitlines()
    if sys.platform == 'win32':
        # WDDM also lists desktop graphics clients as compute apps with N/A usage.
        active = [row for row in active if ',' not in row or any(name in row.lower() for name in
                  ('python', 'julia', 'matlab', 'blender', 'cuda'))]
    if active:
        raise RuntimeError('GPU compute processes are active. Keep training running; use --device cpu or wait for an idle GPU.')


def reconstruction_metrics(structure, target, kind):
    """Supplement the existing Boolean matcher with aligned physical RMSD in angstrom."""
    import numpy as np
    from pymatgen.analysis.structure_matcher import StructureMatcher
    from pymatgen.core import Lattice, Structure
    from unifiedgroupgen.evaluation.utils import layer_structure, structure_match
    result = {'reference_match': structure_match(structure, target, kind), 'rmsd_angstrom': None}
    if not result['reference_match']:
        return result
    if kind == 'space':
        a, b = target, structure
    else:
        height = 10*max(*target.lattice.abc[:2], *structure.lattice.abc[:2],
                        np.ptp(target.cart_coords[:, 2]), np.ptp(structure.cart_coords[:, 2]), 4)
        a, b = layer_structure(target, height), layer_structure(structure, height)
        a = Structure(Lattice(a.lattice.matrix), a.species, a.frac_coords)
        b = Structure(Lattice(b.lattice.matrix), b.species, b.frac_coords)
    matcher = StructureMatcher(ltol=.2, stol=.3 if kind == 'space' else 2, angle_tol=5,
                               scale=False, primitive_cell=False, attempt_supercell=True)
    mapped = matcher.get_s2_like_s1(a, b)
    if mapped is None or len(mapped) != len(a):
        return result
    if kind == 'space':
        distances = [a.lattice.get_distance_and_image(x, y)[0] for x, y in zip(a.frac_coords, mapped.frac_coords)]
        result['rmsd_angstrom'] = float(np.sqrt(np.mean(np.square(distances))))
    else:
        delta = mapped.frac_coords-a.frac_coords
        delta[:, :2] -= np.round(delta[:, :2])
        delta[:, 2] -= delta[:, 2].mean()
        result['rmsd_angstrom'] = float(np.sqrt(np.mean(np.sum((delta @ a.lattice.matrix)**2, axis=1))))
    return result


def symmetry_tolerances(structure, kind):
    """One isolated diagnostic process; a native abort remains an explicit unknown."""
    request = {'kind': kind, 'cell': [structure.lattice.matrix.tolist(), structure.frac_coords.tolist(),
                                     list(structure.atomic_numbers)]}
    code = """import json, sys, spglib
r = json.load(sys.stdin)
out = {}
for tolerance in (0.0001, 0.001, 0.01):
    value = (spglib.get_symmetry_dataset(tuple(r['cell']), symprec=tolerance) if r['kind']=='space'
             else spglib.get_layergroup(tuple(r['cell']), aperiodic_dir=2, symprec=tolerance))
    out[str(tolerance)] = int(value.number) if value is not None else None
print(json.dumps(out))
"""
    try:
        result = subprocess.run(module_command([sys.executable, '-c', code]), input=json.dumps(request),
                                capture_output=True, text=True, timeout=30,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        if result.returncode:
            return {'error': f'Native diagnostic exited {result.returncode}'}
        return json.loads(result.stdout)
    except (subprocess.TimeoutExpired, ValueError) as error:
        return {'error': str(error)}


def component_test(proposal, flow, records, saved, output, batch_size):
    import torch
    from unifiedgroupgen.tests.legacy.evaluation.components import descriptor_batch
    from unifiedgroupgen.pl_modules.objectives.flow_matching import flow_loss_batch
    from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
    if (output/'summary.json').exists():
        return
    allowed = set(saved['config']['allowed_elements'])
    rows, unsupported = [], []
    torch.manual_seed(918)
    with torch.inference_mode():
        for start in range(0, len(records), batch_size):
            batch = []
            for record in records[start:start+batch_size]:
                unseen = sorted({o['atomic_number'] for o in record['descriptor']['orbits']}-allowed)
                if unseen:
                    unsupported.append({'dataset': record['dataset'], 'id': record['id'], 'unseen_elements': unseen})
                else:
                    batch.append(record)
            if not batch:
                continue
            ds = [Descriptor.from_dict(r['descriptor']) for r in batch]
            cs = [compile_descriptor(d) for d in ds]
            props = [contexts(saved['config'], r['dataset']) for r in batch]
            tokens = descriptor_batch(proposal, ds, props)
            endpoints = [next(flow.parameters()).new_tensor(r['state']) for r in batch]
            continuous = flow_loss_batch(flow, cs, endpoints, props)
            for i, record in enumerate(batch):
                k, q = float(continuous['lattice'][i]), float(continuous['coordinates'][i])
                rows.append({'dataset': record['dataset'], 'id': record['id'], **tokens[i],
                             'lattice': k, 'coordinates_raw': q, 'coordinates_dimensionless': q/9, 'flow': k+q/9})
            save(output/'status.json', {'state': 'running', 'completed': min(start+batch_size, len(records)), 'requested': len(records)})
    def aggregate(selected):
        n = len(selected)
        result = {'evaluated': n}
        for key in ('descriptor_loss', 'lattice', 'coordinates_raw', 'coordinates_dimensionless', 'flow'):
            result[key] = sum(r[key] for r in selected)/n if n else None
        result['teacher_forced_sequence_accuracy'] = sum(r['teacher_forced_sequence_correct'] for r in selected)/n if n else None
        result['tokens'] = {}
        for token in ('G', 'W', 'Z', 'STOP'):
            count = sum(r['parts'][token]['tokens'] for r in selected)
            result['tokens'][token] = {'tokens': count,
                'nll': sum(r['parts'][token]['nll_sum'] for r in selected)/count if count else None,
                'top1': sum(r['parts'][token]['correct'] for r in selected)/count if count else None}
        return result
    output.mkdir(parents=True, exist_ok=True)
    (output/'records.jsonl').write_text(''.join(json.dumps(r, allow_nan=False)+'\n' for r in rows), encoding='utf-8')
    save(output/'summary.json', {'state': 'complete', 'requested': len(records), 'unsupported': unsupported,
         'records_sha256': digest(output/'records.jsonl'),
         'overall': aggregate(rows), 'sources': {d: aggregate([r for r in rows if r['dataset'] == d]) for d in sorted({r['dataset'] for r in records})},
         'protocol': 'True-prefix G/W/Z/STOP; true descriptor flow at one fixed-seed random path/time; contexts only; coordinate error /9.'})


def sample_job(proposal, flow, saved, model, dataset, mode, records, args):
    import torch
    from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
    from unifiedgroupgen.generation.flow import sample_flow
    from unifiedgroupgen.group_constraints.validation import summarize_attempts, to_structure, validate
    from unifiedgroupgen.tests.legacy.evaluation_tasks import structure_from_record
    folder = args.output/model/dataset/mode
    folder.mkdir(parents=True, exist_ok=True)
    config = saved['config']
    selected = sorted(records, key=lambda r: hashlib.sha256(f"{args.seed}/{r['id']}".encode()).hexdigest())
    count = group_limit(dataset)*args.per_group if mode == 'fixed_group' else domain_count(model, args.count)
    if mode in ('composition', 'fixed_description') and len(selected) < count:
        raise ValueError('Not enough distinct test cases')
    rows = read_attempts(folder)
    if len(rows) > count:
        raise ValueError('Existing attempts exceed the planned count')
    props = contexts(config, dataset)
    kind = 'layer' if dataset == 'c2db_51' else 'space'
    train_groups = set()
    from unifiedgroupgen.pl_data.encoding import read_jsonl
    for path in config['train_files']:
        train_groups.update(r['descriptor']['number'] for r in read_jsonl(resolve_input(path)) if r['dataset'] == dataset)
    started = time.monotonic()
    previous_seconds = sum(r.get('wall_seconds', 0) for r in rows)
    for index in range(len(rows), count):
        before = time.monotonic()
        seed = seed_for(args.seed, model, dataset, mode, index)
        torch.manual_seed(seed)
        generator = torch.Generator(device=args.device).manual_seed(seed)
        group = index//args.per_group+1 if mode == 'fixed_group' else None
        row = {'index': index, 'seed': seed, 'task': mode, 'requested_group': group,
               'group_seen_in_training': group in train_groups if group is not None else None,
               'properties_requested': props, 'status': 'failed'}
        try:
            reference = selected[index] if mode in ('composition', 'fixed_description') else None
            counts, fixed = None, None
            if reference:
                row['reference_id'] = reference['id']
                truth = compile_descriptor(Descriptor.from_dict(reference['descriptor']))
                row['reference_coordinate_dof'] = truth.coordinate_dof
                row['reference_lattice_dof'] = truth.lattice_dof
                counts = dict(Counter(truth.elements.tolist())) if mode == 'composition' else None
                unseen = set(truth.elements.tolist())-set(config['allowed_elements'])
                if unseen:
                    raise ValueError(f'Unsupported test elements: {sorted(unseen)}')
                fixed = Descriptor.from_dict(reference['descriptor']) if mode == 'fixed_description' else None
            prefix = [proposal.vocab.index['BOS'], proposal.vocab.index[f'G:{kind}:{group}']] if group else None
            descriptor = fixed or proposal.sample(props, kind, counts, 0, 1., generator,
                prefix=prefix, condition_composition=False)
            compiled = compile_descriptor(descriptor)
            actual_counts = dict(Counter(compiled.elements.tolist()))
            if counts is not None and actual_counts != counts:
                raise ValueError('Generated counts differ from requested conventional-cell composition')
            generated = sample_flow(flow, compiled, props, args.steps, 0, generator=generator,
                                    condition_observation=False)
            report = validate(compiled, generated, isolate_identifier=True)
            structure = to_structure(compiled, generated)
            cif = folder/f'sample_{index:05d}.cif'
            structure.to(filename=str(cif))
            row.update(status='generated', descriptor=descriptor.to_dict(), state=generated['state'].cpu().tolist(),
                       validation=report, cif_sha256=digest(cif), composition_satisfied=counts is None or actual_counts == counts,
                       chemical_request={'composition': counts}, generator='learned_flow')
            if reference:
                row.update(reconstruction_metrics(structure, structure_from_record(reference), kind))
        except (ValueError, RuntimeError, FloatingPointError, KeyError, IndexError) as error:
            row['reason'] = f'{type(error).__name__}: {error}'
        row['wall_seconds'] = time.monotonic()-before
        # Serialization failures (including nonfinite output) must not corrupt a resumable ledger.
        try:
            line = json.dumps(row, allow_nan=False)
        except ValueError as error:
            row = {'index': index, 'seed': seed, 'task': mode, 'requested_group': group,
                   'status': 'failed', 'reason': str(error), 'wall_seconds': row['wall_seconds']}
            line = json.dumps(row, allow_nan=False)
        with (folder/'samples.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(line+'\n')
            stream.flush()
        rows.append(row)
        status = {'state': 'running', 'attempted': len(rows), 'requested': count,
                  'generated': sum(r['status'] == 'generated' for r in rows),
                  'seconds': previous_seconds+time.monotonic()-started}
        save(folder/'sampling.status.json', status)
        if len(rows) % 10 == 0 or len(rows) == count:
            print(json.dumps({'model': model, 'dataset': dataset, 'task': mode, **status}), flush=True)
    summary = summarize_attempts(rows)
    if mode in ('composition', 'fixed_description'):
        matched = [r for r in rows if r.get('reference_match')]
        values = [r['rmsd_angstrom'] for r in matched if r.get('rmsd_angstrom') is not None]
        summary.update(match_at_1=sum(bool(r.get('reference_match')) for r in rows)/count,
                       composition_satisfaction=sum(bool(r.get('composition_satisfied')) for r in rows)/count if mode == 'composition' else None,
                       matched_rmsd_evaluated=len(values), matched_rmsd_mean_angstrom=sum(values)/len(values) if values else None,
                       zero_coordinate_dof_cases=sum(r.get('reference_coordinate_dof') == 0 for r in rows))
        summary['coordinate_dof_strata'] = {}
        for name, predicate in (('zero', lambda r: r.get('reference_coordinate_dof') == 0),
                                ('positive', lambda r: (r.get('reference_coordinate_dof') or 0) > 0),
                                ('unknown', lambda r: r.get('reference_coordinate_dof') is None)):
            selected_rows = [r for r in rows if predicate(r)]
            summary['coordinate_dof_strata'][name] = {'attempted': len(selected_rows),
                'matched': sum(bool(r.get('reference_match')) for r in selected_rows),
                'match_at_1': sum(bool(r.get('reference_match')) for r in selected_rows)/len(selected_rows) if selected_rows else None}
    summary.update(state='complete', checkpoint_sha256=args.checkpoint_sha, steps=args.steps, guidance=0,
                   source_context=props, seen_target_groups=sorted(train_groups))
    save(folder/'summary.json', summary)
    save(folder/'sampling.status.json', {'state': 'complete', 'attempted': count, 'requested': count,
         'generated': summary['generated'], 'seconds': sum(r['wall_seconds'] for r in rows)})


def inference_worker(args):
    import torch
    sys.path[:0] = [str(package_parent()), str(resolve_path(ROOT/'scripts',ROOT))]
    from unifiedgroupgen.pl_modules.factory import build_models

    checkpoint = checkpoint_path(args, args.model)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    args.checkpoint_sha = digest(checkpoint)
    records, audit = verify_inputs(saved)
    if saved.get('runtime_options', {}).get('coordinate_length_scale') != 3.:
        raise ValueError('Expected the approved 3 angstrom coordinate loss scale')
    save(args.output/args.model/'data_identity.json', {'checkpoint_sha256': args.checkpoint_sha, **audit,
         'inference_runtime': {'device': args.device, 'threads': args.threads, 'test_batch_size': args.test_batch_size,
                               'steps': args.steps, 'guidance': 0, 'edge_chunk_size': 32768,
                               'activation_checkpointing': False, 'coordinate_length_scale': 3.0}})
    inference_config = {**saved['config'], 'model': {**saved['config']['model'],
                         'checkpoint_layers': False, 'edge_chunk_size': 32768}}
    proposal, flow = build_models(inference_config, args.device)
    proposal.load_state_dict(saved['proposal']); flow.load_state_dict(saved['flow'])
    proposal.eval(); flow.eval()
    torch.set_num_threads(args.threads)
    component_test(proposal, flow, records, saved, args.output/args.model/'test', args.test_batch_size)
    with torch.inference_mode():
        for dataset in MODELS[args.model]:
            subset = [r for r in records if r['dataset'] == dataset]
            for mode in MODES:
                sample_job(proposal, flow, saved, args.model, dataset, mode, subset, args)


def metric_worker(args):
    import torch
    sys.path[:0] = [str(package_parent()), str(resolve_path(ROOT/'scripts',ROOT))]
    import pandas as pd
    from unifiedgroupgen.tests.legacy.evaluation_workflows.evaluate_generation import evaluate_samples, plot_job, record_structure, reference_fingerprints
    from unifiedgroupgen.pl_data.encoding import read_jsonl
    torch.set_num_threads(args.threads)
    checkpoint = checkpoint_path(args, args.model)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    _, audit = verify_inputs(saved)
    kind = 'layer' if args.dataset == 'c2db_51' else 'space'
    training = defaultdict(list)
    for item in audit['files']['train']:
        for record in read_jsonl(resolve_path(ROOT/item['path'],ROOT)):
            if record['descriptor']['kind'] == kind:
                structure = record_structure(record)
                training[structure.composition.reduced_formula].append(structure)
    references = reference_fingerprints([resolve_path(ROOT/item['path'],ROOT) for item in audit['files']['test']],
        args.dataset, args.output/'reference_cache'/args.model, True, workers=1)
    for mode in MODES:
        folder = args.output/args.model/args.dataset/mode
        metrics = folder/'metrics'
        metrics.mkdir(exist_ok=True)
        if (metrics/'summary.json').exists():
            continue
        rows, report = evaluate_samples(folder/'samples.jsonl', references, training, kind, reidentify=True)
        source = read_attempts(folder)
        for row, original in zip(rows, source):
            if row['index'] != original['index']:
                raise ValueError('Metric/sample order changed')
            for key in ('requested_group', 'group_seen_in_training', 'reference_id', 'reference_match',
                        'rmsd_angstrom', 'reference_coordinate_dof', 'composition_satisfied'):
                row[key] = original.get(key)
            if mode == 'fixed_group' and row['generated']:
                from pymatgen.io.cif import CifParser
                cif = folder/f"sample_{row['index']:05d}.cif"
                try:
                    structure = CifParser(cif, frac_tolerance=0).parse_structures(primitive=False)[0]
                    identified = symmetry_tolerances(structure, kind)
                except (ValueError, RuntimeError, TypeError) as error:
                    identified = {'error': str(error)}
                row['tolerance_identifier_error'] = identified.get('error')
                for tolerance in ('0.0001', '0.001', '0.01'):
                    number = identified.get(tolerance)
                    row['exact_group_at_'+tolerance] = number == row['requested_group'] if number is not None else None
        report.update(state='complete', checkpoint_sha256=digest(checkpoint), samples_sha256=digest(folder/'samples.jsonl'),
                      data_identity=audit, task=mode, dataset=args.dataset)
        pd.DataFrame(rows).to_csv(metrics/'records.csv', index=False)
        report['records_csv_sha256'] = digest(metrics/'records.csv')
        if mode == 'fixed_group':
            grouped = []
            for group in range(1, group_limit(args.dataset)+1):
                selected = [r for r in rows if r['requested_group'] == group]
                grouped.append({'group': group, 'attempted': len(selected),
                    'seen': selected[0]['group_seen_in_training'],
                    **{key: sum(bool(r.get(key)) for r in selected)/len(selected) for key in
                       ('generated', 'target_group_containment', 'exact_target_group', 'joint_valid',
                        'exact_group_at_0.0001', 'exact_group_at_0.001', 'exact_group_at_0.01')},
                    **{key+'_unknown': sum(r.get(key) is None for r in selected) for key in
                       ('exact_group_at_0.0001', 'exact_group_at_0.001', 'exact_group_at_0.01')}})
            pd.DataFrame(grouped).to_csv(metrics/'groups.csv', index=False)
            report['groups_csv_sha256'] = digest(metrics/'groups.csv')
        plot_job(rows, references, report, metrics/'quality.png', f'{args.model}/{args.dataset}/{mode}', kind)
        save(metrics/'summary.json', report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('artifacts/evaluations/baselines_256_seed142'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--threads', type=int, default=1)
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--per-group', type=int, default=5)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--seed', type=int, default=142)
    parser.add_argument('--test-batch-size', type=int, default=8)
    parser.add_argument('--checkpoint-root', type=Path, default=Path('artifacts/runs/main15_seed42'))
    parser.add_argument('--update-manuscript', action='store_true', help='Insert the verified final report into the existing article')
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--run', action='store_true')
    action.add_argument('--status', action='store_true')
    action.add_argument('--pack', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--worker', choices=['inference', 'metrics'], help=argparse.SUPPRESS)
    parser.add_argument('--model', choices=list(MODELS), help=argparse.SUPPRESS)
    parser.add_argument('--dataset', choices=['mp_20', 'c2db_51'], help=argparse.SUPPRESS)
    args = parser.parse_args()
    if min(args.count, args.per_group, args.steps, args.threads, args.test_batch_size) < 1 or args.count % 2:
        parser.error('Use positive settings and an even count (joint domain split)')
    args.output = (resolve_path(ROOT/args.output,ROOT)).resolve()
    if not args.output.is_relative_to(resolve_path(ROOT/'artifacts/evaluations',ROOT)):
        parser.error('Use a dedicated artifacts/evaluations output directory')
    os.environ.update(MKL_THREADING_LAYER='SEQUENTIAL', OMP_NUM_THREADS=str(args.threads), MKL_NUM_THREADS=str(args.threads))
    if args.worker:
        if args.worker == 'inference':
            gpu_guard(args.device)
            inference_worker(args)
        else:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            metric_worker(args)
        return
    pipeline = args.output/'pipeline.json'
    if args.status:
        print(json.dumps(load(pipeline) if pipeline.exists() else {'state': 'not_started'}, indent=2))
        for path in args.output.glob('**/*status.json'):
            print(path.relative_to(args.output), json.dumps(load(path)))
        return
    if args.pack:
        if not pipeline.exists() or load(pipeline)['state'] != 'complete':
            parser.error('Only package a complete campaign; inspect --status first')
        archive = args.output.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in args.output.rglob('*'):
                if path.is_file():
                    bundle.write(path, str(path.relative_to(args.output.parent)))
            report_directory = resolve_path(ROOT/load(pipeline)['report_directory'],ROOT)
            for path in report_directory.rglob('*'):
                if path.is_file():
                    bundle.write(path, str(Path(args.output.name)/'report'/path.relative_to(report_directory)))
        archive.with_suffix('.zip.sha256').write_text(digest(archive)+'  '+archive.name+'\n', encoding='ascii')
        print(archive)
        return
    jobs = plan(args.count, args.per_group)
    if not args.run:
        print(json.dumps({'full_test_requested': 24854, 'generation_attempts': sum(sum(j[k] for k in MODES) for j in jobs),
                          'device': args.device, 'jobs': jobs, 'training_jobs': 0}, indent=2))
        return
    gpu_guard(args.device)
    if os.name != 'nt':
        os.nice(10)
    args.output.mkdir(parents=True, exist_ok=True)
    lock = args.output/'controller.lock'
    if lock.exists():
        pid = int(lock.read_text())
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            if not args.resume:
                raise RuntimeError('Stale lock: use --resume')
            lock.unlink()
        else:
            raise RuntimeError(f'Campaign controller PID {pid} exists; refusing duplicate launch')
    with lock.open('x') as stream:
        stream.write(str(os.getpid()))
    started = False
    try:
        code_files = [Path(__file__), resolve_path(ROOT/'scripts/summarize_baselines.py',ROOT), resolve_path(ROOT/'scripts/legacy_eval/components.py',ROOT),
                      resolve_path(ROOT/'src/unifiedgroupgen/flow.py',ROOT), resolve_path(ROOT/'src/unifiedgroupgen/proposal.py',ROOT),
                      resolve_path(ROOT/'src/unifiedgroupgen/symmetry.py',ROOT), resolve_path(ROOT/'src/unifiedgroupgen/standard_metrics.py',ROOT),
                      resolve_path(ROOT/'src/unifiedgroupgen/evaluation.py',ROOT), resolve_path(ROOT/'src/unifiedgroupgen/native_identifier.py',ROOT),
                      resolve_path(ROOT/'src/unifiedgroupgen/conditioning.py',ROOT),
                      resolve_path(ROOT/'scripts/evaluate_generation.py',ROOT)]
        identity = {'count': args.count, 'per_group': args.per_group, 'steps': args.steps, 'seed': args.seed,
                    'threads': args.threads, 'test_batch_size': args.test_batch_size,
                    'device': args.device, 'code_sha256': {source_key(p,ROOT): digest(p) for p in code_files},
                    'checkpoints': {m: digest(checkpoint_path(args, m)) for m in MODELS}}
        report_directory = resolve_path(ROOT/'assets/results'/args.output.name,ROOT)
        state = {'state': 'running', 'pid': os.getpid(), 'identity': identity, 'completed_jobs': [], 'current': None,
                 'report_directory': source_key(report_directory,ROOT)}
        if pipeline.exists():
            previous = load(pipeline)
            if not args.resume or previous['identity'] != identity:
                raise ValueError('Resume requires identical checkpoints, code, seed and evaluation settings')
            state['completed_jobs'] = previous['completed_jobs']
        started = True
        commands = [(model+'/inference', 'inference', model, None) for model in MODELS]
        commands += [(model+'/'+dataset+'/metrics', 'metrics', model, dataset) for model, ds in MODELS.items() for dataset in ds]
        for name, phase, model, dataset in commands:
            if name in state['completed_jobs']:
                continue
            state.update(current=name, updated=time.time())
            save(pipeline, state)
            command = [sys.executable, '-u', str(Path(__file__)), '--worker', phase, '--model', model,
                       '--output', str(args.output), '--device', args.device, '--threads', str(args.threads),
                       '--count', str(args.count), '--per-group', str(args.per_group), '--steps', str(args.steps),
                       '--seed', str(args.seed), '--test-batch-size', str(args.test_batch_size)]
            command += ['--checkpoint-root', str(args.checkpoint_root)]
            if dataset:
                command += ['--dataset', dataset]
            print('[START] '+name, flush=True)
            with (args.output/(name.replace('/', '_')+'.log')).open('a', encoding='utf-8') as stream:
                process = subprocess.run(module_command(command), cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                    env={**os.environ, 'PYTHONPATH': runtime_pythonpath()})
            if process.returncode:
                state.update(state='failed', returncode=process.returncode)
                save(pipeline, state)
                raise RuntimeError(f'{name} failed; inspect its log; resume will reuse recorded attempts')
            state['completed_jobs'].append(name)
            save(pipeline, state)
            print('[DONE] '+name, flush=True)
        state['current'] = 'report'
        save(pipeline, state)
        report_command = [sys.executable, str(resolve_path(ROOT/'scripts/summarize_baselines.py',ROOT)), '--study', str(args.output),
                          '--output', str(report_directory)]
        if args.update_manuscript:
            report_command.append('--update-manuscript')
        subprocess.run(module_command(report_command), cwd=ROOT, check=True)
        state.update(state='complete', current=None, updated=time.time())
        save(pipeline, state)
        print('[COMPLETE] '+str(args.output), flush=True)
    except BaseException as error:
        if started:
            state.update(state='failed', error=f'{type(error).__name__}: {error}', updated=time.time())
            save(pipeline, state)
        raise
    finally:
        lock.unlink(missing_ok=True)


if __name__ == '__main__':
    main()
