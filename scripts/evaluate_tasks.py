"""Held-out chemical-system/formula/required-element/scaffold evaluation without fine-tuning."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import torch
from pymatgen.core import Element, Composition

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.cli import root_path, artifact_path
from unifiedgroupgen.data import read_jsonl
from unifiedgroupgen.symmetry import Descriptor, compile_descriptor
from unifiedgroupgen.training import file_hash
from unifiedgroupgen.task_evaluation import evaluate_task
from unifiedgroupgen.structure_controls import Scaffold
from unifiedgroupgen.task_evaluation import structure_from_record


TASKS = ('system', 'formula', 'required', 'completion', 'scaffold', 'fixed')


def write_scaffold(record, path):
    """Keep only the first complete orbit, without changing its group-origin height."""
    import numpy as np
    from pymatgen.core import Lattice, Structure
    c = compile_descriptor(Descriptor.from_dict(record['descriptor']))
    generated = c.expand(torch.tensor(record['state'], dtype=torch.float64))
    count = len(c.blocks[0])
    coordinates = generated['coordinates'].numpy()[:count].copy()
    lattice = generated['lattice'].numpy().copy()
    if c.descriptor.kind != 'space':
        height = max(40., 2*float(np.max(np.abs(coordinates[:, 2])))+20)
        lattice[2, 2] = height
        coordinates[:, 2] = coordinates[:, 2]/height+.5
    Structure(Lattice(lattice.T), c.elements[:count], coordinates).to(filename=str(path))
    return {str(i): i for i in range(count)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--dataset', choices=['mp_20', 'c2db_51'], required=True)
    parser.add_argument('--tasks', nargs='+', choices=TASKS, default=list(TASKS))
    parser.add_argument('--cases', type=int, default=32)
    parser.add_argument('--count', type=int, default=20, help='All attempts retained per held-out case')
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--seed', type=int, default=142)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--output', required=True)
    parser.add_argument('--plan', action='store_true')
    parser.add_argument('--postprocess-only', action='store_true')
    args = parser.parse_args()
    if min(args.cases, args.count, args.steps) < 1:
        parser.error('Budgets must be positive')
    checkpoint = root_path(args.checkpoint)
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    test_paths = [root_path(p) for p in config['test_files']]
    # Bind the complete test inventory to the checkpoint's audited training inputs.
    sys.path.insert(0, str(ROOT/'scripts'))
    from evaluate_checkpoint import check_heldout_inputs
    check_heldout_inputs(saved, test_paths)
    records = [r for p in test_paths for r in read_jsonl(p) if r['dataset'] == args.dataset]
    ordered = sorted(records, key=lambda r: hashlib.sha256(f'{args.seed}/{r["id"]}'.encode()).hexdigest())
    chosen = ordered[:args.cases]
    if not chosen:
        raise ValueError('No test records for this checkpoint/domain')
    output = artifact_path(args.output)
    if args.plan:
        print(json.dumps({'dataset': args.dataset, 'test_records': len(records), 'cases': len(chosen),
                          'tasks': args.tasks, 'maximum_attempts': len(chosen)*len(args.tasks)*args.count,
                          'checkpoint_sha256': file_hash(checkpoint)}))
        return
    if output.exists() and not args.postprocess_only:
        raise ValueError('Use a new output directory, or --postprocess-only to evaluate completed attempts')
    output.mkdir(parents=True, exist_ok=True)
    identity = {'checkpoint_sha256': file_hash(checkpoint),
                'test_sha256': [file_hash(p) for p in test_paths],
                'seed': args.seed, 'steps': args.steps, 'count': args.count, 'tasks': args.tasks,
                'case_ids': [r['id'] for r in chosen], 'dataset': args.dataset}
    identity_path = output/'identity.json'
    if args.postprocess_only:
        if json.loads(identity_path.read_text(encoding='utf-8')) != identity:
            raise ValueError('Checkpoint, test inventory or evaluation settings changed')
    else:
        identity_path.write_text(json.dumps(identity, indent=2), encoding='utf-8')
    reports, skipped = [], []
    kind = 'layer' if args.dataset == 'c2db_51' else 'space'
    for i, record in enumerate(chosen):
        case = output/f'case_{i:04d}'
        case.mkdir(exist_ok=True)
        c = compile_descriptor(Descriptor.from_dict(record['descriptor']))
        counts = {int(z): int(n) for z, n in Counter(c.elements.tolist()).items()}
        symbols = [Element.from_Z(z).symbol for z in sorted(counts)]
        descriptor_path = case/'descriptor.json'
        descriptor_path.write_text(json.dumps(record['descriptor']), encoding='utf-8')
        properties = ({'source_domain': config['dataset_domains'][args.dataset]}
                      if config.get('dataset_domains') else {})
        for task in args.tasks:
            directory = case/task
            request, extra = {}, []
            observed = None
            if task == 'system':
                request = {'allowed': list(counts), 'required': list(counts)}
                extra = ['--chemical-system', '-'.join(symbols)]
            elif task == 'required':
                required = sorted(counts)[:max(1, len(counts)//2)]
                request = {'required': required}
                extra = ['--required-elements', '-'.join(Element.from_Z(z).symbol for z in required)]
            elif task == 'formula':
                request = {'composition': counts}
                # Exact counts preserve the conventional-cell atom budget for CSP reconstruction.
                exact = Composition({Element.from_Z(z): n for z, n in counts.items()}).formula.replace(' ', '')
                extra = ['--composition', exact]
            elif task in ('completion', 'scaffold'):
                if len(c.blocks) < 2 or c.offsets[1] == c.coordinate_dof:
                    skipped.append({'id': record['id'], 'task': task,
                                    'reason': 'First known orbit leaves no unknown coordinate degrees of freedom'})
                    continue
                cif = case/'scaffold.cif'
                mapping = write_scaffold(record, cif)
                request = {'scaffold': True, 'descriptor_known': task == 'completion'}
                if task == 'completion':
                    mapping_path = case/'atom_map.json'
                    mapping_path.write_text(json.dumps(mapping), encoding='utf-8')
                    extra = ['--descriptor', str(descriptor_path), '--scaffold', str(cif), '--atom-map', str(mapping_path)]
                    observed = Scaffold.from_cif(cif, kind, c.descriptor.number, mapping)
                else:
                    try:
                        observed = Scaffold.from_cif(cif, kind, c.descriptor.number)
                    except ValueError as error:
                        skipped.append({'id': record['id'], 'task': task, 'reason': str(error)})
                        continue
                    extra = ['--group', str(c.descriptor.number), '--scaffold', str(cif)]
            elif task == 'fixed':
                request = {'descriptor_known': True}
                extra = ['--descriptor', str(descriptor_path)]
            command = ['-m', 'unifiedgroupgen.cli', '--seed', str(args.seed+i), 'sample',
                       '--checkpoint', str(checkpoint), '--device', args.device, '--kind', kind,
                       '--properties', json.dumps(properties), '--guidance', '0', '--steps', str(args.steps),
                       '--count', str(args.count), '--output', str(directory), *extra]
            if not args.postprocess_only:
                case_log = case/f'{task}.log'
                with case_log.open('w', encoding='utf-8') as stream:
                    process = subprocess.run([sys.executable, '-u', *command], cwd=ROOT,
                        stdout=stream, stderr=subprocess.STDOUT,
                        env={**os.environ, 'PYTHONPATH': str(ROOT/'src')})
                if process.returncode:
                    raise RuntimeError(f'{task} failed; inspect {case_log}')
            attempts = read_jsonl(directory/'samples.jsonl')
            if len(attempts) != args.count or [r['index'] for r in attempts] != list(range(args.count)):
                raise ValueError('Incomplete/duplicate task attempts')
            report = evaluate_task(attempts, record, kind, request, observed)
            report.update(id=record['id'], task=task, request=request, command=command,
                          samples_sha256=file_hash(directory/'samples.jsonl'))
            (case/f'{task}.metrics.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
            reports.append(report)
            print(f'[DONE] {record["id"]} {task}', flush=True)
    grouped = {}
    for task in args.tasks:
        selected = [r for r in reports if r['task'] == task]
        total = sum(r['attempted'] for r in selected)
        grouped[task] = {'cases': len(selected), 'attempts': total,
            'chemical_satisfaction_all_attempts': sum(r['chemical_satisfaction_all_attempts']*r['attempted'] for r in selected)/total if total else None,
            'geometry_validity_all_attempts': sum(r['geometry_validity_all_attempts']*r['attempted'] for r in selected)/total if total else None,
            'match_at_1': sum(r['match_at_1'] for r in selected)/len(selected) if selected else None,
            'match_at_k': sum(r['match_at_k'] for r in selected)/len(selected) if selected else None}
    (output/'summary.json').write_text(json.dumps({'identity': identity, 'tasks': grouped,
        'skipped': skipped, 'selected_test_cases': len(chosen), 'property_accuracy': None,
        'scope': 'No fine-tuning. Composition masks and scaffold projections need not sample the exact conditional density. '
                 'Completion uses full descriptor; scaffold uses only group plus the observed first orbit. '
                 'System/required reference matches are descriptive, not a unique ground-truth design objective.'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
