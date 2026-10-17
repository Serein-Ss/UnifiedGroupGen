"""Export independently assessable structures, or evaluate them with an installed ASE calculator.

The calculator factory is an explicit module:callable supplied by the researcher. No model
weights are downloaded and no requested property is used as an evaluated property.
"""

import argparse
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Structure

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from unifiedgroupgen.cli import artifact_path, root_path
from unifiedgroupgen.data import read_jsonl
from unifiedgroupgen.standard_metrics import layer_structure
from unifiedgroupgen.training import file_hash


def export_tasks(run, output, reference_config=None):
    state = json.loads((run/'pipeline.json').read_text(encoding='utf-8'))
    if state['state'] != 'complete':
        raise ValueError('Only completed generation runs can be exported')
    tasks = []
    for job, item in state['jobs'].items():
        if job == 'test':
            continue
        folder = run/item['directory']
        for name, checksum in item['output_sha256'].items():
            if file_hash(folder/name) != checksum:
                raise ValueError('Generation files changed after completion')
        for row in read_jsonl(folder/'samples.jsonl'):
            if row['status'] != 'generated':
                continue
            cif = folder/f"sample_{row['index']:05d}.cif"
            tasks.append({'job': job, 'index': row['index'], 'kind': row['descriptor']['kind'],
                          'cif': str(cif.relative_to(ROOT)).replace('\\', '/'), 'sha256': file_hash(cif)})
    if reference_config:
        from evaluate_generation import record_structure
        config = yaml.safe_load(reference_config.read_text(encoding='utf-8'))
        refs = output/'references'
        refs.mkdir()
        for path in config['test_files']:
            for record in read_jsonl(root_path(path)):
                cif = refs/f'{len(tasks):07d}.cif'
                record_structure(record).to(filename=str(cif))
                tasks.append({'dataset': record['dataset'], 'id': record['id'], 'kind': record['descriptor']['kind'],
                              'cif': str(cif.relative_to(ROOT)).replace('\\', '/'), 'sha256': file_hash(cif)})
    report = {'schema': 1, 'generation_pipeline_sha256': file_hash(run/'pipeline.json'), 'tasks': tasks,
              'reference_test_sha256': [file_hash(root_path(p)) for p in config['test_files']] if reference_config else None,
              'scope': 'Targets intentionally absent. Reference predictions for distribution comparison must '
                       'use the same evaluator and unrelaxed reference structures.'}
    (output/'tasks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def assess_one(task, calculator, output, relax, relax_cell, fmax, steps, energy_reference=None):
    from pymatgen.io.ase import AseAtomsAdaptor
    from pymatgen.io.cif import CifParser
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    cif = root_path(task['cif'])
    if file_hash(cif) != task['sha256']:
        raise ValueError('Input CIF changed')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        structure = CifParser(cif, frac_tolerance=0).parse_structures(primitive=False)[0]
    if task['kind'] == 'layer':
        structure = layer_structure(structure)
    atoms = AseAtomsAdaptor.get_atoms(structure)
    atoms.calc = calculator
    # References stay unrelaxed; this is the distribution of the test structures themselves.
    optimize = relax and 'job' in task
    result = {'relaxation_success': False, 'cell_relaxed': False}
    if optimize:
        moving_cell = relax_cell and task['kind'] == 'space'
        target = FrechetCellFilter(atoms) if moving_cell else atoms
        converged = bool(FIRE(target, logfile=None).run(fmax=fmax, steps=steps))
        result.update(relaxation_success=converged, cell_relaxed=moving_cell)
        destination = output/f'relaxed_{task["job"]}_{task["index"]:05d}.cif'
        AseAtomsAdaptor.get_structure(atoms).to(filename=str(destination))
        result['relaxed_cif'] = str(destination.relative_to(ROOT)).replace('\\', '/')
        result['relaxed_cif_sha256'] = file_hash(destination)
    total_energy = float(atoms.get_potential_energy())
    if not np.isfinite(total_energy):
        raise ValueError('Nonfinite independent energy')
    result['total_energy_eV'] = total_energy
    result['energy_per_atom_eV'] = total_energy/len(atoms)
    # An ASE plugin can also provide independently predicted scalar properties.
    for name in ('formation_energy', 'band_gap'):
        value = calculator.results.get(name)
        if value is not None and np.ndim(value) == 0 and np.isfinite(value):
            result[name] = float(value)
    if energy_reference:
        counts = structure.composition.get_el_amt_dict()
        elemental = energy_reference['elemental_eV_per_atom']
        if set(counts) <= set(elemental):
            result['formation_energy'] = (total_energy-sum(counts[e]*elemental[e] for e in counts))/len(atoms)
        # A compatible reference phase set, rather than the generated batch, defines the hull.
        if energy_reference.get('phase_entries'):
            from pymatgen.analysis.phase_diagram import PhaseDiagram, PDEntry
            phases = [PDEntry(e['composition'], e['energy_eV']) for e in energy_reference['phase_entries']
                      if set(e['composition']) <= set(counts)]
            if phases:
                result['energy_above_hull'] = float(PhaseDiagram(phases).get_e_above_hull(
                    PDEntry(counts, total_energy), allow_negative=True))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', help='Export a completed evaluate_joint run for external assessment')
    source.add_argument('--tasks', help='Evaluate an exported tasks.json')
    parser.add_argument('--reference-config', help='Also export all test reference CIFs')
    parser.add_argument('--calculator-factory', help='Installed module:callable returning an ASE calculator')
    parser.add_argument('--calculator-kwargs', default='{}', help='JSON factory arguments, e.g. a local model path')
    parser.add_argument('--evaluator', help='Versioned evaluator/model identifier, including training/calibration scope')
    parser.add_argument('--energy-reference', help='JSON compatible elemental energies and optional phase_entries')
    parser.add_argument('--relax', action='store_true')
    parser.add_argument('--relax-cell', action='store_true', help='3D only; 2D cell is held fixed')
    parser.add_argument('--fmax', type=float, default=.05)
    parser.add_argument('--steps', type=int, default=300)
    parser.add_argument('--output', required=True)
    parser.add_argument('--resume', action='store_true', help='Continue an unchanged assessment from its durable journal')
    args = parser.parse_args()
    output = artifact_path(args.output)
    if output.exists() and any(output.iterdir()) and not args.resume:
        raise ValueError('Use an empty assessment output directory')
    output.mkdir(parents=True, exist_ok=True)
    if args.run:
        if args.resume:
            parser.error('Export is immutable; --resume applies only to calculator assessment')
        report = export_tasks(root_path(args.run), output,
                              root_path(args.reference_config) if args.reference_config else None)
        print(json.dumps({'exported_tasks': len(report['tasks']), 'output': str(output)}))
        return
    if not args.calculator_factory or not args.evaluator:
        parser.error('--tasks requires --calculator-factory and --evaluator')
    if args.relax_cell and not args.relax:
        parser.error('--relax-cell requires --relax')
    if args.steps < 1 or args.fmax <= 0:
        parser.error('steps and fmax must be positive')
    task_path = root_path(args.tasks)
    report = json.loads(task_path.read_text(encoding='utf-8'))
    keys = [('generated', t['job'], t['index']) if 'job' in t else ('reference', t['dataset'], t['id'])
            for t in report['tasks']]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate independent assessment task keys')
    if any(file_hash(root_path(t['cif'])) != t['sha256'] for t in report['tasks']):
        raise ValueError('Input CIF changed')
    module, name = args.calculator_factory.split(':')
    calculator_module = importlib.import_module(module)
    calculator_kwargs = json.loads(args.calculator_kwargs)
    calculator = getattr(calculator_module, name)(**calculator_kwargs)
    reference = json.loads(root_path(args.energy_reference).read_text(encoding='utf-8')) if args.energy_reference else None
    if reference and (reference.get('evaluator') != args.evaluator or not reference.get('scope')):
        raise ValueError('Energy references must specify the same evaluator and a reference scope')
    if reference and (not isinstance(reference.get('elemental_eV_per_atom'), dict) or
                      not all(np.isfinite(v) for v in reference['elemental_eV_per_atom'].values())):
        raise ValueError('Energy reference requires finite elemental_eV_per_atom values')
    metadata = {'evaluator': args.evaluator, 'reference_evaluator': args.evaluator,
        'units': {'formation_energy': 'eV/atom', 'band_gap': 'eV', 'energy_above_hull': 'eV/atom'},
        'calculator_factory': args.calculator_factory, 'calculator_kwargs': json.loads(args.calculator_kwargs),
        'calculator_module_sha256': file_hash(calculator_module.__file__),
        'calculator_files_sha256': {value: file_hash(Path(value)) for value in calculator_kwargs.values()
                                    if isinstance(value, str) and Path(value).is_file()},
        'tasks_sha256': file_hash(task_path), 'code_sha256': file_hash(Path(__file__)),
        'reference_test_sha256': report.get('reference_test_sha256'),
        'ase_version': importlib.metadata.version('ase'), 'relax': args.relax, 'relax_cell_3d_only': args.relax_cell,
        'fmax_eV_per_angstrom': args.fmax, 'maximum_relax_steps': args.steps,
        'energy_reference': {'sha256': file_hash(root_path(args.energy_reference)), 'scope': reference['scope']}
            if reference else None,
        'scope': 'Independent calculator assessment, not DFT certification. 2D uses partial PBC and fixed '
                 'in-plane cell; the calculator must support this geometry. No normal periodic copies are '
                 'silently introduced. Missing properties/hull references remain unevaluated.'}
    metadata_path, journal = output/'metadata.json', output/'assessments.jsonl'
    if args.resume:
        if json.loads(metadata_path.read_text(encoding='utf-8')) != metadata:
            raise ValueError('Assessment inputs, calculator, code or settings changed; use a new output')
    else:
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    previous = read_jsonl(journal) if journal.exists() else []
    previous_keys = [('generated', r['job'], r['index']) if 'job' in r else ('reference', r['dataset'], r['id'])
                     for r in previous]
    if previous_keys != keys[:len(previous_keys)]:
        raise ValueError('Assessment journal is not an exact completed prefix of the task list')
    for i, task in enumerate(report['tasks'][len(previous):], start=len(previous)):
        row = {key: task[key] for key in ('job', 'index', 'dataset', 'id') if key in task}
        row['input_cif_sha256'] = task['sha256']
        try:
            row.update(assess_one(task, calculator, output, args.relax, args.relax_cell, args.fmax, args.steps, reference))
        except (ValueError, RuntimeError, OSError, KeyError, NotImplementedError, np.linalg.LinAlgError) as error:
            row['assessment_error'] = str(error)
        with journal.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, allow_nan=False)+'\n')
        if (i+1) % 25 == 0 or i+1 == len(report['tasks']):
            print(json.dumps({'assessed': i+1, 'requested': len(report['tasks'])}), flush=True)
    rows = read_jsonl(journal) if journal.exists() else []
    for key, filename in [('job', 'generated.csv'), ('dataset', 'references.csv')]:
        selected = [r for r in rows if key in r]
        if selected:
            pd.DataFrame(selected).to_csv(output/filename, index=False)
    result = {'state': 'complete', 'output': str(output), 'requested': len(report['tasks']),
              'assessed': len(rows), 'errors': sum('assessment_error' in r for r in rows)}
    (output/'summary.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result))


if __name__ == '__main__':
    main()
