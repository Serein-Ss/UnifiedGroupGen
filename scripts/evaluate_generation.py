"""Offline standard generation metrics for completed shared runs or explicit sample files."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import importlib.metadata
import json
import multiprocessing
from pathlib import Path
import sys
import warnings
import numpy as np
import pandas as pd
import yaml
from pymatgen.core import Lattice, Structure

ROOT = Path(os.environ.get('UGG_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.cli import artifact_path, root_path
from unifiedgroupgen.data import read_jsonl
from unifiedgroupgen.representation_audit import compiled_chart, expand_numpy
from unifiedgroupgen.symmetry import Descriptor
from unifiedgroupgen.training import file_hash
from unifiedgroupgen.standard_metrics import (composition_validity, coverage, distribution_distances,
    external_metrics, fingerprints, geometry_metrics, layer_structure, structure_match, validity_summary,
    property_distances)


def relaxed_metrics(path, training, unique, kind):
    from pymatgen.io.cif import CifParser
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        structure = CifParser(path, frac_tolerance=0).parse_structures(primitive=False)[0]
    geometry = geometry_metrics(structure, kind)
    chemistry, _ = composition_validity(structure)
    from unifiedgroupgen.native_identifier import identify_isolated
    identified = identify_isolated(layer_structure(structure) if kind == 'layer' else structure, kind)
    physical = {'relaxed_recovered_group': identified['number'],
                'relaxed_group_identifier_error': identified.get('error'),
                'relaxed_minimum_pair_distance_angstrom': geometry.get('minimum_pair_distance_angstrom')}
    formula = structure.composition.reduced_formula
    if not geometry['legacy_structural_valid'] or not chemistry:
        return {**physical, 'relaxed_joint_valid': False, 'relaxed_unique': False,
                'relaxed_novel_against_reference': False, 'relaxed_cif_sha256': file_hash(path)}
    is_unique = not any(structure_match(structure, old, kind) for old in unique[formula])
    if is_unique:
        unique[formula].append(structure)
    return {**physical, 'relaxed_joint_valid': bool(geometry['legacy_structural_valid'] and chemistry),
            'relaxed_unique': is_unique,
            'relaxed_novel_against_reference': not any(structure_match(structure, old, kind)
                                                       for old in training.get(formula, [])),
            'relaxed_cif_sha256': file_hash(path)}


def record_structure(record):
    c = compiled_chart(Descriptor.from_dict(record['descriptor']))
    lattice, coords, _ = expand_numpy(c, record['state'])
    structure = Structure(Lattice(lattice.T, pbc=tuple(c.periodic)), c.elements, coords)
    return layer_structure(structure) if c.descriptor.kind != 'space' else structure


def reference_record_metrics(payload):
    record, enabled = payload
    kind = record['descriptor']['kind']
    try:
        structure = record_structure(record)
        row = {'id': record['id'], **geometry_metrics(structure, kind)}
    except (ValueError, RuntimeError, TypeError, KeyError, ZeroDivisionError, np.linalg.LinAlgError) as error:
        return {'id': record['id'], 'error': str(error)}, None
    pair = None
    if enabled:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                pair = [item.tolist() for item in fingerprints(structure, kind)]
        except (ValueError, RuntimeError, TypeError, KeyError, ZeroDivisionError, np.linalg.LinAlgError) as error:
            row['fingerprint_error'] = str(error)
    return row, pair


def reference_fingerprints(paths, dataset, output, enabled, workers=1):
    versions = {name: importlib.metadata.version(name) for name in ('matminer', 'pymatgen', 'numpy')}
    identity = {'files': {str(path): file_hash(path) for path in paths}, 'dataset': dataset, 'versions': versions,
                'fingerprints_enabled': enabled,
                'metric_code_sha256': file_hash(ROOT/'src/unifiedgroupgen/standard_metrics.py')}
    cache = output/f'{dataset}_reference.json'
    if enabled and cache.exists():
        saved = json.loads(cache.read_text(encoding='utf-8'))
        if saved['identity'] == identity:
            return saved
    records = [r for path in paths for r in read_jsonl(path) if r['dataset'] == dataset]
    payloads = [(r, enabled) for r in records]
    rows, features = [], []
    # Evaluation follows CUDA inference. Fresh interpreters avoid inheriting
    # locks from the parent's PyTorch/OpenMP/BLAS thread pools on Linux.
    pool = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'))
            if workers > 1 else None)
    print(json.dumps({'phase': 'reference_start', 'dataset': dataset, 'requested': len(records),
                      'workers': workers, 'start_method': 'spawn' if pool else 'serial'}), flush=True)
    try:
        results = pool.map(reference_record_metrics, payloads, chunksize=4) if pool else map(reference_record_metrics, payloads)
        for count, (row, pair) in enumerate(results, 1):
            rows.append(row)
            if pair is not None:
                features.append(pair)
            if count == 1 or count % 100 == 0:
                print(json.dumps({'phase': 'reference', 'dataset': dataset, 'records': count,
                                  'requested': len(records), 'workers': workers}), flush=True)
    finally:
        if pool:
            pool.shutdown()
    saved = {'identity': identity, 'rows': rows, 'fingerprints': features, 'requested': len(records)}
    temporary = cache.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(saved, allow_nan=False), encoding='utf-8')
    temporary.replace(cache)
    return saved


def evaluate_samples(samples, references, training, kind, assessments=None, tolerances=None,
                     use_fingerprints=True, reidentify=False, structural_cutoff=0.4, composition_cutoff=10):
    records = read_jsonl(samples)
    indices = [r['index'] for r in records]
    if len(indices) != len(set(indices)):
        raise ValueError('Duplicate attempt indices in sample file')
    rows, features, unique_structures, relaxed_unique = [], [], defaultdict(list), defaultdict(list)
    requested_properties = {}
    for record in records:
        target = {key: value for key, value in record.get('properties_requested', {}).items()
                  if key != 'source_domain' and value is not None}
        if target:
            if requested_properties and target != requested_properties:
                raise ValueError('A metrics job must have fixed requested property targets')
            requested_properties = target
        row = {'index': record['index'], 'generated': record['status'] == 'generated', 'joint_valid': None}
        row.update({f'target_{key}': value for key, value in target.items()})
        row.update({f'predicted_{key}': value for key, value in (assessments or {}).get(record['index'], {}).items()
                    if key in ('formation_energy', 'band_gap', 'energy_above_hull')})
        description, validation = record.get('descriptor', {}), record.get('validation', {})
        row.update(target_kind=kind, target_group=description.get('number'),
                   coordinate_dof=validation.get('coordinate_dof'), lattice_dof=validation.get('lattice_dof'),
                   observation_max_residual=validation.get('observation_max_residual'),
                   recovered_group=validation.get(f'recovered_{kind}_group'),
                   target_group_containment=validation.get('group_operation_closure'),
                   metric_residual=validation.get('metric_residual'),
                   orbit_multiplicity_regular=validation.get('orbit_multiplicity_regular'),
                   no_coincident_orbits=validation.get('no_coincident_orbits'))
        if not row['generated']:
            row['reason'] = record.get('reason')
            rows.append(row)
            continue
        cif = samples.parent/f"sample_{record['index']:05d}.cif"
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                from pymatgen.io.cif import CifParser
                structure = CifParser(cif, frac_tolerance=0).parse_structures(primitive=False)[0]
            row.update(geometry_metrics(structure, kind))
            row['composition_valid'], row['composition_error'] = composition_validity(structure)
            row['chemical_geometry_valid'] = (row['legacy_structural_valid'] and row['composition_valid']
                                              if row['composition_valid'] is not None else None)
            row['joint_valid'] = row['chemical_geometry_valid']
            row['exact_target_group'] = record.get('validation', {}).get('exact_target_group')
            if reidentify:
                from unifiedgroupgen.native_identifier import identify_isolated
                identified = identify_isolated(layer_structure(structure) if kind != 'space' else structure, kind)
                row['recovered_group'] = identified['number']
                row['exact_target_group'] = (identified['number'] == description['number']
                    if identified['number'] is not None and description.get('number') is not None else None)
                row['identifier_error'] = identified.get('error')
            if use_fingerprints and row['legacy_structural_valid']:
                try:
                    with warnings.catch_warnings():
                        warnings.simplefilter('ignore')
                        pair = fingerprints(structure, kind)
                    features.append(pair)
                    row['fingerprint_valid'] = True
                except (ValueError, RuntimeError, TypeError, KeyError, ZeroDivisionError) as error:
                    row.update(fingerprint_valid=False, fingerprint_error=str(error), joint_valid=False)
            if row.get('joint_valid'):
                formula = structure.composition.reduced_formula
                try:
                    row['unique'] = not any(structure_match(structure, previous, kind) for previous in unique_structures[formula])
                    if row['unique']:
                        unique_structures[formula].append(structure)
                    row['novel_against_reference'] = not any(structure_match(structure, old, kind)
                                                              for old in training.get(formula, []))
                except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                    row['matching_error'] = str(error)
            row['cif_sha256'] = file_hash(cif)
            assessment = (assessments or {}).get(record['index'], {})
            if assessment.get('relaxed_cif') and assessment.get('relaxation_success'):
                try:
                    row.update(relaxed_metrics(root_path(assessment['relaxed_cif']), training, relaxed_unique, kind))
                except (ValueError, RuntimeError, OSError, np.linalg.LinAlgError) as error:
                    row['relaxed_evaluation_error'] = str(error)
        except (ValueError, RuntimeError, TypeError, KeyError, OSError, ZeroDivisionError, np.linalg.LinAlgError) as error:
            row.update(evaluation_error=str(error), joint_valid=False)
        rows.append(row)
        if len(rows) % 25 == 0:
            print(json.dumps({'phase': 'generated', 'attempted': len(rows), 'requested': len(records)}), flush=True)
    report = validity_summary(rows)
    report['kind'] = kind
    report['requested_properties'] = requested_properties
    report['protocol'] = {'fingerprints_enabled': use_fingerprints, 'structural_cutoff': structural_cutoff,
                          'composition_cutoff': composition_cutoff, 'matcher_scale': False,
                          'geometry_cutoff_angstrom': 0.5, 'hull_threshold_ev_per_atom': 0.1}
    report['fingerprint_protocol'] = ('CrystalNN-ops (61), Magpie (132), prior CGDiT scaler; '
        '2D uses h=max(24 Å, thickness+12 Å) auxiliary Voronoi closure, rejects nonzero normal-image chemical weights')
    report['valid_unique_novel_count'] = sum(bool(r.get('joint_valid') and r.get('unique')
                                                  and r.get('novel_against_reference')) for r in rows)
    report['valid_unique_novel_rate_over_all_attempts'] = report['valid_unique_novel_count']/len(rows) if rows else None
    report['coverage'] = (coverage(features, references['fingerprints'], len(rows), references['requested'],
                                  structural_cutoff, composition_cutoff) if use_fingerprints else {'status': 'disabled'})
    report['distributions'] = distribution_distances(rows, references['rows'])
    report['external'] = external_metrics(rows, requested_properties, assessments or {}, tolerances or
        {'formation_energy': 0.1, 'band_gap': 0.2, 'energy_above_hull': 0.05})
    report['scope'] = ('Legacy validity is the prior 0.5 Å / volume protocol. Joint validity additionally includes '
        'fingerprint success when enabled. Periodic-image/radius checks are separate. Uniqueness/novelty are '
        'evaluated on joint-valid structures; novelty is against the selected same-kind TRAIN references. '
        'Layer coverage/matching are adapted metrics, not published MP20 scores. External quantities require '
        'independent assessments. Symmetry uses stored generation identification unless --reidentify is used.')
    return rows, report


def plot_job(rows, references, report, output, title, kind):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 10})
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
    keys = ['legacy_structural_valid', 'composition_valid', 'joint_valid', 'exact_target_group']
    values = [report[key]['rate_over_all_attempts'] or 0 for key in keys]
    categories = ['Geometry', 'SMACT', 'Joint', 'Exact group']
    unknown = [report[key]['unknown']/report['attempted'] if report['attempted'] else 0 for key in keys]
    failed = [1-a-b if report['attempted'] else 0 for a,b in zip(values, unknown)]
    axes[0].bar(categories, values, color='#437da6', label='Passed')
    axes[0].bar(categories, failed, bottom=values, color='#e6b7a4', label='Failed check')
    axes[0].bar(categories, unknown, bottom=np.array(values)+failed, color='#d8dde2', label='Unknown / no output')
    axes[0].set_ylim(0, 1.05); axes[0].set_ylabel('Passed / all attempts')
    axes[0].legend(fontsize=6, loc='lower center')
    for i, key in enumerate(keys):
        axes[0].text(i, values[i]+.025, f"{report[key]['passed']}/{report['attempted']}", ha='center', fontsize=8)
    for axis, name, label in zip(axes[1:],
        ['density_g_cm3' if kind == 'space' else 'area_per_atom_angstrom2', 'num_elements'],
        ['Density (g/cm³)' if kind == 'space' else 'Area per atom (Å²)', 'Number of elements']):
        for source, label_source, color in ((rows, 'Generated (joint valid)', '#437da6'),
                                           (references['rows'], 'Reference test', '#d68939')):
            values = np.sort([r[name] for r in source if r.get(name) is not None
                              and (source is not rows or r.get('joint_valid'))])
            if len(values):
                axis.step(np.r_[values[0], values], np.r_[0, np.arange(1, len(values)+1)/len(values)],
                          where='post', label=label_source, color=color)
        axis.set_ylim(0, 1.02)
        axis.set_xlabel(label); axis.set_ylabel('Cumulative fraction'); axis.legend(fontsize=8)
    fig.suptitle(f'{title}: {report["attempted"]} attempts', fontsize=12)
    fig.savefig(output, dpi=220); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument('--run', help='Completed evaluate_joint output directory')
    source.add_argument('--samples', help='Explicit samples.jsonl; references are user-selected, not checkpoint-bound')
    parser.add_argument('--dataset', choices=['mp_20', 'mpts_52', 'c2db_51'])
    parser.add_argument('--reference-config', default='configs/joint_space_layer_large.yaml')
    parser.add_argument('--checkpoint', default='artifacts/runs/joint_space_layer_large_seed42/model.best.pt')
    parser.add_argument('--output', default='artifacts/evaluations/standard_metrics')
    parser.add_argument('--figures', default='assets/generation_metrics')
    parser.add_argument('--no-fingerprints', action='store_true', help='Diagnostic only; no COV or fingerprint validity')
    parser.add_argument('--reidentify', action='store_true')
    parser.add_argument('--assessments', help='Independent per-structure CSV: job,index,property values,relaxation_success')
    parser.add_argument('--assessment-metadata', help='JSON provenance: evaluator, units, energy_reference for hull values')
    parser.add_argument('--reference-assessments', help='Same evaluator on TEST references: dataset,id,property values CSV')
    parser.add_argument('--structural-cutoff', type=float, default=0.4)
    parser.add_argument('--composition-cutoff', type=float, default=10)
    parser.add_argument('--reference-workers', type=int, default=1, help='CPU processes for reference metrics; ordered results and identical definitions')
    args = parser.parse_args()
    if args.reference_workers < 1:
        parser.error('--reference-workers must be positive')
    if args.samples and not args.dataset:
        parser.error('--samples requires --dataset')
    if bool(args.assessments) != bool(args.assessment_metadata):
        parser.error('Independent assessments and their provenance metadata must be provided together')
    if args.reference_assessments and not args.assessments:
        parser.error('Reference predictions require generated predictions and common evaluator metadata')
    config = yaml.safe_load(root_path(args.reference_config).read_text(encoding='utf-8'))
    output, figures = artifact_path(args.output), artifact_path(args.figures)
    output.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    train_paths = [root_path(path) for path in config['train_files']]
    test_paths = [root_path(path) for path in config['test_files']]
    jobs, identity, sampling_settings = [], 'user_selected_references', {}
    if args.run:
        run = root_path(args.run)
        state = json.loads((run/'pipeline.json').read_text(encoding='utf-8'))
        if state['state'] != 'complete':
            raise ValueError('Wait for evaluate_joint to complete before postprocessing the complete run')
        import torch
        checkpoint = root_path(args.checkpoint)
        if file_hash(checkpoint) != state['identity']['checkpoint_sha256']:
            raise ValueError('Checkpoint does not match generation run')
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if [file_hash(path) for path in train_paths] != saved['train_split_sha256']:
            raise ValueError('Novelty training references differ from the actual checkpoint training inputs')
        if [file_hash(path) for path in test_paths] != state['identity']['test_sha256']:
            raise ValueError('Reference tests differ from the generation run')
        del saved
        identity = 'checkpoint_and_test_bound'
        for name, job in state['jobs'].items():
            if name == 'test':
                continue
            directory = run/job['directory']
            for file, checksum in job['output_sha256'].items():
                if file_hash(directory/file) != checksum:
                    raise ValueError(f'Generation artifact changed: {name}/{file}')
            dataset = next(d for d in config['dataset_domains'] if name.startswith(d+'_'))
            jobs.append((name, dataset, directory/'samples.jsonl'))
            command = job.get('command', [])
            sampling_settings[name] = {key: command[command.index('--'+key)+1] for key in
                ('seed', 'guidance', 'steps') if '--'+key in command}
            sampling_settings[name]['prior'] = '--prior' in command
            sampling_settings[name]['fixed_descriptor'] = '--descriptor' in command
    else:
        jobs = [(args.dataset+'_explicit', args.dataset, root_path(args.samples))]
    assessments, provenance = defaultdict(dict), None
    property_tolerances = None
    if args.assessments:
        provenance = json.loads(root_path(args.assessment_metadata).read_text(encoding='utf-8'))
        if not provenance.get('evaluator') or not provenance.get('units'):
            raise ValueError('Assessment metadata requires evaluator and units')
        property_tolerances = provenance.get('property_tolerances')
        if property_tolerances is not None and (set(property_tolerances) !=
                {'formation_energy', 'band_gap', 'energy_above_hull'} or
                any(not np.isfinite(v) or v <= 0 for v in property_tolerances.values())):
            raise ValueError('Property tolerances must specify all three finite positive thresholds')
        values = pd.read_csv(root_path(args.assessments))
        if 'energy_above_hull' in values and not provenance.get('energy_reference'):
            raise ValueError('Hull assessments require an explicit compatible energy reference')
        if values.duplicated(['job', 'index']).any():
            raise ValueError('Duplicate independent assessment keys')
        attempts = {name: {r['index'] for r in read_jsonl(path)} for name, _, path in jobs}
        sample_folders = {name: path.parent for name, _, path in jobs}
        for _, row in values.iterrows():
            if row['job'] not in attempts or int(row['index']) not in attempts[row['job']]:
                raise ValueError('Assessment key does not correspond to a requested generation attempt')
            if pd.notna(row.get('input_cif_sha256')):
                cif = sample_folders[row['job']]/f"sample_{int(row['index']):05d}.cif"
                if file_hash(cif) != row['input_cif_sha256']:
                    raise ValueError('Assessment input CIF differs from the evaluated sample')
            item = {key: float(row[key]) for key in ('formation_energy', 'band_gap', 'energy_above_hull')
                    if key in row and pd.notna(row[key]) and np.isfinite(row[key])}
            item['relaxation_success'] = str(row.get('relaxation_success', '')).lower() in ('true', '1', '1.0')
            if 'relaxed_cif' in row and pd.notna(row['relaxed_cif']):
                item['relaxed_cif'] = str(row['relaxed_cif'])
                if pd.notna(row.get('relaxed_cif_sha256')) and file_hash(root_path(item['relaxed_cif'])) != row['relaxed_cif_sha256']:
                    raise ValueError('Relaxed CIF changed after independent assessment')
            assessments[row['job']][int(row['index'])] = item
    reference_predictions = defaultdict(dict)
    if args.reference_assessments:
        if provenance.get('reference_evaluator') != provenance['evaluator']:
            raise ValueError('Property Wasserstein distances require the same generated/reference evaluator')
        if provenance.get('reference_test_sha256') is not None and provenance['reference_test_sha256'] != [file_hash(p) for p in test_paths]:
            raise ValueError('Independent reference predictions used a different test split')
        predictions = pd.read_csv(root_path(args.reference_assessments), dtype={'id': str})
        if predictions.duplicated(['dataset', 'id']).any():
            raise ValueError('Duplicate test reference prediction keys')
        valid_keys = {(r['dataset'], str(r['id'])) for path in test_paths for r in read_jsonl(path)}
        for _, row in predictions.iterrows():
            if (row['dataset'], row['id']) not in valid_keys:
                raise ValueError('Reference prediction does not correspond to a test record')
            reference_predictions[row['dataset']][row['id']] = {
                name: float(row[name]) for name in ('formation_energy', 'band_gap', 'energy_above_hull')
                if name in row and pd.notna(row[name]) and np.isfinite(row[name])}
    training = {'space': defaultdict(list), 'layer': defaultdict(list)}
    train_failures = []
    required_kinds = {'layer' if dataset == 'c2db_51' else 'space' for _, dataset, _ in jobs}
    for path in train_paths:
        for record in read_jsonl(path):
            kind = record['descriptor']['kind']
            if kind not in required_kinds:
                continue
            try:
                structure = record_structure(record)
                training[kind][structure.composition.reduced_formula].append(structure)
            except (ValueError, RuntimeError, np.linalg.LinAlgError) as error:
                train_failures.append({'dataset': record['dataset'], 'id': record['id'], 'error': str(error)})
    if train_failures:
        (output/'reference_failures.json').write_text(json.dumps(train_failures, indent=2))
        raise ValueError('Novelty references could not all be expanded; refusing incomplete novelty evaluation')
    reports = {'state': 'running', 'reference_identity': identity, 'train_sha256': [file_hash(p) for p in train_paths],
               'checkpoint_sha256': file_hash(root_path(args.checkpoint)) if args.run else None,
               'generation_pipeline_sha256': file_hash(root_path(args.run)/'pipeline.json') if args.run else None,
               'evaluation_code_sha256': {path.name: file_hash(path) for path in
                   (Path(__file__), ROOT/'src/unifiedgroupgen/standard_metrics.py',
                    ROOT/'src/unifiedgroupgen/representation_audit.py', ROOT/'src/unifiedgroupgen/benchmark_comp_scaler.json')},
               'test_sha256': [file_hash(p) for p in test_paths], 'assessment_provenance': provenance,
               'assessments_sha256': file_hash(root_path(args.assessments)) if args.assessments else None,
               'reference_assessments_sha256': file_hash(root_path(args.reference_assessments)) if args.reference_assessments else None,
               'versions': {n: importlib.metadata.version(n) for n in ('smact', 'matminer', 'pymatgen', 'numpy')},
               'jobs': {}}
    references = {}
    for name, dataset, samples in jobs:
        kind = 'layer' if dataset == 'c2db_51' else 'space'
        print(f'[START] {name}', flush=True)
        if dataset not in references:
            references[dataset] = reference_fingerprints(test_paths, dataset, output, not args.no_fingerprints,
                                                        args.reference_workers)
        rows, report = evaluate_samples(samples, references[dataset], training[kind], kind, assessments[name],
            tolerances=property_tolerances,
            use_fingerprints=not args.no_fingerprints, reidentify=args.reidentify,
            structural_cutoff=args.structural_cutoff, composition_cutoff=args.composition_cutoff)
        report['samples_sha256'] = file_hash(samples)
        report['dataset'] = dataset
        report['sampling'] = sampling_settings.get(name, {})
        status_path = samples.parent/'sampling.status.json'
        if status_path.exists():
            timing = json.loads(status_path.read_text(encoding='utf-8'))
            if timing.get('state') == 'complete' and timing.get('attempted') == report['attempted']:
                report['sampling']['wall_seconds'] = timing['seconds']
                report['sampling']['seconds_per_attempt'] = timing['seconds']/report['attempted']
        report['property_distributions'] = property_distances(rows, assessments[name], reference_predictions[dataset],
                                                             references[dataset]['requested'])
        reports['jobs'][name] = report
        pd.DataFrame(rows).to_csv(output/f'{name}.csv', index=False)
        report['metrics_csv_sha256'] = file_hash(output/f'{name}.csv')
        grouped = pd.DataFrame(rows).groupby('target_group', dropna=False).agg(
            attempted=('index', 'count'), generated=('generated', 'sum'),
            joint_valid=('joint_valid', lambda x: int(x.fillna(False).astype(bool).sum())),
            joint_valid_unknown=('joint_valid', lambda x: int(x.isna().sum())))
        grouped.to_csv(output/f'{name}_groups.csv')
        (output/f'{name}.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        plot_job(rows, references[dataset], report, figures/f'{name}.png', name, kind)
        (output/'summary.json').write_text(json.dumps(reports, indent=2, allow_nan=False), encoding='utf-8')
        print(f'[DONE] {name}', flush=True)
    table = [{'job': name, 'attempted': report['attempted'], 'generated': report['generated'],
              **{key: report[key]['rate_over_all_attempts'] for key in ('legacy_structural_valid', 'composition_valid',
                    'joint_valid', 'exact_target_group', 'unique', 'novel_against_reference')},
              'cov_recall': report['coverage'].get('cov_recall'), 'cov_precision': report['coverage'].get('cov_precision')}
             for name, report in reports['jobs'].items()]
    pd.DataFrame(table).to_csv(output/'summary.csv', index=False)
    reports['state'] = 'complete'
    (output/'summary.json').write_text(json.dumps(reports, indent=2, allow_nan=False), encoding='utf-8')
    print(f'[COMPLETE] {output}/summary.json', flush=True)


if __name__ == '__main__':
    main()
