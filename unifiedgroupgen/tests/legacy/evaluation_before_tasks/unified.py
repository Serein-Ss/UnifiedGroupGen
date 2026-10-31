"""Raw-structure evaluation shared by CSP and de novo generation (no calculators)."""
from collections import defaultdict
import importlib.metadata
import json
from pathlib import Path
import time
import warnings

import numpy as np

from unifiedgroupgen.tests.legacy.evaluation_before_tasks.io import ReferenceBank, counts_of, formula_key, load_structure, read_records, sha256, write_json
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.context import EvaluationContext
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.structure_checks import (geometry, identify, operation_check, descriptor_metadata,
    check_declared_orbits, check_requested_chemistry, check_symmetry, check_reconstruction)
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.summaries import PROTOCOL, BOOL_FIELDS, csp_summary, build_summary
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.reporting import write_tables
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.properties import evaluate_properties
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.statistics import numeric, logical_and
from unifiedgroupgen.tests.legacy.evaluation_before_tasks.standard import composition_validity, fingerprints, structure_match


def reference_metrics(bank, kind, use_fingerprints, cache):
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pymatgen', 'matminer', 'smact', 'pyxtal', 'spglib')}
    identity = {'files': bank.identity(), 'kind': kind, 'fingerprints': use_fingerprints, 'versions': versions,
        'code': {Path(p).name: sha256(p) for p in (
            Path(__file__), Path(__file__).with_name('structure_checks.py'), Path(__file__).with_name('standard.py'), Path(__file__).with_name('io.py'),
            Path(__file__).parents[3]/'group_constraints/compiler.py',
            Path(__file__).parents[3]/'tests/representation.py')}}
    key = __import__('hashlib').sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache)/(key+'.json')
    if path.exists():
        saved = json.loads(path.read_text(encoding='utf-8'))
        if saved['identity'] != identity:
            raise ValueError('Reference cache identity mismatch')
        return saved
    rows, features = [], []
    for index, (record, base) in enumerate(bank.rows):
        row = {'id': str(record.get('id', index)), **descriptor_metadata(record, kind)}
        row['recovered_group'] = row.get('declared_group')  # Audited reference label, explicitly identified in protocol.
        try:
            structure = load_structure(record, base)
            row.update(geometry(structure, kind))
            row['elements'] = sorted(counts_of(structure))
            row['chemical_valid'], row['chemical_error'] = composition_validity(structure)
            row['basic_valid'] = logical_and(row['periodic_distance_valid'], row['chemical_valid'])
            if use_fingerprints:
                pair = fingerprints(structure, kind)
                features.append([p.tolist() for p in pair])
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, ZeroDivisionError, np.linalg.LinAlgError) as error:
            row['evaluation_error'] = str(error)
        rows.append(row)
        if (index+1)%100==0:
            print(json.dumps({'phase': 'reference', 'completed': index+1, 'requested': len(bank.rows)}), flush=True)
    saved = {'identity': identity, 'rows': rows, 'fingerprints': features}
    write_json(path, saved)
    return saved


def evaluate(samples, task, kind, output, reference_files=(), novelty_files=(), dataset=None, request=None,
             assessments=None, assessment_metadata=None, use_fingerprints=True, reidentify=True,
             cache=None, expected_attempts=None, embeddings=None, reference_assessments=None):
    if task not in ('csp', 'dng') or kind not in ('space', 'layer'):
        raise ValueError('task=csp/dng and kind=space/layer are required')
    samples, output = Path(samples).resolve(), Path(output).resolve()
    request = request or {}
    records = read_records(samples)
    if not records or len({r['index'] for r in records}) != len(records):
        raise ValueError('A nonempty ledger with unique attempt indices is required')
    if expected_attempts is not None and len(records)!=expected_attempts:
        raise ValueError('Attempt count differs from the requested budget')
    refs, train = ReferenceBank(reference_files, dataset), ReferenceBank(novelty_files, dataset)
    predictions, metadata = assessments or {}, assessment_metadata or {}
    if predictions and (not metadata.get('evaluator') or metadata.get('samples_sha256')!=sha256(samples)):
        raise ValueError('Independent assessments require evaluator identity and matching samples_sha256')
    if reference_assessments and metadata.get('reference_sha256')!=refs.identity():
        raise ValueError('Reference predictions do not bind the exact reference files')
    reference = reference_metrics(refs, kind, use_fingerprints, cache or output/'reference_cache') if refs.rows else {'rows': [], 'fingerprints': []}
    if task=='csp' and any(str(r.get('case_id', r.get('reference_id'))) not in refs.by_id for r in records):
        raise ValueError('Each CSP attempt must name an existing reference case, including failures')
    context = EvaluationContext(task, kind, dataset, request, refs, train, use_fingerprints, reidentify, novelty_files, samples)
    tolerances = request.get('property_tolerances', {'formation_energy': .1, 'band_gap': .2, 'energy_above_hull': .05})
    started = time.perf_counter()
    rows, features = evaluate_records(records, context)
    properties = evaluate_properties(rows, predictions, metadata, tolerances, reference_assessments)
    report = build_summary(rows, features, reference, properties, context, embeddings)
    elapsed = time.perf_counter()-started
    generation_times = [r['generation_seconds'] for r in rows if r.get('generation_seconds') is not None]
    report['efficiency'] = {'metric_seconds_excluding_reference_cache':elapsed,'generation_seconds':numeric(generation_times),
        'generation_time_coverage':len(generation_times),'generated_structures_per_second':report['generation']['generated']/sum(generation_times)
            if len(generation_times)==len(rows) and sum(generation_times)>0 else None}
    write_tables(output, rows, report, kind)
    return rows,report,reference['rows']


def evaluate_records(records, context):
    task, kind, dataset, request = context.task, context.kind, context.dataset, context.request
    refs, train, use_fingerprints = context.refs, context.train, context.use_fingerprints
    reidentify, novelty_files, samples = context.reidentify, context.novelty_files, context.samples
    rows, features, unique = [], [], defaultdict(list)
    for record in records:
        index = record['index']
        row = {'index': index, 'case_id': str(record.get('case_id', record.get('reference_id'))) if record.get('case_id', record.get('reference_id')) is not None else None,
               'generation_success': record.get('status', 'generated')=='generated', 'parse_success': None,
               'task': task, 'dataset': dataset, 'kind': kind, 'basic_valid': None,
               'generation_seconds': record.get('wall_seconds'), **descriptor_metadata(record, kind)}
        reference_record = refs.by_id.get(row['case_id']) if row['case_id'] is not None else None
        if reference_record:
            row['reference_coordinate_dof'] = descriptor_metadata(reference_record[0], kind).get('coordinate_dof')
        row['target_group'] = request.get('group') or record.get('requested_group') or row.get('declared_group')
        row['property_targets'] = request.get('properties', {k:v for k,v in record.get('properties_requested', {}).items() if k!='source_domain' and v is not None})
        validation = record.get('validation', {})
        for name in ('orbit_multiplicity_regular','no_coincident_orbits'):
            row[name] = validation.get(name)
        if 'observation_max_residual' in validation:
            row['observation_max_residual'] = validation['observation_max_residual']
            row['observation_satisfied'] = validation['observation_max_residual']<=request.get('observation_tolerance', 1e-5)
        if not row['generation_success']:
            row.update(basic_valid=False,legacy_joint_valid=False,unique=False)
            if task=='csp': row.update(composition_satisfied=False,formula_satisfied=False,atom_count_satisfied=False)
            if row['target_group'] is not None: row.update(exact_target_group=False,target_operation_containment=False)
            if novelty_files: row['novel_against_reference']=False
            row['failure_reason'] = record.get('reason'); rows.append(row); continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                structure = load_structure(record, samples.parent)
            row['parse_success'] = True
            row['finite'] = bool(np.isfinite(structure.lattice.matrix).all() and np.isfinite(structure.frac_coords).all())
            row['lattice_valid'] = bool(row['finite'] and abs(np.linalg.det(structure.lattice.matrix))>1e-10)
            if not row['lattice_valid']:
                row.update(generation_success=False,basic_valid=False,failure_stage='numeric_validation')
            row.update(geometry(structure, kind))
            row['chemical_valid'], row['chemical_error'] = composition_validity(structure)
            row['basic_valid'] = logical_and(row['periodic_distance_valid'], row['chemical_valid'])
            row['legacy_joint_valid'] = (bool(row['legacy_structural_valid'] and row['chemical_valid']) if row['chemical_valid'] is not None else None)
            counts = counts_of(structure)
            row['elements'] = sorted(counts)
            check_declared_orbits(row, record, structure, samples, counts)
            reference_record = check_requested_chemistry(row, record, request, refs, task, counts)
            check_symmetry(row, record, request, structure, kind, reidentify)
            check_reconstruction(row, reference_record, refs, structure, kind, task)
            if use_fingerprints:
                try:
                    pair = fingerprints(structure, kind)
                    row['fingerprint_success'] = True
                    if row.get('basic_valid'):
                        features.append([p.tolist() for p in pair])
                except (ValueError, TypeError, KeyError, RuntimeError, ZeroDivisionError, np.linalg.LinAlgError) as error:
                    row['fingerprint_success'] = False; row['fingerprint_error'] = str(error)
                row['legacy_joint_valid'] = bool(row.get('legacy_joint_valid') and row.get('fingerprint_success')) if row['chemical_valid'] is not None else None
            if row.get('basic_valid'):
                key = formula_key(counts)
                bucket = (row['case_id'], key) if task=='csp' else key
                row['unique'] = not any(structure_match(structure, old, kind) for old in unique[bucket])
                if row['unique']:
                    unique[bucket].append(structure)
                if novelty_files:
                    row['novel_against_reference'] = not any(structure_match(structure, train.structure(identifier), kind) for identifier in train.by_formula.get(key, []))
            elif row.get('basic_valid') is False:
                row['unique']=False
                if novelty_files: row['novel_against_reference']=False
        except (ValueError, TypeError, KeyError, OSError, RuntimeError, ZeroDivisionError, np.linalg.LinAlgError) as error:
            row['evaluation_error'] = str(error)
            if row['parse_success'] is None:
                row['parse_success'] = False
                row.update(generation_success=False,basic_valid=False,legacy_joint_valid=False,unique=False,
                           failure_stage='structure_readback',error_type=type(error).__name__)
        rows.append(row)
        if len(rows)%25==0:
            print(json.dumps({'phase': 'unified_metrics', 'completed':len(rows), 'requested':len(records)}), flush=True)
    return rows, features
