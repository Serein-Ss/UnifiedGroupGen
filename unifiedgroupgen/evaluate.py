"""Calculate metrics from existing structures; this entry never samples structures."""
from pathlib import Path
import sys
if __package__ in (None,''):
    sys.path[0]=str(Path(__file__).resolve().parent.parent)
import csv
import hashlib
import importlib.metadata
import json
import time
import hydra
import torch
import numpy as np
from filelock import FileLock
from omegaconf import OmegaConf
from unifiedgroupgen.common.configuration import project_root, resolve_path, source_key, task_options, evaluation_directory
from unifiedgroupgen.common.structures import ReferenceBank, counts_of, load_structure, read_records, sha256, write_json
from unifiedgroupgen.evaluation.common import EvaluationContext, evaluate_records, build_summary, geometry, descriptor_metadata, operation_check
from unifiedgroupgen.evaluation.utils import numeric, logical_and, composition_validity, fingerprints
from unifiedgroupgen.evaluation.conditional import evaluate_properties
from unifiedgroupgen.common.reports import write_tables, figures
ROOT=project_root()

def metric_sources():
    """Identity of metric implementation, independent of sampling/training entry changes."""
    package=Path(__file__).parent
    return [Path(__file__),*(package/'evaluation').glob('*.py'),
            *(package/'evaluation/resources').glob('*.json'),
            package/'common/structures.py',package/'common/reports.py',
            package/'group_constraints/compiler.py',package/'group_constraints/encoding.py',
            package/'tests/representation.py']


def metric_versions():
    return {name:importlib.metadata.version(name) for name in
            ('numpy','scipy','pymatgen','matminer','smact','pyxtal','spglib')}

def reference_metrics(bank, kind, use_fingerprints, cache):
    versions = {name: importlib.metadata.version(name) for name in ('numpy', 'scipy', 'pymatgen', 'matminer', 'smact', 'pyxtal', 'spglib')}
    identity = {'files': bank.identity(), 'kind': kind, 'fingerprints': use_fingerprints, 'versions': versions,
        'code': {Path(p).name: sha256(p) for p in (
            Path(__file__), Path(__file__).parent/'evaluation/common.py', Path(__file__).parent/'evaluation/utils.py', Path(__file__).parent/'common/structures.py',
            Path(__file__).parent/'group_constraints/compiler.py',
            Path(__file__).parent/'tests/representation.py')}}
    key = __import__('hashlib').sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    path = Path(cache)/(key+'.json')
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path)+'.lock'):
        return _reference_metrics_locked(bank, kind, use_fingerprints, path, identity)


def _reference_metrics_locked(bank, kind, use_fingerprints, path, identity):
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
             cache=None, expected_attempts=None, embeddings=None, reference_assessments=None, *, conditional_enabled=False):
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
    properties = evaluate_properties(rows, predictions, metadata, tolerances, reference_assessments, enabled=conditional_enabled)
    embedding_data = json.loads(Path(embeddings).read_text(encoding='utf-8')) if embeddings else None
    report = build_summary(rows, features, reference, properties, context, embedding_data)
    elapsed = time.perf_counter()-started
    generation_times = [r['generation_seconds'] for r in rows if r.get('generation_seconds') is not None]
    report['efficiency'] = {'metric_seconds_excluding_reference_cache':elapsed,'generation_seconds':numeric(generation_times),
        'generation_time_coverage':len(generation_times),'generated_structures_per_second':report['generation']['generated']/sum(generation_times)
            if len(generation_times)==len(rows) and sum(generation_times)>0 else None}
    write_tables(output, rows, report, kind)
    return rows,report,reference['rows']


def evaluate_task(args):
    if not args.conditional_enabled and (args.assessments or args.assessment_metadata or args.reference_assessments):
        raise ValueError('Enable evaluation.conditional.enabled for property predictions')
    if bool(args.assessments)!=bool(args.assessment_metadata):
        raise ValueError('Assessments CSV and evaluator metadata must be supplied together')
    inputs = [args.samples, *args.reference_files, *args.novelty_files,
              *[p for p in (args.request,args.assessments,args.assessment_metadata,args.reference_assessments,args.embeddings) if isinstance(p,Path)]]
    records = read_records(args.samples)
    for r in records:
        if 'cif' in r:
            continue
        if r.get('cif_path'):
            p=Path(r['cif_path']); inputs.append(p if p.is_absolute() else args.samples.parent/p)
        elif r.get('status')=='generated' and (args.samples.parent/f"sample_{r['index']:05d}.cif").exists():
            inputs.append(args.samples.parent/f"sample_{r['index']:05d}.cif")
    code = metric_sources()
    identity = {'inputs':{str(p.resolve()):sha256(p) for p in inputs},
                'code':{source_key(p,ROOT):sha256(p) for p in code},
                'versions':metric_versions(),
                'options':{k:v for k,v in vars(args).items() if k not in ('output','cache') and not isinstance(v,Path) and k not in ('reference_files','novelty_files')}}
    manifest = args.output/'manifest.json'
    if manifest.exists():
        raise ValueError('Completed metrics already exist. The campaign may verify and skip them; choose a new output directory to recompute.')
    def prediction_csv(path,index_key):
        predictions={}
        if path is None: return predictions
        with path.open(encoding='utf-8-sig',newline='') as stream:
            for row in csv.DictReader(stream):
                index=row.pop(index_key)
                if index in predictions:
                    raise ValueError('Duplicate prediction index')
                predictions[index]={}
                for name,value in row.items():
                    if name=='prediction_error':
                        predictions[index][name]=value
                        continue
                    try: predictions[index][name]=float(value) if value else None
                    except ValueError: pass
        return predictions
    predictions=prediction_csv(args.assessments,'index')
    reference_predictions=prediction_csv(args.reference_assessments,'id')
    if set(predictions)-{str(r['index']) for r in records}:
        raise ValueError('Predictions contain indices outside the attempt ledger')
    metadata=json.loads(args.assessment_metadata.read_text(encoding='utf-8')) if args.assessment_metadata else {}
    for name, expected in metadata.get('structure_files_sha256', {}).items():
        actual = sha256(Path(name)) if Path(name).is_file() else None
        if actual != expected: raise ValueError('Property predictor input CIF changed: '+name)
    if args.assessments and metadata.get('predictions_sha256') and metadata['predictions_sha256']!=sha256(args.assessments):
        raise ValueError('Prediction CSV hash mismatch')
    if args.reference_assessments and metadata.get('reference_predictions_sha256')!=sha256(args.reference_assessments):
        raise ValueError('Reference prediction CSV hash mismatch')
    request=args.request if isinstance(args.request,dict) else json.loads(args.request.read_text(encoding='utf-8')) if args.request else {}
    rows,report,refs=evaluate(args.samples,args.task,args.kind,args.output,args.reference_files,args.novelty_files,
        args.dataset,request,predictions,metadata,not args.no_fingerprints,not args.no_reidentify,args.cache,args.expected_attempts,args.embeddings,reference_predictions,conditional_enabled=args.conditional_enabled)
    for row in refs:
        row.update({'predicted_'+p:v for p,v in reference_predictions.get(row['id'],{}).items()})
    figures(rows,report,refs,args.output)
    outputs=[p for p in args.output.iterdir() if p.suffix in ('.csv','.json','.png') and p.name not in ('manifest.json','metrics.status.json')]
    write_json(manifest,{'identity':identity,'outputs':{p.name:sha256(p) for p in outputs},'state':'complete'})
    print(json.dumps({'state':'complete','requested':len(rows),'generated':report['generation']['generated'],'output':str(args.output)},ensure_ascii=False))


def execute(cfg):
    torch.set_num_threads(cfg.get('threads',2))
    if cfg.task=='predict_properties':
        from unifiedgroupgen.common.predictors.service import predict
        return predict(task_options(cfg,'property_predictor'))
    if cfg.task!='evaluate':
        raise ValueError('evaluate.py accepts evaluate or predict_properties; generation uses generate.py')
    args=task_options(cfg,'evaluation')
    args.conditional_enabled=args.conditional['enabled']; del args.conditional
    if args.samples is None:
        raise ValueError('evaluation.samples is required')
    if args.output is None:
        args.output = evaluation_directory(args.samples)
    if (args.output/'manifest.json').exists():
        raise ValueError('Completed metrics already exist; use the campaign to skip them or choose a new output directory')
    args.output.mkdir(parents=True,exist_ok=True)
    status=args.output/'metrics.status.json'
    write_json(status,{'state':'running','pid':__import__('os').getpid(),'started':time.time(),
                       'samples':str(args.samples),'scope':'Metrics only; no structure sampling'})
    started=time.perf_counter()
    try:
        if args.conditional_enabled and cfg.get('property_predictor',{}).get('enabled',False):
            from unifiedgroupgen.common.predictors.service import predict
            predictor=task_options(cfg,'property_predictor')
            predictor.samples=args.samples; predictor.reference_files=args.reference_files
            predictor.dataset=args.dataset; predictor.kind=args.kind
            predictor.output=args.output/'properties'
            receipt=predictor.output/'evaluator.json'
            if receipt.exists():
                metadata=json.loads(receipt.read_text(encoding='utf-8'))
                if (metadata.get('samples_sha256')!=sha256(args.samples) or
                    metadata.get('predictions_sha256')!=sha256(predictor.output/'predictions.csv') or
                    metadata.get('reference_predictions_sha256')!=sha256(predictor.output/'reference_predictions.csv')):
                    raise ValueError('Existing property predictions changed; preserve them and select a new output')
            else:
                predict(predictor)
            args.assessments=predictor.output/'predictions.csv'
            args.assessment_metadata=receipt
            args.reference_assessments=predictor.output/'reference_predictions.csv'
        result=evaluate_task(args)
    except BaseException as error:
        write_json(status,{'state':'failed','seconds':time.perf_counter()-started,'error':str(error)})
        raise
    write_json(status,{'state':'complete','seconds':time.perf_counter()-started})
    return result


@hydra.main(version_base='1.3',config_path=str(project_root()/'configs'),config_name='evaluate')
def main(cfg):
    execute(cfg)


if __name__=='__main__': main()
