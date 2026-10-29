"""Unified model-independent raw CSP/DNG evaluation. Old entry points are unchanged."""
import argparse
import csv
import json
from pathlib import Path
import sys
import torch  # initialize the project's OpenMP runtime before numpy/matminer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from unifiedgroupgen.evaluation_io import read_records, sha256, write_json
from unifiedgroupgen.evaluation_reporting import figures
from unifiedgroupgen.unified_evaluation import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=Path, required=True)
    parser.add_argument('--task', choices=['csp','dng'], required=True)
    parser.add_argument('--kind', choices=['space','layer'], required=True)
    parser.add_argument('--dataset')
    parser.add_argument('--reference-files', type=Path, nargs='*', default=[])
    parser.add_argument('--novelty-files', type=Path, nargs='*', default=[])
    parser.add_argument('--request', type=Path)
    parser.add_argument('--assessments', type=Path)
    parser.add_argument('--assessment-metadata', type=Path)
    parser.add_argument('--reference-assessments', type=Path)
    parser.add_argument('--embeddings', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cache', type=Path)
    parser.add_argument('--expected-count', '--expected-attempts', dest='expected_attempts', type=int)
    parser.add_argument('--no-fingerprints', action='store_true')
    parser.add_argument('--no-reidentify', action='store_true')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if bool(args.assessments)!=bool(args.assessment_metadata):
        parser.error('Assessments CSV and evaluator metadata must be supplied together')
    inputs = [args.samples, *args.reference_files, *args.novelty_files,
              *[p for p in (args.request,args.assessments,args.assessment_metadata,args.reference_assessments,args.embeddings) if p]]
    records = read_records(args.samples)
    for r in records:
        if 'cif' in r:
            continue
        if r.get('cif_path'):
            p=Path(r['cif_path']); inputs.append(p if p.is_absolute() else args.samples.parent/p)
        elif r.get('status')=='generated' and (args.samples.parent/f"sample_{r['index']:05d}.cif").exists():
            inputs.append(args.samples.parent/f"sample_{r['index']:05d}.cif")
    code = [Path(__file__), *sorted((ROOT/'src/unifiedgroupgen').glob('evaluation_*.py')),
            *[ROOT/'src/unifiedgroupgen'/n for n in ('unified_evaluation.py','standard_metrics.py','symmetry.py','representation_audit.py')]]
    identity = {'inputs':{str(p.resolve()):sha256(p) for p in inputs},
                'code':{str(p.relative_to(ROOT)):sha256(p) for p in code},
                'options':{k:v for k,v in vars(args).items() if k not in ('resume','output','cache') and not isinstance(v,Path) and k not in ('reference_files','novelty_files')}}
    manifest = args.output/'manifest.json'
    if manifest.exists():
        saved=json.loads(manifest.read_text(encoding='utf-8'))
        if not args.resume or saved['identity']!=identity:
            raise ValueError('Existing output has another identity, or --resume was not supplied; use a new directory')
        if all((args.output/p).exists() and sha256(args.output/p)==h for p,h in saved['outputs'].items()):
            print('Verified existing evaluation; no work repeated.'); return
        raise ValueError('Completed output was modified or removed; preserve it and use a new output directory')
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
    if args.assessments and metadata.get('predictions_sha256') and metadata['predictions_sha256']!=sha256(args.assessments):
        raise ValueError('Prediction CSV hash mismatch')
    if args.reference_assessments and metadata.get('reference_predictions_sha256')!=sha256(args.reference_assessments):
        raise ValueError('Reference prediction CSV hash mismatch')
    request=json.loads(args.request.read_text(encoding='utf-8')) if args.request else {}
    rows,report,refs=evaluate(args.samples,args.task,args.kind,args.output,args.reference_files,args.novelty_files,
        args.dataset,request,predictions,metadata,not args.no_fingerprints,not args.no_reidentify,args.cache,args.expected_attempts,args.embeddings,reference_predictions)
    for row in refs:
        row.update({'predicted_'+p:v for p,v in reference_predictions.get(row['id'],{}).items()})
    figures(rows,report,refs,args.output)
    outputs=[p for p in args.output.iterdir() if p.suffix in ('.csv','.json','.png') and p.name!='manifest.json']
    write_json(manifest,{'identity':identity,'outputs':{p.name:sha256(p) for p in outputs},'state':'complete'})
    print(json.dumps({'state':'complete','requested':len(rows),'generated':report['generation']['generated'],'output':str(args.output)},ensure_ascii=False))


if __name__=='__main__':
    main()
