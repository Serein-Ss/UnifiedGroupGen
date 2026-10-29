"""Optional frozen CGDiT FE/BG adapter producing hash-bound CSV assessments on CPU."""
import argparse
import csv
import json
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from unifiedgroupgen.evaluation_io import load_structure, read_records, sha256, write_json, ReferenceBank


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--cgdit-root',type=Path,required=True)
    parser.add_argument('--predictor-root',type=Path,required=True)
    parser.add_argument('--predictor-seed',type=int,default=123)
    parser.add_argument('--allow-proxy-transfer',action='store_true')
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--reference-files',type=Path,nargs='+',required=True)
    parser.add_argument('--reference-cache',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.dataset!='mp_20' and not args.allow_proxy_transfer:
        parser.error('MP-20-trained predictors require --allow-proxy-transfer on other datasets; results are exploratory')
    sys.path.insert(0,str(args.cgdit_root.resolve()))
    from cgdit.evaluation import generated_properties as backend
    from legacy_eval.properties import predictors_at, crystal_array, infer_items
    torch.set_num_threads(1)
    predictors=predictors_at(args.predictor_root,args.predictor_seed,backend)
    bank=ReferenceBank(args.reference_files,args.dataset)
    ref_identity={'files':bank.identity(),'dataset':args.dataset,'predictors':predictors,'backend_sha256':sha256(Path(backend.__file__))}
    import hashlib
    key=hashlib.sha256(json.dumps(ref_identity,sort_keys=True).encode()).hexdigest()
    args.reference_cache.mkdir(parents=True,exist_ok=True)
    ref_path=args.reference_cache/(key+'.csv')
    receipt=args.reference_cache/(key+'.json')
    if receipt.exists():
        cached=json.loads(receipt.read_text(encoding='utf-8'))
        if cached['identity']!=ref_identity or sha256(ref_path)!=cached['sha256']:
            raise ValueError('Reference property cache identity mismatch')
    else:
        reference=[]
        for record,base in bank.rows:
            row={'id':str(record['id']),'_crystal':None}
            try: row['_crystal']=crystal_array(load_structure(record,base))
            except (ValueError,RuntimeError,OSError,KeyError) as error: row['parse_error']=str(error)
            reference.append(row)
        result=infer_items(reference,predictors,backend,args.batch_size)
        with ref_path.open('w',newline='',encoding='utf-8-sig') as stream:
            writer=csv.DictWriter(stream,fieldnames=['id','formation_energy','band_gap']);writer.writeheader()
            writer.writerows({k:r.get(k) for k in writer.fieldnames} for r in result)
        write_json(receipt,{'identity':ref_identity,'sha256':sha256(ref_path)})
    items=[]
    for r in read_records(args.samples):
        row={'index':r['index'],'_crystal':None}
        try:
            if r.get('status','generated')=='generated': row['_crystal']=crystal_array(load_structure(r,args.samples.parent))
        except (ValueError,RuntimeError,OSError,KeyError) as error: row['parse_error']=str(error)
        items.append(row)
    rows=infer_items(items,predictors,backend,args.batch_size)
    args.output.mkdir(parents=True,exist_ok=True)
    import shutil
    shutil.copyfile(ref_path,args.output/'reference_predictions.csv')
    with (args.output/'predictions.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=['index','formation_energy','band_gap','prediction_error']);writer.writeheader()
        for r in rows:
            r['prediction_error']=r.get('graph_error') or r.get('parse_error') or (
                'No finite independent prediction' if any(r.get(p) is None for p in ('formation_energy','band_gap')) else '')
            writer.writerow({k:r.get(k) for k in writer.fieldnames})
    write_json(args.output/'evaluator.json',{'evaluator':'frozen_CGDiT_FE_BG','samples_sha256':sha256(args.samples),
        'predictions_sha256':sha256(args.output/'predictions.csv'),
        'predictors':predictors,'backend_sha256':sha256(Path(backend.__file__)),
        'reference_sha256':bank.identity(),'reference_predictions_sha256':sha256(ref_path),
        'dataset':args.dataset,'domain_status':'in_training_domain' if args.dataset=='mp_20' else 'exploratory_transfer',
        'units':{'formation_energy':'eV/atom','band_gap':'eV'},'energy_above_hull':'not_assessed',
        'scope':'Frozen proxy predictions; not physical validation or certified C2DB accuracy'})


if __name__=='__main__': main()
