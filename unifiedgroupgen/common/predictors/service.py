"""Frozen CGDiT or public MEGNet property predictions with the same assessment CSV protocol."""
from unifiedgroupgen.common.configuration import resolve_path
import csv
import json
from pathlib import Path
import sys
import torch

from unifiedgroupgen.common.structures import load_structure, read_records, sha256, write_json, ReferenceBank


def external_cif_hashes(records):
    result = {}
    for record, base in records:
        if 'cif' in record:
            continue
        name = record.get('cif_path')
        if name is None and record.get('status') == 'generated' and 'index' in record:
            candidate = base/f"sample_{int(record['index']):05d}.cif"
            if candidate.exists(): name = candidate
        if name:
            path = Path(name)
            path = path if path.is_absolute() else base/path
            result[str(path.resolve())] = sha256(path) if path.is_file() else None
    return result


def public_rows(records, predictor, kind, batch_size, key):
    rows = []
    for start in range(0, len(records), batch_size):
        block = records[start:start+batch_size]
        structures, errors = [], []
        for record, base in block:
            try:
                if record.get('status', 'generated') != 'generated':
                    raise ValueError('Generation failed; no structure to predict')
                structures.append(load_structure(record, base)); errors.append('')
            except (ValueError, RuntimeError, OSError, KeyError) as error:
                structures.append(None); errors.append(str(error))
        predictions = predictor.predict_structures(structures, kind=kind, batch_size=batch_size)
        for (record, _), prediction, error in zip(block, predictions, errors):
            rows.append({key: record[key], **prediction,
                         'prediction_error': error or prediction['prediction_error']})
        if len(rows) % 128 == 0 or len(rows) == len(records):
            print(json.dumps({'phase': 'public_property_predictions', 'completed': len(rows),
                              'requested': len(records)}), flush=True)
    return rows


def write_predictions(path, rows, key, properties):
    with path.open('w', newline='', encoding='utf-8-sig') as stream:
        writer = csv.DictWriter(stream, fieldnames=[key, *properties, 'prediction_error'])
        writer.writeheader(); writer.writerows(rows)


def public_predictions(args):
    import hashlib
    import shutil
    from unifiedgroupgen.common.configuration import load_yaml
    from unifiedgroupgen.common.predictors.matgl import MatGLPredictor
    for name in ('samples', 'reference_cache', 'output', 'model_cache'):
        setattr(args, name, resolve_path(getattr(args, name)))
    args.reference_files = [resolve_path(p) for p in args.reference_files]
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Preserve existing predictions; choose an empty --output directory')
    config = load_yaml(args.predictor_config or 'configs/predictors/matgl_mp.yaml')
    predictor = MatGLPredictor(args.model_cache, **config)
    properties = list(predictor.models)
    bank = ReferenceBank(args.reference_files, args.dataset)
    if not bank.rows:
        raise ValueError('No reference records for the selected dataset')
    references = [({**record, 'id': key}, base) for key, (record, base) in bank.by_id.items()]
    samples = [(r, args.samples.parent) for r in read_records(args.samples)]
    indices = [str(r['index']) for r, _ in samples]
    if len(set(indices)) != len(indices):
        raise ValueError('Duplicate sample indices')
    implementation = [Path(__file__), Path(__file__).with_name('matgl.py'),
                      Path(__file__).with_name('structures.py'), Path(__file__).parents[1]/'structures.py',
                      Path(__file__).parents[2]/'evaluation/utils.py']
    code = {str(p.resolve()): sha256(p) for p in implementation}
    ref_identity = {'files': bank.identity(), 'external_cifs': external_cif_hashes(bank.rows),
                    'dataset': args.dataset, 'kind': args.kind, 'predictor': predictor.identity, 'code': code}
    cache_key = hashlib.sha256(json.dumps(ref_identity, sort_keys=True).encode()).hexdigest()
    args.reference_cache.mkdir(parents=True, exist_ok=True)
    reference_csv = args.reference_cache/(cache_key+'.csv')
    receipt = args.reference_cache/(cache_key+'.json')
    if receipt.exists():
        cached = json.loads(receipt.read_text(encoding='utf-8'))
        if cached['identity'] != ref_identity or sha256(reference_csv) != cached['sha256']:
            raise ValueError('Public predictor reference cache identity mismatch')
    else:
        reference = public_rows(references, predictor, args.kind, args.batch_size, 'id')
        write_predictions(reference_csv, reference, 'id', properties)
        write_json(receipt, {'identity': ref_identity, 'sha256': sha256(reference_csv)})
    rows = public_rows(samples, predictor, args.kind, args.batch_size, 'index')
    args.output.mkdir(parents=True, exist_ok=True)
    write_predictions(args.output/'predictions.csv', rows, 'index', properties)
    shutil.copyfile(reference_csv, args.output/'reference_predictions.csv')
    write_json(args.output/'evaluator.json', {
        'evaluator': 'public_MatGL_MEGNet', 'samples_sha256': sha256(args.samples),
        'predictions_sha256': sha256(args.output/'predictions.csv'),
        'reference_predictions_sha256': sha256(reference_csv), 'reference_sha256': bank.identity(),
        'structure_files_sha256': {**external_cif_hashes(bank.rows), **external_cif_hashes(samples)},
        'predictors': predictor.identity, 'adapter_sha256': code, 'kind': args.kind, 'dataset': args.dataset,
        'domain_status': 'MP_sourced_proxy_overlap_not_audited' if args.kind == 'space' and args.dataset == 'mp_20'
                         else 'exploratory_transfer',
        'units': {name: predictor.registry['models'][name]['unit'] for name in properties},
        'energy_above_hull': 'not_assessed',
        'counts': {'requested': len(rows), 'complete_predictions': sum(not r['prediction_error'] for r in rows),
                   'failed_predictions': sum(bool(r['prediction_error']) for r in rows)},
        'scope': 'Frozen property proxy on raw input geometry; no relaxation or physical stability claim. '
                 'Public predictor training overlap and C2DB accuracy have not been established.'})
    print(json.dumps({'state': 'complete', 'requested': len(rows), 'output': str(args.output)}), flush=True)




def predict(args):
    if args.batch_size < 1:
        raise ValueError('--batch-size must be positive')
    if args.backend == 'matgl':
        if args.kind is None: raise ValueError('Public predictors require explicit --kind space/layer')
        if (args.kind == 'layer' or args.dataset != 'mp_20') and not args.allow_proxy_transfer:
            raise ValueError('Layer/non-MP predictions require --allow-proxy-transfer; accuracy is unverified')
        torch.set_num_threads(1)
        return public_predictions(args)
    if args.cgdit_root is None or args.predictor_root is None:
        raise ValueError('CGDiT backend requires --cgdit-root and --predictor-root')
    if args.dataset!='mp_20' and not args.allow_proxy_transfer:
        raise ValueError('MP-20-trained predictors require --allow-proxy-transfer on other datasets; results are exploratory')
    sys.path.insert(0,str(args.cgdit_root.resolve()))
    from cgdit.evaluation import generated_properties as backend
    from unifiedgroupgen.common.predictors.cgdit import predictors_at, crystal_array, infer_items
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
