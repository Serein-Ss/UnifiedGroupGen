"""Verify existing baseline inputs/CIFs and prepare explicit prior-batch receipts."""
import argparse
import json
from pathlib import Path

from unifiedgroupgen.common.configuration import project_root,resolve_path
from unifiedgroupgen.common.structures import read_records,sha256,verified_prior_records,write_json


def prepare(source,output):
    root=project_root()
    source=resolve_path(source)
    pipeline=json.loads((source/'pipeline.json').read_text(encoding='utf-8'))
    if pipeline['state']!='complete': raise ValueError('Baseline campaign has not completed')
    jobs={}
    totals={'batches':0,'records':0,'generated':0,'failed':0,'component_records':0}
    for model,checkpoint_hash in pipeline['identity']['checkpoints'].items():
        model_root=source/model
        identity_path=model_root/'data_identity.json'
        identity=json.loads(identity_path.read_text(encoding='utf-8'))
        if identity['checkpoint_sha256']!=checkpoint_hash: raise ValueError('Baseline checkpoint identity changed')
        for group in identity['files'].values():
            for item in group:
                if sha256(resolve_path(item['path']))!=item['sha256']:
                    raise ValueError('Baseline audited dataset changed')
        check=model_root/'test'
        summary=json.loads((check/'summary.json').read_text(encoding='utf-8'))
        if summary['state']!='complete' or summary['records_sha256']!=sha256(check/'records.jsonl'):
            raise ValueError('Baseline component diagnostics are incomplete or changed')
        jobs[model+'/components']={'checkpoint_sha256':checkpoint_hash,
            'files':{name:{'path':str(check/name),'sha256':sha256(check/name)} for name in ('summary.json','records.jsonl')},
            'data_identity':{'path':str(identity_path),'sha256':sha256(identity_path)},
            'scope':'Previously completed full component checks; not newly recomputed'}
        totals['component_records']+=summary['requested']
        for ledger in model_root.glob('*/*/samples.jsonl'):
            dataset,mode=ledger.parent.parent.name,ledger.parent.name
            report_path=ledger.parent/'summary.json'
            report=json.loads(report_path.read_text(encoding='utf-8'))
            if report['state']!='complete' or report['checkpoint_sha256']!=checkpoint_hash:
                raise ValueError('Previous structure batch does not bind this checkpoint')
            if report['steps']!=64 or report['guidance']!=0:
                raise ValueError('Previous structure batch used another generation protocol')
            records=read_records(ledger)
            spec={'samples':str(ledger),'samples_sha256':sha256(ledger),'records':len(records),
                'metadata_sha256':{str(report_path):sha256(report_path),str(identity_path):sha256(identity_path),
                                   str(source/'pipeline.json'):sha256(source/'pipeline.json')},
                'seed_contract':'Historical hash(seed/model/dataset/mode/index); original seeds preserved'}
            restored=verified_prior_records(spec)
            if len(restored)!=report['attempted']: raise ValueError('Previous generation budget changed')
            label={'de_novo':'dng','composition':'csp','fixed_description':'fixed_descriptor','fixed_group':'group_sweep'}[mode]
            jobs[model+'/'+dataset+'_'+label]={'checkpoint_sha256':checkpoint_hash,'prior_samples':spec}
            totals['batches']+=1; totals['records']+=len(restored)
            totals['generated']+=sum(r['status']=='generated' for r in restored)
            totals['failed']+=sum(r['status']=='failed' for r in restored)
    result={'source':str(source),'source_pipeline_sha256':sha256(source/'pipeline.json'),'jobs':jobs,'verified':totals,
            'scope':'Preserve original generation, failures, seeds, CIF bytes and component statistics; no metric reuse across protocols'}
    write_json(output,result)
    print(json.dumps({'passed':True,**totals,'manifest':str(output)}))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();prepare(args.source,args.output)


if __name__=='__main__': main()
