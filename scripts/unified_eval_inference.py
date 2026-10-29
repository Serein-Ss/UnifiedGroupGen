"""Native model adapter: held-out components, hard-composition CSP and de novo attempts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time
import torch
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
from unifiedgroupgen.evaluation_io import inline_cif_bytes, read_records, record_counts, sha256, write_json


def checked_config(saved):
    from evaluate_checkpoint import check_heldout_inputs
    config=saved['config']
    paths=[ROOT/p for p in config['test_files']]
    check_heldout_inputs(saved,paths)
    # Checkpoint audit also binds TRAIN/VAL bytes, not just paths in the metadata.
    if config.get('require_data_audit'):
        audit=json.loads((ROOT/config['data_audit']).read_text(encoding='utf-8'))
        for split in ('train','val','test'):
            expected=[r['sha256'] for r in audit['inputs'][split]]
            if [sha256(ROOT/p) for p in config[split+'_files']]!=expected:
                raise ValueError(f'{split} bytes differ from the audited checkpoint data')
    return config


def components(proposal,flow,saved,output,device,batch_size,seed):
    from unifiedgroupgen.evaluation_components import descriptor_batch
    from unifiedgroupgen.conditioning import attach_dataset_domains
    from unifiedgroupgen.symmetry import Descriptor, compile_descriptor
    from unifiedgroupgen.flow import flow_loss_batch
    config=saved['config']
    records=attach_dataset_domains([r for p in config['test_files'] for r in read_records(ROOT/p)],config)
    scale=saved.get('runtime_options',{}).get('coordinate_length_scale',1.)
    rows=[]
    with torch.inference_mode():
        for start in range(0,len(records),batch_size):
            batch=records[start:start+batch_size]
            supported=[r for r in batch if all(o['atomic_number'] in proposal.legality.elements for o in r['descriptor']['orbits'])]
            unsupported=[r for r in batch if r not in supported]
            rows.extend({'id':r['id'],'dataset':r['dataset'],'status':'unsupported_elements'} for r in unsupported)
            if not supported: continue
            descriptors=[Descriptor.from_dict(r['descriptor']) for r in supported]
            compiled=[compile_descriptor(d) for d in descriptors]
            props=[r['properties'] for r in supported]
            discrete=descriptor_batch(proposal,descriptors,props)
            losses=flow_loss_batch(flow,compiled,[torch.tensor(r['state'],device=device) for r in supported],props)
            for i,r in enumerate(supported):
                row={'id':r['id'],'dataset':r['dataset'],'status':'evaluated',
                    'descriptor_loss':discrete[i]['descriptor_loss'],
                    'teacher_forced_sequence_correct':discrete[i]['teacher_forced_sequence_correct'],
                    'lattice':float(losses['lattice'][i]),'coordinates_raw':float(losses['coordinates'][i])}
                row['coordinates_dimensionless']=row['coordinates_raw']/scale**2
                row['flow']=row['lattice']+row['coordinates_dimensionless']
                for part,value in discrete[i]['parts'].items():
                    for key in ('tokens','correct','nll_sum'): row[part+'_'+key]=value[key]
                rows.append(row)
            print(json.dumps({'phase':'components','completed':len(rows),'requested':len(records)}),flush=True)
    fields=sorted({k for r in rows for k in r})
    with (output/'records.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    def aggregate(group):
        good=[r for r in group if r['status']=='evaluated']
        values={k:sum(r[k] for r in good)/len(good) if good else None for k in
                ('descriptor_loss','flow','lattice','coordinates_raw','coordinates_dimensionless','teacher_forced_sequence_correct')}
        tokens={part:{'count':sum(r[part+'_tokens'] for r in good),
                      'correct':sum(r[part+'_correct'] for r in good),
                      'nll_sum':sum(r[part+'_nll_sum'] for r in good)} for part in ('G','W','Z','STOP')}
        for value in tokens.values():
            value['accuracy']=value['correct']/value['count'] if value['count'] else None
            value['nll']=value['nll_sum']/value['count'] if value['count'] else None
        return {'requested':len(group),'evaluated':len(good),'unsupported':len(group)-len(good),'means':values,'tokens':tokens}
    result={'overall':aggregate(rows),'sources':{d:aggregate([r for r in rows if r['dataset']==d]) for d in sorted({r['dataset'] for r in rows})},
            'conditioning':{'mode':'available_true_test_properties','properties':[p for p,s in config.get('properties',{}).items() if not s.get('context')],
                            'source_context':config.get('dataset_domains',{}),'composition_embedding':False,'observations':False},
            'coordinate_length_scale':scale,'seed':seed,'scope':'Full test; teacher-forced tokens and ground-truth-description flow error. Not reconstruction or property accuracy.'}
    write_json(output/'summary.json',result)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(9,3.5),constrained_layout=True)
    for source,value in result['sources'].items():
        axes[0].plot(['G','W','Z','STOP'],[value['tokens'][k]['accuracy'] for k in ('G','W','Z','STOP')],'o-',label=source)
        axes[1].plot(['Lattice','Coordinates / scale2'],[value['means']['lattice'],value['means']['coordinates_dimensionless']],'o-',label=source)
    axes[0].set(ylabel='Teacher-forced token accuracy',ylim=(0,1.05));axes[1].set(ylabel='Mean flow-matching loss')
    axes[0].legend(fontsize=8);axes[1].legend(fontsize=8)
    fig.savefig(output/'components.png',dpi=200);plt.close(fig)


def generate(proposal,flow,config,job,output,device,seed,steps,guidance):
    from unifiedgroupgen.chemical_controls import ChemicalRules
    from unifiedgroupgen.symmetry import Descriptor, compile_descriptor
    from unifiedgroupgen.cli import composition_of
    from unifiedgroupgen.flow import sample_flow
    from unifiedgroupgen.evaluation import to_structure
    refs=({str(r['id']):r for p in config['test_files'] for r in read_records(ROOT/p) if r['dataset']==job['dataset']}
          if job['task']=='csp' else {})
    if job['task']=='csp':
        attempts=[(key,candidate) for key in job['case_ids'] for candidate in range(job['k'])]
    else:
        attempts=[(None,i) for i in range(job['count'])]
    ledger=output/'samples.jsonl'
    previous=read_records(ledger) if ledger.exists() else []
    sampling_seconds=sum(r.get('wall_seconds',0.) for r in previous)
    if [r['index'] for r in previous]!=list(range(len(previous))) or len(previous)>len(attempts):
        raise ValueError('Attempt ledger is incomplete/corrupt; it was preserved')
    for r in previous:
        if r['status']=='generated':
            if 'cif' in r:
                inline_cif_bytes(r)
            elif sha256(output/r['cif_path'])!=r['cif_sha256']:
                raise ValueError('Existing CIF no longer matches its receipt')
    with torch.inference_mode():
        for index,(case,candidate) in enumerate(attempts):
            if index<len(previous): continue
            started=time.perf_counter()
            # Per-attempt seeds make interrupted/resumed generation identical.
            attempt_seed=seed+index;random.seed(attempt_seed);np.random.seed(attempt_seed);torch.manual_seed(attempt_seed)
            generator=torch.Generator(device=device).manual_seed(attempt_seed)
            requested_group=(job['groups'][index//job['per_group']] if job.get('groups') else job.get('group'))
            row={'index':index,'case_id':case,'candidate':candidate,'properties_requested':job['properties'],
                 'requested_group':requested_group,'seed':attempt_seed}
            phase='descriptor_sampling'
            try:
                reference=refs[case] if case is not None else None
                counts=record_counts(reference,ROOT) if reference else None
                row['chemical_request']={'composition':counts}
                fixed=Descriptor.from_dict(reference['descriptor']) if reference and job.get('fixed_descriptor') else None
                prefix=[proposal.vocab.index['BOS'],proposal.vocab.index[f"G:{job['kind']}:{requested_group}"]] if requested_group else None
                trained_composition=config.get('composition_probability',.5)>0
                descriptor=fixed or proposal.sample(job['properties'],job['kind'],counts,guidance,1.,generator,
                    chemical=ChemicalRules(),prefix=prefix,condition_composition=trained_composition)
                compiled=compile_descriptor(descriptor)
                if counts is not None and composition_of(compiled)!=counts: raise ValueError('Hard composition violation')
                phase='flow_integration'
                generated=sample_flow(flow,compiled,job['properties'],steps,guidance,None,
                    counts if trained_composition else None,generator,
                    condition_observation=config.get('observation_probability',0)>0)
                phase='numeric_validation'
                if not torch.isfinite(generated['state']).all():
                    raise FloatingPointError('Nonfinite continuous state rejected before CIF export')
                structure=to_structure(compiled,generated)
                phase='cif_export'
                cif=structure.to(fmt='cif')
                row.update(cif=cif,cif_sha256=hashlib.sha256(cif.encode('utf-8')).hexdigest())
                # Native output is only successful after in-memory CIF round-trip verification.
                from unifiedgroupgen.evaluation_io import load_structure, counts_of
                phase='cif_readback'
                restored=load_structure(row,output)
                if counts_of(restored)!=counts_of(structure):
                    raise RuntimeError('CIF export/readback changed integer composition')
                if not np.isfinite(restored.lattice.matrix).all() or not np.isfinite(restored.frac_coords).all():
                    raise FloatingPointError('Nonfinite CIF readback')
                row.update(status='generated',descriptor=descriptor.to_dict(),state=generated['state'].cpu().tolist())
            except (ValueError,RuntimeError,FloatingPointError,KeyError) as error:
                if isinstance(error,torch.cuda.OutOfMemoryError): raise
                row.update(status='failed',failure_stage=phase,error_type=type(error).__name__,reason=str(error))
            row['wall_seconds']=time.perf_counter()-started
            sampling_seconds+=row['wall_seconds']
            with ledger.open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,allow_nan=False)+'\n');stream.flush()
            write_json(output/'sampling.status.json',{'state':'running','attempted':index+1,'requested':len(attempts),
                'last_attempt_seconds':row['wall_seconds'],'sampling_seconds':sampling_seconds})
            if (index+1)%10==0 or index+1==len(attempts):
                print(json.dumps({'phase':'sample','attempted':index+1,'requested':len(attempts),'sampling_seconds':sampling_seconds}),flush=True)
    write_json(output/'sampling.status.json',{'state':'complete','attempted':len(attempts),'requested':len(attempts),'sampling_seconds':sampling_seconds})


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--job',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--seed',type=int,default=142)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--guidance',type=float,default=0.)
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    saved=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    config=checked_config(saved)
    from unifiedgroupgen.cli import build_models
    proposal,flow=build_models(config,args.device)
    proposal.load_state_dict(saved['proposal']);flow.load_state_dict(saved['flow']);proposal.eval();flow.eval()
    job=json.loads(args.job.read_text(encoding='utf-8'))
    if job.get('checkpoint_sha256') and job['checkpoint_sha256']!=sha256(args.checkpoint):
        raise ValueError('Checkpoint changed after the evaluation plan was created')
    identity={'checkpoint_sha256':sha256(args.checkpoint),'job':job,'seed':args.seed,'steps':args.steps,
              'guidance':args.guidance,'test':{p:sha256(ROOT/p) for p in config['test_files']},
              'structure_storage':'inline_cif_utf8_v1',
              'adapter_sha256':sha256(__file__)}
    args.output.mkdir(parents=True,exist_ok=True)
    receipt=args.output/'identity.json'
    if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8'))!=identity:
        raise ValueError('Inference identity changed; use a new output directory')
    write_json(receipt,identity)
    if job['task']=='components': components(proposal,flow,saved,args.output,args.device,args.batch_size,args.seed)
    else: generate(proposal,flow,config,job,args.output,args.device,args.seed,args.steps,args.guidance)


if __name__=='__main__': main()
