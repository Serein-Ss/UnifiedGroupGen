from unifiedgroupgen.common.configuration import project_root, resolve_path
import csv
import json
import torch
from torch.nn import functional as F
from unifiedgroupgen.common.structures import read_records, sha256, write_json
from unifiedgroupgen.common.source import file_hash
from unifiedgroupgen.common.configuration import root_path
from unifiedgroupgen.common.precision import resolve_precision, inference_autocast
ROOT=project_root()


def execute(cfg):
    """Run full held-out component diagnostics, independently of generation/metrics."""
    from unifiedgroupgen.common.configuration import task_options
    from unifiedgroupgen.pl_modules.factory import build_models,load_model_checkpoint
    import random
    import numpy as np
    args=task_options(cfg,'component_check')
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    if args.device=='auto': args.device='cuda' if torch.cuda.is_available() else 'cpu'
    if args.checkpoint is None: raise ValueError('component_check.checkpoint is required')
    saved=load_model_checkpoint(args.checkpoint,map_location='cpu',weights_only=False)
    config=checked_config(saved)
    args.output.mkdir(parents=True,exist_ok=True)
    identity={'checkpoint_sha256':sha256(args.checkpoint),'seed':args.seed,
              'test':{p:sha256(resolve_path(p)) for p in config['test_files']},
              'batch_size':args.batch_size,'code':sha256(__file__),
              'precision':resolve_precision(args.device,getattr(args,'precision','fp32'))}
    receipt=args.output/'identity.json'
    if receipt.exists() and json.loads(receipt.read_text(encoding='utf-8'))!=identity:
        raise ValueError('Component-check inputs changed; choose a new output directory')
    write_json(receipt,identity)
    proposal,flow=build_models(config,args.device)
    proposal.load_state_dict(saved['proposal'],strict=True); flow.load_state_dict(saved['flow'],strict=True)
    proposal.eval(); flow.eval()
    proposal.inference_precision = flow.inference_precision = identity['precision']
    return components(proposal,flow,saved,args.output,args.device,args.batch_size,args.seed)

def descriptor_batch(model, descriptors, properties):
    sequences=[model.vocab.encode(d) for d in descriptors]
    device=model.embedding.weight.device
    width=max(len(s)-1 for s in sequences)
    ids=torch.full((len(sequences),width),model.vocab.index['PAD'],dtype=torch.long,device=device)
    targets=torch.full_like(ids,-100)
    legal=torch.ones(len(sequences),width,len(model.vocab.tokens),dtype=torch.bool,device=device)
    for i,(sequence,descriptor) in enumerate(zip(sequences,descriptors)):
        n=len(sequence)-1
        ids[i,:n]=torch.tensor(sequence[:-1],device=device)
        targets[i,:n]=torch.tensor(sequence[1:],device=device)
        legal[i,:n]=torch.stack([model.legality.allowed(sequence[:j],None,descriptor.kind) for j in range(1,len(sequence))]).to(device)
    h=model.embedding(ids)+model.position(torch.arange(width,device=device))[None]
    h=h+model.condition.forward_batch(properties,h)[:,None]
    causal=torch.ones(width,width,dtype=torch.bool,device=device).triu(1)
    logits=model.head(model.transformer(h,mask=causal,src_key_padding_mask=ids.eq(model.vocab.index['PAD'])))
    logits=logits.masked_fill(~legal,-torch.inf)
    losses=F.cross_entropy(logits.transpose(1,2),targets,reduction='none')
    predictions=logits.argmax(-1)
    result=[]
    for i,sequence in enumerate(sequences):
        n=len(sequence)-1
        parts={p:{'nll_sum':0.,'tokens':0,'correct':0} for p in ('G','W','Z','STOP')}
        for j,token in enumerate(sequence[1:]):
            part=parts[model.vocab.tokens[token].split(':')[0]]
            part['nll_sum']+=float(losses[i,j]);part['tokens']+=1;part['correct']+=int(predictions[i,j]==token)
        result.append({'descriptor_loss':float(losses[i,:n].mean()),
            'teacher_forced_sequence_correct':bool(predictions[i,:n].eq(targets[i,:n]).all()),'parts':parts})
    return result

def check_heldout_inputs(checkpoint, paths):
    """Formal evaluation must cover the audited test files and the matching training run."""
    config = checkpoint['config']
    if not config.get('require_data_audit'):
        return None
    audit_path = root_path(config['data_audit'])
    report = json.loads(audit_path.read_text(encoding='utf-8'))
    if not report.get('passed'):
        raise ValueError('Held-out data audit has not passed')
    for split in ('train', 'val'):
        if checkpoint.get(f'{split}_split_sha256') != [item['sha256'] for item in report['inputs'][split]]:
            raise ValueError('Checkpoint training/validation files differ from the held-out audit')
    actual = sorted((str(path.resolve()), file_hash(path)) for path in paths)
    expected = sorted((str(root_path(item['path'])), item['sha256']) for item in report['inputs']['test'])
    if actual != expected:
        raise ValueError('Formal held-out evaluation requires exactly the complete audited test files')
    return {'path': str(audit_path), 'sha256': file_hash(audit_path)}

def checked_config(saved):
    config=saved['config']
    paths=[resolve_path(ROOT/p,ROOT) for p in config['test_files']]
    check_heldout_inputs(saved,paths)
    # Checkpoint audit also binds TRAIN/VAL bytes, not just paths in the metadata.
    if config.get('require_data_audit'):
        audit=json.loads((resolve_path(ROOT/config['data_audit'],ROOT)).read_text(encoding='utf-8'))
        for split in ('train','val','test'):
            expected=[r['sha256'] for r in audit['inputs'][split]]
            if [sha256(resolve_path(ROOT/p,ROOT)) for p in config[split+'_files']]!=expected:
                raise ValueError(f'{split} bytes differ from the audited checkpoint data')
    return config


def components(proposal,flow,saved,output,device,batch_size,seed):
    from unifiedgroupgen.conditional.properties import attach_dataset_domains
    from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
    from unifiedgroupgen.pl_modules.objectives.flow_matching import flow_loss_batch
    config=saved['config']
    records=attach_dataset_domains([r for p in config['test_files'] for r in read_records(resolve_path(ROOT/p,ROOT))],config)
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
            with inference_autocast(flow):
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
            'coordinate_length_scale':scale,'seed':seed,'precision':getattr(flow,'inference_precision','fp32'),
            'evaluation_type':'model_component_check','scope':'Full test; teacher-forced tokens and ground-truth-description flow error. Not reconstruction or property accuracy.'}
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


import hydra


@hydra.main(version_base='1.3',config_path=str(project_root()/'configs'),config_name='check_components')
def main(cfg):
    execute(cfg)


if __name__=='__main__': main()
