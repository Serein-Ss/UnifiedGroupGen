"""Paired FP32/BF16 forward/backward on the same real records; no optimizer updates."""
import argparse
import gc
import json
from pathlib import Path
import time
import torch

from unifiedgroupgen.common.configuration import project_root
from unifiedgroupgen.common.precision import inference_autocast
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
from unifiedgroupgen.pl_modules.factory import build_models
from unifiedgroupgen.pl_modules.objectives.flow_matching import conditional_path, velocity_loss


def records(root, count):
    result=[]
    for domain in ('mp_20','c2db_51'):
        with (root/'data/dataset/full_verified_v1'/domain/'train.jsonl').open(encoding='utf-8') as stream:
            for line in stream:
                record=json.loads(line); c=compile_descriptor(Descriptor.from_dict(record['descriptor']))
                if c.num_atoms <= 16:
                    result.append(record)
                    if sum(r['dataset']==domain for r in result) == count: break
    return result


def benchmark(name, rows, steps, reference_size=False):
    choices={'default':None,
        'cspnet':{'_target_':'unifiedgroupgen.pl_modules.backbones.cspnet.CSPNetBackbone','layers':2,'frequencies':16},
        'mattergen':{'_target_':'unifiedgroupgen.pl_modules.backbones.mattergen.MatterGenBackbone','blocks':2,'max_neighbors':32},
        'equiformer_v2':{'_target_':'unifiedgroupgen.pl_modules.backbones.equiformer_v2.EquiformerV2Backbone',
                         'layers':2,'channels':32,'lmax':2,'mmax':1,'max_neighbors':32}}
    if reference_size:
        choices['cspnet'].update(layers=6,frequencies=128)
        choices['mattergen'].update(blocks=3,max_neighbors=50)
        choices['equiformer_v2'].update(layers=12,channels=128,lmax=6,mmax=2,max_neighbors=50)
    config={'groups':sorted({(r['descriptor']['kind'],r['descriptor']['number']) for r in rows}),
        'properties':{},'cond_dropout':0.,'model':{'hidden':512 if reference_size else 128,'heads':4,'proposal_layers':1,
                                               'flow_layers':6 if reference_size else 2,'frequencies':128 if reference_size else 16}}
    if choices[name]: config['model']['candidate_backbone']=choices[name]
    torch.manual_seed(42); _, model=build_models(config,'cuda'); model.train()
    charts=[compile_descriptor(Descriptor.from_dict(r['descriptor'])) for r in rows]
    endpoints=[torch.tensor(r['state'],device='cuda') for r in rows]
    paths=[conditional_path(c,z,z.new_tensor(.5),generator=torch.Generator(device='cuda').manual_seed(142+i))
           for i,(c,z) in enumerate(zip(charts,endpoints))]
    states,targets=zip(*paths); outputs={}; gradients={}; results={}
    for precision in ('fp32','bf16'):
        model.inference_precision=precision
        def iteration():
            torch.manual_seed(142); model.zero_grad(set_to_none=True)
            with inference_autocast(model):
                prediction=model.forward_batch(charts,states,[.5]*len(rows))
            with torch.autocast('cuda',enabled=False):
                parts=[velocity_loss(c,z,t,p) for c,z,t,p in zip(charts,states,targets,prediction)]
                loss=torch.stack([p['lattice']+p['coordinates']/9 for p in parts]).mean()
            loss.backward()
            assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
            return loss.detach(),torch.cat([p.detach() for p in prediction])
        for _ in range(2):iteration()
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();started=time.perf_counter()
        for _ in range(steps):loss,out=iteration()
        torch.cuda.synchronize();duration=time.perf_counter()-started
        outputs[precision]=out.cpu(); gradients[precision]=torch.cat([p.grad.detach().flatten().cpu() for p in model.parameters() if p.grad is not None])
        results[precision]={'seconds_per_step':duration/steps,'records_per_second':len(rows)*steps/duration,
            'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'loss':float(loss),'finite_gradients':True}
    results['output_relative_l2']=float((outputs['bf16']-outputs['fp32']).norm()/outputs['fp32'].norm().clamp_min(1e-12))
    results['gradient_cosine']=float(torch.nn.functional.cosine_similarity(gradients['bf16'].double(),gradients['fp32'].double(),dim=0))
    results['speedup']=results['fp32']['seconds_per_step']/results['bf16']['seconds_per_step']
    results['configuration']=config['model']
    del model;gc.collect();torch.cuda.empty_cache()
    return results


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--records-per-domain',type=int,default=4);parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--models',nargs='+',default=['default','cspnet','mattergen','equiformer_v2'])
    parser.add_argument('--reference-size',action='store_true')
    args=parser.parse_args();torch.set_num_threads(1)
    if not torch.cuda.is_bf16_supported():raise RuntimeError('Native BF16 CUDA required')
    rows=records(project_root(),args.records_per_domain)
    result={'gpu':torch.cuda.get_device_name(),'torch':torch.__version__,
        'records':[(r['dataset'],r['id']) for r in rows],
        'scope':'Fixed real states and targets, OrbitFlow forward/backward only, no optimizer updates; not formal training/quality',
        'reference_size':args.reference_size,
        'models':{}}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    for name in args.models:
        result['models'][name]=benchmark(name,rows,args.steps,args.reference_size)
        args.output.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        print(name,json.dumps(result['models'][name]),flush=True)


if __name__=='__main__':main()
