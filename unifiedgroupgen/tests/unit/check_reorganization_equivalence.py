"""Bounded checkpoint probe for comparing original and reorganized imports."""
import argparse
import json
from pathlib import Path

import torch


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--workspace',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    from unifiedgroupgen.pl_modules.factory import build_models

    from unifiedgroupgen.group_constraints.compiler import Descriptor,compile_descriptor
    from unifiedgroupgen.generation.flow import sample_flow
    torch.set_num_threads(1)
    results={}
    for name in ['mp20_u','c2db51_u','joint_u']:
        checkpoint=torch.load(args.workspace/'artifacts/evaluation_inputs/main15_seed42'/name/'model.best.pt',map_location='cpu',weights_only=False)
        proposal,flow=build_models(checkpoint['config'],'cpu')
        proposal.load_state_dict(checkpoint['proposal']);flow.load_state_dict(checkpoint['flow'])
        proposal.eval();flow.eval()
        domains=['mp_20','c2db_51'] if name=='joint_u' else ['c2db_51' if name=='c2db51_u' else 'mp_20']
        for domain in domains:
            with (args.workspace/'data/dataset/full_verified_v1'/domain/'test.jsonl').open(encoding='utf-8') as stream:
                record=json.loads(next(stream))
            descriptor=Descriptor.from_dict(record['descriptor']);compiled=compile_descriptor(descriptor)
            properties=dict(record['properties'])
            if name=='joint_u':properties['source_domain']=0 if domain=='mp_20' else 1
            generator=torch.Generator().manual_seed(142)
            state=compiled.prior('cpu',torch.float32,generator)*.05
            with torch.inference_mode():
                loss=float(proposal.loss(descriptor,properties))
                velocity=flow(compiled,state,.4,properties).tolist()
                output=sample_flow(flow,compiled,properties,steps=2,guidance=0,generator=torch.Generator().manual_seed(142))
            results[name+'/'+domain]={'descriptor_loss':loss,'velocity':velocity,'sample_state':output['state'].tolist(),
                                     'lattice':output['lattice'].tolist(),'coordinates':output['coordinates'].tolist()}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(results,allow_nan=False),encoding='utf-8')


if __name__=='__main__': main()
