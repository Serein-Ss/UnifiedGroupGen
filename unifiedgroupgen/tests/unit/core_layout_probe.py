"""Reproducible pre/post-refactor CPU evidence; no optimizer or Trainer calls."""
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


def tensor(value):
    value=value.detach().cpu().contiguous()
    return dict(shape=list(value.shape),dtype=str(value.dtype),
                sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())


def parameters(model):
    return {name:dict(value=tensor(p),gradient=None if p.grad is None else tensor(p.grad))
            for name,p in model.named_parameters()}


def capture():
    from unifiedgroupgen.tests.unit.test_joint_flow_v2 import config,record
    from unifiedgroupgen.pl_modules.factory import build_joint_model,build_models
    from unifiedgroupgen.pl_data.joint_path import target_from_record
    from unifiedgroupgen.pl_data.task_path import task_target
    from unifiedgroupgen.joint_flow.path import sample_path
    from unifiedgroupgen.joint_flow.objective import loss
    from unifiedgroupgen.joint_flow.sampling import sample_batch
    from unifiedgroupgen.group_constraints.unified import catalog_spec
    from unifiedgroupgen.group_constraints.compiler import Descriptor,OrbitSpec,compile_descriptor
    from unifiedgroupgen.pl_modules.objectives.joint import joint_loss
    result={}
    for backend in ['equiformer_v2','joint_atom_graph']:
        for dataset in ['mp_20','c2db_51']:
            for task in ['dng','csp','property']:
                torch.manual_seed(91);random.seed(91);np.random.seed(91)
                cfg=config(backend,task);model=build_joint_model(cfg).eval()
                target=task_target(target_from_record(record(dataset)),cfg,12)
                samples=[sample_path(target,k,time=t,allowed_elements=cfg['allowed_elements'],budget=24)
                         for k,t in [(3,.2),(4,.67),(5,.92)]]
                losses=loss(model,samples);losses['loss'].backward()
                values=dict(losses={k:tensor(v) for k,v in losses.items()},parameters=parameters(model))
                outputs=model.forward_batch([s.state for s in samples],[s.time for s in samples],[s.properties for s in samples])
                values['outputs']=[{k:tensor(v) for k,v in row.items() if torch.is_tensor(v)} for row in outputs]
                if task=='dng':
                    sampled=sample_batch(model,[catalog_spec(kind,1) for kind in ['space','layer','plane']],
                                         seeds=[7,8,9],steps=3,budget=6,epsilon=.005)
                    values['samples']=[dict(metric=tensor(s.state.metric),coordinates=tensor(s.state.coordinates),
                        elements=list(s.state.elements),letters=list(s.state.letters),status=s.state.status,
                        events=s.events,time=s.time) for s in sampled]
                result[backend+'/'+dataset+'/'+task]=values
    torch.manual_seed(91);random.seed(91);np.random.seed(91)
    cfg=dict(groups=[('space',136)],properties={},allowed_elements=[8,22],max_atoms=16,max_orbits=6,
             model=dict(hidden=16,heads=4,proposal_layers=1,flow_layers=1,frequencies=2))
    proposal,flow=build_models(cfg,'cpu');proposal.eval();flow.eval()
    descriptor=Descriptor('space',136,(OrbitSpec('a',22),OrbitSpec('f',8)))
    compiled=compile_descriptor(descriptor);endpoint=compiled.prior()*.1
    row=dict(id='fixture',dataset='fixture',descriptor=descriptor.to_dict(),state=endpoint.tolist(),properties={})
    losses=joint_loss(proposal,flow,[row],0.,0.)
    (losses['descriptor'].mean()+losses['flow'].mean()).backward()
    result['two_stage']=dict(losses={k:tensor(v) for k,v in losses.items()},
                             proposal=parameters(proposal),flow=parameters(flow))
    return result


if __name__=='__main__':
    torch.set_num_threads(1)
    Path(sys.argv[1]).write_text(json.dumps(capture(),sort_keys=True,indent=2,allow_nan=False)+'\n',encoding='utf-8')
