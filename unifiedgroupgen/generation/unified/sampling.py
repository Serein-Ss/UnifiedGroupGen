"""Asynchronous event-aware Heun sampling with packed neural evaluations."""
from dataclasses import replace
import math
import torch
from unifiedgroupgen.generation.types import JointSample
from unifiedgroupgen.group_constraints.flow_state import initial,advance,apply,assign,actions
from unifiedgroupgen.conditional.guidance import guided_forward, validate_guidance

@torch.no_grad()
def sample_batch(model,specs,steps=64,epsilon=1e-4,budget=None,birth_scale=None,
                 properties=None,chemistries=None,seeds=None,generators=None,frames=None,
                 trajectory_callback=None,guidance_scale=1.):
    """Optionally report committed states, not Heun predictors or root-search trials.

    callback(local_index, stage, state, time, integration_step, event, reason)
    observes initial, continuous, event and final states without consuming RNG.
    """
    if steps<1 or not 0<epsilon<.1:raise ValueError('Invalid integration controls')
    n=len(specs)
    if not n:return []
    like=next(model.parameters());budget=budget or model.event_budget
    birth_scale=model.birth_scale if birth_scale is None else birth_scale
    properties=properties or [{} for _ in specs];chemistries=chemistries or [None]*n
    validate_guidance(guidance_scale, any(properties), getattr(getattr(model,'property_encoder',None),'dropout',0.))
    frames=frames or [None]*n;seeds=seeds or list(range(n))
    generators=generators or [torch.Generator(device=like.device).manual_seed(seed) for seed in seeds]
    states=[initial(g,generators[i],budget,chemistries[i],like.device,like.dtype,frames[i]) for i,g in enumerate(specs)]
    times=[0.]*n;events=[[] for _ in specs];iterations=[0]*n;reason=[None]*n
    if trajectory_callback is not None:
        for i,s in enumerate(states):trajectory_callback(i,'initial',s,0.,0,None,None)
    def exponential(i):
        return -math.log(max(float(torch.rand((),generator=generators[i],device=like.device)),1e-12))
    thresholds=[exponential(i) for i in range(n)]
    def evaluate(ids,batch,clock,decide=False):
        try:
            return guided_forward(model,batch,clock,[properties[i] for i in ids],
                                  None if decide else [()]*len(ids),guidance_scale)
        except (ValueError,RuntimeError):
            values=[]
            for i,s,t in zip(ids,batch,clock):
                try:
                    values.append(guided_forward(model,[s],[t],[properties[i]],
                                  None if decide else [()],guidance_scale)[0])
                except (ValueError,RuntimeError) as error:
                    reason[i]='numerical_or_graph_failure: '+str(error)
                    values.append(None)
            return values
    mode=model.training;model.eval()
    try:
        while True:
            ids=[i for i in range(n) if reason[i] is None and times[i]<1-epsilon-1e-12]
            if not ids:break
            active=[states[i] for i in ids];t=[times[i] for i in ids]
            dt=[min(1/steps,.2*(1-v),1-epsilon-v) for v in t]
            p=[properties[i] for i in ids]
            out=evaluate(ids,active,t)
            good=[k for k,o in enumerate(out) if o is not None]
            ids=[ids[k] for k in good];active=[active[k] for k in good];t=[t[k] for k in good];dt=[dt[k] for k in good];out=[out[k] for k in good]
            if not ids:continue
            p=[properties[i] for i in ids]
            velocity=[(o['metric_velocity_scaled']/(1-v),o['coordinate_velocity_scaled']/(1-v)) for o,v in zip(out,t)]
            predicted=[advance(s,*v,d) for s,v,d in zip(active,velocity,dt)]
            end=evaluate(ids,predicted,[v+d for v,d in zip(t,dt)])
            end=[o if o is not None else out[k] for k,o in enumerate(end)]
            v1=[(o['metric_velocity_scaled']/(1-v-d),o['coordinate_velocity_scaled']/(1-v-d)) for o,v,d in zip(end,t,dt)]
            event_ids=[]
            for k,i in enumerate(ids):
                s=active[k];v=t[k];d=dt[k];iterations[i]+=1
                masks=sum(z==0 for z in s.elements)
                rho0=(float(out[k]['remaining'])+masks)/(1-v)
                rho1=(float(end[k]['remaining'])+masks)/(1-v-d)
                integrated=.5*d*(rho0+rho1)
                fraction=1.
                if integrated>=thresholds[i] and integrated>0:
                    lo,hi=0.,1.
                    for _ in range(24):
                        f=.5*(lo+hi);hazard=d*(rho0*f+.5*(rho1-rho0)*f*f)
                        if hazard<thresholds[i]:lo=f
                        else:hi=f
                    fraction=.5*(lo+hi);event_ids.append(i)
                else:thresholds[i]-=integrated
                a,b=velocity[k],v1[k]
                vm=a[0]+.5*fraction*(b[0]-a[0]);vx=a[1]+.5*fraction*(b[1]-a[1])
                states[i]=advance(s,vm,vx,d*fraction);times[i]=v+d*fraction
                if not torch.isfinite(states[i].metric).all() or not torch.isfinite(states[i].coordinates).all():
                    reason[i]='nonfinite_continuous_state'
                if trajectory_callback is not None:
                    trajectory_callback(i,'continuous',states[i],times[i],iterations[i],None,reason[i])
            event_ids=[i for i in event_ids if reason[i] is None]
            if event_ids:
                decision=evaluate(event_ids,[states[i] for i in event_ids],[times[i] for i in event_ids],True)
                for i,o in zip(event_ids,decision):
                    if o is None:continue
                    s=states[i];old=s.num_atoms
                    masked=[j for j,z in enumerate(s.elements) if not z]
                    mu=float(o['remaining'])
                    pick=float(torch.rand((),generator=generators[i],device=like.device))*(mu+len(masked))
                    if pick<mu:
                        if not o['actions']:
                            reason[i]='no_legal_orbit_action';continue
                        ix=int(torch.multinomial(o['action_logits'].softmax(0),1,generator=generators[i]))
                        action=o['actions'][ix]
                        states[i]=apply(s,action,generators[i],birth_scale)
                        entry=action.to_dict()
                    else:
                        orbit=masked[min(int(pick-mu),len(masked)-1)]
                        element=int(torch.multinomial(o['element_logits'][orbit].softmax(0),1,generator=generators[i]))
                        states[i]=assign(s,orbit,element)
                        entry=dict(kind='element',orbit=orbit,element=element)
                    events[i].append(dict(time=times[i],action=entry,atoms_before=old,atoms_after=states[i].num_atoms))
                    thresholds[i]=exponential(i)
                    if states[i].status=='failed':reason[i]='orbit_budget_or_no_legal_completion'
                    if trajectory_callback is not None:
                        trajectory_callback(i,'event',states[i],times[i],iterations[i],events[i][-1],reason[i])
        results=[]
        for i,s in enumerate(states):
            done=reason[i] is None and s.frozen and bool(s.num_atoms) and all(s.elements)
            if done:s=replace(s,status='done')
            else:s=replace(s,status='failed')
            if trajectory_callback is not None:
                trajectory_callback(i,'final',s,times[i],iterations[i],None,
                                    reason[i] or ('complete' if done else 'unfinished_at_terminal_time'))
            results.append(JointSample(s,times[i],events[i],reason[i] or ('complete' if done else 'unfinished_at_terminal_time'),iterations[i],epsilon))
        return results
    finally:model.train(mode)

def sample_one(model,spec,**kwargs):
    generator=kwargs.pop('generator',None)
    magnetic=kwargs.pop('magnetic',False)
    if magnetic:raise ValueError('No magnetic fields in joint_flow_v2')
    chemistry=kwargs.pop('chemistry',None);properties=kwargs.pop('properties',None)
    return sample_batch(model,[spec],generators=None if generator is None else [generator],
                        properties=[properties or {}],chemistries=[chemistry],**kwargs)[0]
