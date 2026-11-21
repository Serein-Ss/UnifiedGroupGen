"""Exact auxiliary conditional paths: orbit trajectories, branch bridges and MASK CTMC."""
from dataclasses import dataclass, replace
import numpy as np
import torch
from unifiedgroupgen.group_constraints.state import JointAction
from unifiedgroupgen.pl_data.programs import inventory_program
from unifiedgroupgen.pl_data.orbit_order import endpoint_orbit_order
from unifiedgroupgen.group_constraints.flow_state import initial, apply

@dataclass
class FlowSample:
    state: object
    time: float
    metric_anchor: torch.Tensor
    coordinate_anchor: torch.Tensor
    element_targets: tuple
    remaining: int
    next_action: object
    properties: dict
    identity: str
    trajectory: tuple = ()
    reference_orbits: tuple[bool, ...] | None = None

    def to(self, device, dtype=None):
        return replace(self,state=self.state.to(device,dtype),
                       metric_anchor=self.metric_anchor.to(device=device,dtype=dtype),
                       coordinate_anchor=self.coordinate_anchor.to(device=device,dtype=dtype))

def orbit_program(target, rng, allowed, max_atoms, max_orbits, budget, augment, orbit_order='random'):
    """Build the teacher edits using chemistry, independent of property labels."""
    endpoint=target.state
    anchors={i:x for i,x in enumerate(endpoint.coordinates.split([w.multiplicity for w in endpoint.compiled.templates]))}
    labels={i:z for i,z in enumerate(endpoint.elements)}
    program=[];roles=[];letters=[]
    if endpoint.chemistry.counts:
        # Exact formula retains its coupled, typed, inventory-conserving trajectory.
        inventory, values=inventory_program(target,rng,allowed,max_atoms,max_orbits,budget,False,orbit_order)
        anchors={i:v[0] for i,v in values.items()}
        for entry in inventory:
            program.append((replace(entry.action,kind='freeze') if entry.action.kind=='end' else entry.action,entry.roles))
    else:
        ghost=-1
        for j in endpoint_orbit_order(endpoint,rng,orbit_order):
            letter=endpoint.letters[j]
            occupied_atoms=sum(endpoint.spec.template(s).multiplicity for s in letters)
            if augment and roles and rng.random()<.25 and len(roles)<max_orbits:
                w=endpoint.spec.template(letter)
                if occupied_atoms+w.multiplicity<=max_atoms and (w.dof or letter not in letters):
                    a,b=w.arrays()
                    anchors[ghost]=endpoint.metric.new_tensor(b+np.einsum('nbq,q->nb',a,rng.normal(0,.2,w.dof)))
                    labels[ghost]=int(rng.choice(allowed))
                    roles.append(ghost);letters.append(letter)
                    program.append((JointAction('birth',letter=letter),tuple(roles)))
                    program.append((JointAction('delete',orbit=len(roles)-1),tuple(roles[:-1])))
                    roles.pop();letters.pop();ghost-=1
            wrong=[w for w in endpoint.spec.templates if w.letter!=letter
                   and (w.dof or w.letter not in letters)
                   and occupied_atoms+w.multiplicity<=max_atoms] if augment else []
            if augment and wrong and len(roles)<max_orbits and rng.random()<.2:
                w=wrong[int(rng.integers(len(wrong)))]
                a,b=w.arrays()
                anchors[ghost]=endpoint.metric.new_tensor(b+np.einsum('nbq,q->nb',a,rng.normal(0,.2,w.dof)))
                labels[ghost]=int(rng.choice(allowed))
                roles.append(ghost);letters.append(w.letter)
                program.append((JointAction('birth',letter=w.letter),tuple(roles)))
                roles[-1]=int(j);letters[-1]=letter
                program.append((JointAction('replace',len(roles)-1,letter),tuple(roles)))
                ghost-=1
                continue
            parents=[i for i,w in enumerate(letters) if w==letter and endpoint.spec.template(w).dof]
            action=(JointAction('split',orbit=int(rng.choice(parents)),letter=letter)
                    if parents and rng.random()<.5 else JointAction('birth',letter=letter))
            roles.append(int(j));letters.append(letter);program.append((action,tuple(roles)))
        program.append((JointAction('freeze'),tuple(roles)))
    return program, anchors, labels


def sample_path(target, seed, time=None, allowed_elements=None, max_atoms=192,max_orbits=52,
                budget=96,augment=True,birth_scale=.2,orbit_order='random',orbit_ablation='joint'):
    streams=np.random.SeedSequence(int(seed)).spawn(6)
    endpoint=target.state
    allowed=endpoint.chemistry.domain(allowed_elements or sorted(set(endpoint.elements)))
    if endpoint.num_atoms>max_atoms or len(endpoint.letters)>max_orbits:
        raise ValueError('Endpoint exceeds model limits')
    if set(endpoint.elements)-set(allowed):
        raise ValueError('Endpoint element outside TRAIN domain')
    program,anchors,labels=orbit_program(target,np.random.default_rng(streams[0]),allowed,
                                         max_atoms,max_orbits,budget,augment,orbit_order)
    if len(program)>budget-2:
        if augment:
            return sample_path(target,seed,time,allowed_elements,max_atoms,max_orbits,budget,False,
                               birth_scale,orbit_order,orbit_ablation)
        raise ValueError('Orbit event budget cannot represent endpoint')
    time_rng=np.random.default_rng(streams[1])
    event_rng=np.random.default_rng(streams[2])
    initial_generator=torch.Generator(device=target.state.metric.device).manual_seed(
        int(streams[3].generate_state(1,dtype=np.uint64)[0]))
    generator=torch.Generator(device=target.state.metric.device).manual_seed(
        int(streams[4].generate_state(1,dtype=np.uint64)[0]))
    element_rng=np.random.default_rng(streams[5])
    t=float(time_rng.uniform(1e-4,1-1e-4) if time is None else time)
    if not 0<=t<=1:raise ValueError('time outside [0,1]')
    times=np.sort(event_rng.random(len(program)))
    state=initial(endpoint.spec,initial_generator,budget,endpoint.chemistry,endpoint.metric.device,endpoint.metric.dtype)
    roles=();births={};last=0.;completed=0
    def bridge(s,stop):
        if orbit_ablation == 'fixed_orbits' and (not s.frozen or stop == last):
            return s
        x=torch.cat([anchors[i] for i in roles]) if roles else s.coordinates.new_zeros((0,3))
        ratio=(1-stop)/(1-last)
        return replace(s,metric=endpoint.metric+(s.metric-endpoint.metric)*ratio,
                       coordinates=x+(s.coordinates-x)*ratio)
    for tau,(action,next_roles) in zip(times,program):
        if tau>t:break
        state=bridge(state,float(tau))
        state=apply(state,action,generator,birth_scale)
        for role in next_roles:
            if role not in roles:births[role]=(float(tau),float(element_rng.random()))
        roles=next_roles;last=float(tau);completed+=1
    state=bridge(state,t)
    if not endpoint.chemistry.counts:
        elements=[]
        for role in roles:
            birth,u=births[role]
            elements.append(labels[role] if u<=(t-birth)/(1-birth) else 0)
        state=replace(state,elements=tuple(elements))
    x=torch.cat([anchors[i] for i in roles]) if roles else state.coordinates.new_zeros((0,3))
    metric_anchor = endpoint.metric
    if orbit_ablation == 'fixed_orbits' and not state.frozen:
        metric_anchor, x = state.metric, state.coordinates
    return FlowSample(state,t,metric_anchor,x,tuple(labels[i] for i in roles) if not endpoint.chemistry.counts else state.elements,
                      len(program)-completed,program[completed][0] if completed<len(program) else None,
                      target.properties,target.identity,
                      tuple(dict(time=float(tau),action=a.to_dict()) for tau,(a,_) in zip(times,program)),
                      reference_orbits=tuple(role >= 0 for role in roles))
