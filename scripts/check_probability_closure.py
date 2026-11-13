"""Analytic/independent numerical checks for the closure appendix. No training."""
from collections import Counter
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import math
import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.optimize import brentq, linear_sum_assignment
from scipy.special import betaln, gammaincc
import torch

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'intermediate/checks/probability_closure_20261113'
SITE=ROOT/'web/symmetry-lab'


def event_checks():
    rng=np.random.default_rng(20261113);n=131072;m=5
    clocks=np.sort(rng.random((n,m)),axis=1);birth=clocks[:,m-2]
    reveal=birth+(1-birth)*rng.random(n)
    rows=[]
    for eps in (.1,.03,.01,.003,.001):
        terminal=1-eps
        late=1-terminal**m;mask=quad(lambda x:eps/(1-x)*math.exp((m-2)*math.log(x)+math.log1p(-x)-betaln(m-1,2)),0,terminal)[0]
        late_birth=1-(m*terminal**(m-1)-(m-1)*terminal**m)
        # Full MASK probability includes births still in the future.
        exact_mask=mask+late_birth
        observed=float(np.mean(reveal>terminal));se=math.sqrt(max(exact_mask*(1-exact_mask),1/n)/n)
        assert abs(observed-exact_mask)<6*se+2/n
        observed_late=float(np.mean(clocks[:,-1]>terminal));se_late=math.sqrt(max(late*(1-late),1/n)/n)
        assert abs(observed_late-late)<6*se_late+2/n
        assert mask<=m*eps+1e-12
        rows.append(dict(epsilon=eps,freeze_pending_exact=late,freeze_pending_mc=observed_late,
                         final_life_mask_exact=exact_mask,final_life_mask_mc=observed,
                         prefreeze_bridge_expectation=mask,bridge_over_epsilon=mask/eps,
                         no_freeze_last_birth_cost=eps*(1+math.log(1/eps))))
    # Independent quadrature of the no-freeze uniform last-birth example.
    for eps in (.05,.005,.0005):
        value=quad(lambda x:eps/(1-x),0,1-eps)[0]+eps
        assert abs(value-eps*(1+math.log(1/eps)))<1e-10
    waits=rng.exponential(size=(n,8)).sum(1);tail=[]
    for s in (3.,8.,12.):
        exact=float(gammaincc(8,s));mc=float(np.mean(waits>s))
        assert abs(mc-exact)<6*math.sqrt(exact*(1-exact)/n)+2/n
        tail.append(dict(log_time=s,erlang_tail=exact,mc=mc))
    # Smooth bounded trajectory with no terminal limit; this is not a trained model.
    rotation=solve_ivp(lambda s,y:[-y[1],y[0]],(0,4*math.pi),[1.,0.],rtol=1e-11,atol=1e-12,dense_output=True)
    points=rotation.sol(np.array([math.pi,2*math.pi,3*math.pi,4*math.pi])).T
    assert np.max(np.abs(points[:,0]-[-1,1,-1,1]))<1e-9
    return dict(monte_carlo_samples=n,program_length=m,terminal_rows=rows,erlang_check=tail,
                nonconvergent_rotation_points=points.tolist(),claim='Analytic identities and toy Monte Carlo, not material-model results.')


def heun_events(step,iterations):
    t=0.;y=.8;threshold=.7;event_time=None
    while t<1-1e-14:
        dt=min(step,1-t);v0=-.7*y;predict=y+dt*v0;v1=-.7*predict
        rho0=1+.2*y+.1*t;rho1=1+.2*predict+.1*(t+dt)
        risk=.5*dt*(rho0+rho1);fraction=1.
        if event_time is None and risk>=threshold:
            lo,hi=0.,1.
            for _ in range(iterations):
                middle=(lo+hi)/2
                integrated=dt*(rho0*middle+.5*(rho1-rho0)*middle**2)
                if integrated<threshold:lo=middle
                else:hi=middle
            fraction=(lo+hi)/2
        elif event_time is None:threshold-=risk
        y+=dt*fraction*(v0+.5*fraction*(v1-v0));t+=dt*fraction
        if fraction<1:
            event_time=t;y+=.3
    return y,event_time


def numerical_checks():
    integrated=lambda t:t+(.16/.7)*(1-math.exp(-.7*t))+.05*t*t
    exact_time=brentq(lambda t:integrated(t)-.7,0,1,xtol=1e-14)
    exact_y=.8*math.exp(-.7)+.3*math.exp(-.7*(1-exact_time))
    rows=[]
    for size in (8,16,32,64,128,256):
        y,t=heun_events(1/size,24);fine_y,fine_t=heun_events(1/size,52)
        coarse_y,coarse_t=heun_events(1/size,4)
        assert abs(coarse_t-fine_t)<=1/size*2**-4+1e-12
        rows.append(dict(steps=size,h=1/size,terminal_error=abs(y-exact_y),event_time_error=abs(t-exact_time),
                         q4_vs_q52_time_difference=abs(coarse_t-fine_t),q4_root_interval_bound=1/size*2**-4))
    orders=[math.log(rows[i-1]['terminal_error']/rows[i]['terminal_error'],2) for i in range(1,len(rows))]
    assert min(orders[:4])>1.8 and max(orders[:4])<2.25,orders
    from unifiedgroupgen.pl_modules.backbones.periodic_graph import radius_graph
    counts=[]
    for a in (6.-1e-5,6.+1e-5):
        cell=torch.diag(torch.tensor([a,10.,10.],dtype=torch.float64))[None]
        edge,*_=radius_graph(torch.zeros((1,3),dtype=torch.float64),cell,torch.tensor([1]),
                            torch.tensor([[True,True,True]]),6.,64,16,keep_shells=True)
        counts.append(int(edge.shape[1]))
    assert counts==[2,0],counts
    return dict(smooth_reference_exact_event_time=exact_time,smooth_reference_exact_terminal=exact_y,
                rows=rows,observed_terminal_orders=orders,actual_radius_graph_edges_across_cutoff=counts,
                claim='Smooth scalar reference only; topology switch test does not prove every network output is discontinuous.')


def mixture_checks():
    groups=np.array([0]*10+[1]*10);matches=np.array([1]*9+[0]+[1]+[0]*9,dtype=bool)
    empirical=np.bincount(groups)/len(groups)
    conditional=np.bincount(groups[matches],minlength=2)/matches.sum()
    assert np.allclose(empirical,[.5,.5]) and np.allclose(conditional,[.9,.1])
    alpha=np.array([.2,.8]);accepted=empirical*alpha;accepted/=accepted.sum()
    proposal=empirical/alpha;proposal/=proposal.sum();corrected=proposal*alpha;corrected/=corrected.sum()
    assert np.allclose(accepted,[.2,.8]) and np.allclose(corrected,empirical)
    weights=np.linspace(.1,2.,len(groups));zg=np.array([weights[groups==g].mean() for g in (0,1)])
    corrected_group=empirical*zg;corrected_group/=corrected_group.sum()
    full=weights/weights.sum();reconstructed=np.zeros(len(groups))
    for g in (0,1):reconstructed[groups==g]=corrected_group[g]*weights[groups==g]/weights[groups==g].sum()
    assert np.max(np.abs(full-reconstructed))<1e-15
    return dict(empirical_group=empirical.tolist(),inventory_conditional_group=conditional.tolist(),
                feasible_only_group=empirical.tolist(),marked_weight_tv=float(abs(empirical-conditional).sum()/2),
                accepted_group=accepted.tolist(),inverse_acceptance_proposal=proposal.tolist(),
                corrected_accepted_group=corrected.tolist(),tilt_identity_max_error=float(abs(full-reconstructed).max()))


def train_audit(dataset,inventory):
    from unifiedgroupgen.group_constraints.unified import catalog_spec
    from unifiedgroupgen.group_constraints.chemistry import ChemicalControl,can_complete
    source=ROOT/'data/dataset/full_verified_v1'/dataset/'train.jsonl'
    totals=Counter();matched=Counter();property_weights=Counter();labelled=Counter()
    sumw=sumw2=0.;records=0
    with source.open(encoding='utf-8') as stream:
        for line in stream:
            row=json.loads(line);d=row['descriptor'];key=(d['kind'],d['number']);totals[key]+=1;records+=1
            spec=catalog_spec(*key);counts=Counter()
            for orbit in d['orbits']:counts[orbit['atomic_number']]+=spec.template(orbit['letter']).multiplicity
            if dict(counts)==inventory:matched[key]+=1
            value=row.get('properties',{}).get('band_gap')
            if value is not None and math.isfinite(float(value)):
                labelled[key]+=1;w=math.exp(-.5*((float(value)-1.5)/.2)**2)
                property_weights[key]+=w;sumw+=w;sumw2+=w*w
    control=ChemicalControl.from_inputs(counts=inventory)
    feasible={g:can_complete(catalog_spec(*g),(),(),control,94,52,192) for g in totals}
    norm=sum(totals[g] for g in totals if feasible[g]);nmatch=sum(matched.values())
    assert nmatch>0
    rows=[]
    for g,n in sorted(totals.items()):
        assert not matched[g] or feasible[g]
        rows.append(dict(kind=g[0],group=g[1],train_count=n,inventory_count=matched[g],feasible=feasible[g],
                         unconditional_weight=n/records,feasible_weight=n/norm if feasible[g] else 0.,
                         inventory_conditional_weight=matched[g]/nmatch,observed_property_count=labelled[g],
                         property_kernel_weight=property_weights[g]/sumw))
    tv=sum(abs(x['feasible_weight']-x['inventory_conditional_weight']) for x in rows)/2
    property_tv=sum(abs(x['unconditional_weight']-x['property_kernel_weight']) for x in rows)/2
    return dict(dataset=dataset,source=source.relative_to(ROOT).as_posix(),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                split='train',train_records=records,groups=len(totals),inventory=inventory,inventory_records=nmatch,
                feasible_groups=sum(feasible.values()),inventory_weight_tv=tv,property_target_eV=1.5,
                property_kernel_bandwidth_eV=.2,property_kernel_ess=sumw*sumw/sumw2,
                property_observed_records=sum(labelled.values()),property_weight_tv=property_tv,rows=rows,
                scope='TRAIN empirical weights only. Property is explicitly a kernel-tilted target, not a measured conditional-density truth.')


def actual_sampler_check():
    from unifiedgroupgen.group_constraints.unified import catalog_spec
    from unifiedgroupgen.joint_flow.state import JointAction
    from unifiedgroupgen.joint_flow.sampling import sample_batch
    spec=catalog_spec('space',1)
    class Oracle(torch.nn.Module):
        allowed_elements=(8,);max_atoms=12;max_orbits=4;event_budget=8;birth_scale=.2
        def __init__(self):
            super().__init__();self.dummy=torch.nn.Parameter(torch.zeros((),dtype=torch.float64))
        def forward_batch(self,states,times,properties=None,actions=None):
            result=[]
            for state in states:
                support=() if state.frozen else ((JointAction('birth',letter='a'),) if not state.letters else (JointAction('freeze'),))
                logits=state.metric.new_full((len(state.letters),119),-torch.inf);logits[:,8]=0
                target=state.coordinates.new_tensor([[.13,.25,.41]]) if state.num_atoms else state.coordinates
                result.append(dict(metric_velocity_scaled=-state.metric,coordinate_velocity_scaled=target-state.coordinates,
                    remaining=state.metric.new_tensor(0 if state.frozen else 2-state.step),element_logits=logits,
                    actions=support,action_logits=state.metric.new_zeros(len(support))))
            return result
    model=Oracle();rows=[]
    for eps in (.01,.001,.0001):
        for steps in (16,32,64):
            samples=sample_batch(model,[spec]*6,seeds=list(range(6)),steps=steps,epsilon=eps,budget=8)
            completed=[r for r in samples if r.state.status=='done'];assert completed
            for sample in samples:
                assert len(sample.events)<=16
                assert torch.isfinite(sample.state.metric).all() and torch.isfinite(sample.state.coordinates).all()
            rows.append(dict(epsilon=eps,steps=steps,attempts=6,complete=len(completed),
                failures=Counter(r.reason for r in samples if r.state.status!='done'),
                completed_coordinate_errors=[float((r.state.coordinates-r.state.coordinates.new_tensor([[.13,.25,.41]])).norm()) for r in completed]))
    return dict(rows=rows,claim='Actual production sampler with an analytic oracle, no learned network, optimizer or material-quality evidence.')


def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    value=dict(version='probability-closure-2026-11-13',event_terminal=event_checks(),
               numerics=numerical_checks(),mixture_toy=mixture_checks(),
               train_audits=[train_audit('mp_20',{22:2,8:4}),train_audit('c2db_51',{42:1,16:2})],
               actual_sampler=actual_sampler_check(),scope='Mathematical diagnostic and TRAIN audit only. No complete training, model performance claim or formal proof assistant.')
    protected=json.loads((OUT/'protected_before.json').read_text(encoding='utf-8'))
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protected.items())
    value['protected_core_files_unchanged']=len(protected);value['passed']=True
    content=json.dumps(value,ensure_ascii=False,indent=2)+'\n'
    (OUT/'verification.json').write_text(content,encoding='utf-8')
    (SITE/'dist/tables/probability_closure_verification.json').write_text(content,encoding='utf-8')
    print(json.dumps(dict(passed=True,train=[{k:a[k] for k in ('dataset','train_records','inventory_records','inventory_weight_tv','property_weight_tv')} for a in value['train_audits']],
                         smooth_orders=value['numerics']['observed_terminal_orders'],oracle_attempts=54,protected_core_files=len(protected)),ensure_ascii=False))


if __name__=='__main__':main()
