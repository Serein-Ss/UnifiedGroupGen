"""Clock laws and random-stream isolation for auxiliary mixed-flow paths."""
import math
import numpy as np
import pytest
import torch
import unifiedgroupgen.joint_flow.path as path
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.joint_flow.state import initial, apply, assign, JointAction
from unifiedgroupgen.pl_data.joint_path import JointTarget


@pytest.fixture(scope='module')
def target():
    state=initial(catalog_spec('space',1),budget=24,generator=torch.Generator().manual_seed(4))
    for z in (6,8,6):
        state=apply(state,JointAction('birth',letter='a'),torch.Generator().manual_seed(z))
        state=assign(state,len(state.elements)-1,z)
    return JointTarget(state,'three_orbit_clock_fixture',{})


def draw(target,seed,**options):
    return path.sample_path(target,seed,allowed_elements=[6,8],max_atoms=40,max_orbits=12,
                            budget=6,**options)


def test_fallback_keeps_time_clock_and_initial_noise_independent(target,monkeypatch):
    original=path.sample_path;fallbacks=[]
    def observed(*args,**kwargs):
        if len(args)>7 and args[7] is False:fallbacks.append(args[1])
        return original(*args,**kwargs)
    monkeypatch.setattr(path,'sample_path',observed)
    for seed in range(32):
        a=draw(target,seed,augment=True)
        b=draw(target,seed,augment=False)
        assert a.time==b.time
        assert [v['time'] for v in a.trajectory]==[v['time'] for v in b.trajectory]
        # The metric has no jump reset: its bridge recovers the same initial noise.
        torch.testing.assert_close(a.state.metric,b.state.metric,rtol=0,atol=0)
    assert fallbacks,'The regression must actually exercise budget fallback'


@pytest.mark.parametrize('stream',[0,1,2,5])
def test_consuming_one_numpy_stream_preserves_other_streams(target,monkeypatch,stream):
    # 0 program, 1 training time, 2 orbit clock, 5 element lifetime.
    original_rng=np.random.default_rng
    original_initial=path.initial;original_apply=path.apply
    captured=[]
    def capture_initial(*args,**kwargs):
        captured.append(('initial',args[1].get_state().clone()))
        return original_initial(*args,**kwargs)
    def capture_apply(*args,**kwargs):
        captured.append(('birth',args[2].get_state().clone()))
        return original_apply(*args,**kwargs)
    monkeypatch.setattr(path,'initial',capture_initial)
    monkeypatch.setattr(path,'apply',capture_apply)
    a=draw(target,81,time=1,augment=False);before=list(captured);captured.clear()
    def shifted(seed=None):
        rng=original_rng(seed)
        if isinstance(seed,np.random.SeedSequence) and seed.spawn_key==(stream,):rng.random(113)
        return rng
    monkeypatch.setattr(path.np.random,'default_rng',shifted)
    b=draw(target,81,time=1,augment=False)
    if stream!=2:assert [e['time'] for e in a.trajectory]==[e['time'] for e in b.trajectory]
    assert torch.equal(before[0][1],captured[0][1])
    assert not torch.equal(before[0][1],before[1][1]),'Initial and birth noise must use distinct streams'
    if stream!=0:
        assert len(before)==len(captured)
        assert all(torch.equal(x[1],y[1]) for x,y in zip(before,captured))
    torch.testing.assert_close(a.state.metric,b.state.metric,rtol=0,atol=0)
    # The endpoint remains physical data even when a stream's draw count changes.
    for z in set(a.state.elements):
        x=a.state.coordinates[[i for i,v in enumerate(a.state.elements) if v==z]]
        y=b.state.coordinates[[i for i,v in enumerate(b.state.elements) if v==z]]
        assert all(torch.cdist(x,y).min(1).values<1e-6)


def test_explicit_training_time_does_not_shift_other_streams(target):
    a=draw(target,17,augment=True)
    b=draw(target,17,time=a.time,augment=True)
    assert a.trajectory==b.trajectory and a.state.elements==b.state.elements
    torch.testing.assert_close(a.state.metric,b.state.metric,rtol=0,atol=0)
    torch.testing.assert_close(a.state.coordinates,b.state.coordinates,rtol=0,atol=0)


def test_extra_initial_noise_draws_do_not_change_birth_noise_or_clocks(target,monkeypatch):
    original_initial=path.initial;original_apply=path.apply;births=[]
    def capture(*args,**kwargs):
        births.append(args[2].get_state().clone())
        return original_apply(*args,**kwargs)
    monkeypatch.setattr(path,'apply',capture)
    a=draw(target,31,time=1,augment=False);before=list(births);births.clear()
    def shifted(*args,**kwargs):
        torch.randn(37,generator=args[1])
        return original_initial(*args,**kwargs)
    monkeypatch.setattr(path,'initial',shifted)
    b=draw(target,31,time=1,augment=False)
    assert a.trajectory==b.trajectory
    assert len(before)==len(births) and all(torch.equal(x,y) for x,y in zip(before,births))


@pytest.fixture(scope='module')
def clock_samples(target):
    clocks=[];times=[];mask_error=0.;mask_variance=0.;observations=0
    for seed in range(12000):
        sample=draw(target,seed,augment=True)
        event_times=np.array([v['time'] for v in sample.trajectory])
        assert len(event_times)==4 and np.all(np.diff(event_times)>0)
        completed=int((event_times<=sample.time).sum())
        assert sample.remaining==4-completed
        expected=sample.trajectory[completed]['action'] if completed<4 else None
        assert (None if sample.next_action is None else sample.next_action.to_dict())==expected
        for i,z in enumerate(sample.state.elements):
            probability=(sample.time-event_times[i])/(1-event_times[i])
            mask_error+=int(z!=0)-probability
            mask_variance+=probability*(1-probability);observations+=1
            if z:assert z==sample.element_targets[i]
        clocks.append(event_times);times.append(sample.time)
    return np.array(clocks),np.array(times),mask_error,mask_variance,observations


def test_ordered_uniform_clocks_and_training_time(clock_samples):
    clocks,times,*_=clock_samples
    assert abs(float((clocks[:,0]<=.2).mean())-(1-.8**4))<.02
    assert abs(float(clocks.mean())-.5)<.01
    assert abs(float(times.mean())-.5)<.01
    assert abs(float((times<=.2).mean())-(.2-1e-4)/(1-2e-4))<.02
    assert abs(float(np.corrcoef(times,clocks[:,0])[0,1]))<.04


def test_remaining_event_hazard_matches_order_statistics(clock_samples):
    clocks=clock_samples[0];t=.35;delta=.08
    remaining=(clocks>t).sum(1)
    for r in range(1,5):
        selected=clocks[remaining==r];count=len(selected)
        assert count>100
        next_time=selected[:,4-r]
        expected=1-((1-t-delta)/(1-t))**r
        observed=float((next_time<=t+delta).mean())
        assert abs(observed-expected)<5*math.sqrt(expected*(1-expected)/count)+1/count


def test_element_lifetime_matches_its_conditional_clock(clock_samples):
    _,_,error,variance,count=clock_samples
    assert count>10000
    assert abs(error)<5*math.sqrt(variance)


def test_variable_program_length_retains_conditional_uniform_clocks(target):
    groups={}
    for seed in range(4000):
        sample=path.sample_path(target,seed,time=0,allowed_elements=[6,8],budget=24)
        m=len(sample.trajectory);first=sample.trajectory[0]['time']
        groups.setdefault(m,[]).append(1-(1-first)**m)
    assert set(groups)=={4,6,8}
    for values in groups.values():
        assert len(values)>100
        assert abs(float(np.mean(values))-.5)<5/math.sqrt(12*len(values))


def test_repeated_seed_is_identical_and_terminal_is_data(target):
    for time in (0.,.37,1.):
        a=draw(target,29,time=time,augment=True);b=draw(target,29,time=time,augment=True)
        assert a.trajectory==b.trajectory and a.state.elements==b.state.elements
        assert a.time==b.time and a.next_action==b.next_action
        torch.testing.assert_close(a.state.coordinates,b.state.coordinates,rtol=0,atol=0)
        if time==0:assert a.state.num_atoms==0
        if time==1:
            assert a.state.frozen and a.remaining==0
            # Program order is a nuisance permutation, not the stored record order.
            for z in set(a.state.elements):
                x=a.state.coordinates[[i for i,v in enumerate(a.state.elements) if v==z]]
                y=target.state.coordinates[[i for i,v in enumerate(target.state.elements) if v==z]]
                assert all(torch.cdist(x,y).min(1).values<1e-6)
