"""Clock laws and random-stream isolation for auxiliary mixed-flow paths."""
import math
import numpy as np
import pytest
import torch
import unifiedgroupgen.pl_data.unified.path as path
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.group_constraints.flow_state import initial, apply, assign, JointAction
from unifiedgroupgen.common.data_utils import JointTarget


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


def test_site_symmetry_priority_randomizes_equal_stabilizers():
    from types import SimpleNamespace
    from unifiedgroupgen.pl_data.orbit_order import endpoint_orbit_order
    multiplicities = {'general': 8, 'special_a': 2, 'special_b': 2}
    spec = SimpleNamespace(template=lambda letter: SimpleNamespace(multiplicity=multiplicities[letter]))
    state = SimpleNamespace(letters=tuple(multiplicities), spec=spec)
    precedences = set()
    for seed in range(12):
        order = endpoint_orbit_order(state, np.random.default_rng(seed), 'site_symmetry')
        assert order[-1]==0
        precedences.add(tuple(order[:2]))
    assert precedences=={(1,2),(2,1)}


@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
@pytest.mark.parametrize('task',['dng','csp'])
def test_orbit_order_preserves_endpoint_and_csp_inventory(dataset,task):
    from unifiedgroupgen.tests.unit.test_joint_flow_v2 import config, record
    from unifiedgroupgen.common.data_utils import target_from_record
    from unifiedgroupgen.group_constraints.chemistry import orbit_counts
    from unifiedgroupgen.pl_data.unified.tasks import task_target
    cfg=config(task=task)
    endpoint=task_target(target_from_record(record(dataset)),cfg,42)
    options=dict(allowed_elements=cfg['allowed_elements'],augment=False,budget=24)
    prior=path.sample_path(endpoint,42,time=1,**options)
    for strategy in ('random','site_symmetry'):
        final=path.sample_path(endpoint,42,time=1,orbit_order=strategy,**options)
        assert [item['time'] for item in final.trajectory]==[item['time'] for item in prior.trajectory]
        generated=final.state
        assert generated.frozen and final.remaining==0
        expected=endpoint.state
        if strategy=='site_symmetry':
            sizes=[generated.spec.template(letter).multiplicity for letter in generated.letters]
            assert sizes==sorted(sizes)
        torch.testing.assert_close(generated.metric,expected.metric)
        assert orbit_counts(generated.spec,generated.letters,generated.elements)==orbit_counts(
            expected.spec,expected.letters,expected.elements)
        reference=list(zip(expected.letters,expected.elements,expected.coordinates.split(
            [expected.spec.template(letter).multiplicity for letter in expected.letters])))
        for letter,element,coords in zip(generated.letters,generated.elements,generated.coordinates.split(
                [generated.spec.template(letter).multiplicity for letter in generated.letters])):
            match=next(i for i,(name,z,original) in enumerate(reference)
                       if (name,z)==(letter,element) and torch.allclose(coords,original))
            reference.pop(match)
        assert not reference
        if task=='csp':
            for time in (0.,*(event['time'] for event in final.trajectory),1.):
                state=path.sample_path(endpoint,42,time=time,orbit_order=strategy,**options).state
                assert all(count>=0 for _,count in state.chemistry.remaining(state.spec,state.letters,state.elements))
                assert all(state.elements)


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


@pytest.mark.parametrize('dataset', ['mp_20', 'c2db_51'])
def test_csp_teacher_checks_preserve_exact_inventory_trajectory(dataset, monkeypatch):
    from unifiedgroupgen.group_constraints.state import JointState, legal_actions
    from unifiedgroupgen.group_constraints.unified import compile_constraints
    from unifiedgroupgen.pl_data import programs
    from unifiedgroupgen.pl_data.unified.tasks import task_target
    from unifiedgroupgen.tests.unit.test_joint_flow_v2 import config, record
    from unifiedgroupgen.common.data_utils import target_from_record

    options = config(task='csp')
    endpoint = task_target(target_from_record(record(dataset)), options, 42)
    draws = [(seed, order, time) for seed in range(4)
             for order in ('random', 'site_symmetry') for time in (.67, 1.)]

    def draw(case):
        seed, order, time = case
        return path.sample_path(endpoint, seed, time=time, orbit_order=order, budget=24,
                                allowed_elements=options['allowed_elements'])

    optimized = [draw(case) for case in draws]
    def original_check(spec, letters, elements, control, allowed, max_atoms, max_orbits,
                       remaining_events, action):
        chart = compile_constraints(spec, letters)
        state = JointState(spec, letters, elements, torch.zeros(chart.metric_dimension),
                           torch.zeros((chart.num_atoms, 3)), chemistry=control, budget=remaining_events)
        return action in legal_actions(state, allowed, max_atoms, max_orbits)

    monkeypatch.setattr(programs, 'inventory_teacher_allows', original_check)
    for actual, case in zip(optimized, draws):
        expected = draw(case)
        assert actual.trajectory == expected.trajectory
        assert actual.remaining == expected.remaining and actual.next_action == expected.next_action
        assert actual.state.letters == expected.state.letters and actual.state.elements == expected.state.elements
        torch.testing.assert_close(actual.state.metric, expected.state.metric, atol=0, rtol=0)
        torch.testing.assert_close(actual.state.coordinates, expected.state.coordinates, atol=0, rtol=0)
