"""Full committed-state recording, selection, failures and fixed-seed parity."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl
from unifiedgroupgen.joint_flow.sampling import sample_batch
from unifiedgroupgen.joint_flow.state import initial,JointAction
from unifiedgroupgen.joint_flow.trajectory import TrajectoryWriter


@pytest.fixture(autouse=True)
def threads():
    old=torch.get_num_threads();torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def read(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]


def begin(writer,index,spec):
    writer.start(index,index+7,spec,spec.number,ChemicalControl(),{})


class Program(torch.nn.Module):
    """Sampler fixture, no training or material-quality claim."""
    backend='fixture';event_budget=10;birth_scale=.2;max_atoms=24;max_orbits=4
    allowed_elements=(8,)
    def __init__(self,fail=False):
        super().__init__();self.dummy=torch.nn.Parameter(torch.zeros(()));self.fail=fail
    def forward_batch(self,states,times,properties=None,actions=None):
        if self.fail:raise RuntimeError('fixture graph failure')
        result=[]
        for s in states:
            general=next(w for w in s.spec.templates if w.dof)
            fixed=next(w for w in s.spec.templates if not w.dof)
            program=(JointAction('birth',letter=general.letter),
                     JointAction('split',orbit=0,letter=general.letter),
                     JointAction('replace',orbit=1,letter=fixed.letter),
                     JointAction('delete',orbit=1),JointAction('freeze'))
            support=() if s.frozen else (program[s.step],)
            logits=s.metric.new_full((len(s.letters),119),-torch.inf);logits[:,8]=0
            result.append(dict(metric_velocity_scaled=torch.zeros_like(s.metric),
                coordinate_velocity_scaled=torch.zeros_like(s.coordinates),element_logits=logits,
                remaining=s.metric.new_tensor(0 if s.frozen else len(program)-s.step),
                actions=support,action_logits=s.metric.new_zeros(len(support))))
        return result


def test_records_every_committed_step_and_both_sides_of_events(tmp_path):
    spec=catalog_spec('space',2);model=Program()
    with TrajectoryWriter(tmp_path,1) as writer:
        begin(writer,0,spec)
        result=sample_batch(model,[spec],seeds=[7],steps=4,epsilon=1e-4,
                            trajectory_callback=writer.record)[0]
        writer.finish(0,dict(status=result.state.status))
        assert not writer.streams
    rows=read(tmp_path/'sample_000000.jsonl');frames=[r for r in rows if r['record_type']=='frame']
    assert result.state.status=='done'
    assert frames[0]['stage']=='initial' and frames[0]['num_atoms']==0
    assert frames[-1]['stage']=='final' and frames[-1]['state_status']=='done'
    assert [f['frame_index'] for f in frames]==list(range(len(frames)))
    assert [f['time'] for f in frames]==sorted(f['time'] for f in frames)
    assert sum(f['stage']=='continuous' for f in frames)==result.integration_steps
    event_frames=[f for f in frames if f['stage']=='event']
    assert [f['event'] for f in event_frames]==result.events
    assert {'birth','split','replace','delete','freeze','element'} <= {f['event']['action']['kind'] for f in event_frames}
    for j,f in enumerate(frames):
        assert len(f['atomic_numbers'])==len(f['coordinates'])==len(f['cartesian_coordinates'])==f['num_atoms']
        if f['stage']=='event':
            assert frames[j-1]['stage']=='continuous' and frames[j-1]['time']==f['time']
            assert frames[j-1]['num_atoms']==f['event']['atoms_before']
            assert f['num_atoms']==f['event']['atoms_after']
    assert rows[-1]['record_type']=='result' and rows[-1]['frame_count']==len(frames)
    assert frames[-1]['coordinate_state']==result.state.coordinates.tolist()


@pytest.mark.parametrize('backend',['equiformer_v2','joint_atom_graph'])
@pytest.mark.parametrize('kind,number',[('space',2),('layer',78),('plane',17)])
def test_recording_does_not_change_real_network_samples(tmp_path,backend,kind,number):
    from unifiedgroupgen.tests.unit.test_joint_flow_v2 import config
    from unifiedgroupgen.pl_modules.factory import build_joint_model
    cfg=config(backend);cfg.update(event_budget=5,max_atoms=24,max_orbits=3)
    torch.manual_seed(17);model=build_joint_model(cfg)
    spec=catalog_spec(kind,number)
    # Native EqV2 uses the global RNG for an auxiliary edge reference frame.
    rng=torch.get_rng_state()
    original=sample_batch(model,[spec],seeds=[7],steps=3,epsilon=.005)[0]
    after=torch.get_rng_state();torch.set_rng_state(rng)
    with TrajectoryWriter(tmp_path,1) as writer:
        begin(writer,0,spec)
        logged=sample_batch(model,[spec],seeds=[7],steps=3,epsilon=.005,
                            trajectory_callback=writer.record)[0]
        writer.finish(0,dict(status=logged.state.status))
    assert torch.equal(after,torch.get_rng_state())
    assert model.training
    assert original.events==logged.events and original.time==logged.time
    assert original.state.elements==logged.state.elements and original.state.letters==logged.state.letters
    assert original.state.status==logged.state.status and original.reason==logged.reason
    assert original.integration_steps==logged.integration_steps
    torch.testing.assert_close(original.state.metric,logged.state.metric,atol=0,rtol=0)
    torch.testing.assert_close(original.state.coordinates,logged.state.coordinates,atol=0,rtol=0)
    final=read(tmp_path/'sample_000000.jsonl')[-2]
    assert final['stage']=='final' and final['periodic']==list(spec.periodic)


def test_failed_sampler_keeps_empty_initial_and_final(tmp_path):
    spec=catalog_spec('space',2)
    with TrajectoryWriter(tmp_path,1) as writer:
        begin(writer,0,spec)
        r=sample_batch(Program(fail=True),[spec],trajectory_callback=writer.record)[0]
        writer.finish(0,dict(status='failed',reason=r.reason))
    frames=[r for r in read(tmp_path/'sample_000000.jsonl') if r['record_type']=='frame']
    assert [r['stage'] for r in frames]==['initial','final']
    assert frames[-1]['state_status']=='failed' and 'fixture graph failure' in frames[-1]['reason']


def test_exact_inventory_is_saved_through_typed_orbit_birth(tmp_path):
    spec=catalog_spec('space',2)
    general=next(w for w in spec.templates if w.dof)
    control=ChemicalControl.from_inputs(counts={'O':general.multiplicity})
    class Inventory(Program):
        def forward_batch(self,states,times,properties=None,actions=None):
            result=[]
            for s in states:
                action=JointAction('freeze') if s.num_atoms else JointAction('birth',letter=general.letter,element=8)
                result.append(dict(metric_velocity_scaled=torch.zeros_like(s.metric),
                    coordinate_velocity_scaled=torch.zeros_like(s.coordinates),
                    element_logits=s.metric.new_zeros((len(s.letters),119)),
                    remaining=s.metric.new_tensor(0 if s.frozen else 2-s.step),
                    actions=() if s.frozen else (action,),action_logits=s.metric.new_zeros(0 if s.frozen else 1)))
            return result
    with TrajectoryWriter(tmp_path,1) as writer:
        writer.start(0,7,spec,2,control,{})
        r=sample_batch(Inventory(),[spec],seeds=[7],steps=4,chemistries=[control],trajectory_callback=writer.record)[0]
        writer.finish(0,dict(status=r.state.status))
    frames=[v for v in read(tmp_path/'sample_000000.jsonl') if v['record_type']=='frame']
    assert r.state.status=='done'
    assert frames[0]['remaining_inventory']=={'8':general.multiplicity}
    assert frames[-1]['remaining_inventory']=={'8':0}
    assert all(not f['num_atoms'] or set(f['atomic_numbers'])=={8} for f in frames)
    assert all(e['action']['kind']!='element' for e in r.events)


def test_nonfinite_state_and_interruption_remain_readable(tmp_path):
    spec=catalog_spec('space',2);s=initial(spec)
    with pytest.raises(RuntimeError,match='interrupted fixture'):
        with TrajectoryWriter(tmp_path,1) as writer:
            begin(writer,0,spec)
            writer.record(0,'final',replace(s,metric=s.metric*float('nan')),0.,0)
            raise RuntimeError('interrupted fixture')
    text=(tmp_path/'sample_000000.jsonl').read_text(encoding='utf-8')
    assert 'NaN' not in text
    rows=read(tmp_path/'sample_000000.jsonl')
    assert rows[1]['geometry_finite'] is False and rows[1]['metric_state']==[None]*len(s.metric)
    assert rows[-1]['record_type']=='interrupted'


@pytest.mark.parametrize('indices',[None,[0],[0,2]])
def test_cli_global_selection_across_batches_and_no_feasible_group(tmp_path,monkeypatch,indices):
    import unifiedgroupgen.joint_flow.cli as cli
    import unifiedgroupgen.common.configuration as configuration
    from unifiedgroupgen.group_constraints.group_prior import NoFeasibleGroup
    spec=catalog_spec('space',2)
    def group(prior,generator,**kwargs):
        if generator.initial_seed()==9:raise NoFeasibleGroup('fixture no feasible group')
        return spec
    monkeypatch.setattr(cli,'sample_group',group)
    monkeypatch.setattr(configuration,'artifact_path',lambda p:Path(p))
    args=SimpleNamespace(count=4,batch_size=2,seed=7,device='cpu',kind='space',group=None,
        steps=4,exact_group=False,case_id=None,checkpoint='fixture.pt',
        trajectory_dir=str(tmp_path/'traces'),trajectory_indices=indices)
    checkpoint={'config':dict(model={},group_prior={},dataset='fixture',generation_task='dng',event_budget=10)}
    output=tmp_path/'samples.jsonl'
    summary=cli.generate(args,checkpoint,Program(),None,ChemicalControl(),{},output)
    chosen=set(range(4) if indices is None else indices)
    assert len(read(output))==4 and summary['trajectories']==len(chosen)
    assert {p.name for p in (tmp_path/'traces').glob('*.jsonl')}=={f'sample_{i:06d}.jsonl' for i in chosen}
    for row in read(output):
        if row['index'] not in chosen:
            assert 'trajectory_path' not in row
            continue
        rows=read(tmp_path/'traces'/f"sample_{row['index']:06d}.jsonl")
        assert rows[0]['index']==row['index'] and rows[0]['seed']==7+row['index']
        assert rows[-1]['sample']==row
        if row['index']==2:
            assert row['status']=='failed' and rows[-1]['frame_count']==0


def test_existing_trajectory_is_not_overwritten(tmp_path):
    p=tmp_path/'sample_000000.jsonl';p.write_text('previous result',encoding='utf-8')
    with pytest.raises(FileExistsError):TrajectoryWriter(tmp_path,1)
    assert p.read_text(encoding='utf-8')=='previous result'


@pytest.mark.parametrize('flags',[['--trajectory-indices','0'],
    ['--trajectory-dir','traces','--trajectory-indices','-1'],
    ['--trajectory-dir','traces','--trajectory-indices','1']])
def test_invalid_cli_selection_rejected_before_checkpoint_load(flags):
    from unifiedgroupgen.joint import main
    with pytest.raises(SystemExit) as error:
        main(['sample','--checkpoint','does-not-exist.pt','--output','unused.jsonl',*flags])
    assert error.value.code==2
