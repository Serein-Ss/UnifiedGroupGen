"""Hard composition support, task paths and checkpoint-compatible independent tasks."""
from dataclasses import replace
import pytest
import torch
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl, can_complete, orbit_counts
from unifiedgroupgen.group_constraints.state import initial_state, legal_actions, apply_action, JointAction
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.pl_data.joint_path import target_from_record, sample_training_path
from unifiedgroupgen.pl_data.task_path import task_target, validate_task_config
from unifiedgroupgen.group_constraints.group_prior import fit_group_prior, sample_group, NoFeasibleGroup
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_modules.objectives.joint_generator import joint_generator_loss
from unifiedgroupgen.tests.unit.test_joint_generator import real_record


def config(task):
    return dict(generation_task=task, model=dict(family='joint_symmetry_v1', hidden=16, layers=1,
                heads=4, task_controls=True, constraint_layers=True), allowed_elements=list(range(1,119)),
                properties={'band_gap':dict(type='scalar',mean=0.,std=1.)} if task=='property' else {},
                max_atoms=192,max_orbits=52,event_budget=96,cond_dropout=0.,chemical_set_probability=1.)


@pytest.mark.parametrize('kind,number',[('space',1),('layer',1),('plane',1)])
def test_allowed_set_is_not_an_all_elements_requirement(kind,number):
    chemistry=ChemicalControl.from_inputs(['Li','Fe','O'])
    state=initial_state(catalog_spec(kind,number),budget=8,chemistry=chemistry)
    letter=next(w.letter for w in state.spec.templates if w.multiplicity==1 and w.dof)
    state=apply_action(state,JointAction('birth',letter=letter))
    state=apply_action(state,JointAction('element',orbit=0,element=8))
    assert JointAction('end') in legal_actions(state,range(1,119))
    state=apply_action(state,JointAction('end'))
    assert orbit_counts(state.spec,state.letters,state.elements)=={8:1}
    assert all(a.element in (0,3,8,26) for a in legal_actions(replace(state,status='active'),range(1,119)))
    with pytest.raises(ValueError,match='forbidden'):
        replace(state,elements=(14,))


def test_inventory_joint_birth_and_exact_end_with_failure_slot():
    control=ChemicalControl.from_inputs(counts={'Ti':1,'O':2})
    state=initial_state(catalog_spec('space',1),budget=5,chemistry=control)
    assert JointAction('end') not in legal_actions(state,range(1,119))
    for z in (22,8,8):
        support=legal_actions(state,range(1,119))
        action=JointAction('birth',letter='a',element=z)
        assert action in support
        state=apply_action(state,action,check=support)
        assert all(n>=0 for _,n in state.chemistry.remaining(state.spec,state.letters,state.elements))
    assert legal_actions(state,range(1,119))==(JointAction('end'),)
    state=apply_action(state,JointAction('end'))
    assert state.status=='done' and state.num_atoms==3
    assert not can_complete(state.spec,(),(),control,2)
    too_short=initial_state(state.spec,budget=4,chemistry=control)
    assert legal_actions(too_short,range(1,119))==(JointAction('fail'),)


def test_special_position_capacity_is_shared_between_elements():
    spec=catalog_spec('space',2)  # eight distinct multiplicity-one inversion centres
    assert not can_complete(spec,(),(),ChemicalControl.from_inputs(counts={z:1 for z in range(1,10)}),20)
    assert can_complete(spec,(),(),ChemicalControl.from_inputs(counts={z:1 for z in range(1,9)}),20)
    assert not can_complete(catalog_spec('space',225),(),(),ChemicalControl.from_inputs(counts={14:1}),20)


@pytest.mark.parametrize('task',['dng','property','csp'])
@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
def test_real_task_teacher_support_endpoint_and_shared_gradient(task,dataset):
    cfg=config(task)
    target=target_from_record(real_record(dataset))
    for seed in range(5):
        prepared=task_target(target,cfg,seed)
        sample=sample_training_path(prepared,seed,time=.45,allowed_elements=cfg['allowed_elements'],budget=96)
        assert sample.next_action in legal_actions(sample.state,cfg['allowed_elements'])
        assert bool(sample.properties)==(task=='property')
        if task=='csp':
            assert sample.state.chemistry.counts and all(sample.state.elements)
        final=sample_training_path(prepared,seed,time=1.,allowed_elements=cfg['allowed_elements'],budget=96)
        assert final.state.status=='done'
        assert orbit_counts(final.state.spec,final.state.letters,final.state.elements)==orbit_counts(target.state.spec,target.state.letters,target.state.elements)
        expected=target.state.decoded(wrap=False)
        actual=final.state.decoded(wrap=False)
        torch.testing.assert_close(actual['lattice'],expected['lattice'])
        # Orbit instances may be permuted; every role still reaches its actual block.
        for z in set(target.state.elements):
            lhs=actual['coordinates'][actual['atomic_numbers']==z]
            rhs=expected['coordinates'][expected['atomic_numbers']==z]
            delta=lhs[:,None]-rhs[None]
            delta=delta-torch.tensor(target.state.spec.periodic)*delta.round()
            assert (delta.norm(dim=-1).min(1).values<2e-5).all()
    model=build_joint_model(cfg)
    loss=joint_generator_loss(model,[sample])['loss'];loss.backward()
    assert torch.isfinite(loss) and model.chemical_encoder[0].weight.grad.abs().sum()>0
    rebuilt=build_joint_model(cfg);rebuilt.load_state_dict(model.state_dict(),strict=True)


def test_chemical_features_expose_unknown_versus_fixed_inventory():
    state=initial_state(catalog_spec('space',1),budget=16)
    controlled=replace(state,chemistry=ChemicalControl.from_inputs(counts={8:2}))
    a=state.chemistry.features(state,range(1,119));b=controlled.chemistry.features(controlled,range(1,119))
    assert a.shape==b.shape==(357,) and not torch.equal(a,b)
    model=build_joint_model(config('csp')).eval()
    x=model(state,0.,actions=())['metric_anchor'];y=model(controlled,0.,actions=())['metric_anchor']
    assert not torch.equal(x,y)


def test_empirical_prior_type_support_and_no_test_group_input():
    rows=[real_record('mp_20'),real_record('c2db_51')]
    prior=fit_group_prior(rows)
    spec=sample_group(prior,torch.Generator().manual_seed(0),kind='layer')
    assert spec.kind=='layer' and spec.number==rows[1]['descriptor']['number']
    with pytest.raises(NoFeasibleGroup):
        sample_group(prior,torch.Generator().manual_seed(0),kind='plane')
    with pytest.raises(ValueError,match='TRAIN'):
        sample_group({**prior,'source_split':'test'},torch.Generator())


@pytest.mark.parametrize('task',['dng','property','csp'])
@pytest.mark.parametrize('dataset',['mp20','c2db51'])
def test_six_hydra_selectors_are_independent(task,dataset):
    from unifiedgroupgen.common.configuration import compose_experiment,training_definition
    cfg=training_definition(compose_experiment(f'task_{dataset}_{task}'))
    assert cfg['generation_task']==task and cfg['model']['task_controls']
    assert cfg['model']['constraint_layers'] and not cfg['model']['operation_compiler']
    validate_task_config(cfg)


def test_properties_cannot_hide_chemistry_or_source_context():
    cfg=config('property');cfg['properties']={'source_domain':dict(type='scalar')}
    with pytest.raises(ValueError):validate_task_config(cfg)
    with pytest.raises(ValueError):ChemicalControl.from_inputs(counts={'O':0})
    with pytest.raises(ValueError):ChemicalControl.from_inputs(counts={'O':1.5})


@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
def test_csp_match_uses_post_generation_reference_and_keeps_missing_denominator(dataset):
    from unifiedgroupgen.evaluation.joint_csp import target_view,match_csp,evaluate_csp
    record=real_record(dataset);view=target_view(record)
    sample={**view,'dataset':record['dataset'],'case_id':record['id'],'candidate':0}
    matched=match_csp(view,sample)
    assert matched['matched'] and matched['normalized_rmse']<1e-6
    report=evaluate_csp([sample],[record])
    assert report['summary']['1']['match_rate']==1
    report=evaluate_csp([],[record])
    assert report['summary']['1']['match_rate']==0
    assert report['summary']['1']['matched_only_normalized_rmse'] is None
    shifted={**view,'coordinates':[[p[0]+.231,p[1]-.14,-p[2]+7.] for p in view['coordinates']]}
    if view['kind']=='layer':
        assert match_csp(view,shifted)['matched']
        vacuum={**shifted,'lattice':[row[:] for row in shifted['lattice']]};vacuum['lattice'][2][2]=10000
        assert match_csp(view,vacuum)['matched']
    wrong={**view,'atomic_numbers':[1]*len(view['atomic_numbers'])}
    assert not match_csp(view,wrong)['composition_match']


@pytest.mark.parametrize('kind,limit',[('space',230),('layer',80),('plane',17)])
def test_inventory_kernel_all_standard_groups(kind,limit):
    for number in range(1,limit+1):
        spec=catalog_spec(kind,number)
        w=min(spec.templates,key=lambda x:(x.multiplicity,x.dof))
        chemistry=ChemicalControl.from_inputs(counts={14:w.multiplicity})
        state=initial_state(spec,budget=4,chemistry=chemistry,generator=torch.Generator().manual_seed(number))
        options=legal_actions(state,(14,))
        action=JointAction('birth',letter=w.letter,element=14)
        assert action in options,(kind,number)
        state=apply_action(state,action,check=options)
        state=apply_action(state,JointAction('end'),check=legal_actions(state,(14,)))
        decoded=state.decoded();residual=state.compiled.residuals(decoded['coordinates'],decoded['atomic_numbers'],decoded['lattice'])
        assert residual['elements'] and residual['coordinates']<2e-4,(kind,number,residual)


@pytest.mark.parametrize('kind',['space','layer','plane'])
def test_inventory_oracle_runs_actual_event_sampler(kind):
    from unifiedgroupgen.generation.joint import sample_joint
    class Oracle(torch.nn.Module):
        def __init__(self):
            super().__init__();self.p=torch.nn.Parameter(torch.zeros(()))
            self.allowed_elements=(8,22);self.max_atoms=16;self.max_orbits=8
        def forward(self,state,time,properties=None,actions=None):
            options=legal_actions(state,self.allowed_elements,self.max_atoms,self.max_orbits) if actions is None else tuple(actions)
            desired=next((a for a in options if a.kind=='birth'),next((a for a in options if a.kind=='end'),None))
            logits=state.metric.new_full((len(options),),-100.)
            if desired is not None:logits[options.index(desired)]=100.
            remaining=sum(n for _,n in state.chemistry.remaining(state.spec,state.letters,state.elements))+1
            return dict(metric_anchor=torch.zeros_like(state.metric),coordinate_anchor=torch.zeros_like(state.coordinates),
                        moment_anchor=None,remaining=state.metric.new_tensor(remaining if state.status=='active' else 0),
                        actions=options,action_logits=logits)
    result=sample_joint(Oracle(),catalog_spec(kind,1),steps=12,budget=8,
                        chemistry=ChemicalControl.from_inputs(counts={8:2,22:1}),
                        generator=torch.Generator().manual_seed(42))
    assert result.state.status=='done' and result.state.num_atoms==3
    assert orbit_counts(result.state.spec,result.state.letters,result.state.elements)=={8:2,22:1}
    assert all(event['atoms_after']<=3 for event in result.events)


@pytest.mark.parametrize('task',['dng','property','csp'])
def test_independent_hard_task_native_resume_is_exact(tmp_path,task):
    from pytorch_lightning import seed_everything
    from unifiedgroupgen.pl_modules.joint_training import fit_joint,load_joint_checkpoint
    cfg=config(task);train=[real_record('mp_20'),real_record('c2db_51')]
    val=[real_record('mp_20','val'),real_record('c2db_51','val')]
    runtime=dict(batch_size=1,edge_budget=200000)
    seed_everything(142,workers=True)
    full=tmp_path/'full/model.pt';fit_joint(cfg,train,val,full,epochs=3,runtime=runtime,max_steps=3)
    seed_everything(142,workers=True)
    partial=tmp_path/'partial/model.pt';fit_joint(cfg,train,val,partial,epochs=3,runtime=runtime,max_steps=1)
    fit_joint(cfg,train,val,partial,epochs=3,runtime=runtime,max_steps=3,resume=partial.parent/'last.ckpt')
    expected,actual=load_joint_checkpoint(full)['model'],load_joint_checkpoint(partial)['model']
    for name in expected:torch.testing.assert_close(expected[name],actual[name],rtol=0,atol=0)



def test_property_objective_keeps_missing_labels_missing():
    from unifiedgroupgen.pl_modules.joint_training import JointGeneratorModule
    good=real_record('mp_20');missing={**real_record('c2db_51'),'properties':{'band_gap':None}}
    module=JointGeneratorModule(config('property'))
    result=module.objective([good,missing],validation=True)
    assert torch.isfinite(result['loss'])
    assert (missing['dataset'],missing['id']) not in module._targets
    with pytest.raises(ValueError,match='No observed'):
        module.objective([missing],validation=True)



def test_property_values_change_predictions_without_changing_hard_support():
    cfg=config('property');cfg['properties']['formation_energy']=dict(type='scalar')
    record=real_record('mp_20');target=task_target(target_from_record(record),cfg,10)
    sample=sample_training_path(target,10,time=.4,allowed_elements=cfg['allowed_elements'],budget=96)
    model=build_joint_model(cfg).eval()
    first=model(sample.state,sample.time,{'band_gap':0.,'formation_energy':-1.})
    second=model(sample.state,sample.time,{'band_gap':2.,'formation_energy':-3.})
    assert first['actions']==second['actions']
    assert not torch.equal(first['metric_anchor'],second['metric_anchor'])
    assert not torch.equal(first['action_logits'],second['action_logits'])
    with pytest.raises(ValueError):ChemicalControl.from_inputs(counts={'O':True})
    with pytest.raises(ValueError):ChemicalControl.from_inputs(counts=[8,8])



def test_multi_property_subset_training_configuration_is_wired():
    from unifiedgroupgen.common.configuration import compose_experiment,training_definition
    cfg=training_definition(compose_experiment('task_mp20_property',overrides=['conditioning=multi_property']))
    cfg['allowed_elements']=[8,22]
    model=build_joint_model(cfg)
    assert model.property_encoder.subset_training
    assert set(model.property_encoder.configs)=={'band_gap','formation_energy','energy_above_hull'}



def test_operation_recompilation_preserves_exact_inventory():
    from unifiedgroupgen.pl_data.operation_path import recompile_target
    target=task_target(target_from_record(real_record('c2db_51')),config('csp'),4)
    converted=recompile_target(target)
    assert converted.state.chemistry==target.state.chemistry
    end=sample_training_path(converted,4,time=1.,allowed_elements=range(1,119),budget=96)
    assert end.state.status=='done' and end.state.chemistry.counts==target.state.chemistry.counts
