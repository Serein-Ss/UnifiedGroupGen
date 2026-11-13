"""Generic crystal feature contract, legacy recovery and real-data gradients."""
from dataclasses import replace
from pathlib import Path
import json
import pytest
import torch
from unifiedgroupgen.pl_modules.joint_graph import JointAtomGraph
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_modules.joint_training import load_joint_checkpoint, fit_joint
from unifiedgroupgen.pl_data.joint_path import target_from_record, sample_training_path
from unifiedgroupgen.pl_data.task_path import task_target
from unifiedgroupgen.group_constraints.state import initial_state, apply_action, JointAction
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.pl_modules.objectives.joint_generator import joint_generator_loss

ROOT = Path(__file__).resolve().parents[3]


def configuration(task='dng'):
    return dict(model=dict(family='joint_symmetry_v1', feature_schema='nonmagnetic_v1',
                           hidden=16,layers=1,heads=4,constraint_layers=True,task_controls=True),
                allowed_elements=list(range(1,119)), properties={'band_gap':dict(type='scalar')} if task=='property' else {},
                generation_task=task, cond_dropout=0., chemical_set_probability=0.,
                max_atoms=192,max_orbits=52,event_budget=96)


def record(dataset, split='train', labelled=False):
    path=ROOT/'data/dataset/full_verified_v1'/dataset/(split+'.jsonl')
    with path.open(encoding='utf-8') as stream:
        return next(r for line in stream if (r:=json.loads(line))
                    if not labelled or r.get('properties',{}).get('band_gap') is not None)


def test_default_has_no_auxiliary_field_parameters():
    model=JointAtomGraph([8,14],hidden=16,layers=2,heads=4)
    assert model.feature_schema=='nonmagnetic_v1'
    assert model.moment_encoder is None and model.moment_head is None
    assert not any('moment' in name for name,_ in model.named_parameters())
    assert model.global_encoder[0].in_features==17
    assert model.operation_encoder[0].in_features==12
    assert all(block.edge[0].in_features==24 for block in model.blocks)


@pytest.mark.parametrize('kind,number',[('space',2),('layer',78),('plane',17)])
def test_three_families_have_finite_feature_graph_and_legal_decode(kind,number):
    model=build_joint_model(configuration())
    state=initial_state(catalog_spec(kind,number),budget=96)
    nodes,edges=model._features(state,.1,{})
    assert nodes.shape==(1,16) and edges.shape==(1,1,24)
    w=next(w for w in state.spec.templates if w.dof)
    state=apply_action(state,JointAction('birth',letter=w.letter))
    state=apply_action(state,JointAction('element',orbit=0,element=8))
    nodes,edges=model._features(state,.5,{})
    assert edges.shape==(state.num_atoms+1,state.num_atoms+1,24)
    assert torch.isfinite(nodes).all() and torch.isfinite(edges).all()
    decoded=state.decoded()
    residual=state.compiled.residuals(decoded['coordinates'],decoded['atomic_numbers'],decoded['lattice'])
    assert residual['elements'] and residual['coordinates']<2e-5


@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
@pytest.mark.parametrize('task',['dng','property','csp'])
def test_actual_task_paths_train_generic_heads(dataset,task):
    torch.manual_seed(17)
    cfg=configuration(task);model=build_joint_model(cfg)
    target=task_target(target_from_record(record(dataset,labelled=task=='property')),cfg,31)
    sample=sample_training_path(target,31,time=.67,allowed_elements=cfg['allowed_elements'],budget=96)
    loss=joint_generator_loss(model,[sample])
    loss['loss'].backward()
    for name in ['metric_head.weight','coordinate_head.weight','count_head.weight','action_head.3.weight']:
        gradient=dict(model.named_parameters())[name].grad
        assert gradient is not None and torch.isfinite(gradient).all(),name
    assert model.moment_head is None
    assert not any('moment' in name for name in model.state_dict())


def test_periodic_feature_values_are_invariant_to_integer_cell_translation():
    model=build_joint_model(configuration());state=initial_state(catalog_spec('space',1),budget=96)
    state=apply_action(state,JointAction('birth',letter='a'))
    state=apply_action(state,JointAction('element',orbit=0,element=8))
    state=replace(state,coordinates=state.coordinates.new_tensor([[.2,.3,.4]]))
    shifted=replace(state,coordinates=state.coordinates+state.coordinates.new_tensor([[1.,-2.,3.]]))
    a,ea=model._features(state,.4,{});b,eb=model._features(shifted,.4,{})
    torch.testing.assert_close(a,b,atol=2e-5,rtol=2e-5)
    torch.testing.assert_close(ea,eb,atol=2e-5,rtol=2e-5)


def test_generic_schema_does_not_silently_accept_an_internal_field():
    state=initial_state(catalog_spec('space',1))
    state=replace(state,moments=torch.zeros((0,3)))
    with pytest.raises(ValueError,match='nonmagnetic feature schema'):
        build_joint_model(configuration())(state,.5)


def test_unversioned_checkpoint_preserves_exact_predictions(tmp_path):
    from unifiedgroupgen.pl_modules.joint_training import FORMAT
    cfg=configuration()
    cfg['model'].pop('feature_schema')
    original=JointAtomGraph(cfg['allowed_elements'],hidden=16,layers=1,heads=4,
                           constraint_layers=True,task_controls=True,cond_dropout=0.,
                           feature_schema='legacy_v1')
    original.eval()
    state=initial_state(catalog_spec('space',2),budget=96)
    state=apply_action(state,JointAction('birth',letter='i'))
    state=apply_action(state,JointAction('element',orbit=0,element=16))
    path=tmp_path/'unversioned.pt'
    torch.save(dict(format=FORMAT,config=cfg,model=original.state_dict()),path)
    value=load_joint_checkpoint(path)
    rebuilt=build_joint_model(value['config']);rebuilt.load_state_dict(value['model'],strict=True);rebuilt.eval()
    assert rebuilt.feature_schema=='legacy_v1'
    with torch.no_grad():
        expected=original(state,.67,{})
        actual=rebuilt(state,.67,{})
    for name in ['metric_anchor','coordinate_anchor','remaining','action_logits']:
        torch.testing.assert_close(actual[name],expected[name],rtol=0,atol=0)


@pytest.mark.parametrize('dataset',['mp20','c2db51'])
@pytest.mark.parametrize('task',['dng','property','csp'])
def test_current_selectors_explicitly_choose_generic_features(dataset,task):
    from unifiedgroupgen.common.configuration import compose_experiment,training_definition
    cfg=training_definition(compose_experiment('task_'+dataset+'_'+task))
    assert cfg['model']['feature_schema']=='nonmagnetic_v1'


def test_new_native_checkpoint_reloads_and_resumes_same_architecture(tmp_path):
    from lightning_fabric.utilities.seed import seed_everything
    cfg=configuration();train=[record('mp_20'),record('c2db_51')]
    val=[record('mp_20','val'),record('c2db_51','val')]
    seed_everything(23,workers=True)
    output=tmp_path/'model.pt'
    fit_joint(cfg,train,val,output,epochs=1,max_steps=1,runtime=dict(batch_size=2,edge_budget=200000))
    saved=load_joint_checkpoint(output);native=load_joint_checkpoint(tmp_path/'last.ckpt')
    assert saved['config']['model']['feature_schema']=='nonmagnetic_v1'
    for value in [saved,native]:
        model=build_joint_model(value['config']);model.load_state_dict(value['model'],strict=True)
        assert model.moment_encoder is None
    continued=fit_joint(cfg,train,val,output,epochs=2,max_steps=2,
                        resume=tmp_path/'last.ckpt',runtime=dict(batch_size=2,edge_budget=200000))
    assert continued.global_step==2
