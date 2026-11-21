from types import SimpleNamespace

import pytest
import torch

from unifiedgroupgen.conditional.tasks import validate_task_config
from unifiedgroupgen.experiment_config import compose_experiment, training_definition
from unifiedgroupgen.group_constraints.flow_state import from_state, initial
from unifiedgroupgen.group_constraints.state import JointState
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.pl_data.unified.path import FlowSample
from unifiedgroupgen.pl_modules.unified.objectives import layer_geometry_errors, loss


def test_layer_log_area_matches_the_decoded_cell():
    state=initial(catalog_spec('layer',11),torch.Generator().manual_seed(2),dtype=torch.float64)
    metric_error=state.metric.new_tensor([.3,-.1,.23]).requires_grad_()
    area_error,height_error=layer_geometry_errors(state,metric_error,state.coordinates,None)

    def log_area(metric):
        return torch.linalg.slogdet(state.compiled.lattice(metric)[:2,:2]).logabsdet

    expected=(log_area(state.metric+metric_error)-log_area(state.metric)).square()
    torch.testing.assert_close(area_error,expected)
    assert height_error.item()==0.
    area_error.backward()
    assert torch.isfinite(metric_error.grad).all() and metric_error.grad.norm()>0


def test_layer_height_is_translation_invariant_and_ignores_temporary_atoms():
    spec=catalog_spec('layer',1)
    label=spec.templates[0].letter
    s=initial(spec,torch.Generator().manual_seed(3),dtype=torch.float64)
    state=from_state(JointState(spec,(label,label,label),(14,14,14),s.metric,
                               s.coordinates.new_zeros((3,3))))
    error=s.metric.new_tensor([[4.,0.,1.], [.2,9.,3.], [7.,5.,50.]]).requires_grad_()
    _,height=layer_geometry_errors(state,torch.zeros_like(s.metric),error,(True,True,False))
    assert height.item()==1.
    _,shift=layer_geometry_errors(state,torch.zeros_like(s.metric),
                                   error + error.new_tensor([.8,1.,7.]),(True,True,False))
    torch.testing.assert_close(shift,height)
    height.backward()
    torch.testing.assert_close(error.grad,s.metric.new_tensor([[0.,0.,-1.],[0.,0.,1.],[0.,0.,0.]]))


@pytest.mark.parametrize('kind', ['layer','space'])
def test_objective_wiring_zero_diagnostics_and_layer_scope(kind):
    spec=catalog_spec(kind,1)
    label=spec.templates[0].letter
    src=initial(spec,torch.Generator().manual_seed(4),dtype=torch.float64)
    coords=src.metric.new_zeros((2,3))
    state=from_state(JointState(spec,(label,label),(14,14),src.metric,coords))
    pred=dict(metric_velocity_scaled=torch.ones_like(src.metric,requires_grad=True),
              coordinate_velocity_scaled=coords.new_tensor([[0.,0.,-1.],[0.,0.,1.]]),
              remaining=src.metric.new_zeros(()))
    model=SimpleNamespace(forward_batch=lambda *_:[pred])
    sample=FlowSample(state,.7,src.metric,coords,(14,14),0,None,{},'probe',
                      reference_orbits=(True,True))
    base=loss(model,[sample])
    off=loss(model,[sample],layer_log_area_weight=0.,layer_height_weight=0.)
    on=loss(model,[sample],layer_log_area_weight=1.,layer_height_weight=.25)
    torch.testing.assert_close(base['loss'],off['loss'],atol=0,rtol=0)
    if kind=='layer':
        torch.testing.assert_close(on['loss']-base['loss'],base['loss'].new_tensor(1.25))
    else:
        torch.testing.assert_close(on['loss'],base['loss'],atol=0,rtol=0)
        assert on['layer_log_area'].item()==on['layer_height'].item()==0.


def test_layer_weight_config():
    name='full_mp20_eqv2_csp'
    base=training_definition(compose_experiment(name))
    assert not {'layer_log_area_weight','layer_height_weight'} & set(base)
    cfg=training_definition(compose_experiment(name,
        overrides=['train.layer_log_area_weight=1.0','train.layer_height_weight=0.25']))
    assert cfg['layer_log_area_weight']==1. and cfg['layer_height_weight']==.25
    with pytest.raises(ValueError,match='layer_height_weight'):
        validate_task_config(dict(cfg,layer_height_weight=-1))
