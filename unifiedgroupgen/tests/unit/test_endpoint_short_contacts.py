from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from unifiedgroupgen.conditional.tasks import validate_task_config
from unifiedgroupgen.experiment_config import compose_experiment, training_definition
from unifiedgroupgen.group_constraints.flow_state import initial
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.pl_modules.unified.objectives import loss
from unifiedgroupgen.pl_modules.unified.short_contacts import _periodic_short_contacts


def test_layer_wraps_in_plane_only_and_has_coordinate_and_cell_gradients():
    cell = torch.diag(torch.tensor([2., 2., 1.], dtype=torch.float64)).requires_grad_()
    coords = torch.tensor([[.99, 0., 0.], [.04, 0., .01]], dtype=torch.float64, requires_grad=True)
    value = _periodic_short_contacts(coords, cell, (True, True, False), .6)
    assert value.item() == pytest.approx((1-(.1**2 + .01**2)**.5/.6)**2/2)
    value.backward()
    assert coords.grad[1, 0] < 0 and cell.grad[0, 0] < 0
    across_z = torch.tensor([[.99, 0., 0.], [.04, 0., .96]], dtype=torch.float64)
    assert _periodic_short_contacts(across_z, cell, (True, True, False), .6).item() == 0.
    assert _periodic_short_contacts(across_z, cell, (True, True, True), .6).item() > 0.


def test_skew_short_periodic_self_image_keeps_lattice_gradient():
    cell = torch.tensor([[.8,0.,0.], [.75,.2,0.], [0.,0.,1.]], dtype=torch.float64,
                        requires_grad=True)
    coords = torch.zeros((1, 3), dtype=cell.dtype)
    value = _periodic_short_contacts(coords, cell, (True,True,False), .6)
    assert value.item() == pytest.approx((1-(.05**2+.2**2)**.5/.6)**2)
    value.backward()
    assert torch.isfinite(cell.grad).all() and cell.grad[1, 1] < 0


def test_auxiliary_loss_is_gated_and_zero_weight_preserves_the_objective():
    state = initial(catalog_spec('space',1), torch.Generator().manual_seed(2), dtype=torch.float64)
    template = state.spec.templates[0]
    coords = state.metric.new_tensor([[0.,0.,0.],[.04,0.,0.]])
    velocity = torch.zeros_like(coords, requires_grad=True)
    state = replace(state, metric=torch.zeros_like(state.metric), coordinates=coords,
                    letters=(template.letter,template.letter), elements=(14,14), frozen=True)
    prediction = dict(metric_velocity_scaled=state.metric, coordinate_velocity_scaled=velocity,
                      remaining=state.metric.new_zeros(()))
    model = SimpleNamespace(forward_batch=lambda *_:[prediction])
    sample = SimpleNamespace(state=state,time=.8,properties={},metric_anchor=state.metric,
                             coordinate_anchor=coords,remaining=0)
    base = loss(model,[sample])
    off = loss(model,[sample],short_contact_weight=0.)
    on = loss(model,[sample],short_contact_weight=1.)
    assert 'short_contact' not in base and off['loss'].item()==base['loss'].item()
    assert off['short_contact'].item()==on['short_contact'].item()>0
    assert on['short_contact_eligible'].item()==on['short_contact_violation'].item()==1
    on['loss'].backward()
    assert velocity.grad[1,0] < 0
    sample.state = replace(state,frozen=False)
    active = loss(model,[sample],short_contact_weight=1.)
    assert active['short_contact_eligible'].item()==active['short_contact'].item()==0


def test_hydra_default_and_weight_validation():
    experiment = 'full_mp20_eqv2_csp'
    assert 'short_contact_weight' not in training_definition(compose_experiment(experiment))
    value = training_definition(compose_experiment(experiment, overrides=['train.short_contact_weight=0.5']))
    assert value['short_contact_weight'] == .5
    with pytest.raises(ValueError, match='short_contact_weight'):
        validate_task_config(dict(value, short_contact_weight=-1))
