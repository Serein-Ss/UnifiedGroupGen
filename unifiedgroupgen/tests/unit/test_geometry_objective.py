"""Verify shape/size separation and orbit weighting without changing the base flow."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from unifiedgroupgen.experiment_config import compose_experiment, training_definition
from unifiedgroupgen.group_constraints.flow_state import JointAction, apply, initial
from unifiedgroupgen.group_constraints.operations import operation_spec
from unifiedgroupgen.pl_data.unified.path import FlowSample
from unifiedgroupgen.pl_modules.unified.objectives import geometry_errors, loss


def metric_coefficients(state, matrix):
    return torch.einsum('kij,ij->k',matrix.new_tensor(state.compiled.standard),matrix)


@pytest.mark.parametrize('kind,rank',[('space',3),('layer',2),('rod',1)])
def test_isotropic_change_only_has_size_error(kind,rank):
    state=initial(operation_spec(kind,1),dtype=torch.float64)
    isotropic=0.2*torch.eye(rank,dtype=state.metric.dtype)
    values=metric_coefficients(state,isotropic)
    shape,size,site,scaled=geometry_errors(state,values,state.coordinates)
    torch.testing.assert_close(shape,torch.zeros_like(shape),atol=1e-14,rtol=0)
    torch.testing.assert_close(size,state.metric.new_tensor((.1*rank)**2))
    assert site==0
    torch.testing.assert_close(scaled,state.metric.new_tensor(.04*rank))
    before=state.compiled.lattice(state.metric)
    after=state.compiled.lattice(state.metric+values)
    log_measure=torch.logdet(after)-torch.logdet(before)
    torch.testing.assert_close(log_measure.square(),size)


def test_shape_has_zero_volume_and_fixed_orbits_do_not_dilute_site():
    state=initial(operation_spec('space',2),dtype=torch.float64)
    state=replace(state,metric=torch.zeros_like(state.metric))
    free=next(w for w in state.spec.templates if w.dof==3)
    fixed=next(w for w in state.spec.templates if w.dof==0)
    state=apply(state,JointAction('birth',letter=free.letter,element=8),torch.Generator().manual_seed(87))
    state=apply(state,JointAction('birth',letter=fixed.letter,element=14),torch.Generator().manual_seed(98))
    error=metric_coefficients(state,torch.diag(state.metric.new_tensor([.3,-.3,0])))
    delta=state.coordinates.new_full(state.coordinates.shape,.2)
    delta[-1]=8.
    shape,size,site,scaled=geometry_errors(state,error,delta)
    torch.testing.assert_close(shape,state.metric.new_tensor(.18/5))
    torch.testing.assert_close(site,state.metric.new_tensor(.04))
    assert size==0 and scaled==0


def test_objective_reports_same_components_and_keeps_original_without_override():
    s=initial(operation_spec('space',1))
    coeff=metric_coefficients(s,torch.eye(3)).requires_grad_()
    sample=FlowSample(s,.3,s.metric,s.coordinates,(),0,None,{},'probe')
    def fake_forward(states,times,properties):
        v=s.metric.new_zeros(())
        return [dict(metric_velocity_scaled=coeff,coordinate_velocity_scaled=s.coordinates,
                     remaining=v,element_logits=s.metric.new_zeros((0,119)),actions=())]
    model=SimpleNamespace(forward_batch=fake_forward)
    original=loss(model,[sample])
    legacy=loss(model,[sample],geometry_objective='legacy')
    split=loss(model,[sample],geometry_objective='decomposed')
    torch.testing.assert_close(original['loss'],legacy['loss'],atol=0,rtol=0)
    assert split['loss'] != legacy['loss']
    for key in ('lattice_shape','lattice_log_measure','orbit_site','geometry_legacy','element','jump'):
        torch.testing.assert_close(split[key],legacy[key],atol=0,rtol=0)
    split['loss'].backward()
    assert torch.isfinite(coeff.grad).all() and coeff.grad.norm()>0


def test_hydra_geometry_override_keeps_legacy_defaults_compatible():
    name='full_mp20_eqv2_csp'
    default=training_definition(compose_experiment(name))
    updated=training_definition(compose_experiment(name,overrides=['train.geometry_objective=decomposed']))
    assert 'geometry_objective' not in default
    assert updated.pop('geometry_objective')=='decomposed'
    assert updated==default

