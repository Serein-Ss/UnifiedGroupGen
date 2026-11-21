"""CFG branch visibility, stable signed fields and legal stochastic rates."""
from types import SimpleNamespace

import pytest
import torch

from unifiedgroupgen.conditional.guidance import guided_forward, validate_guidance
from unifiedgroupgen.tests.unit.test_joint_flow_v2 import config, record
from unifiedgroupgen.common.data_utils import target_from_record
from unifiedgroupgen.pl_data.unified.tasks import task_target
from unifiedgroupgen.pl_data.unified.path import sample_path
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.generation.unified.sampling import sample_batch
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl
from unifiedgroupgen.group_constraints.state import JointAction
from unifiedgroupgen.group_constraints.unified import catalog_spec


@pytest.mark.parametrize('task',['dng','csp'])
def test_property_cfg_preserves_constraints_and_trains_null_embedding(task,monkeypatch):
    cfg=config('joint_atom_graph',task,True)
    cfg['cond_dropout']=.5
    model=build_joint_model(cfg)
    endpoint=task_target(target_from_record(record('mp_20')),cfg,5)
    sample=sample_path(endpoint,5,time=.65,allowed_elements=cfg['allowed_elements'])
    state=sample.state
    encoder=model.property_encoder
    with torch.no_grad():
        encoder.null['band_gap'].fill_(.2)
    like=torch.zeros((2,encoder.hidden))
    model.eval()
    encoder.train()
    conditioned=[sample.properties]*2
    with monkeypatch.context() as patch:
        patch.setattr(torch,'rand',lambda count,device:torch.tensor([.1,.9],device=device))
        explicit=encoder.forward_batch(conditioned,like)
    empty=encoder.forward_batch([{},{}],like)
    torch.testing.assert_close(explicit[0],empty[0])
    assert not torch.equal(explicit[1],empty[1])
    explicit[0].sum().backward()
    assert encoder.null['band_gap'].grad is not None
    model.eval()
    with torch.no_grad():
        null=guided_forward(model,[state],[sample.time],[sample.properties],None,0.)[0]
        direct=guided_forward(model,[state],[sample.time],[sample.properties],None,1.)[0]
        guided=guided_forward(model,[state],[sample.time],[sample.properties],None,2.)[0]
    assert null['actions']==direct['actions']==guided['actions']
    assert torch.equal(torch.isneginf(guided['element_logits']),torch.isneginf(direct['element_logits']))
    assert not torch.isnan(guided['element_logits']).any()
    assert 0<=guided['remaining']<=state.budget-state.step+1e-5
    torch.testing.assert_close(guided['metric_velocity_scaled'],2*direct['metric_velocity_scaled']-null['metric_velocity_scaled'])
    assert state.chemistry.counts==endpoint.state.chemistry.counts


@pytest.mark.parametrize('scale,has_properties,dropout', [(2,False,.1),(2,True,0.),(2,True,1.),(-1,True,.1),(float('nan'),True,.1)])
def test_invalid_cfg_requests_fail(scale,has_properties,dropout):
    with pytest.raises(ValueError):
        validate_guidance(scale,has_properties,dropout)


def test_cfg_sampler_keeps_inventory_in_both_passes_and_freezes():
    chemistry=ChemicalControl.from_inputs(counts={'O':1})
    seen=set()

    class InventoryOracle(torch.nn.Module):
        event_budget=8
        birth_scale=.2
        property_encoder=SimpleNamespace(dropout=.1)

        def __init__(self):
            super().__init__()
            self.anchor=torch.nn.Parameter(torch.zeros(()))

        def forward_batch(self,states,times,properties,actions=None):
            result=[]
            for state,props in zip(states,properties):
                assert state.chemistry==chemistry
                seen.add(bool(props))
                edits=() if state.frozen else ((JointAction('freeze'),) if state.num_atoms else
                                                (JointAction('birth',letter='a',element=8),))
                if actions is not None:
                    edits=()
                logits=state.metric.new_full((len(state.letters),119),-torch.inf)
                logits[:,8]=0
                result.append(dict(metric_velocity_scaled=-state.metric,
                    coordinate_velocity_scaled=-state.coordinates,element_logits=logits,
                    remaining=state.metric.new_tensor(0. if state.frozen else (3. if props else 2.)),
                    actions=edits,action_logits=state.metric.new_zeros(len(edits))))
            return result

    result=sample_batch(InventoryOracle(),[catalog_spec('space',1)],properties=[{'band_gap':1.5}],
                        chemistries=[chemistry],seeds=[1],steps=8,epsilon=.001,guidance_scale=1.5)[0]
    assert seen=={False,True}
    assert result.state.status=='done' and result.state.elements==(8,)
    assert [event['action']['kind'] for event in result.events]==['birth','freeze']
