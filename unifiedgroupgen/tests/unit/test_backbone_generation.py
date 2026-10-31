"""Bounded CUDA generation checks for optional official cores; no trained-quality claim."""
import importlib.util
import pytest
import torch

from unifiedgroupgen.group_constraints.compiler import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.generation.flow import sample_flow_batch
from unifiedgroupgen.pl_modules.factory import build_models
from unifiedgroupgen.common.configuration import compose_experiment, training_definition


@pytest.mark.parametrize('name,target,options', [
    ('cspnet','CSPNetBackbone',dict(layers=1,frequencies=2)),
    ('mattergen','MatterGenBackbone',dict(blocks=1,cutoff=4.,max_neighbors=16)),
    ('equiformer_v2','EquiformerV2Backbone',dict(layers=1,channels=16,lmax=2,mmax=1,cutoff=4.,max_neighbors=16))])
def test_official_backbone_packed_generation_in_inference_mode(name, target, options):
    if not torch.cuda.is_available(): pytest.skip('CUDA unavailable')
    dependency={'cspnet':'einops','mattergen':'torch_sparse','equiformer_v2':'e3nn'}[name]
    if importlib.util.find_spec(dependency) is None: pytest.skip('Optional core unavailable')
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        chart=compile_descriptor(Descriptor('space',136,(OrbitSpec('a',22),OrbitSpec('f',8))))
        config={'groups':[['space',136]],'properties':{},'max_atoms':16,'max_orbits':4,
            'model':{'hidden':32,'heads':4,'proposal_layers':1,'flow_layers':1,
                     'candidate_backbone':{'_target_':f'unifiedgroupgen.pl_modules.backbones.{name}.{target}',**options}}}
        _,model=build_models(config,'cuda'); model.eval()
        starts=[chart.prior('cuda',generator=torch.Generator(device='cuda').manual_seed(i))*.03 for i in (1,2)]
        with torch.inference_mode():
            results=sample_flow_batch(model,[chart,chart],steps=2,initial_states=starts)
        assert len(results)==2 and all(torch.isfinite(r['state']).all() for r in results)
    finally:
        torch.set_num_threads(previous)


@pytest.mark.parametrize('name', ['cspnet', 'mattergen', 'equiformer_v2'])
def test_reference_size_backbone_configuration_constructs(name):
    dependency={'cspnet':'einops','mattergen':'torch_sparse','equiformer_v2':'e3nn'}[name]
    if importlib.util.find_spec(dependency) is None: pytest.skip('Optional core unavailable')
    previous=torch.get_num_threads(); torch.set_num_threads(1)
    try:
        config=training_definition(compose_experiment('new_joint_u',overrides=('model='+name,)))
        proposal, flow=build_models(config,'cpu')
        assert not flow.property_encoder.configs and not proposal.condition.configs
        assert hasattr(flow,'candidate_backbone') and flow.backbone_identity['files']
        assert sum(p.numel() for p in flow.candidate_backbone.parameters()) > 100000
    finally:
        torch.set_num_threads(previous)
