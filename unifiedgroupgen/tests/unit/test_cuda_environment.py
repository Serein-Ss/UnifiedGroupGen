"""The maintained environment and precision boundaries, including bundled core provenance."""
import hashlib
import json
from pathlib import Path
import pytest
import torch
import yaml

from unifiedgroupgen.common.precision import resolve_precision
from unifiedgroupgen.group_constraints.compiler import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.pl_modules.factory import build_models
from unifiedgroupgen.generation.flow import sample_flow_batch

ROOT=Path(__file__).resolve().parents[3]


def test_environment_excludes_conflicting_upstream_training_distributions():
    config=yaml.safe_load((ROOT/'unifiedgroupgen/environment.yml').read_text(encoding='utf-8'))
    requirements=next(c['pip'] for c in config['dependencies'] if isinstance(c,dict))
    assert 'numpy==1.26.4' in requirements and 'torch==2.7.1+cu118' in requirements
    assert all(not r.startswith(('mattergen','fairchem-core')) for r in requirements)
    assert all(r in requirements for r in ('e3nn==0.5.1','matgl==0.9.2','dgl==1.1.2'))


def test_bundled_core_and_resource_bytes_match_the_extraction_manifest():
    directory=ROOT/'unifiedgroupgen/pl_modules/backbones/official'
    manifest=json.loads((directory/'vendor_manifest.json').read_text())
    assert len(manifest)>30
    for name,record in manifest.items():
        data=(directory/name).read_bytes()
        assert hashlib.sha256(data).hexdigest()==record['local_sha256'],name
        if name.endswith('.py') and not name.endswith('globals.py'):
            restored=data.decode('utf-8').replace('unifiedgroupgen.pl_modules.backbones.official.mattergen.', 'mattergen.').replace(
                'unifiedgroupgen.pl_modules.backbones.official.fairchem.', 'fairchem.')
            assert hashlib.sha256(restored.encode()).hexdigest()==record['normalized_upstream_sha256'],name
            assert record['changes']=='absolute import namespace only'
    assert (directory/'MatterGen.LICENSE').is_file() and (directory/'FAIRChem.LICENSE').is_file()


def test_precision_selection_is_explicit_on_cpu():
    assert resolve_precision('cpu','auto')=='fp32'
    with pytest.raises(ValueError):resolve_precision('cpu','bf16')


def test_sample_reuse_rejects_mixed_precisions():
    from types import SimpleNamespace
    from unifiedgroupgen.generate import check_record_precision
    check_record_precision([{}],SimpleNamespace(inference_precision='fp32'))
    with pytest.raises(ValueError,match='different inference precision'):
        check_record_precision([{'precision':'fp32'}],SimpleNamespace(inference_precision='bf16'))


@pytest.mark.parametrize('name,target,options',[
    ('mattergen','MatterGenBackbone',{'blocks':1,'cutoff':4.,'max_neighbors':16}),
    ('equiformer_v2','EquiformerV2Backbone',{'layers':1,'channels':16,'lmax':2,'mmax':1,'cutoff':4.,'max_neighbors':16})])
def test_amp_generation_keeps_state_float32_and_uses_bf16_linear_layers(name,target,options):
    if not torch.cuda.is_available():pytest.skip('CUDA unavailable')
    previous=torch.get_num_threads();torch.set_num_threads(1)
    try:
        config={'groups':[['space',136]],'properties':{},'max_atoms':16,'max_orbits':4,
            'model':{'hidden':32,'heads':4,'proposal_layers':1,'flow_layers':1,
                'candidate_backbone':{'_target_':f'unifiedgroupgen.pl_modules.backbones.{name}.{target}',**options}}}
        _,model=build_models(config,'cuda');model.eval();model.inference_precision='bf16'
        dtypes=[]
        hooks=[m.register_forward_hook(lambda m,i,o:dtypes.append(o.dtype))
               for m in model.candidate_backbone.core.modules() if isinstance(m,torch.nn.Linear)]
        chart=compile_descriptor(Descriptor('space',136,(OrbitSpec('a',22),OrbitSpec('f',8))))
        result=sample_flow_batch(model,[chart],steps=2,initial_states=[chart.prior('cuda')*.03])[0]
        for hook in hooks:hook.remove()
        assert torch.bfloat16 in dtypes
        assert result['state'].dtype==torch.float32 and torch.isfinite(result['state']).all()
        h=result['lattice'].T@result['lattice']
        torch.testing.assert_close(h[0,0],h[1,1],rtol=1e-5,atol=1e-5)
    finally:torch.set_num_threads(previous)
