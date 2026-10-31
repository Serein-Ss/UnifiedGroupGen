"""Scientific contracts for source-free models, priors, shared CFG, and packed flows."""
import copy
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from unifiedgroupgen.common.configuration import compose_experiment, training_definition, load_experiments
from unifiedgroupgen.conditional.structure import training_observation
from unifiedgroupgen.generation.flow import sample_flow, sample_flow_batch
from unifiedgroupgen.generation.prior import sample_prior, fit_prior
from unifiedgroupgen.group_constraints.compiler import Descriptor, OrbitSpec, compile_descriptor, group_table
from unifiedgroupgen.pl_modules.factory import build_models
from unifiedgroupgen.pl_modules.objectives.joint import joint_loss
from unifiedgroupgen.pl_modules.objectives.flow_matching import conditional_path
from unifiedgroupgen.tests.legacy.generation_before_entry.flow import sample_flow as old_sample_flow

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture(autouse=True)
def bounded_threads():
    previous = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def small(properties=None):
    return {'groups': [['space', 1], ['space', 2], ['layer', 1]], 'properties': properties or {},
            'max_atoms': 8, 'max_orbits': 4, 'cond_dropout': .1,
            'model': {'hidden': 16, 'heads': 4, 'proposal_layers': 1, 'flow_layers': 1, 'frequencies': 2}}


def charts():
    return [compile_descriptor(Descriptor('space', 2, (OrbitSpec('i', 14),))),
            compile_descriptor(Descriptor('layer', 1, (OrbitSpec('a', 16), OrbitSpec('a', 16))))]


def test_new_default_and_all_fifteen_new_recipes_are_source_free():
    cfg = training_definition(compose_experiment('new_joint_all'))
    assert 'source_domain' not in cfg['properties'] and not cfg.get('dataset_domains')
    assert cfg['condition_mask_policy'] == 'shared'
    assert set(cfg['evaluation_domains']) == {'mp_20', 'c2db_51'}
    entries = load_experiments('configs/experiments_v2.yaml')
    assert len(entries) == 15
    for entry in entries.values():
        config = entry['config']
        assert config['input_contract'] == 'source_free_v2'
        assert 'source_domain' not in config['properties'] and not config.get('dataset_domains')
        assert not any(p.get('per_domain') for p in config['properties'].values())
    old = training_definition(compose_experiment('joint_all'))
    assert old['properties']['source_domain']['context'] and old['dataset_domains']


def test_new_contract_rejects_accidental_source_injection():
    cfg = small({'source_domain': {'type': 'categorical', 'num_classes': 2, 'context': True}})
    cfg['input_contract'] = 'source_free_v2'
    with pytest.raises(ValueError, match='source labels'): build_models(cfg, 'cpu')


def test_legacy_fifteen_definitions_and_audit_hashes_still_match():
    import hashlib
    for name, entry in load_experiments('configs/experiments.yaml').items():
        audit = json.loads((ROOT/'data/dataset/audits/main15'/f'{name}.json').read_text())
        digest = hashlib.sha256(json.dumps(entry['config'], sort_keys=True).encode()).hexdigest()
        assert digest == audit['config_sha256']


@pytest.mark.parametrize('guidance', [0., 1., 2.])
def test_legacy_sampler_is_unchanged_and_packed_flow_matches_singles(guidance):
    _, model = build_models(small({'gap': {}}), 'cpu'); model.eval()
    cc = charts(); props = [{'gap': 1.}, {'gap': 2.}]
    expected = []
    for i, (c, p) in enumerate(zip(cc, props)):
        old = old_sample_flow(model, c, p, 3, guidance, generator=torch.Generator().manual_seed(142+i))
        new = sample_flow(model, c, p, 3, guidance, generator=torch.Generator().manual_seed(142+i))
        for key in ('state', 'coordinates', 'lattice'): torch.testing.assert_close(new[key], old[key], rtol=0, atol=0)
        expected.append(new)
    actual = sample_flow_batch(model, cc, props, steps=3, guidance=guidance,
                                generators=[torch.Generator().manual_seed(142+i) for i in range(2)])
    for new, old in zip(actual, expected):
        for key in ('state', 'coordinates', 'lattice'):
            torch.testing.assert_close(new[key], old[key], atol=2e-5, rtol=2e-5)


@pytest.mark.parametrize('name', ['scaled_normal', 'wrapped_normal', 'physical_metric', 'train_gaussian'])
@pytest.mark.parametrize('chart', [0, 1])
def test_each_prior_expands_to_a_compatible_positive_definite_metric(name, chart):
    c = charts()[chart]; spec = {'name': name}
    z = sample_prior(c, dtype=torch.float64, generator=torch.Generator().manual_seed(3), specification=spec)
    full = c.expand(z); h = full['lattice'].T @ full['lattice']
    assert torch.linalg.eigvalsh(h).min() > 0
    periodic = c.tensor(c.state_periodic, z, torch.bool)
    assert ((z[periodic] >= 0) & (z[periodic] < 1)).all()
    for operation in group_table(c.descriptor.kind, c.descriptor.number)[0].ops:
        r = torch.tensor(operation.rotation_matrix, dtype=z.dtype)
        torch.testing.assert_close(r.T @ h @ r, h, atol=1e-7, rtol=1e-7)


def test_empirical_prior_statistics_only_use_the_supplied_train_records():
    c = charts()[0]
    records = [{'descriptor': c.descriptor.to_dict(), 'state': [value]*c.dof} for value in (1., 3.)]
    fitted = fit_prior(records, {'name': 'train_gaussian'})
    stats = fitted['lattice_statistics']['space:2']
    assert stats['mean'] == [2.]*c.lattice_dof and stats['std'] == [1.]*c.lattice_dof
    assert fitted['fitted_split'] == 'train' and stats['records'] == 2


def test_nonorthogonal_hexagonal_metric_needs_a_compatible_reference():
    c = compile_descriptor(Descriptor('space',194,(OrbitSpec('f',22),)))
    r = torch.tensor([[0.,-1.,0.],[1.,-1.,0.],[0.,0.,1.]],dtype=torch.float64)
    assert not torch.allclose(r.T@r,torch.eye(3,dtype=r.dtype))
    for name in ('scaled_normal','wrapped_normal','physical_metric'):
        z=sample_prior(c,dtype=torch.float64,specification={'name':name},generator=torch.Generator().manual_seed(3))
        h=c.metric(z[:c.lattice_dof])
        torch.testing.assert_close(r.T@h@r,h,rtol=1e-9,atol=1e-9)


@pytest.mark.parametrize('kind',['space','layer'])
def test_packed_entry_preserves_cases_cif_ledger_and_completed_resume(tmp_path,kind):
    from unifiedgroupgen.generate import generate_records
    from unifiedgroupgen.common.structures import read_records
    c=charts()[0 if kind=='space' else 1]
    config=small();config.update(composition_probability=0.,observation_probability=0.)
    proposal,flow=build_models(config,'cpu');proposal.eval();flow.eval()
    reference=tmp_path/'reference.jsonl'
    reference.write_text(json.dumps({'id':'r','dataset':'fixture','descriptor':c.descriptor.to_dict()})+'\n')
    config['test_files']=[str(reference)]
    job={'task':'csp','kind':kind,'dataset':'fixture','case_ids':['r'],'k':3,'properties':{},'fixed_descriptor':True}
    serial,packed=tmp_path/'serial',tmp_path/'packed';serial.mkdir();packed.mkdir()
    generate_records(proposal,flow,config,job,serial,'cpu',142,2,0.,batch_size=1)
    generate_records(proposal,flow,config,job,packed,'cpu',142,2,0.,batch_size=3)
    a,b=read_records(serial/'samples.jsonl'),read_records(packed/'samples.jsonl')
    assert len(a)==len(b)==3 and [r['index'] for r in b]==[0,1,2]
    assert all(r['status']=='generated' and 'cif' in r for r in b)
    for x,y in zip(a,b):
        assert x['descriptor']==y['descriptor'] and x['case_id']==y['case_id']=='r'
        np.testing.assert_allclose(x['state'],y['state'],atol=2e-5,rtol=2e-5)
    assert not list(packed.glob('*.cif'))
    before=(packed/'samples.jsonl').read_bytes()
    generate_records(proposal,flow,config,job,packed,'cpu',142,2,0.,batch_size=3)
    assert (packed/'samples.jsonl').read_bytes()==before


def test_source_free_joint_evaluation_stratifies_by_kind_without_model_source_input(tmp_path):
    from scripts.evaluate_models import build_jobs
    records=[]
    for dataset,c in zip(('mp_20','c2db_51'),charts()):
        records.append({'id':dataset,'dataset':dataset,'descriptor':c.descriptor.to_dict(),'properties':{}})
    data=tmp_path/'data.jsonl';data.write_text(''.join(json.dumps(r)+'\n' for r in records))
    config={'dataset':'joint','kinds':['space','layer'],'properties':{},'evaluation_domains':['mp_20','c2db_51'],
            'test_files':[str(data)],'train_files':[str(data)]}
    jobs=build_jobs(config,2,2,1,1,142,tasks=['dng'],root=tmp_path,split_domain_budget=True)
    assert {j['kind'] for j in jobs}=={'space','layer'}
    assert all(j['properties']=={} and j['count']==1 for j in jobs)


def test_untrained_checkpoint_cannot_claim_learned_completion():
    from types import SimpleNamespace
    from unifiedgroupgen.generate import prepare_request
    with pytest.raises(ValueError,match='trained on completion masks'):
        prepare_request(SimpleNamespace(completion_method='masked_flow'),{},None,None)


def test_gaussian_bridge_target_is_the_time_derivative_and_preserves_observations():
    c = charts()[0]; endpoint = c.prior()*.2
    observation = training_observation(c, endpoint, {'mode': 'parameters'})
    def path(t):
        return conditional_path(c, endpoint, endpoint.new_tensor(t), observation,
                                torch.Generator().manual_seed(18), path={'sigma': .1})
    state, velocity = path(.4); epsilon = 1e-3
    finite_difference = (path(.4+epsilon)[0]-path(.4-epsilon)[0])/(2*epsilon)
    torch.testing.assert_close(velocity, finite_difference, atol=1e-4, rtol=1e-4)
    a, b = observation._tensors(state)
    torch.testing.assert_close(a@state, b, atol=1e-5, rtol=0)
    torch.testing.assert_close(a@velocity, torch.zeros_like(b), atol=1e-5, rtol=0)


def test_two_stages_receive_the_same_visibility_decision(monkeypatch):
    cfg = small({'fe': {}, 'gap': {}, 'eh': {}})
    cfg.update(condition_mask_policy='shared', property_subset_training=True)
    p, f = build_models(cfg, 'cpu'); observed = []
    for encoder in (p.condition, f.property_encoder):
        original = encoder.forward_batch
        def spy(*args, _call=original, **kwargs):
            observed.append(kwargs['visibility'])
            return _call(*args, **kwargs)
        monkeypatch.setattr(encoder, 'forward_batch', spy)
    c = charts()[0]
    records = [{'id': 'a', 'descriptor': c.descriptor.to_dict(), 'state': c.prior().tolist(),
                'properties': {'fe': -1., 'gap': 1., 'eh': .1}}]
    losses = joint_loss(p, f, records, 0., 0., 3.)
    (losses['descriptor'].mean()+losses['flow'].mean()).backward()
    assert observed[0] is observed[1] and observed[0] is not None


@pytest.mark.parametrize('mode', ['parameters', 'orbit_block', 'spatial_patch', 'mixed'])
def test_learned_completion_masks_keep_a_free_subspace_and_hard_observed_values(mode):
    c = charts()[1]; endpoint = c.prior()*.2
    o = training_observation(c, endpoint, {'mode': mode, 'minimum_fraction': .4, 'maximum_fraction': .4})
    a, b = o._tensors(endpoint)
    assert 0 < torch.linalg.matrix_rank(a) < c.dof
    torch.testing.assert_close(a@endpoint, b, atol=1e-5, rtol=0)
    _, flow = build_models(small(), 'cpu'); flow.eval()
    generated = sample_flow_batch(flow, [c], [{}], observations=[o], steps=2, solver='euler')
    torch.testing.assert_close(a@generated[0]['state'], b, atol=1e-5, rtol=0)


def test_patch_mask_handles_fixed_wyckoff_coordinates():
    c = compile_descriptor(Descriptor('space', 2, (OrbitSpec('a', 14),)))
    assert c.coordinate_dof == 0 and c.dof > 1
    endpoint = c.prior()*.1
    observation = training_observation(c, endpoint, {'mode':'spatial_patch'})
    a, b = observation._tensors(endpoint)
    assert 0 < torch.linalg.matrix_rank(a) < c.dof
    torch.testing.assert_close(a@endpoint, b, atol=1e-5, rtol=0)


@pytest.mark.parametrize('kind', ['space', 'layer'])
def test_public_packed_generation_saves_inline_cifs_and_hard_observations(tmp_path, kind):
    from types import SimpleNamespace
    from unifiedgroupgen.generate import sample_batched
    from unifiedgroupgen.common.structures import read_records
    config=small(); config.update(composition_probability=0.,observation_probability=0.)
    proposal,flow=build_models(config,'cpu')
    checkpoint=tmp_path/'model.pt'
    torch.save({'config':config,'proposal':proposal.state_dict(),'flow':flow.state_dict()},checkpoint)
    c=charts()[0 if kind=='space' else 1]
    descriptor=tmp_path/'descriptor.json'; descriptor.write_text(json.dumps(c.descriptor.to_dict()))
    observations=tmp_path/'observation.json'; observations.write_text(json.dumps({'indices':[0],'values':[0.]}))
    args=SimpleNamespace(checkpoint=str(checkpoint),device='cpu',kind=kind,properties='{}',
        composition=None,formula=None,formula_units=[1],chemical_system=None,allowed_elements=None,
        required_elements=None,group=None,scaffold=None,atom_map=None,scaffold_tolerance=1e-4,
        descriptor=str(descriptor),observations=str(observations),prior=False,guidance=0.,
        temperature=1.,steps=2,count=3,seed=142,output=str(tmp_path/'generated'),batch_size=2,
        solver='heun',completion_method='auto')
    sample_batched(args)
    rows=read_records(tmp_path/'generated/samples.jsonl')
    assert len(rows)==3 and all(r['status']=='generated' and r['cif'] and r['cif_sha256'] for r in rows)
    assert all(r['validation']['observation_max_residual'] < 1e-5 for r in rows)
    assert not list((tmp_path/'generated').glob('*.cif'))
    assert json.loads((tmp_path/'generated/sampling.status.json').read_text())['state']=='complete'
