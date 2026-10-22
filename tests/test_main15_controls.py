from collections import Counter
from pathlib import Path
import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure
from unifiedgroupgen.chemical_controls import ChemicalRules, feasible_formula_counts
from unifiedgroupgen.conditioning import PropertyEncoder, fit_property_scalers
from unifiedgroupgen.main_experiments import configurations
from unifiedgroupgen.proposal import DescriptorTransformer
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.structure_controls import Scaffold
from unifiedgroupgen.flow import OrbitFlow, sample_flow
from unifiedgroupgen.task_evaluation import evaluate_task


def tiny_proposal(group=1, atoms=8):
    return DescriptorTransformer([('space', group)], {}, hidden=16, layers=1, heads=4,
                                 max_atoms=atoms, max_orbits=8, allowed_elements=[8, 14, 22]).eval()


def test_all_fifteen_are_property_only_and_keep_same_source_splits():
    configs = configurations(Path(__file__).resolve().parents[1])
    assert len(configs) == 15
    for name, config in configs.items():
        assert config['composition_probability'] == config['observation_probability'] == 0
        assert not any('mpts' in path for path in config['train_files'])
        properties = [p for p, spec in config['properties'].items() if not spec.get('context')]
        assert len(properties) == (0 if name.endswith('_u') else 3 if name.endswith('_all') else 1)
        if name.startswith('joint'):
            assert config['train_files'] == (configs['mp20_u']['train_files']+configs['c2db51_u']['train_files'])


def test_chemical_system_and_required_elements_are_enforced_without_cfg():
    model = tiny_proposal()
    for rules in [ChemicalRules(allowed=['Si', 'O'], required=['Si', 'O']), ChemicalRules(required=['Ti'])]:
        for seed in range(4):
            descriptor = model.sample(kind='space', chemical=rules,
                generator=torch.Generator().manual_seed(seed), condition_composition=False)
            counts = Counter(compile_descriptor(descriptor).elements.tolist())
            assert rules.matches(counts)
    with pytest.raises(ValueError, match='outside'):
        model.sample(chemical=ChemicalRules(required=['U']))


def test_required_elements_respect_fixed_site_occupancy_and_budget():
    model = tiny_proposal(group=2, atoms=1)
    mask = ChemicalRules(required=['Si', 'O']).mask(model.legality, [model.vocab.index['BOS']])
    assert not mask.any()
    model = tiny_proposal(group=2, atoms=2)
    prefix = [model.vocab.index[t] for t in ['BOS', 'G:space:2', 'W:a', 'Z:14']]
    mask = ChemicalRules(required=['Si', 'O']).mask(model.legality, prefix)
    assert not mask[model.vocab.index['STOP']]
    assert not mask[model.vocab.index['W:a']]


def test_formula_units_are_screened_by_the_fixed_group():
    model = tiny_proposal(group=136, atoms=12)
    prefix = [model.vocab.index[t] for t in ['BOS', 'G:space:136']]
    choices = feasible_formula_counts('TiO2', [1, 2], model.legality, 'space', prefix=prefix)
    assert {22: 2, 8: 4} in choices and {22: 1, 8: 2} not in choices
    d = model.sample(composition={22: 2, 8: 4}, prefix=prefix, condition_composition=False)
    assert Counter(compile_descriptor(d).elements.tolist()) == {22: 2, 8: 4}


def test_known_orbit_prefix_is_preserved_and_conflicts_fail():
    model = tiny_proposal()
    known = Descriptor('space', 1, (OrbitSpec('a', 14),))
    prefix = model.vocab.encode(known)[:-1]
    descriptor = model.sample(composition={14: 1, 8: 2}, prefix=prefix, condition_composition=False)
    assert descriptor.orbits[0] == known.orbits[0]
    assert Counter(compile_descriptor(descriptor).elements.tolist()) == {14: 1, 8: 2}
    with pytest.raises(ValueError):
        model.sample(composition={8: 2}, prefix=prefix)


@pytest.mark.parametrize('kind,number,letter', [('space', 1, 'a'), ('layer', 1, 'a')])
def test_scaffold_preserves_known_atoms_and_leaves_unknown_motion(kind, number, letter):
    descriptor = Descriptor(kind, number, (OrbitSpec(letter, 14), OrbitSpec(letter, 8)))
    c = compile_descriptor(descriptor)
    z = torch.zeros(c.dof, dtype=torch.float64)
    z[c.lattice_dof:c.lattice_dof+3] = torch.tensor([.2, .3, .4])
    z[-3:] = torch.tensor([.6, .7, .8])
    generated = c.expand(z)
    rows = generated['lattice'].numpy().T.copy()
    coords = generated['coordinates'].numpy()[:1].copy()
    if kind == 'layer':
        rows[2, 2] = 40
        coords[:, 2] = coords[:, 2]/40+.5
    partial = Structure(Lattice(rows), [14], coords)
    scaffold = Scaffold(partial, kind, number)
    observation = scaffold.observation(c)
    model = OrbitFlow({}, hidden=16, layers=1, frequencies=2).double().eval()
    result = sample_flow(model, c, steps=2, observation=observation, condition_observation=False)
    report = scaffold.report(c, result['state'])
    assert report['scaffold_preserved']
    assert not torch.allclose(result['state'][-3:], z[-3:])


def test_mapped_incomplete_orbit_uses_periodic_lift_and_rejects_conflicts():
    descriptor = Descriptor('space', 2, (OrbitSpec('i', 14), OrbitSpec('i', 8)))
    c = compile_descriptor(descriptor)
    z = torch.zeros(c.dof, dtype=torch.float64); z[-6:] = torch.tensor([.1, .2, .3, .4, .5, .6])
    expanded = c.expand(z)
    partial = Structure(Lattice(expanded['lattice'].numpy().T), [14, 14], expanded['coordinates'].numpy()[:2])
    scaffold = Scaffold(partial, 'space', 2, {'0': 0, '1': 1})
    observation = scaffold.observation(c)
    state = observation.retract(torch.randn(c.dof, dtype=torch.float64))
    assert scaffold.report(c, state)['scaffold_preserved']
    partial.translate_sites([1], [.1, 0, 0], frac_coords=True)
    with pytest.raises(ValueError, match='conflict'):
        Scaffold(partial, 'space', 2, {'0': 0, '1': 1}).observation(c)


def test_domain_scalers_and_subset_training_keep_null_branch_and_property_identity():
    configs = {'formation_energy': {'type': 'scalar', 'per_domain': True},
               'band_gap': {'type': 'scalar', 'per_domain': True},
               'source_domain': {'type': 'categorical', 'num_classes': 2, 'context': True}}
    records = [{'properties': {'source_domain': domain, 'formation_energy': value, 'band_gap': value+1}}
               for domain, value in [(0, 0), (0, 2), (1, 100), (1, 104)]]
    fitted = fit_property_scalers(records, configs)
    encoder = PropertyEncoder(fitted, 16, dropout=1, subset_training=True).train()
    p = records[0]['properties']
    assert encoder._normalized('formation_energy', 0, p) == -1
    assert encoder._normalized('formation_energy', 100, records[2]['properties']) == -1
    like = torch.zeros(16)
    assert torch.allclose(encoder(p, like), encoder({**p, 'band_gap': 999}, like))
    encoder.dropout = 0
    seen = {tuple(sorted(encoder._subset(p, like.device))) for _ in range(80)}
    assert len(seen) == 3


def test_failed_attempts_stay_in_task_denominator():
    d = Descriptor('space', 1, (OrbitSpec('a', 14),))
    c = compile_descriptor(d)
    record = {'descriptor': d.to_dict(), 'state': [0.]*c.dof}
    samples = [{'index': 0, 'status': 'failed', 'reason': 'constraint conflict'},
               {'index': 1, 'status': 'generated', **record}]
    report = evaluate_task(samples, record, 'space', {'required': [14]})
    assert report['attempted'] == 2
    assert report['chemical_satisfaction_all_attempts'] == .5
    assert report['match_at_1'] is False and report['match_at_k'] is True


@pytest.mark.parametrize('mode', ['system', 'formula', 'required', 'scaffold', 'mapped'])
def test_cli_control_requests_use_real_sampler_and_persist_checks(tmp_path, monkeypatch, mode):
    import json
    import sys
    from unifiedgroupgen import cli
    monkeypatch.setattr(cli, 'ROOT', tmp_path)
    original_validate = cli.validate
    monkeypatch.setattr(cli, 'validate', lambda c, x, **kw: original_validate(c, x, isolate_identifier=False))
    config = {'kind': 'space', 'groups': [['space', 1]], 'properties': {},
              'max_atoms': 6, 'max_orbits': 4, 'allowed_elements': [8, 14],
              'composition_probability': 0, 'observation_probability': 0,
              'model': {'hidden': 16, 'proposal_layers': 1, 'flow_layers': 1, 'heads': 4, 'frequencies': 2}}
    proposal, flow = cli.build_models(config, 'cpu')
    checkpoint = tmp_path/'tiny.pt'
    torch.save({'config': config, 'proposal': proposal.state_dict(), 'flow': flow.state_dict()}, checkpoint)
    extra = {'system': ['--chemical-system', 'Si-O'], 'formula': ['--formula', 'SiO2', '--formula-units', '1'],
             'required': ['--required-elements', 'Si']} .get(mode, [])
    if mode in ('scaffold', 'mapped'):
        descriptor = Descriptor('space', 1, (OrbitSpec('a', 14), OrbitSpec('a', 8)))
        (tmp_path/'descriptor.json').write_text(json.dumps(descriptor.to_dict()))
        Structure(Lattice.cubic(3), [14], [[.2, .3, .4]]).to(filename=str(tmp_path/'partial.cif'))
        extra = ['--scaffold', str(tmp_path/'partial.cif')]
        if mode == 'scaffold':
            extra += ['--group', '1', '--composition', 'SiO']
        else:
            (tmp_path/'map.json').write_text('{"0": 0}')
            extra += ['--descriptor', str(tmp_path/'descriptor.json'), '--atom-map', str(tmp_path/'map.json')]
    monkeypatch.setattr(sys, 'argv', ['ugg', 'sample', '--checkpoint', str(checkpoint), '--device', 'cpu',
                        '--steps', '2', '--count', '1', '--output', str(tmp_path/'out'), *extra])
    cli.main()
    record = json.loads((tmp_path/'out/samples.jsonl').read_text())
    assert record['status'] == 'generated', record
    counts = Counter(compile_descriptor(Descriptor.from_dict(record['descriptor'])).elements.tolist())
    if mode == 'system':
        assert set(counts) == {8, 14}
    if mode == 'formula':
        assert counts == {14: 1, 8: 2}
    if mode == 'required':
        assert 14 in counts
    if mode in ('scaffold', 'mapped'):
        assert record['validation']['scaffold']['scaffold_preserved']
