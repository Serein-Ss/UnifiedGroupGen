import torch
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.proposal import Vocabulary, Legality, DescriptorTransformer
from unifiedgroupgen.flow import OrbitFlow, flow_loss, sample_flow
from unifiedgroupgen.constraints import AffineObservation
from unifiedgroupgen.conditioning import PropertyEncoder, fit_property_scalers


def test_prefix_completion_masks_fixed_sites_and_exact_composition():
    vocabulary = Vocabulary([('space', 2), ('space', 225)])
    legality = Legality(vocabulary, max_atoms=20, max_orbits=8)
    descriptor = Descriptor('space', 2, (OrbitSpec('a', 8), OrbitSpec('i', 14)))
    tokens = vocabulary.encode(descriptor)
    for i in range(1, len(tokens)):
        allowed = legality.allowed(tokens[:i], {8: 1, 14: 2}, kind='space')
        assert allowed[tokens[i]]
    first = legality.allowed(tokens[:1], {8: 1}, kind='space')
    assert first[vocabulary.index['G:space:2']]
    assert not first[vocabulary.index['G:space:225']]
    prefix = tokens[:4]
    assert not legality.allowed(prefix)[vocabulary.index['W:a']]


def test_train_and_sample_both_models_with_property_cfg_and_observations():
    descriptor = Descriptor('space', 2, (OrbitSpec('i', 14), OrbitSpec('a', 8)))
    compiled = compile_descriptor(descriptor)
    properties = {'formation_energy': {'type': 'scalar', 'shift': -1, 'scale': 1}}
    proposal = DescriptorTransformer([('space', 2)], properties, hidden=32, layers=1, max_atoms=12, max_orbits=4)
    flow = OrbitFlow(properties, hidden=32, layers=1, frequencies=2)
    endpoint = compiled.prior()
    discrete = proposal.loss(descriptor, {'formation_energy': -1.2}, composition={14: 2, 8: 1})
    continuous = flow_loss(flow, compiled, endpoint, {'formation_energy': -1.2})
    (discrete+continuous['loss']).backward()
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in proposal.parameters())
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in flow.parameters())
    proposal.eval()
    flow.eval()
    generated = proposal.sample({'formation_energy': -1.2}, composition={14: 2, 8: 1}, guidance=1.5)
    c = compile_descriptor(generated)
    assert c.num_atoms == 3
    observation = AffineObservation.fixed(compiled.dof, list(range(compiled.lattice_dof)), [0]*compiled.lattice_dof)
    output = sample_flow(flow, compiled, {'formation_energy': -1.2}, steps=3, guidance=2.0,
                         observation=observation, return_trajectory=True)
    for state in output['trajectory']:
        assert torch.allclose(state[:compiled.lattice_dof], torch.zeros(compiled.lattice_dof), atol=1e-7)
        compiled.permutations(compiled.expand(state)['coordinates'].numpy(), tolerance=1e-5)


def test_missing_labels_are_not_zero_and_scalers_are_training_only():
    configs = {'gap': {'type': 'scalar'}}
    records = [{'properties': {'gap': 1}}, {'properties': {'gap': 3}}, {'properties': {'gap': None}}]
    fitted = fit_property_scalers(records, configs)
    assert fitted['gap']['shift'] == 2 and fitted['gap']['scale'] == 1
    encoder = PropertyEncoder(fitted, 16).eval()
    like = torch.zeros(1)
    assert torch.allclose(encoder({}, like), encoder({'gap': float('nan')}, like))
    assert torch.allclose(encoder({'gap': 0}, like, force_null=True), encoder({}, like))


def test_sampling_summary_counts_failed_attempts_and_unknown_layer_groups():
    from unifiedgroupgen.evaluation import summarize_attempts
    reports = [{'status': 'generated', 'validation': {'group_operation_closure': True,
                'distance_valid': True, 'exact_layer_group': None}}, {'status': 'failed'}]
    summary = summarize_attempts(reports)
    assert summary['attempted'] == 2 and summary['failed'] == 1
    assert summary['rates_over_all_attempts']['distance_valid'] == 0.5
    assert summary['exact_group_identified'] == 0 and summary['exact_group_unidentified_generated'] == 1
    assert summary['counts']['chemistry_validated'] == 0
