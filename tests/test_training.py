import copy
import json
import random
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from torch.nn import functional as F
from unifiedgroupgen.cli import build_models
from unifiedgroupgen.data import prepare_csv
from unifiedgroupgen.flow import MessageLayer, OrbitFlow
from unifiedgroupgen.proposal import DescriptorTransformer, Legality, Vocabulary
from unifiedgroupgen.preflight import inspect_records, audit_files, check_training_gate
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.training import batch_indices, run_training


def record(identifier, descriptor=None):
    descriptor = descriptor or Descriptor('space', 2, (OrbitSpec('i', 14),))
    compiled = compile_descriptor(descriptor)
    return {'id': identifier, 'dataset': 'synthetic', 'descriptor': descriptor.to_dict(),
            'state': compiled.prior().tolist(), 'properties': {'gap': 1.0}}


def test_packed_message_layer_matches_dense_reference():
    torch.manual_seed(7)
    layer = MessageLayer(8, 5)
    h, edges = torch.randn(4, 8), torch.randn(4, 4, 5)
    messages = layer.edge(torch.cat((h[:, None].expand(4, 4, 8), h[None].expand(4, 4, 8), edges), -1))
    diagonal = ~torch.eye(4, dtype=torch.bool)
    aggregate = (messages*diagonal[..., None]).sum(1)/3
    expected = layer.norm(h+layer.node(torch.cat((h, aggregate), -1)))
    indices = diagonal.nonzero().T
    actual = layer(h, edges[diagonal], indices, torch.full((4,), 3), chunk_size=3)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_batched_flow_matches_independent_graphs_and_gradients():
    torch.manual_seed(4)
    descriptors = [Descriptor('space', 2, (OrbitSpec('i', 14),)),
                   Descriptor('layer', 37, (OrbitSpec('r', 8),)),
                   Descriptor('space', 1, (OrbitSpec('a', 14),))]
    charts = [compile_descriptor(d) for d in descriptors]
    states = [c.prior() for c in charts]
    model = OrbitFlow({}, hidden=16, layers=2, frequencies=2).eval()
    reference = copy.deepcopy(model)
    batched = model.forward_batch(charts, states, [0.3]*3)
    singles = [reference(c, z, 0.3) for c, z in zip(charts, states)]
    for actual, expected in zip(batched, singles):
        assert torch.allclose(actual, expected, atol=2e-6)
    sum(v.square().sum() for v in batched).backward()
    sum(v.square().sum() for v in singles).backward()
    for actual, expected in zip(model.parameters(), reference.parameters()):
        if actual.grad is not None:
            assert torch.allclose(actual.grad, expected.grad, atol=1e-4, rtol=1e-4)


def test_padded_descriptor_loss_matches_single_sequence_reference():
    descriptors = [Descriptor('space', 2, (OrbitSpec('i', 14),)),
                   Descriptor('space', 2, (OrbitSpec('a', 8), OrbitSpec('i', 14)))]
    model = DescriptorTransformer([('space', 2)], {}, hidden=16, layers=1, max_orbits=4).eval()
    batch = model.loss_batch(descriptors, [{}, {}], [None, {8: 1, 14: 2}])
    for i, (descriptor, composition) in enumerate(zip(descriptors, [None, {8: 1, 14: 2}])):
        tokens = model.vocab.encode(descriptor)
        logits = model(tokens[:-1], {}, composition=composition)
        legal = torch.stack([model.legality.allowed(tokens[:j], composition, 'space') for j in range(1, len(tokens))])
        expected = F.cross_entropy(logits.masked_fill(~legal, -torch.inf), torch.tensor(tokens[1:]))
        assert torch.allclose(batch[i], expected, atol=1e-6)


def test_element_domain_masks_sampling_and_rejects_outside_composition():
    model = DescriptorTransformer([('space', 2)], {}, hidden=16, layers=1, allowed_elements=[8, 14]).eval()
    prefix = [model.vocab.index['BOS'], model.vocab.index['G:space:2'], model.vocab.index['W:i']]
    legal = model.legality.allowed(prefix)
    assert legal[model.vocab.index['Z:8']] and not legal[model.vocab.index['Z:104']]
    with pytest.raises(ValueError, match='outside'):
        model.legality.allowed(prefix[:1], {104: 2})


def test_edge_budget_never_splits_an_orbit_and_rejects_oversize_graph():
    records = [record(str(i)) for i in range(5)]
    batches = batch_indices(records, 4, 4, shuffle=False)
    assert [len(b) for b in batches] == [2, 2, 1]
    assert sorted(i for batch in batches for i in batch) == list(range(5))
    with pytest.raises(ValueError, match='Single structure'):
        batch_indices(records, 4, 1)


def test_data_audit_blocks_overlap_nonfinite_and_unknown_elements():
    train, val = record('train'), record('val')
    val['state'] = list(train['state'])
    report = inspect_records({'train': [train], 'val': [val]}, {'properties': {'gap': {}}})
    assert not report['passed']
    assert any(e['type'] == 'verified_duplicate_structure' for e in report['errors'])
    val['state'][0] = float('nan')
    report = inspect_records({'train': [train], 'val': [val]}, {'properties': {'gap': {}}})
    assert any(e['type'] == 'invalid_record' for e in report['errors'])


def test_audit_gate_detects_data_changes(tmp_path):
    train, val = tmp_path/'train.jsonl', tmp_path/'val.jsonl'
    train.write_text(json.dumps(record('train'))+'\n')
    val.write_text(json.dumps(record('val'))+'\n')
    config = {'properties': {'gap': {}}, 'require_data_audit': True, 'data_audit': 'audit.json'}
    report = audit_files({'train': [train], 'val': [val]}, config, tmp_path/'audit.json')
    assert report['passed']
    check_training_gate(config, [train], [val], tmp_path)
    train.write_text(train.read_text()+'\n')
    with pytest.raises(ValueError, match='differ'):
        check_training_gate(config, [train], [val], tmp_path)


def test_preparation_resume_and_joint_orbit_reordering(tmp_path):
    from pathlib import Path
    source = Path(__file__).resolve().parents[2]/'data/mp_20/train.csv'
    if not source.exists():
        source = Path(__file__).resolve().parents[1]/'raw_data/mp_20/train.csv'
    config = {'dataset': 'mp_20', 'kind': 'space', 'properties': {}, 'max_atoms': 192, 'max_orbits': 32}
    output = tmp_path/'prepared.jsonl'
    first = prepare_csv(source, output, config, limit=2)
    before = output.read_bytes()
    second = prepare_csv(source, output, config, limit=2, resume=True)
    assert first == second and output.read_bytes() == before
    assert len((tmp_path/'prepared.jsonl.attempts.jsonl').read_text().splitlines()) == 2
    journal = tmp_path/'prepared.jsonl.attempts.jsonl'
    with journal.open('ab') as stream:
        stream.write(b'{"row":')
    prepare_csv(source, output, config, limit=2, resume=True)
    assert output.read_bytes() == before
    assert (tmp_path/'prepared.jsonl.attempts.jsonl.incomplete_tail').read_bytes() == b'{"row":'
    with pytest.raises(ValueError, match='differs'):
        prepare_csv(source, output, {**config, 'max_orbits': 16}, limit=2, resume=True)


@pytest.mark.parametrize('device,precision', [('cpu', 'fp32'), ('cuda', 'fp32'), ('cuda', 'bf16')])
@pytest.mark.parametrize('legacy_amp', [False, True])
def test_uninterrupted_and_resumed_training_are_identical(tmp_path, device, precision, legacy_amp, monkeypatch):
    if device == 'cuda' and not torch.cuda.is_available():
        pytest.skip('CUDA unavailable')
    if precision == 'bf16' and not torch.cuda.is_bf16_supported():
        pytest.skip('CUDA BF16 unavailable')
    if legacy_amp:
        monkeypatch.delattr(torch.amp, 'GradScaler', raising=False)
    torch.set_num_threads(1)
    config = {'groups': [('space', 2)], 'properties': {'gap': {}}, 'model': {'hidden': 16, 'proposal_layers': 1,
              'flow_layers': 1, 'heads': 4, 'frequencies': 2}, 'max_orbits': 4, 'batch_size': 2,
              'epochs': 2, 'early_stopping_patience': 0, 'precision': precision}
    records = [record('a'), record('b')]; validation = [record('v')]
    train_file, val_file = tmp_path/'train.jsonl', tmp_path/'val.jsonl'
    train_file.write_text('train'); val_file.write_text('val')
    def run(path, epochs, resume=None):
        random.seed(42); np.random.seed(42); torch.manual_seed(42)
        proposal, flow = build_models(config, device)
        args = SimpleNamespace(device=device, resume=resume, batch_size=None, epochs=epochs)
        run_training(args, config, records, validation, proposal, flow, [train_file], [val_file], path)
        return torch.load(path, weights_only=False)
    full = run(tmp_path/'full.pt', 2)
    run(tmp_path/'resumed.pt', 1)
    resumed = run(tmp_path/'resumed.pt', 2, tmp_path/'resumed.pt')
    status = json.loads((tmp_path/'resumed.status.json').read_text())
    assert status['state'] == 'complete' and status['completed_epochs'] == 2
    assert status['reason'] == 'maximum_epochs'
    assert status['train_records'] == 2 and status['val_records'] == 1
    for model in ('proposal', 'flow'):
        for name in full[model]:
            assert torch.equal(full[model][name], resumed[model][name])
    assert full['best_val_loss'] == resumed['best_val_loss']


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_amp_and_checkpointed_flow_have_finite_backward():
    c = compile_descriptor(Descriptor('space', 2, (OrbitSpec('i', 14),)))
    flow = OrbitFlow({}, hidden=32, layers=2, frequencies=2, checkpoint_layers=True).cuda().train()
    z = c.prior('cuda')
    with torch.autocast('cuda', dtype=torch.float16):
        loss = flow(c, z, 0.2).square().sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in flow.parameters() if p.grad is not None)


def test_formal_gate_rejects_partial_preparation_and_unverified_sources(tmp_path):
    path = tmp_path/'train.jsonl'
    path.write_text(json.dumps(record('a'))+'\n')
    path.with_suffix(path.suffix+'.report.json').write_text(json.dumps({'input': 1, 'accepted': 1, 'full_source': False}))
    config = {'properties': {'gap': {}}, 'require_full_preparation': True, 'source_verified': False}
    report = audit_files({'train': [path], 'val': [path]}, config, tmp_path/'audit.json')
    types = {e['type'] for e in report['errors']}
    assert {'partial_preparation', 'unverified_coordinate_source'} <= types
    with pytest.raises(ValueError, match='unverified'):
        check_training_gate({**config, 'require_data_audit': True, 'data_audit': 'missing.json'}, [path], [path], tmp_path)


def test_empty_element_domain_is_rejected():
    with pytest.raises(ValueError, match='Allowed elements'):
        DescriptorTransformer([('space', 2)], {}, hidden=16, layers=1, allowed_elements=[])


def test_completion_pruning_matches_exhaustive_small_compositions():
    legality = Legality(Vocabulary([('space', 2), ('space', 225)]), max_atoms=16, max_orbits=3)
    def brute(group, counts, occupied, slots):
        if not any(counts):
            return True
        if not slots:
            return False
        element = next(i for i, n in enumerate(counts) if n)
        for letter, (multiplicity, fixed) in legality.rows[group].items():
            if counts[element] >= multiplicity and not (fixed and letter in occupied):
                updated = list(counts); updated[element] -= multiplicity
                if brute(group, updated, occupied | ({letter} if fixed else set()), slots-1):
                    return True
        return False
    for group in legality.rows:
        for first in range(5):
            for second in range(5):
                for slots in range(4):
                    for occupied in (set(), {'a'}, {'a', 'b'}):
                        assert legality._completion(group, [first, second], occupied, slots) == brute(group, [first, second], occupied, slots)


def test_unseen_test_elements_are_reported_and_retained():
    train, val = record('train'), record('val')
    test = record('test', Descriptor('space', 2, (OrbitSpec('i', 18),)))
    report = inspect_records({'train': [train], 'val': [val], 'test': [test]}, {'properties': {'gap': {}}})
    assert report['passed']
    assert report['splits']['test']['records'] == 1
    assert any(w['type'] == 'test_out_of_domain_elements' for w in report['warnings'])


@pytest.mark.parametrize('guidance,expected_null', [(0, [True]*6), (1, [False]*6), (2, [False, True]*6)])
def test_cfg_skips_unused_branch_without_changing_velocity(guidance, expected_null):
    from unifiedgroupgen.flow import sample_flow
    class ConstantFlow(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor(0.0))
            self.calls = []
        def forward(self, c, z, t, properties, force_null=False, **kwargs):
            self.calls.append(force_null)
            return torch.ones_like(z)*(0.01 if force_null else 0.02)
    c = compile_descriptor(Descriptor('space', 2, (OrbitSpec('i', 14),)))
    generator = torch.Generator().manual_seed(9)
    start = c.prior(generator=generator)
    model = ConstantFlow().eval()
    output = sample_flow(model, c, steps=3, guidance=guidance, generator=torch.Generator().manual_seed(9))
    assert model.calls == expected_null
    assert torch.allclose(output['state'], start+0.01+guidance*0.01, atol=1e-6)
