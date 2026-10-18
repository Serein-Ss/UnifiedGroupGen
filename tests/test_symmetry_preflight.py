import copy
import json
import random

import numpy as np
import pytest
import torch

from unifiedgroupgen import symmetry_preflight as gate
from unifiedgroupgen.cli import build_models
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor


def test_complete_layer_catalog_preserves_rng_and_has_fixed_lg52_orbit():
    gate.check_layer_catalog.cache_clear()
    python_rng, numpy_rng, torch_rng = random.getstate(), np.random.get_state(), torch.get_rng_state()
    report = gate.check_layer_catalog()
    assert report['passed'] and report['layer_groups'] == 80 and report['single_orbit_types'] == 477
    assert report['pyxtal'] == '1.1.3'
    assert random.getstate() == python_rng
    assert np.array_equal(np.random.get_state()[1], numpy_rng[1])
    assert torch.equal(torch.get_rng_state(), torch_rng)
    c = compile_descriptor(Descriptor('layer', 52, (OrbitSpec('c', 14),)))
    assert set(map(tuple, c.b)) == {(0., 0., 0.), (.5, .5, 0.), (.5, 0., 0.), (0., .5, 0.)}


def test_unverified_version_blocks_even_without_formal_data_audit(monkeypatch, tmp_path):
    from unifiedgroupgen.preflight import check_training_gate
    gate.check_layer_catalog.cache_clear()
    monkeypatch.setattr(gate, 'version', lambda package: '1.1.1')
    try:
        with pytest.raises(ValueError, match='PyXtal 1.1.1.*verified'):
            check_training_gate({}, [], [], tmp_path)
        with pytest.raises(ValueError, match='PyXtal 1.1.1.*verified'):
            build_models({'kind': 'layer'}, 'cpu')
    finally:
        gate.check_layer_catalog.cache_clear()


def test_version_string_alone_cannot_bypass_corrupt_catalog(monkeypatch):
    c = copy.copy(compile_descriptor(Descriptor('layer', 52, (OrbitSpec('c', 14),))))
    c.b = c.b.copy(); c.b[-1] = c.b[0]
    original = gate.compiled_chart
    def corrupted(descriptor):
        return c if descriptor.kind == 'layer' and descriptor.number == 52 and descriptor.orbits[0].letter == 'c' else original(descriptor)
    gate.check_layer_catalog.cache_clear()
    monkeypatch.setattr(gate, 'compiled_chart', corrupted)
    try:
        with pytest.raises(ValueError, match='LG52 4c'):
            gate.check_layer_catalog()
    finally:
        gate.check_layer_catalog.cache_clear()


def test_fixed_orbit_numpy_expansion_does_not_contract_an_empty_axis(monkeypatch):
    from unifiedgroupgen.representation_audit import expand_numpy
    c = compile_descriptor(Descriptor('layer', 52, (OrbitSpec('c', 14),)))
    original = np.einsum
    def no_empty_coordinates(expression, *args, **kwargs):
        if expression == 'ijq,q->ij':
            raise AssertionError('A fixed orbit must expand directly from its offset')
        return original(expression, *args, **kwargs)
    monkeypatch.setattr(np, 'einsum', no_empty_coordinates)
    _, coordinates, _ = expand_numpy(c, np.zeros(c.dof))
    assert np.array_equal(coordinates, c.b)


def test_runtime_verifier_strict_loads_existing_schema_and_keeps_data(tmp_path, monkeypatch):
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1]/'scripts/verify_symmetry_runtime.py'
    spec = importlib.util.spec_from_file_location('verify_symmetry_runtime', path)
    script = importlib.util.module_from_spec(spec); spec.loader.exec_module(script)
    descriptors = [Descriptor('space', 2, (OrbitSpec('i', 14),)),
                   Descriptor('layer', 78, (OrbitSpec('a', 42), OrbitSpec('f', 16)))]
    records = [{'id': str(i), 'dataset': d.kind, 'descriptor': d.to_dict(),
                'state': compile_descriptor(d).prior().tolist(), 'properties': {}} for i, d in enumerate(descriptors)]
    data = tmp_path/'records.jsonl'
    data.write_text(''.join(json.dumps(r)+'\n' for r in records))
    before = data.read_bytes()
    config = {'groups': [('space', 2), ('layer', 78)], 'model': {'hidden': 16, 'proposal_layers': 1, 'flow_layers': 1, 'frequencies': 2}}
    proposal, flow = build_models(config, 'cpu')
    checkpoint, output = tmp_path/'model.pt', tmp_path/'report.json'
    torch.save({'config': config, 'epoch': 3, 'proposal': proposal.state_dict(), 'flow': flow.state_dict()}, checkpoint)
    monkeypatch.setattr('sys.argv', ['verify_symmetry_runtime', '--checkpoint', str(checkpoint), '--records', str(data), '--output', str(output)])
    script.main()
    report = json.loads(output.read_text())
    assert report['passed'] and report['records'][0]['count'] == 2
    assert report['checkpoint']['strict_state_dict_load']
    assert set(report['checkpoint']['forward_probes']) == {'space', 'layer'}
    assert data.read_bytes() == before
    monkeypatch.setattr('sys.argv', ['verify_symmetry_runtime', '--checkpoint', str(checkpoint)])
    with pytest.raises(ValueError, match='supply --records'):
        script.main()
