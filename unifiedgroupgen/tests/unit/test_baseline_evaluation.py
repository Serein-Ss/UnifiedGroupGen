import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]


def runner():
    spec = importlib.util.spec_from_file_location('baseline_evaluation', ROOT/'unifiedgroupgen/tests/legacy/scripts/evaluate_baselines.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_budget_and_joint_domain_split():
    module = runner()
    jobs = module.plan()
    assert len(jobs) == 4
    assert sum(j['fixed_group'] for j in jobs) == 3100
    for mode in ('de_novo', 'composition', 'fixed_description'):
        assert sum(j[mode] for j in jobs) == 768
    assert sum(sum(j[k] for k in module.MODES) for j in jobs) == 5404
    assert [j['de_novo'] for j in jobs if j['model'] == 'joint_u'] == [128, 128]


def test_resume_preserves_failures_and_verifies_cifs(tmp_path):
    module = runner()
    cif = tmp_path/'sample_00000.cif'
    cif.write_text('original', encoding='ascii')
    rows = [{'index': 0, 'status': 'generated', 'cif_sha256': module.digest(cif)},
            {'index': 1, 'status': 'failed', 'reason': 'unsupported element'}]
    path = tmp_path/'samples.jsonl'
    complete = ''.join(json.dumps(r)+'\n' for r in rows).encode()
    path.write_bytes(complete+b'{"index": 2')
    assert module.read_attempts(tmp_path) == rows
    assert path.read_bytes() == complete
    assert (tmp_path/'samples.interrupted-tail.bin').exists()
    cif.write_text('changed', encoding='ascii')
    with pytest.raises(ValueError, match='changed'):
        module.read_attempts(tmp_path)


def test_independent_attempt_seed():
    module = runner()
    before = module.seed_for(142, 'joint_u', 'mp_20', 'composition', 12)
    assert before == module.seed_for(142, 'joint_u', 'mp_20', 'composition', 12)
    assert before != module.seed_for(142, 'joint_u', 'mp_20', 'de_novo', 12)


def test_cuda_guard_does_not_stop_existing_training(monkeypatch):
    module = runner()
    class Result:
        stdout = '24469\n'
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: Result())
    with pytest.raises(RuntimeError, match='Keep training running'):
        module.gpu_guard('cuda')
    module.gpu_guard('cpu')


def test_known_server_prefix_relocation_only():
    module = runner()
    assert module.resolve_input('/root/private_data/rszhong/workspace/UnifiedGroupGen/artifacts/test.json') == ROOT/'artifacts/test.json'
    assert module.resolve_input('artifacts/test.json') == ROOT/'artifacts/test.json'


def test_baseline_data_audit_checks_hashes(tmp_path, monkeypatch):
    module = runner()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    inputs = {}
    config = {'data_revision': 'source_integrity_v1', 'require_data_audit': True,
              'data_audit': 'audit.json', 'properties': {}}
    saved = {'config': config}
    for split in ('train', 'val', 'test'):
        path = tmp_path/(split+'.jsonl')
        path.write_text(json.dumps({'dataset': 'mp_20', 'id': split})+'\n', encoding='utf-8')
        inputs[split] = [{'path': str(path), 'sha256': module.digest(path)}]
        config[split+'_files'] = [path.name]
        if split != 'test':
            saved[split+'_split_sha256'] = [module.digest(path)]
    (tmp_path/'audit.json').write_text(json.dumps({'passed': True, 'inputs': inputs}))
    records, audit = module.verify_inputs(saved)
    assert len(records) == 1
    assert len(audit['files']) == 3
    (tmp_path/'test.jsonl').write_text('{}\n')
    with pytest.raises(ValueError, match='test bytes'):
        module.verify_inputs(saved)


def test_component_nll_matches_existing_loss():
    import torch
    import sys
    sys.path.insert(0, str(ROOT/'unifiedgroupgen/tests/legacy/scripts'))
    from legacy_eval.components import descriptor_batch
    from unifiedgroupgen.pl_modules.descriptor import DescriptorTransformer
    from unifiedgroupgen.group_constraints.compiler import Descriptor, OrbitSpec
    model = DescriptorTransformer([('space', 1)], {}, hidden=32, layers=1, heads=4,
                                  max_atoms=4, max_orbits=2, allowed_elements=[8])
    model.eval()
    descriptor = Descriptor('space', 1, (OrbitSpec('a', 8),))
    with torch.inference_mode():
        expected = float(model.loss(descriptor, {}))
        actual = descriptor_batch(model, [descriptor], [{}])[0]
    assert actual['descriptor_loss'] == pytest.approx(expected, abs=1e-6)


def test_reconstruction_identical_real_geometries():
    from pymatgen.core import Lattice, Structure
    module = runner()
    crystal = Structure(Lattice.cubic(4), ['Si'], [[0, 0, 0]])
    result = module.reconstruction_metrics(crystal, crystal.copy(), 'space')
    assert result['reference_match']
    assert result['rmsd_angstrom'] == pytest.approx(0, abs=1e-8)
    layer = Structure(Lattice.from_parameters(3, 3, 20, 90, 90, 120), ['Mo', 'S', 'S'],
                      [[0, 0, .5], [1/3, 2/3, .42], [1/3, 2/3, .58]])
    result = module.reconstruction_metrics(layer, layer.copy(), 'layer')
    assert result['reference_match']
    assert result['rmsd_angstrom'] == pytest.approx(0, abs=1e-8)


def test_isolated_symmetry_tolerances():
    from pymatgen.core import Lattice, Structure
    module = runner()
    crystal = Structure(Lattice.cubic(4), ['Si'], [[0, 0, 0]])
    identified = module.symmetry_tolerances(crystal, 'space')
    assert identified == {'0.0001': 221, '0.001': 221, '0.01': 221}
