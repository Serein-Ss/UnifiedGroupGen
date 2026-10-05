import importlib.util
import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
import torch
from unifiedgroupgen.cli import build_models
from unifiedgroupgen.data import parse_cif
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor

DATA = Path(__file__).resolve().parents[2]/'data'
if not DATA.exists():
    DATA = Path(__file__).resolve().parents[1]/'raw_data'


def script(name):
    path = Path(__file__).resolve().parents[1]/'scripts'/f'{name}.py'
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_c2db_recovery_requires_matching_source_structure():
    source = DATA/'c2db_51/train.csv'
    row = pd.read_csv(source, nrows=1).iloc[0]
    original = parse_cif(row['cif'], 'cartesian_in_fractional_fields')
    atoms = {'cell': original.lattice.matrix.tolist(), 'numbers': list(original.atomic_numbers),
             'positions': original.cart_coords.tolist(), 'pbc': [True, True, False], 'energy': row['energy']}
    recovery = script('recover_c2db')
    restored, residual = recovery.verify_structure(row, atoms)
    assert residual < 1e-9 and len(restored) == int(row['natoms'])
    atoms['positions'][0][0] += 0.1
    with pytest.raises(ValueError, match='Cartesian'):
        recovery.verify_structure(row, atoms)
    atoms['positions'] = original.cart_coords.tolist()
    atoms['energy'] += 0.1
    with pytest.raises(ValueError, match='energy'):
        recovery.verify_structure(row, atoms)


def test_full_test_evaluation_retains_unsupported_labels(tmp_path, monkeypatch):
    evaluation = script('evaluate_checkpoint')
    config = {'kind': 'space', 'groups': [('space', 2)], 'properties': {}, 'allowed_elements': [14],
              'model': {'hidden': 16, 'proposal_layers': 1, 'flow_layers': 1, 'heads': 4, 'frequencies': 2},
              'max_orbits': 4, 'batch_size': 2, 'edge_budget': 100}
    proposal, flow = build_models(config, 'cpu')
    checkpoint, records, output = tmp_path/'model.pt', tmp_path/'test.jsonl', tmp_path/'result.json'
    torch.save({'config': config, 'proposal': proposal.state_dict(), 'flow': flow.state_dict(), 'epoch': 0}, checkpoint)
    rows = []
    for element in (14, 18):
        descriptor = Descriptor('space', 2, (OrbitSpec('i', element),))
        rows.append({'id': str(element), 'descriptor': descriptor.to_dict(), 'properties': {},
                     'state': compile_descriptor(descriptor).prior().tolist()})
    records.write_text('\n'.join(json.dumps(row) for row in rows))
    monkeypatch.setattr(evaluation, 'artifact_path', lambda value: Path(value))
    monkeypatch.setattr(sys, 'argv', ['evaluate_checkpoint.py', '--checkpoint', str(checkpoint),
                                     '--records', str(records), '--output', str(output), '--device', 'cpu'])
    evaluation.main()
    result = json.loads(output.read_text())
    assert result['requested_records'] == 2 and result['evaluated_records'] == 1
    assert result['label_coverage'] == 0.5
    assert result['unsupported_records'] == [{'id': '18', 'unseen_elements': [18]}]
    assert all(np.isfinite(value) for value in result['loss_on_supported_records'].values())


def test_original_ase_recovery_requires_exact_identity_and_fidelity(tmp_path):
    from ase import Atoms
    from ase.db import connect
    from ase.calculators.singlepoint import SinglePointCalculator
    source = DATA/'c2db_51/train.csv'
    row = pd.read_csv(source, nrows=1).iloc[0]
    original = parse_cif(row['cif'], 'cartesian_in_fractional_fields')
    atoms = Atoms(numbers=original.atomic_numbers, positions=original.cart_coords,
                  cell=original.lattice.matrix, pbc=[True, True, False])
    atoms.calc = SinglePointCalculator(atoms, energy=float(row['energy']))
    database = connect(tmp_path/'original.db')
    database.write(atoms, uid=row['uid'])
    recovery = script('recover_c2db')
    recovered, identifier = recovery.atoms_from_database(database, row)
    assert identifier == 1 and np.allclose(recovered['cell'], original.lattice.matrix)
    # Same composition cannot replace a missing exact UID/unique_id.
    changed = row.copy(); changed['uid'] = 'absent'; changed['unique_id'] = '0'*32
    with pytest.raises(KeyError):
        recovery.atoms_from_database(database, changed)
    changed = row.copy(); changed['energy'] += 0.1
    with pytest.raises(ValueError, match='energy'):
        recovery.atoms_from_database(database, changed)
    # Original ASE unique_id is also an exact identity, independent of UID aliases.
    changed = row.copy(); changed['uid'] = 'old_alias'; changed['unique_id'] = database.get(id=1).unique_id
    assert recovery.atoms_from_database(database, changed)[1] == 1


def test_cached_local_source_recovery_is_diagnostic_until_all_rows_verified(tmp_path, monkeypatch):
    from ase import Atoms
    from ase.db import connect
    from ase.calculators.singlepoint import SinglePointCalculator
    from unifiedgroupgen.training import file_hash
    source = DATA/'c2db_51/train.csv'
    row = pd.read_csv(source, nrows=1).iloc[0]
    original = parse_cif(row['cif'], 'cartesian_in_fractional_fields')
    atoms = Atoms(numbers=original.atomic_numbers, positions=original.cart_coords,
                  cell=original.lattice.matrix, pbc=[True, True, False])
    atoms.calc = SinglePointCalculator(atoms, energy=float(row['energy']))
    database_path = tmp_path/'original.db'
    connect(database_path).write(atoms, uid=row['uid'])
    csvs = tmp_path/'csvs'; csvs.mkdir()
    for split in ('train', 'val', 'test'):
        pd.DataFrame([row]).to_csv(csvs/f'{split}.csv', index=False)
    recovery = script('recover_c2db')
    monkeypatch.setattr(recovery, 'root_path', lambda value: Path(value))
    monkeypatch.setattr(recovery, 'artifact_path', lambda value: Path(value))
    destination = tmp_path/'recovered'
    monkeypatch.setattr(sys, 'argv', ['recover_c2db.py', '--source', str(csvs), '--output', str(destination),
                                     '--ase-db', str(database_path), '--limit', '1'])
    recovery.main()
    report = json.loads((destination/'recovery.json').read_text())
    assert not report['passed']
    assert report['original_databases'] == {str(database_path.resolve()): file_hash(database_path)}
    recovered = pd.read_csv(destination/'train.csv').iloc[0]
    assert recovered['official_structure_url'].startswith('ase-db:')
    assert json.loads(recovered['original_ase_source'])['kind'] == 'original_ase_database'


def test_formal_evaluation_requires_complete_audited_test_and_matching_checkpoint(tmp_path, monkeypatch):
    from unifiedgroupgen.training import file_hash
    evaluation = script('evaluate_checkpoint')
    audit, first, second = tmp_path/'audit.json', tmp_path/'first.jsonl', tmp_path/'second.jsonl'
    first.write_text('first test'); second.write_text('second test')
    report = {'passed': True, 'inputs': {'train': [{'sha256': 'train'}], 'val': [{'sha256': 'val'}],
              'test': [{'path': str(path), 'sha256': file_hash(path)} for path in (first, second)]}}
    audit.write_text(json.dumps(report))
    checkpoint = {'config': {'require_data_audit': True, 'data_audit': str(audit)},
                  'train_split_sha256': ['train'], 'val_split_sha256': ['val']}
    assert evaluation.check_heldout_inputs(checkpoint, [second, first])['sha256'] == file_hash(audit)
    with pytest.raises(ValueError, match='complete audited test'):
        evaluation.check_heldout_inputs(checkpoint, [first])
    checkpoint['train_split_sha256'] = ['different run']
    with pytest.raises(ValueError, match='training/validation'):
        evaluation.check_heldout_inputs(checkpoint, [first, second])
    checkpoint['train_split_sha256'] = ['train']
    second.write_text('modified labels')
    with pytest.raises(ValueError, match='complete audited test'):
        evaluation.check_heldout_inputs(checkpoint, [first, second])


def test_shared_evaluation_reports_source_losses_and_unsupported_denominators(tmp_path, monkeypatch):
    evaluation = script('evaluate_checkpoint')
    config = {'kind': 'space', 'groups': [('space', 2), ('layer', 37)], 'allowed_elements': [14],
              'model': {'hidden': 16, 'proposal_layers': 1, 'flow_layers': 1, 'heads': 4, 'frequencies': 2},
              'max_orbits': 4, 'batch_size': 3, 'edge_budget': 1000,
              'properties': {'gap': {'type': 'scalar'}, 'source_domain': {'type': 'categorical', 'num_classes': 2, 'context': True}},
              'dataset_domains': {'bulk': 0, 'sheet': 1}}
    proposal, flow = build_models(config, 'cpu')
    checkpoint = tmp_path/'model.pt'
    torch.save({'config': config, 'proposal': proposal.state_dict(), 'flow': flow.state_dict(), 'epoch': 0}, checkpoint)
    rows = []
    for kind, group, letter, element, dataset, gap in (
        ('space', 2, 'i', 14, 'bulk', 1.0), ('space', 2, 'i', 18, 'bulk', None), ('layer', 37, 'r', 14, 'sheet', 2.0)):
        descriptor = Descriptor(kind, group, (OrbitSpec(letter, element),))
        rows.append({'id': str(element), 'dataset': dataset, 'descriptor': descriptor.to_dict(),
                     'properties': {'gap': gap}, 'state': compile_descriptor(descriptor).prior().tolist()})
    paths = [tmp_path/'bulk.jsonl', tmp_path/'sheet.jsonl']
    paths[0].write_text('\n'.join(json.dumps(row) for row in rows[:2]))
    paths[1].write_text(json.dumps(rows[2]))
    output = tmp_path/'result.json'
    monkeypatch.setattr(evaluation, 'artifact_path', lambda value: Path(value))
    monkeypatch.setattr(sys, 'argv', ['evaluate_checkpoint.py', '--checkpoint', str(checkpoint), '--device', 'cpu',
                                     '--records', *(str(path) for path in paths), '--output', str(output)])
    evaluation.main()
    report = json.loads(output.read_text())
    assert report['requested_records'] == 3 and report['evaluated_records'] == 2
    sources = report['source_breakdown']
    assert sources['bulk']['label_coverage'] == 0.5 and sources['sheet']['label_coverage'] == 1
    assert sources['bulk']['unsupported_records'] == 1 and sources['sheet']['unsupported_records'] == 0
    assert sources['bulk']['property_label_counts'] == {'gap': 1}
    assert report['unsupported_records'][0]['dataset'] == 'bulk'
    for name, loss in report['loss_on_supported_records'].items():
        assert np.isfinite(loss)
        assert loss == pytest.approx(sum(source['loss_on_supported_records'][name] for source in sources.values())/2)
