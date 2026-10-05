import json
from pathlib import Path
import pytest
import torch
import yaml
from unifiedgroupgen.conditioning import PropertyEncoder, attach_dataset_domains
from unifiedgroupgen.data import write_jsonl
from unifiedgroupgen.joint import grouped_splits, inventory_hash
from unifiedgroupgen.preflight import check_recovered_source, audit_files
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.training import file_hash


def make_record(dataset, identifier, element=14):
    descriptor = Descriptor('space', 2, (OrbitSpec('i', element),))
    return {'id': identifier, 'dataset': dataset, 'descriptor': descriptor.to_dict(),
            'state': compile_descriptor(descriptor).prior().tolist(), 'properties': {}}


def test_cfg_retains_calculation_context_and_rejects_missing_context():
    encoder = PropertyEncoder({'source_domain': {'type': 'categorical', 'num_classes': 2, 'context': True},
                               'gap': {'type': 'scalar'}}, hidden=8, dropout=1.0).eval()
    like = torch.zeros(8)
    first = encoder({'source_domain': 1, 'gap': 1.0}, like, force_null=True)
    second = encoder({'source_domain': 0, 'gap': 1.0}, like, force_null=True)
    assert torch.allclose(first-second, encoder.encoders['source_domain'].weight[1]-encoder.encoders['source_domain'].weight[0])
    encoder.train()
    assert torch.allclose(first, encoder({'source_domain': 1, 'gap': 2.0}, like))
    with pytest.raises(ValueError, match='required context'):
        encoder({'gap': 1.0}, like, force_null=True)


def test_attaching_domains_preserves_original_records_and_missing_labels():
    rows = [make_record('mp_20', 'm'), make_record('c2db_51', 'c')]
    rows[0]['properties']['gap'] = None
    config = {'dataset_domains': {'mp_20': 0, 'c2db_51': 1},
              'properties': {'source_domain': {'type': 'categorical', 'num_classes': 2, 'context': True}}}
    attached = attach_dataset_domains(rows, config)
    assert [r['properties']['source_domain'] for r in attached] == [0, 1]
    assert attached[0]['properties']['gap'] is None
    assert all('source_domain' not in r['properties'] for r in rows)


def test_recovery_gate_requires_all_rows_and_matching_original_and_recovered_files(tmp_path):
    folder = tmp_path/'recovered'
    folder.mkdir()
    report = {'passed': True, 'splits': {}}
    for split in ('train', 'val', 'test'):
        original, restored = tmp_path/f'original_{split}.csv', folder/f'{split}.csv'
        original.write_text('original'); restored.write_text('restored')
        report['splits'][split] = {'full_source': True, 'accepted': 1, 'input': 1, 'source_rows': 1, 'rejected': [],
                                  'source': str(original), 'source_sha256': file_hash(original),
                                  'recovered_sha256': file_hash(restored)}
    path = tmp_path/'recovery.json'
    path.write_text(json.dumps(report))
    config = {'data_root': 'recovered', 'recovery_report': 'recovery.json'}
    check_recovered_source(config, tmp_path)
    report['splits']['train']['source_rows'] = 2
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match='Incomplete'):
        check_recovered_source(config, tmp_path)
    report['splits']['train']['source_rows'] = 1
    path.write_text(json.dumps(report))
    (folder/'train.csv').write_text('changed')
    with pytest.raises(ValueError, match='differs'):
        check_recovered_source(config, tmp_path)
    (folder/'train.csv').write_text('restored')
    database = tmp_path/'source.db'
    database.write_text('original database')
    report['original_databases'] = {str(database): file_hash(database)}
    path.write_text(json.dumps(report))
    check_recovered_source(config, tmp_path)
    database.write_text('changed database')
    with pytest.raises(ValueError, match='ASE database differs'):
        check_recovered_source(config, tmp_path)


def test_joint_splits_keep_shared_materials_together_and_preserve_all_records():
    torch.manual_seed(31)
    records = [make_record('mp_20', f'mp-{i}', 8 if i == 10 else 14) for i in range(20)]
    records += [make_record('mpts_52', 'mp-1'), make_record('mpts_52', 'mp-10', 8)]
    splits, report = grouped_splits(records, {'mp_20': 'mp', 'mpts_52': 'mp'})
    assert inventory_hash(records) == inventory_hash(r for rows in splits.values() for r in rows)
    for identifier in ('mp-1', 'mp-10'):
        locations = [name for name, rows in splits.items() if any(r['id'] == identifier for r in rows)]
        assert len(locations) == 1
    assert report['input_records'] == 22 and report['components'] == 20
    trained = {o['atomic_number'] for r in splits['train'] for o in r['descriptor']['orbits']}
    assert trained == {8, 14}


def test_audit_duplicate_candidate_uses_the_correct_record_across_source_ids():
    from copy import deepcopy
    from unifiedgroupgen.preflight import inspect_records
    torch.manual_seed(31)
    first = make_record('mp_20', 'shared-id')
    second = make_record('mpts_52', 'shared-id')
    duplicate = deepcopy(second)
    duplicate.update({'dataset': 'mp_20', 'id': 'duplicate-of-second'})
    report = inspect_records({'train': [first, second], 'val': [duplicate]}, {'properties': {}})
    errors = [entry for entry in report['errors'] if entry['type'] == 'verified_duplicate_structure']
    assert len(errors) == 1
    assert errors[0]['id'] == 'duplicate-of-second'
    assert errors[0]['previous'] == ['train', 'shared-id']


def test_joint_lineage_detects_geometry_or_label_changes_even_with_updated_file_hash(tmp_path, monkeypatch):
    import unifiedgroupgen.joint as joint
    import unifiedgroupgen.preflight as preflight
    monkeypatch.setattr(joint, 'ROOT', tmp_path)
    monkeypatch.setattr(preflight, 'ROOT', tmp_path)
    torch.manual_seed(18)
    source_configs = {}
    for dataset, identifiers in {'mp_20': ['m0', 'm1', 'm2'], 'mpts_52': ['m3', 'm0', 'm4']}.items():
        folder = tmp_path/dataset
        folder.mkdir()
        config = {'dataset': dataset, 'kind': 'space', 'prepared_root': dataset, 'properties': {},
                  'require_data_audit': True, 'require_full_preparation': True, 'data_audit': f'{dataset}/audit.json'}
        config_path = tmp_path/f'{dataset}.yaml'
        config_path.write_text(yaml.safe_dump(config))
        source_configs[dataset] = config_path.name
        paths = {}
        for split, identifier in zip(('train', 'val', 'test'), identifiers):
            path = folder/f'{split}.jsonl'
            write_jsonl(path, [make_record(dataset, identifier)])
            Path(str(path)+'.report.json').write_text(json.dumps({'dataset': dataset, 'input': 1, 'accepted': 1, 'full_source': True}))
            Path(str(path)+'.manifest.json').write_text(json.dumps({'conversion_config': {'dataset': dataset, 'kind': 'space'}}))
            paths[split] = [path]
        assert audit_files(paths, config, folder/'audit.json')['passed']
    config = {'source_configs': source_configs, 'dataset_families': {'mp_20': 'mp', 'mpts_52': 'mp'},
              'prepared_root': 'joint', 'joint_split_manifest': 'joint/manifest.json'}
    manifest = joint.prepare_joint(config)
    assert joint.check_joint_lineage(config)['input_records'] == 6
    item = next(item for item in manifest['outputs'] if item['records'])
    path = Path(item['path'])
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows[0]['properties']['gap'] = 5.0
    write_jsonl(path, rows)
    item['sha256'] = file_hash(path)
    (tmp_path/'joint/manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='preserve every source record'):
        joint.check_joint_lineage(config)


def test_joint_space_layer_backward_with_retained_domain_context():
    from unifiedgroupgen.cli import build_models
    from unifiedgroupgen.training import joint_loss
    config = {'groups': [('space', 2), ('layer', 37)],
              'properties': {'gap': {'type': 'scalar'}, 'source_domain': {'type': 'categorical', 'num_classes': 2, 'context': True}},
              'max_orbits': 4, 'model': {'hidden': 16, 'proposal_layers': 1, 'flow_layers': 1, 'heads': 4, 'frequencies': 2}}
    first = make_record('mp_20', 'bulk')
    descriptor = Descriptor('layer', 37, (OrbitSpec('r', 8),))
    second = {'id': 'sheet', 'dataset': 'c2db_51', 'descriptor': descriptor.to_dict(),
              'state': compile_descriptor(descriptor).prior().tolist(), 'properties': {}}
    first['properties'].update({'gap': 1.0, 'source_domain': 0})
    second['properties'].update({'gap': 2.0, 'source_domain': 1})
    proposal, flow = build_models(config, 'cpu')
    losses = joint_loss(proposal, flow, [first, second], 0.5, 0.25)
    loss = losses['descriptor'].mean()+losses['flow'].mean()
    loss.backward()
    assert torch.isfinite(loss)
    for model, encoder in ((proposal, proposal.condition), (flow, flow.property_encoder)):
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        assert encoder.encoders['source_domain'].weight.grad is not None


def test_sampling_rejects_property_condition_without_labels_in_requested_domain(tmp_path):
    from types import SimpleNamespace
    from unifiedgroupgen.cli import build_models, sample
    config = {'groups': [('space', 2)], 'properties': {'gap': {'type': 'scalar'},
              'source_domain': {'type': 'categorical', 'num_classes': 2, 'context': True}},
              'max_orbits': 4, 'model': {'hidden': 16, 'proposal_layers': 1, 'flow_layers': 1, 'heads': 4, 'frequencies': 2},
              'domain_label_support': {'1': {'kinds': ['space'], 'property_counts': {'gap': 0}}}}
    proposal, flow = build_models(config, 'cpu')
    checkpoint = tmp_path/'model.pt'
    torch.save({'config': config, 'proposal': proposal.state_dict(), 'flow': flow.state_dict()}, checkpoint)
    args = SimpleNamespace(checkpoint=str(checkpoint), device='cpu', properties='{"source_domain":1,"gap":1}',
                           descriptor=None, composition=None, kind='space')
    with pytest.raises(ValueError, match='No training property labels'):
        sample(args)
