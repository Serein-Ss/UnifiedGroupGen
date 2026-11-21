"""Current CSP manifest and CLI contracts without training."""
import json
from types import SimpleNamespace

import pytest

from unifiedgroupgen.common import configuration
from unifiedgroupgen.common.csp_cases import parse_csp_manifest, read_csp_manifest
from unifiedgroupgen.common.structures import read_records
from unifiedgroupgen.evaluation.joint_csp import evaluate_csp, target_view
from unifiedgroupgen.generation.unified import runner
from unifiedgroupgen.group_constraints.group_prior import NoFeasibleGroup
from unifiedgroupgen.joint import main
from unifiedgroupgen.pl_modules import factory
from unifiedgroupgen.pl_modules.unified import module
from unifiedgroupgen.tests.unit.test_joint_flow_v2 import record


@pytest.fixture
def mixed_checkpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(configuration, 'package_parent', lambda: tmp_path)
    (tmp_path/'.env').write_text(
        f'PROJECT_ROOT="{tmp_path.as_posix()}"\nHYDRA_JOBS="outputs"\n'
        'WANDB_DIR="wandb"\nUSE_WANDB_LOGGING="0"\n', encoding='utf-8')
    config = dict(model={'task_controls': True}, generation_task='csp',
                  dataset='mp20_c2db51', group_prior={})
    model = SimpleNamespace(backend='fixture', allowed_elements=range(1, 119),
                            max_atoms=192, max_orbits=52, load_state_dict=lambda *a, **k: None)
    monkeypatch.setattr(module, 'load_joint_checkpoint', lambda *a: dict(config=config, model={}))
    monkeypatch.setattr(factory, 'build_joint_model', lambda *a: model)
    return ['sample', '--checkpoint', str(tmp_path/'best.ckpt')]


def test_mixed_manifest_cli_roundtrip(tmp_path, mixed_checkpoint, monkeypatch):
    references = [dict(record(dataset), dataset=dataset, id='same-id') for dataset in ('mp_20', 'c2db_51')]
    inputs = []
    for index, reference in enumerate(references):
        path = tmp_path/f'{index}.jsonl'
        path.write_text(json.dumps(reference)+'\n', encoding='utf-8')
        inputs.extend(['--references', str(path)])
    manifest_path = tmp_path/'cases.json'
    main(['prepare-csp', *inputs, '--candidates', '2', '--output', str(manifest_path)])
    manifest = read_csp_manifest(manifest_path)
    assert [case.key for case in manifest.cases] == [('mp_20', 'same-id'), ('c2db_51', 'same-id')]
    assert all(set(case.to_dict()) == {'dataset', 'case_id', 'kind', 'counts'} for case in manifest.cases)
    views = {view['kind']: view for view in map(target_view, references)}
    seen = []

    def draw_group(prior, generator, *, kind, **kwargs):
        if generator.initial_seed() == 44:
            raise NoFeasibleGroup('fixture no group')
        return SimpleNamespace(kind=kind, number=1)

    def sample(model, specs, *, chemistries, **kwargs):
        seen.append([(spec.kind, dict(control.counts)) for spec, control in zip(specs, chemistries)])
        return [views[spec.kind] for spec in specs]

    monkeypatch.setattr(runner, 'sample_group', draw_group)
    monkeypatch.setattr(runner, 'sample_batch', sample)
    monkeypatch.setattr(runner, 'evaluation_record',
                        lambda view, *args, **kwargs: dict(view, status='generated'))
    ledger = tmp_path/'samples.jsonl'
    main([*mixed_checkpoint, '--case-manifest', str(manifest_path), '--batch-size', '4', '--output', str(ledger)])
    samples = read_records(ledger)
    assert len(seen) == 1 and [kind for kind, _ in seen[0]] == ['space', 'space', 'layer']
    assert seen[0][0][1] == dict(manifest.cases[0].chemistry.counts)
    assert seen[0][-1][1] == dict(manifest.cases[1].chemistry.counts)
    assert [(row['dataset'], row['candidate']) for row in samples] == [
        ('mp_20', 0), ('mp_20', 1), ('c2db_51', 0), ('c2db_51', 1)]
    assert all(row['training_dataset'] == 'mp20_c2db51' for row in samples)
    assert all(row['case_manifest_sha256'] == manifest.digest for row in samples)
    assert samples[2]['status'] == 'failed' and samples[2]['reason'] == 'fixture no group'

    report_path = tmp_path/'metrics.json'
    main(['evaluate-csp', '--samples', str(ledger), '--case-manifest', str(manifest_path),
          *inputs, '--ks', '1', '2', '--output', str(report_path)])
    report = json.loads(report_path.read_text(encoding='utf-8'))
    assert report['summary']['1']['match_rate'] == .5
    assert report['summary']['2']['match_rate'] == 1.
    assert report['summary']['2']['unattempted_candidates'] == 0
    assert report['summary']['2']['candidate_outcomes'] == {'generated': 3, 'failed': 1}
    assert report['by_dataset']['c2db_51']['1']['match_rate'] == 0.
    assert report['case_manifest_sha256'] == manifest.digest

    partial = evaluate_csp([samples[1], samples[3]], references, (1, 2), manifest)
    assert partial['summary']['1']['match_rate'] == 0.
    assert partial['summary']['1']['unattempted_candidates'] == 2
    assert partial['summary']['2']['match_rate'] == 1.
    with pytest.raises(ValueError, match='Duplicate CSP candidate'):
        evaluate_csp([samples[0], samples[0]], references, manifest=manifest)
    with pytest.raises(ValueError, match='manifest identity'):
        evaluate_csp([dict(samples[0], case_manifest_sha256='foreign')], references, manifest=manifest)


def test_mixed_checkpoint_single_case_requires_source_dataset(tmp_path, mixed_checkpoint, monkeypatch):
    monkeypatch.setattr(runner, 'sample_group',
                        lambda *a, **k: SimpleNamespace(number=1, kind='space'))
    monkeypatch.setattr(runner, 'sample_batch', lambda *a, **k: [object()])
    monkeypatch.setattr(runner, 'evaluation_record', lambda *a, **k: dict(status='failed'))
    ledger = tmp_path/'samples.jsonl'
    args = [*mixed_checkpoint, '--kind', 'space', '--formula', 'TiO2', '--case-id', 'fixture',
            '--output', str(ledger)]
    with pytest.raises(SystemExit):
        main(args)
    assert not ledger.exists()
    main([*args, '--case-dataset', 'mp_20'])
    row, = read_records(ledger)
    assert row['dataset'] == 'mp_20' and row['training_dataset'] == 'mp20_c2db51'


def test_manifest_rejects_geometry_and_duplicate_keys():
    case = dict(dataset='mp_20', case_id='example', kind='space', counts={'Ti': 2, 'O': 4})
    data = dict(schema='ugg_csp_cases_v1', candidates_per_case=1, cases=[case])
    with pytest.raises(ValueError, match='only dataset'):
        parse_csp_manifest(dict(data, cases=[dict(case, coordinates=[[0, 0, 0]])]))
    with pytest.raises(ValueError, match='Duplicate CSP case'):
        parse_csp_manifest(dict(data, cases=[case, case]))
