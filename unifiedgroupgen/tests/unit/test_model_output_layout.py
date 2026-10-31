import csv
import json
from pathlib import Path

import pytest

from unifiedgroupgen.common.configuration import model_result_directory, evaluation_directory
from unifiedgroupgen.tests.workflows.reorganize_model_outputs import migrate, digest, TRAINING_FILES


def test_checkpoint_local_task_paths(tmp_path):
    checkpoint = tmp_path/'model.pt'
    generated = model_result_directory(checkpoint, 'generate', 'mp20_dng_seed142')
    assert generated == tmp_path/'generate/mp20_dng_seed142'
    assert evaluation_directory(generated/'samples.jsonl') == tmp_path/'evaluate/mp20_dng_seed142'
    for name in ('../escape', 'a/b', 'a\\b', '.', ''):
        with pytest.raises(ValueError):
            model_result_directory(checkpoint, 'generate', name)
    with pytest.raises(ValueError, match='evaluation.output'):
        evaluation_directory(tmp_path/'external/samples.jsonl')


def test_public_entries_default_to_checkpoint_local_results(tmp_path, monkeypatch):
    from omegaconf import OmegaConf
    from unifiedgroupgen import generate, evaluate
    root = Path(__file__).resolve().parents[3]
    options = OmegaConf.load(root/'configs/generation/default.yaml')
    options.checkpoint = str(tmp_path/'model.best.pt')
    options.batch_size = 1
    captured = {}
    monkeypatch.setattr(generate, 'sample', lambda args: captured.update(output=args.output))
    generate.execute(OmegaConf.create({'task': 'generate', 'device': 'cpu', 'seed': 142,
                                      'threads': 1, 'generation': options}))
    assert captured['output'] == tmp_path/'generate/space_dng_seed142'
    metrics = OmegaConf.load(root/'configs/evaluation/default.yaml')
    metrics.samples = str(captured['output']/'samples.jsonl')
    monkeypatch.setattr(evaluate, 'evaluate_task', lambda args: captured.update(metrics=args.output))
    evaluate.execute(OmegaConf.create({'task': 'evaluate', 'device': 'cpu', 'threads': 1,
                                      'evaluation': metrics}))
    assert captured['metrics'] == tmp_path/'evaluate/space_dng_seed142'


def fixture(root):
    run = root/'outputs/mp20_u/26-10-30-12-50-31'
    run.mkdir(parents=True)
    for name in TRAINING_FILES:
        (run/name).write_text(name, encoding='utf-8')
    (run/'provenance').mkdir()
    (run/'provenance/source_config.yaml').write_text('model: original')
    with (root/'outputs/model_inventory.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['name', 'directory', 'best_checkpoint_sha256', 'last_checkpoint_sha256']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerow(dict(zip(fields, ['mp20_u', str(run.relative_to(root)),
                                         digest(run/'model.best.pt'), digest(run/'model.pt')])))
    task = root/'outputs/baselines_256_local_seed142/date/mp20_u/mp_20/de_novo'
    task.mkdir(parents=True)
    (task/'sample_00000.cif').write_bytes(b'exact CIF bytes\n')
    rows = [{'index': 0, 'status': 'generated', 'cif_sha256': digest(task/'sample_00000.cif')},
            {'index': 1, 'status': 'failed', 'reason': 'preserve this failure'}]
    (task/'samples.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    (task/'metrics').mkdir()
    (task/'metrics/summary.json').write_text('{"rate": 0.5}')
    component = task.parents[1]/'test'
    component.mkdir()
    (component/'summary.json').write_text('{"state": "complete"}')
    return run, task


def test_migration_preserves_weights_candidates_cifs_and_metrics(tmp_path):
    run, task = fixture(tmp_path)
    weights = {name: digest(run/name) for name in ('model.pt', 'model.best.pt')}
    metric = (task/'metrics/summary.json').read_bytes()
    assert migrate(tmp_path, 'test')['tasks'] == 1
    result = migrate(tmp_path, 'test', run=True)
    assert result['state'] == 'verified' and result['component_checks'] == 1
    assert {p.name for p in (tmp_path/'outputs').iterdir()} == {'mp20_u'}
    assert {p.name for p in run.iterdir()} == TRAINING_FILES | {'generate', 'evaluate'}
    generated = run/'generate/baseline256_mp_20_de_novo_seed142'
    rows = [json.loads(line) for line in (generated/'samples.jsonl').read_text().splitlines()]
    assert len(rows) == 2 and rows[1]['reason'] == 'preserve this failure'
    assert rows[0]['cif'].encode() == b'exact CIF bytes\n'
    assert not list((tmp_path/'outputs').rglob('*.cif'))
    assert (run/'evaluate/baseline256_mp_20_de_novo_seed142/summary.json').read_bytes() == metric
    assert weights == {name: digest(run/name) for name in weights}
    assert (tmp_path/'intermediate/output_layout/test/migration.json').exists()


def test_corrupt_original_cif_aborts_before_cleanup(tmp_path):
    run, task = fixture(tmp_path)
    (task/'sample_00000.cif').write_text('changed')
    with pytest.raises(ValueError, match='CIF hash'):
        migrate(tmp_path, 'test', run=True)
    assert (task/'samples.jsonl').exists()
    assert (run/'provenance/source_config.yaml').exists()
