import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
import loss_balance_study as study_runner
from pack_loss_study import archive_results


def test_frozen_evaluator_includes_composition_scaler(tmp_path):
    frozen = study_runner.freeze_sources(tmp_path/'study')
    resource = frozen/'src/unifiedgroupgen/benchmark_comp_scaler.json'
    assert resource.read_bytes() == (study_runner.ROOT/'src/unifiedgroupgen/benchmark_comp_scaler.json').read_bytes()
    spec = importlib.util.spec_from_file_location('frozen_metrics_probe', frozen/'src/unifiedgroupgen/standard_metrics.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    _, _, means, stds = module._fingerprint_tools()
    assert len(means) == len(stds) and len(means) > 0


def test_archive_preserves_evidence_and_excludes_weights_and_reference_cache(tmp_path):
    study = tmp_path/'study'
    study.mkdir()
    (study/'status.json').write_text(json.dumps({'state': 'running'}))
    with pytest.raises(ValueError, match='not complete'):
        archive_results(study, tmp_path/'early.zip')
    (study/'status.json').write_text(json.dumps({'state': 'complete'}))
    (study/'model.pt').write_bytes(b'weights stay on server')
    (study/'model.history.jsonl').write_text('{"epoch":0}\n')
    (study/'reference_metrics').mkdir()
    (study/'reference_metrics/reference.json').write_text('{}')
    (study/'initial').mkdir()
    (study/'initial/private.json').write_text('{}')
    (study/'generated.cif').write_text('data_example')
    output = tmp_path/'audit.zip'
    report = archive_results(study, output)
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        assert 'study/generated.cif' in archive.namelist()
        assert not any(n.endswith('.pt') or '/initial/' in n or '/reference_metrics/' in n for n in archive.namelist())
        manifest = json.loads(archive.read('audit_manifest.json'))
        for entry in manifest['files']:
            assert hashlib.sha256(archive.read('study/'+entry['path'])).hexdigest() == entry['sha256']
    assert output.with_suffix('.zip.sha256').read_text().startswith(report['sha256'])


def test_finished_controller_automatically_packs_results(tmp_path, monkeypatch):
    study = tmp_path/'artifacts/study'
    study_runner.freeze_sources(study)
    monkeypatch.setattr(study_runner, 'ROOT', tmp_path)
    monkeypatch.setattr(study_runner, 'prepare', lambda *_: {'limits': 'test'})
    monkeypatch.setattr(study_runner, 'summarize', lambda *_: (study/'REPORT.md').write_text('test archive only'))
    for dataset, version in study_runner.ORDER:
        run = study/'runs'/dataset/version
        run.mkdir(parents=True)
        checkpoint = run/'model.pt'
        checkpoint.write_bytes(b'archive test fixture')
        marker = {'checkpoint_sha256': hashlib.sha256(checkpoint.read_bytes()).hexdigest()}
        (run/'training_complete.json').write_text(json.dumps(marker))
        evaluation = study/'evaluations'/dataset/version
        evaluation.mkdir(parents=True)
        (evaluation/'complete.json').write_text(json.dumps(marker))
    study_runner.controller(study, SimpleNamespace(wait_for=None))
    assert json.loads((study/'status.json').read_text())['state'] == 'complete'
    archives = list((tmp_path/'artifacts/exports').glob('*.zip'))
    assert len(archives) == 1
