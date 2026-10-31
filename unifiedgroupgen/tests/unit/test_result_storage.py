"""Permanent evidence and disposable checks use distinct storage roots."""
import json
from pathlib import Path
import hashlib

from unifiedgroupgen.common.configuration import resolve_path, artifact_path
from unifiedgroupgen.tests.legacy.paths import resolve_path as old_resolve


def test_both_readers_resolve_frozen_audit_locations(tmp_path):
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/paths.yaml').write_text('mounts:\n  artifacts/main15/audits: data/dataset/audits/main15\n', encoding='utf-8')
    logical='artifacts/main15/audits/custom.json'
    expected=tmp_path/'data/dataset/audits/main15/custom.json'
    assert resolve_path(logical,tmp_path)==expected
    assert old_resolve(logical,tmp_path)==expected
    assert not (tmp_path/'artifacts').exists()


def test_intermediate_report_creation_does_not_create_artifacts(tmp_path):
    path=artifact_path('intermediate/checks/run/report.json',tmp_path)
    path.write_text('{"passed":true}',encoding='utf-8')
    assert path==tmp_path/'intermediate/checks/run/report.json'
    assert not (tmp_path/'artifacts').exists()


def test_unregistered_paths_are_not_silently_redirected(tmp_path):
    assert old_resolve('artifacts/unregistered/result.json',tmp_path)==tmp_path/'artifacts/unregistered/result.json'
    assert resolve_path('artifacts/unregistered/result.json',tmp_path)==tmp_path/'artifacts/unregistered/result.json'


def test_project_training_receipts_match_current_configuration():
    from unifiedgroupgen.common.configuration import load_experiments
    root=Path(__file__).resolve().parents[3]
    entries=load_experiments(root/'configs/experiments.yaml',root)
    assert len(entries)==15 and not (root/'artifacts').exists()
    for entry in entries.values():
        config=entry['config']
        path=old_resolve(config['data_audit'],root)
        assert path.is_relative_to(root/'data/dataset/audits/main15')
        saved=json.loads(path.read_text(encoding='utf-8'))
        assert saved['config_sha256']==hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
