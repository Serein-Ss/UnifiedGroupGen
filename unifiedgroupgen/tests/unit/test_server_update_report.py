"""Report updates without modifying experiments or exporting private/large contents."""
import csv
import io
import json
from pathlib import Path
import subprocess
import zipfile

import pytest

from submit_files.server_update_report import collect, export_report


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding='utf-8')


@pytest.fixture
def campaign(tmp_path):
    root = tmp_path/'project'
    root.mkdir()
    folder = root/'intermediate/hpc_runs/example'
    manifest = folder/'campaign.json'
    write(manifest, {'name': 'example', 'revision': 'frozen-training-code', 'folder': str(folder)})
    write(folder/'state/train_model.json', {'phase': 'train', 'model': 'model', 'state': 'complete'})
    generated = root/'outputs/model/example/generate/dng'
    metrics = root/'outputs/model/example/evaluate/dng'
    write(generated/'sampling.status.json', {'attempted': 3})
    (generated/'samples.jsonl').write_text('PRIVATE_LARGE_LEDGER')
    write(metrics/'summary.json', {'requested': 3})
    write(folder/'evaluation/model/pipeline.json', {
        'state': 'complete', 'identity': {'jobs': [{'name': 'dng'}]},
        'jobs': [{'state': 'complete', 'directory': str(generated),
                  'metrics': {'metric_directory': str(metrics)}}]})
    (folder/'evaluation/model/summary.csv').write_text('model,requested,basic_valid\nmodel,3,0.5\n')
    (root/'.env').write_text('PRIVATE_ENV_SECRET')
    (root/'outputs/model/example/model.best.pt').write_bytes(b'PRIVATE_WEIGHT_CONTENT')
    subprocess.run(['git', 'init', str(root)], check=True, capture_output=True)
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '--allow-empty', '-m', 'Initial code'], check=True, capture_output=True)
    return root, manifest, folder


def test_portable_report_keeps_inputs_and_private_contents_untouched(campaign, tmp_path):
    root, manifest, folder = campaign
    before = {p: p.read_bytes() for p in root.rglob('*') if p.is_file() and '.git' not in p.parts}
    archive, result = export_report(root, manifest, tmp_path/'reports')
    assert all(p.read_bytes() == content for p, content in before.items())
    assert result['git']['head']['returncode'] == 0
    assert result['campaign']['revision'] == 'frozen-training-code'
    assert result['evaluations']['model']['task_states'] == {'complete': 1}
    assert not result['changes']['baseline_available']
    weight = result['files']['outputs/model/example/model.best.pt']
    assert 'sha256' not in weight and weight['bytes'] == len(b'PRIVATE_WEIGHT_CONTENT')
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert set(bundle.namelist()) == {'snapshot.json', 'report.md', 'git_log.txt', 'metrics.csv'}
        content = b''.join(bundle.read(name) for name in bundle.namelist())
        for secret in (b'PRIVATE_ENV_SECRET', b'PRIVATE_WEIGHT_CONTENT', b'PRIVATE_LARGE_LEDGER'):
            assert secret not in content
        rows = list(csv.DictReader(io.StringIO(bundle.read('metrics.csv').decode('utf-8-sig'))))
        assert rows == [{'basic_valid': '0.5', 'model': 'model', 'requested': '3'}]


def test_repeat_snapshot_records_changes_and_missing_files(campaign, tmp_path):
    root, manifest, folder = campaign
    output = tmp_path/'reports'
    export_report(root, manifest, output)
    _, unchanged = export_report(root, manifest, output)
    assert unchanged['changes']['baseline_available']
    assert not any(unchanged['changes'][key] for key in ('added', 'modified', 'removed'))
    assert not unchanged['changes']['code_commit_changed']
    subprocess.run(['git', '-C', str(root), '-c', 'user.name=Fixture', '-c', 'user.email=fixture@example.invalid',
                    'commit', '--allow-empty', '-m', 'Update code'], check=True, capture_output=True)
    write(folder/'state/train_model.json', {'phase': 'train', 'model': 'model', 'state': 'failed'})
    weight = root/'outputs/model/example/model.best.pt'
    weight.unlink()
    _, changed = export_report(root, manifest, output)
    assert changed['changes']['code_commit_changed']
    assert 'Update code' in changed['git']['log']['stdout']
    assert 'intermediate/hpc_runs/example/state/train_model.json' in changed['changes']['modified']
    assert 'outputs/model/example/model.best.pt' in changed['changes']['removed']
    summary = folder/'evaluation/model/summary.csv'
    summary.unlink()
    result, rows = collect(root, manifest)
    assert rows == [] and result['warnings']
    assert result['files'][summary.relative_to(root).as_posix()] == {'exists': False}
    pipeline = folder/'evaluation/model/pipeline.json'
    data = json.loads(pipeline.read_text(encoding='utf-8'))
    data['jobs'] = []
    write(pipeline, data)
    result, _ = collect(root, manifest)
    assert any('incomplete task records' in warning for warning in result['warnings'])


def test_missing_campaign_refuses_to_create_report(tmp_path):
    output = tmp_path/'reports'
    with pytest.raises(ValueError, match='existing campaign'):
        export_report(tmp_path, tmp_path/'missing.json', output)
    assert not output.exists()
