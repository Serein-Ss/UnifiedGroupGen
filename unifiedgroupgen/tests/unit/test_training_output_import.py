"""Historical directory dates must not masquerade as training timestamps."""
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

spec = importlib.util.spec_from_file_location('training_import', Path(__file__).resolve().parents[3] / 'scripts/import_training_outputs.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture_archive(path, missing=None, unsafe=False):
    sources = [(f'artifacts/runs/main15_seed42/{name}/', f'configs/main15/{name}.yaml') for name in module.MAIN15]
    sources += [('artifacts/runs/local_large_seed42/mp20/', 'configs/mp20_local.yaml'),
                ('artifacts/runs/local_large_seed42/mpts52/', 'configs/mpts52_local.yaml'),
                ('artifacts/runs/joint_space_layer_large_seed42/', 'configs/joint_space_layer_large.yaml')]
    with ZipFile(path, 'w') as archive:
        archive.writestr('UnifiedGroupGen/artifacts/runs/main15_seed42/runner.json', json.dumps({'jobs': []}))
        for source, config in sources:
            for name in module.REQUIRED:
                if source + name == missing:
                    continue
                value = json.dumps({'state': 'complete', 'completed_epochs': 1}) if name.endswith('status.json') else (
                    '{"epoch": 0}\n' if name.endswith('history.jsonl') else 'fixture checkpoint bytes, not a trained model')
                archive.writestr('UnifiedGroupGen/' + source + name, value)
            archive.writestr('UnifiedGroupGen/' + config, 'seed: 42\n')
            archive.writestr('UnifiedGroupGen/' + source + 'train.log', 'training log\n')
        if unsafe:
            archive.writestr('UnifiedGroupGen/' + sources[0][0] + '../../escape.pt', 'bad')
    return path


def test_copy_records_import_time_and_preserves_source(tmp_path, capsys):
    source = fixture_archive(tmp_path / 'source.zip')
    before = module.digest(source)
    output = tmp_path / 'outputs'
    index = module.import_runs(source, output, datetime(2026, 10, 30, 6, 7, 8, tzinfo=timezone.utc))
    registry = json.loads(index.read_text(encoding='utf-8'))
    assert registry['directory_date_kind'] == 'import_time'
    assert registry['training_started_at'] is None
    assert module.digest(source) == before
    assert len(registry['models']) == 15
    assert (output / 'mp20_u/26-10-30-14-07-08/model.best.pt').is_file()
    assert module.verify_import(index) == {'models': 15, 'verified_files': 75, 'state': 'verified'}
    manifest = json.loads(Path(registry['models'][0]['manifest']).read_text(encoding='utf-8'))
    assert not list(output.rglob('provenance'))
    assert not list(output.rglob('import_manifest.json'))
    assert '非训练开始时间' in manifest['date_note']
    assert manifest['training_started_at'] is None
    with pytest.raises(FileExistsError):
        module.import_runs(source, output, datetime(2026, 10, 30, 6, 7, 8, tzinfo=timezone.utc))
    (output / 'mp20_u/26-10-30-14-07-08/model.best.pt').write_bytes(b'altered')
    with pytest.raises(ValueError, match='hash mismatch'):
        module.verify_import(index)


def test_missing_weight_rejected_before_output_creation(tmp_path):
    source = fixture_archive(tmp_path / 'source.zip', missing='artifacts/runs/main15_seed42/joint_eh/model.best.pt')
    output = tmp_path / 'outputs'
    with pytest.raises(ValueError, match='Missing training artifact'):
        module.import_runs(source, output)
    assert not output.exists()


def test_path_traversal_rejected_before_output_creation(tmp_path):
    source = fixture_archive(tmp_path / 'source.zip', unsafe=True)
    output = tmp_path / 'outputs'
    with pytest.raises(ValueError, match='Unsafe archive path'):
        module.import_runs(source, output)
    assert not output.exists()
