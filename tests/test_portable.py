import importlib.util
import io
import json
from pathlib import Path
import tarfile

import pytest


def script(name):
    path = Path(__file__).resolve().parents[1] / 'scripts' / f'{name}.py'
    specification = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module


def bundle(tmp_path, member='artifacts/train.jsonl'):
    installer = script('unpack_training_data')
    source = tmp_path / 'bundle'
    source.mkdir()
    part = source / 'full_training.tar.gz.part000'
    content = b'{"state": [1]}\n'
    metadata = json.dumps({'inputs': [{'path': 'E:\\old\\project\\artifacts\\train.jsonl'}],
                           'original_databases': {'E:\\old\\project\\archive.db': 'digest'},
                           'source': 'E:\\old\\data\\c2db_51\\train.csv'}).encode()
    files = []
    with tarfile.open(part, 'w:gz') as archive:
        for name, data in ((member, content), ('artifacts/audit.json', metadata)):
            info = tarfile.TarInfo(name)
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
            import hashlib
            files.append({'path': name, 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
    digest = installer.sha256(part)
    manifest = {'original_root': 'E:\\old\\project', 'original_data_root': 'E:\\old\\data',
                'parts': [{'name': part.name, 'bytes': part.stat().st_size, 'sha256': digest}],
                'archive_sha256': digest, 'files': files, 'total_source_records': 1}
    (source / 'manifest.json').write_text(json.dumps(manifest))
    return installer, source


def test_portable_import_preserves_records_and_rebinds_metadata_idempotently(tmp_path):
    installer, source = bundle(tmp_path)
    root = tmp_path / 'new_checkout'
    installer.unpack(source, root)
    assert (root / 'artifacts/train.jsonl').read_bytes() == b'{"state": [1]}\n'
    audit = json.loads((root / 'artifacts/audit.json').read_text())
    assert audit['inputs'][0]['path'] == str(root / 'artifacts/train.jsonl')
    assert list(audit['original_databases']) == [str(root / 'archive.db')]
    assert audit['source'] == str(root / 'raw_data/c2db_51/train.csv')
    installer.unpack(source, root)
    (root / 'artifacts/train.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='Existing data differ'):
        installer.unpack(source, root)


def test_portable_import_rejects_corrupt_archive_before_extracting(tmp_path):
    installer, source = bundle(tmp_path)
    part = source / 'full_training.tar.gz.part000'
    part.write_bytes(part.read_bytes()[:-1] + b'x')
    root = tmp_path / 'new_checkout'
    with pytest.raises(ValueError, match='Corrupt'):
        installer.unpack(source, root)
    assert not (root / 'artifacts/train.jsonl').exists()


def test_portable_import_rejects_path_traversal(tmp_path):
    installer, source = bundle(tmp_path, '../outside.jsonl')
    with pytest.raises(ValueError, match='Unsafe inventory'):
        installer.unpack(source, tmp_path / 'new_checkout')
    assert not (tmp_path / 'outside.jsonl').exists()


def test_remote_plan_covers_full_shared_evaluation_and_all_sources():
    runner = script('train_remote')
    commands = runner.jobs('all', False)
    assert [name for name, _ in commands[:3]] == ['references', 'joint_train', 'joint_test']
    assert len(commands) == 9
    assert all('--count' in command and command[command.index('--count') + 1] == '64'
               for name, command in commands[3:])
    assert {json.loads(command[command.index('--properties') + 1])['source_domain']
            for name, command in commands[3:]} == {0, 1, 2}
    assert not any('complete_pipeline.py' in command for _, command in commands)
