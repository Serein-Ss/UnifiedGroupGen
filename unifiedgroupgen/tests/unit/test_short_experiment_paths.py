import json
from pathlib import Path

import pytest

from unifiedgroupgen.tests.legacy.paths import resolve_path
from relocate_short_experiments import finalize, verify_files
from import_training_outputs import digest


def test_archived_test_alias_leaves_main15_unchanged(tmp_path):
    registry = tmp_path/'outputs/tests/path_aliases.json'
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({'artifacts/loss_balance_20261023': 'outputs/tests/loss_balance_local/26-10-30-14-00-00'}))
    assert resolve_path('artifacts/loss_balance_20261023/runs/mp20/before/model.pt', tmp_path) == tmp_path/'outputs/tests/loss_balance_local/26-10-30-14-00-00/runs/mp20/before/model.pt'
    assert resolve_path('artifacts/runs/main15_seed42/mp20_u/model.pt', tmp_path) == tmp_path/'artifacts/runs/main15_seed42/mp20_u/model.pt'
    old = tmp_path/'artifacts/loss_balance_20261023/status.json'
    old.parent.mkdir(parents=True)
    old.write_text('{}')
    assert resolve_path(old, tmp_path) == old


def test_registry_cannot_redirect_outside_test_outputs(tmp_path):
    registry = tmp_path/'outputs/tests/path_aliases.json'
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({'artifacts/study': '../outside'}))
    with pytest.raises(ValueError, match='permitted directory'):
        resolve_path('artifacts/study/file.json', tmp_path)


def test_finalize_requires_old_source_to_be_moved(tmp_path):
    source = tmp_path/'artifacts/study'
    source.mkdir(parents=True)
    plan = {'state': 'ready_for_native_move', 'root': str(tmp_path), 'groups': ['study'],
            'moves': [{'source': str(source), 'target': str(tmp_path/'outputs/tests/study/date'), 'files': []}]}
    path = tmp_path/'plan.json'
    path.write_text(json.dumps(plan))
    with pytest.raises(ValueError, match='not moved'):
        finalize(path)
    assert not (tmp_path/'outputs/tests/path_aliases.json').exists()


def test_verification_detects_changed_bytes(tmp_path):
    path = tmp_path/'model.pt'
    path.write_bytes(b'original')
    files = [{'path': 'model.pt', 'size': path.stat().st_size, 'sha256': digest(path)}]
    verify_files(tmp_path, files)
    path.write_bytes(b'modified')
    with pytest.raises(ValueError, match='Hash mismatch'):
        verify_files(tmp_path, files)
