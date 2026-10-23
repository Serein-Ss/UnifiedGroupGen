import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
import resume_loss_evaluation as recovery


def fixture(study):
    study.mkdir(parents=True)
    (study/'protocol.json').write_text(json.dumps({'epochs': 20, 'files': {}, 'frozen_files': {}}))
    for dataset, version in recovery.ORDER:
        run = study/'runs'/dataset/version
        run.mkdir(parents=True)
        (run/'model.pt').write_bytes(b'checkpoint test fixture')
        receipt = {'completed_epochs': 20, 'checkpoint_sha256': hashlib.sha256(b'checkpoint test fixture').hexdigest()}
        (run/'training_complete.json').write_text(json.dumps(receipt))
    return study


def test_plan_is_read_only_and_reuses_completed_evaluations(tmp_path):
    study = fixture(tmp_path/'study')
    done = study/'evaluations/mp20/before'
    done.mkdir(parents=True)
    receipt = json.loads((study/'runs/mp20/before/training_complete.json').read_text())
    (done/'complete.json').write_text(json.dumps(receipt))
    before = {p.relative_to(study): p.read_bytes() for p in study.rglob('*') if p.is_file()}
    plan = recovery.make_plan(study)
    assert plan['training_jobs'] == 0 and plan['cpu_start_method'] == 'spawn'
    assert [job['action'] for job in plan['jobs']] == ['reuse']+['evaluate']*5
    assert before == {p.relative_to(study): p.read_bytes() for p in study.rglob('*') if p.is_file()}


def test_changed_checkpoint_blocks_recovery(tmp_path):
    study = fixture(tmp_path/'study')
    (study/'runs/c2db51/after/model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError, match='checkpoint changed'):
        recovery.make_plan(study)


def test_active_controller_blocks_run_without_stopping_anything(tmp_path, monkeypatch):
    study = fixture(tmp_path/'study')
    (study/'status.json').write_text(json.dumps({'state': 'running', 'pid': 3586}))
    (tmp_path/'proc/3586').mkdir(parents=True)
    original_path = Path
    monkeypatch.setattr(recovery.sys, 'platform', 'linux')
    monkeypatch.setattr(recovery, 'Path', lambda p: tmp_path/str(p).lstrip('/') if str(p).startswith('/proc') else original_path(p))
    with pytest.raises(ValueError, match='still present'):
        recovery.require_inactive(study)
    assert json.loads((study/'status.json').read_text())['state'] == 'running'


def test_worker_selects_spawn_and_only_calls_evaluation(tmp_path, monkeypatch):
    study = tmp_path/'artifacts/study'
    calls = []
    monkeypatch.setattr(recovery, 'ROOT', tmp_path)
    monkeypatch.setattr(recovery.multiprocessing, 'set_start_method', lambda method, **kw: calls.append(method))
    monkeypatch.setattr(recovery, 'frozen_runner', lambda *_: SimpleNamespace(worker=lambda *args: calls.append(args)))
    monkeypatch.setattr(sys, 'argv', ['resume_loss_evaluation.py', '--study', str(study), '--worker',
                                     '--dataset', 'c2db51', '--version', 'after'])
    recovery.main()
    assert calls == ['spawn', (study, 'c2db51', 'after', 'evaluate')]
