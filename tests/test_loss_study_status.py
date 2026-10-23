import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from loss_study_status import DATASETS, SOURCES, snapshot, read_rows, display


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_initial_study_and_read_only_snapshot(tmp_path):
    write(tmp_path/'protocol.json', {'epochs': 20})
    write(tmp_path/'status.json', {'state': 'running', 'current_job': 'train/mp20/before'})
    original = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    result = snapshot(tmp_path)
    assert len(result['runs']) == 6
    assert result['requested'] == 3072 and result['attempted'] == 0
    assert result['metric_jobs'] == 16
    assert 2.5*3600 < result['training_remaining_seconds'] < 4*3600
    assert 70*60 < result['sampling_remaining_seconds'] < 85*60
    assert original == {p.name: p.read_bytes() for p in tmp_path.iterdir()}


def test_partial_epoch_eta_and_incomplete_last_line(tmp_path):
    write(tmp_path/'protocol.json', {'epochs': 3, 'validation_interval': 5})
    write(tmp_path/'status.json', {'state': 'running', 'current_job': 'train/mp20/before'})
    run = tmp_path/'runs/mp20/before'
    row = {'epoch': 0, 'training_seconds': 400, 'validation_seconds': 0,
           'checkpoint_seconds': 1, 'train': {'descriptor': 2, 'flow': 20, 'lattice': 1, 'coordinates': 19},
           'peak_cuda_bytes': 8*2**30, 'peak_cuda_reserved_bytes': 9*2**30}
    write(run/'model.status.json', {'epoch': 1, 'phase': 'train', 'step': 50, 'steps': 100, 'seconds': 30})
    history = run/'model.history.jsonl'
    history.write_text(json.dumps(row)+'\n{"epoch":', encoding='utf-8')
    assert read_rows(history) == [row]
    result = snapshot(tmp_path)
    model = result['runs'][0]
    assert model['epochs'] == 1 and model['training_seconds'] == 400
    assert model['training_remaining_seconds'] == pytest.approx(30+1+56.24+113.02+1)
    assert model['allocated_GiB'] == 8 and model['reserved_GiB'] == 9


def test_sampling_eta_uses_current_observed_speed(tmp_path):
    write(tmp_path/'protocol.json', {'epochs': 20})
    original = snapshot(tmp_path)
    write(tmp_path/'evaluations/mp20/before/mp_20_unconditional/sampling.status.json',
          {'state': 'running', 'attempted': 128, 'requested': 256, 'seconds': 256})
    current = snapshot(tmp_path)
    assert current['attempted'] == 128
    assert current['sampling_remaining_seconds'] == pytest.approx(
        original['sampling_remaining_seconds']-4*262.61+(4*256-128)*2)


def test_finished_study_has_zero_remaining_work(tmp_path):
    write(tmp_path/'protocol.json', {'epochs': 1})
    write(tmp_path/'status.json', {'state': 'complete'})
    for dataset in DATASETS:
        for version in ('before', 'after'):
            run = tmp_path/'runs'/dataset/version
            write(run/'model.status.json', {'state': 'complete'})
            (run/'model.history.jsonl').write_text(json.dumps({'epoch': 0})+'\n')
            evaluation = tmp_path/'evaluations'/dataset/version
            write(evaluation/'test.json', {})
            write(evaluation/'complete.json', {})
            for source in SOURCES[dataset]:
                for mode, count in (('unconditional', 256), ('conditional', 128)):
                    job = evaluation/(source+'_'+mode)
                    write(job/'sampling.status.json', {'attempted': count, 'seconds': count})
                    write(job/'standard_metrics.json', {})
    result = snapshot(tmp_path)
    assert result['total_remaining_seconds'] == [0, 0]
    assert result['training_remaining_seconds'] == 0 and result['sampling_remaining_seconds'] == 0
    assert result['test_remaining_seconds'] == 0 and result['cpu_reserve_seconds'] == [0, 0]
    assert result['test_done'] == 6 and result['metrics_done'] == 16


def test_display_includes_child_log_and_live_gpu(tmp_path, monkeypatch, capsys):
    write(tmp_path/'status.json', {'state': 'running', 'current_job': 'train/mp20/before'})
    (tmp_path/'train_mp20_before.log').write_text('Epoch 1/20 running\n', encoding='utf-8')
    monkeypatch.setattr('loss_study_status.subprocess.run', lambda *a, **k: SimpleNamespace(stdout='TEST_GPU'))
    display(tmp_path, snapshot(tmp_path))
    output = capsys.readouterr().out
    assert 'Epoch 1/20 running' in output and 'TEST_GPU' in output
