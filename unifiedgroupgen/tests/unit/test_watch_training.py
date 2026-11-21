"""Read-only panel tests using synthetic log records; never launch training."""
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from watch_training import JsonlTail, RunReader, phase_lines, render, watch


def append(path, row):
    with path.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row) + '\n')


def step(phase='train', batch=5, epoch=120, seconds=10, time=20):
    return dict(time=f'2026-11-15T21:05:{time:02}+08:00', phase=phase, event='step', epoch=epoch,
                batch=batch, batches=10, global_step=409000+batch, loss=10 if phase=='train' else 8,
                continuous=.2, element=4, jump=5.8 if phase=='train' else 3.8,
                batch_size=8, phase_seconds=seconds, lr=.00015)


def history():
    return dict(epoch=119, global_step=407040, metrics={'train/loss':10.5, 'val/loss':8.3},
                learning_rates_used=[.00015], early_stopping=dict(wait_count=12, patience=30, best_score=8.2))


@pytest.fixture
def run(tmp_path):
    directory = tmp_path/'full_mp20_eqv2_dng'/'jointflow_mp20_fp32_offline30_v2'
    directory.mkdir(parents=True)
    (directory/'model.status.json').write_text(json.dumps(dict(state='running', phase='training', epoch=120,
        global_step=409000, max_epochs=1000, batch_size=8, updated=1)), encoding='utf-8')
    append(directory/'model.history.jsonl', history())
    return directory


def test_train_validation_and_epoch_end_layout_and_frozen_clocks(run):
    path = run/'model.steps.jsonl'
    append(path, step(batch=10, seconds=20))
    reader = RunReader(run)
    first = reader.poll()
    text = render([first], width=200)
    assert 'Epoch 121/1000 (bs=8)' in text and 'early=12/30' in text and 'best=8.2000' in text
    assert 'Validate' not in text and 'geom= 0.2000' in text
    frozen = phase_lines('Train', first['train'], 200)
    for batch, seconds in ((1, 2), (5, 8)):
        append(path, step('validation', batch=batch, seconds=seconds, time=20+batch))
        snapshot = reader.poll()
        lines = render([snapshot], width=200).splitlines()
        train = next(line for line in lines if line.startswith('Train '))
        val = next(line for line in lines if line.startswith('Validate '))
        assert [i for i,c in enumerate(train) if c=='|'] == [i for i,c in enumerate(val) if c=='|']
        assert phase_lines('Train', snapshot['train'], 200) == frozen
        assert snapshot['validation']['phase_seconds'] == seconds
        assert snapshot['lr']==.00015
    append(path, dict(time='2026-11-15T21:05:30+08:00', phase='train', event='epoch_end', epoch=120,
                     global_step=410432, loss=10.1, validation_loss=8.1, lr=.00015,
                     loss_parts={'train':dict(continuous=.2, element=4, jump=5.9),
                                 'val':dict(continuous=.2, element=3.8, jump=4.1)},
                     best_validation_loss=8.1, early_stopping=dict(wait_count=0, patience=30),
                     train_seconds=20, validation_seconds=16))
    end = reader.poll()
    text = render([end], width=200)
    assert 'Validate' not in text and 'val=8.1000' in text and 'early=0/30' in text
    assert 'best=8.1000' in text and 'loss=10.1000' in text
    assert end['train']['phase_seconds']==20 and end['train']['jump']==5.9
    append(path, step(epoch=121, batch=1, seconds=1, time=31))
    new = reader.poll()
    assert new['validation']=={} and new['train']['batch']==1 and new['epoch']==121


def test_incremental_reader_waits_for_complete_newline_and_skips_bad_records(tmp_path):
    path = tmp_path/'steps.jsonl'
    append(path, {'batch':1})
    reader = JsonlTail(path)
    assert reader.poll()==[{'batch':1}] and reader.reset
    assert reader.poll()==[] and not reader.reset
    offset = reader.offset
    with path.open('ab') as stream:
        stream.write(b'{"batch":2')
    assert reader.poll()==[] and reader.offset==offset
    with path.open('ab') as stream:
        stream.write(b'}\nmalformed\n{"batch":3}\n')
    assert reader.poll()==[{'batch':2},{'batch':3}] and reader.invalid==1
    assert reader.poll()==[]


def test_opening_mid_validation_recovers_train_beyond_suffix(tmp_path):
    path = tmp_path/'steps.jsonl'
    append(path, step(batch=10, seconds=20))
    for i in range(100):
        append(path, step('validation', batch=i+1))
    reader = JsonlTail(path, steps=True, window=100)
    rows = reader.poll()
    assert any(r['phase']=='train' for r in rows) and rows[-1]['batch']==100
    assert reader.poll()==[]


def test_truncation_and_atomic_replacement_reset_tail(tmp_path):
    path = tmp_path/'history.jsonl'
    append(path, {'value':'long first record'})
    reader = JsonlTail(path)
    reader.poll()
    path.write_text('{}\n',encoding='utf-8')
    assert reader.poll()==[{}] and reader.reset
    replacement = tmp_path/'replacement'
    replacement.write_text('{"new":true}\n',encoding='utf-8')
    replacement.replace(path)
    assert reader.poll()==[{'new':True}] and reader.reset


def test_missing_or_partial_status_does_not_hide_steps(run):
    (run/'model.status.json').write_text('{',encoding='utf-8')
    append(run/'model.steps.jsonl', step())
    snapshot = RunReader(run).poll()
    assert snapshot['epoch']==120 and snapshot['global_step']==409005
    assert snapshot['train']['loss']==10


def test_status_only_fallback_does_not_fabricate_counts_or_components(run):
    snapshot = RunReader(run).poll()
    text = render([snapshot], width=200)
    assert '    -/-' in text and 'geom=      -' in text and 'ETA=       -' in text
    assert snapshot['best']==8.2 and 'Validate' not in text


def test_status_validation_start_shows_empty_validate_without_train_metric_leak(run):
    append(run/'model.steps.jsonl', step())
    status = json.loads((run/'model.status.json').read_text())
    status.update(phase='validation',updated=2000000000)
    (run/'model.status.json').write_text(json.dumps(status),encoding='utf-8')
    snapshot = RunReader(run).poll()
    assert snapshot['phase']=='validation' and snapshot['validation']=={}
    text = render([snapshot], width=200)
    val = next(line for line in text.splitlines() if line.startswith('Validate '))
    assert 'loss=      -' in val


def test_complete_epoch_count_is_not_incremented_twice(run):
    status = dict(state='complete',phase='finished',epoch=130,epochs_completed=130,
                  max_epochs=1000,batch_size=8,updated=2000000000)
    (run/'model.status.json').write_text(json.dumps(status),encoding='utf-8')
    assert 'Epoch 130/1000' in render([RunReader(run).poll()])


def test_narrow_terminal_keeps_all_loss_fields(run):
    append(run/'model.steps.jsonl',step())
    text = render([RunReader(run).poll()],width=110)
    assert all(key+'=' in text for key in ('loss','geom','elem','jump','best','early'))
    assert '\x1b' not in text


def test_live_view_restores_terminal_and_never_changes_inputs(run, monkeypatch):
    append(run/'model.steps.jsonl',step())
    paths = list(run.iterdir())
    before = {p:(p.read_bytes(),p.stat().st_mtime_ns) for p in paths}
    class Terminal(io.StringIO):
        def isatty(self):return True
    terminal = Terminal()
    monkeypatch.setattr(sys,'stdout',terminal)
    monkeypatch.setattr('watch_training.time.sleep',lambda _:(_ for _ in ()).throw(KeyboardInterrupt()))
    watch([RunReader(run)],interval=2)
    assert '\x1b[?1049h' in terminal.getvalue() and terminal.getvalue().endswith('\x1b[?25h\x1b[?1049l')
    assert set(run.iterdir())==set(paths)
    assert all((p.read_bytes(),p.stat().st_mtime_ns)==v for p,v in before.items())


def test_non_tty_requires_once(run, monkeypatch):
    monkeypatch.setattr(sys,'stdout',io.StringIO())
    with pytest.raises(ValueError,match='--once'):
        watch([RunReader(run)])


def test_standalone_cli_has_no_project_or_ml_dependency_and_two_models(run,tmp_path):
    script = Path(__file__).resolve().parents[3]/'scripts'/'watch_training.py'
    csp = tmp_path/'full_mp20_eqv2_csp'/'run'
    csp.mkdir(parents=True)
    append(run/'model.steps.jsonl',step())
    result = subprocess.run([sys.executable,'-B','-I',str(script),'--once',str(run),str(csp)],
                            cwd=tmp_path,text=True,capture_output=True,check=True)
    assert 'full_mp20_eqv2_dng' in result.stdout and 'full_mp20_eqv2_csp' in result.stdout
    assert '\x1b' not in result.stdout and not result.stderr


@pytest.mark.parametrize('value',['0','nan','inf'])
def test_cli_rejects_invalid_refresh_interval(run,value):
    script = Path(__file__).resolve().parents[3]/'scripts'/'watch_training.py'
    result = subprocess.run([sys.executable,'-B',str(script),'--once','--interval',value,str(run)],
                            text=True,capture_output=True)
    assert result.returncode!=0 and 'at least 0.5' in result.stderr
