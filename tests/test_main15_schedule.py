import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import pytest
import yaml

from unifiedgroupgen.main_experiments import configurations

ROOT = Path(__file__).resolve().parents[1]
ORDER = ['mp20_u', 'c2db51_u', 'joint_u', 'mp20_all', 'c2db51_all', 'joint_all',
         'mp20_ef', 'mp20_eg', 'mp20_eh', 'c2db51_ef', 'c2db51_eg', 'c2db51_eh',
         'joint_ef', 'joint_eg', 'joint_eh']


def runner():
    spec = importlib.util.spec_from_file_location('main15_schedule', ROOT/'scripts/train_main15.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_plan_has_approved_order_runtime_and_unchanged_configs(tmp_path, monkeypatch, capsys):
    configs = configurations(ROOT)
    assert list(configs) == ORDER
    hashes = {}
    for name, config in configs.items():
        path = ROOT/f'configs/main15/{name}.yaml'
        assert yaml.safe_load(path.read_text(encoding='utf-8')) == config
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    module = runner()
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'configurations', lambda _: configs)
    def forbidden(*args, **kwargs):
        pytest.fail('Plan mode must not launch subprocesses')
    monkeypatch.setattr(module.subprocess, 'run', forbidden)
    monkeypatch.setattr(module.subprocess, 'Popen', forbidden)
    monkeypatch.setattr(sys, 'argv', ['train_main15', '--plan'])
    module.main()
    jobs = json.loads(capsys.readouterr().out)
    assert [job['name'] for job in jobs] == ORDER
    assert not list(tmp_path.iterdir())
    for job in jobs:
        command = job['command']
        def value(flag):
            return command[command.index(flag)+1]
        c2db = job['name'].startswith('c2db51_')
        assert int(value('--batch-size')) == (768 if c2db else 256)
        assert int(value('--edge-budget')) == (200000 if c2db else 400000)
        assert value('--threads') == '4' and command.index('--threads') < command.index('train')
        assert float(value('--coordinate-length-scale')) == 3
        assert value('--edge-chunk-size') == '32768'
        assert value('--validation-interval') == '5'
        assert '--no-checkpoint' in command
        expected = 0 if job['name'].endswith('_u') else 3 if job['name'].endswith('_all') else 1
        assert len(job['target_properties']) == expected
        assert job['cond_dropout'] == (0 if expected == 0 else .1)
        path = ROOT/f'configs/main15/{job["name"]}.yaml'
        assert hashlib.sha256(path.read_bytes()).hexdigest() == hashes[job['name']]


def test_serial_execution_resume_and_runtime_mismatch(tmp_path, monkeypatch):
    module = runner()
    configs = configurations(ROOT)
    monkeypatch.setattr(module, 'ROOT', tmp_path)
    monkeypatch.setattr(module, 'configurations', lambda _: configs)
    for name in configs:
        path = tmp_path/f'configs/main15/{name}.yaml'
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes((ROOT/f'configs/main15/{name}.yaml').read_bytes())
    monkeypatch.setattr(module.subprocess, 'run', lambda *a, **k: None)
    events = []
    class Process:
        pid = 123
        def __init__(self, command, **kwargs):
            self.name = Path(command[command.index('--config')+1]).stem
            events.append(('start', self.name))
            assert kwargs['env']['OMP_NUM_THREADS'] == '4'
            output = tmp_path/command[command.index('--output')+1]
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(b'mocked checkpoint')
            output.with_suffix('.status.json').write_text('{"state":"complete"}')
        def wait(self):
            events.append(('wait', self.name))
            return 0
    monkeypatch.setattr(module.subprocess, 'Popen', Process)
    monkeypatch.setattr(sys, 'argv', ['train_main15', '--models', *ORDER[:6]])
    module.main()
    assert events == [event for name in ORDER[:6] for event in [('start', name), ('wait', name)]]
    directory = tmp_path/'artifacts/runs/main15_seed42'
    identity = (directory/'identity.json').read_bytes()
    assert len(json.loads(identity)['runtime_options']) == 15
    assert not (directory/'runner.lock').exists()
    events.clear()
    monkeypatch.setattr(sys, 'argv', ['train_main15', '--resume', '--models', *ORDER[:6]])
    module.main()
    assert events == []
    monkeypatch.setattr(sys, 'argv', ['train_main15', '--resume', '--models', 'mp20_u',
                                    '--coordinate-length-scale', '1'])
    with pytest.raises(ValueError, match='runtime options'):
        module.main()
    assert events == [] and (directory/'identity.json').read_bytes() == identity
    assert not (directory/'runner.lock').exists()


@pytest.mark.parametrize('arguments', [['--threads', '0'], ['--edge-chunk-size', '0'],
                                      ['--coordinate-length-scale', 'nan'],
                                      ['--coordinate-length-scale', '0']])
def test_invalid_runtime_rejected_in_plan(arguments, monkeypatch):
    module = runner()
    monkeypatch.setattr(sys, 'argv', ['train_main15', '--plan', *arguments])
    with pytest.raises(SystemExit):
        module.main()
