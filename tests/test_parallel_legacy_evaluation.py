import importlib.util
import json
from pathlib import Path
import signal
from types import SimpleNamespace

import pytest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/parallel_legacy_evaluation.py'
spec = importlib.util.spec_from_file_location('parallel_legacy', SCRIPT)
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)


def job(tmp_path, name='M1_mp_20_conditional', stage='sample', count=2):
    folder = tmp_path/name
    folder.mkdir(exist_ok=True)
    command = ['--output', str(folder)]
    if stage == 'sample':
        command = ['-m', 'unifiedgroupgen.cli', '--device', 'cpu', '--threads', '1',
                   '--count', str(count), *command]
    if stage == 'properties':
        command += ['--reference-output', str(tmp_path/'shared_reference')]
    return {'name': name, 'stage': stage, 'command': command, 'marker': str(folder/'summary.json')}


def process(args, ppid=1, start='123'):
    return {'args': args, 'ppid': ppid, 'start': start, 'state': 'S', 'rss': 1000}


def finish_sample(j):
    folder = Path(j['marker']).parent
    count = int(m.option(j['command'], '--count'))
    # Keep failed attempts, rather than silently topping up successes.
    rows = [{'index': i, 'status': 'failed' if i % 2 else 'generated'} for i in range(count)]
    (folder/'samples.jsonl').write_text('\n'.join(json.dumps(r) for r in rows))
    for row in rows:
        if row['status'] == 'generated':
            (folder/f'sample_{row["index"]:05d}.cif').write_text('fixture CIF')
    m.save(folder/'sampling.status.json', {'state': 'complete'})
    m.save(folder/'summary.json', {'attempted': count})


def test_dependencies_and_reference_cache_mutex(tmp_path):
    sample = job(tmp_path)
    prop = job(tmp_path, sample['name']+'_properties', 'properties')
    other = job(tmp_path, 'M1_mp_20_unconditional_properties', 'properties')
    metric = job(tmp_path, sample['name']+'_metrics', 'metrics')
    jobs = [sample, prop, other, metric]
    results = {'M1_test': {'state': 'complete'}, sample['name']: {'state': 'complete'},
               'M1_mp_20_unconditional': {'state': 'complete'}}
    assert [j['name'] for j in m.ready_jobs(jobs, results, {})] == [prop['name'], other['name']]
    assert not m.ready_jobs(jobs, results, {prop['name']: {}})
    results[prop['name']] = {'state': 'complete'}
    assert metric in m.ready_jobs(jobs, results, {})
    results[sample['name']] = {'state': 'failed'}
    assert prop not in m.ready_jobs(jobs, results, {})


def rows(rates):
    return [{'epoch': i, 'train_records_per_second': rate} for i, rate in enumerate(rates)]


def test_training_guard_promotes_only_after_three_fresh_epochs_and_throttles():
    g = m.TrainingGuard()
    assert g.update('joint_u', rows([100]*3), 3)[0] == 2
    assert g.update('joint_u', rows([100]*3), 3)[0] == 2
    assert g.update('joint_u', rows([100]*5), 3)[0] == 2
    assert g.update('joint_u', rows([100]*6), 3)[0] == 3
    assert g.update('joint_u', rows([100]*6), 2)[0] == 2
    assert g.update('joint_u', rows([100]*6), 64)[0] == 3
    assert g.update('joint_u', rows([100]*6+[80, 80]), 3)[0] == 1
    assert g.update('mp20_all', rows([200]*8), 3)[0] == 2
    assert g.data['baseline'] == 200


def test_training_guard_mild_slowdown_and_missing_baseline():
    g = m.TrainingGuard()
    assert g.update('joint_u', rows([100]*2), 3)[0] == 2
    g.update('joint_u', rows([100]*3), 3)
    assert g.update('joint_u', rows([100]*3+[93]*3), 3)[0] == 2
    assert m.TrainingGuard(g.data).data == g.data


def test_resources_respect_cgroup_quota_memory_and_cpu_reserve(tmp_path, monkeypatch):
    (tmp_path/'cpu.max').write_text('800000 100000')
    (tmp_path/'memory.max').write_text(str(32*m.GIB))
    (tmp_path/'memory.current').write_text(str(16*m.GIB))
    original = Path.read_text
    def text(path, *args, **kwargs):
        if path.as_posix() == '/proc/meminfo':
            return 'MemTotal: 1073741824 kB\nMemAvailable: 536870912 kB\n'
        if '/topology/' in path.as_posix():
            raise OSError('No fixture topology')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'read_text', text)
    monkeypatch.setattr(m.os, 'sched_getaffinity', lambda pid: set(range(64)), raising=False)
    monkeypatch.setattr(m, 'cgroup_paths', lambda name: [tmp_path])
    r = m.resources()
    assert r['effective_cpus'] == 8 and r['capacity'] == 3
    assert r['memory_limit_gib'] == 32 and r['evaluation_budget_gib'] == 4
    for quota, capacity in [(7, 3), (6, 2), (5, 1), (4, 0)]:
        (tmp_path/'cpu.max').write_text(f'{quota*100000} 100000')
        result = m.resources()
        assert result['effective_cpus'] == quota and result['capacity'] == capacity
        assert result['training_cpu_reserve'] == 4


def test_seal_preserves_failures_and_ignores_future_child_outputs(tmp_path):
    j = job(tmp_path)
    finish_sample(j)
    result = m.seal(tmp_path, j, adopted=True)
    assert result['adopted'] and 'seconds' not in result
    m.verify_completed(tmp_path, j, result)
    (Path(j['marker']).parent/'properties').mkdir()
    (Path(j['marker']).parent/'properties'/'later.txt').write_text('later task')
    m.verify_completed(tmp_path, j, result)
    (Path(j['marker']).parent/'samples.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='Completed output changed'):
        m.verify_completed(tmp_path, j, result)


@pytest.mark.parametrize('problem', ['partial', 'duplicate_index', 'missing_cif', 'running'])
def test_seal_refuses_incomplete_samples(tmp_path, problem):
    j = job(tmp_path)
    finish_sample(j)
    folder = Path(j['marker']).parent
    if problem == 'partial':
        m.save(folder/'summary.json', {'attempted': 1})
    elif problem == 'duplicate_index':
        (folder/'samples.jsonl').write_text('{"index":0,"status":"failed"}\n'*2)
    elif problem == 'missing_cif':
        (folder/'sample_00000.cif').unlink()
    else:
        m.save(folder/'sampling.status.json', {'state': 'running'})
    with pytest.raises(ValueError):
        m.seal(tmp_path, j)


def test_property_summary_alone_is_not_completion(tmp_path):
    j = job(tmp_path, 'M1_mp_20_conditional_properties', 'properties')
    m.save(Path(j['marker']), {'state': 'complete'})
    with pytest.raises(ValueError, match='Incomplete property outputs'):
        m.seal(tmp_path, j)


def handoff_fixture(tmp_path, monkeypatch, duplicate=False, training_child=False):
    j = job(tmp_path)
    m.save(tmp_path/'pipeline.json', {'jobs': {j['name']: {'state': 'running'}}})
    table = {10: process(['python', '-u', '/frozen/campaign.py', '--output', str(tmp_path)]),
             11: process(['python', '-u', *j['command']], 10),
             12: process(['native_cpu_worker'], 11),
             99: process(['python', '-m', 'unifiedgroupgen.cli', 'train'])}
    if duplicate:
        table[13] = process(['python', '-u', *j['command']], 10)
    if training_child:
        table[12]['args'] = ['python', 'scripts/train_main15.py']
    monkeypatch.setattr(m, 'processes', lambda: dict(table))
    for name, value in [('SIGSTOP', 19), ('SIGCONT', 18), ('SIGKILL', 9)]:
        monkeypatch.setattr(signal, name, value, raising=False)
    called = []
    def kill(pid, sig):
        called.append((pid, sig))
        if sig in (signal.SIGTERM, signal.SIGKILL):
            table.pop(pid)
    monkeypatch.setattr(m.os, 'kill', kill)
    monkeypatch.setattr(m.os, 'sysconf', lambda name: 100, raising=False)
    original = Path.read_text
    monkeypatch.setattr(Path, 'read_text', lambda path, *a, **k:
                        '10000 0' if path.as_posix() == '/proc/uptime' else original(path, *a, **k))
    return j, table, called


def test_handoff_only_signals_controller_retains_sampling_descendants_and_training(tmp_path, monkeypatch):
    j, table, called = handoff_fixture(tmp_path, monkeypatch)
    _, active = m.handoff(tmp_path, [j], True)
    assert active[j['name']]['pid'] == 11 and active[j['name']]['start'] == '123'
    assert {pid for pid, sig in called} == {10}
    assert {11, 12, 99} <= set(table)
    assert list(tmp_path.glob('parallel_handoff_*.json'))


@pytest.mark.parametrize('problem', ['duplicates', 'training_descendant', 'missing_authorization'])
def test_handoff_refuses_ambiguous_workers_before_termination(tmp_path, monkeypatch, problem):
    j, table, called = handoff_fixture(tmp_path, monkeypatch, problem == 'duplicates', problem == 'training_descendant')
    with pytest.raises(ValueError):
        m.handoff(tmp_path, [j], problem != 'missing_authorization')
    assert not any(sig in (signal.SIGTERM, signal.SIGKILL) for pid, sig in called)
    assert 10 in table and 11 in table and 99 in table


def test_affinity_skips_reused_pid_and_never_changes_training(tmp_path, monkeypatch):
    monkeypatch.setattr(m, 'processes', lambda: {11: process(['train'], start='new')})
    monkeypatch.setattr(m.os, 'sched_setaffinity', lambda *a: pytest.fail('Changed unrelated affinity'), raising=False)
    assert m.set_eval_affinity({'sample': {'pid': 11, 'start': 'old'}}, [1, 2]) == 0
    with pytest.raises(ValueError, match='training affinity'):
        m.set_eval_affinity({'sample': {'pid': 11, 'start': 'new'}}, [1, 2])


def test_launcher_cpu_only_single_thread_and_unchanged_scientific_arguments(tmp_path, monkeypatch):
    j = job(tmp_path)
    captured = {}
    def popen(command, **kwargs):
        captured.update(command=command, **kwargs)
        return SimpleNamespace(pid=11)
    monkeypatch.setattr(m.subprocess, 'Popen', popen)
    monkeypatch.setattr(m, 'processes', lambda: {11: process(['fixture'])})
    m.launch(tmp_path, tmp_path, j, [5, 6])
    assert captured['command'][-len(j['command']):] == j['command']
    assert captured['command'][:9] == ['nice', '-n', '19', 'ionice', '-c', '3', 'taskset', '-c', '5,6']
    assert captured['env']['CUDA_VISIBLE_DEVICES'] == ''
    assert captured['env']['OMP_NUM_THREADS'] == captured['env']['MKL_NUM_THREADS'] == '1'
    assert captured['start_new_session']


def test_reference_cache_only_copies_completed_matching_dataset_config(tmp_path):
    first = job(tmp_path, 'M1_mp_20_unconditional_metrics', 'metrics')
    second = job(tmp_path, 'M1_mp_20_conditional_metrics', 'metrics')
    for j in (first, second):
        j['command'] += ['--dataset', 'mp_20', '--reference-config', 'same.yaml']
    source = Path(first['marker']).parent/'mp_20_reference.json'
    source.write_text('verified-cache')
    dest = Path(second['marker']).parent/source.name
    m.share_reference_cache(second, [first, second], {})
    assert not dest.exists()
    m.share_reference_cache(second, [first, second], {first['name']: {'state': 'complete'}})
    assert dest.read_text() == source.read_text()


def test_scheduler_adopts_completes_and_reuses_without_launching_training(tmp_path, monkeypatch):
    j = job(tmp_path)
    finish_sample(j)
    source = {'identity': {'jobs': [j]}, 'jobs': {j['name']: {'state': 'running'}}}
    m.save(tmp_path/'pipeline.json', source)
    limits = {'capacity': 3, 'evaluation_cpus': list(range(3)), 'evaluation_budget_gib': 16,
              'available_memory_gib': 64}
    monkeypatch.setattr(m, 'validate_manifest', lambda *a: [j])
    monkeypatch.setattr(m, 'handoff', lambda *a: (source, {j['name']: {
        'pid': 11, 'start': 'old', 'started_at': m.time.time()-10, 'adopted': True}}))
    monkeypatch.setattr(m, 'processes', lambda: {m.os.getpid(): process(['controller'])})
    monkeypatch.setattr(m, 'resources', lambda: limits)
    monkeypatch.setattr(m, 'set_eval_affinity', lambda *a: 0)
    monkeypatch.setattr(m, 'launch', lambda *a: pytest.fail('Duplicated adopted sampling'))
    m.run(tmp_path, tmp_path, limits, True, False)
    state = m.read(tmp_path/'parallel.status.json')
    assert state['state'] == 'complete' and state['jobs'][j['name']]['adopted']
    assert not (tmp_path/'parallel.lock').exists()
    assert (tmp_path/'pipeline.lock').exists()


def test_active_parallel_lock_prevents_second_controller(tmp_path, monkeypatch):
    (tmp_path/'parallel.lock').write_text('77')
    m.save(tmp_path/'parallel.status.json', {'process_start': 'valid'})
    monkeypatch.setattr(m, 'processes', lambda: {77: process(['controller'], start='valid')})
    with pytest.raises(ValueError, match='already active'):
        m.run(tmp_path, tmp_path, {}, True, True)


def manifest_fixture(tmp_path, monkeypatch):
    import yaml
    tools = SCRIPT.parent/'legacy_eval'
    monkeypatch.syspath_prepend(str(tools))
    campaign_spec = importlib.util.spec_from_file_location('campaign_fixture', tools/'campaign.py')
    campaign = importlib.util.module_from_spec(campaign_spec)
    campaign_spec.loader.exec_module(campaign)
    (tmp_path/'configs').mkdir()
    for _, _, config_name, _, _ in campaign.MODELS:
        prepared = tmp_path/'data'/config_name
        prepared.mkdir(parents=True)
        for split in ('train', 'val', 'test'):
            (prepared/(split+'.jsonl')).write_text('{}\n')
        (tmp_path/'configs'/(config_name+'.yaml')).write_text(yaml.safe_dump({'prepared_root': str(prepared)}))
    study = tmp_path/'artifacts/evaluations/legacy_1024_cpu_seed142'
    models, jobs = campaign.plan(tmp_path, study, tmp_path/'cgdit', tmp_path/'predictors', count=1024)
    model_identity = {}
    for model in models:
        checkpoint = Path(model['checkpoint'])
        checkpoint.parent.mkdir(parents=True)
        checkpoint.write_bytes(b'old-checkpoint')
        model_identity[model['name']] = {'checkpoint_sha256': m.digest(checkpoint),
            'test_sha256': [m.digest(p) for p in model['tests']],
            **{split+'_sha256': [m.digest(p) for p in paths] for split, paths in model['training_inputs'].items()}}
    identity = {'jobs': jobs, 'models': model_identity, 'predictors': {}, 'generator_sha256': {},
                'tools_sha256': {p.name: m.digest(p) for p in tools.glob('*.py')},
                'predictor_code_sha256': {}, 'metric_sha256': m.digest(SCRIPT.parent/'evaluate_generation.py')}
    return study, {'identity': identity, 'jobs': {}}


def test_actual_33_job_manifest_and_reference_hashes(tmp_path, monkeypatch):
    study, state = manifest_fixture(tmp_path, monkeypatch)
    before = json.dumps(state, sort_keys=True)
    assert len(m.validate_manifest(tmp_path, study, state)) == 33
    assert json.dumps(state, sort_keys=True) == before
    (tmp_path/'data/mp20_local/train.jsonl').write_text('changed')
    with pytest.raises(ValueError, match='Frozen input/code changed'):
        m.validate_manifest(tmp_path, study, state)


@pytest.mark.parametrize('problem', ['gpu', 'wrong_count', 'training', 'outside_output'])
def test_manifest_rejects_wrong_budget_gpu_or_training(tmp_path, monkeypatch, problem):
    study, state = manifest_fixture(tmp_path, monkeypatch)
    j = next(j for j in state['identity']['jobs'] if j['stage'] == 'sample')
    if problem == 'gpu':
        j['command'][j['command'].index('--device')+1] = 'cuda'
    elif problem == 'wrong_count':
        j['command'][j['command'].index('--count')+1] = '4096'
    elif problem == 'training':
        j['command'].append('train')
    else:
        j['command'][j['command'].index('--output')+1] = str(tmp_path/'artifacts/runs/main15_seed42')
    with pytest.raises(ValueError):
        m.validate_manifest(tmp_path, study, state)


def test_full_dag_reuses_tests_adopts_sample_and_observes_two_before_three(tmp_path, monkeypatch):
    study, source = manifest_fixture(tmp_path, monkeypatch)
    study.mkdir(parents=True)
    jobs = source['identity']['jobs']
    for j in jobs:
        if j['stage'] == 'test':
            source['jobs'][j['name']] = {'state': 'complete', 'reused_from': 'previous'}
    adopted = next(j['name'] for j in jobs if j['stage'] == 'sample')
    source['jobs'][adopted] = {'state': 'running'}
    m.save(study/'pipeline.json', source)
    trainer = tmp_path/'artifacts/runs/main15_seed42'
    trainer.mkdir(parents=True)
    m.save(trainer/'runner.json', {'state': 'running', 'pid': 90,
                                 'jobs': [{'name': 'joint_u', 'state': 'running', 'pid': 99}]})
    (trainer/'joint_u').mkdir()
    m.save(trainer/'joint_u/model.status.json', {'state': 'running'})
    table = {m.os.getpid(): process(['parallel']), 99: process(['python', 'train'])}
    calls = {'history': 0}
    launched, verified, snapshots = [], [], []
    limits = {'capacity': 3, 'evaluation_cpus': list(range(3)), 'evaluation_budget_gib': 16,
              'available_memory_gib': 64}
    monkeypatch.setattr(m, 'validate_manifest', lambda *a: jobs)
    monkeypatch.setattr(m, 'verify_completed', lambda study, j, r: verified.append(j['name']))
    monkeypatch.setattr(m, 'handoff', lambda *a: (source, {adopted: {
        'pid': 11, 'start': 'old', 'started_at': m.time.time()-10, 'adopted': True}}))
    monkeypatch.setattr(m, 'processes', lambda: dict(table))
    monkeypatch.setattr(m, 'resources', lambda: limits)
    monkeypatch.setattr(m, 'set_eval_affinity', lambda *a: 0)
    monkeypatch.setattr(m, 'share_reference_cache', lambda *a: None)
    monkeypatch.setattr(m.time, 'sleep', lambda seconds: None)
    def hist(path):
        calls['history'] += 1
        return rows([100]*(calls['history']+2))
    monkeypatch.setattr(m, 'history', hist)
    monkeypatch.setattr(m, 'seal', lambda study, j, *args, **kwargs: {'state': 'complete'})
    def launch(root, study, j, cpus):
        # Dependencies must have been persisted complete before dispatch.
        launched.append(j['name'])
        return SimpleNamespace(poll=lambda: 0), {'pid': 200+len(launched), 'start': 'new',
            'started_at': m.time.time(), 'adopted': False}
    monkeypatch.setattr(m, 'launch', launch)
    save = m.save
    def record(path, state):
        if path.name == 'parallel.status.json':
            snapshots.append((state.get('worker_limit'), len(state.get('active', {}))))
        save(path, state)
    monkeypatch.setattr(m, 'save', record)
    m.run(tmp_path, study, limits, True, False)
    result = m.read(study/'parallel.status.json')
    assert result['state'] == 'complete' and len(result['jobs']) == 33
    assert len(verified) == 3 and len(launched) == 29
    assert adopted not in launched and len(set(launched)) == len(launched)
    assert all(n <= slots for slots, n in snapshots if slots)
    assert 2 in [s for s, _ in snapshots] and 3 in [s for s, _ in snapshots]
    assert max(n for _, n in snapshots) <= 3
    assert result['parallel_protocol'] == 'cpu_dependency_v2_2to3'


@pytest.mark.parametrize('capacity, allowed', [(3, True), (2, True), (1, False)])
def test_main_allows_two_to_three_workers_without_reducing_training_reserve(tmp_path, monkeypatch, capacity, allowed):
    limits = {'capacity': capacity, 'effective_cpus': capacity+4, 'training_cpu_reserve': 4,
              'tools': {'nice': '/usr/bin/nice', 'ionice': '/usr/bin/ionice', 'taskset': '/usr/bin/taskset'}}
    called = []
    monkeypatch.setattr(m.sys, 'platform', 'linux')
    monkeypatch.setattr(m.sys, 'argv', ['parallel', '--project-root', str(tmp_path), '--run', '--takeover-serial'])
    monkeypatch.setattr(m, 'resources', lambda: limits)
    monkeypatch.setattr(m, 'run', lambda *args: called.append(args))
    if allowed:
        m.main()
        assert len(called) == 1 and called[0][2]['training_cpu_reserve'] == 4
        assert called[0][3:] == (True, False)
    else:
        with pytest.raises(SystemExit) as error:
            m.main()
        assert error.value.code == 2 and not called


def test_status_reports_actual_worker_and_serial_fallback_read_only(tmp_path, monkeypatch, capsys):
    j = job(tmp_path)
    m.save(tmp_path/'pipeline.json', {'pid': 10, 'state': 'running',
        'identity': {'jobs': [j]}, 'jobs': {j['name']: {'state': 'running'}}})
    table = {10: process(['python', 'campaign.py']), 11: process(['python', '-u', *j['command']], 10)}
    monkeypatch.setattr(m, 'processes', lambda: table)
    monkeypatch.setattr(m.os, 'kill', lambda *args: pytest.fail('Read-only status sent a signal'))
    monkeypatch.setattr(m, 'launch', lambda *args: pytest.fail('Read-only status started work'))
    m.show_status(tmp_path, tmp_path)
    output = capsys.readouterr().out
    assert 'serial; no parallel takeover recorded' in output
    assert 'Actual live evaluation tasks: 1' in output
    assert not (tmp_path/'parallel.status.json').exists()
