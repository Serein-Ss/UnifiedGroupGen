"""CPU-only dependency scheduler for an existing legacy campaign; never starts training."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import time

GIB = 2**30


def read(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return default


def save(path, value):
    temporary = path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def tail(path, limit=262144):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, 2)
            stream.seek(max(0, stream.tell()-limit))
            return stream.read().decode('utf-8', errors='replace').splitlines()
    except OSError:
        return []


def history(path):
    rows = {}
    for line in tail(path):
        try:
            row = json.loads(line)
            if row.get('train_records_per_second', 0) > 0:
                rows[row['epoch']] = row
        except (ValueError, KeyError):
            pass
    return sorted(rows.values(), key=lambda row: row['epoch'])


def option(command, name):
    return command[command.index(name)+1]


def processes():
    result = {}
    for path in Path('/proc').glob('[0-9]*'):
        try:
            fields = (path/'stat').read_text().rsplit(')', 1)[1].split()
            result[int(path.name)] = {
                'start': fields[19], 'state': fields[0], 'ppid': int(fields[1]),
                'rss': int(fields[21])*os.sysconf('SC_PAGE_SIZE'),
                'args': [s.decode(errors='replace') for s in (path/'cmdline').read_bytes().split(b'\0') if s]}
        except (OSError, ValueError, IndexError):
            pass
    return result


def live(process, start=None):
    return bool(process and process['state'] != 'Z' and (start is None or process['start'] == start))


def descendants(table, roots):
    found = set(roots) & set(table)
    while True:
        extra = {pid for pid, p in table.items() if p['ppid'] in found}-found
        if not extra:
            return found
        found |= extra


def training_process(process):
    return 'train' in process['args'] or any(Path(s).name.startswith('train_') for s in process['args'])


def cgroup_paths(controller):
    """Read visible v1/v2 ancestors; namespaces may expose only the mount root."""
    entries = []
    try:
        for line in Path('/proc/self/cgroup').read_text().splitlines():
            _, names, relative = line.split(':', 2)
            if not names or controller in names.split(','):
                entries.append(relative.lstrip('/'))
    except OSError:
        pass
    bases = [Path('/sys/fs/cgroup'), Path('/sys/fs/cgroup')/controller]
    if controller == 'cpu':
        bases.append(Path('/sys/fs/cgroup/cpu,cpuacct'))
    found = set()
    for base in bases:
        if not base.exists():
            continue
        found.add(base)
        for relative in entries:
            child = base/relative
            if not child.is_dir() or not child.resolve().is_relative_to(base.resolve()):
                continue
            while child != base:
                found.add(child)
                child = child.parent
    return found


def resources():
    allowed = sorted(os.sched_getaffinity(0))
    quotas, limits, memory_free = [], [], []
    mem = {s.split(':')[0]: int(s.split()[1])*1024 for s in Path('/proc/meminfo').read_text().splitlines()}
    for folder in cgroup_paths('cpu'):
        try:
            if (folder/'cpu.max').exists():
                quota, period = (folder/'cpu.max').read_text().split()
                if quota != 'max':
                    quotas.append(int(quota)/int(period))
            elif (folder/'cpu.cfs_quota_us').exists():
                quota = int((folder/'cpu.cfs_quota_us').read_text())
                if quota > 0:
                    quotas.append(quota/int((folder/'cpu.cfs_period_us').read_text()))
        except (OSError, ValueError, ZeroDivisionError):
            continue
    for folder in cgroup_paths('memory'):
        for maximum, current in [('memory.max', 'memory.current'), ('memory.limit_in_bytes', 'memory.usage_in_bytes')]:
            try:
                raw = (folder/maximum).read_text().strip()
                if raw != 'max':
                    limit = int(raw)
                    used = int((folder/current).read_text())
                    limits.append(limit)
                    memory_free.append(max(0, limit-used))
            except (OSError, ValueError):
                continue
    effective = min([len(allowed), *quotas])
    available = min([mem['MemAvailable'], *memory_free])
    budget = min(16*GIB, available/4)
    capacity = max(0, min(8, math.floor(effective)-4, math.floor(budget/GIB)))
    # Prefer distinct physical cores; do not change any training affinity.
    chosen, cores = [], set()
    for cpu in reversed(allowed):
        base = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
        try:
            core = ((base/'physical_package_id').read_text().strip(), (base/'core_id').read_text().strip())
        except OSError:
            core = (str(cpu), str(cpu))
        if core not in cores:
            chosen.append(cpu)
            cores.add(core)
    chosen += [cpu for cpu in reversed(allowed) if cpu not in chosen]
    return {'allowed_cpus': allowed, 'cpu_quotas': quotas, 'effective_cpus': effective,
            'memory_limit_gib': min([mem['MemTotal'], *limits])/GIB,
            'available_memory_gib': available/GIB, 'evaluation_budget_gib': budget/GIB,
            'capacity': capacity, 'evaluation_cpus': chosen[:capacity], 'training_cpu_reserve': 4,
            'tools': {name: shutil.which(name) for name in ('nice', 'ionice', 'taskset')}}


def dependency(job):
    if job['stage'] == 'test':
        return None
    if job['stage'] == 'sample':
        return job['name'].split('_', 1)[0]+'_test'
    suffix = '_properties' if job['stage'] == 'properties' else '_metrics'
    base = job['name'][:-len(suffix)]
    return base if job['stage'] == 'properties' else base+'_properties'


def cache_key(job):
    return option(job['command'], '--reference-output') if job['stage'] == 'properties' else None


def ready_jobs(jobs, results, active):
    busy = {cache_key(job) for job in jobs if job['name'] in active and cache_key(job)}
    result = []
    for job in jobs:
        name, parent = job['name'], dependency(job)
        if name in active or results.get(name, {}).get('state') in ('complete', 'failed'):
            continue
        if parent and results.get(parent, {}).get('state') != 'complete':
            continue
        if cache_key(job) in busy:
            continue
        result.append(job)
    # Pipeline available postprocessing instead of waiting for all sampling.
    return sorted(result, key=lambda job: {'properties': 0, 'metrics': 1, 'test': 2, 'sample': 3}[job['stage']])


def validate_manifest(root, study, state):
    identity = state['identity']
    jobs = identity['jobs']
    if Counter(j['stage'] for j in jobs) != {'test': 3, 'sample': 10, 'properties': 10, 'metrics': 10}:
        raise ValueError('Expected the existing 33-job legacy campaign')
    if len({j['name'] for j in jobs}) != 33:
        raise ValueError('Duplicate job names')
    for job in jobs:
        command = job['command']
        if 'train' in command or any(Path(s).name.startswith('train_') for s in command):
            raise ValueError('Refusing a training command')
        folder = Path(option(command, '--output')).resolve()
        if not folder.is_relative_to(study) or Path(job['marker']).resolve() != folder/'summary.json':
            raise ValueError('Unexpected output path: '+job['name'])
        if job['stage'] == 'sample':
            if option(command, '--device') != 'cpu' or option(command, '--threads') != '1':
                raise ValueError('Sampling must remain CPU / one thread')
            if option(command, '--count') != '1024':
                raise ValueError('This takeover requires 1024 attempts per group')
        if job['stage'] == 'metrics' and option(command, '--reference-workers') != '1':
            raise ValueError('Nested worker pools are not enabled by this scheduler')
    # Keep all frozen scientific code and weights bound to their original hashes.
    tools = Path(next(j for j in jobs if j['stage'] == 'test')['command'][0]).parent
    cgdit = Path(option(next(j for j in jobs if j['stage'] == 'properties')['command'], '--cgdit-root'))
    checks = [(root/name, sha) for name, sha in identity['generator_sha256'].items()]
    checks += [(tools/name, sha) for name, sha in identity['tools_sha256'].items()]
    metric = Path(next(j for j in jobs if j['stage'] == 'metrics')['command'][0])
    checks.append((metric, identity['metric_sha256']))
    checks += [(cgdit/name, sha) for name, sha in identity['predictor_code_sha256'].items()]
    for info in identity['predictors'].values():
        checks.append((Path(info['checkpoint']), info['sha256']))
    for model, info in identity['models'].items():
        command = next(j['command'] for j in jobs if j['stage'] == 'sample' and j['name'].startswith(model+'_'))
        checkpoint = Path(option(command, '--checkpoint'))
        if 'main15' in str(checkpoint):
            raise ValueError('Refusing a formal-training checkpoint')
        checks.append((checkpoint, info['checkpoint_sha256']))
        import yaml
        config_path = Path(option(next(j['command'] for j in jobs if j['stage'] == 'metrics'
                                      and j['name'].startswith(model+'_')), '--reference-config'))
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        for split in ('train', 'val', 'test'):
            paths = config.get(split+'_files', [config['prepared_root']+'/'+split+'.jsonl'])
            if len(paths) != len(info[split+'_sha256']):
                raise ValueError('Reference split changed: '+model+'/'+split)
            checks += [(root/path, sha) for path, sha in zip(paths, info[split+'_sha256'])]
    for path, sha in checks:
        if digest(path) != sha:
            raise ValueError('Frozen input/code changed: '+str(path))
    return jobs


def verify_completed(study, job, result):
    marker = Path(job['marker'])
    if digest(marker) != result['summary_sha256']:
        raise ValueError('Completed summary changed: '+job['name'])
    if not result.get('output_sha256'):
        raise ValueError('Missing output hashes: '+job['name'])
    for relative, sha in result['output_sha256'].items():
        path = (study/relative).resolve()
        if not path.is_relative_to(study) or digest(path) != sha:
            raise ValueError('Completed output changed: '+relative)


def seal(study, job, seconds=None, adopted=False):
    marker = Path(job['marker'])
    report = read(marker, {})
    folder = marker.parent
    if job['stage'] == 'sample':
        count = int(option(job['command'], '--count'))
        rows = [json.loads(s) for s in (folder/'samples.jsonl').read_text(encoding='utf-8').splitlines() if s.strip()]
        if report.get('attempted') != count or [r['index'] for r in rows] != list(range(count)):
            raise ValueError('Incomplete attempts: '+job['name'])
        if read(folder/'sampling.status.json', {}).get('state') != 'complete':
            raise ValueError('Sampling did not finish: '+job['name'])
        for row in rows:
            if row['status'] == 'generated' and not (folder/f'sample_{row["index"]:05d}.cif').is_file():
                raise ValueError('Missing generated CIF')
        files = [p for p in folder.iterdir() if p.is_file()]
    else:
        if report.get('state') != 'complete':
            raise ValueError('Incomplete report: '+job['name'])
        if job['stage'] == 'properties':
            for name in ('assessments.csv', 'reference_assessments.csv', 'assessment_metadata.json', 'property_distributions.png'):
                if not (folder/name).is_file():
                    raise ValueError('Incomplete property outputs: '+name)
        files = [p for p in folder.rglob('*') if p.is_file()]
    result = {'state': 'complete', 'summary_sha256': digest(marker),
              'output_sha256': {str(p.relative_to(study)): digest(p) for p in files}, 'adopted': adopted}
    if seconds is not None:
        result['seconds'] = seconds
    return result


def matching_worker(process, job):
    return live(process) and process['args'][1:] == ['-u', *job['command']]


def serial_process(process, study):
    return (live(process) and any(Path(s).name == 'campaign.py' for s in process['args'])
            and '--output' in process['args'] and Path(option(process['args'], '--output')).resolve() == study)


def handoff(study, jobs, take_over):
    """Stop the verified serial controller ONLY; adopt live worker interpreters."""
    source = read(study/'pipeline.json', {})
    table = processes()
    controllers = {pid: p for pid, p in table.items() if serial_process(p, study)}
    if controllers and not take_over:
        raise ValueError('Serial controller active; use --takeover-serial')
    stopped, frozen = [], []
    try:
        for pid, p in controllers.items():
            if live(processes().get(pid), p['start']):
                try:
                    os.kill(pid, signal.SIGSTOP)
                    frozen.append((pid, p['start']))
                except ProcessLookupError:
                    pass
        source = read(study/'pipeline.json', {})
        table = processes()
        adopted = {}
        for job in jobs:
            matches = [(pid, p) for pid, p in table.items() if matching_worker(p, job)]
            if len(matches) > 1:
                raise ValueError('Duplicate workers: '+job['name'])
            if matches:
                pid, p = matches[0]
                if source.get('jobs', {}).get(job['name'], {}).get('state') == 'complete':
                    raise ValueError('Live worker for completed job')
                for child in descendants(table, [pid]):
                    if training_process(table[child]):
                        raise ValueError('Training process in evaluation lineage')
                age = float(Path('/proc/uptime').read_text().split()[0])-int(p['start'])/os.sysconf('SC_CLK_TCK')
                adopted[job['name']] = {'pid': pid, 'start': p['start'], 'started_at': time.time()-age, 'adopted': True}
        children = descendants(table, controllers)-set(controllers)
        registered = descendants(table, [a['pid'] for a in adopted.values()])
        if any(live(table[p]) for p in children-registered):
            raise ValueError('Unrecognized serial child; no controller terminated')
        for pid, p in controllers.items():
            if live(processes().get(pid), p['start']):
                try:
                    os.kill(pid, signal.SIGTERM)
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass
                stopped.append(pid)
        deadline = time.monotonic()+5
        while any(live(processes().get(pid), start) for pid, start in frozen) and time.monotonic() < deadline:
            time.sleep(.1)
        for pid, start in frozen:
            if live(processes().get(pid), start):
                try:
                    os.kill(pid, signal.SIGKILL)  # controller only, never the worker tree
                except ProcessLookupError:
                    pass
        deadline = time.monotonic()+2
        while any(live(processes().get(pid), start) for pid, start in frozen) and time.monotonic() < deadline:
            time.sleep(.1)
        if any(live(processes().get(pid), start) for pid, start in frozen):
            raise ValueError('Serial controller did not exit')
        save(study/f'parallel_handoff_{time.time_ns()}.json', {
            'previous_pipeline': source, 'stopped_controllers_only': stopped,
            'adopted_workers': adopted, 'time': time.time()})
        return source, adopted
    finally:
        for pid, start in frozen:
            if live(processes().get(pid), start):
                try:
                    os.kill(pid, signal.SIGCONT)
                except ProcessLookupError:
                    pass


class TrainingGuard:
    def __init__(self, saved=None):
        self.data = saved or {'model': None, 'baseline': None, 'last_epoch': -1, 'new_rates': []}

    def update(self, model, rows, capacity):
        if model != self.data['model']:
            self.data = {'model': model, 'baseline': None, 'last_epoch': -1, 'new_rates': []}
        d = self.data
        if d['baseline'] is None:
            if len(rows) < 3:
                return min(4, capacity), 'Waiting for 3 baseline training epochs'
            d.update(baseline=statistics.median(r['train_records_per_second'] for r in rows[-5:]),
                     last_epoch=rows[-1]['epoch'], new_rates=[])
            return min(4, capacity), 'Baseline recorded; observe 3 new epochs at four workers'
        fresh = [r for r in rows if r['epoch'] > d['last_epoch']]
        if fresh:
            d['new_rates'] = (d['new_rates']+[r['train_records_per_second'] for r in fresh])[-5:]
            d['last_epoch'] = fresh[-1]['epoch']
        rates = d['new_rates']
        if len(rates) >= 2 and statistics.median(rates[-2:]) < d['baseline']*.90:
            return min(2, capacity), 'Training throughput below 90% baseline; hold new work'
        if len(rates) >= 3:
            if statistics.median(rates[-3:]) < d['baseline']*.95:
                return min(4, capacity), 'Training throughput below 95% baseline; restrict to four'
            return min(8, capacity), '3 new epochs stable; eight workers permitted within resource cap'
        return min(4, capacity), 'Observing training before promotion'


def set_eval_affinity(active, cpus):
    table = processes()
    roots = [a['pid'] for a in active.values() if a['start'] is not None and live(table.get(a['pid']), a['start'])]
    for pid in descendants(table, roots):
        if training_process(table[pid]):
            raise ValueError('Refusing to set training affinity')
        for thread in Path(f'/proc/{pid}/task').glob('[0-9]*'):
            try:
                os.sched_setaffinity(int(thread.name), cpus)
            except ProcessLookupError:
                pass
    return sum(table[pid]['rss'] for pid in descendants(table, roots))


def share_reference_cache(job, jobs, results):
    if job['stage'] != 'metrics':
        return
    name = option(job['command'], '--dataset')+'_reference.json'
    dest = Path(job['marker']).parent/name
    if dest.exists():
        return
    for prior in jobs:
        if prior['stage'] != 'metrics' or results.get(prior['name'], {}).get('state') != 'complete':
            continue
        if all(option(prior['command'], flag) == option(job['command'], flag)
               for flag in ('--dataset', '--reference-config')):
            source = Path(prior['marker']).parent/name
            if source.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                temporary = dest.with_suffix('.copy.tmp')
                shutil.copyfile(source, temporary)
                temporary.replace(dest)  # worker independently checks cache identity
                return


def launch(root, study, job, cpus):
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'MKL_THREADING_LAYER': 'SEQUENTIAL',
           'PYTHONPATH': str(root/'src'), 'UGG_PROJECT_ROOT': str(root), 'PYTHONIOENCODING': 'utf-8'}
    command = ['nice', '-n', '19', 'ionice', '-c', '3', 'taskset', '-c', ','.join(map(str, cpus)),
               sys.executable, '-u', *job['command']]
    with (study/(job['name']+'.log')).open('a', encoding='utf-8') as stream:
        child = subprocess.Popen(command, cwd=root, env=env, stdout=stream,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    p = processes().get(child.pid)
    return child, {'pid': child.pid, 'start': p['start'] if p else None,
                   'started_at': time.time(), 'adopted': False}


def run(root, study, capacity, take_over, resume):
    lock = study/'parallel.lock'
    previous = read(study/'parallel.status.json', {})
    if lock.exists():
        owner = int(lock.read_text())
        if live(processes().get(owner), previous.get('process_start')):
            raise ValueError('Parallel controller already active: '+str(owner))
        if not resume:
            raise ValueError('Stale parallel lock; use --resume after inspecting status')
        lock.rename(study/f'parallel.lock.previous.{time.time_ns()}')
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(fd, str(os.getpid()).encode()); os.close(fd)
    children, active, status = {}, {}, None
    try:
        source = read(study/'pipeline.json', {})
        jobs = validate_manifest(root, study, source)
        verified_results = {}
        for job in jobs:
            prior = source.get('jobs', {}).get(job['name'], {})
            if prior.get('state') == 'complete':
                verify_completed(study, job, prior)
                verified_results[job['name']] = prior
        identity = source['identity']
        source, active = handoff(study, jobs, take_over)
        if source['identity'] != identity:
            raise ValueError('Campaign identity changed during takeover')
        for job in jobs:
            prior = source.get('jobs', {}).get(job['name'], {})
            if prior.get('state') == 'complete' and prior != verified_results.get(job['name']):
                verify_completed(study, job, prior)
        # Do not allow the serial scheduler to restart while the new owner runs.
        serial_lock = study/'pipeline.lock'
        if not serial_lock.exists():
            serial_lock.write_text(str(os.getpid()))
        status = {**source, 'state': 'running', 'pid': os.getpid(),
                  'process_start': processes()[os.getpid()]['start'],
                  'parallel_tool_sha256': digest(Path(__file__)), 'parallel_protocol': 'cpu_dependency_v1',
                  'active': active, 'initial_resource_check': capacity, 'updated_at': time.time()}
        status.pop('child_pid', None)
        guard = TrainingGuard(previous.get('guard'))
        indexed = {j['name']: j for j in jobs}
        for name, result in list(status['jobs'].items()):
            if result.get('state') == 'running' and name not in active:
                status['jobs'][name] = seal(study, indexed[name], adopted=True)
        def persist():
            status.update(active=active, guard=guard.data, updated_at=time.time())
            status['current'] = ','.join(active)
            save(study/'parallel.status.json', status)
            save(study/'pipeline.json', status)
        persist()
        while True:
            table = processes()
            for name, info in list(active.items()):
                code = children[name].poll() if name in children else None
                ended = code is not None if name in children else not live(table.get(info['pid']), info['start'])
                if not ended:
                    continue
                try:
                    if code not in (None, 0):
                        raise RuntimeError(f'Worker exited {code}')
                    status['jobs'][name] = seal(study, indexed[name], time.time()-info['started_at'], info['adopted'])
                    print('[DONE]', name, flush=True)
                except (OSError, ValueError, KeyError, RuntimeError) as error:
                    status['jobs'][name] = {'state': 'failed', 'error': str(error)}
                    print('[FAILED]', name, str(error), flush=True)
                del active[name]
            limits = resources()
            cap = min(capacity['capacity'], limits['capacity'])
            trainer = read(root/'artifacts/runs/main15_seed42/runner.json', {})
            current = next((j for j in trainer.get('jobs', []) if j.get('state') == 'running'), None)
            if trainer.get('state') == 'complete':
                slots, reason = cap, 'Formal training complete; resource cap applies'
            elif current:
                tp = processes().get(current.get('pid'))
                if not live(tp) or not training_process(tp):
                    slots, reason = 2, 'Training PID unavailable; hold new work'
                else:
                    rows = history(root/f'artifacts/runs/main15_seed42/{current["name"]}/model.history.jsonl')
                    slots, reason = guard.update(current['name'], rows, cap)
                    progress = root/f'artifacts/runs/main15_seed42/{current["name"]}/model.status.json'
                    maximum_age = max(600, 3*statistics.median(r.get('seconds', 600) for r in rows[-5:])) if rows else 1800
                    if not progress.exists() or time.time()-progress.stat().st_mtime > maximum_age:
                        slots, reason = min(2, cap), 'Training progress stale; hold new work'
            else:
                slots, reason = min(4, cap), 'Training transition or unknown status; hold new work'
            cpus = capacity['evaluation_cpus'][:max(1, slots)]
            rss = set_eval_affinity(active, cpus)
            hold = slots < 4 or current is None and trainer.get('state') != 'complete'
            budget = capacity['evaluation_budget_gib']*GIB
            if limits['available_memory_gib'] < 4 or rss > budget:
                hold, reason = True, 'Memory guard: no new jobs'
            status.update(worker_limit=slots, evaluation_cpus=cpus, evaluation_rss_gib=rss/GIB,
                          guard_reason=reason, dispatch_held=hold, training_current=current['name'] if current else None)
            while not hold and len(active) < slots:
                ready = ready_jobs(jobs, status['jobs'], active)
                if not ready:
                    break
                job = ready[0]
                if job['stage'] == 'sample' and (Path(job['marker']).parent/'samples.jsonl').exists():
                    raise ValueError('Partial sampling has no live owner; refusing overwrite: '+job['name'])
                required = GIB if job['stage'] == 'sample' else 2*GIB
                if rss+required > budget:
                    break
                share_reference_cache(job, jobs, status['jobs'])
                child, info = launch(root, study, job, cpus)
                active[job['name']], children[job['name']] = info, child
                status['jobs'][job['name']] = {'state': 'running', **info}
                rss += required
                print('[START]', job['name'], '| workers', len(active), '/', slots, flush=True)
            persist()
            complete = sum(r.get('state') == 'complete' for r in status['jobs'].values())
            if complete == len(jobs):
                status['state'] = 'complete'; persist(); break
            if not active and any(r.get('state') == 'failed' for r in status['jobs'].values()):
                status['state'] = 'failed'; persist(); break
            time.sleep(10)
    except BaseException as error:
        if status is not None:
            status.update(state='controller_error', error=str(error), active=active, updated_at=time.time())
            save(study/'parallel.status.json', status)
            save(study/'pipeline.json', status)
        raise
    finally:
        # Workers are intentionally left alive on controller failure for --resume adoption.
        if lock.exists() and lock.read_text() == str(os.getpid()):
            lock.unlink()


def show_status(root, study):
    state = read(study/'parallel.status.json', read(study/'pipeline.json', {}))
    print(time.strftime('%F %T'), '| state:', state.get('state'), '| updated seconds ago:', round(time.time()-state.get('updated_at', time.time()), 1))
    table = processes()
    print('Controller PID:', state.get('pid'), '| alive:', live(table.get(state.get('pid')), state.get('process_start')))
    print('Worker limit:', state.get('worker_limit'), '| CPUs:', state.get('evaluation_cpus'), '| RSS GiB:', state.get('evaluation_rss_gib'))
    print('Guard:', state.get('guard_reason'), '| new dispatch held:', state.get('dispatch_held'))
    print('Training throughput guard:', state.get('guard'))
    jobs, done = state.get('identity', {}).get('jobs', []), state.get('jobs', {})
    print('Complete:', sum(r.get('state') == 'complete' for r in done.values()), '/', len(jobs))
    print('Previously reused:', sum(r.get('state') == 'complete' and bool(r.get('reused_from')) for r in done.values()),
          '| adopted tasks:', sum(bool(r.get('adopted')) for r in done.values()))
    for stage in ('test', 'sample', 'properties', 'metrics'):
        selected = [j for j in jobs if j['stage'] == stage]
        print(stage, dict(Counter(done.get(j['name'], {}).get('state', 'pending') for j in selected)))
    for job in jobs:
        r = done.get(job['name'], {})
        if r.get('state') not in ('running', 'failed'):
            continue
        print('\n', job['name'], r)
        if job['stage'] == 'sample':
            sample = read(Path(job['marker']).parent/'sampling.status.json', {})
            n, total, seconds = sample.get('attempted', 0), sample.get('requested', 1024), sample.get('seconds')
            print('Attempts:', n, '/', total, '| remaining minutes:', round(seconds/n*(total-n)/60, 1) if n and seconds else 'unknown')
        print('\n'.join(tail(study/(job['name']+'.log'))[-2:]))
    trainer = read(root/'artifacts/runs/main15_seed42/runner.json', {})
    print('\nFormal training:', trainer.get('state'), '| PID:', trainer.get('pid'), '| alive:', live(table.get(trainer.get('pid'))))
    for job in trainer.get('jobs', []):
        if job.get('state') == 'running':
            print(job['name'], '| PID:', job.get('pid'), '| alive:', live(table.get(job.get('pid'))))
            print(read(root/f'artifacts/runs/main15_seed42/{job["name"]}/model.status.json', {}))
    print('ETA for all jobs requires measured property/metric speeds; concurrent task ETAs must not be summed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--study', default='artifacts/evaluations/legacy_1024_cpu_seed142')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--resources', action='store_true')
    mode.add_argument('--status', action='store_true')
    mode.add_argument('--plan', action='store_true')
    mode.add_argument('--run', action='store_true')
    parser.add_argument('--takeover-serial', action='store_true', help='Terminate only verified serial controller, retain its workers')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if sys.platform != 'linux':
        parser.error('Linux /proc and affinity are required; unit tests cover portable logic')
    root = args.project_root.resolve()
    study = (root/args.study).resolve()
    if not study.is_relative_to(root/'artifacts/evaluations'):
        parser.error('Keep legacy evaluation under artifacts/evaluations')
    if args.status:
        show_status(root, study); return
    capacity = resources()
    print(json.dumps(capacity, indent=2), flush=True)
    if args.resources:
        return
    if not args.run:
        source = read(study/'pipeline.json', {})
        jobs = validate_manifest(root, study, source)
        print(json.dumps({'initial_workers': 4, 'maximum_workers': 8, 'capacity': capacity['capacity'],
                          'training_cpu_reserve': 4, 'jobs': [{**j, 'depends_on': dependency(j),
                          'state': source.get('jobs', {}).get(j['name'], {}).get('state', 'pending')} for j in jobs]}, indent=2))
        return
    if capacity['capacity'] < 4 or not all(capacity['tools'].values()):
        parser.error('Insufficient verified CPU/RAM or missing nice/ionice/taskset; serial evaluation unchanged')
    run(root, study, capacity, args.takeover_serial, args.resume)


if __name__ == '__main__':
    main()
