"""Plan or resume evaluation only, using frozen models/code and spawned CPU workers."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
from datetime import datetime
import hashlib
import importlib.util
import json
import multiprocessing
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(os.environ.get('UGG_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
ORDER = [('mp20', 'before'), ('mp20', 'after'), ('c2db51', 'after'),
         ('c2db51', 'before'), ('joint', 'before'), ('joint', 'after')]


def digest(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def write(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def make_plan(study):
    protocol = json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    for key in ('files', 'frozen_files'):
        base = ROOT if key == 'files' else study
        for relative, expected in protocol[key].items():
            if digest(base/relative) != expected:
                raise ValueError('Original input/source changed: '+relative)
    jobs = []
    for dataset, version in ORDER:
        run = study/'runs'/dataset/version
        receipt = json.loads((run/'training_complete.json').read_text(encoding='utf-8'))
        if receipt['completed_epochs'] != protocol['epochs']:
            raise ValueError('Training budget is incomplete: '+str(run))
        if digest(run/'model.pt') != receipt['checkpoint_sha256']:
            raise ValueError('Trained checkpoint changed: '+str(run))
        complete = study/'evaluations'/dataset/version/'complete.json'
        reused = False
        if complete.exists():
            if json.loads(complete.read_text())['checkpoint_sha256'] != receipt['checkpoint_sha256']:
                raise ValueError('Evaluation/checkpoint mismatch: '+str(complete))
            reused = True
        jobs.append({'dataset': dataset, 'version': version, 'action': 'reuse' if reused else 'evaluate',
                     'checkpoint_sha256': receipt['checkpoint_sha256']})
    return {'study': str(study), 'training_jobs': 0, 'cpu_start_method': 'spawn', 'jobs': jobs,
            'scope': 'Reuse completed tests, sampling and metrics; finish missing evaluations only'}


def frozen_runner(study):
    os.environ['UGG_PROJECT_ROOT'] = str(ROOT)
    sys.path[:0] = [str(study/'versions/after/scripts'), str(study/'versions/after/src')]
    path = study/'versions/after/scripts/loss_balance_study.py'
    spec = importlib.util.spec_from_file_location('frozen_loss_evaluation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def require_inactive(study):
    if not sys.platform.startswith('linux'):
        raise ValueError('--run recovery requires Linux process checks; --plan is cross-platform')
    state = json.loads((study/'status.json').read_text())
    pid = state.get('pid')
    if pid and Path(f'/proc/{pid}').exists():
        raise ValueError(f'Original controller PID {pid} is still present; no process will be stopped automatically')
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            command = (entry/'cmdline').read_bytes().decode(errors='replace')
        except OSError:
            continue
        if 'loss_balance_study.py' in command and study.name in command:
            raise ValueError('Original study worker is still present: '+entry.name)


def resume(study, plan):
    require_inactive(study)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    recovery = study/'recovery'/stamp
    recovery.mkdir(parents=True, exist_ok=False)
    launcher = recovery/Path(__file__).name
    shutil.copyfile(__file__, launcher)
    old = json.loads((study/'status.json').read_text())
    write(recovery/'receipt.json', {'previous_status': old, 'plan': plan,
        'launcher_sha256': digest(launcher), 'protocol_sha256': digest(study/'protocol.json'),
        'change': 'CPU multiprocessing start method only; frozen training/evaluation source unchanged'})
    lock = study/'pipeline.lock'
    if lock.exists():
        lock.rename(recovery/'previous_pipeline.lock')
    with lock.open('x', encoding='utf-8') as stream:
        stream.write(str(os.getpid()))
    state = {**old, 'state': 'running', 'pid': os.getpid(), 'recovery': str(recovery.relative_to(study))}
    state.pop('error', None)
    state.setdefault('jobs', {})
    try:
        for job in plan['jobs']:
            key = 'evaluate/'+job['dataset']+'/'+job['version']
            if job['action'] == 'reuse':
                state['jobs'][key] = {'state': 'complete', 'reused_verified': True}
                continue
            state.update(current_job=key)
            state['jobs'][key] = {'state': 'running', 'started_at': time.time(), 'cpu_start_method': 'spawn'}
            write(study/'status.json', state)
            print('[START] '+key+' (spawn)', flush=True)
            with (study/(key.replace('/', '_')+'.log')).open('a', encoding='utf-8') as stream:
                subprocess.run([sys.executable, '-u', str(launcher), '--study', str(study), '--worker',
                                '--dataset', job['dataset'], '--version', job['version']],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True,
                    env={**os.environ, 'UGG_PROJECT_ROOT': str(ROOT), 'PYTHONIOENCODING': 'utf-8',
                         'PYTHONPATH': str(study/'versions/after/src')})
            state['jobs'][key].update(state='complete', finished_at=time.time())
            print('[DONE] '+key, flush=True)
        runner = frozen_runner(study)
        runner.summarize(study, json.loads((study/'protocol.json').read_text()))
        state.update(state='complete', finished_at=time.time())
    except BaseException as error:
        state.update(state='failed', error=str(error))
        raise
    finally:
        write(study/'status.json', state)
        lock.unlink()
    print('[COMPLETE] evaluation recovery', flush=True)
    subprocess.run([sys.executable, str(ROOT/'scripts/pack_loss_study.py'), '--study', str(study)],
                   cwd=ROOT, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', default='artifacts/loss_balance_4090_seed42')
    parser.add_argument('--run', action='store_true', help='Resume only after the old process tree is inactive')
    parser.add_argument('--plan', action='store_true', help='Read and validate files only (default)')
    parser.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    parser.add_argument('--dataset', choices=('mp20', 'c2db51', 'joint'))
    parser.add_argument('--version', choices=('before', 'after'))
    args = parser.parse_args()
    study = (ROOT/args.study).resolve()
    if not study.is_relative_to((ROOT/'artifacts').resolve()):
        parser.error('Study must be inside project artifacts')
    if args.worker:
        if not args.dataset or not args.version:
            parser.error('Worker requires dataset/version')
        # Set before importing the frozen PyTorch evaluator. Its default pool
        # then uses spawn without editing the frozen files or their hashes.
        multiprocessing.set_start_method('spawn', force=True)
        print('[RECOVERY] CPU start method: '+multiprocessing.get_start_method(), flush=True)
        frozen_runner(study).worker(study, args.dataset, args.version, 'evaluate')
        return
    if args.run and args.plan:
        parser.error('Choose --run or --plan')
    plan = make_plan(study)
    if args.run:
        resume(study, plan)
    else:
        print(json.dumps(plan, indent=2), flush=True)


if __name__ == '__main__':
    main()
