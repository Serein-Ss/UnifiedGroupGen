"""Finish official 2D preparation and the full shared 2D/3D model after existing jobs."""

import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['MKL_NUM_THREADS'] = '4'

import argparse
import ctypes
from ctypes import wintypes
import datetime
import json
from pathlib import Path
import subprocess
import sys
import traceback


ROOT = Path(__file__).resolve().parents[1]
os.environ['PYTHONPATH'] = str(ROOT/'src')


def open_process(pid):
    if os.name != 'nt':
        raise RuntimeError('This existing-job pipeline is for the current Windows device')
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.OpenProcess(0x100000 | 0x1000, False, pid)
    if not handle:
        raise ctypes.WinError(ctypes.get_last_error())
    return kernel, handle


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recovery-pid', type=int, required=True)
    parser.add_argument('--reference-runner-pid', type=int, required=True)
    parser.add_argument('--ase-db', help='Original C2DB ASE database for exact records absent from the public site')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    folder = ROOT/'artifacts/runs/full_pipeline'
    folder.mkdir(parents=True, exist_ok=True)
    lock = folder/'pipeline.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    handles = []
    status = {'pid': os.getpid(), 'state': 'running', 'jobs': [],
              'started': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    def update():
        path = folder/'pipeline.json'
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(status, indent=2), encoding='utf-8')
        temporary.replace(path)
    def wait(name, pid, process, allowed_codes=(0,)):
        status.update({'phase': name, 'waiting_pid': pid})
        update()
        if process is None:
            return 0
        kernel, handle = process
        while True:
            result = kernel.WaitForSingleObject(handle, 60000)
            if result == 0:
                code = wintypes.DWORD()
                if not kernel.GetExitCodeProcess(handle, ctypes.byref(code)):
                    raise ctypes.WinError(ctypes.get_last_error())
                status.setdefault('existing_jobs', {})[name] = {'pid': pid, 'exit_code': code.value}
                update()
                if code.value not in allowed_codes:
                    raise RuntimeError(f'Existing {name} process exited with {code.value}; inspect its logs')
                return code.value
            if result != 258:
                raise ctypes.WinError(ctypes.get_last_error())
    def run(name, command):
        job = {'name': name, 'command': command, 'state': 'running'}
        status['jobs'].append(job)
        status['phase'] = name
        status.pop('waiting_pid', None)
        with (folder/f'{name}.stdout.log').open('a', encoding='utf-8') as out, \
             (folder/f'{name}.stderr.log').open('a', encoding='utf-8') as err:
            process = subprocess.Popen([sys.executable, '-u', *command], cwd=ROOT, stdout=out, stderr=err)
            job['pid'] = process.pid
            update()
            code = process.wait()
        job.update({'state': 'complete' if code == 0 else 'failed', 'exit_code': code})
        update()
        if code:
            raise RuntimeError(f'{name} failed with {code}; inspect {name}.stderr.log')
    try:
        # Hold actual OS handles before waiting; later PID reuse cannot attach a different job.
        recovered = ROOT/'artifacts/source_recovery/c2db_official/recovery.json'
        finished_recovery_attempt = args.resume and recovered.exists()
        recovery = None if finished_recovery_attempt else open_process(args.recovery_pid)
        if recovery:
            handles.append(recovery)
        reference_path = ROOT/'artifacts/runs/local_large_seed42/runner.json'
        completed_references = args.resume and reference_path.exists() and json.loads(reference_path.read_text()).get('state') == 'complete'
        references = None if completed_references else open_process(args.reference_runner_pid)
        if references:
            handles.append(references)
        recovery_code = wait('source_recovery', args.recovery_pid, recovery, allowed_codes=(0, 2))
        if recovery_code == 2 or (recovered.exists() and not json.loads(recovered.read_text()).get('passed')):
            command = ['scripts/recover_c2db.py', '--output', 'artifacts/source_recovery/c2db_official']
            if args.ase_db:
                command += ['--ase-db', args.ase_db]
            # Retry from the cache, including any newly supplied original ASE records.
            # Exit 2 still fails the pipeline; no missing source rows are dropped.
            run('source_recovery_retry', command)
        run('c2db_prepare', ['scripts/prepare_training.py', '--config', 'configs/c2db_official_local.yaml',
                             '--workers', '2', '--resume'])
        manifest = ROOT/'artifacts/full_joint_sourcegrouped/manifest.json'
        if args.resume and manifest.exists():
            run('joint_reaudit', ['-m', 'unifiedgroupgen.cli', 'preflight', '--config',
                                 'configs/joint_space_layer_large.yaml', '--output', 'artifacts/full_joint_sourcegrouped/audit.json'])
        else:
            command = ['scripts/prepare_joint.py', '--config', 'configs/joint_space_layer_large.yaml']
            if args.resume:
                command += ['--resume']
            run('joint_prepare', command)
        wait('reference_training', args.reference_runner_pid, references)
        reference_status = json.loads((ROOT/'artifacts/runs/local_large_seed42/runner.json').read_text())
        if reference_status['state'] != 'complete':
            raise ValueError('Existing reference pipeline did not complete normally')
        model = 'artifacts/runs/joint_space_layer_large_seed42/model.pt'
        command = ['-m', 'unifiedgroupgen.cli', '--seed', '42', 'train', '--config',
                   'configs/joint_space_layer_large.yaml', '--device', 'cuda', '--output', model]
        if args.resume and (ROOT/model).exists():
            command += ['--resume', model]
        run('joint_train', command)
        best = str(Path(model).with_suffix('.best.pt'))
        tests = [f'artifacts/full_joint_sourcegrouped/{dataset}/test.jsonl' for dataset in ('mp_20', 'mpts_52', 'c2db_51')]
        run('joint_test', ['scripts/evaluate_checkpoint.py', '--checkpoint', best, '--device', 'cuda',
                           '--records', *tests, '--output', 'artifacts/runs/joint_space_layer_large_seed42/test.json'])
        for dataset, domain, kind, targets in (
            ('mp_20', 0, 'space', {'formation_energy': -1.0, 'band_gap': 1.0}),
            ('mpts_52', 1, 'space', {'formation_energy': -1.0, 'energy_above_hull': 0.0}),
            ('c2db_51', 2, 'layer', {'formation_energy': -1.0, 'band_gap': 1.0}),
        ):
            for mode in ('unconditional', 'conditional'):
                properties = {'source_domain': domain, **(targets if mode == 'conditional' else {})}
                run(f'{dataset}_{mode}', ['-m', 'unifiedgroupgen.cli', '--seed', '142', 'sample',
                                         '--checkpoint', best, '--device', 'cuda', '--kind', kind,
                                         '--properties', json.dumps(properties), '--guidance', '2' if mode == 'conditional' else '0',
                                         '--count', '64', '--steps', '64', '--output',
                                         f'artifacts/runs/joint_space_layer_large_seed42/{dataset}/{mode}'])
        status['state'] = 'complete'
        status['scope'] = 'Full-data shared model trained and evaluated; physical property accuracy and stability still need certification'
    except BaseException:
        status['state'] = 'failed'
        status['error'] = traceback.format_exc()
        raise
    finally:
        for kernel, handle in handles:
            kernel.CloseHandle(handle)
        status['finished'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
        update()
        lock.unlink()


if __name__ == '__main__':
    main()
