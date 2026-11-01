"""Retry failed phases using their original source, excluding a suspect node."""
import argparse
import getpass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from filelock import FileLock
from submit_files.hpc5090 import job_plan, read, write


def active_jobs():
    output = subprocess.check_output(['yhq', '-h', '-u', getpass.getuser(), '-o', '%i %T'], text=True)
    return {row[0]: row[1] for line in output.splitlines()
            if len(row := line.split()) == 2 and row[0].isdigit()}


def phase_state(folder, phase, model):
    path = folder/'state'/(phase+'_'+model+'.json')
    return read(path).get('state') if path.exists() else None


def pending_exclusion(control, job, exclude):
    # Recheck before editing; running jobs are never changed or cancelled.
    if active_jobs().get(job) not in ('PENDING', 'PD'):
        return
    details = subprocess.check_output([control, '-o', 'show', 'job', job], text=True)
    match = re.search(r'\bExcNodeList=(\S+)', details)
    previous = match.group(1) if match else '(null)'
    nodes = exclude if previous in ('(null)', 'None', 'N/A') else previous
    if exclude not in nodes.split(','):
        nodes += ','+exclude
    try:
        subprocess.run([control, 'update', 'JobId='+job, 'ExcNodeList='+nodes], check=True)
    except subprocess.CalledProcessError:
        if active_jobs().get(job) in ('PENDING', 'PD'):
            raise
        print('LEFT UNCHANGED: job started or left the queue:', job, flush=True)


def recover(manifest, exclude, submit=False):
    campaign = read(manifest)
    folder = Path(campaign['folder'])
    source = Path(campaign['source'])
    if not (source/'submit_files/hpc5090.sbatch').is_file():
        raise ValueError('Original source snapshot is missing')
    with FileLock(str(folder/'recovery.lock'), timeout=0):
        submitted = read(folder/'submitted.json')
        receipt_path = folder/'recovery.json'
        receipt = read(receipt_path) if receipt_path.exists() else {
            'revision': campaign['revision'], 'exclude': exclude, 'jobs': {}}
        if (receipt['revision'], receipt['exclude']) != (campaign['revision'], exclude):
            raise ValueError('Existing recovery has different source/exclusion; inspect recovery.json first')
        for key, previous in receipt['jobs'].items():
            submitted[key] = previous['job_id']
        live = active_jobs()
        pending = [job for job in submitted.values() if live.get(job) in ('PENDING', 'PD')]
        plan = []
        for key, phase, model, _, hours in job_plan(campaign['legacy_models']):
            if (phase in ('train', 'legacy', 'new') and key not in receipt['jobs']
                    and submitted.get(key) not in live and phase_state(folder, phase, model) == 'failed'):
                plan.append((key, phase, model, hours))
        retried_trains = {r[2] for r in plan if r[1] == 'train'} | {
            key.split(':', 1)[1] for key in receipt['jobs'] if key.startswith('train:')}
        for model in sorted(retried_trains):
            key = 'new:'+model
            if (key not in receipt['jobs'] and submitted.get(key) not in live
                    and phase_state(folder, 'new', model) != 'complete' and key not in {r[0] for r in plan}):
                plan.append((key, 'new', model, 24))
        print('Frozen source:', campaign['revision'])
        print('Exclude:', exclude)
        print('Own pending jobs to update:', ', '.join(pending) or 'none')
        for key, _, _, hours in plan:
            print('RETRY:', key, 'limit='+str(hours)+'h')
        if not submit:
            print('PLAN ONLY. Add --submit to update pending exclusions and queue these retries.')
            return
        control = shutil.which('yhcontrol') or shutil.which('scontrol')
        if pending and not control:
            raise RuntimeError('Neither yhcontrol nor scontrol is available; no jobs were submitted')
        for job in pending:
            pending_exclusion(control, job, exclude)
        backup = folder/'submitted.before_recovery.json'
        if not backup.exists():
            write(backup, submitted)
        write(folder/'submitted.json', submitted)
        for key, phase, model, hours in plan:
            # Stop if another invocation/user has started this phase in the meantime.
            live = active_jobs()
            if submitted.get(key) in live or phase_state(folder, phase, model) in ('running', 'complete'):
                print('SKIP ACTIVE/COMPLETE:', key, flush=True)
                continue
            dependency = None
            train_key = 'train:'+model
            if phase == 'new' and model in retried_trains:
                if phase_state(folder, 'train', model) != 'complete':
                    dependency = submitted.get(train_key)
                    if dependency not in live:
                        raise RuntimeError('Retry training is not active or complete: '+model)
            command = ['yhbatch', '--parsable', '--export=ALL', '--kill-on-invalid-dep=yes',
                       '--exclude='+exclude, '--chdir='+str(source), '--time='+str(hours)+':00:00',
                       '--job-name=ugg_retry_'+phase+'_'+model,
                       '--output='+str(folder/'logs/%x-%j.out')]
            if dependency:
                command.append('--dependency=afterok:'+dependency)
            command += [str(source/'submit_files/hpc5090.sbatch'), '--campaign', str(manifest),
                        '--phase', phase, '--model', model]
            new = subprocess.check_output(command, text=True,
                env={**os.environ, 'UGG_GIT_COMMIT': campaign['revision']}).strip().split(';')[0]
            if not new.isdigit():
                raise RuntimeError('Unexpected yhbatch reply: '+new)
            receipt['jobs'][key] = {'previous_job': submitted.get(key), 'job_id': new}
            write(receipt_path, receipt)
            submitted[key] = new
            write(folder/'submitted.json', submitted)
            print('QUEUED:', key, new, flush=True)
        print('Recovery record:', receipt_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', required=True, type=Path)
    parser.add_argument('--exclude', required=True)
    parser.add_argument('--submit', action='store_true')
    args = parser.parse_args()
    recover(args.campaign.resolve(), args.exclude, args.submit)
