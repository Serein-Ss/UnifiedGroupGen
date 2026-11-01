"""One allocated GPU generates while bounded CPU subprocesses evaluate earlier jobs."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

from omegaconf import OmegaConf

from scripts import evaluate_models as suite
from submit_files.hpc5090 import CATALOG, TRAIN_MODELS, checkpoint, read, write
from unifiedgroupgen.common.configuration import model_result_directory, project_root
from unifiedgroupgen.common.structures import read_records, sha256, verified_prior_records


def arguments(campaign, model, phase):
    root = project_root()
    options = {}
    for name in ('default', 'main15_local'):
        options.update({k: v for k, v in OmegaConf.to_container(OmegaConf.load(
            root/'configs/evaluation_suite'/f'{name}.yaml'), resolve=True).items() if k != 'defaults'})
    folder = Path(campaign['folder'])
    options.update(models=[model], experiments=folder/('legacy_catalog.yaml' if phase == 'legacy' else 'new_catalog.yaml'),
                   output=folder/'evaluation'/model, device='cuda', threads=3, seed=142,
                   run=True, plan=False, task_prefix=campaign['name']+'_', stage='generate',
                   generation_batch_size=32 if model.startswith('c2db51_') else 16,
                   generation_precision='fp32', generation_solver='heun', test_batch_size=32,
                   reference_cache=folder/'shared/reference_cache',
                   property_reference_cache=folder/'shared/property_reference_cache')
    if phase == 'legacy':
        options['reuse_manifest'] = folder/'reuse_manifest.json'
    else:
        options.update(reuse_manifest=None, conditional_tasks=[], target_properties=[])
    options['metric_workers'] = campaign.get('metric_workers', 3)
    budget = folder/'legacy_budget.json'
    if phase == 'legacy' and budget.exists() and read(budget)['models'].get(model, {}).get('new_attempts') == 0:
        # With no fresh sampling, spend the otherwise idle CPU cores on metrics.
        options.update(threads=1, metric_workers=6)
    return SimpleNamespace(**options)


def import_previous(job, campaign, inventory_path):
    """Import directory-provenance legacy candidates; never invent missing old identity receipts."""
    logical = job['name'][len(campaign['name'])+1:]
    base = Path(job['checkpoint']).parent/'generate'
    legacy_names = [base/('diagnostic256_'+logical+'_seed142')]
    aliases = {'dng': 'de_novo', 'csp': 'composition', 'fixed_descriptor': 'fixed_description'}
    suffix = logical.removeprefix(job['dataset']+'_')
    if suffix in aliases:
        legacy_names.append(base/('baseline256_'+job['dataset']+'_'+aliases[suffix]+'_seed142'))
    rows = {}; sources = {}; rejected = []
    limit = len(job['case_ids'])*job['k'] if job['task'] == 'csp' else job['count']
    for old in legacy_names:
        if len(rows) == limit:
            break
        ledger = old/'samples.jsonl'; status = old/'sampling.status.json'
        if not ledger.exists():
            continue
        try:
            count = len(read_records(ledger))
            if status.exists() and read(status)['attempted'] != count:
                raise ValueError('Sampling status and ledger length disagree')
            spec = {'samples': str(ledger), 'samples_sha256': sha256(ledger), 'records': count}
            previous = verified_prior_records(spec)
            accepted = 0
            for row in previous:
                if row.get('precision', 'fp32') != 'fp32' or row.get('properties_requested', {}) != job['properties']:
                    continue
                if job['task'] == 'csp':
                    case = str(row.get('case_id', row.get('reference_id')))
                    candidate = int(row.get('candidate', 0))
                    if case not in job['case_ids'] or not 0 <= candidate < job['k']:
                        continue
                    slot = (case, candidate)
                else:
                    slot = row['index']
                    if slot >= limit:
                        continue
                    group = job['groups'][slot//job['per_group']] if job.get('groups') else job.get('group')
                    if row.get('requested_group') != group:
                        continue
                if slot not in rows:
                    rows[slot] = row; accepted += 1
            if accepted:
                sources[str(ledger)] = spec['samples_sha256']
                if status.exists():
                    sources[str(status)] = sha256(status)
        except (ValueError, FileNotFoundError, KeyError) as error:
            rejected.append({'directory': str(old), 'reason': str(error)})
    if job['task'] == 'csp':
        retained = [rows[(case, k)] for case in job['case_ids'] for k in range(job['k']) if (case, k) in rows]
    else:
        # The generator imports non-CSP candidates by position, so only a contiguous prefix is valid.
        retained = []
        for i in range(limit):
            if i not in rows:
                break
            retained.append(rows[i])
    if not retained:
        return None, rejected
    target = Path(campaign['folder'])/'reuse'/job['model']/job['name']
    target.mkdir(parents=True, exist_ok=True)
    ledger = target/'samples.jsonl'
    with ledger.open('w', encoding='utf-8') as stream:
        for i, row in enumerate(retained):
            row = {**row, 'index': i, 'legacy_origin': row['reused_from']}
            stream.write(json.dumps(row, allow_nan=False)+'\n')
    receipt = target/'import.json'
    write(receipt, {'checkpoint_sha256': job['checkpoint_sha256'], 'source_files_sha256': sources,
                   'provenance_basis': 'retained named diagnostic/baseline directories plus audited checkpoint inventory',
                   'limitation': 'Historical folders have no original code/options identity receipt; 64-step Heun FP32 is the retained campaign protocol, not newly proven provenance.',
                   'records': len(retained), 'failed_attempts_retained': sum(r['status'] == 'failed' for r in retained)})
    return {'checkpoint_sha256': job['checkpoint_sha256'], 'prior_samples': {
        'samples': str(ledger), 'samples_sha256': sha256(ledger), 'records': len(retained),
        'metadata_sha256': {str(receipt): sha256(receipt), str(inventory_path): sha256(inventory_path), **sources}}}, rejected


def prepare_legacy(campaign):
    root = project_root(); folder = Path(campaign['folder'])
    inventory_path = root/'configs/hpc5090_legacy_checkpoints.json'
    inventory = {m['name']: m for m in read(inventory_path)['models']}
    catalog = OmegaConf.to_container(OmegaConf.load(root/'configs/experiments.yaml'), resolve=True)
    selected = {}
    for model in campaign['legacy_models']:
        print('VERIFY HISTORICAL MODEL', model, flush=True)
        item = inventory[model]
        path = root/item['directory']/'model.best.pt'
        if sha256(path) != item['best_checkpoint_sha256']:
            raise ValueError('Historical checkpoint differs from the audited inventory: '+str(path))
        selected[model] = {**catalog['experiments'][model], 'checkpoint': str(path)}
    OmegaConf.save(OmegaConf.create({'experiments': selected}), folder/'legacy_catalog.yaml')
    new = OmegaConf.to_container(OmegaConf.load(root/CATALOG), resolve=True)
    for model in TRAIN_MODELS:
        new['experiments'][model]['checkpoint'] = str(checkpoint(campaign, model))
    OmegaConf.save(OmegaConf.create(new), folder/'new_catalog.yaml')
    imports = {'jobs': {}}; summary = {'models': {}, 'rejected_sources': []}
    # Manifest must exist while build_plan reads it. It is populated only after complete verification.
    write(folder/'reuse_manifest.json', imports)
    for model in campaign['legacy_models']:
        args = arguments(campaign, model, 'legacy'); args.reuse_manifest = None
        plan, total, _ = suite.build_plan(args, root)
        reused = failed = 0
        with ThreadPoolExecutor(max_workers=4) as pool:
            pending = [(job, pool.submit(import_previous, job, campaign, inventory_path))
                       for job in plan if job['task'] != 'components']
            for job, future in pending:
                imported, rejected = future.result()
                summary['rejected_sources'].extend(rejected)
                if imported:
                    imports['jobs'][model+'/'+job['name']] = imported
                    reused += imported['prior_samples']['records']
                    failed += sum(r['status'] == 'failed' for r in read_records(imported['prior_samples']['samples']))
        summary['models'][model] = {'planned_attempts': total, 'retained_attempts': reused,
                                    'retained_failures': failed, 'new_attempts': total-reused}
        print('PREPARED', model, summary['models'][model], flush=True)
    write(folder/'reuse_manifest.json', imports)
    write(folder/'legacy_budget.json', summary)


def metric_worker(specification):
    spec = read(specification)
    args = arguments(spec['campaign'], spec['job']['model'], spec['phase'])
    args.device = 'cpu'; args.threads = 1
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'}
    start = time.perf_counter()
    suite.evaluate_metrics(spec['job'], args, Path(spec['target']), {},
                           Path(specification).with_suffix('.progress.json'), env)
    metrics = model_result_directory(spec['job']['checkpoint'], 'evaluate', spec['job']['name'])
    write(Path(specification).with_suffix('.done.json'), {
        'metric_directory': str(metrics), 'outputs': suite.output_hashes(metrics),
        'evaluation_seconds': time.perf_counter()-start})


def reap(active, state, pipeline, wait=False):
    if wait:
        while all(proc.poll() is None for proc, _, _, _ in active):
            time.sleep(.2)
    for proc, log, entry, spec in list(active):
        code = proc.poll()
        if code is None:
            continue
        log.close(); active.remove((proc, log, entry, spec))
        if code:
            raise RuntimeError('Metric worker failed: '+entry['name']+'; inspect '+str(spec.with_suffix('.log')))
        receipt = read(spec.with_suffix('.done.json'))
        entry.update(state='complete', metrics=receipt, metric_directory=receipt['metric_directory'],
                     evaluation_seconds=receipt['evaluation_seconds'],
                     seconds=entry['generation']['seconds']+receipt['evaluation_seconds'])
        write(pipeline, state)
        print('METRICS COMPLETE', entry['name'], flush=True)


def evaluate_model(campaign, model, phase):
    args = arguments(campaign, model, phase)
    args.output.mkdir(parents=True, exist_ok=True)
    plan, total, identity = suite.build_plan(args, project_root())
    identity['controller_code'] = {p.name: sha256(p) for p in (Path(__file__), Path(__file__).with_name('hpc5090.py'))}
    if any(j['checkpoint_sha256'] is None for j in plan):
        raise ValueError('Evaluation requires trained checkpoints')
    pipeline = args.output/'pipeline.json'
    state = read(pipeline) if pipeline.exists() else {
        'state': 'running', 'identity': identity, 'jobs': [], 'generation_attempts': total, 'started_at': time.time()}
    if state['identity'] != identity:
        raise ValueError('Changed checkpoint/data/code/options: choose a new campaign name')
    state['state'] = 'running'
    active = []
    env = {**os.environ, 'OMP_NUM_THREADS': str(args.threads), 'MKL_NUM_THREADS': str(args.threads)}
    try:
        for job in plan:
            reap(active, state, pipeline)
            key = model+'/'+job['name']; target = suite.job_directory(job)
            entry = next((r for r in state['jobs'] if r['name'] == key), None)
            if entry and entry.get('state') == 'complete':
                suite.verify_files(target, entry.get('generation', {}).get('outputs', entry.get('outputs', {})))
                if entry.get('metrics'):
                    suite.verify_files(Path(entry['metrics']['metric_directory']), entry['metrics']['outputs'])
                continue
            if entry is None:
                entry = {'name': key, 'state': 'pending'}; state['jobs'].append(entry)
            entry.update(directory=str(target), started=time.time())
            target.mkdir(parents=True, exist_ok=True)
            write(target/'job.json', job)
            suite.run_job(job, args, target, entry, state, pipeline, env)
            if job['task'] == 'components':
                continue
            while len(active) >= getattr(args, 'metric_workers', campaign['metric_workers']):
                reap(active, state, pipeline, wait=True)
            spec = args.output/(job['name']+'.worker.json')
            write(spec, {'campaign': campaign, 'phase': phase, 'job': job, 'target': str(target)})
            log = spec.with_suffix('.log').open('a', encoding='utf-8')
            proc = subprocess.Popen([sys.executable, '-u', '-m', 'submit_files.hpc5090_eval', '--metrics-worker', str(spec)],
                                    env={**env, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1'},
                                    stdout=log, stderr=subprocess.STDOUT)
            active.append((proc, log, entry, spec))
        while active:
            reap(active, state, pipeline, wait=True)
        state['state'] = 'complete'; write(pipeline, state)
        from unifiedgroupgen.common.reports import campaign_summary
        campaign_summary(args.output, state)
    except BaseException as error:
        # Let already running metrics finish cleanly; preserve their receipts for retry.
        for proc, log, _, _ in active:
            proc.wait(); log.close()
        state.update(state='failed', error=str(error)); write(pipeline, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metrics-worker', required=True, type=Path)
    metric_worker(parser.parse_args().metrics_worker)
