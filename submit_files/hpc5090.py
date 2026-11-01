"""Submit independent TH-HPC6 jobs from one immutable source snapshot."""
import argparse
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import subprocess
import tarfile


TRAIN_MODELS = (
    'mp20_u_comp_v2', 'c2db51_u_comp_v2', 'joint_u_comp_v2',
    'joint_u_comp_source_control_v2',
)
CATALOG = 'configs/experiments_hpc5090.yaml'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    temporary.replace(path)


def job_plan(legacy_models):
    jobs = [('prepare', 'prepare', '', [], 4), ('profile', 'profile', '', ['prepare'], 2)]
    jobs += [('train:'+m, 'train', m, ['profile'], 72) for m in TRAIN_MODELS]
    jobs += [('legacy:'+m, 'legacy', m, ['prepare'], 24) for m in legacy_models]
    jobs += [('new:'+m, 'new', m, ['train:'+m], 24) for m in TRAIN_MODELS]
    return jobs


def snapshot(root, folder, revision):
    destination = folder/'source'
    destination.mkdir()
    archive = subprocess.Popen(['git', 'archive', revision, 'unifiedgroupgen', 'configs', 'scripts', 'submit_files'],
                               cwd=root, stdout=subprocess.PIPE)
    with tarfile.open(fileobj=archive.stdout, mode='r|') as stream:
        stream.extractall(destination)
    if archive.wait():
        raise RuntimeError('Cannot freeze tracked source')
    for name in ('data', 'outputs', 'assets', 'intermediate', 'wandb'):
        (root/name).mkdir(exist_ok=True)
        (destination/name).symlink_to(root/name, target_is_directory=True)
    if (root/'.env').exists():
        (destination/'.env').symlink_to(root/'.env')
    return destination


def submit(name):
    import yaml
    root = Path(__file__).resolve().parents[1]
    if not name or Path(name).name != name or any(c in name for c in '/\\:'):
        raise ValueError('Use a single campaign directory name')
    folder = root/'intermediate/hpc_runs'/name
    manifest = folder/'campaign.json'
    if manifest.exists():
        campaign = read(manifest)
    else:
        subprocess.run(['git', 'diff', '--quiet', 'HEAD'], cwd=root, check=True)
        revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
        legacy = yaml.safe_load((root/'configs/evaluation_suite/main15_local.yaml').read_text(encoding='utf-8'))['models']
        folder.mkdir(parents=True, exist_ok=False)
        source = snapshot(root, folder, revision)
        campaign = {'name': name, 'root': str(root), 'folder': str(folder), 'source': str(source),
                    'revision': revision, 'legacy_models': legacy, 'train_models': list(TRAIN_MODELS),
                    'training_seed': 42, 'evaluation_seed': 142, 'metric_workers': 3,
                    'resources': {'partition': '5090', 'gpus': 1, 'cpus': 8, 'memory': '64G'}}
        write(manifest, campaign)
    (folder/'logs').mkdir(exist_ok=True)
    submitted = read(folder/'submitted.json') if (folder/'submitted.json').exists() else {}
    env = {**os.environ, 'UGG_GIT_COMMIT': campaign['revision']}
    for key, phase, model, dependencies, hours in job_plan(campaign['legacy_models']):
        if key in submitted:
            print('ALREADY QUEUED', key, submitted[key], flush=True)
            continue
        command = ['yhbatch', '--parsable', '--export=ALL', '--kill-on-invalid-dep=yes',
                   '--job-name=ugg_'+phase+('_'+model if model else ''),
                   '--chdir='+campaign['source'], '--time='+str(hours)+':00:00',
                   '--output='+str(folder/'logs/%x-%j.out')]
        if dependencies:
            command.append('--dependency=afterok:'+':'.join(submitted[d] for d in dependencies))
        command += [str(Path(campaign['source'])/'submit_files/hpc5090.sbatch'),
                    '--campaign', str(manifest), '--phase', phase]
        if model:
            command += ['--model', model]
        job = subprocess.check_output(command, cwd=root, env=env, text=True).strip().split(';')[0]
        if not job.isdigit():
            raise RuntimeError('Unexpected yhbatch job ID: '+job)
        submitted[key] = job
        write(folder/'submitted.json', submitted)
        print('QUEUED', key, job, flush=True)
    print('Campaign:', manifest)
    print('Source revision:', campaign['revision'])
    print('All jobs are submitted; the scheduler enforces account GPU limits.')


def checkpoint(campaign, model):
    return Path(campaign['root'])/'outputs'/model/campaign['name']/'model.best.pt'


def audit_model(model):
    from unifiedgroupgen.pl_data.prepare import audit_experiments
    audit_experiments(CATALOG, [model])


def prepare(campaign):
    from unifiedgroupgen.common.configuration import resolve_path
    from unifiedgroupgen.common.predictors.matgl import MatGLPredictor, model_registry
    from unifiedgroupgen.common.structures import sha256
    registry = model_registry()
    cache = resolve_path('data/scr/pretrained/matgl')
    for spec in registry['models'].values():
        for name, expected in spec['files'].items():
            path = cache/spec['name']/name
            if not path.is_file() or sha256(path) != expected:
                raise ValueError('Prepare the frozen public predictor cache on the login node first: '+str(path))
    # Check Linux imports/loading from the local cache, without a network download.
    predictor = MatGLPredictor(cache)
    from pymatgen.core import Lattice, Structure
    probes = {
        'space': Structure.from_spacegroup(225, Lattice.cubic(5.64), ['Na', 'Cl'], [[0, 0, 0], [.5, .5, .5]]),
        'layer': Structure(Lattice.hexagonal(3.16, 20), ['Mo', 'S', 'S'],
                           [[0, 0, .5], [1/3, 2/3, .578], [1/3, 2/3, .422]]),
    }
    for kind, structure in probes.items():
        result = predictor.predict_structures([structure], kind=kind)[0]
        if result['prediction_error'] or any(result[name] is None for name in registry['models']):
            raise ValueError('Public predictor smoke check failed: '+str(result))
    processes = []
    for model in TRAIN_MODELS:
        log = (Path(campaign['folder'])/'logs'/('audit-'+model+'.log')).open('w', encoding='utf-8')
        proc = subprocess.Popen([os.sys.executable, '-u', 'submit_files/hpc5090.py', 'audit-model', model],
                                stdout=log, stderr=subprocess.STDOUT)
        processes.append((model, proc, log))
    errors = []
    for model, proc, log in processes:
        code = proc.wait(); log.close()
        if code:
            errors.append(model)
    if errors:
        raise RuntimeError('Data audits failed; inspect per-model audit logs: '+', '.join(errors))
    from submit_files.hpc5090_eval import prepare_legacy
    prepare_legacy(campaign)


def train(campaign, model):
    from unifiedgroupgen.common.configuration import load_experiments, project_root
    catalog = load_experiments(CATALOG)
    entry = catalog[model]
    profile = read(Path(campaign['folder'])/'profile.json')
    family = 'joint' if model.startswith('joint_') else ('mp20' if model.startswith('mp20_') else 'c2db51')
    selected = profile['selected'][family]
    output = checkpoint(campaign, model).with_name('model.pt')
    status = output.with_suffix('.status.json')
    if status.exists() and read(status).get('state') == 'complete':
        if not checkpoint(campaign, model).is_file():
            raise ValueError('Complete training status has no best checkpoint: '+str(output))
        print('Training already complete:', output)
        return
    command = [os.sys.executable, '-u', '-m', 'unifiedgroupgen.run', 'experiment='+entry['experiment'],
               *entry.get('overrides', []), 'device=cuda', 'threads=4', 'seed=42', 'logging.mode=offline',
               'runtime.batch_size='+str(selected['batch_size']),
               'runtime.edge_budget='+str(selected['edge_budget']),
               'runtime.edge_chunk_size=32768', 'runtime.checkpoint=false']
    command += ['resume='+str(output)] if output.exists() else ['output='+str(output)]
    print('TRAIN COMMAND', json.dumps(command), flush=True)
    subprocess.run(command, cwd=project_root(), check=True)
    if not status.is_file() or read(status).get('state') != 'complete' or not checkpoint(campaign, model).is_file():
        raise RuntimeError('Training exited without a complete status and best checkpoint')


def run(campaign_path, phase, model):
    from filelock import FileLock
    campaign = read(campaign_path)
    os.environ['UGG_CAMPAIGN_NAME'] = campaign['name']
    if phase in ('train', 'new') and model not in TRAIN_MODELS:
        raise ValueError('Unknown training model')
    if phase == 'legacy' and model not in campaign['legacy_models']:
        raise ValueError('Unknown historical model')
    state = Path(campaign['folder'])/'state'/(phase+('_'+model if model else '')+'.json')
    state.parent.mkdir(exist_ok=True)
    with FileLock(str(state)+'.lock', timeout=0):
        write(state, {'state': 'running', 'phase': phase, 'model': model, 'revision': campaign['revision']})
        try:
            if phase == 'prepare':
                prepare(campaign)
            elif phase == 'profile':
                from submit_files.hpc5090_profile import profile
                profile(campaign)
            elif phase == 'train':
                train(campaign, model)
            else:
                from submit_files.hpc5090_eval import evaluate_model
                evaluate_model(campaign, model, phase)
        except BaseException as error:
            write(state, {'state': 'failed', 'phase': phase, 'model': model, 'error': str(error)})
            raise
        write(state, {'state': 'complete', 'phase': phase, 'model': model, 'revision': campaign['revision']})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest='action', required=True)
    subs.add_parser('plan')
    submitter = subs.add_parser('submit')
    submitter.add_argument('--name', default=datetime.now(timezone(timedelta(hours=8))).strftime('hpc5090_%y%m%d_%H%M%S'))
    worker = subs.add_parser('run')
    worker.add_argument('--campaign', required=True, type=Path)
    worker.add_argument('--phase', required=True, choices=['prepare', 'profile', 'train', 'legacy', 'new'])
    worker.add_argument('--model', default='')
    audit = subs.add_parser('audit-model'); audit.add_argument('model', choices=TRAIN_MODELS)
    status = subs.add_parser('status'); status.add_argument('campaign', type=Path)
    args = parser.parse_args()
    if args.action == 'plan':
        import yaml
        root = Path(__file__).resolve().parents[1]
        legacy = yaml.safe_load((root/'configs/evaluation_suite/main15_local.yaml').read_text(encoding='utf-8'))['models']
        for key, phase, model, dependencies, hours in job_plan(legacy):
            print(key, '| GPU=1 CPU=8 RAM=64G | limit='+str(hours)+'h | afterok='+','.join(dependencies))
        print('25 jobs; no submission or training started. Wall-time limits are not predicted durations.')
    elif args.action == 'submit':
        submit(args.name)
    elif args.action == 'audit-model':
        audit_model(args.model)
    elif args.action == 'run':
        run(args.campaign, args.phase, args.model)
    else:
        campaign = read(args.campaign)
        print('Revision:', campaign['revision'])
        for path in sorted((Path(campaign['folder'])/'state').glob('*.json')):
            print(path.stem, json.dumps(read(path), ensure_ascii=False))
        print('Slurm jobs:', read(Path(campaign['folder'])/'submitted.json'))


if __name__ == '__main__':
    main()
