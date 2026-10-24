"""Serial CPU evaluation of the three historical models: 10 sampling jobs, no training."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from properties import digest

MODELS = (
    ('M1', 'local_large_seed42/mp20', 'mp20_local', 270, [('mp_20', None)]),
    ('M2', 'local_large_seed42/mpts52', 'mpts52_local', 135, [('mpts_52', None)]),
    ('M3', 'joint_space_layer_large_seed42', 'joint_space_layer_large', 195,
     [('mp_20', 0), ('mpts_52', 1), ('c2db_51', 2)]),
)
STAGES = ('test', 'sample', 'properties', 'metrics')


def plan(root, output, cgdit, predictor_root, predictor_seed=123, count=4096, steps=64, seed=142):
    import yaml
    jobs, models = [], []
    tools = Path(__file__).resolve().parent
    for model, run, config_name, epoch, domains in MODELS:
        config_path = root/'configs'/f'{config_name}.yaml'
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        checkpoint = root/'artifacts/runs'/run/'model.best.pt'
        tests = [root/p for p in config.get('test_files', [config['prepared_root']+'/test.jsonl'])]
        models.append({'name': model, 'checkpoint': str(checkpoint), 'expected_epoch': epoch,
                       'tests': [str(p) for p in tests],
                       'training_inputs': {split: [str(root/p) for p in config.get(split+'_files',
                           [config['prepared_root']+'/'+split+'.jsonl'])] for split in ('train', 'val')}})
        test_output = output/model/'test'
        jobs.append({'name': model+'_test', 'stage': 'test', 'marker': str(test_output/'summary.json'),
                     'command': [str(tools/'components.py'), '--project-root', str(root),
                                 '--checkpoint', str(checkpoint), '--output', str(test_output)]})
        for dataset, domain in domains:
            for mode in ('unconditional', 'conditional'):
                kind = 'layer' if dataset == 'c2db_51' else 'space'
                properties = {'source_domain': domain} if domain is not None else {}
                if mode == 'conditional':
                    properties.update(formation_energy=-1.)
                    properties['energy_above_hull' if dataset == 'mpts_52' else 'band_gap'] = 0. if dataset == 'mpts_52' else 1.
                tag = f'{model}_{dataset}_{mode}'
                folder = output/model/f'{dataset}_{mode}'
                samples = folder/'samples.jsonl'
                jobs.append({'name': tag, 'stage': 'sample', 'marker': str(folder/'summary.json'),
                    'command': ['-m', 'unifiedgroupgen.cli', '--threads', '1', '--seed', str(seed), 'sample',
                        '--checkpoint', str(checkpoint), '--device', 'cpu', '--kind', kind,
                        '--properties', json.dumps(properties), '--guidance', '2' if mode == 'conditional' else '0',
                        '--count', str(count), '--steps', str(steps), '--output', str(folder)]})
                prop = folder/'properties'
                jobs.append({'name': tag+'_properties', 'stage': 'properties', 'marker': str(prop/'summary.json'),
                    'command': [str(tools/'properties.py'), '--project-root', str(root), '--cgdit-root', str(cgdit),
                        '--predictor-root', str(predictor_root), '--predictor-seed', str(predictor_seed),
                        '--samples', str(samples), '--dataset', dataset, '--reference-files', *map(str, tests),
                        '--output', str(prop), '--reference-output', str(output/model/'reference_properties'/dataset)]})
                metrics = folder/'metrics'
                jobs.append({'name': tag+'_metrics', 'stage': 'metrics', 'marker': str(metrics/'summary.json'),
                    'command': [str(tools.parent/'evaluate_generation.py'), '--samples', str(samples),
                        '--dataset', dataset, '--reference-config', str(config_path), '--output', str(metrics),
                        '--figures', str(folder/'figures'), '--reidentify', '--reference-workers', '1',
                        '--assessments', str(prop/'assessments.csv'),
                        '--assessment-metadata', str(prop/'assessment_metadata.json'),
                        '--reference-assessments', str(prop/'reference_assessments.csv')]})
    jobs.sort(key=lambda job: STAGES.index(job['stage']))
    return models, jobs


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--cgdit-root', type=Path, required=True)
    parser.add_argument('--predictor-root', type=Path, required=True)
    parser.add_argument('--predictor-seed', type=int, default=123)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--count', type=int, default=4096)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--seed', type=int, default=142)
    parser.add_argument('--stages', nargs='+', choices=STAGES, default=list(STAGES))
    parser.add_argument('--run', action='store_true', help='Without this flag, print the plan only')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.count, args.steps) < 1:
        parser.error('Count and integration steps must be positive')
    root, output = args.project_root.resolve(), args.output.resolve()
    if not output.is_relative_to(root/'artifacts/evaluations'):
        parser.error('Keep this campaign in project-root/artifacts/evaluations, separate from training')
    models, jobs = plan(root, output, args.cgdit_root.resolve(), args.predictor_root.resolve(),
                       args.predictor_seed, args.count, args.steps, args.seed)
    if not args.run:
        print(json.dumps({'models': models, 'sampling_jobs': 10, 'attempts_per_job': args.count,
                          'total_sampling_attempts': 10*args.count, 'device': 'cpu', 'threads': 1,
                          'selected_stages': args.stages, 'jobs': jobs}, indent=2))
        return
    import torch
    torch.set_num_threads(1)
    identities = {}
    for model in models:
        checkpoint = Path(model['checkpoint'])
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if saved['epoch']+1 != model['expected_epoch'] or saved['config'].get('experiment_id'):
            raise ValueError(f"Wrong historical checkpoint for {model['name']}; expected epoch {model['expected_epoch']}")
        identities[model['name']] = {'checkpoint_sha256': digest(checkpoint),
                                    'test_sha256': [digest(p) for p in model['tests']]}
        for split, paths in model['training_inputs'].items():
            hashes = [digest(p) for p in paths]
            if hashes != saved.get(split+'_split_sha256'):
                raise ValueError(f'{model["name"]} {split} references do not match the historical checkpoint')
            identities[model['name']][split+'_sha256'] = hashes
        del saved
    sys.path.insert(0, str(args.cgdit_root.resolve()))
    from cgdit.evaluation import generated_properties as backend
    from properties import predictors_at
    predictors = predictors_at(args.predictor_root.resolve(), args.predictor_seed, backend)
    identity = {'models': identities, 'predictors': predictors, 'jobs': jobs,
                'tools_sha256': {p.name: digest(p) for p in Path(__file__).parent.glob('*.py')},
                'generator_sha256': {str(p.relative_to(root)): digest(p) for p in (root/'src/unifiedgroupgen').glob('*.py')},
                'metric_sha256': digest(Path(__file__).parent.parent/'evaluate_generation.py'),
                'predictor_code_sha256': {name: digest(args.cgdit_root/name) for name in
                    ('cgdit/evaluation/generated_properties.py', 'cgdit/prop_models/gnn_models/m3gnet.py',
                     'cgdit/common/data_utils.py', 'cgdit/prop_models/gnn_models/_layers.py')}}
    output.mkdir(parents=True, exist_ok=True)
    state_path, lock = output/'pipeline.json', output/'pipeline.lock'
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding='utf-8'))
        if not args.resume or state['identity'] != identity:
            raise ValueError('Existing campaign requires --resume with identical code, predictors, models and settings')
    else:
        state = {'identity': identity, 'jobs': {}}
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.write(descriptor, str(os.getpid()).encode()); os.close(descriptor)
    env = {**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'OMP_NUM_THREADS': '1', 'MKL_NUM_THREADS': '1',
           'OPENBLAS_NUM_THREADS': '1', 'NUMEXPR_NUM_THREADS': '1', 'PYTHONPATH': str(root/'src'),
           'UGG_PROJECT_ROOT': str(root),
           'MKL_THREADING_LAYER': 'SEQUENTIAL', 'PYTHONIOENCODING': 'utf-8'}
    state.update(state='running', pid=os.getpid())
    state.pop('error', None)
    try:
        for job in jobs:
            if job['stage'] not in args.stages:
                continue
            previous = state['jobs'].get(job['name'])
            marker = Path(job['marker'])
            if previous and previous['state'] == 'complete':
                if not marker.exists() or digest(marker) != previous['summary_sha256']:
                    raise ValueError('Completed result changed: '+job['name'])
                if any(not (output/path).exists() or digest(output/path) != checksum
                       for path, checksum in previous['output_sha256'].items()):
                    raise ValueError('Completed output files changed: '+job['name'])
                print('[REUSE] '+job['name'], flush=True)
                continue
            state.update(current=job['name']); state['jobs'][job['name']] = {'state': 'running'}
            save(state_path, state)
            print('[START] '+job['name'], flush=True)
            started = time.monotonic()
            with (output/(job['name']+'.log')).open('a', encoding='utf-8') as stream:
                process = subprocess.Popen([sys.executable, '-u', *job['command']], cwd=root, env=env,
                                           stdout=stream, stderr=subprocess.STDOUT)
                state['child_pid'] = process.pid; save(state_path, state)
                code = process.wait()
            if code:
                raise RuntimeError(f'{job["name"]} exited {code}; inspect its log')
            report = json.loads(marker.read_text(encoding='utf-8'))
            if job['stage'] == 'sample' and report['attempted'] != args.count:
                raise ValueError('Incomplete sampling attempts')
            if job['stage'] != 'sample' and report.get('state') != 'complete':
                raise ValueError('Incomplete evaluation result')
            state['jobs'][job['name']] = {'state': 'complete', 'summary_sha256': digest(marker),
                                        'seconds': time.monotonic()-started,
                                        'output_sha256': {str(p.relative_to(output)): digest(p)
                                            for p in marker.parent.rglob('*') if p.is_file()}}
            state.pop('child_pid', None)
            save(state_path, state); print('[DONE] '+job['name'], flush=True)
        state.update(state='complete' if len(state['jobs']) == len(jobs) and
                     all(j['state'] == 'complete' for j in state['jobs'].values()) else 'stages_complete')
        state.pop('current', None)
    except BaseException as error:
        state.update(state='failed', error=str(error))
        raise
    finally:
        save(state_path, state); lock.unlink()


if __name__ == '__main__':
    main()
