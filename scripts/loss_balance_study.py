"""Frozen full-data loss-scaling ablation; never launches or edits main15 runs."""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time
import torch
import numpy as np
import paired_training_study as paired

ROOT = Path(os.environ.get('UGG_PROJECT_ROOT', Path(__file__).resolve().parents[1])).resolve()
paired.ROOT = ROOT
DATASETS = ('mp20', 'c2db51', 'joint')
ORDER = [('mp20', 'before'), ('mp20', 'after'), ('c2db51', 'after'),
         ('c2db51', 'before'), ('joint', 'before'), ('joint', 'after')]


def verify(study, protocol):
    for relative, expected in protocol['files'].items():
        if paired.file_hash(ROOT/relative) != expected:
            raise ValueError('Data/config changed: '+relative)
    for relative, expected in protocol['frozen_files'].items():
        if paired.file_hash(study/relative) != expected:
            raise ValueError('Frozen source/initialization changed: '+relative)


def freeze_sources(study):
    frozen = study/'versions/after'
    sources = sorted((ROOT/'src/unifiedgroupgen').rglob('*.py'))
    sources.append(ROOT/'src/unifiedgroupgen/benchmark_comp_scaler.json')
    sources += [ROOT/'scripts'/name for name in (
        'loss_balance_study.py', 'paired_training_study.py', 'evaluate_checkpoint.py',
        'evaluate_generation.py', 'pack_loss_study.py')]
    for source in sources:
        target = frozen/source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    return frozen


def prepare(study, args):
    if (study/'protocol.json').exists():
        protocol = json.loads((study/'protocol.json').read_text())
        expected = (args.epochs, args.count, args.conditional_count, args.steps)
        actual = tuple(protocol[k] for k in ('epochs', 'unconditional_count', 'conditional_count', 'steps'))
        if expected != actual:
            raise ValueError('Existing study has a different budget')
        verify(study, protocol)
        return protocol
    sys.path.insert(0, str(ROOT/'src'))
    import yaml
    from unifiedgroupgen.cli import build_models, fit_training_config
    from unifiedgroupgen.conditioning import attach_dataset_domains
    from unifiedgroupgen.data import read_jsonl
    from unifiedgroupgen.preflight import check_training_gate
    torch.set_num_threads(4)
    protocol = {'schema': 1, 'epochs': args.epochs, 'seed': 42, 'validation_interval': 5,
                'unconditional_count': args.count, 'conditional_count': args.conditional_count,
                'steps': args.steps, 'sampling_seed': 142, 'guidance': 2.0,
                'order': ORDER, 'datasets': {}, 'files': {}, 'frozen_files': {},
                'primary_checkpoint': 'final_equal_budget',
                'loss_scales_angstrom': {'before': 1.0, 'after': 3.0},
                'common_settings': {'batch_size': 64, 'precision': 'bf16', 'threads': 4,
                    'checkpoint_layers': False, 'edge_chunk_size': 32768,
                    'gradient_diagnostics_interval': 100, 'constant_learning_rate': True},
                'scope': 'Full repaired data, full architecture, single-seed short-budget loss ablation',
                'quality_warning_drop_percentage_points': 10, 'raw_loss_warning_relative_increase': 0.10,
                'limits': 'Regression screen, not statistical noninferiority or final convergence. No property predictor or physics verification.',
                'formal15_started': False, 'formal_loss_config_selected': False}
    frozen = freeze_sources(study)
    for dataset in DATASETS:
        path = ROOT/f'configs/main15/{dataset}_all.yaml'
        config = yaml.safe_load(path.read_text(encoding='utf-8'))
        paths = {phase: [ROOT/p for p in config[phase+'_files']] for phase in ('train', 'val', 'test')}
        check_training_gate(config, paths['train'], paths['val'], ROOT)
        records = attach_dataset_domains([r for p in paths['train'] for r in read_jsonl(p)], config)
        fitted = fit_training_config(config, records)
        torch.manual_seed(42); random.seed(42); np.random.seed(42)
        proposal, flow = build_models(fitted, 'cpu')
        initial = study/'initial'/f'{dataset}.pt'
        initial.parent.mkdir(parents=True, exist_ok=True)
        initial_hash = paired.weight_hash((proposal, flow))
        rng = {'python': random.getstate(), 'numpy': np.random.get_state(),
               'torch': torch.get_rng_state(), 'cuda': []}
        torch.save({'proposal': proposal.state_dict(), 'flow': flow.state_dict(),
                    'rng': rng, 'weights_sha256': initial_hash}, initial)
        targets = {}
        names = [p for p, spec in fitted['properties'].items() if not spec.get('context')]
        for source in sorted({r['dataset'] for r in records}):
            complete = [r for r in records if r['dataset'] == source and
                        all(r['properties'].get(p) is not None and np.isfinite(r['properties'][p]) for p in names)]
            complete.sort(key=lambda r: (r['properties'][names[0]], str(r['id'])))
            if not complete:
                raise ValueError('No complete TRAIN condition tuple for '+source)
            selected = complete[len(complete)//2]
            targets[source] = {'train_id': selected['id'], 'properties': {p: selected['properties'][p] for p in names}}
        protocol['datasets'][dataset] = {'config': str(path.relative_to(ROOT)), 'fitted_config': fitted,
            'counts': {phase: sum(len(read_jsonl(p)) for p in items) for phase, items in paths.items()},
            'initial': str(initial.relative_to(study)), 'initial_weights_sha256': initial_hash,
            'targets': targets, 'training_gate_passed': True}
        for source in [path, ROOT/config['data_audit'], *sum(paths.values(), [])]:
            protocol['files'][str(source.relative_to(ROOT))] = paired.file_hash(source)
        del proposal, flow, records
        print('[PREPARED] '+dataset, flush=True)
    for path in sorted(p for p in frozen.rglob('*') if p.is_file())+sorted((study/'initial').glob('*.pt')):
        protocol['frozen_files'][str(path.relative_to(study))] = paired.file_hash(path)
    protocol['source_identity'] = paired.digest({k: v for k, v in protocol['frozen_files'].items() if k.startswith('versions/')})
    paired.write(study/'protocol.json', protocol)
    return protocol


def worker(study, dataset, version, action):
    protocol = json.loads((study/'protocol.json').read_text())
    verify(study, protocol)
    sys.path.insert(0, str(study/'versions/after/src'))
    from unifiedgroupgen import cli, training
    cli.ROOT = ROOT
    info = protocol['datasets'][dataset]
    run = study/'runs'/dataset/version
    run.mkdir(parents=True, exist_ok=True)
    if action == 'evaluate':
        paired.OPTIMIZED = protocol['source_identity']
        paired.evaluate(study, dataset, version, protocol)
        return
    torch.set_num_threads(4)
    initial = torch.load(study/info['initial'], map_location='cpu', weights_only=False)
    original_run, original_batches = training.run_training, training.batch_indices
    checkpoint = run/'model.pt'
    if checkpoint.exists():
        raise ValueError('Partial run exists: review it before starting a new paired study')
    epoch = 0
    def batches(records, *args, **kwargs):
        nonlocal epoch
        result = original_batches(records, *args, **kwargs)
        if kwargs.get('shuffle', True):
            identities = [[(records[i].get('dataset'), records[i]['id']) for i in batch] for batch in result]
            with (run/'pairing.jsonl').open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'epoch': epoch, 'batch_order_sha256': paired.digest(identities),
                    'rng_sha256': paired.rng_hash(), 'batches': len(result)})+'\n')
            epoch += 1
        return result
    def execute(args, config, records, validation, proposal, flow, *rest):
        if config != info['fitted_config']:
            raise ValueError('Fitted data/scalers do not match the paired initialization')
        proposal.load_state_dict(initial['proposal']); flow.load_state_dict(initial['flow'])
        if paired.weight_hash((proposal, flow)) != info['initial_weights_sha256']:
            raise ValueError('Initial weights differ')
        torch.cuda.manual_seed_all(protocol['seed'])
        paired.restore_rng(initial['rng'])
        config = {**config, 'coordinate_length_scale': protocol['loss_scales_angstrom'][version],
                  'gradient_diagnostics_interval': 100, 'early_stopping_patience': 0,
                  'lr_patience': protocol['epochs']+1}
        return original_run(args, config, records, validation, proposal, flow, *rest)
    training.run_training, training.batch_indices = execute, batches
    sys.argv = [__file__, '--seed', '42', '--threads', '4', 'train', '--config', info['config'],
                '--device', 'cuda', '--epochs', str(protocol['epochs']), '--validation-interval', '5',
                '--no-checkpoint', '--edge-chunk-size', '32768', '--output', str(checkpoint)]
    cli.main()
    final = torch.load(checkpoint, map_location='cpu', weights_only=False)
    if final['epoch']+1 != protocol['epochs']:
        raise ValueError('Unequal training budget')
    paired.write(run/'training_complete.json', {'checkpoint_sha256': paired.file_hash(checkpoint),
        'completed_epochs': final['epoch']+1, 'initial_weights_sha256': initial['weights_sha256']})


def summarize(study, protocol):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = study/'figures'
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 4, figsize=(18, 11), constrained_layout=True)
    gradfig, gradaxes = plt.subplots(3, 2, figsize=(13, 10), constrained_layout=True)
    report = {'state': 'complete', 'datasets': {}, 'formal_loss_config_selected': False,
              'decision': 'Review all raw/common-scale losses, gradients and quality before selecting main15 loss settings',
              'limits': protocol['limits']}
    for row, dataset in enumerate(DATASETS):
        data, traces, gradients = {}, {}, {}
        for version, color in [('before', '#555555'), ('after', '#0072B2')]:
            run = study/'runs'/dataset/version
            data[version] = [json.loads(s) for s in (run/'model.history.jsonl').read_text().splitlines() if s]
            traces[version] = [json.loads(s) for s in (run/'pairing.jsonl').read_text().splitlines() if s]
            gradients[version] = [json.loads(s) for s in (run/'model.gradients.jsonl').read_text().splitlines() if s]
            for column, key in enumerate(('descriptor', 'lattice', 'coordinates', 'common_total')):
                def value(item, phase):
                    loss = item[phase]
                    return loss['descriptor']+loss['lattice']+loss['coordinates']/9 if key == 'common_total' else loss[key]
                history = data[version]
                axes[row, column].plot([r['epoch']+1 for r in history], [value(r, 'train') for r in history],
                                       color=color, label=version+' train')
                valid = [r for r in history if r.get('val')]
                axes[row, column].plot([r['epoch']+1 for r in valid], [value(r, 'val') for r in valid],
                                       '--o', color=color, markersize=3, label=version+' val')
                axes[row, column].set(title=dataset+' / '+key, xlabel='Epoch')
                axes[row, column].grid(alpha=.2); axes[row, column].legend(fontsize=7)
            for key, style in [('shared_lattice_weighted_norm', '-'), ('shared_coordinates_weighted_norm', '--')]:
                gradaxes[row, 0].plot([g['epoch']+g['step']/traces[version][g['epoch']]['batches'] for g in gradients[version]],
                                      [g[key] for g in gradients[version]], style, color=color, label=version+' '+key)
            gradaxes[row, 0].set(title=dataset+' / weighted shared gradients', yscale='symlog', xlabel='Epoch')
            gradaxes[row, 0].legend(fontsize=6)
            gradaxes[row, 1].plot([r['epoch']+1 for r in history], [r['gradient_clipped_fraction'] for r in history],
                                  color=color, label=version)
            gradaxes[row, 1].set(title=dataset+' / clipped batch fraction', ylim=(0, 1.02), xlabel='Epoch')
            gradaxes[row, 1].legend()
        trace_ok = len(traces['before']) == len(traces['after']) == protocol['epochs'] and all(
            a['batch_order_sha256'] == b['batch_order_sha256'] and a['rng_sha256'] == b['rng_sha256']
            for a, b in zip(traces['before'], traces['after']))
        warnings = []
        if not trace_ok:
            warnings.append('Paired batch order or epoch RNG differs')
        for key in ('descriptor', 'lattice', 'coordinates'):
            if data['after'][-1]['val'][key] > data['before'][-1]['val'][key]*1.10:
                warnings.append('Final raw validation '+key+' increased by >10%')
        generations = {v: json.loads((study/'evaluations'/dataset/v/'complete.json').read_text())['jobs']
                       for v in ('before', 'after')}
        for job, old in generations['before'].items():
            new = generations['after'][job]
            for key in ('joint_valid', 'exact_target_group', 'unique', 'novel_against_reference'):
                if new[key]['rate_over_all_attempts'] < old[key]['rate_over_all_attempts']-.10:
                    warnings.append(job+': '+key+' decreased >10 percentage points')
            if new['rule_checks']['target_group_containment']['passed'] != new['generated']:
                warnings.append(job+': target group containment failed or unknown')
        report['datasets'][dataset] = {'batch_order_and_rng_paired': trace_ok, 'warnings': warnings,
            'final_raw_validation': {v: data[v][-1]['val'] for v in ('before', 'after')}, 'generations': generations}
    fig.savefig(figures/'loss_comparison.png', dpi=180); plt.close(fig)
    gradfig.savefig(figures/'gradient_comparison.png', dpi=180); plt.close(gradfig)
    paired.write(study/'comparison.json', report)
    lines = ['# Coordinate-loss scaling study', '', protocol['scope'], '', protocol['limits'], '',
             'Only difference: coordinate_length_scale=1 versus 3 angstrom. Fixed common LR and diagnostics in both arms.',
             'Compare raw components and common_total=descriptor+lattice+coordinates/9; arm-specific total losses are not directly comparable.', '',
             '## Dataset checks', '']
    for name, item in report['datasets'].items():
        lines += ['### '+name, '', 'Paired batches/RNG: '+str(item['batch_order_and_rng_paired']),
                  'Warnings: '+json.dumps(item['warnings']), '']
    lines += ['All per-task standard metrics, confidence intervals and quality.png figures are in evaluations/.',
              'Formal main15 loss settings remain unchanged.']
    (study/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def controller(study, args):
    protocol = prepare(study, args)
    status = {'state': 'queued', 'pid': os.getpid(), 'jobs': {}, 'formal15_started': False}
    paired.write(study/'status.json', status)
    if args.wait_for:
        waiting = (ROOT/args.wait_for).resolve()
        if waiting == study:
            raise ValueError('Cannot wait for this study itself')
        status['waiting_for'] = str(waiting)
        print('[QUEUED] waiting for '+str(waiting), flush=True)
        while True:
            paired.write(study/'status.json', status)
            previous = json.loads((waiting/'status.json').read_text())
            if previous['state'] == 'complete':
                break
            if previous['state'] == 'failed':
                status.update(state='failed', error='Previous GPU study failed; review required')
                paired.write(study/'status.json', status)
                raise RuntimeError(status['error'])
            time.sleep(30)
    try:
        for action in ('train', 'evaluate'):
            for dataset, version in ORDER:
                key = action+'/'+dataset+'/'+version
                marker = (study/'runs'/dataset/version/'training_complete.json' if action == 'train' else
                          study/'evaluations'/dataset/version/'complete.json')
                if marker.exists():
                    if json.loads(marker.read_text())['checkpoint_sha256'] != paired.file_hash(study/'runs'/dataset/version/'model.pt'):
                        raise ValueError('Completed checkpoint changed: '+key)
                    status['jobs'][key] = {'state': 'complete', 'reused_verified': True}
                    continue
                status.update(state='running', current_job=key)
                status['jobs'][key] = {'state': 'running', 'started_at': time.time()}
                paired.write(study/'status.json', status)
                print('[START] '+key, flush=True)
                frozen = study/'versions/after'
                command = [sys.executable, '-u', str(frozen/'scripts/loss_balance_study.py'), '--output', str(study),
                           '--worker', action, '--dataset', dataset, '--version', version]
                with (study/(key.replace('/', '_')+'.log')).open('a', encoding='utf-8') as stream:
                    subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True,
                        env={**os.environ, 'UGG_PROJECT_ROOT': str(ROOT), 'PYTHONPATH': str(frozen/'src'), 'PYTHONIOENCODING': 'utf-8'},
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                status['jobs'][key].update(state='complete', finished_at=time.time())
                print('[DONE] '+key, flush=True)
        summarize(study, protocol)
        status.update(state='complete', finished_at=time.time())
    except BaseException as error:
        status.update(state='failed', error=str(error), finished_at=time.time())
        raise
    finally:
        paired.write(study/'status.json', status)
    if status['state'] == 'complete':
        print('[COMPLETE] '+str(study), flush=True)
        subprocess.run([sys.executable, str(study/'versions/after/scripts/pack_loss_study.py'),
                        '--study', str(study)], cwd=ROOT, check=True,
                       env={**os.environ, 'UGG_PROJECT_ROOT': str(ROOT)})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/loss_balance_20261023')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--conditional-count', type=int, default=128)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--wait-for')
    parser.add_argument('--worker', choices=('train', 'evaluate'))
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--version', choices=('before', 'after'))
    args = parser.parse_args()
    if min(args.epochs, args.count, args.conditional_count, args.steps) < 1:
        parser.error('Budgets must be positive')
    study = (ROOT/args.output).resolve()
    if not study.is_relative_to((ROOT/'artifacts').resolve()):
        parser.error('Study must be inside project artifacts')
    study.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if not args.dataset or not args.version:
            parser.error('Worker requires dataset and version')
        worker(study, args.dataset, args.version, args.worker)
    elif args.prepare_only:
        prepare(study, args)
    else:
        lock = study/'pipeline.lock'
        with lock.open('x', encoding='utf-8') as stream:
            stream.write(str(os.getpid()))
        try:
            controller(study, args)
        finally:
            lock.unlink()


if __name__ == '__main__':
    main()
