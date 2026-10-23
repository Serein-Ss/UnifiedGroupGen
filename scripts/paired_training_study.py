"""Run a frozen, full-data, before/after training study; never starts main15 training."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import argparse
from collections import defaultdict
import hashlib
import importlib.util
import io
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import time
from types import SimpleNamespace
import zipfile
import torch
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASELINE = '26183967082bae4142b785e0306a72c1f5a00134'
OPTIMIZED = '4116dea0e96d7fba39b200ab08ac7979fc486276'
DATASETS = ('mp20', 'c2db51', 'joint')
ORDER = [('mp20', 'before'), ('mp20', 'after'), ('c2db51', 'after'),
         ('c2db51', 'before'), ('joint', 'before'), ('joint', 'after')]


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def file_hash(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            result.update(chunk)
    return result.hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def rate_summary(values):
    """Wilson interval; unknowns remain in the all-attempt denominator."""
    total, known = len(values), sum(value is not None for value in values)
    passed = sum(bool(value) for value in values if value is not None)
    if not total:
        return {'attempted': 0, 'passed': 0, 'unknown': 0, 'rate': None, 'wilson95': None}
    rate, z = passed/total, 1.959963984540054
    center = (rate+z*z/(2*total))/(1+z*z/total)
    radius = z*math.sqrt(rate*(1-rate)/total+z*z/(4*total*total))/(1+z*z/total)
    return {'attempted': total, 'passed': passed, 'unknown': total-known,
            'rate': rate, 'wilson95': [center-radius, center+radius]}


def weight_hash(modules):
    result = hashlib.sha256()
    for module in modules:
        for name, value in module.state_dict().items():
            result.update(name.encode())
            result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def rng_state():
    return {'python': random.getstate(), 'numpy': np.random.get_state(),
            'torch': torch.get_rng_state(), 'cuda': torch.cuda.get_rng_state_all()}


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def rng_hash():
    state = rng_state()
    result = hashlib.sha256(repr((state['python'], state['numpy'][0], state['numpy'][2:])).encode())
    result.update(state['numpy'][1].tobytes())
    for tensor in [state['torch'], *state['cuda']]:
        result.update(tensor.cpu().numpy().tobytes())
    return result.hexdigest()


def source_path(study, version):
    return study/'versions'/version/'src'


def verify_inputs(study, manifest):
    if file_hash(__file__) != manifest['controller_sha256']:
        raise ValueError('Study controller changed after protocol preparation')
    for path, expected in manifest['files'].items():
        if file_hash(ROOT/path) != expected:
            raise ValueError(f'Study input changed: {path}')
    for path, expected in manifest['frozen_files'].items():
        if file_hash(study/path) != expected:
            raise ValueError(f'Frozen study file changed: {path}')


def prepare(study, args):
    if (study/'protocol.json').exists():
        manifest = json.loads((study/'protocol.json').read_text(encoding='utf-8'))
        if (manifest['epochs'], manifest['unconditional_count'], manifest['conditional_count'], manifest['steps']) != (
                args.epochs, args.count, args.conditional_count, args.steps):
            raise ValueError('Existing protocol differs; use its original budgets or a new study directory')
        verify_inputs(study, manifest)
        return manifest
    sys.path.insert(0, str(ROOT/'src'))
    import yaml
    from unifiedgroupgen.cli import build_models, fit_training_config
    from unifiedgroupgen.conditioning import attach_dataset_domains
    from unifiedgroupgen.data import read_jsonl
    from unifiedgroupgen.preflight import check_training_gate
    if not torch.cuda.is_available():
        raise ValueError('The paired study requires the local CUDA GPU')
    torch.set_num_threads(4)
    manifest = {'schema': 1, 'scope': 'Single-seed 30-epoch regression screen, not final convergence proof',
                'epochs': args.epochs, 'seed': 42, 'validation_interval': 5,
                'unconditional_count': args.count, 'conditional_count': args.conditional_count,
                'steps': args.steps, 'sampling_seed': 142, 'guidance': 2.0,
                'primary_checkpoint': 'final_equal_budget', 'inference_version': 'after',
                'commits': {'before': BASELINE, 'after': OPTIMIZED}, 'order': ORDER,
                'gpu': torch.cuda.get_device_name(), 'torch': torch.__version__,
                'datasets': {}, 'files': {}, 'frozen_files': {},
                'interpretation': {'quality_warning_drop_percentage_points': 10,
                    'loss_warning_relative_increase': 0.05,
                    'warning_thresholds_are_diagnostics_not_noninferiority_tests': True,
                    'physical_stability_and_property_hit': 'not assessed',
                    'formal_15_runs': 'not started by this script'}}
    git = ['git', '-c', f'safe.directory={ROOT.as_posix()}']
    for version, revision in manifest['commits'].items():
        blob = subprocess.check_output([*git, 'archive', '--format=zip', revision,
                                       'src/unifiedgroupgen'], cwd=ROOT)
        destination = study/'versions'/version
        destination.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(io.BytesIO(blob)) as archive:
            if any(not (destination/info.filename).resolve().is_relative_to(destination.resolve())
                   for info in archive.infolist()):
                raise ValueError('Archive path escapes frozen source directory')
            archive.extractall(destination)
    # Freeze the shared evaluator as well as both training implementations.
    for name in ('evaluate_checkpoint.py', 'evaluate_generation.py'):
        target = study/'versions/after/scripts'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/'scripts'/name).read_bytes())
    for dataset in DATASETS:
        config_path = ROOT/f'configs/main15/{dataset}_all.yaml'
        config = yaml.safe_load(config_path.read_text(encoding='utf-8'))
        paths = {phase: [ROOT/p for p in config[phase+'_files']] for phase in ('train', 'val', 'test')}
        check_training_gate(config, paths['train'], paths['val'], ROOT)
        records = [r for path in paths['train'] for r in read_jsonl(path)]
        fitted = fit_training_config(config, attach_dataset_domains(records, config))
        torch.manual_seed(42); random.seed(42); np.random.seed(42)
        proposal, flow = build_models(fitted, 'cuda')
        initial_hash = weight_hash((proposal, flow))
        initial = study/'initial'/f'{dataset}.pt'
        initial.parent.mkdir(parents=True, exist_ok=True)
        torch.save({'proposal': proposal.state_dict(), 'flow': flow.state_dict(),
                    'config': fitted, 'rng': rng_state(), 'weights_sha256': initial_hash}, initial)
        targets = {}
        names = [p for p, spec in fitted['properties'].items() if not spec.get('context')]
        for source in sorted({r['dataset'] for r in records}):
            complete = [r for r in records if r['dataset'] == source and
                        all(r['properties'].get(p) is not None and np.isfinite(r['properties'][p]) for p in names)]
            complete.sort(key=lambda r: (r['properties'][names[0]], str(r['id'])))
            if not complete:
                raise ValueError(f'No complete TRAIN property tuple for {source}')
            selected = complete[len(complete)//2]
            targets[source] = {'train_id': selected['id'], 'properties': {p: selected['properties'][p] for p in names}}
        manifest['datasets'][dataset] = {'config': str(config_path.relative_to(ROOT)), 'fitted_config': fitted,
            'counts': {phase: sum(len(read_jsonl(p)) for p in items) for phase, items in paths.items()},
            'initial': str(initial.relative_to(study)), 'initial_weights_sha256': initial_hash,
            'targets': targets, 'training_gate_passed': True}
        for path in [config_path, ROOT/config['data_audit'], *sum(paths.values(), [])]:
            manifest['files'][str(path.relative_to(ROOT))] = file_hash(path)
        del proposal, flow, records
        torch.cuda.empty_cache()
        print(f'[PREPARED] {dataset}: {manifest["datasets"][dataset]["counts"]}', flush=True)
    for path in sorted((study/'versions').rglob('*')) + sorted((study/'initial').glob('*.pt')):
        if path.is_file():
            manifest['frozen_files'][str(path.relative_to(study))] = file_hash(path)
    manifest['controller_sha256'] = file_hash(__file__)
    write(study/'protocol.json', manifest)
    return manifest


def instrument_training(training, initial, expected, trace, resumed, start_epoch=0):
    original_run, original_batches = training.run_training, training.batch_indices
    epoch = start_epoch
    def batches(records, *args, **kwargs):
        nonlocal epoch
        result = original_batches(records, *args, **kwargs)
        if kwargs.get('shuffle', True):
            identities = [[(records[i].get('dataset'), records[i]['id']) for i in batch] for batch in result]
            with trace.open('a', encoding='utf-8') as stream:
                stream.write(json.dumps({'local_epoch': epoch, 'batch_order_sha256': digest(identities),
                    'batches': len(result), 'records': sum(map(len, result)), 'rng_sha256': rng_hash()})+'\n')
            epoch += 1
        return result
    def run(args, config, records, validation, proposal, flow, *rest):
        if config != expected['fitted_config']:
            raise ValueError('The two releases did not fit exactly the same training configuration')
        if not resumed:
            proposal.load_state_dict(initial['proposal'], strict=True)
            flow.load_state_dict(initial['flow'], strict=True)
            if weight_hash((proposal, flow)) != expected['initial_weights_sha256']:
                raise ValueError('Initial model weights do not match the paired protocol')
            restore_rng(initial['rng'])
        return original_run(args, config, records, validation, proposal, flow, *rest)
    training.batch_indices, training.run_training = batches, run


def worker(study, dataset, version, action, resume):
    manifest = json.loads((study/'protocol.json').read_text(encoding='utf-8'))
    verify_inputs(study, manifest)
    sys.path.insert(0, str(source_path(study, version if action in ('train', 'smoke') else 'after')))
    from unifiedgroupgen import cli, training
    cli.ROOT = ROOT
    info = manifest['datasets'][dataset]
    run = study/'runs'/dataset/version
    checkpoint = run/'model.pt'
    if action == 'smoke':
        smoke(study, version, manifest)
    elif action == 'train':
        run.mkdir(parents=True, exist_ok=True)
        initial = torch.load(study/info['initial'], map_location='cpu', weights_only=False)
        completed = torch.load(checkpoint, map_location='cpu', weights_only=False)['epoch']+1 if resume else 0
        trace = run/'pairing.jsonl'
        if trace.exists():
            lines = trace.read_text(encoding='utf-8').splitlines()
            if len(lines) > completed:
                # Preserve interrupted traces, then align the primary trace to saved optimizer/RNG state.
                trace.rename(run/f'pairing_interrupted_{time.time_ns()}.jsonl')
                trace.write_text(''.join(s+'\n' for s in lines[:completed]), encoding='utf-8')
        instrument_training(training, initial, info, trace, resume, completed)
        sys.argv = [__file__, '--seed', '42', 'train', '--config', info['config'], '--device', 'cuda',
                    '--epochs', str(manifest['epochs']), '--validation-interval', '5', '--output', str(checkpoint)]
        if resume:
            sys.argv += ['--resume', str(checkpoint)]
        cli.main()
        final = torch.load(checkpoint, map_location='cpu', weights_only=False)
        if final['epoch']+1 != manifest['epochs']:
            raise ValueError('Training ended before the fixed paired epoch budget')
        write(run/'training_complete.json', {'checkpoint_sha256': file_hash(checkpoint),
            'initial_weights_sha256': initial['weights_sha256'], 'completed_epochs': final['epoch']+1,
            'protocol_sha256': file_hash(study/'protocol.json')})
    else:
        evaluate(study, dataset, version, manifest)


def smoke(study, version, manifest):
    """Separate tiny mixed-kind integration check; never included in study results."""
    from copy import deepcopy
    from unifiedgroupgen import cli, training
    from unifiedgroupgen.conditioning import attach_dataset_domains
    from unifiedgroupgen.data import read_jsonl
    config = deepcopy(manifest['datasets']['joint']['fitted_config'])
    config['model'].update(proposal_hidden=16, proposal_layers=1, heads=4,
                           flow_hidden=16, flow_layers=1, frequencies=2)
    paths = {phase: [ROOT/p for p in config[phase+'_files']] for phase in ('train', 'val', 'test')}
    records = {phase: attach_dataset_domains([r for path in items for r in read_jsonl(path)[:2]], config)
               for phase, items in paths.items()}
    folder = study/'smoke'/version
    folder.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42); random.seed(42); np.random.seed(42)
    proposal, flow = cli.build_models(config, 'cuda')
    initial_hash, initial_rng = weight_hash((proposal, flow)), rng_hash()
    args = SimpleNamespace(device='cuda', resume=None, checkpoint=None, edge_chunk_size=None,
                           validation_interval=1, batch_size=2, epochs=2)
    training.run_training(args, config, records['train'], records['val'], proposal, flow,
                          paths['train'], paths['val'], folder/'model.pt')
    # Test the frozen COMMON sampling and multiprocessing metric APIs separately.
    sys.path.insert(0, str(source_path(study, 'after')))
    fixed = folder/'descriptor.json'
    write(fixed, records['train'][0]['descriptor'])
    sys.argv = [__file__, '--seed', '142', 'sample', '--checkpoint', str(folder/'model.pt'),
                '--device', 'cuda', '--descriptor', str(fixed), '--properties', '{"source_domain":0}',
                '--guidance', '0', '--count', '2', '--steps', '2', '--output', str(folder/'samples')]
    cli.main()
    metrics = load_evaluator(study, 'evaluate_generation')
    reference = folder/'references.jsonl'
    reference.write_text(''.join(json.dumps(r)+'\n' for r in records['test'] if r['dataset'] == 'mp_20'), encoding='utf-8')
    refs = metrics.reference_fingerprints([reference], 'mp_20', folder, True, workers=2)
    rows, report = metrics.evaluate_samples(folder/'samples/samples.jsonl', refs, {}, 'space')
    metrics.plot_job(rows, refs, report, folder/'smoke_quality.png', 'SMOKE ONLY', 'space')
    write(folder/'complete.json', {'scope': 'Tiny integration check only; excluded from paired experiment',
        'initial_weights_sha256': initial_hash, 'initial_rng_sha256': initial_rng, 'metrics': report})


def load_evaluator(study, name):
    path = study/'versions/after/scripts'/f'{name}.py'
    sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    module.ROOT = ROOT
    return module


def evaluate(study, dataset, version, manifest):
    from unifiedgroupgen import cli
    from unifiedgroupgen.data import read_jsonl
    run = study/'runs'/dataset/version
    checkpoint = run/'model.pt'
    output = study/'evaluations'/dataset/version
    output.mkdir(parents=True, exist_ok=True)
    config = manifest['datasets'][dataset]['fitted_config']
    evaluator = load_evaluator(study, 'evaluate_checkpoint')
    heldout = output/'test.json'
    if not heldout.exists():
        sys.argv = [__file__, '--checkpoint', str(checkpoint), '--records', *config['test_files'],
                    '--output', str(heldout), '--device', 'cuda']
        evaluator.main()
    jobs = []
    domains = config.get('dataset_domains') or {('c2db_51' if dataset == 'c2db51' else 'mp_20'): None}
    for source, domain in domains.items():
        context = {'source_domain': domain} if domain is not None else {}
        for mode, count in [('unconditional', manifest['unconditional_count']), ('conditional', manifest['conditional_count'])]:
            target = manifest['datasets'][dataset]['targets'][source]['properties'] if mode == 'conditional' else {}
            folder = output/f'{source}_{mode}'
            jobs.append((source, mode, folder))
            if (folder/'sampling.status.json').exists() and json.loads((folder/'sampling.status.json').read_text())['state'] == 'complete':
                continue
            # Do not overwrite partial attempts. A failed job requires review, not silent retry selection.
            sys.argv = [__file__, '--seed', str(manifest['sampling_seed']), 'sample', '--checkpoint', str(checkpoint),
                '--device', 'cuda', '--kind', 'layer' if source == 'c2db_51' else 'space',
                '--properties', json.dumps({**context, **target}), '--guidance', str(manifest['guidance'] if target else 0),
                '--count', str(count), '--steps', str(manifest['steps']), '--output', str(folder)]
            cli.main()
    metrics = load_evaluator(study, 'evaluate_generation')
    original_hash = metrics.file_hash
    # The metric identity must identify the frozen code actually executed.
    def evaluated_file_hash(path):
        path = Path(path)
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else None
        frozen = study/'versions/after'/relative if relative is not None else None
        return original_hash(frozen if frozen is not None and frozen.is_file() else path)
    metrics.file_hash = evaluated_file_hash
    (study/'reference_metrics').mkdir(exist_ok=True)
    training = {'space': defaultdict(list), 'layer': defaultdict(list)}
    for path in config['train_files']:
        for record in read_jsonl(ROOT/path):
            structure = metrics.record_structure(record)
            training[record['descriptor']['kind']][structure.composition.reduced_formula].append(structure)
    references = {}
    results = {}
    for source, mode, folder in jobs:
        destination = folder/'standard_metrics.json'
        if destination.exists():
            results[folder.name] = json.loads(destination.read_text())
            continue
        if source not in references:
            source_paths = [ROOT/p for p in config['test_files'] if Path(p).parent.name == source]
            if not source_paths:
                raise ValueError(f'No audited TEST files for {source}')
            references[source] = metrics.reference_fingerprints(source_paths, source,
                study/'reference_metrics', True, workers=4)
        kind = 'layer' if source == 'c2db_51' else 'space'
        rows, report = metrics.evaluate_samples(folder/'samples.jsonl', references[source], training[kind], kind)
        import pandas as pd
        pd.DataFrame(rows).to_csv(folder/'metrics.csv', index=False)
        report.update(samples_sha256=file_hash(folder/'samples.jsonl'), checkpoint_sha256=file_hash(checkpoint),
                      protocol_sha256=file_hash(study/'protocol.json'))
        report['rule_checks'] = {key: rate_summary([r.get(key) for r in rows]) for key in
            ('target_group_containment', 'orbit_multiplicity_regular', 'no_coincident_orbits')}
        report['validity_intervals'] = {key: rate_summary([r.get(key) for r in rows]) for key in
            ('joint_valid', 'exact_target_group', 'composition_valid')}
        residuals = [r['metric_residual'] for r in rows if r.get('metric_residual') is not None]
        report['metric_residual'] = {'count': len(residuals), 'nonfinite': sum(not np.isfinite(x) for x in residuals),
                                    'maximum': max(residuals) if residuals else None}
        write(destination, report)
        metrics.plot_job(rows, references[source], report, folder/'quality.png', folder.name, kind)
        results[folder.name] = report
    write(output/'complete.json', {'checkpoint_sha256': file_hash(checkpoint), 'jobs': results,
                                  'inference': OPTIMIZED, 'selection': 'final_equal_budget'})


def summarize(study, manifest):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figures = study/'figures'
    figures.mkdir(exist_ok=True)
    fig, axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    report = {'state': 'complete', 'datasets': {}, 'decision': 'Review the paired evidence before formal main15 training',
              'limits': 'One seed and 30 epochs screen obvious regression only. Conditional property accuracy and stability are unassessed.'}
    for row, dataset in enumerate(DATASETS):
        histories, traces = {}, {}
        for version, color in [('before', '#555555'), ('after', '#0072B2')]:
            run = study/'runs'/dataset/version
            histories[version] = [json.loads(s) for s in (run/'model.history.jsonl').read_text().splitlines() if s]
            traces[version] = [json.loads(s) for s in (run/'pairing.jsonl').read_text().splitlines() if s]
            history = histories[version]
            for column, key in enumerate(('total', 'descriptor', 'flow')):
                def loss(item, phase):
                    values = item[phase]
                    return values['descriptor']+values['flow'] if key == 'total' else values[key]
                axes[row, column].plot([r['epoch']+1 for r in history], [loss(r, 'train') for r in history],
                                      color=color, label=version+' train', linewidth=1.4)
                valid = [r for r in history if r.get('val')]
                axes[row, column].plot([r['epoch']+1 for r in valid], [loss(r, 'val') for r in valid],
                                      '--o', color=color, label=version+' val', markersize=3)
                axes[row, column].set(title=f'{dataset} / {key}', xlabel='Epoch', ylabel='Loss')
                axes[row, column].grid(alpha=.2)
                axes[row, column].legend(fontsize=8)
        paired = len(traces['before']) == len(traces['after']) == manifest['epochs'] and all(
            a['batch_order_sha256'] == b['batch_order_sha256'] and a['rng_sha256'] == b['rng_sha256']
            for a, b in zip(traces['before'], traces['after']))
        b, a = histories['before'][-1]['val_loss'], histories['after'][-1]['val_loss']
        report['datasets'][dataset] = {'batch_order_and_epoch_rng_paired': paired,
            'final_val_before': b, 'final_val_after': a, 'relative_val_change': (a-b)/b,
            'loss_warning': a > b*1.05, 'generations': {}}
        for version in ('before', 'after'):
            result = json.loads((study/'evaluations'/dataset/version/'complete.json').read_text())
            report['datasets'][dataset]['generations'][version] = result['jobs']
        warnings = []
        for job, old in report['datasets'][dataset]['generations']['before'].items():
            new = report['datasets'][dataset]['generations']['after'][job]
            for key in ('joint_valid', 'exact_target_group'):
                old_rate, new_rate = old[key]['rate_over_all_attempts'], new[key]['rate_over_all_attempts']
                if new_rate < old_rate-.10:
                    warnings.append(f'{job}: {key} decreased by more than 10 percentage points')
            if new['rule_checks']['target_group_containment']['passed'] != new['generated']:
                warnings.append(f'{job}: generated structures failed target group containment')
        report['datasets'][dataset]['quality_warnings'] = warnings
    fig.savefig(figures/'paired_loss_curves.png', dpi=180)
    plt.close(fig)
    summary_rows = []
    for dataset, item in report['datasets'].items():
        for version, jobs in item['generations'].items():
            for job, metrics in jobs.items():
                summary_rows.append({'dataset': dataset, 'version': version, 'job': job,
                    'attempted': metrics['attempted'], 'generated': metrics['generated'],
                    **{key: metrics[key]['rate_over_all_attempts'] for key in
                       ('joint_valid', 'exact_target_group', 'unique', 'novel_against_reference')},
                    'target_group_containment': metrics['rule_checks']['target_group_containment']['rate'],
                    'cov_recall': metrics['coverage'].get('cov_recall'),
                    'cov_precision': metrics['coverage'].get('cov_precision')})
    import pandas as pd
    frame = pd.DataFrame(summary_rows)
    frame.to_csv(study/'generation_comparison.csv', index=False)
    fig, axes = plt.subplots(1, 4, figsize=(19, 5), constrained_layout=True)
    for ax, key in zip(axes, ('target_group_containment', 'exact_target_group', 'joint_valid', 'cov_recall')):
        pivot = frame.pivot(index=['dataset', 'job'], columns='version', values=key)
        if not pivot.notna().any().any():
            ax.set(title=key, ylim=(0, 1))
            ax.text(.5, .5, 'Unavailable: no valid fingerprints', ha='center', va='center', transform=ax.transAxes)
            continue
        pivot[['before', 'after']].plot.bar(ax=ax, color=['#555555', '#0072B2'], width=.75)
        ax.set(title=key, ylim=(0, 1), xlabel='Dataset / task')
        ax.tick_params(axis='x', labelsize=7, rotation=70)
        if key != 'cov_recall':
            for container, version in zip(ax.containers, ('before', 'after')):
                errors = []
                for dataset, job in pivot.index:
                    metric = report['datasets'][dataset]['generations'][version][job]
                    entry = metric['rule_checks'][key] if key == 'target_group_containment' else metric['validity_intervals'][key]
                    errors.append((entry['rate']-entry['wilson95'][0], entry['wilson95'][1]-entry['rate']))
                ax.errorbar([bar.get_x()+bar.get_width()/2 for bar in container],
                            [bar.get_height() for bar in container], yerr=np.maximum(np.array(errors).T, 0),
                            fmt='none', ecolor='black', elinewidth=.7, capsize=2)
    fig.savefig(figures/'paired_generation_quality.png', dpi=180)
    plt.close(fig)
    write(study/'comparison.json', report)
    lines = ['# Paired acceleration regression study', '',
        'Full repaired TRAIN/VAL splits; same saved initialization, seed 42, 30 epochs, BF16 and optimizer settings.',
        'Final equal-budget checkpoints are evaluated with one common inference implementation.', '',
        '## Training', '', '| Dataset | Paired batches / RNG | Before val | After val | Change |',
        '|---|---|---:|---:|---:|']
    for dataset, item in report['datasets'].items():
        lines.append(f'| {dataset} | {item["batch_order_and_epoch_rng_paired"]} | {item["final_val_before"]:.5f} | '
                     f'{item["final_val_after"]:.5f} | {item["relative_val_change"]:+.2%} |')
    lines += ['', '## Generation', '', frame.to_markdown(index=False), '',
        'PNG figures: figures/paired_loss_curves.png and figures/paired_generation_quality.png.', '',
        'Per-attempt geometry, exact SG/LG, chemistry, fingerprint, uniqueness/novelty and distribution metrics are in evaluations/.',
        'Failures and unknown identifiers are retained in all-attempt denominators. Reference fingerprints use full TEST; novelty uses full same-kind TRAIN.',
        'One seed and 30 epochs do not establish final convergence or statistical noninferiority. No independent property predictor or physical verification was used.',
        'The 15 formal experiments are not launched automatically.']
    (study/'REPORT.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def controller(study, args):
    manifest = prepare(study, args)
    status = {'state': 'running', 'pid': os.getpid(), 'started_at': time.time(), 'jobs': {}}
    try:
        for action in ('train', 'evaluate'):
            for dataset, version in ORDER:
                key = f'{action}/{dataset}/{version}'
                marker = (study/'runs'/dataset/version/'training_complete.json' if action == 'train'
                          else study/'evaluations'/dataset/version/'complete.json')
                checkpoint = study/'runs'/dataset/version/'model.pt'
                if marker.exists():
                    if json.loads(marker.read_text())['checkpoint_sha256'] != file_hash(checkpoint):
                        raise ValueError(f'Checkpoint changed after completion: {key}')
                    status['jobs'][key] = {'state': 'complete', 'reused_verified': True}
                    continue
                command = [sys.executable, '-u', str(Path(__file__).resolve()), '--output', str(study),
                           '--worker', action, '--dataset', dataset, '--version', version]
                if action == 'train' and checkpoint.exists():
                    command.append('--resume')
                status.update(current_job=key)
                status['jobs'][key] = {'state': 'running', 'started_at': time.time()}
                write(study/'status.json', status)
                print('[START] '+key, flush=True)
                with (study/(key.replace('/', '_')+'.log')).open('a', encoding='utf-8') as stream:
                    subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True,
                        env={**os.environ, 'PYTHONPATH': str(source_path(study, 'after')), 'PYTHONIOENCODING': 'utf-8'},
                        creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
                status['jobs'][key].update(state='complete', finished_at=time.time())
                write(study/'status.json', status)
                print('[DONE] '+key, flush=True)
        summarize(study, manifest)
        status.update(state='complete', finished_at=time.time())
    except BaseException as error:
        status.update(state='failed', error=str(error), finished_at=time.time())
        raise
    finally:
        write(study/'status.json', status)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/paired_training_20261023')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--count', type=int, default=256)
    parser.add_argument('--conditional-count', type=int, default=128)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--worker', choices=['train', 'evaluate', 'smoke'])
    parser.add_argument('--dataset', choices=DATASETS)
    parser.add_argument('--version', choices=['before', 'after'])
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    if min(args.epochs, args.count, args.conditional_count, args.steps) < 1:
        parser.error('All budgets must be positive')
    study = (ROOT/args.output).resolve()
    if not study.is_relative_to((ROOT/'artifacts').resolve()):
        parser.error('Study output must be inside this project artifacts directory')
    study.mkdir(parents=True, exist_ok=True)
    if args.worker:
        if not args.dataset or not args.version:
            parser.error('--worker requires --dataset and --version')
        worker(study, args.dataset, args.version, args.worker, args.resume)
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
