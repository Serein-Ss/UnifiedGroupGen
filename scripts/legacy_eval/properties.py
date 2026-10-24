"""CPU bridge from UnifiedGroupGen CIFs to frozen CGDiT FE/BG predictors."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def predictors_at(root, seed, backend):
    import numpy as np
    import torch
    import yaml
    result = {}
    for short, name, expected in [('fe', 'formation_energy', 'formation_energy_per_atom'),
                                  ('bg', 'band_gap', 'band_gap')]:
        run = root/short/f'seed{seed}'
        checkpoint = backend.resolve_checkpoint(run)
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        hp = saved['hyper_parameters']
        if hp.get('target_prop') != expected:
            raise ValueError(f'{checkpoint} target is not {expected}')
        config_path = next((p for p in (run/'hparams.yaml', run/'.hydra/config.yaml') if p.exists()), None)
        if config_path is None:
            raise ValueError(f'Missing predictor training data configuration: {run}')
        data = yaml.safe_load(config_path.read_text(encoding='utf-8'))['data']
        settings = {k: data[k] for k in ('niggli', 'primitive', 'graph_method')}
        if not all(isinstance(settings[k], bool) for k in ('niggli', 'primitive')):
            raise ValueError('Predictor graph settings must be resolved booleans')
        info = {'checkpoint': str(checkpoint.resolve()), 'sha256': digest(checkpoint),
                'target': expected, 'seed': seed, 'graph_settings': settings,
                'target_mean': hp.get('target_mean', 0.), 'target_std': hp.get('target_std', 1.),
                'training_dataset': data.get('root_path'), 'training_max_atoms': data.get('max_atoms')}
        if float(info['target_std']) <= 0:
            raise ValueError('Invalid predictor target scaling')
        pred_file, truth_file = run/'test_preds.npy', run/'test_targets.npy'
        if not pred_file.exists() or not truth_file.exists():
            raise ValueError(f'Predictor final-test arrays are missing in {run}; select a completed predictor run')
        if pred_file.exists() and truth_file.exists():
            pred, truth = np.load(pred_file).reshape(-1), np.load(truth_file).reshape(-1)
            if pred.shape != truth.shape:
                raise ValueError('Predictor test prediction/target shapes differ')
            finite = np.isfinite(pred)&np.isfinite(truth)
            info['saved_test_array_agreement'] = {
                'requested': len(pred), 'finite': int(finite.sum()),
                'mae': float(np.abs(pred[finite]-truth[finite]).mean()) if finite.any() else None,
                'rmse': float(np.sqrt(np.square(pred[finite]-truth[finite]).mean())) if finite.any() else None,
                'predictions_sha256': digest(pred_file), 'targets_sha256': digest(truth_file)}
        result[name] = info
    if result['formation_energy']['graph_settings'] != result['band_gap']['graph_settings']:
        raise ValueError('FE/BG graph preprocessing differs; shared graph inference is not valid')
    return result


def crystal_array(structure):
    import numpy as np
    return {'frac_coords': np.asarray(structure.frac_coords),
            'atom_types': np.asarray(structure.atomic_numbers),
            'lengths': np.asarray(structure.lattice.abc), 'angles': np.asarray(structure.lattice.angles)}


def infer_items(items, predictors, backend, batch_size):
    import numpy as np
    settings = next(iter(predictors.values()))['graph_settings']
    for start in range(0, len(items), 128):
        block = items[start:start+128]
        selected = [r for r in block if r.get('_crystal') is not None]
        graphs, indices, errors = backend.build_graphs([r['_crystal'] for r in selected], num_workers=1, **settings)
        for row, error in zip(selected, errors):
            row['graph_error'] = error or ''
        for name, info in predictors.items():
            values = backend.predict_graphs(info['checkpoint'], info['target'], graphs, indices,
                                             len(selected), batch_size, 'cpu')
            for row, value in zip(selected, values):
                row[name] = float(value) if np.isfinite(value) else None
        print(json.dumps({'phase': 'property_predictions', 'completed': start+len(block),
                          'requested': len(items)}), flush=True)
    return [{k: v for k, v in r.items() if k != '_crystal'} for r in items]


def summarize(rows, reference, targets, tolerances):
    import numpy as np
    from scipy.stats import wasserstein_distance
    result = {'attempted': len(rows), 'properties': {}, 'targets_requested': targets,
              'targets_unassessed': sorted(set(targets)-set(tolerances))}
    for name, tolerance in tolerances.items():
        values = np.array([r[name] for r in rows if r.get(name) is not None], dtype=float)
        refs = np.array([r[name] for r in reference if r.get(name) is not None], dtype=float)
        p = {'predicted': len(values), 'prediction_rate': len(values)/len(rows) if rows else None}
        if len(values):
            p.update(mean=float(values.mean()), std=float(values.std()), median=float(np.median(values)),
                     p05=float(np.quantile(values, .05)), p95=float(np.quantile(values, .95)),
                     wasserstein_to_same_predictor_test=float(wasserstein_distance(values, refs)) if len(refs) else None)
        if name in targets:
            errors = np.abs(values-float(targets[name])); hits = int((errors <= tolerance).sum())
            p.update(target=targets[name], tolerance=tolerance, hits=hits,
                     hit_rate_all=hits/len(rows) if rows else None,
                     hit_rate_predicted=hits/len(values) if len(values) else None,
                     target_mae=float(errors.mean()) if len(values) else None)
        result['properties'][name] = p
    if targets and not result['targets_unassessed']:
        hits = sum(all(r.get(k) is not None and abs(r[k]-v) <= tolerances[k] for k, v in targets.items()) for r in rows)
        result['all_requested_targets_joint_hit_rate'] = hits/len(rows) if rows else None
    else:
        result['all_requested_targets_joint_hit_rate'] = None
    return result


def plot_distributions(rows, reference, targets, output, dataset):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), constrained_layout=True)
    for ax, (name, label) in zip(axes, [('formation_energy', 'Predicted formation energy (eV/atom)'),
                                      ('band_gap', 'Predicted band gap (eV)')]):
        for data, title, color in [(reference, 'Test reference', '#d68939'), (rows, 'Generated', '#437da6')]:
            values = np.sort([r[name] for r in data if r.get(name) is not None])
            if len(values):
                ax.step(values, np.arange(1, len(values)+1)/len(values), where='post',
                        color=color, label=f'{title}, n={len(values)}')
        if name in targets:
            ax.axvline(targets[name], color='#64727b', ls='--', lw=1, label='Requested target')
        ax.set(xlabel=label, ylabel='Cumulative fraction', ylim=(0, 1.02)); ax.legend(fontsize=8)
    suffix = ' | exploratory 3D-surrogate transfer' if dataset != 'mp_20' else ''
    fig.suptitle(f'{dataset}: {len(rows)} attempts{suffix}')
    fig.savefig(output, dpi=240, facecolor='white'); plt.close(fig)


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--cgdit-root', type=Path, required=True)
    parser.add_argument('--predictor-root', type=Path, required=True)
    parser.add_argument('--predictor-seed', type=int, default=123)
    parser.add_argument('--samples', type=Path)
    parser.add_argument('--dataset', choices=['mp_20', 'mpts_52', 'c2db_51'])
    parser.add_argument('--reference-files', nargs='+', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference-output', type=Path)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--inventory', action='store_true')
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error('batch-size must be positive')
    root = args.project_root.resolve()
    os.environ['PROJECT_ROOT'] = str(args.cgdit_root.resolve())
    sys.path[:0] = [str(args.cgdit_root.resolve()), str(root/'src'), str(root/'scripts')]
    import pandas as pd
    import torch
    from cgdit.evaluation import generated_properties as backend
    torch.set_num_threads(1)
    predictors = predictors_at(args.predictor_root, args.predictor_seed, backend)
    predictor_code = {name: digest(args.cgdit_root/name) for name in
                      ('cgdit/evaluation/generated_properties.py', 'cgdit/prop_models/gnn_models/m3gnet.py',
                       'cgdit/common/data_utils.py', 'cgdit/prop_models/gnn_models/_layers.py')}
    args.output.mkdir(parents=True, exist_ok=True)
    if args.inventory:
        inventory = {'predictors': predictors, 'code_sha256': predictor_code}
        (args.output/'predictors.json').write_text(json.dumps(inventory, indent=2), encoding='utf-8')
        print(json.dumps(inventory, indent=2), flush=True)
        return
    if not args.samples or not args.dataset or not args.reference_files or not args.reference_output:
        parser.error('Prediction needs samples, dataset, reference-files and reference-output')
    from evaluate_generation import record_structure
    from unifiedgroupgen.data import read_jsonl
    from pymatgen.io.cif import CifParser
    records = read_jsonl(args.samples)
    if len({r['index'] for r in records}) != len(records):
        raise ValueError('Duplicate sample attempt indices')
    targets = records[0].get('properties_requested', {}) if records else {}
    if any(r.get('properties_requested', {}) != targets for r in records):
        raise ValueError('A property job requires one fixed target tuple')
    targets = {k: v for k, v in targets.items() if k != 'source_domain' and v is not None}
    identity = {'samples_sha256': digest(args.samples), 'predictors': predictors, 'predictor_code_sha256': predictor_code,
                'reference_sha256': [digest(p) for p in args.reference_files], 'dataset': args.dataset,
                'code_sha256': digest(__file__)}
    args.reference_output.mkdir(parents=True, exist_ok=True)
    cache = args.reference_output/'predictions.json'
    reference_identity = {k: identity[k] for k in
                          ('predictors', 'predictor_code_sha256', 'reference_sha256', 'dataset', 'code_sha256')}
    if cache.exists():
        saved = json.loads(cache.read_text(encoding='utf-8'))
        if saved['identity'] != reference_identity:
            raise ValueError('Reference property cache belongs to another evaluator or test split')
        reference = saved['rows']
    else:
        reference = []
        for path in args.reference_files:
            for r in read_jsonl(path):
                if r['dataset'] != args.dataset:
                    continue
                row = {'dataset': args.dataset, 'id': str(r['id']),
                       **{'true_'+name: r['properties'].get(name)
                          for name in ('formation_energy', 'band_gap')}}
                try:
                    row['_crystal'] = crystal_array(record_structure(r))
                except (ValueError, RuntimeError, KeyError) as error:
                    row['parse_error'] = str(error)
                reference.append(row)
        reference = infer_items(reference, predictors, backend, args.batch_size)
        cache.write_text(json.dumps({'identity': reference_identity, 'rows': reference}, allow_nan=False), encoding='utf-8')
    rows = []
    for r in records:
        row = {'job': args.dataset+'_explicit', 'index': r['index'], 'generation_status': r['status']}
        if r['status'] == 'generated':
            cif = args.samples.parent/f"sample_{r['index']:05d}.cif"
            try:
                row['input_cif_sha256'] = digest(cif)
                row['_crystal'] = crystal_array(CifParser(cif, frac_tolerance=0).parse_structures(primitive=False)[0])
            except (ValueError, RuntimeError, OSError) as error:
                row['parse_error'] = str(error)
        rows.append(row)
    rows = infer_items(rows, predictors, backend, args.batch_size)
    tolerances = {'formation_energy': .06, 'band_gap': .45}
    report = {**summarize(rows, reference, targets, tolerances), 'identity': identity, 'state': 'complete',
              'units': {'formation_energy': 'eV/atom', 'band_gap': 'eV'},
              'domain_status': 'MP20-trained external surrogate; domain calibration not established for '
                               'MPTS52/C2DB or unrelaxed generated structures; not DFT/stability.',
              'energy_reference': 'Predictor training convention; C2DB formation-energy target compatibility unverified.'}
    report['reference_label_agreement'] = {}
    for name in tolerances:
        pairs = [(r[name], r['true_'+name]) for r in reference
                 if r.get(name) is not None and r.get('true_'+name) is not None]
        report['reference_label_agreement'][name] = {
            'pairs': len(pairs), 'mae': sum(abs(a-b) for a, b in pairs)/len(pairs) if pairs else None,
            'scope': 'Label agreement only; predictor holdout membership and energy/DFT convention compatibility not established.'}
    evaluator = 'CGDiT-M3GNet-'+str(args.predictor_seed)+'-'+hashlib.sha256(json.dumps(predictors, sort_keys=True).encode()).hexdigest()
    metadata = {'evaluator': evaluator, 'reference_evaluator': evaluator, 'units': report['units'],
                'reference_test_sha256': identity['reference_sha256'], 'predictors': predictors,
                'predictor_code_sha256': predictor_code,
                'property_tolerances': {**tolerances, 'energy_above_hull': .03},
                'domain_status': report['domain_status'], 'energy_reference_note': report['energy_reference']}
    pd.DataFrame(rows).to_csv(args.output/'assessments.csv', index=False)
    pd.DataFrame(reference).to_csv(args.output/'reference_assessments.csv', index=False)
    (args.output/'assessment_metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    plot_distributions(rows, reference, targets, args.output/'property_distributions.png', args.dataset)


if __name__ == '__main__':
    main()
