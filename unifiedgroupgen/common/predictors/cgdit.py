"""Frozen external CGDiT predictor adapter; no historical campaign commands."""
import hashlib
import json
from pathlib import Path
from unifiedgroupgen.common.configuration import resolve_path

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
        run = resolve_path(root/short/f'seed{seed}',root)
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
