import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def pilot_module():
    path = Path(__file__).resolve().parents[1]/'scripts/train_timing_pilot.py'
    spec = importlib.util.spec_from_file_location('train_timing_pilot', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pilot_command_runs_one_model_for_five_epochs_and_forwards_global_threads(tmp_path):
    module = pilot_module()
    args = SimpleNamespace(threads=2, device='cuda', checkpoint=False, edge_chunk_size=32768)
    job = module.command(tmp_path/'mp20_all.yaml', tmp_path, args)
    assert job[job.index('--epochs')+1] == '5'
    assert job[job.index('--threads')+1] == '2' and job.index('--threads') < job.index('train')
    assert job.count('--config') == 1 and '--no-checkpoint' in job
    assert not any('train_main15' in arg for arg in job)


def test_pilot_summary_separates_phases_and_keeps_cold_epoch(tmp_path):
    module = pilot_module()
    rows = [{'epoch': i, 'validated': i == 4, 'val_loss': 2. if i == 4 else None,
             'training_seconds': 10. if i == 0 else 2., 'validation_seconds': 3. if i == 4 else 0.,
             'checkpoint_seconds': .5, 'other_seconds': .1, 'wall_seconds': 10.6 if i == 0 else 5.6 if i == 4 else 2.6,
             'peak_cuda_bytes': 2**30, 'peak_cuda_reserved_bytes': 2**31} for i in range(5)]
    (tmp_path/'model.history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report = module.summarize(tmp_path, 50.)
    assert report['warm_training_median_epochs_2_to_4'] == 2.
    assert report['totals']['training_seconds'] == 18.
    assert report['totals']['validation_seconds'] == 3.
    assert report['totals']['checkpoint_seconds'] == 2.5
    assert len(report['rows']) == 5 and (tmp_path/'timing.csv').exists()
    rows.pop()
    (tmp_path/'model.history.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='five completed epochs'):
        module.summarize(tmp_path, 50.)


def test_pilot_forwards_batch_and_edge_budget_without_changing_config(tmp_path):
    module = pilot_module()
    args = SimpleNamespace(threads=4, device='cuda', checkpoint=False, edge_chunk_size=32768,
                           batch_size=128, edge_budget=400000, coordinate_length_scale=3.)
    job = module.command(tmp_path/'mp20_all.yaml', tmp_path, args)
    assert job[job.index('--batch-size')+1] == '128'
    assert job[job.index('--edge-budget')+1] == '400000'
    assert job[job.index('--coordinate-length-scale')+1] == '3.0'
    assert job[job.index('--epochs')+1] == '5' and '--resume' not in job


def test_summary_checks_scale_and_exports_component_losses(tmp_path):
    module = pilot_module()
    rows = [{'epoch': i, 'validated': i == 4, 'val_loss': 6. if i == 4 else None,
        'training_seconds': 2., 'validation_seconds': 1. if i == 4 else 0.,
        'checkpoint_seconds': .1, 'other_seconds': .1, 'wall_seconds': 3.2 if i == 4 else 2.2,
        'peak_cuda_bytes': 0, 'peak_cuda_reserved_bytes': 0, 'coordinate_length_scale_angstrom': 3.,
        'train_loss': 6., 'train': {'descriptor': 1., 'lattice': 2., 'coordinates': 27., 'coordinates_dimensionless': 3.},
        'train_batches': 2, 'mean_actual_batch': 2., 'train_records_per_second': 2.,
        'runtime_options': {'batch_size': 4, 'edge_budget': 400000, 'coordinate_length_scale': 3.}}
        for i in range(5)]
    path = tmp_path/'model.history.jsonl'
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    report = module.summarize(tmp_path, 15., expected_scale=3.)
    assert report['coordinate_loss_weight'] == pytest.approx(1/9)
    assert report['rows'][0]['train_coordinates'] == 27. and report['rows'][0]['train_coordinates_dimensionless'] == 3.
    assert report['runtime_options']['batch_size'] == 4
    rows[0]['coordinate_length_scale_angstrom'] = 1.
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError, match='scale differs'):
        module.summarize(tmp_path, 15., expected_scale=3.)
