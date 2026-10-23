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
