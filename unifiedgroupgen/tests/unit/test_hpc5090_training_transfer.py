"""Training archives preserve provenance and refuse corrupt or conflicting restores."""
import hashlib
import json
from pathlib import Path
import zipfile

import pytest

from submit_files.transfer_hpc5090_training import MODELS, export_training, restore_training


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def sidecar(path):
    path.with_suffix('.zip.sha256').write_text(hashlib.sha256(path.read_bytes()).hexdigest()+'  '+path.name)


@pytest.fixture
def bundle(tmp_path):
    root = tmp_path/'server'
    folder = root/'intermediate/hpc_runs/hpc5090_comp_v1'
    campaign = folder/'campaign.json'
    write(campaign, {'name': 'hpc5090_comp_v1', 'revision': 'frozen-code', 'root': str(root)})
    write(folder/'source/configs/experiments_hpc5090.yaml', {'composition_probability': .5})
    source = folder/'source/unifiedgroupgen/model.py'
    source.parent.mkdir(parents=True)
    source.write_text('# original training implementation\n')
    (root/'.env').write_text('ENV_SECRET')
    pipelines = {}
    for model in MODELS:
        run = root/'outputs'/model/'hpc5090_comp_v1'
        write(run/'model.status.json', {'state': 'complete'})
        (run/'model.best.pt').write_bytes(('BEST '+model).encode())
        (run/'model.pt').write_bytes(('RESUME '+model).encode())
        (run/'model.history.jsonl').write_text('{"epoch": 0}\n')
        (run/'train.log').write_text('training finished')
        write(run/'generate/large_results.json', {'do_not_duplicate': True})
        write(run/'evaluate/summary.json', {'do_not_duplicate': True})
        write(folder/'state'/f'train_{model}.json', {'state': 'complete'})
        log = folder/'logs'/f'ugg_train_{model}-123.out'
        log.parent.mkdir(exist_ok=True);log.write_text('scheduler stdout')
        checkpoint_hash = hashlib.sha256((run/'model.best.pt').read_bytes()).hexdigest()
        pipeline = {'identity': {'jobs': [{'checkpoint_sha256': checkpoint_hash}]}}
        write(folder/'evaluation'/model/'pipeline.json', pipeline)
        pipelines[model] = pipeline
    output = export_training(root, campaign, tmp_path/'training.zip')
    results = tmp_path/'all19.zip'
    with zipfile.ZipFile(results, 'w') as archive:
        archive.writestr('manifest.json', json.dumps({'campaign': 'hpc5090_comp_v1', 'source_revision': 'frozen-code'}))
        for model, pipeline in pipelines.items():
            archive.writestr(f'models/{model}/campaign_evaluation/pipeline.json', json.dumps(pipeline))
    sidecar(results)
    return root, output, results


def test_restore_matches_eval_weights_and_is_repeatable_without_overwriting(bundle, tmp_path):
    server, archive, results = bundle
    local = tmp_path/'local'
    with zipfile.ZipFile(archive) as data:
        assert data.testzip() is None
        assert not any('/generate/' in n or '/evaluate/' in n or n.endswith('.env') for n in data.namelist())
    restored = restore_training(local, archive, results)
    assert restored['matched_to_local_evaluation_archive']
    before = {}
    for model in MODELS:
        relative = Path('outputs')/model/'hpc5090_comp_v1/model.best.pt'
        assert (local/relative).read_bytes() == (server/relative).read_bytes()
        before[relative] = (local/relative).stat().st_mtime_ns
    assert (local/'intermediate/hpc_runs/hpc5090_comp_v1/source/configs/experiments_hpc5090.yaml').exists()
    second = restore_training(local, archive, results)
    assert second['files_restored'] == 0 and second['identical_files_skipped'] > 0
    assert all((local/p).stat().st_mtime_ns == modified for p, modified in before.items())


def test_conflicting_weight_stops_before_any_other_file_is_restored(bundle, tmp_path):
    _, archive, results = bundle
    local = tmp_path/'local'
    weight = local/'outputs'/MODELS[0]/'hpc5090_comp_v1/model.best.pt'
    weight.parent.mkdir(parents=True);weight.write_bytes(b'KEEP_EXISTING_WEIGHT')
    with pytest.raises(FileExistsError, match='nothing imported'):
        restore_training(local, archive, results)
    assert weight.read_bytes() == b'KEEP_EXISTING_WEIGHT'
    assert not (local/'intermediate').exists()
    assert not (local/'outputs'/MODELS[1]).exists()


def test_mismatched_evaluation_checkpoint_stops_restore(bundle, tmp_path):
    _, archive, _ = bundle
    results = tmp_path/'different_results.zip'
    with zipfile.ZipFile(results, 'w') as data:
        data.writestr('manifest.json', json.dumps({'campaign': 'hpc5090_comp_v1', 'source_revision': 'frozen-code'}))
        data.writestr(f'models/{MODELS[0]}/campaign_evaluation/pipeline.json',
                      json.dumps({'identity': {'jobs': [{'checkpoint_sha256': 'wrong-weights'}]}}))
    sidecar(results)
    local = tmp_path/'local'
    with pytest.raises(ValueError, match='local evaluated checkpoint'):
        restore_training(local, archive, results)
    assert not local.exists()


def test_export_refuses_changed_best_checkpoint(bundle, tmp_path):
    root, _, _ = bundle
    (root/'outputs'/MODELS[0]/'hpc5090_comp_v1/model.best.pt').write_bytes(b'OTHER_CHECKPOINT')
    output = tmp_path/'must_not_exist.zip'
    with pytest.raises(ValueError, match='differs from evaluated weights'):
        export_training(root, root/'intermediate/hpc_runs/hpc5090_comp_v1/campaign.json', output)
    assert not output.exists() and not output.with_suffix('.zip.partial').exists()


@pytest.mark.parametrize('change', ['checksum', 'unsafe_path'])
def test_corrupt_or_unsafe_archive_never_writes_project_files(bundle, tmp_path, change):
    _, original, _ = bundle
    archive = tmp_path/'changed.zip'
    if change == 'checksum':
        archive.write_bytes(original.read_bytes()+b'changed')
        archive.with_suffix('.zip.sha256').write_bytes(original.with_suffix('.zip.sha256').read_bytes())
    else:
        with zipfile.ZipFile(original) as source, zipfile.ZipFile(archive, 'w') as target:
            manifest = json.loads(source.read('training_manifest.json'))
            for name in source.namelist():
                if name != 'training_manifest.json': target.writestr(name, source.read(name))
            name = f'outputs/{MODELS[0]}/hpc5090_comp_v1/../../../../.env'
            content = b'UNSAFE'
            manifest['files'][name] = {'sha256': hashlib.sha256(content).hexdigest(), 'bytes': len(content), 'mtime_ns': 0}
            target.writestr(name, content)
            target.writestr('training_manifest.json', json.dumps(manifest))
        sidecar(archive)
    local = tmp_path/'local'
    with pytest.raises(ValueError, match='checksum mismatch|Unsafe archive path'):
        restore_training(local, archive)
    assert not local.exists()
