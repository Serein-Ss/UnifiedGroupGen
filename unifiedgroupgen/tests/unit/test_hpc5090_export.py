"""Portable export tests: completion gates, provenance, exact CIFs and untouched inputs."""
import csv
import hashlib
import io
import json
from pathlib import Path
import zipfile

import pytest
from submit_files import export_hpc5090 as exporter


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def completed(tmp_path):
    folder = tmp_path/'campaign'
    manifest = folder/'campaign.json'
    write(manifest, {'name': 'fixture', 'revision': 'original-code', 'folder': str(folder),
                     'legacy_models': list(exporter.DEFAULT_MODELS)})
    (folder/'.env').write_text('SECRET=must-not-export')
    for model in exporter.DEFAULT_MODELS:
        generation = tmp_path/'outputs'/model/'generate/fixture_dng'
        metrics = tmp_path/'outputs'/model/'evaluate/fixture_dng'
        generation.mkdir(parents=True); metrics.mkdir(parents=True)
        cif = 'data_exact\n_chemical_formula_sum C\n'
        checksum = hashlib.sha256(cif.encode()).hexdigest()
        rows = [{'index': 0, 'status': 'failed', 'reason': 'original failure'},
                {'index': 1, 'status': 'generated', 'cif': cif, 'cif_sha256': checksum, 'case_id': 'case', 'candidate': 0},
                {'index': 2, 'status': 'generated', 'cif': cif, 'cif_sha256': checksum, 'case_id': 'case', 'candidate': 1}]
        ledger = generation/'samples.jsonl'
        ledger.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
        (metrics/'samples.csv').write_text('index,basic_valid\n0,False\n1,True\n2,False\n')
        write(metrics/'summary.json', {'requested': 3, 'generated': 2, 'failed': 1})
        evaluation = folder/'evaluation'/model
        write(evaluation/'pipeline.json', {'state': 'complete',
            'identity': {'jobs': [{'name': 'fixture_dng', 'checkpoint_sha256': 'original-weights'}]},
            'jobs': [{'name': model+'/fixture_dng', 'state': 'complete', 'directory': str(generation),
                      'generation': {'outputs': {'samples.jsonl': exporter.digest(ledger)}},
                      'metrics': {'metric_directory': str(metrics),
                                  'outputs': {p.name: exporter.digest(p) for p in metrics.iterdir()}}}]})
        (evaluation/'summary.csv').write_text('model,job,requested,generated,generation_failed\n'+model+',fixture_dng,3,2,1\n')
        write(evaluation/'summary.json', {'state': 'complete'})
        write(folder/'state'/f'legacy_{model}.json', {'state': 'complete'})
        (generation.parents[1]/'model.best.pt').write_text('WEIGHTS must-not-export')
    return manifest


@pytest.mark.parametrize('limit, expected_cifs', [(0, 0), (1, 6), (-1, 12)])
def test_export_preserves_all_attempts_and_exact_structure_metric_pairing(completed, tmp_path, limit, expected_cifs):
    before = {p: p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    output = tmp_path/'downloads/main6.zip'
    inventory = exporter.export(completed, output, cifs_per_task=limit)
    assert all(p.read_bytes() == content for p, content in before.items())
    assert inventory['source_revision'] == 'original-code'
    with zipfile.ZipFile(output) as archive:
        assert archive.testzip() is None
        names = archive.namelist()
        assert not any(name.endswith(('.env', '.pt')) for name in names)
        assert len([name for name in names if '/cifs/' in name]) == expected_cifs
        recorded = json.loads(archive.read('manifest.json'))
        assert len(recorded['tasks']) == 6
        assert all((task['attempted'], task['generated'], task['failed']) == (3, 2, 1) for task in recorded['tasks'])
        for name, receipt in recorded['files'].items():
            assert hashlib.sha256(archive.read(name)).hexdigest() == receipt['sha256']
        summaries = list(csv.DictReader(io.StringIO(archive.read('summary.csv').decode('utf-8-sig'))))
        assert {r['model'] for r in summaries} == set(exporter.DEFAULT_MODELS)
        for task in recorded['tasks']:
            ledger = archive.read(task['samples'])
            source = tmp_path/'outputs'/task['model']/'generate/fixture_dng/samples.jsonl'
            assert ledger == source.read_bytes()
        for row in csv.DictReader(io.StringIO(archive.read('cif_index.csv').decode('utf-8-sig'))):
            content = archive.read(row['cif'])
            assert hashlib.sha256(content).hexdigest() == row['cif_sha256']
            rows = list(csv.DictReader(io.StringIO(archive.read(row['metrics_csv']).decode())))
            assert next(r for r in rows if r['index'] == row['index'])['basic_valid'] in ('True', 'False')
    assert output.with_suffix('.zip.sha256').read_text().split()[0] == exporter.digest(output)
    with pytest.raises(FileExistsError):
        exporter.export(completed, output)


def test_export_refuses_unfinished_model_before_creating_archive(completed, tmp_path):
    write(completed.parent/'state/legacy_joint_u.json', {'state': 'running'})
    output = tmp_path/'incomplete.zip'
    with pytest.raises(ValueError, match='not complete'):
        exporter.export(completed, output)
    assert not output.exists() and not output.with_suffix('.zip.partial').exists()


@pytest.mark.parametrize('changed', ['generation', 'metrics', 'inline_cif'])
def test_export_refuses_corrupted_results(completed, tmp_path, changed):
    pipeline = completed.parent/'evaluation/mp20_u/pipeline.json'
    state = exporter.read(pipeline)
    entry = state['jobs'][0]
    if changed == 'metrics':
        (Path(entry['metrics']['metric_directory'])/'summary.json').write_text('{"changed": true}')
    else:
        ledger = Path(entry['directory'])/'samples.jsonl'
        if changed == 'generation':
            ledger.write_text(ledger.read_text()+'{}\n')
        else:
            rows = [json.loads(line) for line in ledger.read_text().splitlines()]
            rows[1]['cif'] = 'modified CIF'
            ledger.write_text(''.join(json.dumps(row)+'\n' for row in rows))
            entry['generation']['outputs']['samples.jsonl'] = exporter.digest(ledger)
            write(pipeline, state)
    output = tmp_path/'corrupted.zip'
    with pytest.raises(ValueError, match='modified|CIF hash mismatch'):
        exporter.export(completed, output)
    assert not output.exists()


def test_export_includes_component_report_and_only_selected_models(completed, tmp_path):
    folder = completed.parent
    pipeline = folder/'evaluation/mp20_u/pipeline.json'
    state = exporter.read(pipeline)
    components = tmp_path/'outputs/mp20_u/generate/fixture_components'
    write(components/'components.json', {'state': 'complete'})
    state['identity']['jobs'].append({'name': 'fixture_components'})
    state['jobs'].append({'name': 'mp20_u/fixture_components', 'state': 'complete',
                          'directory': str(components),
                          'outputs': {'components.json': exporter.digest(components/'components.json')}})
    write(pipeline, state)
    write(folder/'state/legacy_joint_u.json', {'state': 'running'})
    output = tmp_path/'selected.zip'
    exporter.export(completed, output, models=['mp20_u'])
    with zipfile.ZipFile(output) as archive:
        assert 'models/mp20_u/tasks/fixture_components/generation/components.json' in archive.namelist()
        assert json.loads(archive.read('manifest.json'))['models'] == ['mp20_u']
        assert not any(name.startswith('models/joint_u/') for name in archive.namelist())
