"""Generic preprocessing entry, configuration precedence and CSV-only execution."""
import csv
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from unifiedgroupgen.pl_data import cache

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / 'data/process.py'


@pytest.fixture
def process():
    spec = importlib.util.spec_from_file_location('data_process_cli', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_config(root, name='custom_crystals', **settings):
    config = dict(dataset=name, kind='space', data_root='data/src/' + name,
                  coordinate_mode='fractional', symmetry_tolerance=.001,
                  symmetry_protocol='single-pass-standard-orbits-v1',
                  max_atoms=192, max_orbits=32, property_columns={'band_gap': 'gap'}, **settings)
    path = root / (name + '.yaml')
    path.write_text(yaml.safe_dump(config), encoding='utf-8')
    return path


@pytest.fixture
def calls(monkeypatch):
    result = []
    def ensure(config, split, root, **options):
        result.append(dict(config=config, split=split, root=root, **options))
        return root / 'data/dataset' / config['dataset'] / (split + '.fixture.pt'), {'records': [{'id': 'fixture'}]}, True
    monkeypatch.setattr(cache, 'ensure_tensor_cache', ensure)
    state = SimpleNamespace(spec=SimpleNamespace(number=0), letters=())
    monkeypatch.setattr(cache, 'target_from_cache', lambda *args: SimpleNamespace(state=state))
    return result


def test_generic_source_configs_select_datasets_without_name_mapping(process, calls, tmp_path, capsys):
    a = source_config(tmp_path)
    b = source_config(tmp_path, 'another_dataset',
                      preprocessing={'splits': ['val'], 'workers': 2, 'cutoff': 4., 'max_neighbors': 24, 'max_images': 5})
    process.main(process.parse_args(['--root', str(tmp_path), '--config', str(a), str(b)]))
    assert [(r['config']['dataset'], r['split']) for r in calls] == [
        ('custom_crystals', 'train'), ('custom_crystals', 'val'), ('custom_crystals', 'test'),
        ('another_dataset', 'val')]
    assert calls[0]['graph_options'] == dict(cutoff=6., max_neighbors=64, max_images=16)
    assert calls[0]['workers'] == 0 and calls[0]['operation_compiler']
    assert calls[-1]['graph_options'] == dict(cutoff=4., max_neighbors=24, max_images=5)
    assert calls[-1]['workers'] == 2
    summaries = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith('{')]
    assert len(summaries) == 4 and all(row['state'] == 'cached' for row in summaries)


def test_arguments_override_config_without_rewriting_it(process, calls, tmp_path):
    path = source_config(tmp_path, preprocessing={'splits': ['test'], 'workers': 1, 'cutoff': 4.})
    before = path.read_bytes()
    model = tmp_path/'model.yaml'
    model.write_text(yaml.safe_dump(dict(cutoff=5., max_neighbors=16, max_images=16, operation_compiler=False)), encoding='utf-8')
    args = process.parse_args(['incoming_csv', '--root', str(tmp_path), '--config', str(path),
                               '--model', str(model), '--split', 'train', '--workers', '3', '--cutoff', '2.5',
                               '--max-neighbors', '12', '--max-images', '7'])
    process.main(args)
    assert len(calls) == 1 and calls[0]['split'] == 'train'
    assert calls[0]['config']['data_root'] == str(tmp_path / 'incoming_csv')
    assert calls[0]['graph_options'] == dict(cutoff=2.5, max_neighbors=12, max_images=7)
    assert calls[0]['workers'] == 3 and not calls[0]['operation_compiler']
    assert path.read_bytes() == before


def test_preprocessing_section_overrides_model_defaults(process, calls, tmp_path):
    path = source_config(tmp_path, preprocessing={'splits': ['train'], 'cutoff': 3.})
    model = tmp_path/'model.yaml'
    model.write_text(yaml.safe_dump(dict(cutoff=5., max_neighbors=24, max_images=16, operation_compiler=True)), encoding='utf-8')
    process.main(process.parse_args(['--root', str(tmp_path), '--config', str(path), '--model', str(model)]))
    assert calls[0]['graph_options'] == dict(cutoff=3., max_neighbors=24, max_images=16)


@pytest.mark.parametrize('settings', [dict(splits=['unknown']), dict(splits=[]), dict(workers=-1),
                                     dict(cutoff=0), dict(max_neighbors=0), dict(max_images=-1)])
def test_invalid_config_options_fail_before_cache_processing(process, calls, tmp_path, settings):
    path = source_config(tmp_path, preprocessing=settings)
    with pytest.raises(ValueError):
        process.main(process.parse_args(['--root', str(tmp_path), '--config', str(path)]))
    assert not calls and not (tmp_path / 'data/dataset').exists()


def test_one_directory_cannot_override_multiple_dataset_configs(process):
    with pytest.raises(SystemExit) as error:
        process.parse_args(['incoming', '--config', 'one.yaml', 'two.yaml'])
    assert error.value.code == 2


def test_retired_jsonl_migration_flag_is_rejected(process):
    with pytest.raises(SystemExit) as error:
        process.parse_args(['--config', 'example.yaml', '--migrate-jsonl'])
    assert error.value.code == 2


def test_csv_only_cli_builds_then_reuses_custom_dataset_cache(tmp_path):
    from pymatgen.core import Lattice, Structure
    structure = Structure(Lattice.from_parameters(4., 4.5, 5., 80., 87., 73.),
                          ['Si', 'O'], [[.11, .12, .13], [.37, .52, .61]])
    folder = tmp_path / 'incoming'
    folder.mkdir()
    for split in ('train', 'val'):
        with (folder / (split + '.csv')).open('w', encoding='utf-8', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=['material_id', 'cif', 'gap'])
            writer.writeheader()
            writer.writerow(dict(material_id=split + '0', cif=structure.to(fmt='cif'), gap=1.2))
    config = source_config(tmp_path, preprocessing={'splits': ['train', 'val'], 'workers': 2, 'cutoff': 3.})
    command = [sys.executable, '-X', 'utf8', '-B', str(SCRIPT), str(folder),
               '--root', str(tmp_path), '--config', str(config)]
    paths = []
    for expected in ('created', 'cached'):
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True,
                                encoding='utf-8', check=True, timeout=180)
        summaries = [json.loads(line) for line in result.stdout.splitlines() if line.startswith('{')]
        assert len(summaries) == 2 and all(row['state'] == expected for row in summaries)
        paths.append([row['cache'] for row in summaries])
        assert all(Path(path).parent == tmp_path / 'data/dataset/custom_crystals' for path in paths[-1])
    assert paths[0] == paths[1] and not list(tmp_path.rglob('*.jsonl'))
    for path in map(Path, paths[0]):
        report = json.loads(path.with_suffix('.config.json').read_text(encoding='utf-8'))
        assert report['cache'] == path.name and report['records'] == 1
        assert report['identity']['graph_options']['cutoff'] == 3.
        assert report['build']['mode'] == 'csv' and report['build']['workers'] == 2
    assert not list(tmp_path.rglob('*.records.csv'))
    assert not (tmp_path / 'outputs').exists()
