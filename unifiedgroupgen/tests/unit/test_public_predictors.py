"""Public predictor contracts; no optional library install or network needed here."""
from types import SimpleNamespace
import csv
import hashlib
import json
import sys

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

from unifiedgroupgen.common.predictors.structures import structure_from_graph, predictor_structure
from unifiedgroupgen.common.predictors.matgl import MatGLPredictor, fetch_model, model_registry
from unifiedgroupgen.common.predictors import service as property_inference


def test_expanded_graph_lattice_convention_is_explicit_and_embeddings_unused():
    lattice = torch.tensor([[4., 1., 0.], [0., 3., 0.], [0., 0., 5.]], requires_grad=True)
    graph = {'lattice': lattice, 'coordinates': torch.tensor([[0., 0., 0.], [.5, .5, .5]]),
             'atomic_numbers': torch.tensor([11, 17]), 'node_features': torch.randn(2, 64)}
    column = structure_from_graph(graph, kind='space')
    row = structure_from_graph(graph, kind='space', lattice_columns=False)
    np.testing.assert_allclose(column.lattice.matrix, lattice.detach().numpy().T)
    np.testing.assert_allclose(row.lattice.matrix, lattice.detach().numpy())
    np.testing.assert_allclose(column.cart_coords, graph['coordinates'].numpy() @ lattice.detach().numpy().T)
    converted = predictor_structure(column, 'space', 4.)
    assert converted.lattice.matrix.flags.c_contiguous
    np.testing.assert_allclose(converted.cart_coords, column.cart_coords)
    graph['node_features'].fill_(float('nan'))
    np.testing.assert_allclose(structure_from_graph(graph, kind='space').cart_coords, column.cart_coords)
    graph['coordinates'][0, 0] = float('nan')
    with pytest.raises(ValueError, match='Nonfinite'): structure_from_graph(graph, kind='space')


def test_layer_auxiliary_cell_is_vacuum_invariant_and_preserves_thickness():
    def slab(height):
        return Structure(Lattice.hexagonal(3.2, height), ['Mo', 'S', 'S'],
                         [[0, 0, height/2], [1.6, .7, height/2-1.5], [1.6, .7, height/2+1.5]],
                         coords_are_cartesian=True)
    first = predictor_structure(slab(20), 'layer', 5.)
    second = predictor_structure(slab(50), 'layer', 5.)
    np.testing.assert_allclose(first.lattice.matrix, second.lattice.matrix)
    np.testing.assert_allclose(first.cart_coords, second.cart_coords, atol=1e-12)
    assert np.ptp(first.cart_coords[:, 2]) == pytest.approx(3.)
    assert first.lattice.c == pytest.approx(15.)
    assert first.lattice.pbc == (True, True, False)


def test_model_cache_uses_fixed_hashes_and_detects_tampering(tmp_path, monkeypatch):
    data = b'fixed official bytes'
    spec = {'name': 'fixture', 'files': {'model.json': hashlib.sha256(data).hexdigest()}}
    calls = []
    def get(url, timeout):
        calls.append(url)
        return SimpleNamespace(content=data, raise_for_status=lambda: None)
    monkeypatch.setattr('requests.get', get)
    path = fetch_model(spec, {'base_url': 'https://example.org/fixed/'}, tmp_path)
    assert (path/'model.json').read_bytes() == data
    fetch_model(spec, {'base_url': 'https://example.org/fixed/'}, tmp_path)
    assert len(calls) == 1
    (path/'model.json').write_bytes(b'changed')
    with pytest.raises(ValueError, match='Cached pretrained file changed'):
        fetch_model(spec, {'base_url': 'https://example.org/fixed/'}, tmp_path)


def test_unsupported_property_and_fidelity_fail_before_optional_import(tmp_path):
    with pytest.raises(ValueError, match='supported'): MatGLPredictor(tmp_path, properties=['energy_above_hull'])
    with pytest.raises(ValueError, match='fidelity'): MatGLPredictor(tmp_path, band_gap_fidelity='unknown')
    registry = model_registry()
    assert set(registry['models']) == {'formation_energy', 'band_gap'}
    assert all(len(m['files']) == 3 for m in registry['models'].values())


def test_batch_failure_falls_back_without_losing_other_structures():
    predictor = object.__new__(MatGLPredictor)
    predictor.models = {'formation_energy': object()}
    def graph(value, name, kind):
        if value < 0: raise ValueError('invalid geometry')
        return value
    def forward(graphs, name):
        if len(graphs) > 1: raise RuntimeError('batch fixture error')
        return np.array([float(graphs[0])])
    predictor.graph, predictor._forward = graph, forward
    rows = predictor.predict_structures([1, -1, 2], kind='space', batch_size=3)
    assert [r['formation_energy'] for r in rows] == [1., None, 2.]
    assert [bool(r['prediction_error']) for r in rows] == [False, True, False]


def test_public_cli_retains_failed_indices_and_reuses_reference_cache(tmp_path, monkeypatch):
    from unifiedgroupgen.common.predictors import matgl as adapter
    calls = []
    class FakePredictor:
        def __init__(self, cache, **kwargs):
            self.models = {'formation_energy': object(), 'band_gap': object()}
            self.identity = {'fixture': True}
            self.registry = model_registry()
        def predict_structures(self, structures, **kwargs):
            calls.append(len(structures))
            return [{'formation_energy': -1. if s is not None else None,
                     'band_gap': 2. if s is not None else None,
                     'prediction_error': '' if s is not None else 'missing input'} for s in structures]
    monkeypatch.setattr(adapter, 'MatGLPredictor', FakePredictor)
    config = tmp_path/'predictor.yaml'; config.write_text('properties: [formation_energy, band_gap]\n')
    structure = Structure(Lattice.cubic(4), ['Na', 'Cl'], [[0, 0, 0], [.5, .5, .5]])
    cif = structure.to(fmt='cif')
    refs = tmp_path/'refs.jsonl'
    refs.write_text(json.dumps({'id': 'ref', 'dataset': 'mp_20', 'structure': structure.as_dict()})+'\n')
    samples = tmp_path/'samples.jsonl'
    records = [{'index': 4, 'status': 'generated', 'cif': cif},
               {'index': 9, 'status': 'failed'}, {'index': 12, 'status': 'generated', 'cif': 'invalid CIF'}]
    samples.write_text(''.join(json.dumps(r)+'\n' for r in records))
    command = ['properties', '--backend', 'matgl', '--kind', 'space', '--dataset', 'mp_20',
               '--samples', str(samples), '--reference-files', str(refs), '--reference-cache', str(tmp_path/'cache'),
               '--predictor-config', str(config)]
    for name in ('first', 'second'):
        args=SimpleNamespace(samples=samples, dataset='mp_20', kind='space', model_cache=tmp_path/'models',
                             reference_cache=tmp_path/'cache', reference_files=[refs],
                             predictor_config=config, batch_size=8, output=tmp_path/name)
        property_inference.public_predictions(args)
    assert calls == [1, 3, 3]
    with (tmp_path/'first/predictions.csv').open(encoding='utf-8-sig') as stream:
        values = list(csv.DictReader(stream))
    assert [r['index'] for r in values] == ['4', '9', '12']
    assert values[0]['formation_energy'] == '-1.0'
    assert all(not r['formation_energy'] and r['prediction_error'] for r in values[1:])
    metadata = json.loads((tmp_path/'first/evaluator.json').read_text())
    assert metadata['counts'] == {'requested': 3, 'complete_predictions': 1, 'failed_predictions': 2}
    assert metadata['energy_above_hull'] == 'not_assessed'
    assert (tmp_path/'first/reference_predictions.csv').read_bytes() == (tmp_path/'second/reference_predictions.csv').read_bytes()
    args.output=tmp_path/'first'
    with pytest.raises(ValueError, match='empty'): property_inference.public_predictions(args)


def test_layer_prediction_requires_explicit_proxy_transfer(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, 'argv', ['properties', '--backend', 'matgl', '--kind', 'layer', '--dataset', 'c2db_51',
        '--samples', 'samples', '--reference-files', 'refs', '--reference-cache', str(tmp_path/'cache'),
        '--output', str(tmp_path/'out')])
    with pytest.raises(SystemExit):
        from unifiedgroupgen.tests.legacy.evaluation_before_tasks.property_inference import main
        main()


def test_evaluation_rejects_changed_predictor_structure_file(tmp_path, monkeypatch):
    from unifiedgroupgen.tests.legacy.evaluation_before_tasks import cli
    samples = tmp_path/'samples.jsonl'; samples.write_text('{"index": 0, "status": "failed"}\n')
    predictions = tmp_path/'predictions.csv'; predictions.write_text('index,formation_energy\n0,\n')
    cif = tmp_path/'input.cif'; cif.write_text('changed')
    metadata = tmp_path/'evaluator.json'
    metadata.write_text(json.dumps({'structure_files_sha256': {str(cif): hashlib.sha256(b'original').hexdigest()}}))
    monkeypatch.setattr(sys, 'argv', ['evaluate', '--samples', str(samples), '--task', 'dng', '--kind', 'space',
        '--assessments', str(predictions), '--assessment-metadata', str(metadata), '--output', str(tmp_path/'evaluation')])
    with pytest.raises(ValueError, match='input CIF changed'): cli.main()
