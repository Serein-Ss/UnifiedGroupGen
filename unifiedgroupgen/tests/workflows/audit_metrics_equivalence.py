"""Bounded 3D/2D CSP/DNG metric comparison against extracted server source."""
import argparse
import ast
import copy
import importlib
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from pymatgen.core import Lattice, Structure

from unifiedgroupgen.evaluate import evaluate
from unifiedgroupgen.evaluation.conditional import evaluate_properties
from unifiedgroupgen.evaluation import utils
from unifiedgroupgen.common.configuration import project_root


def run(snapshot, output):
    sys.path.insert(0, str(snapshot))
    old = importlib.import_module('server_original_ugg.unified_evaluation')
    old_properties = importlib.import_module('server_original_ugg.evaluation_properties')
    old_standard = importlib.import_module('server_original_ugg.standard_metrics')
    output.mkdir(parents=True, exist_ok=True)
    def ledger(path, rows):
        path.write_text(''.join(json.dumps(row)+'\n' for row in rows), encoding='utf-8')
        return path
    results, fingerprint_checks = {}, {}
    for kind in ('space', 'layer'):
        structure = (Structure(Lattice.cubic(5.6), ['Na', 'Cl'], [[0, 0, 0], [.5, .5, .5]])
                     if kind == 'space' else Structure(Lattice.hexagonal(3.16, 20), ['Mo', 'S', 'S'],
                         [[0, 0, .5], [1/3, 2/3, .578], [1/3, 2/3, .422]]))
        reference = ledger(output/(kind+'_reference.jsonl'), [{'id': 'r', 'structure': structure.as_dict()}])
        old_features, new_features = old_standard.fingerprints(structure, kind), utils.fingerprints(structure, kind)
        assert all(np.array_equal(a, b) for a, b in zip(old_features, new_features))
        assert old_standard.coverage([old_features], [old_features], 3, 1) == utils.coverage([new_features], [new_features], 3, 1)
        fingerprint_checks[kind] = {'identical': True, 'structural_dimensions': len(new_features[0]),
                                    'composition_dimensions': len(new_features[1]), 'coverage_equal': True}
        for task in ('dng', 'csp'):
            if task == 'csp':
                rows = [{'index': i, 'case_id': 'r', 'candidate': i, 'status': 'failed'} for i in range(20)]
                rows[4].update(status='generated', structure=structure.as_dict())
                request = {'case_ids': ['r'], 'attempts_per_case': 20}
            else:
                rows = [{'index': i, 'status': 'generated', 'structure': structure.as_dict()} for i in range(2)]
                rows.append({'index': 2, 'status': 'failed'})
                request = {}
            samples = ledger(output/(kind+'_'+task+'.jsonl'), rows)
            options = dict(reference_files=[reference], novelty_files=[reference], request=request,
                           use_fingerprints=False, reidentify=False)
            _, before, _ = old.evaluate(samples, task, kind, output/(kind+'_'+task+'_old'), **options)
            _, after, _ = evaluate(samples, task, kind, output/(kind+'_'+task+'_new'), **options)
            for report in (before, after):
                report.pop('efficiency')  # Wall time is not an invariant.
                report.pop('properties')  # The new entry makes property evaluation opt-in.
            assert after == before, (kind, task)
            results[kind+'/'+task] = {'identical': True, 'requested': len(rows),
                                     'matches': after.get('csp')}
    rows = [{'index': i, 'basic_valid': True, 'property_targets': {'band_gap': 1.}} for i in range(3)]
    predictions = {'0': {'band_gap': 1.1}, '1': {'band_gap': 1.5}, '2': {'prediction_error': 'retained failure'}}
    before = old_properties.evaluate_properties(copy.deepcopy(rows), predictions, {}, {'band_gap': .2})
    after = evaluate_properties(copy.deepcopy(rows), predictions, {}, {'band_gap': .2}, enabled=True)
    assert before == after
    def bodies(path, original=False):
        source = path.read_text(encoding='utf-8')
        if original:
            source = source.replace("parent/'benchmark_comp_scaler.json'", "parent/'resources/benchmark_comp_scaler.json'")
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef)}
    original_functions = {name: body for file in ('standard_metrics.py', 'evaluation_protocols.py', 'evaluation_statistics.py')
                          for name, body in bodies(snapshot/'server_original_ugg'/file, original=True).items()}
    current_functions = bodies(project_root()/'unifiedgroupgen/evaluation/utils.py')
    same = {name: current_functions.get(name) == body for name, body in original_functions.items()}
    assert all(same.values()), [name for name, passed in same.items() if not passed]
    old_scaler = snapshot/'server_original_ugg/benchmark_comp_scaler.json'
    new_scaler = project_root()/'unifiedgroupgen/evaluation/resources/benchmark_comp_scaler.json'
    # Formatting changed when the resource moved; compare every numeric value.
    assert json.loads(old_scaler.read_text()) == json.loads(new_scaler.read_text())
    return {'passed': True, 'cases': results, 'conditional_metrics_equal': True,
            'metric_function_bodies_equal_after_resource_path_normalization': same,
            'resource_path_normalization': 'Only benchmark_comp_scaler.json was moved into resources/',
            'scaler_sha256': hashlib.sha256(new_scaler.read_bytes()).hexdigest(),
            'scaler_values_identical': True,
            'fingerprint_probes': fingerprint_checks,
            'limits': ['Bounded software comparison; no new quality benchmark',
                       'Two fingerprint examples checked; full fingerprint distributions not regenerated',
                       'Independent SG/LG identifiers covered by existing dedicated tests, not these cases']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.snapshot.resolve(), args.output.resolve())
    (args.output/'comparison.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'metric_cases': len(result['cases']),
                      'functions': len(result['metric_function_bodies_equal_after_resource_path_normalization'])}))


if __name__ == '__main__':
    main()
