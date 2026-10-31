"""Read-only full C2DB source, label and group-orbit encoding verification."""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
from collections import Counter, deque
import csv
from datetime import datetime, timezone, timedelta
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import time
import warnings

import numpy as np
import yaml
from pymatgen.core import Composition
from scipy.optimize import linear_sum_assignment
from scripts.data_tools.recovery import verify_structure
from unifiedgroupgen.common.configuration import project_root, resolve_path
from unifiedgroupgen.common.source import file_hash
from unifiedgroupgen.pl_data.encoding import parse_cif
from unifiedgroupgen.tests.preflight import check_recovered_source
from unifiedgroupgen.tests.source_integrity import check_record_source, encoding_digest, _isolated_layer
from unifiedgroupgen.tests.representation import compiled_chart, expand_numpy, action_statistics
from unifiedgroupgen.group_constraints.compiler import Descriptor

ROOT = project_root()
PROPERTIES = {'formation_energy': 'hform', 'band_gap': 'gap', 'energy_above_hull': 'ehull'}


def number(value):
    return float(value) if value not in (None, '', 'nan', 'NaN', 'None') else None


def check_cif_export(source, exported):
    """CIF preserves cell parameters and fractional sites, not the Cartesian frame/site order.

    Reverse CIF parser wrapping in its known fractional basis. This serialization
    check does not make the material periodic in z: check_record_source below
    separately compares the isolated layers with only two periodic directions.
    """
    assert len(source) == len(exported) and source.composition == exported.composition, 'CIF composition changed'
    assert np.allclose(source.lattice.abc, exported.lattice.abc, atol=1e-6, rtol=0), 'CIF cell lengths changed'
    assert np.allclose(source.lattice.angles, exported.lattice.angles, atol=1e-6, rtol=0), 'CIF cell angles changed'
    residual = 0.
    for element in set(source.atomic_numbers):
        first = source.frac_coords[np.array(source.atomic_numbers) == element]
        second = exported.frac_coords[np.array(exported.atomic_numbers) == element]
        delta = first[:, None]-second[None]
        delta -= np.round(delta)
        distances = np.linalg.norm(delta @ source.lattice.matrix, axis=-1)
        a, b = linear_sum_assignment(distances)
        residual = max(residual, float(distances[a, b].max()))
    thickness = [float(np.ptp(_isolated_layer(s, 100.).cart_coords[:, 2])) for s in (source, exported)]
    assert residual <= 1e-6 and abs(thickness[0]-thickness[1]) <= 2e-6, 'CIF sites or layer thickness changed'
    return residual


def worker(item):
    split, original, recovered, record, tolerance = item
    result = {'split': split, 'id': record['id'], 'passed': False}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            uid = original['uid']
            assert uid == recovered['uid'] == record['id'], 'UID changed'
            provenance = record['provenance']
            cache = ROOT/'data/scr/source_recovery/c2db_official/official_json'/(hashlib.sha256(uid.encode()).hexdigest()+'.json')
            assert file_hash(cache) == recovered['official_structure_sha256'] == provenance['official_structure_sha256'], 'Official cache hash changed'
            content = json.loads(cache.read_text(encoding='utf-8'))
            entries = [v for v in content.values() if isinstance(v, dict) and 'positions' in v]
            assert len(entries) == 1, 'Official cache must contain exactly one structure'
            assert np.isfinite(float(original['energy'])) and np.isfinite(float(entries[0]['energy'])), 'Missing/nonfinite source energy'
            restored, residual = verify_structure(original, entries[0])
            result.update(source_cartesian_residual_angstrom=residual,
                          authority='ase_database' if content.get('_source') else 'official_download')
            if content.get('_source'):
                declared = json.loads(recovered['original_ase_source'])
                assert declared == content['_source'], 'Original ASE provenance differs'
            source = parse_cif(recovered['cif'], 'fractional', preserve_fractional=True)
            result['cif_roundtrip_max_displacement_angstrom'] = check_cif_export(restored, source)
            assert int(original['natoms']) == len(source), 'Source atom count column differs'
            assert Composition(original['formula']).reduced_composition == source.composition.reduced_composition, 'Source formula column differs'
            assert hashlib.sha256(recovered['cif'].encode()).hexdigest() == provenance['source_cif_sha256'], 'Encoded source CIF changed'
            assert encoding_digest(record) == provenance['source_integrity']['encoding_sha256'], 'Encoded descriptor/state changed'
            comparison = check_record_source(source, record, tolerance)
            result.update({key: comparison[key] for key in ('composition_conserved', 'geometry_equivalent',
                          'numerically_identical', 'max_displacement_angstrom', 'measure_per_atom_relative_error',
                          'source_atoms', 'encoded_atoms')})
            assert comparison['composition_conserved'] and comparison['geometry_equivalent'], comparison.get('failure', 'Source equivalence failed')
            chart = compiled_chart(Descriptor.from_dict(record['descriptor']))
            lattice, coords, metric = expand_numpy(chart, record['state'])
            assert tuple(chart.periodic) == (True, True, False), 'Wrong layer periodicity'
            stats = action_statistics(chart, coords, lattice)
            result.update(operation_residual_angstrom=stats['operation_residual_angstrom'],
                          metric_relative_residual=stats['metric_relative_residual'],
                          metric_min_eigenvalue=float(np.linalg.eigvalsh(metric).min()),
                          source_group=int(original['lgnum']), encoded_group=chart.descriptor.number)
            assert result['operation_residual_angstrom'] < 1e-7, 'Encoded group operation failure'
            assert result['metric_relative_residual'] < 1e-7 and result['metric_min_eigenvalue'] > 0, 'Invalid group-compatible metric'
            for name, column in PROPERTIES.items():
                before, after, encoded = number(original[column]), number(recovered[column]), record['properties'].get(name)
                assert before == after == encoded, 'Property changed: '+name
                assert encoded is None or np.isfinite(encoded), 'Nonfinite property: '+name
                result[name+'_present'] = encoded is not None
            result['passed'] = True
    except Exception as error:
        result['error'] = type(error).__name__+': '+str(error)
    return result


def initialize():
    import torch
    torch.set_num_threads(1)


def stratified_items(items):
    """Two reproducibly selected records per observed split/group, plus hard cases."""
    strata = {}
    for item in items:
        split, _, _, record, _ = item
        strata.setdefault((split, record['descriptor']['number']), []).append(item)
    selected = {}
    for key, rows in sorted(strata.items()):
        ranked = sorted(rows, key=lambda x: hashlib.sha256(
            ('c2db-stratified-v1|'+x[0]+'|'+x[3]['id']).encode()).hexdigest())
        for row in ranked[:2]:
            selected[(row[0], row[3]['id'])] = row
    for split in ('train', 'val', 'test'):
        rows = [x for x in items if x[0] == split]
        for score in (
            lambda x: int(x[1]['natoms']),
            lambda x: x[3]['provenance']['source_integrity']['max_displacement_angstrom'],
        ):
            for row in sorted(rows, key=lambda x: (-score(x), x[3]['id']))[:10]:
                selected[(row[0], row[3]['id'])] = row
    return list(selected.values()), [
        {'split': split, 'layer_group': group, 'population': len(rows),
         'sampled': sum(x[0] == split and x[3]['descriptor']['number'] == group
                        for x in selected.values())}
        for (split, group), rows in sorted(strata.items())]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--limit', type=int, help='Per-split diagnostic subset; never a full-data result')
    selection.add_argument('--stratified', action='store_true', help='Fresh split/layer-group sample plus largest/highest-residual records')
    args = parser.parse_args()
    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=False)
    config = yaml.safe_load((ROOT/'configs/c2db_verified_v1.yaml').read_text(encoding='utf-8'))
    started = time.monotonic()
    status = output/'status.json'
    def save_status(value):
        temp = status.with_suffix('.tmp'); temp.write_text(json.dumps(value), encoding='utf-8'); temp.replace(status)
    save_status({'state': 'checking_sources', 'pid': os.getpid()})
    check_recovered_source(config)
    files, items, counts, ids, hashes, canonical, missing = [], [], {}, {}, {}, {}, Counter()
    for split in ('train', 'val', 'test'):
        paths = [ROOT/f'data/scr/c2db_51/{split}.csv', ROOT/f'data/scr/source_recovery/c2db_official/{split}.csv',
                 ROOT/f'data/dataset/full_verified_v1/c2db_51/{split}.jsonl']
        files.extend(paths)
        original, recovered = [list(csv.DictReader(p.open(encoding='utf-8', newline=''))) for p in paths[:2]]
        encoded = [json.loads(s) for s in paths[2].read_text(encoding='utf-8').splitlines() if s.strip()]
        assert len(original) == len(recovered) == len(encoded), 'Source/encoded row count differs: '+split
        counts[split] = len(encoded); ids[split] = {r['id'] for r in encoded}
        assert len(ids[split]) == len(encoded), 'Duplicate UID within '+split
        for index, (raw, restored, record) in enumerate(zip(original, recovered, encoded)):
            assert raw['uid'] == restored['uid'] == record['id'], 'UID or row order differs'
            assert record['dataset'] == 'c2db_51' and record['descriptor']['kind'] == 'layer'
            assert record['provenance']['source_row'] == index
            assert restored['original_csv_sha256'] == file_hash(paths[0]) if index == 0 else True
            canonical[record['id']] = (encoding_digest(record), record['properties'])
            for name, column in PROPERTIES.items():
                if number(raw[column]) is None: missing[name] += 1
            if args.limit is None or index < args.limit:
                items.append((split, raw, restored, record, config['symmetry_tolerance']))
        report = json.loads(paths[2].with_suffix('.jsonl.report.json').read_text(encoding='utf-8'))
        assert report['input'] == report['accepted'] == len(encoded) and report['rejected'] == 0
        assert report['source_sha256'] == file_hash(paths[1])
        hashes.update({str(p.relative_to(ROOT)): file_hash(p) for p in paths})
    assert counts == {'train': 10143, 'val': 3381, 'test': 3381}
    assert not (ids['train'] & ids['val'] or ids['train'] & ids['test'] or ids['val'] & ids['test']), 'UID split leakage'
    joint = Counter()
    joint_config = yaml.safe_load((ROOT/'configs/data/joint.yaml').read_text(encoding='utf-8'))
    joint_paths = {}
    for split in ('train', 'val', 'test'):
        for logical in joint_config[split+'_files']:
            path = resolve_path(logical, ROOT)
            if path.parent.name != 'c2db_51':
                continue
            joint_paths[split] = str(path.relative_to(ROOT))
            for line in path.read_text(encoding='utf-8').splitlines():
                record = json.loads(line)
                assert canonical[record['id']] == (encoding_digest(record), record['properties']), 'Joint C2DB record changed'
                joint[split] += 1
    assert dict(joint) == counts, 'Joint C2DB counts differ'
    strata = []
    if args.stratified:
        items, strata = stratified_items(items)
        (output/'selection.json').write_text(json.dumps({
            'protocol': 'c2db-stratified-v1', 'strata': strata,
            'selected': [{'split': x[0], 'id': x[3]['id'], 'layer_group': x[3]['descriptor']['number']} for x in items],
        }, indent=2), encoding='utf-8')
    processed = []
    pool = mp.get_context('spawn').Pool(args.workers, initializer=initialize)
    try:
        pending = deque(); iterator = iter(items)
        for _ in range(args.workers*4):
            item = next(iterator, None)
            if item is not None: pending.append(pool.apply_async(worker, (item,)))
        with (output/'records.jsonl').open('w', encoding='utf-8') as stream:
            while pending:
                row = pending.popleft().get(timeout=300)
                processed.append(row); stream.write(json.dumps(row, allow_nan=False)+'\n')
                item = next(iterator, None)
                if item is not None: pending.append(pool.apply_async(worker, (item,)))
                if len(processed) % 100 == 0 or len(processed) == len(items):
                    stream.flush(); progress = {'state': 'running', 'completed': len(processed), 'requested': len(items),
                        'failed': sum(not r['passed'] for r in processed), 'seconds': time.monotonic()-started}
                    save_status(progress); print(json.dumps(progress), flush=True)
        pool.close()
    except BaseException:
        pool.terminate(); save_status({'state': 'failed', 'completed': len(processed), 'requested': len(items)}); raise
    finally:
        pool.join()
    with (output/'records.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({k for r in processed for k in r})); writer.writeheader(); writer.writerows(processed)
    unchanged = all(file_hash(ROOT/p) == digest for p, digest in hashes.items())
    failed = [r for r in processed if not r['passed']]
    keys = ('source_cartesian_residual_angstrom', 'cif_roundtrip_max_displacement_angstrom',
            'max_displacement_angstrom', 'measure_per_atom_relative_error', 'operation_residual_angstrom', 'metric_relative_residual')
    summary = {'complete': args.limit is None and not args.stratified, 'passed': not failed and unchanged, 'records': len(processed),
        'planned_checks_complete': len(processed) == len(items),
        'selection': 'stratified' if args.stratified else ('first_per_split' if args.limit is not None else 'all'),
        'strata': strata,
        'splits': counts, 'failed': len(failed), 'inputs_unchanged': unchanged, 'input_sha256': hashes,
        'uid_overlap_across_splits': 0, 'joint_splits': dict(joint), 'joint_paths': joint_paths,
        'missing_property_labels': dict(missing),
        'audit_script_sha256': file_hash(Path(__file__)),
        'authority_counts': dict(Counter(r.get('authority') for r in processed)),
        'group_label_changes': sum(r.get('source_group') != r.get('encoded_group') for r in processed if r['passed']),
        'cell_atom_count_changes': sum(r.get('source_atoms') != r.get('encoded_atoms') for r in processed if r['passed']),
        'numerically_identical': sum(r.get('numerically_identical', False) for r in processed),
        'maxima': {k: max((r[k] for r in processed if r.get(k) is not None), default=None) for k in keys},
        'seconds': time.monotonic()-started, 'checked_at': datetime.now(timezone(timedelta(hours=8))).isoformat(),
        'scope': 'Fresh source/cache, labels, CIF geometry and encoded group compatibility for the selected records; all-row source-file hashes/UID/split binding. No fresh online download, independent maximal layer-group re-identification or DFT recalculation.'}
    (output/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    save_status({'state': 'complete' if summary['passed'] else 'failed', **summary})
    print(json.dumps(summary), flush=True)
    if failed: raise SystemExit(1)


if __name__ == '__main__':
    main()
