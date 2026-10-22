"""Audit every raw CIF against its saved group-orbit structure; repair into a new tree."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')

import argparse
from collections import Counter
import csv
import hashlib
import json
import multiprocessing
from pathlib import Path
import sys
import time
import warnings
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from unifiedgroupgen.source_integrity import PROTOCOL, check_record_source, encoding_digest
from unifiedgroupgen.data import parse_cif, convert_row

CONFIGS = ['mp20_local', 'mpts52_local', 'c2db_official_local']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def worker(item):
    old, row, config, split, source, repair = item
    result = {'dataset': old['dataset'], 'id': old['id'], 'source_split': split,
              'source_row': old['provenance']['source_row'],
              'source_cif_sha256': hashlib.sha256(row['cif'].encode()).hexdigest()}
    if result['source_cif_sha256'] != old['provenance']['source_cif_sha256']:
        return {**result, 'error': 'Source CIF hash does not match saved provenance'}
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            structure = parse_cif(row['cif'], config.get('coordinate_mode', 'fractional'), preserve_fractional=True)
            result['old'] = check_record_source(structure, old, config['symmetry_tolerance'])
            passed = result['old']['composition_conserved'] and result['old']['geometry_equivalent']
            result['affected'] = not passed
            if repair:
                fresh = old
                if not passed:
                    converted = convert_row((result['source_row'], row, config, source))
                    if 'record' not in converted:
                        raise ValueError(converted['rejection']['reason'])
                    fresh = converted['record']
                result['new'] = result['old'] if passed else check_record_source(structure, fresh, config['symmetry_tolerance'])
                if not (result['new']['composition_conserved'] and result['new']['geometry_equivalent']):
                    raise ValueError('Repaired encoding still differs from source: ' + result['new'].get('failure', 'unknown'))
                fresh['provenance']['source_integrity'] = {
                    **result['new'], 'source_cif_sha256': result['source_cif_sha256'],
                    'encoding_sha256': encoding_digest(fresh)}
                result['record'] = fresh
    except Exception as error:
        result['error'] = type(error).__name__ + ': ' + str(error)
        if repair:
            result['quarantine_record'] = old
    return result


def inputs(limit):
    for name in CONFIGS:
        config = yaml.safe_load((ROOT / 'configs' / f'{name}.yaml').read_text(encoding='utf-8'))
        data_root = (ROOT / config['data_root']).resolve()
        portable = ROOT / 'raw_data' / config['dataset']
        if config['kind'] == 'space' and portable.exists():
            data_root = portable
        for split in ('train', 'val', 'test'):
            saved = ROOT / config['prepared_root'] / f'{split}.jsonl'
            source = data_root / f'{split}.csv'
            records = [json.loads(s) for s in saved.read_text(encoding='utf-8').splitlines() if s.strip()]
            wanted = {int(r['provenance']['source_row']): r for r in records}
            with source.open(encoding='utf-8', newline='') as stream:
                for index, row in enumerate(csv.DictReader(stream)):
                    if index not in wanted:
                        raise ValueError(f'Unaccounted raw row: {source}:{index}')
                    if limit and index >= limit:
                        break
                    row = {key: None if value == '' else value for key, value in row.items()}
                    yield wanted[index], row, config, split, str(source)


def summarize(rows, complete):
    datasets = {}
    for dataset in ('mp_20', 'mpts_52', 'c2db_51'):
        selected = [r for r in rows if r['dataset'] == dataset]
        def counts(part):
            return {'records': len(part), 'composition_failed': sum(r.get('old', {}).get('composition_conserved') is False for r in part),
                    'geometry_failed_after_composition': sum(r.get('old', {}).get('composition_conserved') is True and not r['old']['geometry_equivalent'] for r in part),
                    'affected': sum(r.get('affected', False) for r in part), 'errors': sum('error' in r for r in part),
                    'repaired_passed': sum(r.get('new', {}).get('geometry_equivalent') is True for r in part)}
        datasets[dataset] = {**counts(selected), 'source_splits': {
            split: counts([r for r in selected if r['source_split'] == split]) for split in ('train', 'val', 'test')}}
    affected = {(r['dataset'], r['id']) for r in rows if r.get('affected') or 'error' in r}
    joint = {}
    for split in ('train', 'val', 'test'):
        parts = {}
        for dataset in datasets:
            path = ROOT / 'artifacts/full_joint_sourcegrouped' / dataset / f'{split}.jsonl'
            values = [json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s.strip()]
            parts[dataset] = {'records': len(values), 'affected': sum((r['dataset'], r['id']) in affected for r in values)}
        joint[split] = parts
    return {'protocol': PROTOCOL, 'complete': complete, 'datasets': datasets,
            'joint_historical_split_impact': joint,
            'scope': 'Source CIF equivalence under explicit modelling tolerance, not physical stability; original data and weights retained.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', default='artifacts/source_integrity_20261022')
    parser.add_argument('--workers', type=int, default=6)
    parser.add_argument('--limit', type=int, help='Per source split; marks report partial')
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args()
    output = (ROOT / args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    journal = output / ('repair.jsonl' if args.repair else 'audit.jsonl')
    if journal.exists():
        raise ValueError('Use a new output directory; existing audit evidence is preserved')
    start = time.monotonic()
    streams, rows = {}, []
    with journal.open('w', encoding='utf-8') as stream, (output/'quarantine.jsonl').open('w', encoding='utf-8') as quarantined, multiprocessing.get_context('spawn').Pool(
            args.workers, maxtasksperchild=512) as pool:
        pending = pool.imap(worker, ((*item, args.repair) for item in inputs(args.limit)), chunksize=4)
        for index, result in enumerate(pending, 1):
            record = result.pop('record', None)
            quarantine_record = result.pop('quarantine_record', None)
            if quarantine_record is not None:
                quarantined.write(json.dumps({'record': quarantine_record, 'reason': result['error']}, ensure_ascii=False) + '\n')
            if record is not None:
                key = result['dataset'], result['source_split']
                if key not in streams:
                    path = output / 'corrected' / key[0] / f'{key[1]}.jsonl'
                    path.parent.mkdir(parents=True, exist_ok=True)
                    streams[key] = path.open('w', encoding='utf-8')
                streams[key].write(json.dumps(record, ensure_ascii=False, allow_nan=False) + '\n')
            stream.write(json.dumps(result, ensure_ascii=False, allow_nan=False) + '\n')
            rows.append(result)
            if index % 250 == 0:
                stream.flush()
                status = {'processed': index, 'affected': sum(r.get('affected', False) for r in rows),
                          'errors': sum('error' in r for r in rows), 'seconds': round(time.monotonic() - start, 1)}
                (output / 'status.json').write_text(json.dumps(status), encoding='utf-8')
                print(json.dumps(status), flush=True)
    for stream in streams.values():
        stream.close()
    summary = summarize(rows, args.limit is None)
    summary['seconds'] = time.monotonic() - start
    summary['journal_sha256'] = digest(journal)
    summary['implementation_sha256'] = {name: digest(ROOT/name) for name in (
        'src/unifiedgroupgen/source_integrity.py', 'src/unifiedgroupgen/data.py',
        'scripts/audit_source_integrity.py')}
    summary['inputs'] = []
    for name in CONFIGS:
        config = yaml.safe_load((ROOT/'configs'/f'{name}.yaml').read_text(encoding='utf-8'))
        source_root = (ROOT/config['data_root']).resolve()
        if config['kind'] == 'space' and (ROOT/'raw_data'/config['dataset']).exists():
            source_root = ROOT/'raw_data'/config['dataset']
        for split in ('train', 'val', 'test'):
            source = source_root/f'{split}.csv'
            prepared = ROOT/config['prepared_root']/f'{split}.jsonl'
            summary['inputs'].append({'dataset': config['dataset'], 'split': split,
                                      'raw_sha256': digest(source), 'prepared_sha256': digest(prepared)})
    summary['quarantined_records'] = sum('error' in r for r in rows)
    summary['verified_records'] = sum(r.get('new', {}).get('geometry_equivalent') is True and 'error' not in r for r in rows)
    summary['all_source_records_repaired'] = args.repair and summary['quarantined_records'] == 0 and len(rows) == 102610
    summary['passed'] = args.repair and args.limit is None and len(rows) == 102610 and summary['verified_records'] + summary['quarantined_records'] == len(rows)
    summary['passed_scope'] = 'All raw records accounted for. Only certified output records may train; quarantined inputs remain excluded and visible.'
    (output / 'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
