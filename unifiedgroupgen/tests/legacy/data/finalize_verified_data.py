"""Promote certified records to a new revision, then rebuild audited joint splits."""
from unifiedgroupgen.tests.legacy.paths import package_parent

from unifiedgroupgen.tests.legacy.paths import project_root, resolve_path
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import yaml

ROOT = project_root()
sys.path.insert(0, str(package_parent()))
from unifiedgroupgen.tests.preflight import audit_files
from unifiedgroupgen.pl_data.prepare import prepare_joint

SOURCES = [('mp20_local', 'mp20_verified_v1'), ('mpts52_local', 'mpts52_verified_v1'),
           ('c2db_official_local', 'c2db_verified_v1')]


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--audit', default='artifacts/source_integrity_20261022_v3')
    args = parser.parse_args()
    audit = resolve_path(ROOT/args.audit,ROOT)
    summary = json.loads((audit/'summary.json').read_text(encoding='utf-8'))
    if not summary['complete'] or not summary['passed']:
        raise ValueError('Only an exhaustive accounted audit can be finalized')
    rejected = [json.loads(s) for s in (audit/'repair.jsonl').read_text(encoding='utf-8').splitlines()
                if s.strip() and '"error"' in s]
    for original, name in SOURCES:
        config = yaml.safe_load((resolve_path(ROOT/f'configs/{original}.yaml',ROOT)).read_text(encoding='utf-8'))
        old = resolve_path(ROOT/config['prepared_root'],ROOT)
        config.update(prepared_root=f'artifacts/full_verified_v1/{config["dataset"]}',
                      data_audit=f'artifacts/full_verified_v1/{config["dataset"]}/audit.json',
                      require_source_integrity=True, data_revision='source_integrity_v1')
        target = resolve_path(ROOT/config['prepared_root'],ROOT)
        target.mkdir(parents=True, exist_ok=True)
        (resolve_path(ROOT/f'configs/{name}.yaml',ROOT)).write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
        paths = {}
        for split in ('train', 'val', 'test'):
            source = audit/'corrected'/config['dataset']/f'{split}.jsonl'
            destination = target/f'{split}.jsonl'
            if destination.exists() and destination.read_bytes() != source.read_bytes():
                raise ValueError('Refusing to overwrite a different verified revision')
            shutil.copy2(source, destination)
            records = sum(bool(s.strip()) for s in destination.open(encoding='utf-8'))
            report = json.loads((old/f'{split}.jsonl.report.json').read_text(encoding='utf-8'))
            report.update(accepted=records, rejected=report['input']-records,
                          output=str(destination.resolve()), integrity_protocol=summary['protocol'],
                          integrity_journal_sha256=summary['journal_sha256'])
            write_json(str(destination)+'.report.json', report)
            manifest = json.loads((old/f'{split}.jsonl.manifest.json').read_text(encoding='utf-8'))
            manifest['converter_version'] = 8 if config['kind'] == 'layer' else 3
            write_json(str(destination)+'.manifest.json', manifest)
            quarantine = [r for r in rejected if r['dataset'] == config['dataset'] and r['source_split'] == split]
            Path(str(destination)+'.rejected.jsonl').write_text(''.join(json.dumps({
                'id': r['id'], 'row': r['source_row'], 'reason': r['error']})+'\n' for r in quarantine), encoding='utf-8')
            paths[split] = [destination]
        result = audit_files(paths, config, resolve_path(ROOT/config['data_audit'],ROOT))
        print(json.dumps({'dataset': config['dataset'], 'passed': result['passed'], 'errors': result['errors'][:5]}), flush=True)
        if not result['passed']:
            raise ValueError('Verified source preflight failed')
    config = yaml.safe_load((resolve_path(ROOT/'configs/joint_space_layer_large.yaml',ROOT)).read_text(encoding='utf-8'))
    config.update(prepared_root='artifacts/joint_verified_v1', data_audit='artifacts/joint_verified_v1/audit.json',
                  joint_split_manifest='artifacts/joint_verified_v1/manifest.json',
                  require_source_integrity=True, data_revision='source_integrity_v1',
                  source_configs={dataset: f'configs/{name}.yaml' for dataset, (_, name) in zip(
                      ['mp_20', 'mpts_52', 'c2db_51'], SOURCES)})
    for split in ('train', 'val', 'test'):
        config[f'{split}_files'] = [f'artifacts/joint_verified_v1/{dataset}/{split}.jsonl' for dataset in config['source_configs']]
    (resolve_path(ROOT/'configs/joint_verified_v1.yaml',ROOT)).write_text(yaml.safe_dump(config, sort_keys=False), encoding='utf-8')
    manifest_path = resolve_path(ROOT/config['joint_split_manifest'],ROOT)
    if not manifest_path.exists():
        prepare_joint(config)
    result = audit_files({split: [resolve_path(ROOT/p,ROOT) for p in config[f'{split}_files']] for split in ('train', 'val', 'test')},
                         config, resolve_path(ROOT/config['data_audit'],ROOT))
    print(json.dumps({'joint_passed': result['passed'], 'errors': result['errors'][:5]}), flush=True)
    if not result['passed']:
        raise ValueError('Verified joint preflight failed')
    write_json(resolve_path(ROOT/'artifacts/full_verified_v1/source_integrity_summary.json',ROOT), summary)
    shutil.copy2(audit/'quarantine.jsonl', resolve_path(ROOT/'artifacts/full_verified_v1/quarantine.jsonl',ROOT))
    inputs = {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in (resolve_path(ROOT/'artifacts/full_verified_v1',ROOT)).rglob('*.jsonl')}
    write_json(resolve_path(ROOT/'artifacts/full_verified_v1/verified_inventory.json',ROOT), inputs)


if __name__ == '__main__':
    main()
