"""Convert all official splits and audit them; never starts model training."""

import argparse
import json
import yaml
from unifiedgroupgen.cli import root_path, artifact_path, ROOT
from unifiedgroupgen.data import prepare_csv
from unifiedgroupgen.preflight import audit_files, check_recovered_source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    config = yaml.safe_load(root_path(args.config).read_text(encoding='utf-8'))
    if config.get('source_verified') is False:
        raise ValueError('Original coordinate/cell source is unverified. Confirm it before full preparation; '
                         'use CLI prepare --limit for explicit diagnostics.')
    check_recovered_source(config, ROOT)
    prepared = config['prepared_root']
    paths = {}
    for split in ('train', 'val', 'test'):
        destination = artifact_path(f'{prepared}/{split}.jsonl')
        manifest = destination.with_suffix(destination.suffix+'.manifest.json')
        report = prepare_csv(root_path(config['data_root'])/f'{split}.csv', destination, config,
                             workers=args.workers, resume=args.resume and manifest.exists())
        print(json.dumps({'split': split, **report}), flush=True)
        paths[split] = [destination]
    report = audit_files(paths, config, artifact_path(config['data_audit']))
    print(json.dumps({'passed': report['passed'], 'errors': len(report['errors']),
                      'audit': config['data_audit']}), flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
