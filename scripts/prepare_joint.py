"""Prepare all source records into joint material-grouped splits, then audit the result."""

import argparse
import json
import yaml
from unifiedgroupgen.cli import root_path, artifact_path
from unifiedgroupgen.joint import prepare_joint
from unifiedgroupgen.preflight import audit_files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/joint_space_layer_large.yaml')
    parser.add_argument('--resume', action='store_true', help='Rebuild unfinished outputs only when no final manifest exists')
    args = parser.parse_args()
    config = yaml.safe_load(root_path(args.config).read_text(encoding='utf-8'))
    manifest = prepare_joint(config, args.resume)
    paths = {split: [root_path(p) for p in config[f'{split}_files']] for split in ('train', 'val', 'test')}
    report = audit_files(paths, config, artifact_path(config['data_audit']))
    print(json.dumps({'passed': report['passed'], 'split_report': manifest['report'], 'errors': report['errors']}), flush=True)
    if not report['passed']:
        raise SystemExit(2)


if __name__ == '__main__':
    main()
