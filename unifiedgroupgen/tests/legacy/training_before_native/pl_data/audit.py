"""Audit configured experiments without selecting datasets or changing training inputs."""
import argparse
import json

from unifiedgroupgen.common.configuration import load_experiments
from unifiedgroupgen.common.configuration import project_root, resolve_path
from unifiedgroupgen.tests.preflight import audit_files, check_training_gate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiments', required=True)
    parser.add_argument('--models', nargs='+')
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = project_root()
    entries = load_experiments(args.experiments, root)
    for name in args.models or entries:
        config = entries[name]['config']
        paths = {split: [resolve_path(p, root) for p in config[split+'_files']]
                 for split in ('train', 'val', 'test')}
        if not args.verify_only:
            report = audit_files(paths, config, resolve_path(config['data_audit'], root))
            if not report['passed']:
                raise ValueError(f'{name}: {report["errors"]}')
        check_training_gate(config, paths['train'], paths['val'], root)
        print(json.dumps({'experiment': name, 'passed': True}), flush=True)


if __name__ == '__main__':
    main()
