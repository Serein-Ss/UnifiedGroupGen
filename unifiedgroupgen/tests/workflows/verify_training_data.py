"""Verify all relocated source audits and the complete shared-model training gate."""

from unifiedgroupgen.tests.legacy.paths import project_root, resolve_path, module_command
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import json
from pathlib import Path
import yaml
from unifiedgroupgen.tests.preflight import check_training_gate
from unifiedgroupgen.pl_data.prepare import check_joint_lineage

ROOT = project_root()


def main():
    from unifiedgroupgen.tests.catalog import check_layer_catalog
    catalog = check_layer_catalog()
    print('[PASS] layer catalog: 80 groups, 477 single orbits; PyXtal '+catalog['pyxtal'], flush=True)
    verified = []
    for name in ('mp20_verified_v1', 'mpts52_verified_v1', 'c2db_verified_v1', 'joint_verified_v1'):
        config = yaml.safe_load((resolve_path(ROOT / f'configs/{name}.yaml',ROOT)).read_text(encoding='utf-8'))
        prepared = config['prepared_root']
        train = [resolve_path(ROOT / p,ROOT) for p in config.get('train_files', [f'{prepared}/train.jsonl'])]
        validation = [resolve_path(ROOT / p,ROOT) for p in config.get('val_files', [f'{prepared}/val.jsonl'])]
        check_training_gate(config, train, validation, ROOT)
        verified.append(name)
        print(f'[PASS] {name}', flush=True)
    joint = yaml.safe_load((resolve_path(ROOT / 'configs/joint_verified_v1.yaml',ROOT)).read_text(encoding='utf-8'))
    lineage = check_joint_lineage(joint)
    report = {'passed': True, 'configurations': verified, 'joint_lineage': lineage, 'layer_catalog': catalog,
              'scope': 'Complete bundled inputs and training gates verified; no training or physical certification claim.'}
    path = resolve_path(ROOT / 'intermediate/checks/portable_training_gate.json',ROOT)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
