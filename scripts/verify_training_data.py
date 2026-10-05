"""Verify all relocated source audits and the complete shared-model training gate."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import json
from pathlib import Path
import yaml
from unifiedgroupgen.preflight import check_training_gate
from unifiedgroupgen.joint import check_joint_lineage

ROOT = Path(__file__).resolve().parents[1]


def main():
    verified = []
    for name in ('mp20_local', 'mpts52_local', 'c2db_official_local', 'joint_space_layer_large'):
        config = yaml.safe_load((ROOT / f'configs/{name}.yaml').read_text(encoding='utf-8'))
        prepared = config['prepared_root']
        train = [ROOT / p for p in config.get('train_files', [f'{prepared}/train.jsonl'])]
        validation = [ROOT / p for p in config.get('val_files', [f'{prepared}/val.jsonl'])]
        check_training_gate(config, train, validation, ROOT)
        verified.append(name)
        print(f'[PASS] {name}', flush=True)
    joint = yaml.safe_load((ROOT / 'configs/joint_space_layer_large.yaml').read_text(encoding='utf-8'))
    lineage = check_joint_lineage(joint)
    report = {'passed': True, 'configurations': verified, 'joint_lineage': lineage,
              'scope': 'Complete bundled inputs and training gates verified; no training or physical certification claim.'}
    path = ROOT / 'artifacts/portable_training_gate.json'
    path.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
