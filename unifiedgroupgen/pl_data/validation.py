"""Verify current tensor caches and their explicit training audits."""

import hashlib
import json
from pathlib import Path

from unifiedgroupgen.common.source import file_hash
from unifiedgroupgen.common.structures import write_json
from unifiedgroupgen.pl_data.cache import load_tensor_cache


def audit_files(paths,config,output):
    """Cache identities and split completeness, without repeating structure validation."""
    seen={};errors=[];inputs={};counts={}
    for split,files in paths.items():
        inputs[split]=[];counts[split]=0
        for path in files:
            payload=load_tensor_cache(path)
            if payload['identity']['split']!=split:errors.append(dict(type='split_mismatch',path=str(path)))
            if not payload['records']:errors.append(dict(type='empty_split',path=str(path)))
            for row in payload['records']:
                identity=(row['dataset'],row['id'])
                if identity in seen:errors.append(dict(type='duplicate_identity',id=row['id']))
                seen[identity]=split
            counts[split]+=len(payload['records'])
            inputs[split].append(dict(path=str(Path(path).resolve()),sha256=file_hash(path)))
    report=dict(schema='ugg_tensor_cache_audit_v1',passed=not errors,errors=errors,inputs=inputs,records=counts,
                config_sha256=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest(),
                scope='Cache metadata and split identities; no new geometry or property validation')
    write_json(output, report)
    return report


def verify_training_audit(path, config, splits):
    report = json.loads(Path(path).read_text(encoding='utf-8'))
    digest = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    if report.get('schema') != 'ugg_tensor_cache_audit_v1' or not report.get('passed'):
        raise ValueError(f'Expected a passing Tensor cache audit: {path}')
    if report['config_sha256'] != digest:
        raise ValueError(f'Tensor cache audit belongs to a different experiment: {path}')
    for split, paths in splits.items():
        expected = [{'path': str(p.resolve()), 'sha256': file_hash(p)} for p in paths]
        if expected != report['inputs'][split]:
            raise ValueError(f'{split} caches changed since the audit: {path}')
