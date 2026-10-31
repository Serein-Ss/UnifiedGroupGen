"""Read-only provenance and model-math checks for the evaluation refactor."""
import ast
import hashlib
import json
from pathlib import Path

import yaml

from unifiedgroupgen.common.source import file_hash

ROOT=Path(__file__).resolve().parents[3]


def nodes(path):
    return {n.name:ast.dump(n,include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body
            if isinstance(n,(ast.FunctionDef,ast.ClassDef))}


def main():
    location=ROOT/'intermediate/checks/evaluation_tasks'
    migration=json.loads((location/'migration.json').read_text(encoding='utf-8'))
    backup=Path(migration['backup'])
    models=json.loads((ROOT/'old_results/_audit/retained_models_before.json').read_text(encoding='utf-8'))
    changed=[name for name,digest in models['files'].items() if not (ROOT/name).is_file() or file_hash(ROOT/name)!=digest]
    configurations={}
    for path in (ROOT/'configs/main15').glob('*.yaml'):
        config=yaml.safe_load(path.read_text(encoding='utf-8'))
        audit=json.loads((ROOT/'data/dataset/audits/main15'/str(path.stem+'.json')).read_text(encoding='utf-8'))
        configurations[path.stem]={'raw_bytes_unchanged':path.read_bytes()==(backup/'configs/main15'/path.name).read_bytes(),
            'audit_canonical_config_matches':hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()==audit['config_sha256']}
    model_math={}
    for path in (ROOT/'unifiedgroupgen/pl_modules').rglob('*.py'):
        relative=path.relative_to(ROOT)
        before=backup/relative
        model_math[str(relative)]=ast.dump(ast.parse(path.read_text(encoding='utf-8')),include_attributes=False)==ast.dump(
            ast.parse(before.read_text(encoding='utf-8')),include_attributes=False)
    before,after=nodes(backup/'unifiedgroupgen/run.py'),nodes(ROOT/'unifiedgroupgen/run.py')
    training={name:after[name]==before[name] for name in ('UnifiedGroupGenModule','TrainingArtifacts','run_lightning',
        'fit_training_config','load_pretrained_models','train_args','build_callbacks','runtime_options','build_loggers','resume_checkpoint','training_output')}
    result={'retained_model_files':len(models['files']),'changed_model_files':changed,'configurations':configurations,
            'model_math_ast_unchanged':model_math,'training_functions_ast_unchanged':training,
            'note':'Audit config_sha256 is sorted JSON of parsed configuration, not SHA256 of YAML file bytes.',
            'passed':not changed and all(all(v.values()) for v in configurations.values()) and all(model_math.values()) and all(training.values())}
    (location/'preservation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__': main()
