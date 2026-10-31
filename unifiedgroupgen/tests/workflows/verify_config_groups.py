"""One-shot preservation receipt for the functional Hydra configuration migration."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from unifiedgroupgen.common.configuration import load_experiments, project_root
from unifiedgroupgen.common.source import file_hash


def verify(root):
    before = json.loads((root/'intermediate/checks/config_groups/before.json').read_text())
    entries = load_experiments('configs/experiments.yaml', root)
    comparisons = {}
    for name, entry in entries.items():
        audit = json.loads((root/'data/dataset/audits/main15'/f'{name}.json').read_text())
        comparisons[name] = {
            'definition_equal': entry['config'] == before['configs'][name],
            'runtime_equal': entry['runtime'] == before['profiles'][name.rsplit('_', 1)[0]],
            'audit_sha256_equal': hashlib.sha256(json.dumps(entry['config'], sort_keys=True).encode()).hexdigest() == audit['config_sha256']}
    changed = [name for name, digest in before['retained_files'].items() if file_hash(root/name) != digest]
    backup = Path(before['backup'])
    def nodes(path):
        return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text(encoding='utf-8')).body
                if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    old, new = nodes(backup/'unifiedgroupgen/run.py'), nodes(root/'unifiedgroupgen/run.py')
    unchanged = {name: new[name] == value for name, value in old.items() if name not in ('execute', 'main')}
    return {'passed': not changed and all(unchanged.values()) and len(comparisons) == 15
            and all(all(row.values()) for row in comparisons.values()),
            'experiments': comparisons, 'preserved_model_files': len(before['retained_files']),
            'changed_model_files': changed, 'training_functions_unchanged': unchanged,
            'backup': str(backup), 'scope': 'Configuration refactor only; no experiment launched'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('outputs/validation/config_groups.json'))
    args = parser.parse_args()
    report = verify(project_root())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
