"""Read-only preservation check for the native-Lightning directory migration."""
import argparse
import gc
import json
from pathlib import Path

import torch
from unifiedgroupgen.common.source import file_hash
from unifiedgroupgen.common.configuration import project_root
from unifiedgroupgen.pl_modules.factory import build_models, load_model_checkpoint


def audit(root, backup):
    before = json.loads((root/'old_results/_audit/retained_models_before.json').read_text(encoding='utf-8'))
    inventory = json.loads((root/'outputs/model_inventory.json').read_text(encoding='utf-8'))
    changed = [name for name, digest in before['files'].items()
               if not (root/name).is_file() or file_hash(root/name) != digest]
    configs = sorted((root/'configs/main15').glob('*.yaml'))
    changed_configs = [str(p.relative_to(root)) for p in configs
                       if file_hash(p) != file_hash(backup/p.relative_to(root))]
    loaded = []
    for row in inventory['models']:
        saved = load_model_checkpoint(root/row['directory']/'model.best.pt')
        proposal, flow = build_models(saved['config'], 'cpu')
        proposal.load_state_dict(saved['proposal'], strict=True)
        flow.load_state_dict(saved['flow'], strict=True)
        loaded.append(row['name'])
        del saved, proposal, flow
        gc.collect()
    return {'passed': not changed and not changed_configs and len(loaded) == 15,
            'preserved_files': len(before['files']), 'changed_model_files': changed,
            'checked_main15_configs': len(configs), 'changed_configs': changed_configs,
            'strictly_loaded_models': loaded,
            'scope': 'No training, generation, server access or checkpoint rewrite'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backup', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    report = audit(project_root(), args.backup)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report))
    if not report['passed']: raise SystemExit(1)


if __name__ == '__main__': main()
