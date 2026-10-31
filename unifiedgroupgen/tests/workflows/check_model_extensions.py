"""Collect the bounded v2 checks and independently recheck retained model bytes."""
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def main():
    root = Path(__file__).resolve().parents[3]
    checks = root/'intermediate/checks/model_extensions'
    reports = {}
    for name in ('final_regression', 'backbone_generation', 'public_packed'):
        suite = ET.parse(checks/(name+'.xml')).getroot()
        totals = {key: sum(int(s.get(key, 0)) for s in suite.findall('testsuite'))
                  for key in ('tests', 'failures', 'errors', 'skipped')}
        assert not totals['failures'] and not totals['errors'], (name, totals)
        reports[name] = totals
    original = json.loads((root/'old_results/_audit/retained_models_before.json').read_text(encoding='utf-8'))
    def digest(path):
        h = hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda: stream.read(1024*1024), b''): h.update(chunk)
        return h.hexdigest()
    changed = [p for p, h in original['files'].items() if digest(root/p) != h]
    assert not changed, changed
    legacy = json.loads((checks/'legacy_equivalence/server_comparison.json').read_text(encoding='utf-8'))
    assert legacy['passed']
    result = {'passed': True, 'scope': 'Bounded code/gradient/generation checks, not trained quality',
        'reports': reports, 'retained_files_checked': len(original['files']), 'changed_retained_files': changed,
        'legacy_models_strictly_loaded': len(legacy['strictly_loaded_models']),
        'legacy_real_input_probes': len(legacy['paired_real_record_probes']),
        'new_source_free_models_trained': 0, 'formal_training_started': False,
        'new_configuration_plans': len(json.loads((checks/'config_plans.json').read_text(encoding='utf-8'))),
        'optional_backbones': {'cores_tested': ['DiffCSP CSPNet', 'MatterGen GemNetT', 'FAIRChem EquiformerV2'],
            'equiformer_core_precision': 'float32, including inside AMP',
            'full_upstream_dependency_metadata_clean': False,
            'dependency_report': 'intermediate/checks/model_extensions/optional_pip_check.log',
            'linux_fresh_install_verified': False},
        'source_manifest': {str(p.relative_to(root)).replace('\\','/'): digest(p)
            for p in sorted((root/'unifiedgroupgen').rglob('*.py')) if 'tests' not in p.relative_to(root/'unifiedgroupgen').parts},
        'legacy_report': 'intermediate/checks/model_extensions/legacy_equivalence/server_comparison.json'}
    destination = root/'outputs/validation/model_extensions_v2.json'
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k!='source_manifest'}, ensure_ascii=False, indent=2))


if __name__ == '__main__': main()
