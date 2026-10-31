"""Bind the maintained CUDA environment, numerical checks and unchanged trained outputs."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET


def digest(path):
    value=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(1024*1024),b''):value.update(chunk)
    return value.hexdigest()


def main():
    root=Path(__file__).resolve().parents[3];checks=root/'intermediate/checks/environment_precision'
    environment=json.loads((checks/'environment.json').read_text(encoding='utf-8'))
    assert environment['passed']
    pip=subprocess.run([sys.executable,'-m','pip','check'],capture_output=True,text=True)
    assert pip.returncode==0,pip.stdout+pip.stderr
    reports={}
    for name in ('regression','precision_final','generation_final','metrics_recheck'):
        tree=ET.parse(checks/(name+'.xml')).getroot()
        counts={key:sum(int(s.get(key,0)) for s in tree.findall('testsuite'))
                for key in ('tests','failures','errors','skipped')}
        assert not counts['failures'] and not counts['errors'],(name,counts)
        reports[name]=counts
    initial=ET.parse(checks/'metrics_initial.xml').getroot()
    cases=initial.findall('.//testcase')
    failures=[case for case in cases if case.find('failure') is not None]
    assert len(failures)==1 and failures[0].get('name')=='test_cli_png_manifest_resume'
    assert 'another identity' in failures[0].find('failure').text
    recheck=ET.parse(checks/'metrics_recheck.xml').getroot().findall('.//testcase')
    assert len(recheck)==1 and recheck[0].get('name')==failures[0].get('name')
    assert recheck[0].get('classname')==failures[0].get('classname')
    reports['metrics_initial']={'tests':len(cases),'passed':len(cases)-1,'failures':1,
        'reason':'Source changed during the initial run; identity guard rejected reuse before the expected corrupted-image check.',
        'rechecked_case':'test_cli_png_manifest_resume','raw_report_preserved':True}
    legacy=json.loads((checks/'legacy_equivalence/server_comparison.json').read_text(encoding='utf-8'))
    assert legacy['passed'] and len(legacy['strictly_loaded_models'])==15
    protected=json.loads((root/'old_results/_audit/retained_models_before.json').read_text(encoding='utf-8'))['files']
    changed=[name for name,value in protected.items() if digest(root/name)!=value]
    assert not changed,changed
    wheel=json.loads((checks/'wheel_resources.json').read_text())
    assert wheel['passed']
    benchmark=json.loads((checks/'all_backbones_precision.json').read_text())
    for value in benchmark['models'].values():
        assert value['fp32']['finite_gradients'] and value['bf16']['finite_gradients']
        assert value['gradient_cosine']>.99
    files=['unifiedgroupgen/environment.yml','unifiedgroupgen/pyproject.toml',
        'unifiedgroupgen/common/precision.py','unifiedgroupgen/pl_modules/orbitflow.py',
        'unifiedgroupgen/generate.py',
        'unifiedgroupgen/pl_modules/backbones/official/vendor_manifest.json',
        'unifiedgroupgen/pl_modules/backbones/mattergen.py','unifiedgroupgen/pl_modules/backbones/equiformer_v2.py',
        'configs/learning/unified.yaml','configs/generation/default.yaml',
        'docs/ENVIRONMENT.md','docs/MODEL_ABLATIONS_V2.md','docs/UNIFIEDGROUPGEN_MANUSCRIPT_ZH.md']
    result={'passed':True,'environment_name':'unifiedgroupgen_cuda','python':sys.executable,
        'environment':environment,'pip_check':pip.stdout.strip(),'reports':reports,
        'test_counts_note':'Groups overlap; do not add them as unique cases.',
        'backbone_distribution':'Pinned official computation subsets with relocated imports and licenses; full upstream trainers not installed',
        'precision_policy':'BF16 neural AMP; FP32 weights/state/geometry/constraints/loss; FP32/FP64 metric protocols',
        'reference_precision':json.loads((checks/'reference_precision.json').read_text()),
        'reduced_core_precision':benchmark,'wheel_resources':wheel,
        'legacy_models_checked':15,'legacy_real_input_probes':len(legacy['paired_real_record_probes']),
        'protected_files_checked':len(protected),'changed_protected_files':changed,
        'linux_fresh_install_executed':False,'formal_tasks_started':False,'new_quality_claim':False,
        'manuscript_updates':['6.9','7.1-7.7','10.5.3-10.5.7'],
        'files':{p:digest(root/p) for p in files}}
    destination=root/'outputs/validation/environment_precision.json'
    destination.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in result.items() if k not in ('environment','files','reference_precision','reduced_core_precision')},ensure_ascii=False,indent=2))


if __name__=='__main__':main()
