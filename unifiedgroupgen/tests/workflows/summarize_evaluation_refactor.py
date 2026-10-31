"""Merge regression/rerun evidence without presenting reruns as a fresh full run."""
import ast
from datetime import datetime
import json
from pathlib import Path
import xml.etree.ElementTree as ET

ROOT=Path(__file__).resolve().parents[3]


def main():
    directory=ROOT/'intermediate/checks/evaluation_tasks'
    files=['full.xml','tasks_final.xml','extra.xml','joint_retry.xml','scaler_retry.xml']
    latest={}
    runs=[]
    for name in files:
        cases=ET.parse(directory/name).findall('.//testcase')
        failed=[]
        for case in cases:
            key=case.get('classname').replace('.','/')+'.py::'+case.get('name')
            status=('failed' if case.find('failure') is not None or case.find('error') is not None else
                    'skipped' if case.find('skipped') is not None else 'passed')
            latest[key]={'state':status,'evidence':name}
            if status=='failed': failed.append(key)
        runs.append({'file':name,'cases':len(cases),'failed':failed})
    expected={s.strip() for s in (directory/'current_collection.log').read_text(encoding='utf-8').splitlines() if '::' in s}
    uncovered=sorted(expected-set(latest))
    unresolved={k:v for k,v in latest.items() if v['state']=='failed'}
    preserved=json.loads((directory/'preservation.json').read_text(encoding='utf-8'))
    migration=json.loads((directory/'migration.json').read_text(encoding='utf-8'))
    backup=Path(migration['backup'])
    group_math={p.name:ast.dump(ast.parse(p.read_text(encoding='utf-8')),include_attributes=False)==
        ast.dump(ast.parse((ROOT/p.relative_to(backup)).read_text(encoding='utf-8')),include_attributes=False)
        for p in (backup/'unifiedgroupgen/group_constraints').glob('*.py')}
    public=json.loads((directory/'public_weights/integration.json').read_text(encoding='utf-8'))
    report={'verified_at':datetime.now().astimezone().isoformat(),
        'passed':not uncovered and not unresolved and preserved['passed'] and public['passed'] and all(group_math.values()),
        'scope':'Local metric/entry-point refactor; no formal retraining, benchmark regeneration, server or website actions',
        'metric_modules':['common.py','csp.py','dng.py','conditional.py','utils.py'],
        'entrypoint':'unifiedgroupgen.run:main', 'console_command':'ugg',
        'conditional_evaluation_default':False,'legacy_archive':migration['frozen_reference'],
        'regression':{'unique_cases':len(latest),'passed':sum(v['state']=='passed' for v in latest.values()),
            'skipped':sum(v['state']=='skipped' for v in latest.values()),'unresolved':unresolved,'uncovered':uncovered,
            'runs':runs,'aggregation':'Full run plus explicitly recorded targeted reruns, not a second full green run',
            'initial_failure_explanations':{
                'legacy_joint_resume':'Source changed during initial regression; strict reuse guard rejected it. Fresh isolated rerun passed.',
                'frozen_fingerprint_probe':'Test still opened removed standard.py; updated to utils.py and reran successfully.'}},
        'preservation':preserved,'group_math_ast_unchanged':group_math,
        'real_public_predictor_integration':{'passed':public['passed'],
            'cases':list(public['cases']),'scope':public['scope'],
            'max_batch_vs_native_absolute_difference':public['max_batch_vs_official_single_absolute_difference']},
        'checks_directory':str(directory.relative_to(ROOT)), 'backup':migration['backup'],
        'limits':['Local Windows/CPU and existing CUDA tests; no new Linux environment installation',
                  'No new scientific generation-quality conclusions; input/source-domain semantics unchanged']}
    path=ROOT/'outputs/validation/evaluation_task_refactor.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({'passed':report['passed'],'regression':{k:report['regression'][k] for k in
        ('unique_cases','passed','skipped','unresolved','uncovered')},'report':str(path)},ensure_ascii=False))
    if not report['passed']: raise SystemExit(1)


if __name__=='__main__': main()
