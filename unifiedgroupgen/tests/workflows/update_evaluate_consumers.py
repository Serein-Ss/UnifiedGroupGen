"""Migrate maintained consumers to independent generation and metric entries."""
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]


def main():
    folders=['unifiedgroupgen/tests/unit','unifiedgroupgen/tests/reference_workflows','scripts','examples','assets']
    for directory in folders:
        for path in (ROOT/directory).rglob('*.py'):
            text=path.read_text(encoding='utf-8')
            changed=text.replace('from unifiedgroupgen.run import evaluate','from unifiedgroupgen.evaluate import evaluate')
            changed=changed.replace('from unifiedgroupgen.run import build_jobs','from scripts.evaluate_models import build_jobs')
            if changed!=text: path.write_text(changed,encoding='utf-8')
    path=ROOT/'unifiedgroupgen/tests/unit/test_evaluation_tasks.py'
    text=path.read_text(encoding='utf-8').replace('from unifiedgroupgen import run','from unifiedgroupgen import evaluate as metric_entry')
    text=text.replace('run.evaluate(', 'metric_entry.evaluate(')
    text=text.replace("[sys.executable,'-m','unifiedgroupgen.run','--config-path',str(folder),",
        "[sys.executable,'-m',{'evaluate':'unifiedgroupgen.evaluate','evaluate_suite':'scripts.evaluate_models'}[config['task']],'--config-path',str(folder),")
    text=text.replace("    options['resume']=True\n",'')
    text=text.replace("assert result.returncode==0 and 'Verified existing evaluation' in result.stdout,result.stderr",
        "assert result.returncode!=0 and 'Completed metrics already exist' in result.stderr")
    text=text.replace("assert result.returncode!=0 and 'modified' in result.stderr",
        "assert result.returncode!=0 and 'Completed metrics already exist' in result.stderr")
    text=text.replace("options.update(action='run',stage='generate',resume=True)","options.update(action='run',stage='generate',skip_completed=True)")
    path.write_text(text,encoding='utf-8')
    path=ROOT/'unifiedgroupgen/tests/check_public_predictors.py'
    path.write_text(path.read_text(encoding='utf-8').replace('from unifiedgroupgen.run import execute',
        'from unifiedgroupgen.evaluate import execute'),encoding='utf-8')


if __name__=='__main__': main()
