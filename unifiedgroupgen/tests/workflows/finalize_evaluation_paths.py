"""Update historical aliases and verification consumers after the metric-module move."""
import json
import runpy
from pathlib import Path

ROOT=Path(__file__).resolve().parents[3]
PKG=ROOT/'unifiedgroupgen'
OLD='tests/legacy/evaluation_before_tasks/'


def main():
    for name,key in [('layout.py','MODULE_PATHS'),('workflow_registry.py','WORKFLOWS')]:
        path=PKG/'tests/legacy'/name
        values=runpy.run_path(str(path))[key]
        for label,target in values.items():
            file=target if key=='MODULE_PATHS' else target.replace('.','/')+'.py'
            if not (PKG/file).is_file() and file.startswith('evaluation/'):
                replacement=OLD+file.split('/',1)[1]
                if (PKG/replacement).is_file():
                    values[label]=replacement if key=='MODULE_PATHS' else replacement[:-3].replace('/','.')
        path.write_text('"""Historical command aliases, only for archived reproduction tests."""\n'+key+' = '+repr(values)+'\n',encoding='utf-8')
    # The original public predictor tests exercise its service directly, not the removed CLI.
    path=PKG/'tests/unit/test_public_predictors.py'
    text=path.read_text(encoding='utf-8').replace('from unifiedgroupgen.evaluation import property_inference',
        'from unifiedgroupgen.common.predictors import service as property_inference')
    text=text.replace('from unifiedgroupgen.evaluation import cli',
        'from unifiedgroupgen.tests.legacy.evaluation_before_tasks import cli')
    # Only the old argparse validation test remains a historical command test.
    text=text.replace('with pytest.raises(SystemExit): property_inference.main()',
        'with pytest.raises(SystemExit):\n        from unifiedgroupgen.tests.legacy.evaluation_before_tasks.property_inference import main\n        main()')
    path.write_text(text,encoding='utf-8')
    path=PKG/'tests/unit/test_unified_evaluation.py'
    text=path.read_text(encoding='utf-8').replace("monkeypatch.setattr('unifiedgroupgen.run.fingerprints'",
        "monkeypatch.setattr('unifiedgroupgen.evaluation.common.fingerprints'")
    text=text.replace("monkeypatch.setattr('unifiedgroupgen.run.composition_validity'",
        "monkeypatch.setattr('unifiedgroupgen.evaluation.common.composition_validity'")
    text=text.replace("{'band_gap':.2})", "{'band_gap':.2},enabled=True)")
    text=text.replace("{'r':{'band_gap':2.}})", "{'r':{'band_gap':2.}},enabled=True)")
    path.write_text(text,encoding='utf-8')
    path=PKG/'tests/unit/test_environment_configuration.py'
    path.write_text(path.read_text(encoding='utf-8').replace("{'ugg-train','ugg-generate','ugg-evaluate'}", "{'ugg'}"),encoding='utf-8')
    path=PKG/'tests/unit/test_layered_package.py'
    path.write_text(path.read_text(encoding='utf-8').replace("ROOT/'unifiedgroupgen/evaluation/cli.py'",
        "ROOT/'unifiedgroupgen/tests/legacy/evaluation_before_tasks/cli.py'"),encoding='utf-8')
    # Remove the remaining second public generation entry, preserving it as a historical command.
    path=PKG/'generation/cli.py'
    if path.exists():
        destination=PKG/'tests/legacy/evaluation_before_tasks/generation_cli.py'
        destination.write_text(path.read_text(encoding='utf-8').replace('from .pipeline import sample',
            'from unifiedgroupgen.generation.pipeline import sample'),encoding='utf-8')
        path.unlink()
    for path in PKG.rglob('*.py'):
        if path==Path(__file__).resolve(): continue
        text=path.read_text(encoding='utf-8')
        updated=text.replace('from unifiedgroupgen.generation.cli import',
            'from unifiedgroupgen.tests.legacy.evaluation_before_tasks.generation_cli import')
        if updated!=text: path.write_text(updated,encoding='utf-8')
    print('Historical aliases and metric consumers updated')


if __name__=='__main__': main()
