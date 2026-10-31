"""One-off, source-preserving migration to metric-only evaluation modules."""
import ast
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PKG = ROOT/'unifiedgroupgen'
EVAL = PKG/'evaluation'


def functions(name, selected=None):
    source = (EVAL/name).read_text(encoding='utf-8')
    nodes = ast.parse(source).body
    return '\n\n\n'.join(ast.get_source_segment(source, n) if not getattr(n, 'decorator_list', []) else
        '\n'.join(source.splitlines()[n.decorator_list[0].lineno-1:n.end_lineno])
        for n in nodes if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and (selected is None or n.name in selected))


def header(name):
    source = (EVAL/name).read_text(encoding='utf-8')
    nodes = ast.parse(source).body
    return '\n'.join(ast.get_source_segment(source, n) for n in nodes
        if isinstance(n, (ast.Import, ast.ImportFrom)) and
        not (isinstance(n, ast.ImportFrom) and (n.module or '').startswith('unifiedgroupgen.evaluation')))


def put(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip()+'\n', encoding='utf-8')


def main():
    stamp = datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    backup = ROOT/'old_results/code_refactor'/stamp
    snapshot = PKG/'tests/legacy/evaluation_before_tasks'
    if snapshot.exists():
        raise RuntimeError('Migration already applied')
    for name in ('unifiedgroupgen', 'configs'):
        shutil.copytree(ROOT/name, backup/name, ignore=shutil.ignore_patterns('__pycache__', '*.egg-info', 'build'))
    shutil.copytree(EVAL, snapshot, ignore=shutil.ignore_patterns('__pycache__'))
    for path in snapshot.rglob('*.py'):
        put(path, path.read_text(encoding='utf-8').replace('unifiedgroupgen.evaluation',
            'unifiedgroupgen.tests.legacy.evaluation_before_tasks'))
    sources = {p.name: p.read_text(encoding='utf-8') for p in EVAL.glob('*.py')}

    utils = '"""Metric formulas, matching profiles, fingerprints and population statistics."""\n'
    utils += '\n'.join(dict.fromkeys(header(n) for n in ('standard.py','statistics.py','protocols.py')))+'\n\n'
    utils += functions('standard.py')+'\n\n'+functions('statistics.py')+'\n\n'
    stats = sources_stats = (EVAL/'statistics.py').read_text(encoding='utf-8')
    utils += stats[stats.index('DISTRIBUTIONS ='):stats.index('def population_metrics')]+'\n'
    protocols = sources['protocols.py']
    utils += protocols[protocols.index('PROFILES = '):]
    put(EVAL/'utils.py', utils)

    csp = '"""CSP reconstruction and first-1/5/20 candidate matching metrics."""\n'
    csp += 'from collections import defaultdict\nimport numpy as np\nfrom .utils import PROFILES, PROFILE_LABELS, match_pair, numeric, rate\n\n'
    reconstruction = functions('structure_checks.py', {'check_reconstruction'})
    reconstruction = reconstruction.replace("    if task=='csp'", "    from .common import descriptor_metadata\n    if task=='csp'", 1)
    put(EVAL/'csp.py', csp+reconstruction+'\n\n'+functions('summaries.py', {'csp_summary'}))
    put(EVAL/'dng.py', '"""De novo novelty, uniqueness and distribution metrics."""\n'
        'from .utils import rate, population_metrics\n\n'+functions('summaries.py', {'dng_summary'}))
    conditional = sources['properties.py'].replace('unifiedgroupgen.evaluation.statistics', 'unifiedgroupgen.evaluation.utils')
    conditional = conditional.replace('reference_predictions=None):', 'reference_predictions=None, *, enabled=False):')
    conditional = conditional.replace('    available={',
        "    if not enabled:\n        return {'status': 'disabled', 'quantities': {}, 'target_groups': [],\n"
        "                'joint_hit': rate([], False), 'valid_joint_hit': rate([], False)}\n    available={", 1)
    put(EVAL/'conditional.py', conditional)

    common = '"""Common raw-structure validity, group compliance and aggregate generation metrics."""\n'
    common += 'from collections import defaultdict\nfrom dataclasses import dataclass\nimport json\nimport warnings\nimport subprocess\nimport sys\nfrom pathlib import Path\nimport numpy as np\nfrom scipy.optimize import linear_sum_assignment\n'
    common += 'from unifiedgroupgen.common.structures import ReferenceBank, counts_of, formula_key, load_structure, sha256\n'
    common += 'from .utils import (geometry_metrics, layer_structure, PROFILES, PROFILE_LABELS, numeric, rate, frechet_distance, logical_and, composition_validity, fingerprints, structure_match)\n'
    common += 'from .csp import csp_summary, check_reconstruction\nfrom .dng import dng_summary\n\n'
    common += functions('context.py')+'\n\n'
    common += sources['summaries.py'][sources['summaries.py'].index("PROTOCOL ="):sources['summaries.py'].index('def csp_summary')]
    common += functions('structure_checks.py', {'geometry','identify','operation_check','descriptor_metadata','check_declared_orbits','check_requested_chemistry','check_symmetry'})+'\n\n'
    common += functions('summaries.py', {'group_summary','embedding_summary','build_summary'})+'\n\n'
    common = common.replace("source = json.loads(Path(embeddings).read_text(encoding='utf-8'))", 'source = embeddings')
    common += functions('unified.py', {'evaluate_records'})+'\n\n'+functions('validation.py', {'summarize_attempts'})
    put(EVAL/'common.py', common)

    put(PKG/'common/structures.py', sources['io.py']+'\n\n'+
        (PKG/'common/structures.py').read_text(encoding='utf-8').split('def composition_of',1)[0].replace('"""Small structure metadata helpers shared by model-independent callers."""','')+
        'def composition_of'+(PKG/'common/structures.py').read_text(encoding='utf-8').split('def composition_of',1)[1])
    put(PKG/'common/reports.py', sources['reporting.py'])
    put(PKG/'group_constraints/validation.py', header('validation.py')+'\nfrom unifiedgroupgen.evaluation.common import summarize_attempts\n\n'+functions('validation.py', {'to_structure','validate'}))
    shutil.copytree(EVAL/'predictors', PKG/'common/predictors', ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copyfile(EVAL/'resources/public_megnet_models.json', PKG/'common/predictors/public_megnet_models.json')
    service = sources['property_inference.py'].split('def main():')[0]
    main_tail = sources['property_inference.py'].split('    args=parser.parse_args()\n',1)[1].split("\n\nif __name__")[0]
    main_tail = main_tail.replace('parser.error(', 'raise ValueError(')
    put(PKG/'common/predictors/service.py', service+'\n\ndef predict(args):\n'+main_tail)
    # Component checks remain test code, separately labelled from generation performance.
    component = header('inference.py')+'\nfrom torch.nn import functional as F\nfrom unifiedgroupgen.common.structures import read_records, sha256, write_json\nfrom unifiedgroupgen.common.source import file_hash\nfrom unifiedgroupgen.common.configuration import root_path\nROOT=project_root()\n\n'
    component += functions('components.py')+'\n\n'+functions('checkpoint.py', {'check_heldout_inputs'})+'\n\n'+functions('inference.py', {'checked_config','components'})
    component = component.replace('    from unifiedgroupgen.evaluation.checkpoint import check_heldout_inputs\n','').replace('    from unifiedgroupgen.evaluation.components import descriptor_batch\n','')
    component = component.replace("'scope':'Full test;", "'evaluation_type':'model_component_check','scope':'Full test;")
    put(PKG/'tests/components.py', component)
    generation = header('inference.py')+'\nfrom unifiedgroupgen.common.structures import inline_cif_bytes, read_records, record_counts, sha256, write_json\nROOT=project_root()\n\n'
    put(PKG/'generation/tasks.py', generation+functions('inference.py', {'generate'}))

    # Preserve file-based orchestration as run.py functions; metrics remain in-memory.
    runner = '\n\n# Evaluation and generation entry points share the same Hydra configuration.\n'
    runner += 'import csv\nimport os\nimport random\nimport subprocess\nimport sys\nimport hashlib\nimport importlib.metadata\nfrom unifiedgroupgen.common.configuration import load_experiments, runtime_pythonpath, source_key\n'
    runner += 'from unifiedgroupgen.common.structures import ReferenceBank, counts_of, formula_key, inline_cif_bytes, load_structure, read_records, record_counts, sha256, write_json\n'
    runner += 'from unifiedgroupgen.evaluation.common import EvaluationContext, evaluate_records, build_summary, geometry, operation_check\nfrom unifiedgroupgen.evaluation.utils import numeric, logical_and, composition_validity, fingerprints\n'
    runner += 'from unifiedgroupgen.evaluation.conditional import evaluate_properties\nfrom unifiedgroupgen.common.reports import write_tables, figures, campaign_summary\nfrom unifiedgroupgen.common.source import package_sources\nfrom unifiedgroupgen.pl_modules.factory import load_model_checkpoint\nROOT=project_root()\n'
    runner += functions('unified.py', {'reference_metrics','evaluate'})+'\n\n'
    runner = runner.replace("Path(__file__).with_name('structure_checks.py'), Path(__file__).with_name('standard.py'), Path(__file__).with_name('io.py'),",
        "Path(__file__).parent/'evaluation/common.py', Path(__file__).parent/'evaluation/utils.py', Path(__file__).parent/'common/structures.py',")
    runner = runner.replace("Path(__file__).parents[1]/'group_constraints/compiler.py'", "Path(__file__).parent/'group_constraints/compiler.py'")
    runner = runner.replace("Path(__file__).parents[1]/'tests/representation.py'", "Path(__file__).parent/'tests/representation.py'")
    runner = runner.replace('embeddings=None, reference_assessments=None):', 'embeddings=None, reference_assessments=None, *, conditional_enabled=False):')
    runner = runner.replace('properties = evaluate_properties(rows, predictions, metadata, tolerances, reference_assessments)',
        'properties = evaluate_properties(rows, predictions, metadata, tolerances, reference_assessments, enabled=conditional_enabled)')
    runner = runner.replace('report = build_summary(rows, features, reference, properties, context, embeddings)',
        "embedding_data = json.loads(Path(embeddings).read_text(encoding='utf-8')) if embeddings else None\n    report = build_summary(rows, features, reference, properties, context, embedding_data)")
    cli_body = sources['cli.py'].split('    args = parser.parse_args()\n',1)[1].split("\n\nif __name__")[0]
    cli_body = cli_body.replace('parser.error(', 'raise ValueError(')
    cli_body = cli_body.replace('args.embeddings,reference_predictions)', 'args.embeddings,reference_predictions,conditional_enabled=args.conditional_enabled)')
    cli_body = cli_body.replace("if bool(args.assessments)", "if not args.conditional_enabled and (args.assessments or args.assessment_metadata or args.reference_assessments):\n        raise ValueError('Enable evaluation.conditional.enabled for property predictions')\n    if bool(args.assessments)",1)
    runner += '\n\ndef evaluate_task(args):\n'+cli_body+'\n\n'
    runner += functions('planning.py', {'build_jobs','build_plan'})+'\n\n'
    runner += functions('campaign.py', {'run_command','verify_files','output_hashes'})+'\n\n'
    runner += functions('suite.py', {'show_status','verify_generated','write_plan'})+'\n\n'
    suite = sources['suite.py'].split('    parser, args = parse_args()\n',1)[1].split("\n\nif __name__")[0]
    suite = suite.replace('parser.error(', 'raise ValueError(')
    runner += '\n\ndef evaluate_suite(args):\n'+suite+'\n\n'
    # Dispatch implementations are supplied separately below; remove old active entry modules.
    run = (PKG/'run.py').read_text(encoding='utf-8')
    run = run.replace("@hydra.main(version_base='1.3'", runner+"\n\n@hydra.main(version_base='1.3'",1)
    put(PKG/'run.py', run)

    mappings = {
        'evaluation.io':'common.structures','evaluation.reporting':'common.reports',
        'evaluation.standard':'evaluation.utils','evaluation.statistics':'evaluation.utils',
        'evaluation.protocols':'evaluation.utils','evaluation.structure_checks':'evaluation.common',
        'evaluation.summaries':'evaluation.common','evaluation.context':'evaluation.common',
        'evaluation.unified':'run','evaluation.validation':'group_constraints.validation',
        'evaluation.components':'tests.components','evaluation.checkpoint':'tests.legacy.evaluation_before_tasks.checkpoint',
        'evaluation.properties':'evaluation.conditional','evaluation.predictors':'common.predictors',
        'evaluation.property_inference':'tests.legacy.evaluation_before_tasks.property_inference',
        'evaluation.cli':'tests.legacy.evaluation_before_tasks.cli',
        'evaluation.suite':'tests.legacy.evaluation_before_tasks.suite',
        'evaluation.planning':'run','evaluation.campaign':'tests.legacy.evaluation_before_tasks.campaign',
        'evaluation.inference':'tests.legacy.evaluation_before_tasks.inference',
        'evaluation.export':'tests.legacy.evaluation_before_tasks.export',
        'evaluation.import_samples':'tests.legacy.evaluation_before_tasks.import_samples',
    }
    # Update consumers, including archived command registry; frozen reference stays independent.
    for tree in (PKG, ROOT/'scripts', ROOT/'examples', ROOT/'assets'):
        for path in tree.rglob('*.py'):
            if path.is_relative_to(snapshot) or path==Path(__file__).resolve() or path.is_relative_to(EVAL):
                continue
            text = path.read_text(encoding='utf-8')
            for before, after in mappings.items():
                text = text.replace('unifiedgroupgen.'+before, 'unifiedgroupgen.'+after)
            put(path, text)
    for path in EVAL.glob('*.py'):
        if path.name not in ('__init__.py','common.py','csp.py','dng.py','conditional.py','utils.py'):
            # Absolute checked workspace paths; originals already preserved in snapshot and backup.
            assert path.resolve().parent==EVAL.resolve()
            path.unlink()
    for path in (EVAL/'predictors').glob('*'):
        if path.is_file(): path.unlink()
    assert (EVAL/'predictors').resolve().is_relative_to(PKG.resolve())
    shutil.rmtree(EVAL/'predictors')
    (EVAL/'resources/public_megnet_models.json').unlink()
    (EVAL/'__init__.py').write_text('"""Pure metric calculations: common, CSP, DNG and optional conditional evaluation."""\n',encoding='utf-8')
    # Fix explicit relocated package resources and predictor cache-code identities.
    matgl = PKG/'common/predictors/matgl.py'
    text = matgl.read_text(encoding='utf-8').replace("Path(__file__).parents[1]/'resources/public_megnet_models.json'", "Path(__file__).with_name('public_megnet_models.json')")
    put(matgl,text)
    service_path=PKG/'common/predictors/service.py'
    text=service_path.read_text(encoding='utf-8').replace("Path(__file__).parent/'predictors/matgl.py'", "Path(__file__).with_name('matgl.py')")
    text=text.replace("Path(__file__).parent/'predictors/structures.py'", "Path(__file__).with_name('structures.py')")
    text=text.replace("Path(__file__).parent/'io.py'", "Path(__file__).parents[1]/'structures.py'")
    text=text.replace("Path(__file__).parent/'standard.py'", "Path(__file__).parents[2]/'evaluation/utils.py'")
    put(service_path,text)
    # Change only import locations inside new metric sources, never their formulas.
    for path in EVAL.glob('*.py'):
        text=path.read_text(encoding='utf-8').replace('unifiedgroupgen.evaluation.io','unifiedgroupgen.common.structures')
        put(path,text)
    write = ROOT/'intermediate/checks/evaluation_tasks/migration.json'
    put(write,json.dumps({'backup':str(backup),'frozen_reference':str(snapshot),'formula_migration':'AST extraction; no formula changes'},indent=2))
    print(write)


if __name__ == '__main__':
    main()
