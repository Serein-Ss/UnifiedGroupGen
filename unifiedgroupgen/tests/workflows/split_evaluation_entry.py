"""Source-preserving separation of training, generation and metric entry points."""
import ast
from datetime import datetime
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[3]
PKG=ROOT/'unifiedgroupgen'


def main():
    source=(PKG/'run.py').read_text(encoding='utf-8')
    nodes={n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body
           if isinstance(n,(ast.FunctionDef,ast.ClassDef))}
    backup=ROOT/'old_results/code_refactor'/datetime.now().strftime('%y-%m-%d-%H-%M-%S')
    for folder in ('unifiedgroupgen','configs','scripts','docs'):
        shutil.copytree(ROOT/folder,backup/folder,ignore=shutil.ignore_patterns('__pycache__','*.egg-info','build','node_modules'))
    shutil.copyfile(PKG/'run.py',PKG/'tests/legacy/training_and_evaluation_before_split.py')
    def save(path,text):
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(text.rstrip()+'\n',encoding='utf-8')
    training=source[:source.index('# Evaluation and generation entry points share')]
    training=training.replace("    if cfg.task not in ('train', 'plan'):\n        return execute_analysis(cfg)\n",'')
    training+='''@hydra.main(version_base='1.3', config_path=str(project_root()/'configs'), config_name='run')
def main(cfg: DictConfig):
    execute(cfg)


if __name__ == '__main__': main()
'''
    save(PKG/'run.py',training)
    header='''"""Calculate metrics from existing structures; this entry never samples structures."""
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import random
import time
from types import SimpleNamespace
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
from unifiedgroupgen.common.configuration import project_root, resolve_path, source_key
from unifiedgroupgen.common.structures import ReferenceBank, counts_of, load_structure, read_records, sha256, write_json
from unifiedgroupgen.evaluation.common import EvaluationContext, evaluate_records, build_summary, geometry, descriptor_metadata
from unifiedgroupgen.evaluation.utils import numeric, logical_and, composition_validity, fingerprints
from unifiedgroupgen.evaluation.conditional import evaluate_properties
from unifiedgroupgen.common.reports import write_tables, figures
ROOT=project_root()

'''
    evaluator=header+'\n\n\n'.join(nodes[n] for n in ('reference_metrics','evaluate','evaluate_task'))
    evaluator=evaluator.replace("    from unifiedgroupgen.common.source import package_sources\n    code = package_sources()",
        "    code = metric_sources()")
    evaluator=evaluator.replace("k not in ('resume','output','cache')", "k not in ('output','cache')")
    start=evaluator.index('    if manifest.exists():',evaluator.index('def evaluate_task'))
    end=evaluator.index('    def prediction_csv',start)
    evaluator=evaluator[:start]+'''    if manifest.exists():
        raise ValueError('Completed metrics already exist. The campaign may verify and skip them; choose a new output directory to recompute.')
'''+evaluator[end:]
    save(PKG/'evaluate.py',evaluator)
    # Generation owns checkpoint loading, sample continuation and all stochastic integration.
    generator='''"""Generate structures; metric calculation is a separate evaluate.py command."""
from pathlib import Path
import json
import random
from types import SimpleNamespace
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
from unifiedgroupgen.common.configuration import project_root, resolve_path, source_key
from unifiedgroupgen.common.structures import read_records, sha256, write_json, export_cif
from unifiedgroupgen.common.source import package_sources
from unifiedgroupgen.pl_modules.factory import build_models, load_model_checkpoint
from unifiedgroupgen.tests.components import checked_config
ROOT=project_root()

'''
    inference=nodes['inference_task'].replace('def inference_task(args):','def generate_job(args):')
    inference=inference.replace('from unifiedgroupgen.tests.components import checked_config, components','from unifiedgroupgen.tests.components import checked_config')
    inference=inference.replace("    if job['task'] == 'components':\n        return components(proposal, flow, saved, args.output, args.device, args.batch_size, args.seed)\n",'')
    save(PKG/'generate.py',generator+inference)
    # The optional experiment campaign belongs to scripts, not to training or metric modules.
    scheduler='''"""Coordinate generation, component checks and metrics via separate entry points."""
from pathlib import Path
from types import SimpleNamespace
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
import hydra
import torch
import numpy as np
from omegaconf import OmegaConf
from unifiedgroupgen.common.configuration import project_root, resolve_path, source_key, load_experiments, RunSettings, runtime_pythonpath
from unifiedgroupgen.common.source import package_sources
from unifiedgroupgen.common.structures import inline_cif_bytes, read_records, sha256, write_json
from unifiedgroupgen.common.reports import campaign_summary
from unifiedgroupgen.pl_modules.factory import load_model_checkpoint
ROOT=project_root()
TASKS=('components','dng','csp','fixed_descriptor','group')

'''
    scheduler+='\n\n\n'.join(nodes[n] for n in ('build_jobs','build_plan','run_command','verify_files','output_hashes',
        'show_status','verify_generated','write_plan','evaluate_suite','run_worker','evaluate_metrics','run_job'))
    scheduler=scheduler.replace("['-m','unifiedgroupgen.run','--config-path',target,'--config-name',config_path.stem]",
        "['-m', {'generate':'unifiedgroupgen.generate','check_components':'unifiedgroupgen.tests.components',\n                             'evaluate':'unifiedgroupgen.evaluate','predict_properties':'unifiedgroupgen.evaluate'}[task],\n                     '--config-path',target,'--config-name',config_path.stem]")
    save(ROOT/'scripts/evaluate_models.py',scheduler)
    save(ROOT/'intermediate/checks/evaluate_entry/migration.json',json.dumps({'backup':str(backup),
        'scope':'Entry point separation; model math and stored scientific results unchanged'},indent=2))
    print(backup)


if __name__=='__main__': main()
