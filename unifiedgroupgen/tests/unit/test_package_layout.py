"""Installation/workspace boundaries introduced by the directory-only refactor."""
from pathlib import Path
import json
import os
import subprocess
import sys

from unifiedgroupgen.tests.legacy.paths import module_command, project_root, resolve_path

ROOT=Path(__file__).resolve().parents[3]


def test_recorded_source_paths_relocate_without_rewriting_data(tmp_path):
    import yaml
    from unifiedgroupgen.common.configuration import resolve_path as current_resolve
    (tmp_path/'configs').mkdir()
    settings = {'recorded_roots': {'E:/old/UnifiedGroupGen': '.', 'E:/old/data': 'data/scr', tmp_path.parent.as_posix(): '.'},
                'mounts': {'artifacts/source_recovery': 'data/scr/source_recovery'}}
    (tmp_path/'configs/paths.yaml').write_text(yaml.safe_dump(settings), encoding='utf-8')
    for resolver in (current_resolve, resolve_path):
        assert resolver(r'E:\old\UnifiedGroupGen\artifacts\source_recovery\c2db_official\train.csv', tmp_path) == tmp_path/'data/scr/source_recovery/c2db_official/train.csv'
        assert resolver('E:/old/data/c2db_51/train.csv', tmp_path) == tmp_path/'data/scr/c2db_51/train.csv'
        current = tmp_path/'artifacts/source_recovery/c2db_official/train.csv'
        assert resolver(current, tmp_path) == tmp_path/'data/scr/source_recovery/c2db_official/train.csv'
        external = Path(tmp_path.anchor)/'external_unmapped.csv'
        assert resolver(external, tmp_path) == external


def test_legacy_input_paths_resolve_without_redirecting_outputs(tmp_path):
    assert resolve_path('artifacts/full_verified_v1/mp_20/train.jsonl',tmp_path)==tmp_path/'data/dataset/full_verified_v1/mp_20/train.jsonl'
    assert resolve_path(tmp_path/'artifacts/source_recovery/c2db_official/train.csv',tmp_path)==tmp_path/'data/scr/source_recovery/c2db_official/train.csv'
    assert resolve_path('../data/mp_20/train.csv',tmp_path)==tmp_path/'data/scr/mp_20/train.csv'
    assert resolve_path('artifacts/runs/main15_seed42/joint_u/model.pt',tmp_path)==tmp_path/'artifacts/runs/main15_seed42/joint_u/model.pt'


def test_workflow_commands_do_not_depend_on_checkout_wrappers():
    old=[sys.executable,'-u','scripts/evaluate_joint_pair.py','--plan']
    assert module_command(old)==[sys.executable,'-u','-m','unifiedgroupgen.tests.legacy.evaluation_workflows.evaluate_joint_pair','--plan']
    archived=[sys.executable,str(ROOT/'artifacts/frozen/scripts/evaluate_joint_pair.py')]
    assert module_command(archived)==archived


def test_installed_workflow_can_plan_outside_checkout(tmp_path):
    env={**os.environ,'UGG_PROJECT_ROOT':str(ROOT)}
    result=subprocess.run([sys.executable,'-m','unifiedgroupgen.tests.legacy.workflows','train_main15','--plan'],
                          cwd=tmp_path,env=env,capture_output=True,text=True)
    assert result.returncode==0,result.stderr
    jobs=json.loads(result.stdout)
    assert len(jobs)==15 and jobs[0]['name']=='mp20_u' and jobs[5]['name']=='joint_all'
    assert not list(tmp_path.iterdir())


def test_workspace_override_does_not_target_installed_package(tmp_path,monkeypatch):
    monkeypatch.setenv('UGG_PROJECT_ROOT',str(tmp_path))
    assert project_root()==tmp_path
    assert resolve_path('artifacts/sample.json')==tmp_path/'artifacts/sample.json'


def test_compiler_constants_created_during_inference_allow_later_backward():
    import torch
    from unifiedgroupgen.group_constraints.compiler import Descriptor,OrbitSpec,compile_descriptor
    compiled=compile_descriptor(Descriptor('space',2,(OrbitSpec('i',14),)))
    with torch.inference_mode():
        constant=compiled.tensor(compiled.coordinate_basis,torch.zeros(compiled.dof))
    assert not torch.is_inference(constant)
    parameter=torch.ones_like(constant,requires_grad=True)
    (parameter*constant).sum().backward()
    torch.testing.assert_close(parameter.grad,constant)
