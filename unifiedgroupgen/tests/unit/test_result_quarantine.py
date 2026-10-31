import json

import pytest

from unifiedgroupgen.tests.workflows.archive_model_results import checked_paths, move_results, digest, filesystem


def test_quarantine_preserves_bytes_and_current_model(tmp_path):
    (tmp_path/'outputs/current').mkdir(parents=True)
    current=tmp_path/'outputs/current/model.pt'; current.write_bytes(b'current')
    (tmp_path/'outputs/old').mkdir()
    old=tmp_path/'outputs/old/model.pt';old.write_bytes(b'old')
    plan={'protected':['outputs/current'],'moves':[{'source':'outputs/old','reason':'old model'}]}
    expected=digest(old)
    result=move_results(tmp_path,tmp_path/'old_results',plan)
    assert result['state']=='verified' and result['files']==1
    assert digest(tmp_path/'old_results/outputs/old/model.pt')==expected
    assert current.read_bytes()==b'current' and not old.exists()
    assert json.loads((tmp_path/'old_results/migration.json').read_text())['original_files_deleted'] is False


@pytest.mark.parametrize('source', ['outputs/current','outputs','../outside','.git'])
def test_quarantine_rejects_protected_or_outside_targets(tmp_path,source):
    (tmp_path/'outputs/current').mkdir(parents=True)
    (tmp_path/'.git').mkdir()
    plan={'protected':['outputs/current'],'moves':[{'source':source,'reason':'invalid'}]}
    with pytest.raises(ValueError):checked_paths(tmp_path,tmp_path/'old_results',plan)
    assert (tmp_path/'outputs/current').exists()


def test_resume_verifies_an_already_moved_directory(tmp_path):
    archive=tmp_path/'old_results';(archive/'old').mkdir(parents=True)
    target=archive/'old/result.bin';target.write_bytes(b'unchanged')
    receipt={'source':'old','state':'prepared','files':[{'source':'old/result.bin','bytes':9,'sha256':digest(target)}]}
    (archive/'moves.jsonl').write_text(json.dumps(receipt)+'\n',encoding='utf-8')
    plan={'protected':[],'moves':[{'source':'old','reason':'old model'}]}
    assert move_results(tmp_path,archive,plan,resume=True)['state']=='verified'
    assert target.read_bytes()==b'unchanged'


def test_long_nested_artifact_is_preserved(tmp_path):
    source=tmp_path/'old'
    target=source/('/'.join(['nested_segment']*12))/'result.bin'
    filesystem(target.parent).mkdir(parents=True)
    filesystem(target).write_bytes(b'long-path')
    plan={'protected':[],'moves':[{'source':'old','reason':'test artifact'}]}
    result=move_results(tmp_path,tmp_path/'old_results',plan)
    assert result['files']==1
    assert filesystem(tmp_path/'old_results'/target.relative_to(tmp_path)).read_bytes()==b'long-path'
