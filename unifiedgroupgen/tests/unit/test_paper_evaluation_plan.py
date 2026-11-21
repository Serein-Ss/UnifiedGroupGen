"""Paired inference and matching use the same held-out candidate denominator."""
import pytest

from scripts.plan_paper_evaluation import catalog, evaluation_plan
from submit_files.joint_flow5090 import main_selection, plan
from unifiedgroupgen.evaluation.joint_csp import match_csp


def test_dng_preview_keeps_training_novelty_and_heldout_reference_separate():
    caches = {s: dict(path=f'{s}.pt',ready=(s != 'val')) for s in ('train','val','test')}
    plan = evaluation_plan(catalog(),'paper','full_mp20_eqv2_dng',42,caches,group=19)
    assert not plan['cache_ready'] and plan['prepare_once'] is None
    assert len(plan['runs']) == 3
    for run in plan['runs']:
        assert run['sample'][-2:] == ['--group','19']
        assert '10042' in run['sample'] and '10000' in run['sample']
        assert 'evaluation.reference_files=[test.pt]' in run['evaluate']
        assert 'evaluation.novelty_files=[train.pt]' in run['evaluate']
        assert 'evaluation.expected_attempts=10000' in run['evaluate']


def test_csp_preview_shares_manifest_and_uses_final_budget_checkpoint():
    caches = {s: dict(path=f'{s}.pt',ready=True) for s in ('train','val','test')}
    plan = evaluation_plan(catalog(),'paper','full_c2db51_eqv2_csp',44,caches,candidates=5)
    assert plan['cache_ready'] and 'test.pt' in plan['prepare_once']
    manifest = plan['prepare_once'][-1]
    for run in plan['runs']:
        assert run['sample'][-2:] == ['--case-manifest',manifest]
        assert 'test.pt' not in run['sample'] and '--group' not in run['sample']
        assert any(item.endswith('/model.pt') for item in run['sample'])
        assert manifest in run['evaluate'] and run['evaluate'][-3:] == ['--ks','1','5']


def test_rod_csp_does_not_silently_use_layer_matching():
    with pytest.raises(ValueError,match='1D matching'):
        match_csp({'kind':'rod'}, {})


def test_main_preview_excludes_ablations_and_includes_mixed_data():
    data = catalog()
    selected = main_selection(data)
    assert len(selected) == 6 and set(selected) == {
        f'full_{domain}_eqv2_{task}'
        for domain in ('mp20','c2db51','joint') for task in ('dng','csp')}
    assert plan(selected)['jobs'] == 6
    with pytest.raises(ValueError, match='Main experiments'):
        main_selection(data, ['full_joint_eqv2_property'])


@pytest.mark.parametrize('task', ['dng','csp'])
def test_mixed_full_model_uses_one_checkpoint_for_two_held_out_domains(task):
    data = catalog()
    model = f'full_joint_eqv2_{task}'
    kind = {'mp_20':'space','c2db_51':'layer'}
    for domain in ('mp_20','c2db_51'):
        caches = {s:dict(path=f'{domain}/{s}.pt',ready=True) for s in ('train','val','test')}
        preview = evaluation_plan(data,'paper',model,42,caches,main=True,domain=domain,split='test')
        assert preview['training_dataset']=='mp20_c2db51' and preview['evaluation_dataset']==domain
        screening = evaluation_plan(data,'paper',model,42,caches,main=True,domain=domain)
        screening_args = screening['runs'][0]['sample']
        assert screening['evaluation_split']=='val'
        assert screening_args[screening_args.index('--output')+1].endswith(f'_{domain}_val/samples.jsonl')
        run, = preview['runs']
        assert f'outputs/{model}/paper/best.ckpt' in run['sample']
        assert run['sample'][run['sample'].index('--output')+1].endswith(f'_{domain}_test/samples.jsonl')
        if task == 'dng':
            assert run['sample'][run['sample'].index('--kind')+1]==kind[domain]
            assert f'evaluation.reference_files=[{domain}/test.pt]' in run['evaluate']
        else:
            assert preview['prepare_once'][preview['prepare_once'].index('--references')+1]==f'{domain}/test.pt'
            assert '--case-manifest' in run['sample'] and f'{domain}/test.pt' not in run['sample']
