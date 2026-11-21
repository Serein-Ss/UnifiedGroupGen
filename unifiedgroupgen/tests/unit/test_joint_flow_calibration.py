"""Batch calibration policy and orchestration only; no CUDA trials or training."""
import copy
import sys
import pytest
import torch
from torch.utils.data import BatchSampler, RandomSampler
from submit_files import joint_flow5090_calibrate as calibration

POLICY=dict(candidates=[1,8,16],memory_fraction=.85,reserve_gib=2,
            warmup_steps=2,stress_times=[.05,.75,.95])


def passed(batch,speed):
    return dict(state='pass',batch_size=batch,records_per_second=speed,
                precision='fp32',peak_reserved_bytes=100,memory_limit_bytes=200)


def test_selection_prefers_measured_throughput_over_largest_batch():
    trials=[passed(1,2.),passed(8,7.),passed(16,6.),
            dict(state='oom',batch_size=32),dict(state='memory_margin_failed',records_per_second=100.)]
    assert calibration.select_trial(trials)['batch_size']==8
    with pytest.raises(RuntimeError):calibration.select_trial(trials[3:])


def test_workload_uses_actual_batch_sampler_and_stresses_large_records():
    sizes=[2,3,4,8,9,10,2,2,2,3,4,4]
    cases=calibration.workloads(sizes,4,POLICY)
    assert len(cases)==11 and sum(c['timed'] for c in cases)==5
    assert cases[0] is not cases[1]
    assert {c['time'] for c in cases if c['time'] is not None}=={.05,.75,.95}
    expected = list(BatchSampler(RandomSampler(sizes, generator=torch.Generator().manual_seed(42)), 4, False))
    assert all(len(c['indices']) == 4 for c in cases)
    assert all(c['indices'] in expected for c in cases if c['timed'])
    assert all(c['indices'] == [5,4,3,2] for c in cases if c['time'] is not None)


def test_profile_reuse_requires_exact_identity_and_memory_criteria():
    expected=dict(policy=POLICY,revision='a');profile=dict(state='complete',identity=expected,selected=passed(8,7))
    assert calibration.validate_profile(profile,expected)['batch_size']==8
    for key,value in [('identity',{'revision':'b'}),('state','running')]:
        changed={**profile,key:value}
        with pytest.raises(ValueError):calibration.validate_profile(changed,expected)
    changed=copy.deepcopy(profile);changed['selected']['peak_reserved_bytes']=201
    with pytest.raises(ValueError):calibration.validate_profile(changed,expected)


def test_each_candidate_runs_in_a_separate_mock_subprocess_and_completed_profile_reuses(tmp_path,monkeypatch):
    campaign=dict(name='test',revision='a',entries={'model':{'experiment':'fixture'}},calibration=POLICY)
    manifest=tmp_path/'campaign.json';calibration.write(manifest,campaign)
    calibration.write(tmp_path/'audits/model.json',dict(config_sha256='hash',inputs={'train':[]}))
    monkeypatch.setattr(calibration,'prepare_inputs',lambda c,m,folder:calibration.write(folder/'inputs.json',{}))
    calls=[]
    def subprocess_only(command,**kwargs):
        calls.append(command);index=int(command[command.index('--index')+1]);batch=POLICY['candidates'][index]
        calibration.write(command[command.index('--output')+1],passed(batch,[1.,8.,5.][index]))
    monkeypatch.setattr(calibration.subprocess,'run',subprocess_only)
    calibration.calibrate(manifest,'model')
    assert len(calls)==3 and all('trial' in c for c in calls)
    assert len({c[c.index('--index')+1] for c in calls})==3
    saved=calibration.read(tmp_path/'profiles/model.json')
    assert saved['selected']['batch_size']==8 and saved['temporary_weights_discarded']
    calibration.calibrate(manifest,'model');assert len(calls)==3
    calibration.write(tmp_path/'audits/model.json',dict(config_sha256='changed',inputs={'train':[]}))
    with pytest.raises(ValueError):calibration.calibrate(manifest,'model')


def test_cli_refuses_unallocated_local_calibration(monkeypatch):
    monkeypatch.delenv('SLURM_JOB_ID',raising=False);monkeypatch.delenv('YH_JOB_ID',raising=False)
    monkeypatch.setattr(sys,'argv',['calibrate','trial','--input','unused','--index','0','--output','unused'])
    monkeypatch.setattr(calibration,'trial',lambda *a:pytest.fail('No local CUDA trial may run'))
    with pytest.raises(RuntimeError,match='allocated server GPU'):calibration.main()


@pytest.mark.parametrize('task',['dng','csp','property'])
def test_input_preparation_remaps_all_workloads_and_uses_only_eligible_train(tmp_path,monkeypatch,task):
    from unifiedgroupgen.pl_data import dataset
    rows=[dict(id=str(i),dataset='mp_20',descriptor=dict(kind='space',number=1,
          orbits=[dict(letter='a',atomic_number=6)]*(i+1)),properties={'band_gap':i/10 if i%2 else None}) for i in range(12)]
    monkeypatch.setattr(dataset,'load_training_splits',lambda *args:(rows,[],{}))
    c=dict(name='test',entries={'model':{'experiment':'full_mp20_eqv2_'+task}},calibration=POLICY)
    calibration.prepare_inputs(c,'model',tmp_path)
    value=calibration.read(tmp_path/'inputs.json')
    assert value['source_split']=='train' and value['eligible_train_records']==(6 if task=='property' else 12)
    assert value['config']['allowed_elements']==[6]
    assert len(value['records'])==len(value['expected_atoms'])
    if task=='property':assert all(r['properties']['band_gap'] is not None for r in value['records'])
    for candidate in value['candidates']:
        for case in candidate['cases']:
            assert all(0<=i<len(value['records']) for i in case['indices'])
            assert len(case['indices'])<=candidate['batch_size']
            assert all(value['expected_atoms'][i]==len(value['records'][i]['descriptor']['orbits']) for i in case['indices'])
