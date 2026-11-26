"""Configuration and mocked scheduler checks: never build or train a model."""
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
import pytest
import yaml
from filelock import FileLock, Timeout
from submit_files import joint_flow5090 as jobs
from submit_files import joint_flow5090_calibrate as calibration
from unifiedgroupgen.experiment_config import compose_experiment, training_definition

ROOT=Path(__file__).resolve().parents[3]
DATA=yaml.safe_load((ROOT/jobs.CATALOG).read_text(encoding='utf-8'))


@pytest.mark.parametrize('name,entry',DATA['experiments'].items())
def test_full_configuration_has_correct_task_data_backbone_and_no_limits(name,entry):
    cfg=compose_experiment(entry['experiment']);c=training_definition(cfg)
    assert cfg.experiment_name==name and c['experiment_id']==name
    assert c['model']['family']=='joint_flow_v2' and c['model']['backbone']==entry['backbone']
    assert c['dataset']==entry['dataset'] and c['generation_task']==entry['task']
    assert c['epochs']==1000 and c['joint_max_steps']==-1
    assert c['lr_schedule']==dict(factor=.5,patience=10,threshold=.0001,min_lr=.000001)
    is_comparison = name in DATA['backbone_comparison']['models']
    is_geometry_v2 = name.endswith('_geometry_v2')
    assert c['early_stopping']==dict(patience=30 if is_comparison or is_geometry_v2 else 60,min_delta=.0001)
    assert c['validation_seed']==918
    assert c['joint_train_limit'] is c['joint_val_limit'] is None
    assert cfg.runtime.joint_train_limit is cfg.runtime.joint_val_limit is cfg.runtime.joint_max_steps is None
    assert c['chemical_set_probability']==0.
    assert not c.get('dataset_domains') and c['cond_dropout']==(.1 if 'property_conditioning' in entry else 0.)
    assert set(c['properties'])==({entry['property_conditioning']} if 'property_conditioning' in entry else set())
    assert not any(p.get('context') for p in c['properties'].values())
    assert cfg.runtime.precision=='fp32' and cfg.runtime.strict_fp32 and cfg.runtime.fused_adamw
    sizes = DATA['backbone_comparison']['batch_sizes']
    default_batch = sizes[entry['backbone']] if is_comparison else 8
    assert cfg.runtime.batch_size == entry.get('batch_size', default_batch) and 'batch_size' not in c
    assert c['model']['hidden']==128 and c['model']['layers']==4 and not c['model']['checkpoint_layers']
    assert c['model']['heads']==(8 if entry['backbone']=='equiformer_v2' else 4)
    expected = ['mp_20', 'c2db_51'] if entry['dataset'] == 'mp20_c2db51' else [entry['dataset']]
    assert c['source_configs'] == {name: f'configs/sources/{name}.yaml' for name in expected}
    assert c['data_audit'] is None
    assert cfg.resume is None and cfg.initialize_from is None
    if is_geometry_v2:
        assert cfg.logging.mode == 'offline' and c['terminal_contact_repair'] is True
        assert c['geometry_objective'] == 'decomposed' and 'short_contact_weight' not in c
        assert c.get('csp_group_policy', 'prior') == ('composition_neighbors' if entry['task'] == 'csp' else 'prior')


def test_matrix_and_default_plan_are_exactly_requested():
    assert len(DATA['experiments'])==20
    combinations={(e['dataset'],e['backbone'],e['task'],e.get('property_conditioning')) for e in DATA['experiments'].values()}
    variants=(('dng',None),('csp',None),('dng','band_gap'))
    expected={(d,'equiformer_v2',t,p) for d in ('mp_20','c2db_51','mp20_c2db51') for t,p in variants}
    expected|={('mp_20','joint_atom_graph',t,p) for t,p in variants}
    expected|={('mp_20','cspnet',t,None) for t in ('dng','csp')}
    assert combinations==expected
    plan=jobs.plan()
    assert plan['selected']==['full_mp20_eqv2_dng','full_mp20_eqv2_csp'] and plan['jobs']==2
    assert plan['phase']=='audit_calibrate_train' and not plan['local_training_started']
    assert plan['calibration']['candidates'] == [1,2,4,8,16,32,64,128]
    assert plan['resources']==dict(partition='5090',gpus=1,cpus=8,memory='64G',hours=72)
    assert jobs.plan(['full_joint_eqv2_property'])['jobs']==1
    geometry = [f'full_{scope}_eqv2_{task}_geometry_v2'
                for scope in ('mp20', 'joint') for task in ('dng', 'csp')]
    geometry_plan = jobs.plan(geometry)
    assert geometry_plan['selected'] == geometry
    assert [calibration.model_policy(geometry_plan, name)['candidates'] for name in geometry] == [
        [8], [8], [4], [4]]


@pytest.mark.parametrize('models',[[],['missing'],['full_mp20_eqv2_dng']*2])
def test_bad_selection_is_rejected(models):
    with pytest.raises(ValueError):jobs.selection(DATA,models)


def campaign(root,source):
    models=DATA['default_models']
    return dict(schema=jobs.CAMPAIGN_SCHEMA,root=str(root),source=str(source),name='test_full',
                revision='frozen-commit',models=models,entries={m:DATA['experiments'][m] for m in models},
                resources=DATA['resources'],calibration=DATA['calibration'])


def make_profile(c,model):
    folder=Path(c['root'])/'intermediate/hpc_runs'/c['name']
    audit=dict(passed=True,config_sha256='fixture',inputs={'train':[]})
    jobs.write(folder/'audits'/(model+'.json'),audit)
    chosen=dict(state='pass',batch_size=16,records_per_second=10.,precision='fp32',
                peak_reserved_bytes=100,memory_limit_bytes=200)
    jobs.write(folder/'profiles'/(model+'.json'),dict(state='complete',identity=calibration.identity(c,model,audit),selected=chosen))


def test_commands_share_audit_definition_and_resume_native_last_checkpoint(tmp_path):
    c=campaign(tmp_path,tmp_path/'source');model=c['models'][0]
    audit,train=jobs.commands(c,model)
    assert audit[-1] in train and audit[-2]=='--override'
    assert 'experiment='+model in train and 'device=cuda:0' in train
    assert not any(v.startswith('resume=') or 'joint_max_steps=' in v or '_limit=' in v for v in train)
    path=jobs.output(c,model);path.parent.mkdir(parents=True)
    (path.parent/'last.ckpt').write_bytes(b'native-checkpoint-fixture')
    with pytest.raises(ValueError,match='original batch calibration'):jobs.commands(c,model)
    make_profile(c,model)
    _,train=jobs.commands(c,model)
    assert 'resume='+str(path.parent/'last.ckpt') in train
    assert 'output='+str(path) in train
    assert 'runtime.batch_size=16' in train


def test_incomplete_unresumable_results_are_preserved(tmp_path):
    c=campaign(tmp_path,tmp_path/'source');model=c['models'][0]
    p=jobs.output(c,model).with_suffix('.status.json');p.parent.mkdir(parents=True)
    p.write_text('{"state":"running"}')
    with pytest.raises(ValueError,match='last.ckpt'):jobs.commands(c,model)
    assert p.read_text()=='{"state":"running"}'


def test_completed_run_requires_all_actual_checkpoints(tmp_path):
    c=campaign(tmp_path,tmp_path/'source');model=c['models'][0];p=jobs.output(c,model)
    p.parent.mkdir(parents=True);p.with_suffix('.status.json').write_text('{"state":"complete"}')
    with pytest.raises(ValueError,match='checkpoints'):jobs.complete(c,model)
    for n in ('model.pt','last.ckpt','best.ckpt'):(p.parent/n).write_bytes(b'fixture')
    assert jobs.complete(c,model)


def test_submit_only_two_jobs_once_and_freezes_source_without_real_scheduler(tmp_path,monkeypatch):
    monkeypatch.setattr(jobs,'ROOT',tmp_path);monkeypatch.setattr(jobs,'catalog',lambda:DATA)
    calls=[]
    def checked(command,**kwargs):
        if command[0]=='git':return '' if 'ls-files' in command else 'frozen-commit\n'
        assert command[0]=='yhbatch';calls.append(command);return str(100+len(calls))+';cluster\n'
    monkeypatch.setattr(jobs.subprocess,'check_output',checked)
    monkeypatch.setattr(jobs.subprocess,'run',lambda *a,**k:SimpleNamespace(returncode=0))
    def snapshot(root,folder,revision):
        assert root==tmp_path and revision=='frozen-commit'
        s=folder/'source';s.mkdir();return s
    monkeypatch.setattr(jobs,'snapshot',snapshot)
    jobs.submit('test_full');jobs.submit('test_full')
    assert len(calls)==2
    assert {v[-1] for v in calls}==set(DATA['default_models'])
    for command in calls:
        assert '--gres=gpu:1' in command and '--partition=5090' in command
        assert '--cpus-per-task=8' in command and '--mem=64G' in command
        assert any(v.endswith('joint_flow5090.sbatch') for v in command)
    folder=tmp_path/'intermediate/hpc_runs/test_full';frozen=jobs.read(folder/'campaign.json')
    assert frozen['models']==DATA['default_models'] and frozen['revision']=='frozen-commit'
    with pytest.raises(ValueError,match='Selection is frozen'):jobs.submit('test_full',['full_joint_eqv2_csp'])
    jobs.submit('test_full',retry=True);assert len(calls)==4
    assert len((folder/'submission_history.jsonl').read_text().splitlines())==4
    finished=jobs.output(frozen,DATA['default_models'][0]);finished.parent.mkdir(parents=True)
    for n in ('model.pt','best.ckpt','last.ckpt'):(finished.parent/n).write_bytes(b'fixture')
    jobs.write(finished.with_suffix('.status.json'),dict(state='complete'))
    jobs.submit('test_full',retry=True)
    assert len(calls)==5 and calls[-1][-1]=='full_mp20_eqv2_csp'


def test_untracked_runtime_source_is_not_silently_omitted(tmp_path,monkeypatch):
    monkeypatch.setattr(jobs,'ROOT',tmp_path);monkeypatch.setattr(jobs,'catalog',lambda:DATA)
    monkeypatch.setattr(jobs.subprocess,'run',lambda *a,**k:None)
    monkeypatch.setattr(jobs.subprocess,'check_output',lambda *a,**k:'unifiedgroupgen/new_module.py\n')
    monkeypatch.setattr(jobs,'snapshot',lambda *a,**k:pytest.fail('Uncommitted source must not freeze'))
    with pytest.raises(ValueError,match='Commit runtime source'):jobs.submit('untracked')
    assert not (tmp_path/'intermediate/hpc_runs/untracked').exists()


@pytest.mark.parametrize('name',['../bad','a/b','.','', 'a\nb'])
def test_campaign_path_cannot_escape(tmp_path,monkeypatch,name):
    monkeypatch.setattr(jobs,'ROOT',tmp_path)
    with pytest.raises(ValueError):jobs.submit(name)
    assert not (tmp_path/'intermediate').exists()


def worker_fixture(tmp_path,monkeypatch):
    source=tmp_path/'source';source.mkdir();monkeypatch.setattr(jobs,'ROOT',source)
    c=campaign(tmp_path,source);folder=tmp_path/'intermediate/hpc_runs/test_full';folder.mkdir(parents=True)
    p=folder/'campaign.json';jobs.write(p,c)
    monkeypatch.setattr(jobs,'device_info',lambda:dict(gpu='mock5090'))
    return c,p


def test_failed_audit_cannot_start_training(tmp_path,monkeypatch):
    c,p=worker_fixture(tmp_path,monkeypatch);calls=[]
    def fail(command,**kwargs):
        calls.append(command);raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(jobs.subprocess,'run',fail)
    with pytest.raises(subprocess.CalledProcessError):jobs.run_worker(p,c['models'][0])
    assert len(calls)==1 and 'audit' in calls[0] and 'unifiedgroupgen.run' not in calls[0]


def test_worker_audits_before_training_with_mock_commands_only(tmp_path,monkeypatch):
    c,p=worker_fixture(tmp_path,monkeypatch);model=c['models'][0];calls=[]
    def execute(command,**kwargs):
        calls.append(command)
        if 'audit' in command:jobs.write(p.parent/'audits'/(model+'.json'),dict(passed=True))
        elif 'calibrate' in command:make_profile(c,model)
        else:
            out=jobs.output(c,model);out.parent.mkdir(parents=True)
            for n in ('model.pt','best.ckpt','last.ckpt'):(out.parent/n).write_bytes(b'fixture')
            jobs.write(out.with_suffix('.status.json'),dict(state='complete'))
    monkeypatch.setattr(jobs.subprocess,'run',execute)
    jobs.run_worker(p,model);assert len(calls)==3 and 'audit' in calls[0] and 'calibrate' in calls[1] and 'unifiedgroupgen.run' in calls[2]
    assert 'runtime.batch_size=16' in calls[2]
    jobs.run_worker(p,model);assert len(calls)==3


def test_failed_calibration_cannot_start_training(tmp_path,monkeypatch):
    c,p=worker_fixture(tmp_path,monkeypatch);model=c['models'][0];calls=[]
    def execute(command,**kwargs):
        calls.append(command)
        if 'audit' in command:jobs.write(p.parent/'audits'/(model+'.json'),dict(passed=True))
        else:raise subprocess.CalledProcessError(1,command)
    monkeypatch.setattr(jobs.subprocess,'run',execute)
    with pytest.raises(subprocess.CalledProcessError):jobs.run_worker(p,model)
    assert len(calls)==2 and all('unifiedgroupgen.run' not in c for c in calls)


def test_concurrent_worker_is_rejected_before_any_command(tmp_path,monkeypatch):
    c,p=worker_fixture(tmp_path,monkeypatch);model=c['models'][0];locks=p.parent/'locks';locks.mkdir()
    monkeypatch.setattr(jobs.subprocess,'run',lambda *a,**k:pytest.fail('No subprocess may run'))
    with FileLock(str(locks/(model+'.lock')),timeout=0):
        with pytest.raises(Timeout):jobs.run_worker(p,model)


def test_live_checkout_cannot_replace_frozen_worker_source(tmp_path,monkeypatch):
    c,p=worker_fixture(tmp_path,monkeypatch)
    monkeypatch.setattr(jobs,'ROOT',tmp_path)
    monkeypatch.setattr(jobs.subprocess,'run',lambda *a,**k:pytest.fail('No subprocess may run'))
    with pytest.raises(ValueError,match='frozen source'):jobs.run_worker(p,c['models'][0])


def test_sbatch_uses_current_joint_worker_and_no_install_or_legacy_campaign():
    text=(ROOT/'submit_files/joint_flow5090.sbatch').read_text(encoding='utf-8')
    assert '#SBATCH -p 5090' in text and '#SBATCH --gres=gpu:1' in text
    assert 'joint_flow5090.py run "$@"' in text and 'export UGG_PROJECT_ROOT' in text and 'Missing project root from submission' in text
    assert 'CUDA/12.9' in text and 'conda activate unifiedgroupgen' in text
    assert not any(x in text for x in ('pip install','conda install','hpc5090.py run','git pull'))


def test_paper_matrix_freezes_same_budget_across_three_arms(tmp_path, monkeypatch):
    paper = dict(seeds=None, max_steps=10000, batch_size=4)
    planned = jobs.plan(paper=paper)
    assert planned['jobs'] == len(set(planned['selected'])) == 36
    assert planned['calibration']['candidates'] == [4]
    assert list(planned['entries'])[0] == 'full_mp20_eqv2_dng_control_s42'
    assert not planned['local_training_started']

    model = 'full_c2db51_eqv2_csp'
    paper['seeds'] = [43]
    submit_calls = []
    monkeypatch.setattr(jobs, 'catalog', lambda: DATA)
    monkeypatch.setattr(jobs, 'ROOT', tmp_path)
    monkeypatch.setattr(jobs, 'snapshot', lambda root, folder, revision: folder/'source')
    monkeypatch.setattr(jobs.subprocess, 'run', lambda *a,**k: None)

    def scheduler(command, **kwargs):
        if command[0] == 'git':
            return '' if 'ls-files' in command else 'frozen-commit'
        assert command[0] == 'yhbatch'
        submit_calls.append(command)
        return str(len(submit_calls))

    monkeypatch.setattr(jobs.subprocess, 'check_output', scheduler)
    jobs.submit('paper', [model], paper=paper)
    c = jobs.read(tmp_path/'intermediate/hpc_runs/paper/campaign.json')
    assert c['models'] == [f'{model}_{mode}_s43' for mode in DATA['paper']['ablations']]
    assert [call[-1] for call in submit_calls] == c['models']
    for key in c['models']:
        audit, train = jobs.commands(c, key)
        overrides = jobs.run_overrides(c, key)
        assert audit[-6:]==[v for item in overrides for v in ('--override',item)]
        assert set(overrides)<=set(train)
        assert 'seed=43' in train and 'train.joint_max_steps=10000' in train
        definition = training_definition(compose_experiment(model, overrides=overrides))
        assert definition['joint_max_steps']==10000 and definition['early_stopping'] is None
        assert definition['dataset']=='c2db_51' and definition['generation_task']=='csp'
    with pytest.raises(ValueError,match='settings are frozen'):
        jobs.submit('paper', [model], paper={**paper,'batch_size': 8})
    assert len(submit_calls)==3

    key = c['models'][0]
    path = jobs.output(c, key)
    path.parent.mkdir(parents=True)
    jobs.write(path.with_suffix('.status.json'),dict(state='complete',global_step=9999))
    with pytest.raises(ValueError,match='optimization budget'):
        jobs.complete(c,key)
