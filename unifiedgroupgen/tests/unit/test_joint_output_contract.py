"""Output contracts using temporary files and mocks, never real training or inference."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf
from unifiedgroupgen.common.configuration import sampling_files
from unifiedgroupgen.common.structures import write_json
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl
from unifiedgroupgen.joint_flow import cli
from unifiedgroupgen.pl_modules import joint_training as training


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


@pytest.fixture
def generation(monkeypatch):
    args=SimpleNamespace(count=4,batch_size=2,seed=7,device='cpu',kind='space',group=None,
        steps=4,exact_group=False,case_id=None,checkpoint='fixture.pt',trajectory_dir=None)
    config=dict(model={},group_prior={},generation_task='dng',dataset='fixture')
    model=SimpleNamespace(backend='fixture',max_atoms=24,max_orbits=4)
    spec=SimpleNamespace(number=1)
    monkeypatch.setattr(cli,'sample_group',lambda *a,**k:spec)
    monkeypatch.setattr(cli,'sample_batch',lambda model,specs,**k:[object() for _ in specs])
    def record(sample,i,seed,*a):
        status=['generated','rejected','failed','generated'][i]
        return dict(index=i,seed=seed,status=status,accepted=status=='generated')
    monkeypatch.setattr(cli,'evaluation_record',record)
    return args,{'config':config},model,ChemicalControl()


def generate(fixture,path):
    args,checkpoint,model,chemistry=fixture
    return cli.generate(args,checkpoint,model,None,chemistry,{},path)


def test_generation_flushes_progress_and_saves_summary_without_random_draws(tmp_path,generation,monkeypatch):
    output=tmp_path/'samples.jsonl';summary,status=sampling_files(output);seen=[]
    def sample(model,specs,**kwargs):
        progress=read(status)
        seen.append((progress['attempted'],len(output.read_text().splitlines())))
        return [object() for _ in specs]
    monkeypatch.setattr(cli,'sample_batch',sample)
    rng=torch.get_rng_state().clone()
    result=generate(generation,output)
    assert torch.equal(rng,torch.get_rng_state())
    assert seen==[(0,0),(2,2)] and read(status)==read(summary)==result
    assert result['state']=='complete' and result['attempted']==result['requested']==4
    assert result['generated']==result['accepted']==2 and result['rejected']==result['failed']==1
    assert result['unattempted']==0 and result['seconds']>=0 and result['updated']>=result['started']
    rows=[json.loads(s) for s in output.read_text().splitlines()]
    assert [r['index'] for r in rows]==list(range(4)) and rows[2]['status']=='failed'
    assert not list(tmp_path.glob('*.tmp'))


@pytest.mark.parametrize('error',[RuntimeError('fixture failure'),KeyboardInterrupt(),SystemExit(2)])
def test_generation_exception_keeps_partial_ledger_and_explicit_denominator(tmp_path,generation,monkeypatch,error):
    calls=[]
    def sample(model,specs,**kwargs):
        calls.append(1)
        if len(calls)==2:raise error
        return [object() for _ in specs]
    monkeypatch.setattr(cli,'sample_batch',sample)
    output=tmp_path/'samples.jsonl'
    with pytest.raises(type(error)):generate(generation,output)
    summary,status=sampling_files(output);value=read(summary)
    assert value==read(status)
    assert value['state']==('failed' if isinstance(error,RuntimeError) else 'interrupted')
    assert value['attempted']==2 and value['unattempted']==2 and value['requested']==4
    assert len(output.read_text().splitlines())==2 and value['error_type']==type(error).__name__


def test_missing_sampler_results_cannot_be_reported_complete(tmp_path,generation,monkeypatch):
    monkeypatch.setattr(cli,'sample_batch',lambda *a,**k:[])
    output=tmp_path/'samples.jsonl'
    with pytest.raises(RuntimeError,match='every requested'):generate(generation,output)
    assert read(sampling_files(output)[0])['state']=='failed'


def test_no_feasible_group_is_a_recorded_failure_not_a_dropped_attempt(tmp_path,generation,monkeypatch):
    from unifiedgroupgen.group_constraints.group_prior import NoFeasibleGroup
    def group(prior,generator,**kwargs):
        if generator.initial_seed()==9:raise NoFeasibleGroup('fixture')
        return SimpleNamespace(number=1)
    monkeypatch.setattr(cli,'sample_group',group)
    result=generate(generation,tmp_path/'samples.jsonl')
    assert result['attempted']==4 and result['failed']==1 and result['state']=='complete'


@pytest.mark.parametrize('existing',['samples.jsonl','summary.json','sampling.status.json'])
def test_existing_generation_files_are_preserved(tmp_path,generation,existing,monkeypatch):
    path=tmp_path/existing;path.write_bytes(b'existing result')
    before={p.name:p.read_bytes() for p in tmp_path.iterdir()}
    monkeypatch.setattr(cli,'sample_batch',lambda *a,**k:pytest.fail('Must not sample'))
    with pytest.raises(FileExistsError):generate(generation,tmp_path/'samples.jsonl')
    assert before=={p.name:p.read_bytes() for p in tmp_path.iterdir()}


def test_explicit_ledger_names_keep_distinct_sidecars(tmp_path,generation):
    for name in ('dng.jsonl','csp.jsonl'):
        path=tmp_path/name;result=generate(generation,path)
        summary,status=sampling_files(path)
        assert summary.name==name+'.summary.json' and status.name==name+'.sampling.status.json'
        assert read(summary)==read(status)==result
    assert not (tmp_path/'summary.json').exists()


def test_invalid_row_is_not_counted_as_a_written_attempt(tmp_path,generation,monkeypatch):
    monkeypatch.setattr(cli,'evaluation_record',lambda *a:dict(status='generated',value=float('nan')))
    output=tmp_path/'samples.jsonl'
    with pytest.raises(ValueError):generate(generation,output)
    value=read(sampling_files(output)[0])
    assert value['state']=='failed' and value['attempted']==0 and value['unattempted']==4
    assert output.read_bytes()==b''


@pytest.fixture
def joint_entry(tmp_path,monkeypatch):
    from unifiedgroupgen.pl_modules import factory
    config=dict(model={'family':'joint_flow_v2','task_controls':True},generation_task='dng',
                group_prior={},properties={})
    checkpoint=dict(config=config,model={})
    model=SimpleNamespace(flow_version='joint_flow_v2',allowed_elements=(8,),load_state_dict=lambda *a,**k:None)
    monkeypatch.setattr(training,'load_joint_checkpoint',lambda *a,**k:checkpoint)
    monkeypatch.setattr(factory,'build_joint_model',lambda *a,**k:model)
    import unifiedgroupgen.common.configuration as configuration
    monkeypatch.setattr(configuration,'project_root',lambda:tmp_path)
    captured={}
    monkeypatch.setattr(cli,'generate',lambda *a:captured.update(output=a[-1]) or {'fixture':True})
    return model,captured,['sample','--checkpoint',str(tmp_path/'run/best.ckpt'),'--kind','space']


@pytest.mark.parametrize('name',[None,'comparison_seed7'])
def test_joint_entry_defaults_beside_checkpoint(tmp_path,joint_entry,name):
    from unifiedgroupgen.joint import main
    _,captured,args=joint_entry
    main(args+(['--task-name',name] if name else []))
    assert captured['output']==tmp_path/'run/generate'/(name or 'space_dng_seed42')/'samples.jsonl'


def test_explicit_output_path_remains_exact(tmp_path,joint_entry):
    from unifiedgroupgen.joint import main
    _,captured,args=joint_entry;output=tmp_path/'explicit/custom.jsonl'
    main(args+['--output',str(output)])
    assert captured['output']==output


@pytest.mark.parametrize('task,flags',[('csp',['--formula','O2']),('property',['--properties','{"band_gap":1.5}'])])
def test_other_tasks_use_the_same_directory_contract(tmp_path,joint_entry,monkeypatch,task,flags):
    from unifiedgroupgen.joint import main
    model,captured,args=joint_entry
    config=dict(model={'family':'joint_flow_v2','task_controls':True},generation_task=task,
        group_prior={},properties={'band_gap':{'type':'scalar'}} if task=='property' else {})
    monkeypatch.setattr(training,'load_joint_checkpoint',lambda *a,**k:dict(config=config,model={}))
    main(args+flags)
    assert captured['output']==tmp_path/f'run/generate/space_{task}_seed42/samples.jsonl'


def test_unsafe_task_name_cannot_escape_run(tmp_path,joint_entry):
    from unifiedgroupgen.joint import main
    with pytest.raises(ValueError):main(joint_entry[2]+['--task-name','../escape'])
    assert not (tmp_path/'escape').exists()


def test_legacy_joint_still_requires_explicit_output(joint_entry):
    from unifiedgroupgen.joint import main
    model,_,args=joint_entry;model.flow_version='joint_symmetry_v1'
    with pytest.raises(SystemExit):main(args)


def test_legacy_explicit_generation_does_not_gain_v2_sidecars(tmp_path,joint_entry,monkeypatch):
    import unifiedgroupgen.joint as joint
    import unifiedgroupgen.generation.joint as legacy
    model,_,args=joint_entry;model.flow_version='joint_symmetry_v1'
    monkeypatch.setattr(legacy,'sample_joint',lambda *a,**k:object())
    monkeypatch.setattr(joint,'evaluation_record',lambda *a,**k:dict(status='generated'))
    output=tmp_path/'legacy/samples.jsonl'
    joint.main(args+['--group','1','--output',str(output)])
    assert len(output.read_text().splitlines())==1
    assert not any(p.exists() for p in sampling_files(output))


@pytest.fixture
def metric_config(tmp_path):
    root=Path(__file__).resolve().parents[3]
    options=OmegaConf.load(root/'configs/evaluation/default.yaml')
    samples=tmp_path/'run/generate/fixture/samples.jsonl'
    samples.parent.mkdir(parents=True);samples.write_text('{"index":0,"status":"failed"}\n')
    options.samples=str(samples)
    return OmegaConf.create(dict(task='evaluate',device='cpu',threads=1,evaluation=options))


def test_evaluation_uses_matching_task_directory_and_preserves_inputs(metric_config,monkeypatch):
    from unifiedgroupgen import evaluate
    samples=Path(metric_config.evaluation.samples);before=samples.read_bytes();captured={}
    def metrics(args):
        captured.update(status=read(args.output/'metrics.status.json'),output=args.output)
        write_json(args.output/'summary.json',{'fixture':True})
        (args.output/'samples.csv').write_text('index,status\n0,failed\n')
        write_json(args.output/'manifest.json',{'state':'complete'})
    monkeypatch.setattr(evaluate,'evaluate_task',metrics)
    evaluate.execute(metric_config)
    output=samples.parents[2]/'evaluate/fixture';status=read(output/'metrics.status.json')
    assert captured['output']==output and captured['status']['state']=='running'
    assert status['state']=='complete' and status['phase']=='finished'
    assert status['samples']==str(samples) and status['updated']>=status['started']
    assert samples.read_bytes()==before
    preserved={p.name:p.read_bytes() for p in output.iterdir()}
    with pytest.raises(ValueError,match='already exist'):evaluate.execute(metric_config)
    assert preserved=={p.name:p.read_bytes() for p in output.iterdir()}


@pytest.mark.parametrize('error',[RuntimeError('fixture'),KeyboardInterrupt()])
def test_evaluation_failure_status_keeps_source_identity(metric_config,monkeypatch,error):
    from unifiedgroupgen import evaluate
    def fail(args):raise error
    monkeypatch.setattr(evaluate,'evaluate_task',fail)
    with pytest.raises(type(error)):evaluate.execute(metric_config)
    samples=Path(metric_config.evaluation.samples)
    status=read(samples.parents[2]/'evaluate/fixture/metrics.status.json')
    assert status['state']==('failed' if isinstance(error,RuntimeError) else 'interrupted')
    assert status['samples']==str(samples) and status['error_type']==type(error).__name__


def test_training_lifecycle_and_callbacks_without_real_model_or_fit(tmp_path,monkeypatch):
    output=tmp_path/'model.pt';status=output.with_suffix('.status.json');seen=[]
    def phase():seen.append(read(status)['phase'])
    class FakeModule:
        def __init__(self,config,*args):
            phase();self.config=config;self.model=SimpleNamespace(state_dict=lambda:{})
    def data(*args):phase()
    class FakeTrainer:
        def __init__(self,**options):
            phase();self.callbacks=options['callbacks'];self.current_epoch=0;self.global_step=1
            self.interrupted=False;self.callback_metrics={'val/loss':torch.tensor(2.)}
            self.optimizers=[SimpleNamespace(param_groups=[{'lr':.0003}])]
        def fit(self,module,**kwargs):
            phase();artifacts=next(c for c in self.callbacks if isinstance(c,training.JointTrainingArtifacts))
            artifacts.on_train_batch_end(self,module,{'loss':torch.tensor(3.)},[],0);phase()
            artifacts.on_validation_start(self,module);phase()
            artifacts.on_train_epoch_end(self,module);phase()
            for cb in self.callbacks:
                if isinstance(cb,training.ModelCheckpoint):cb.best_model_score=torch.tensor(2.)
            self.current_epoch=1
        def save_checkpoint(self,path):phase();path.write_bytes(b'fixture, not real checkpoint')
    monkeypatch.setattr(training,'JointGeneratorModule',FakeModule)
    monkeypatch.setattr(training,'JointDataModule',data)
    monkeypatch.setattr(training.pl,'Trainer',FakeTrainer)
    config=dict(model={'family':'joint_flow_v2','task_controls':True},generation_task='dng',properties={},group_prior={})
    training.fit_joint(config,[{'id':'train'}],[{'id':'val'}],output,epochs=1)
    assert seen==['building_model','preparing_targets','trainer_setup','waiting_for_first_batch',
                  'training','validation','epoch_end','saving']
    value=read(status)
    assert value['state']=='complete' and value['phase']=='finished' and value['global_step']==1
    assert read(output.with_suffix('.history.jsonl'))['metrics']['val/loss']==2.
    assert value['updated']>=value['started'] and value['validation_loss']==2.


@pytest.mark.parametrize('error',[RuntimeError('fixture initialization'),KeyboardInterrupt()])
def test_training_initialization_failure_is_visible_without_training(tmp_path,monkeypatch,error):
    def fail(*args):raise error
    monkeypatch.setattr(training,'JointGeneratorModule',fail)
    output=tmp_path/'model.pt'
    config=dict(model={'family':'joint_flow_v2','task_controls':True},generation_task='dng',properties={},group_prior={})
    with pytest.raises(type(error)):training.fit_joint(config,[{'id':'train'}],[{'id':'val'}],output)
    status=read(output.with_suffix('.status.json'))
    assert status['phase']=='building_model' and status['global_step'] is None
    assert status['state']==('failed' if isinstance(error,RuntimeError) else 'interrupted')
    assert not output.exists() and not list(tmp_path.glob('*.ckpt'))


@pytest.mark.parametrize('name',['best.ckpt','best-v1.ckpt','last.ckpt','model.history.jsonl','model.status.json'])
def test_training_refuses_an_existing_run_before_any_model_construction(tmp_path,monkeypatch,name):
    (tmp_path/name).write_bytes(b'existing result')
    monkeypatch.setattr(training,'JointGeneratorModule',lambda *a:pytest.fail('No model construction'))
    with pytest.raises(FileExistsError):
        training.fit_joint({'model':{'family':'joint_flow_v2'}},[{'id':'train'}],[{'id':'val'}],tmp_path/'model.pt')
    assert list(tmp_path.iterdir())==[tmp_path/name] and (tmp_path/name).read_bytes()==b'existing result'


@pytest.mark.parametrize('suffix',['.history.jsonl','.status.json'])
def test_programmatic_custom_training_name_preserves_sidecars(tmp_path,monkeypatch,suffix):
    output=tmp_path/'custom.pt';existing=output.with_suffix(suffix);existing.write_bytes(b'existing custom result')
    monkeypatch.setattr(training,'JointGeneratorModule',lambda *a:pytest.fail('No model construction'))
    with pytest.raises(FileExistsError):
        training.fit_joint({'model':{'family':'joint_flow_v2'}},[{'id':'train'}],[{'id':'val'}],output)
    assert existing.read_bytes()==b'existing custom result' and not output.exists()


def test_legacy_lifecycle_does_not_add_initialization_files(tmp_path):
    output=tmp_path/'model.pt'
    with training._flow_status(output,{'model':{'family':'joint_symmetry_v1'}}) as progress:
        progress('preparing_targets')
        assert not list(tmp_path.iterdir())
