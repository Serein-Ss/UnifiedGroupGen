import copy
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip('pytorch_lightning')
from unifiedgroupgen.pl_modules.factory import build_models, load_model_checkpoint

from unifiedgroupgen.pl_data.datamodule import CrystalDataModule
from unifiedgroupgen.run import UnifiedGroupGenModule
from unifiedgroupgen.run import run_lightning
from unifiedgroupgen.common.configuration import RunSettings
from unifiedgroupgen.run import training_output
from unifiedgroupgen.group_constraints.compiler import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.pl_modules.objectives.joint import joint_loss


def seed():
    random.seed(42); np.random.seed(42); torch.manual_seed(42)


def record(name, descriptor=None):
    c = compile_descriptor(descriptor or Descriptor('space', 2, (OrbitSpec('i', 14),)))
    return {'id': name, 'dataset': 'fixture', 'descriptor': c.descriptor.to_dict(),
            'state': c.prior().tolist(), 'properties': {}}


def config():
    return {'groups': [('space', 2)], 'properties': {}, 'model': {'hidden': 16, 'proposal_layers': 1,
            'flow_layers': 1, 'heads': 4, 'frequencies': 2}, 'max_orbits': 4,
            'batch_size': 2, 'epochs': 2, 'precision': 'fp32', 'learning_rate': .0003,
            'early_stopping_patience': 0, 'composition_probability': 0., 'observation_probability': 0.,
            'validation_interval': 1}


def settings(root, mode='disabled'):
    return RunSettings(root, root/'outputs', root/'wandb', mode)


def test_settings_and_real_start_directories(tmp_path, monkeypatch):
    monkeypatch.delenv('HYDRA_JOBS', raising=False)
    monkeypatch.delenv('WANDB_DIR', raising=False)
    monkeypatch.delenv('USE_WANDB_LOGGING', raising=False)
    (tmp_path/'.env').write_text('HYDRA_JOBS="outputs"\nWANDB_DIR="wandb"\nUSE_WANDB_LOGGING=0\n')
    s = RunSettings.load(tmp_path)
    assert s.wandb_mode == 'offline' and s.jobs == tmp_path/'outputs'
    output = training_output(s, 'mp20_u')
    assert output.name == 'model.pt'
    assert not (output.parent/'run.json').exists()
    assert output.parent.parent.name == 'mp20_u'
    assert len(output.parent.name.split('-')) == 6
    output.write_bytes(b'existing checkpoint')
    with pytest.raises(FileExistsError):
        training_output(s, 'mp20_u', output=output)
    with pytest.raises(ValueError):
        training_output(s, '../escape')


def test_joint_loss_and_all_gradients_are_unchanged(tmp_path):
    torch.set_num_threads(1)
    seed(); cfg = config(); cfg['groups'].append(('layer', 1))
    records = [record('a'), record('b', Descriptor('layer', 1, (OrbitSpec('a', 16),)))]
    p, f = build_models(cfg, 'cpu'); p2, f2 = copy.deepcopy(p), copy.deepcopy(f)
    runtime = {'coordinate_length_scale': 3.}
    module = UnifiedGroupGenModule(p2, f2, cfg, runtime, {}, tmp_path/'model.pt')
    seed(); losses = joint_loss(p, f, records, 0, 0, 3.)
    original = losses['descriptor'].mean()+losses['flow'].mean()
    original.backward()
    seed(); actual, total = module.objective(records)
    total.backward()
    torch.testing.assert_close(original, total, rtol=0, atol=0)
    for key in losses:
        torch.testing.assert_close(losses[key], actual[key], rtol=0, atol=0)
    for before, after in zip(list(p.parameters())+list(f.parameters()), module.parameters()):
        if before.grad is not None:
            torch.testing.assert_close(before.grad, after.grad, rtol=0, atol=0)


def test_batch_sampler_is_reproducible_without_consuming_training_rng():
    seed(); records = [record(str(i)) for i in range(5)]
    runtime = {'batch_size': 2, 'edge_budget': 8}
    data = CrystalDataModule(records, records[:1], runtime)
    state = random.getstate(); torch_state = torch.get_rng_state().clone()
    loader = data.train_dataloader(); len(loader); list(loader)
    assert random.getstate() == state
    assert torch.equal(torch_state, torch.get_rng_state())
    assert [list(x) for x in loader.batch_sampler] == [list(x) for x in data.train_dataloader().batch_sampler]
    with pytest.raises(ValueError, match='held-out'):
        data.test_dataloader()


def run_fixture(root, epochs, resume=None, interval=1, mode='disabled', device='cpu', precision='fp32', keep_logging=False):
    root.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1); seed()
    cfg = config(); cfg.update(validation_interval=interval, precision=precision)
    records = [record('a'), record('b')]; val = [record('v')]
    for name in ('train', 'val'):
        (root/(name+'.jsonl')).write_text(name)
    p, f = build_models(cfg, 'cpu')
    args = SimpleNamespace(device=device, resume=resume, epochs=epochs, seed=42, initialize_from=None)
    trainer = run_lightning(args, cfg, records, val, p, f, [root/'train.jsonl'], [root/'val.jsonl'],
                            root/'model.pt', settings(root, mode), finish_logging=not keep_logging)
    return load_model_checkpoint(root/'model.pt', weights_only=False), trainer


def test_native_resume_and_native_inference_loading(tmp_path):
    full, _ = run_fixture(tmp_path/'full', 2)
    partial, _ = run_fixture(tmp_path/'resumed', 1)
    resumed, _ = run_fixture(tmp_path/'resumed', 2, tmp_path/'resumed/model.pt')
    for network in ('proposal', 'flow'):
        for key in full[network]:
            torch.testing.assert_close(full[network][key], resumed[network][key], rtol=0, atol=0)
    assert resumed['epoch'] == 1
    assert resumed['trainer_backend'] == 'lightning'
    p, f = build_models(resumed['config'], 'cpu')
    p.load_state_dict(resumed['proposal']); f.load_state_dict(resumed['flow'])
    history = [json.loads(s) for s in (tmp_path/'resumed/model.history.jsonl').read_text().splitlines()]
    assert [row['epoch'] for row in history] == [0, 1]
    assert not (tmp_path/'resumed/logs').exists()
    assert not (tmp_path/'resumed/run.json').exists()
    assert not (tmp_path/'resumed/.hydra').exists()
    assert {p.name for p in (tmp_path/'resumed').iterdir() if p.name not in ('train.jsonl','val.jsonl')} == {
        'model.pt','model.best.pt','model.history.jsonl','model.status.json','train.log'}
    assert (tmp_path/'resumed/train.log').is_file()
    native_full = torch.load(tmp_path/'full/model.pt', weights_only=False)
    native_resumed = torch.load(tmp_path/'resumed/model.pt', weights_only=False)
    assert native_full['lr_schedulers'] == native_resumed['lr_schedulers']
    assert torch.equal(native_full['ugg_state']['rng']['torch'], native_resumed['ugg_state']['rng']['torch'])


def test_final_epoch_validation_and_offline_wandb(tmp_path):
    state, trainer = run_fixture(tmp_path/'offline', 2, interval=5, mode='offline', keep_logging=True)
    rows = [json.loads(s) for s in (tmp_path/'offline/model.history.jsonl').read_text().splitlines()]
    assert [r['validated'] for r in rows] == [False, True]
    assert (tmp_path/'offline/model.best.pt').exists()
    import wandb
    try:
        data = CrystalDataModule(trainer.datamodule.records, trainer.datamodule.validation,
                                 trainer.datamodule.runtime, test_records=[record('heldout')])
        result = trainer.test(trainer.lightning_module, datamodule=data, verbose=False)
        assert np.isfinite(result[0]['test/loss'])
        assert wandb.run is not None
        assert wandb.run.config['runtime']['coordinate_length_scale'] == 3.
        assert wandb.run.config['seed'] == 42
    finally:
        wandb.finish()
    assert list((tmp_path/'offline/wandb').rglob('run-*.wandb'))
    assert not (tmp_path/'offline/provenance').exists()
    assert not (tmp_path/'offline/run.json').exists()


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA unavailable')
def test_cuda_bf16_train_save_and_test(tmp_path):
    if not torch.cuda.is_bf16_supported():
        pytest.skip('BF16 unavailable')
    state, trainer = run_fixture(tmp_path/'cuda', 1, device='cuda', precision='bf16')
    assert state['epoch'] == 0
    data = CrystalDataModule(trainer.datamodule.records, trainer.datamodule.validation,
                             trainer.datamodule.runtime, test_records=[record('heldout')])
    status = (tmp_path/'cuda/model.status.json').read_bytes()
    trainer.validate(trainer.lightning_module, datamodule=data, verbose=False)
    result = trainer.test(trainer.lightning_module, datamodule=data, verbose=False)
    assert np.isfinite(result[0]['test/loss'])
    assert (tmp_path/'cuda/model.status.json').read_bytes() == status


def test_regular_and_final_validation_schedule(tmp_path):
    run_fixture(tmp_path/'seven', 7, interval=5)
    rows = [json.loads(s) for s in (tmp_path/'seven/model.history.jsonl').read_text().splitlines()]
    assert [row['validated'] for row in rows] == [False, False, False, False, True, False, True]


def test_interrupted_trainer_propagates_failure_to_serial_scheduler(tmp_path, monkeypatch):
    import pytorch_lightning as pl
    from pytorch_lightning.trainer.states import TrainerStatus
    def interrupted(self, *args, **kwargs):
        self.state.status = TrainerStatus.INTERRUPTED
    monkeypatch.setattr(pl.Trainer, 'fit', interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_fixture(tmp_path/'stopped', 1)
    state = json.loads((tmp_path/'stopped/model.status.json').read_text())
    assert state['state'] == 'interrupted'


def test_lightning_plan_preserves_approved_runtime_and_allocates_no_output(tmp_path):
    from unifiedgroupgen.tests.reference_workflows.train_lightning_experiments import plan_jobs
    root = Path(__file__).resolve().parents[3]
    args = SimpleNamespace(models=['mp20_u', 'c2db51_all', 'joint_all'], seed=42, threads=4,
                           epochs=None, device='cuda', wandb_mode='offline')
    jobs = plan_jobs(root, args)
    assert [job['name'] for job in jobs] == args.models
    assert [len(job['target_properties']) for job in jobs] == [0, 3, 3]
    for job in jobs:
        command = job['command']
        assert command[command.index('--trainer')+1] == 'lightning'
        assert '--output' not in command
        assert job['runtime_options']['coordinate_length_scale'] == 3.
        assert job['runtime_options']['batch_size'] == (768 if job['name'].startswith('c2db') else 256)
        assert job['runtime_options']['edge_budget'] == (200000 if job['name'].startswith('c2db') else 400000)
    assert not list(tmp_path.iterdir())


def test_native_resume_replays_pending_plateau_before_next_optimizer_step(tmp_path, monkeypatch):
    original_config = config
    def forced_schedule():
        value = original_config();value['lr_patience'] = 0
        return value
    monkeypatch.setattr(__import__(__name__, fromlist=['config']), 'config', forced_schedule)
    objective = UnifiedGroupGenModule.objective
    def increasing_validation(self, records, validation=False):
        if not validation:return objective(self, records, validation)
        value = torch.full((len(records),), 1.+self.current_epoch, device=self.device)
        return {key:value for key in self.keys}, value.mean()*2
    monkeypatch.setattr(UnifiedGroupGenModule, 'objective', increasing_validation)
    full,_ = run_fixture(tmp_path/'full',3)
    run_fixture(tmp_path/'resumed',2)
    resumed,_ = run_fixture(tmp_path/'resumed',3,tmp_path/'resumed/model.pt')
    for network in ('proposal','flow'):
        for key in full[network]:torch.testing.assert_close(full[network][key],resumed[network][key],rtol=0,atol=0)
    a=torch.load(tmp_path/'full/model.pt',weights_only=False)
    b=torch.load(tmp_path/'resumed/model.pt',weights_only=False)
    assert a['lr_schedulers']==b['lr_schedulers']
    assert a['optimizer_states'][0]['param_groups'][0]['lr'] < .0003


def test_native_checkpoint_runs_the_public_generation_pipeline(tmp_path):
    from unifiedgroupgen.generate import sample
    run_fixture(tmp_path/'trained', 1)
    descriptor = tmp_path/'descriptor.json'
    descriptor.write_text(json.dumps(Descriptor('space', 2, (OrbitSpec('i', 14),)).to_dict()))
    args = SimpleNamespace(checkpoint=str(tmp_path/'trained/model.best.pt'), device='cpu', kind='space',
        properties='{}', composition=None, formula=None, formula_units=[1], chemical_system=None,
        allowed_elements=None, required_elements=None, group=None, scaffold=None, atom_map=None,
        scaffold_tolerance=1e-4, descriptor=str(descriptor), observations=None, prior=False,
        guidance=0., temperature=1., steps=2, count=1, seed=142, output=str(tmp_path/'generated'))
    sample(args, root=tmp_path)
    records=[json.loads(line) for line in (tmp_path/'generated/samples.jsonl').read_text().splitlines()]
    assert len(records)==1 and records[0]['status']=='generated'
    assert records[0]['cif']
    assert not list((tmp_path/'generated').glob('*.cif'))
