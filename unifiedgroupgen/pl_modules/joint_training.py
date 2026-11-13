"""Opt-in native Lightning training and versioned checkpoints for the joint generator."""
from pathlib import Path
import hashlib
import json
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from lightning_fabric.utilities.seed import _collect_rng_states, _set_rng_states
from unifiedgroupgen.common.configuration import RunSettings, project_root
from unifiedgroupgen.common.source import file_hash
from unifiedgroupgen.pl_data.joint_datamodule import JointDataModule
from unifiedgroupgen.pl_data.joint_path import target_from_record, sample_training_path
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_modules.objectives.joint_generator import joint_generator_loss


FORMAT = 'ugg_joint_symmetry_v1'
FLOW_FORMAT = 'ugg_joint_flow_v2'

def checkpoint_format(config):
    return FLOW_FORMAT if config.get('model', {}).get('family') == 'joint_flow_v2' else FORMAT


def _normalize_epoch_boundary(checkpoint, batches):
    """Serialize completed and partial epochs with their actual batch progress.

    PL 2.5's last-batch restart stage retains per-epoch completed batches even
    after fit returns. Without this normalization a one-batch epoch is replayed
    without an optimizer step. Only this new checkpoint format is affected.
    """
    loop = checkpoint['loops']['fit_loop']
    batch = loop['epoch_loop.batch_progress']
    if not batch['is_last_batch'] and 0 < batch['current']['completed'] < batches:
        for values in loop['epoch_progress'].values():
            values['processed'] = values['completed'] = values['started']-1
        checkpoint['epoch'] = loop['epoch_progress']['current']['completed']
        return False
    if not batch['is_last_batch'] or batch['current']['completed'] != batches:
        return False
    completed = loop['epoch_progress']['total']['started']
    for values in loop['epoch_progress'].values():
        for name in values:
            values[name] = completed
    checkpoint['epoch'] = completed

    def clear_current(value):
        for name, child in value.items():
            if name == 'current':
                for counter in child:
                    child[counter] = 0
            elif isinstance(child, dict):
                clear_current(child)

    for name in ('epoch_loop.batch_progress', 'epoch_loop.scheduler_progress',
                 'epoch_loop.automatic_optimization.optim_progress'):
        clear_current(loop[name])
    batch['is_last_batch'] = False
    validation = loop['epoch_loop.val_loop.batch_progress']
    for scope in ('total', 'current'):
        for name in validation[scope]:
            validation[scope][name] = 0
    validation['is_last_batch'] = False
    return True


def load_joint_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get('format') in (FORMAT, FLOW_FORMAT):
        return saved
    state = saved.get('joint_state')
    if not state or state.get('format') not in (FORMAT, FLOW_FORMAT):
        raise ValueError('Expected a joint-symmetry checkpoint, not a legacy two-stage checkpoint')
    return dict(state, model={k.removeprefix('model.'): v for k, v in saved['state_dict'].items()
                              if k.startswith('model.')}, epoch=saved['epoch'])


class JointGeneratorModule(pl.LightningModule):
    def __init__(self, config, runtime=None, split_hashes=None):
        super().__init__()
        self.config, self.runtime = dict(config), dict(runtime or {})
        self.split_hashes = dict(split_hashes or {})
        self.model = build_joint_model(config)
        self._targets = {}
        self._pending_rng = None
        self.save_hyperparameters(dict(config=self.config, runtime=self.runtime, split_hashes=self.split_hashes))

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch

    def objective(self, records, validation=False):
        if self.config.get('generation_task') == 'property':
            from unifiedgroupgen.pl_data.task_path import has_property_target
            records = [row for row in records if has_property_target(row, self.config['properties'])]
            if not records:
                raise ValueError('No observed target properties in this independent-task batch')
        mixed = self.config.get("model", {}).get("family") == "joint_flow_v2"
        if mixed:
            from unifiedgroupgen.joint_flow.path import sample_path as path_sampler
            from unifiedgroupgen.joint_flow.objective import loss as objective
        else:
            path_sampler, objective = sample_training_path, joint_generator_loss
        samples = []
        for row in records:
            identity = (row.get('dataset', ''), row['id'])
            if identity not in self._targets:
                self._targets[identity] = row.get('_joint_target') or target_from_record(
                    row, operation_compiler=self.config.get('model', {}).get('operation_compiler', False))
            digest = int.from_bytes(hashlib.sha256(repr(identity).encode()).digest()[:4], 'little')
            seed = (digest+918 if validation else digest+int(self.global_step)*100003+int(self.current_epoch)*977)
            from unifiedgroupgen.pl_data.task_path import task_target
            target = task_target(self._targets[identity], self.config, seed)
            sample = path_sampler(target, seed,
                     allowed_elements=self.model.allowed_elements, max_atoms=self.model.max_atoms,
                     max_orbits=self.model.max_orbits, budget=self.config.get('event_budget', 256),
                     augment=not validation, birth_scale=self.config.get('birth_scale', .2))
            samples.append(sample.to(self.device, next(self.model.parameters()).dtype))
        return objective(self.model, samples, self.config.get('jump_weight', 1.))

    def training_step(self, batch, batch_idx):
        losses = self.objective(batch)
        self.log_dict({'train/'+name: value for name, value in losses.items()},
                      on_step=True, on_epoch=True, batch_size=len(batch))
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        losses = self.objective(batch, validation=True)
        self.log_dict({'val/'+name: value for name, value in losses.items()},
                      on_step=False, on_epoch=True, batch_size=len(batch))

    def configure_optimizers(self):
        options={'fused':True} if self.runtime.get('fused_adamw',False) else {}
        return torch.optim.AdamW(self.parameters(), lr=self.config.get('learning_rate', 1e-3),**options)

    def configure_gradient_clipping(self,optimizer,gradient_clip_val=None,gradient_clip_algorithm=None):
        # Lightning 2.5 rejects fused clipping even without a gradient scaler.
        if self.runtime.get('fused_adamw',False) and self.trainer.precision in ('32-true','bf16-mixed'):
            if gradient_clip_val:
                parameters=[p for group in optimizer.param_groups for p in group['params']]
                algorithm=getattr(gradient_clip_algorithm,'value',gradient_clip_algorithm)
                if algorithm=='value':torch.nn.utils.clip_grad_value_(parameters,gradient_clip_val)
                else:torch.nn.utils.clip_grad_norm_(parameters,gradient_clip_val)
            return
        super().configure_gradient_clipping(optimizer,gradient_clip_val,gradient_clip_algorithm)

    def on_save_checkpoint(self, checkpoint):
        checkpoint['joint_state'] = dict(format=checkpoint_format(self.config), config=self.config, runtime=self.runtime,
                                        split_hashes=self.split_hashes, rng=_collect_rng_states(),
                                        epoch_boundary=_normalize_epoch_boundary(checkpoint, self.trainer.num_training_batches))

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get('joint_state', {})
        if state.get('format') not in (FORMAT, FLOW_FORMAT) or state.get('config') != self.config:
            raise ValueError('Joint resume architecture/training configuration differs')
        if state.get('runtime') != self.runtime or state.get('split_hashes') != self.split_hashes:
            raise ValueError('Joint resume execution settings or dataset hashes differ')
        self._pending_rng = state['rng']

    def on_train_start(self):
        if self._pending_rng is not None:
            _set_rng_states(self._pending_rng)
            self._pending_rng = None


class JointTrainingArtifacts(pl.Callback):
    def __init__(self, output):
        self.output = Path(output)

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if batch_idx == 0 or trainer.global_step % 25 == 0:
            value = outputs['loss'] if isinstance(outputs,dict) else outputs
            status = dict(format=checkpoint_format(module.config),state='running',epoch=trainer.current_epoch,
                          global_step=trainer.global_step,loss=float(value.detach()))
            self.output.with_suffix('.status.json').write_text(json.dumps(status,indent=2),encoding='utf-8')

    def on_train_epoch_end(self, trainer, module):
        metrics = {name: float(value.detach()) for name,value in trainer.callback_metrics.items()
                   if torch.is_tensor(value) and value.numel() == 1}
        row = dict(epoch=trainer.current_epoch,global_step=trainer.global_step,metrics=metrics)
        with self.output.with_suffix('.history.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')


def fit_joint(config, records, validation, output, device='cpu', epochs=1, runtime=None,
              split_hashes=None, resume=None, initialize_from=None, max_steps=-1, logger=False):
    """Reusable implementation for the normal Hydra entry and bounded local smoke runs."""
    identities = lambda rows: {(row.get('dataset', ''), row['id']) for row in rows}
    if not records or not validation or identities(records) & identities(validation):
        raise ValueError('Joint training requires nonempty, identifier-disjoint train and validation records')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() and not resume:
        raise FileExistsError('Use a new joint training output; existing checkpoints are preserved')
    from unifiedgroupgen.pl_data.task_path import validate_task_config, has_property_target
    validate_task_config(config)
    prior_records = records
    coverage = dict(train_requested=len(records), val_requested=len(validation))
    if config.get('generation_task') == 'property':
        records = [r for r in records if has_property_target(r, config['properties'])]
        validation = [r for r in validation if has_property_target(r, config['properties'])]
        if not records or not validation:
            raise ValueError('Property task needs observed target labels in train and validation')
    coverage.update(train_labelled=len(records), val_labelled=len(validation))
    if config.get('generation_task') and 'group_prior' not in config:
        from unifiedgroupgen.group_constraints.group_prior import fit_group_prior
        config = {**config, 'group_prior': fit_group_prior(prior_records)}
    config = {**config, 'magnetic_training': any(r.get('magnetic_moments') is not None for r in records)}
    actual_device = torch.device(device)
    device_name = 'cuda:'+str(actual_device.index or 0) if actual_device.type == 'cuda' else str(actual_device)
    runtime = {'batch_size': 4, 'edge_budget': 200000, **(runtime or {}),
               'device': device_name, 'train_records': len(records), 'val_records': len(validation)}
    precision=runtime.get('precision',config.get('precision','fp32'))
    modes={'fp32':'32-true','bf16':'bf16-mixed','fp16':'16-mixed'}
    if precision not in modes:raise ValueError('Joint precision must be fp32, bf16 or fp16')
    if config.get('model',{}).get('family') != 'joint_flow_v2' and precision != 'fp32':
        raise ValueError('Mixed precision is validated only for joint_flow_v2')
    if runtime.get('fused_adamw',False) and actual_device.type != 'cuda':
        raise ValueError('Fused AdamW profile requires CUDA')
    if runtime.get('fused_adamw',False) and precision=='fp16' and config.get('gradient_clip',10.):
        raise ValueError('FP16 with clipping requires unfused AdamW; BF16 fused clipping is supported')
    module = JointGeneratorModule(config, runtime, split_hashes)
    if initialize_from:
        module.model.load_state_dict(load_joint_checkpoint(initialize_from)['model'], strict=True)
    data = JointDataModule(records, validation, runtime, config.get('model', {}).get('operation_compiler', False))
    callback = ModelCheckpoint(dirpath=output.parent, filename='best', monitor='val/loss',
                               mode='min', save_top_k=1, save_last=True, save_on_train_epoch_end=True)
    trainer = pl.Trainer(accelerator='gpu' if str(device).startswith('cuda') else 'cpu',
                         devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else 1,
                         max_epochs=epochs, max_steps=max_steps, precision=modes[precision], logger=logger,
                         callbacks=[JointTrainingArtifacts(output),callback], default_root_dir=str(output.parent),
                         enable_model_summary=False, enable_progress_bar=False, num_sanity_val_steps=0,
                         gradient_clip_val=config.get('gradient_clip', 10.),
                         check_val_every_n_epoch=1, log_every_n_steps=runtime.get('log_every_n_steps',1), inference_mode=False,
                         reload_dataloaders_every_n_epochs=1)
    trainer.fit(module, datamodule=data, ckpt_path=str(resume) if resume else None)
    if trainer.interrupted:
        raise KeyboardInterrupt('Joint training interrupted; native last.ckpt is resumable')
    # Capture the completed loop boundary, rather than the callback's in-hook
    # snapshot. This is a public Lightning API and preserves optimizer/RNG state.
    trainer.save_checkpoint(output.parent/'last.ckpt')
    torch.save(dict(format=checkpoint_format(config), config=config, runtime=runtime, split_hashes=split_hashes or {},
                    model=module.model.state_dict(), epoch=trainer.current_epoch,
                    global_step=trainer.global_step), output)
    status = dict(format=checkpoint_format(config), state='complete', global_step=trainer.global_step,
                  train_records=len(records), val_records=len(validation),
                  magnetic_label_records=sum(r.get('magnetic_moments') is not None for r in records),
                  validation_loss=float(trainer.callback_metrics['val/loss']) if 'val/loss' in trainer.callback_metrics else None,
                  last_checkpoint=str(output.parent/'last.ckpt'),
                  task=config.get('generation_task', 'legacy_joint'), label_coverage=coverage,
                  scope='Training execution validated; material quality is not certified')
    output.with_suffix('.status.json').write_text(json.dumps(status, indent=2), encoding='utf-8')
    return trainer


def run_joint_training(args, config, records, validation, paths, resolved_config=None):
    from unifiedgroupgen.run import training_output, build_loggers
    settings = RunSettings.load(project_root(), getattr(args, 'wandb_mode', None))
    output = training_output(settings, args.experiment, args.output, args.resume)
    print(json.dumps(dict(event='run_directory',trainer='lightning',architecture=checkpoint_format(config),output=str(output))),flush=True)
    if config.get('generation_task') and 'group_prior' not in config:
        from unifiedgroupgen.group_constraints.group_prior import fit_group_prior
        config = {**config, 'group_prior': fit_group_prior(records)}
    previous_runtime = load_joint_checkpoint(args.resume)['runtime'] if args.resume else {}
    train_limit = getattr(args, 'joint_train_limit', None)
    val_limit = getattr(args, 'joint_val_limit', None)
    train_limit = config.get('joint_train_limit') if train_limit is None else train_limit
    val_limit = config.get('joint_val_limit') if val_limit is None else val_limit
    if args.resume:
        train_limit = previous_runtime.get('train_records') if train_limit is None else train_limit
        val_limit = previous_runtime.get('val_records') if val_limit is None else val_limit
    records = records if train_limit is None else records[:int(train_limit)]
    validation = validation if val_limit is None else validation[:int(val_limit)]
    if not records or not validation:
        raise ValueError('Joint train/validation limits must leave nonempty datasets')
    hashes = {split: {str(path): file_hash(path) for path in files} for split, files in paths.items()}
    loggers = build_loggers(args, output, settings, args.resume)
    runtime = {key: getattr(args, key) if getattr(args, key) is not None else previous_runtime.get(key)
               for key in ('batch_size', 'edge_budget')}
    for key in ('precision','fused_adamw','log_every_n_steps'):
        value=getattr(args,key,None)
        if value is not None:runtime[key]=value
        elif key in previous_runtime:runtime[key]=previous_runtime[key]
    max_steps = getattr(args, 'joint_max_steps', None)
    return fit_joint(config, records, validation, output, args.device,
                     args.epochs or config.get('epochs', 100),
                     runtime, hashes,
                     args.resume, getattr(args, 'initialize_from', None),
                     config.get('joint_max_steps', -1) if max_steps is None else max_steps, loggers or False)
