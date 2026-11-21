"""Opt-in native Lightning training and versioned checkpoints for the joint generator."""
from pathlib import Path
from unifiedgroupgen.pl_modules.training_utils import strict_fp32, _normalize_epoch_boundary
from contextlib import contextmanager, nullcontext
import os
import hashlib
from dataclasses import replace
import json
import time
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger
from lightning_fabric.utilities.seed import _collect_rng_states, _set_rng_states
from unifiedgroupgen.common.configuration import RunSettings, project_root
from unifiedgroupgen.common.source import file_hash
from unifiedgroupgen.common.structures import write_json
from unifiedgroupgen.common.progress import interactive
from unifiedgroupgen.pl_modules.progress import progress_callbacks, cache_progress_losses
from unifiedgroupgen.pl_data.datamodule import JointDataModule
from unifiedgroupgen.common.data_utils import target_from_record
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_modules.debug import DebugBatches, debugging, debug_trace, debug_trainer_options, observe, stage


FLOW_FORMAT = 'ugg_joint_flow_v2'


def _current_checkpoint(state):
    return state.get('format') == FLOW_FORMAT


def checkpoint_format(config):
    if config.get('model', {}).get('family') != 'joint_flow_v2':
        raise ValueError('Production training requires joint_flow_v2')
    return FLOW_FORMAT


def _update_flow_status(output, **values):
    path=Path(output).with_suffix('.status.json')
    previous=json.loads(path.read_text(encoding='utf-8')) if path.exists() else {}
    write_json(path,{**previous,**values,'updated':time.time(),'pid':os.getpid()})


@contextmanager
def _flow_status(output,config):
    """Observe v2 initialization, execution and exceptions without changing training."""
    checkpoint_format(config)
    write_json(Path(output).with_suffix('.status.json'),dict(format=FLOW_FORMAT,
        state='initializing',phase='building_model',started=time.time(),updated=time.time(),
        pid=os.getpid(),epoch=None,global_step=None))
    def progress(phase,**values):
        _update_flow_status(output,phase=phase,**values)
    try:
        yield progress
    except BaseException as error:
        _update_flow_status(output,state='interrupted' if isinstance(error,(KeyboardInterrupt,SystemExit)) else 'failed',
                            error_type=type(error).__name__,error=str(error))
        raise




def joint_callbacks(config,output):
    monitor=config.get('validation_monitor','val/loss')
    callbacks=[JointTrainingArtifacts(output), *progress_callbacks(Path(output))]
    if config.get('early_stopping'):
        callbacks.append(EarlyStopping(monitor=monitor,mode='min',check_finite=True,
            check_on_train_epoch_end=False,verbose=False,**config['early_stopping']))
    callbacks.append(ModelCheckpoint(dirpath=Path(output).parent,filename='best',monitor=monitor,
                     mode='min',save_top_k=1,save_last=True,save_on_train_epoch_end=True))
    return callbacks




def load_joint_checkpoint(path, device='cpu'):
    saved = torch.load(path, map_location=device, weights_only=False)
    if _current_checkpoint(saved):
        return saved
    state = saved.get('joint_state')
    if not state or not _current_checkpoint(state):
        raise ValueError('Expected a ugg_joint_flow_v2 checkpoint')
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
        self._pending_lr_metric = None
        self._validation_rng = None
        self.save_hyperparameters(dict(config=self.config, runtime=self.runtime, split_hashes=self.split_hashes))

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return batch

    def objective(self, records, validation=False):
        if hasattr(records,'records'):
            if debugging():
                atom_counts = records.natoms.tolist()
                observe('endpoint_batch', structures=records.num_graphs, nodes=records.num_nodes,
                        atom_counts=atom_counts,
                        edge_index=records.edge_index, atomic_numbers=records.atomic_numbers, lattice=records.cell)
            records=records.records
        from unifiedgroupgen.conditional.tasks import eligible_records
        records = eligible_records(records, self.config)
        if not records:
            raise ValueError('No eligible records for this training batch')
        from unifiedgroupgen.pl_data.unified.path import sample_path as path_sampler
        from unifiedgroupgen.pl_modules.unified.objectives import loss as objective
        samples, recovery = [], []
        recovery_weight = 0. if validation else self.config.get('orbit_recovery_weight', 0.)
        if recovery_weight:
            from unifiedgroupgen.pl_data.unified.recovery import recovery_example
        for row in records:
            identity = (row.get('dataset', ''), row['id'])
            if identity not in self._targets:
                self._targets[identity] = row.get('_joint_target') or target_from_record(
                    row, operation_compiler=self.config.get('model', {}).get('operation_compiler', False))
            digest = int.from_bytes(hashlib.sha256(repr(identity).encode()).digest()[:4], 'little')
            seed = (digest+918 if validation else digest+int(self.global_step)*100003+int(self.current_epoch)*977)
            from unifiedgroupgen.pl_data.unified.tasks import task_target
            with stage('data.sample_path'):
                target = task_target(self._targets[identity], self.config, seed)
                sample = path_sampler(target, seed,
                         allowed_elements=self.model.allowed_elements, max_atoms=self.model.max_atoms,
                         max_orbits=self.model.max_orbits, budget=self.config.get('event_budget', 256),
                         augment=not validation, birth_scale=self.config.get('birth_scale', .2),
                         orbit_order=self.config.get('orbit_order', 'random'),
                         orbit_ablation=self.model.orbit_ablation)
            with stage('data.transfer_intermediate'):
                dtype = next(self.model.parameters()).dtype
                samples.append(sample.to(self.device, dtype))
                if recovery_weight:
                    example = recovery_example(sample, target.state, seed, self.model.allowed_elements,
                                               self.model.max_atoms, self.model.max_orbits,
                                               self.config.get('birth_scale', .2))
                    if example is not None:
                        recovery.append(replace(example, state=example.state.to(self.device, dtype)))
            if debugging():
                observe('intermediate', sample_id=row['id'], kind=sample.state.spec.kind,
                        atoms=sample.state.num_atoms, orbits=len(sample.state.letters), time=sample.time,
                        next_action=sample.next_action.kind if sample.next_action is not None else None,
                        metric=samples[-1].state.metric, coordinates=samples[-1].state.coordinates)
        with stage('model.forward_and_loss'):
            return objective(self.model, samples, self.config.get('jump_weight', 1.),
                             recovery=recovery, recovery_weight=recovery_weight,
                             geometry_objective=self.config.get('geometry_objective'),
                             short_contact_weight=self.config.get('short_contact_weight'),
                             layer_log_area_weight=self.config.get('layer_log_area_weight'),
                             layer_height_weight=self.config.get('layer_height_weight'))

    def training_step(self, batch, batch_idx):
        losses = self.objective(batch)
        cache_progress_losses(self,'train',losses)
        self.log_dict({'train/'+name: value for name, value in losses.items()},
                      on_step=True, on_epoch=True, batch_size=batch.num_graphs if hasattr(batch,'num_graphs') else len(batch))
        return losses['loss']

    def validation_step(self, batch, batch_idx):
        losses = self.objective(batch, validation=True)
        cache_progress_losses(self,'validation',losses)
        self.log_dict({'val/'+name: value for name, value in losses.items()},
                      on_step=False, on_epoch=True, batch_size=batch.num_graphs if hasattr(batch,'num_graphs') else len(batch))
        return losses['loss'].detach()

    def configure_optimizers(self):
        options={'fused':True} if self.runtime.get('fused_adamw',False) else {}
        optimizer=torch.optim.AdamW(self.parameters(), lr=self.config.get('learning_rate', 1e-3),**options)
        if not self.config.get('lr_schedule'):return optimizer
        scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',
                         threshold_mode='abs',**self.config['lr_schedule'])
        return dict(optimizer=optimizer,lr_scheduler=dict(scheduler=scheduler,
            monitor=self.config.get('validation_monitor','val/loss'),interval='epoch',frequency=1,strict=True))

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
        boundary=_normalize_epoch_boundary(checkpoint,self.trainer.num_training_batches)
        pending=None
        # PL 2.5 saves epoch callbacks before stepping ReduceLROnPlateau.
        # Final public save_checkpoint occurs after the step and needs no replay.
        if boundary and self.config.get('lr_schedule') and self.lr_schedulers().last_epoch<checkpoint['epoch']:
            pending=float(self.trainer.callback_metrics[self.config.get('validation_monitor','val/loss')])
        checkpoint['joint_state'] = dict(format=checkpoint_format(self.config), config=self.config, runtime=self.runtime,
                                        split_hashes=self.split_hashes, rng=_collect_rng_states(),
                                        epoch_boundary=boundary,pending_lr_metric=pending)
        logger=getattr(self,'logger',None)
        if isinstance(logger,WandbLogger):
            checkpoint['joint_state'].update(wandb_id=logger.version,
                wandb_run_directory=str(Path(self.trainer.default_root_dir).resolve()))

    def on_load_checkpoint(self, checkpoint):
        state = checkpoint.get('joint_state', {})
        if not _current_checkpoint(state) or state.get('config') != self.config:
            raise ValueError('Joint resume architecture/training configuration differs')
        if state.get('runtime') != self.runtime or state.get('split_hashes') != self.split_hashes:
            raise ValueError('Joint resume execution settings or dataset hashes differ')
        self._pending_rng = state['rng']
        self._pending_lr_metric = state.get('pending_lr_metric')

    def on_train_start(self):
        if self._pending_lr_metric is not None:
            self.lr_schedulers().step(self._pending_lr_metric)
            self._pending_lr_metric=None
        if self._pending_rng is not None:
            _set_rng_states(self._pending_rng)
            self._pending_rng = None

    def on_train_epoch_start(self):
        if self.config.get('lr_schedule'):
            self.log('train/learning_rate',self.optimizers(use_pl_optimizer=False).param_groups[0]['lr'],
                     on_step=False,on_epoch=True,batch_size=1)

    def on_validation_epoch_start(self):
        if self.config.get('validation_seed') is not None:
            self._validation_rng=_collect_rng_states()
            torch.manual_seed(self.config['validation_seed'])

    def on_validation_epoch_end(self):
        if self._validation_rng is not None:
            _set_rng_states(self._validation_rng)
            self._validation_rng=None


class JointTrainingArtifacts(pl.Callback):
    def __init__(self, output):
        self.output = Path(output)
        self.orbit_example = None

    def on_train_batch_end(self, trainer, module, outputs, batch, batch_idx):
        if batch_idx == 0:
            self.orbit_example = None
            records = getattr(batch, 'records', ())
            if records and (target := records[0].get('_joint_target')) is not None:
                from unifiedgroupgen.group_constraints.wyckoff import record_orbit_labels
                record = records[0]
                self.orbit_example = dict(id=target.identity, dataset=record.get('dataset'),
                    kind=target.state.spec.kind, group=(record.get('descriptor') or {}).get('number'),
                    orbit_labels=record_orbit_labels(record, target.state), letters=list(target.state.letters))
        if batch_idx == 0 or trainer.global_step % 25 == 0:
            value = outputs['loss'] if isinstance(outputs,dict) else outputs
            status = dict(format=checkpoint_format(module.config),state='running',epoch=trainer.current_epoch,
                          global_step=trainer.global_step,loss=float(value.detach()))
            if self.orbit_example is not None:
                status['target_orbit_example'] = self.orbit_example
            _update_flow_status(self.output,phase='training',**status)

    def on_validation_start(self,trainer,module):
        _update_flow_status(self.output,state='running',phase='validation',
                            epoch=trainer.current_epoch,global_step=trainer.global_step)

    def on_train_epoch_end(self, trainer, module):
        metrics = {name: float(value.detach()) for name,value in trainer.callback_metrics.items()
                   if torch.is_tensor(value) and value.numel() == 1}
        row = dict(epoch=trainer.current_epoch,global_step=trainer.global_step,metrics=metrics,
                   learning_rates_used=[g['lr'] for opt in trainer.optimizers for g in opt.param_groups])
        if self.orbit_example is not None:
            row['target_orbit_example'] = self.orbit_example
        for callback in trainer.callbacks:
            if isinstance(callback,EarlyStopping):
                score=float(callback.best_score)
                row['early_stopping']=dict(wait_count=callback.wait_count,patience=callback.patience,
                                           best_score=score if torch.isfinite(callback.best_score) else None)
        with self.output.with_suffix('.history.jsonl').open('a',encoding='utf-8') as stream:
            stream.write(json.dumps(row,allow_nan=False)+'\n')
        _update_flow_status(self.output,state='running',phase='epoch_end',epoch=trainer.current_epoch,
                            global_step=trainer.global_step,validation_loss=metrics.get('val/loss'))


def fit_joint(config, records, validation, output, device='cpu', epochs=1, runtime=None,
              split_hashes=None, resume=None, initialize_from=None, max_steps=-1, logger=False, debug=None):
    """Reusable implementation for the normal Hydra entry and bounded local smoke runs."""
    checkpoint_format(config)
    if debug is not None and resume:
        raise ValueError('Use a new run directory for Lightning debugging')
    identities = lambda rows: {(row.get('dataset', ''), row['id']) for row in rows}
    if not records or not validation or identities(records) & identities(validation):
        raise ValueError('Joint training requires nonempty, identifier-disjoint train and validation records')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    protected=(output,output.with_suffix('.history.jsonl'),output.with_suffix('.status.json'),output.with_suffix('.steps.jsonl'),
               output.parent/'model.best.pt',output.parent/'train.log',*output.parent.glob('*.ckpt'))
    if not resume and any(path.exists() or path.is_symlink() for path in protected):
        raise FileExistsError('Use a new joint training output; existing checkpoints are preserved')
    from unifiedgroupgen.conditional.tasks import eligible_records, generation_mode, validate_task_config
    validate_task_config(config)
    prior_records = records
    coverage = dict(train_requested=len(records), val_requested=len(validation))
    records, validation = eligible_records(records, config), eligible_records(validation, config)
    if not records or not validation:
        raise ValueError('Property conditioning needs observed target labels in train and validation')
    coverage.update(train_labelled=len(records), val_labelled=len(validation))
    if config.get('generation_task') and 'group_prior' not in config:
        from unifiedgroupgen.group_constraints.group_prior import fit_group_prior
        config = {**config, 'group_prior': fit_group_prior(prior_records, composition_ranked=config.get('csp_group_policy')=='composition_ranked')}
    config = {**config, 'magnetic_training': any(r.get('magnetic_moments') is not None for r in records)}
    actual_device = torch.device(device)
    device_name = 'cuda:'+str(actual_device.index or 0) if actual_device.type == 'cuda' else str(actual_device)
    runtime = {'batch_size': 4, **(runtime or {}),
               'device': device_name, 'train_records': len(records), 'val_records': len(validation)}
    precision=runtime.get('precision',config.get('precision','fp32'))
    modes={'fp32':'32-true','bf16':'bf16-mixed','fp16':'16-mixed'}
    if precision not in modes:raise ValueError('Joint precision must be fp32, bf16 or fp16')
    if runtime.get('fused_adamw',False) and actual_device.type != 'cuda':
        raise ValueError('Fused AdamW profile requires CUDA')
    if runtime.get('fused_adamw',False) and precision=='fp16' and config.get('gradient_clip',10.):
        raise ValueError('FP16 with clipping requires unfused AdamW; BF16 fused clipping is supported')
    strict_fp32(runtime)
    with _flow_status(output,config) as progress:
        progress('building_model',max_epochs=1 if debug is not None else epochs,
                 train_records=len(records),val_records=len(validation),
                 batch_size=runtime['batch_size'],precision=precision,resumed_from=str(resume) if resume else None)
        module = JointGeneratorModule(config, runtime, split_hashes)
        if initialize_from:
            module.model.load_state_dict(load_joint_checkpoint(initialize_from)['model'], strict=True)
        progress('preparing_targets')
        data = JointDataModule(records, validation, runtime, config.get('model', {}).get('operation_compiler', False))
        progress('trainer_setup')
        options = debug_trainer_options(debug, output.parent, actual_device) if debug is not None else {}
        callbacks = [DebugBatches()] if debug is not None else joint_callbacks(config, output)
        trainer = pl.Trainer(accelerator='gpu' if str(device).startswith('cuda') else 'cpu',
                             devices=[torch.device(device).index or 0] if str(device).startswith('cuda') else 1,
                             max_epochs=epochs, max_steps=max_steps, precision=modes[precision],
                             logger=False if debug is not None else logger, callbacks=callbacks, default_root_dir=str(output.parent),
                             enable_model_summary=False, enable_progress_bar=interactive(), num_sanity_val_steps=0,
                             gradient_clip_val=config.get('gradient_clip', 10.),
                             check_val_every_n_epoch=1, log_every_n_steps=runtime.get('log_every_n_steps',1), inference_mode=False,
                             reload_dataloaders_every_n_epochs=1, **options)
        progress('waiting_for_first_batch')
        trace = debug_trace(output.parent/'debug.batches.jsonl') if debug is not None else nullcontext()
        with trace:
            trainer.fit(module, datamodule=data, ckpt_path=str(resume) if resume else None)
        if trainer.interrupted:
            raise KeyboardInterrupt('Joint training interrupted; native last.ckpt is resumable')
        if debug is not None:
            progress('finished', state='complete', lightning_debug=True, global_step=trainer.global_step)
            return trainer
        # Capture the completed loop boundary, rather than the callback's in-hook
        # snapshot. This is a public Lightning API and preserves optimizer/RNG state.
        progress('saving',state='running')
        trainer.save_checkpoint(output.parent/'last.ckpt')
        torch.save(dict(format=checkpoint_format(config), config=config, runtime=runtime, split_hashes=split_hashes or {},
                        model=module.model.state_dict(), epoch=trainer.current_epoch,
                        global_step=trainer.global_step), output)
        checkpoint_callback=next(c for c in trainer.callbacks if isinstance(c,ModelCheckpoint))
        status = dict(format=checkpoint_format(config), state='complete', global_step=trainer.global_step,
                      epochs_completed=trainer.current_epoch,max_epochs=epochs,
                      stop_reason='early_stopping' if any(isinstance(c,EarlyStopping) and c.stopped_epoch>0 for c in trainer.callbacks) else 'training_budget_reached',
                      best_validation_loss=float(checkpoint_callback.best_model_score) if checkpoint_callback.best_model_score is not None else None,
                      train_records=len(records), val_records=len(validation),
                      magnetic_label_records=sum(r.get('magnetic_moments') is not None for r in records),
                      validation_loss=float(trainer.callback_metrics['val/loss']) if 'val/loss' in trainer.callback_metrics else None,
                      last_checkpoint=str(output.parent/'last.ckpt'),
                      task=generation_mode(config) or 'dng', property_conditioning=sorted(config.get('properties', {})),
                      label_coverage=coverage,
                      scope='Training execution validated; material quality is not certified')
        _update_flow_status(output,phase='finished',**status)
        return trainer


def run_joint_training(args, config, records, validation, paths):
    from unifiedgroupgen.run import training_output, build_loggers
    settings = RunSettings.load(project_root(), getattr(args, 'wandb_mode', None))
    debug = getattr(args, 'lightning_debug', None)
    if debug is not None and args.resume:
        raise ValueError('Use a new run directory for Lightning debugging')
    output = training_output(settings, args.experiment, args.output, args.resume)
    print(f'Training | {checkpoint_format(config)} | output={output}',flush=True)
    if config.get('generation_task') and 'group_prior' not in config:
        from unifiedgroupgen.group_constraints.group_prior import fit_group_prior
        config = {**config, 'group_prior': fit_group_prior(records, composition_ranked=config.get('csp_group_policy')=='composition_ranked')}
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
    from unifiedgroupgen.pl_data.cache import training_split_hashes
    hashes = training_split_hashes(paths)
    loggers = build_loggers(args, output, settings, args.resume) if debug is None else False
    runtime = {}
    for key in ('batch_size','precision','fused_adamw','log_every_n_steps','strict_fp32'):
        value=getattr(args,key,None)
        if value is not None:runtime[key]=value
        elif key in previous_runtime:runtime[key]=previous_runtime[key]
    max_steps = getattr(args, 'joint_max_steps', None)
    return fit_joint(config, records, validation, output, args.device,
                     args.epochs or config.get('epochs', 100),
                     runtime, hashes,
                     args.resume, getattr(args, 'initialize_from', None),
                     config.get('joint_max_steps', -1) if max_steps is None else max_steps,
                     loggers or False, debug=debug)
