"""Opt-in Lightning profiling and compact observations of the joint-flow batches."""
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
import json

import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.profilers import PyTorchProfiler


_trace = ContextVar('joint_flow_debug_trace', default=None)


@contextmanager
def debug_trace(path):
    with path.open('x', encoding='utf-8') as stream:
        token = _trace.set(stream)
        try:
            yield
        finally:
            _trace.reset(token)


def debugging():
    return _trace.get() is not None


def stage(name):
    return torch.profiler.record_function('ugg.' + name) if debugging() else nullcontext()


def observe(name, **fields):
    stream = _trace.get()
    if stream is not None:
        for key, value in fields.items():
            if torch.is_tensor(value):
                fields[key] = dict(shape=list(value.shape), dtype=str(value.dtype), device=str(value.device))
        stream.write(json.dumps(dict(stage=name, **fields), allow_nan=False) + '\n')


class DebugBatches(Callback):
    def _start(self, module, phase, idx):
        if module.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(module.device)
        observe('batch_start', phase=phase, batch=idx)

    def _end(self, module, phase, idx, outputs):
        loss = outputs.get('loss') if isinstance(outputs, dict) else outputs
        memory = {}
        if module.device.type == 'cuda':
            memory = dict(peak_allocated_bytes=torch.cuda.max_memory_allocated(module.device),
                          peak_reserved_bytes=torch.cuda.max_memory_reserved(module.device))
        parts = getattr(module, '_progress_losses', {}).get(phase, {})
        scalars = torch.stack([loss.detach(), *parts.values()]).cpu().tolist()
        observe('batch_end', phase=phase, batch=idx, loss=scalars[0],
                loss_parts=dict(zip(parts, scalars[1:])), **memory)

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        self._start(pl_module, 'train', batch_idx)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        self._end(pl_module, 'train', batch_idx, outputs)

    def on_validation_batch_start(self, trainer, pl_module, batch, batch_idx, dataloader_idx=0):
        self._start(pl_module, 'validation', batch_idx)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        self._end(pl_module, 'validation', batch_idx, outputs)


def debug_trainer_options(settings, directory, device):
    batches = settings.get('fast_dev_run')
    if type(batches) is not int or batches < 1:
        raise ValueError('lightning_debug.fast_dev_run must be a positive integer')
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == 'cuda':
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    profiler = PyTorchProfiler(dirpath=directory, filename='lightning-profile',
                               activities=activities, schedule=None, record_shapes=True,
                               profile_memory=True, record_module_names=False, row_limit=40)
    return dict(fast_dev_run=batches, detect_anomaly=settings.get('detect_anomaly', False),
                profiler=profiler, enable_checkpointing=False)
