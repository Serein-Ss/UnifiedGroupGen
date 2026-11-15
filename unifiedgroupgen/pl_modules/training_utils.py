"""Shared execution policy and native checkpoint loop normalization."""
import os
import torch

def strict_fp32(runtime):
    if runtime.get('strict_fp32',False):
        if runtime.get('precision','fp32')!='fp32':
            raise ValueError('strict_fp32 requires precision=fp32')
        if os.environ.get('TORCH_ALLOW_TF32_CUBLAS_OVERRIDE')=='1':
            raise ValueError('Disable TORCH_ALLOW_TF32_CUBLAS_OVERRIDE for strict FP32')
        torch.set_float32_matmul_precision('highest')
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False


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
