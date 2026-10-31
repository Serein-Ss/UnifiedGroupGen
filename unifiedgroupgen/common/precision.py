"""One inference policy: AMP for neural work, float32 for chart states and solves."""
from contextlib import nullcontext
import torch


def resolve_precision(device, requested='auto'):
    device = torch.device(device)
    if requested == 'auto':
        return 'bf16' if device.type == 'cuda' and torch.cuda.is_bf16_supported() else 'fp32'
    if requested not in ('fp32', 'bf16'):
        raise ValueError('Inference precision must be auto, fp32 or bf16')
    if requested == 'bf16' and (device.type != 'cuda' or not torch.cuda.is_bf16_supported()):
        raise ValueError('BF16 inference requires a CUDA GPU with BF16 support')
    return requested


def inference_autocast(model):
    precision = getattr(model, 'inference_precision', 'fp32')
    return torch.autocast('cuda', dtype=torch.bfloat16) if precision == 'bf16' else nullcontext()
