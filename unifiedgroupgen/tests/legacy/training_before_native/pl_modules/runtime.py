"""Validated runtime and native-resume policy, independent of Trainer construction."""
from pathlib import Path
import numpy as np
import torch


def resume_checkpoint(path):
    resume = None
    previous = None
    if path:
        resume = Path(path).resolve()
        previous = torch.load(resume, map_location='cpu', weights_only=False)
        if previous.get('trainer_backend') == 'lightning' and 'resume_checkpoint' in previous:
            resume = resume.parent/previous['resume_checkpoint']
            previous = torch.load(resume, map_location='cpu', weights_only=False)
        if 'ugg_state' not in previous:
            raise ValueError('Not a native Lightning checkpoint; use initialize_from for weights-only initialization')
    return resume, previous


def runtime_options(args, config, flow, previous, hashes):
    runtime = (dict(previous['ugg_state']['runtime']) if previous else
               {'checkpoint_layers': flow.checkpoint_layers, 'edge_chunk_size': flow.edge_chunk_size,
                'batch_size': config.get('batch_size', 8), 'edge_budget': config.get('edge_budget', 200000),
                'validation_interval': config.get('validation_interval', 5),
                'coordinate_length_scale': config.get('coordinate_length_scale', 3.)})
    for flag, key in [('checkpoint', 'checkpoint_layers'), ('edge_chunk_size', 'edge_chunk_size'),
                      ('batch_size', 'batch_size'), ('edge_budget', 'edge_budget'),
                      ('validation_interval', 'validation_interval'), ('coordinate_length_scale', 'coordinate_length_scale')]:
        value = getattr(args, flag, None)
        if value is not None:
            runtime[key] = value
    for key in ('batch_size', 'edge_budget', 'edge_chunk_size', 'validation_interval'):
        if not isinstance(runtime[key], int) or isinstance(runtime[key], bool) or runtime[key] < 1:
            raise ValueError(key+' must be a positive integer')
    if not np.isfinite(runtime['coordinate_length_scale']) or runtime['coordinate_length_scale'] <= 0:
        raise ValueError('Coordinate length scale must be finite and positive')
    for key in ('composition_probability', 'observation_probability'):
        if not 0 <= config.get(key, .5 if key == 'composition_probability' else .25) <= 1:
            raise ValueError(key+' must lie in [0,1]')
    if previous and (previous['ugg_state']['config'] != config or previous['ugg_state']['runtime'] != runtime
                     or previous['ugg_state']['hashes'] != hashes):
        raise ValueError('Resume configuration, runtime or split hashes differ')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('This trainer supports a single CPU or CUDA device')
    diagnostic_interval = config.get('gradient_diagnostics_interval', 0)
    if not isinstance(diagnostic_interval, int) or isinstance(diagnostic_interval, bool) or diagnostic_interval < 0:
        raise ValueError('gradient_diagnostics_interval must be a nonnegative integer')
    return runtime
