"""Continuous teacher states for the two-stage geometric model."""
import torch
from unifiedgroupgen.generation.two_stage.prior import sample_prior

def conditional_path(compiled, endpoint, time, observation=None, generator=None, prior=None, path=None):
    start = sample_prior(compiled, endpoint.device, endpoint.dtype, generator, prior)
    if observation is not None:
        start = observation.retract(start)
        if not torch.allclose(observation.retract(endpoint), endpoint, atol=1e-5):
            raise ValueError("Training endpoint does not satisfy the observations")
        # General linear combinations need a consistent lift. Independent wrapping could leave Az=b.
        displacement = endpoint-start
    else:
        periodic = compiled.tensor(compiled.state_periodic, endpoint, torch.bool)
        delta = endpoint-start
        displacement = torch.where(periodic, torch.remainder(delta+0.5, 1)-0.5, delta)
    state = start + time*displacement
    sigma = (path or {}).get('sigma', 0.)
    if sigma:
        if sigma < 0: raise ValueError('Bridge sigma must be nonnegative')
        time = torch.as_tensor(time, device=endpoint.device, dtype=endpoint.dtype)
        noise = torch.randn(endpoint.shape, device=endpoint.device, dtype=endpoint.dtype, generator=generator)
        if observation is not None: noise = observation.project(noise)
        state = state + sigma*torch.sin(torch.pi*time)*noise
        displacement = displacement + sigma*torch.pi*torch.cos(torch.pi*time)*noise
    return state, displacement
