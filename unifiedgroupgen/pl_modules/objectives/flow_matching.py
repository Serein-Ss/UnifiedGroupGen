import torch
from unifiedgroupgen.generation.prior import sample_prior

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

def flow_loss(model, compiled, endpoint, properties=None, observation=None, composition=None, time=None):
    t = torch.rand((), device=endpoint.device) if time is None else endpoint.new_tensor(time)
    state, target = conditional_path(compiled, endpoint, t, observation,
                                     prior=getattr(model, 'prior', None), path=getattr(model, 'flow_path', None))
    prediction = model(compiled, state, t, properties, observation=observation, composition=composition)
    return velocity_loss(compiled, state, target, prediction)

def velocity_loss(compiled, state, target, prediction, lattice=None):
    error = prediction-target
    lattice_error = error[:compiled.lattice_dof].square().mean()
    if compiled.coordinate_dof:
        lattice = compiled.expand(state)["lattice"] if lattice is None else lattice
        physical_error = compiled.coordinate_velocity(error[compiled.lattice_dof:]) @ lattice.T
        coordinate_error = physical_error.square().sum()/max(compiled.coordinate_dof, 1)
    else:
        coordinate_error = lattice_error*0
    return {"loss": lattice_error+coordinate_error, "lattice": lattice_error, "coordinates": coordinate_error}

def flow_loss_batch(model, compiled, endpoints, properties, observations=None, compositions=None, times=None,
                    visibility=None):
    observations = observations or [None]*len(compiled)
    times = times if times is not None else torch.rand(len(compiled), device=endpoints[0].device)
    with torch.autocast(device_type=endpoints[0].device.type, enabled=False):
        paths = [conditional_path(c, z, t, o, prior=getattr(model, 'prior', None), path=getattr(model, 'flow_path', None))
                 for c, z, t, o in zip(compiled, endpoints, times, observations)]
    states, targets = zip(*paths)
    predictions, lattices = model.forward_batch(compiled, states, times, properties, observations=observations,
                                                compositions=compositions, return_lattices=True, property_visibility=visibility)
    with torch.autocast(device_type=endpoints[0].device.type, enabled=False):
        losses = [velocity_loss(c, z, target, prediction, lattice) for c, z, target, prediction, lattice
                  in zip(compiled, states, targets, predictions, lattices)]
    return {key: torch.stack([loss[key] for loss in losses]) for key in ('loss', 'lattice', 'coordinates')}
