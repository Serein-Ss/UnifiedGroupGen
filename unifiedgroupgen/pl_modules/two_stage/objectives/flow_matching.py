import torch
from unifiedgroupgen.pl_data.two_stage.path import conditional_path
from unifiedgroupgen.generation.two_stage.prior import sample_prior  # Legacy imported helper.


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
