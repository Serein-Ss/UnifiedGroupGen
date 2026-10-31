import torch

@torch.no_grad()
def sample_flow(model, compiled, properties=None, steps=64, guidance=1.0, observation=None,
                composition=None, generator=None, return_trajectory=False, condition_observation=True):
    if model.training:
        raise RuntimeError("Call eval() before sampling")
    if steps < 1:
        raise ValueError("steps must be positive")
    parameter = next(model.parameters())
    state = compiled.prior(parameter.device, parameter.dtype, generator)
    if observation is not None:
        state = observation.retract(state)
    trajectory = [state.clone()] if return_trajectory else None
    def velocity(z, t):
        def predict(force_null=False):
            value = model(compiled, z, t, properties, force_null=force_null,
                          observation=observation if condition_observation else None, composition=composition)
            return observation.project(value) if observation is not None else value
        if guidance == 0:
            return predict(True)
        conditional = predict()
        if guidance == 1:
            return conditional
        unconditional = predict(True)
        return unconditional + guidance*(conditional-unconditional)
    dt = 1/steps
    for step in range(steps):
        t = step*dt
        v1 = velocity(state, t)
        trial = state + dt*v1
        v2 = velocity(trial, t+dt)
        state = state + dt*(v1+v2)/2
        if observation is not None:
            state = observation.retract(state)
        if not torch.isfinite(state).all():
            raise FloatingPointError("Non-finite flow state; sample was not repaired or silently clipped")
        if trajectory is not None:
            trajectory.append(state.clone())
    result = compiled.expand(state)
    result["state"] = state
    if trajectory is not None:
        result["trajectory"] = torch.stack(trajectory)
    return result
