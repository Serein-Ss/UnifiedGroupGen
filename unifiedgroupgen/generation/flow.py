import torch
from unifiedgroupgen.generation.prior import sample_prior
from unifiedgroupgen.common.precision import inference_autocast

@torch.no_grad()
def sample_flow(model, compiled, properties=None, steps=64, guidance=1.0, observation=None,
                composition=None, generator=None, return_trajectory=False, condition_observation=True, solver='heun'):
    if model.training:
        raise RuntimeError("Call eval() before sampling")
    if steps < 1:
        raise ValueError("steps must be positive")
    if solver not in ('euler', 'heun'): raise ValueError('solver must be euler or heun')
    parameter = next(model.parameters())
    state = sample_prior(compiled, parameter.device, parameter.dtype, generator, getattr(model, 'prior', None))
    if observation is not None:
        state = observation.retract(state)
    trajectory = [state.clone()] if return_trajectory else None
    def velocity(z, t):
        def predict(force_null=False):
            with inference_autocast(model):
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
        if solver == 'heun':
            v2 = velocity(trial, t+dt)
            state = state + dt*(v1+v2)/2
        else:
            state = trial
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


@torch.no_grad()
def sample_flow_batch(model, compiled, properties=None, steps=64, guidance=0., observations=None,
                      compositions=None, generators=None, solver='heun', condition_observation=True,
                      initial_states=None):
    """One packed graph-network call for all structures at each velocity evaluation."""
    if model.training: raise RuntimeError('Call eval() before sampling')
    if steps < 1 or solver not in ('euler', 'heun'): raise ValueError('Invalid integration options')
    count = len(compiled)
    if not count: return []
    properties, observations = properties or [{} for _ in compiled], observations or [None]*count
    compositions, generators = compositions or [None]*count, generators or [None]*count
    parameter = next(model.parameters())
    states = ([z.clone() for z in initial_states] if initial_states is not None else
              [sample_prior(c, parameter.device, parameter.dtype, g, getattr(model, 'prior', None))
               for c, g in zip(compiled, generators)])
    states = [o.retract(z) if o is not None else z for o, z in zip(observations, states)]
    def velocity(current, time):
        def predict(null):
            with inference_autocast(model):
                values = model.forward_batch(compiled, current, [time]*count, properties, force_null=null,
                    observations=observations if condition_observation else None, compositions=compositions)
            return [o.project(v) if o is not None else v for o, v in zip(observations, values)]
        if guidance == 0: return predict(True)
        conditional = predict(False)
        if guidance == 1: return conditional
        return [u+guidance*(v-u) for u, v in zip(predict(True), conditional)]
    dt = 1/steps
    for step in range(steps):
        first = velocity(states, step*dt)
        trial = [z+dt*v for z, v in zip(states, first)]
        states = ([z+dt*(v+w)/2 for z, v, w in zip(states, first, velocity(trial, (step+1)*dt))]
                  if solver == 'heun' else trial)
        states = [o.retract(z) if o is not None else z for o, z in zip(observations, states)]
        if any(not torch.isfinite(z).all() for z in states):
            raise FloatingPointError('Nonfinite packed flow state; no silent repair')
    return [{**c.expand(z), 'state': z} for c, z in zip(compiled, states)]
