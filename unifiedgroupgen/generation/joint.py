"""Event-aware Heun sampler for the shared mixed generator.

Heun's local dense output locates hazard events before accepting a step. The
interpolation is a numerical approximation; steps and terminal epsilon are saved.
"""
from dataclasses import dataclass, replace
import math
import torch
from unifiedgroupgen.group_constraints.state import initial_state, apply_action


@dataclass
class JointSample:
    state: object
    time: float
    events: list
    reason: str
    integration_steps: int
    epsilon: float


def _velocity(output, state, time):
    divisor = 1-time
    return ((output['metric_anchor']-state.metric)/divisor,
            (output['coordinate_anchor']-state.coordinates)/divisor,
            None if state.moments is None else (output['moment_anchor']-state.moments)/divisor)


def _advance(state, first, second, dt, fraction=1.):
    def value(current, a, b):
        return current+dt*(fraction*a+.5*fraction*fraction*(b-a))
    return replace(state, metric=value(state.metric, first[0], second[0]),
                   coordinates=value(state.coordinates, first[1], second[1]),
                   moments=None if state.moments is None else value(state.moments, first[2], second[2]))


@torch.no_grad()
def sample_joint(model, spec, steps=128, epsilon=1e-4, budget=None,
                 birth_scale=None, magnetic=False, properties=None, generator=None, chemistry=None):
    if getattr(model, 'flow_version', None) == 'joint_flow_v2':
        from unifiedgroupgen.joint_flow.sampling import sample_one
        return sample_one(model, spec, steps=steps, epsilon=epsilon,
                          budget=model.event_budget if budget is None else budget,
                          birth_scale=model.birth_scale if birth_scale is None else birth_scale, magnetic=magnetic, properties=properties,
                          generator=generator, chemistry=chemistry)
    budget = 256 if budget is None else budget
    birth_scale = .2 if birth_scale is None else birth_scale
    if steps < 1 or not 0 < epsilon < .1 or budget < 1:
        raise ValueError('Invalid integration steps, terminal epsilon or event budget')
    if (spec.magnetic or spec.spin_generators or any(op.spin is not None for op in spec.operations)) and not magnetic:
        raise ValueError('A magnetic/spin target requires an enabled moment field; missing moments are not a magnetic structure')
    like = next(model.parameters())
    state = initial_state(spec, like.device, like.dtype, generator, budget, magnetic, chemistry)
    if state.chemistry.counts:
        from unifiedgroupgen.group_constraints.chemistry import can_complete
        if not can_complete(spec, (), (), state.chemistry, budget-2, model.max_orbits, model.max_atoms):
            state = replace(state, status='failed')
            return JointSample(state, 0., [], 'infeasible_inventory', 0, epsilon)
    previous_mode = model.training
    model.eval()
    time, integration_steps, events = 0., 0, []

    def exponential():
        return -math.log(max(float(torch.rand((), device=like.device, generator=generator)), 1e-12))

    threshold = exponential()
    try:
        while time < 1-epsilon and state.status != 'failed':
            dt = min(1/steps, .2*(1-time), 1-epsilon-time)
            if dt <= 1e-12:
                break
            before = model(state, time, properties, actions=())
            v0 = _velocity(before, state, time)
            predictor = _advance(state, v0, v0, dt)
            after = model(predictor, time+dt, properties, actions=())
            v1 = _velocity(after, predictor, time+dt)
            rho0 = float(before['remaining'])/(1-time)
            rho1 = float(after['remaining'])/(1-time-dt)
            increment = .5*dt*(rho0+rho1)
            integration_steps += 1
            if increment < threshold or state.status != 'active':
                state = _advance(state, v0, v1, dt)
                threshold -= increment
                time += dt
            else:
                lo, hi = 0., 1.
                for _ in range(32):
                    fraction = .5*(lo+hi)
                    hazard = dt*(rho0*fraction+.5*(rho1-rho0)*fraction*fraction)
                    if hazard < threshold:
                        lo = fraction
                    else:
                        hi = fraction
                fraction = .5*(lo+hi)
                state = _advance(state, v0, v1, dt, fraction)
                time += dt*fraction
                output = model(state, time, properties)
                if not output['actions']:
                    state = replace(state, status='failed')
                    break
                probabilities = torch.softmax(output['action_logits'], 0)
                index = int(torch.multinomial(probabilities, 1, generator=generator))
                action = output['actions'][index]
                old_count = state.num_atoms
                state = apply_action(state, action, generator, birth_scale, check=output['actions'])
                events.append(dict(time=time, action=action.to_dict(), atoms_before=old_count,
                                   atoms_after=state.num_atoms))
                threshold = exponential()
            if not all(torch.isfinite(v).all() for v in (state.metric, state.coordinates)
                       if v.numel()) or (state.moments is not None and not torch.isfinite(state.moments).all()):
                state = replace(state, status='failed')
                return JointSample(state, time, events, 'nonfinite_continuous_state', integration_steps, epsilon)
        reason = 'complete' if state.status == 'done' else 'event_budget' if state.step >= state.budget else 'unfinished_at_terminal_time'
        if state.status == 'failed' and events and events[-1]['action']['kind'] == 'fail' and state.step < state.budget:
            reason = 'no_legal_completion'
        if state.status != 'done':
            state = replace(state, status='failed')
        return JointSample(state, time, events, reason, integration_steps, epsilon)
    except (torch.linalg.LinAlgError, FloatingPointError) as error:
        state = replace(state, status='failed')
        return JointSample(state, time, events, 'numerical_failure: '+str(error), integration_steps, epsilon)
    finally:
        model.train(previous_mode)
