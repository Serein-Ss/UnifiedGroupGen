"""Property-only classifier-free guidance of the mixed crystal generator."""
import math

import torch


def validate_guidance(scale, has_properties, dropout):
    if not math.isfinite(scale) or scale < 0:
        raise ValueError('guidance_scale must be finite and nonnegative')
    if scale != 1 and (not has_properties or not 0 < dropout < 1):
        raise ValueError('Non-unit CFG scale requires target properties and a checkpoint trained with property dropout')


def guided_forward(model, states, times, properties, actions, scale):
    if scale == 1 or not any(properties):
        return model.forward_batch(states, times, properties, actions=actions)
    if scale == 0:
        return model.forward_batch(states, times, [{} for _ in states], actions=actions)
    conditioned = model.forward_batch(states, times, properties, actions=actions)
    indices = [i for i, value in enumerate(properties) if value]
    null = model.forward_batch([states[i] for i in indices], [times[i] for i in indices],
                              [{} for _ in indices],
                              actions=None if actions is None else [actions[i] for i in indices])
    for index, empty in zip(indices, null):
        conditioned[index] = blend_generator(empty, conditioned[index], states[index], scale)
    return conditioned


def blend_generator(null, condition, state, scale):
    if null['actions'] != condition['actions']:
        raise ValueError('Property dropout changed the hard orbit action support')
    guided = dict(condition)
    for field in ('metric_velocity_scaled', 'coordinate_velocity_scaled', 'action_logits'):
        guided[field] = null[field] + scale * (condition[field] - null[field])
    baseline, target = null['element_logits'], condition['element_logits']
    forbidden = torch.isneginf(target)
    if not torch.equal(forbidden, torch.isneginf(baseline)):
        raise ValueError('Property dropout changed the hard element mask')
    baseline, target = baseline.masked_fill(forbidden, 0), target.masked_fill(forbidden, 0)
    guided['element_logits'] = (baseline + scale * (target - baseline)).masked_fill(forbidden, -torch.inf)
    if not state.frozen and state.status == 'active':
        lower = torch.finfo(torch.float32).tiny
        baseline = null['remaining'].float().clamp_min(lower).log()
        target = condition['remaining'].float().clamp_min(lower).log()
        guided['remaining'] = (baseline + scale * (target - baseline)).clamp(
            min=0., max=math.log(max(1, state.budget-state.step))).exp()
    for field in ('metric_velocity_scaled', 'coordinate_velocity_scaled', 'action_logits', 'remaining'):
        if not torch.isfinite(guided[field]).all():
            raise ValueError('Nonfinite property-guided generator output: ' + field)
    if not torch.isfinite(guided['element_logits'][~forbidden]).all():
        raise ValueError('Nonfinite property-guided element logits')
    return guided
