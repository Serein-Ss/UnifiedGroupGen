"""Version-one conditional path, retained for native checkpoint compatibility."""
from dataclasses import dataclass, replace
import numpy as np
import torch
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
from unifiedgroupgen.group_constraints.unified import catalog_spec, SymmetrySpec
from unifiedgroupgen.group_constraints.state import JointState, JointAction, initial_state, apply_action

from unifiedgroupgen.pl_data.targets import JointTarget, target_from_record, _target_from_record
from unifiedgroupgen.pl_data.programs import PathStep, _program

@dataclass
class PathSample:
    state: JointState
    time: float
    metric_anchor: torch.Tensor
    coordinate_anchor: torch.Tensor
    moment_anchor: torch.Tensor | None
    remaining: int
    next_action: JointAction | None
    properties: dict
    identity: str

    def to(self, device, dtype=None):
        return replace(self, state=self.state.to(device, dtype),
                       metric_anchor=self.metric_anchor.to(device=device, dtype=dtype),
                       coordinate_anchor=self.coordinate_anchor.to(device=device, dtype=dtype),
                       moment_anchor=None if self.moment_anchor is None else self.moment_anchor.to(device=device, dtype=dtype))


def _anchors(state, roles, anchors, metric):
    x = torch.cat([anchors[j][0] for j in roles]) if roles else state.coordinates.new_zeros((0, 3))
    m = None
    if state.moments is not None:
        m = torch.cat([anchors[j][1] for j in roles]) if roles else state.moments.new_zeros((0, 3))
    return metric, x, m


def _bridge(state, anchors, start, stop):
    ratio = (1-stop)/(1-start)
    metric, x, m = anchors
    return replace(state, metric=metric+(state.metric-metric)*ratio,
                   coordinates=x+(state.coordinates-x)*ratio,
                   moments=None if m is None else m+(state.moments-m)*ratio)


def sample_training_path(target, seed, time=None, allowed_elements=None, max_atoms=192,
                         max_orbits=52, budget=256, augment=True, birth_scale=.2):
    """Sample the actual conditional PDMP, including correlated birth history.

    The auxiliary program never fixes event times or birth noise. Those are sampled
    here, with the same K_a used by inference. At t=1 its decoded endpoint is data.
    """
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=target.state.metric.device).manual_seed(int(seed))
    allowed = target.state.chemistry.domain(tuple(allowed_elements or sorted(set(target.state.elements))))
    program, anchors = _program(target, rng, allowed, max_atoms, max_orbits, budget, augment)
    time = float(rng.uniform(1e-4, 1-1e-4) if time is None else time)
    if not 0 <= time <= 1:
        raise ValueError('Training time must be in [0, 1]')
    events = np.sort(rng.random(len(program)))
    state = initial_state(target.state.spec, target.state.metric.device, target.state.metric.dtype,
                          generator, budget, target.state.moments is not None, target.state.chemistry)
    roles, last, completed = (), 0., 0
    for event_time, entry in zip(events, program):
        if event_time > time:
            break
        state = _bridge(state, _anchors(state, roles, anchors, target.state.metric), last, float(event_time))
        if target.state.chemistry.counts:
            from unifiedgroupgen.group_constraints.state import legal_actions
            support = legal_actions(state, allowed, max_atoms, max_orbits)
        else:
            support = None
        state = apply_action(state, entry.action, generator, birth_scale, check=support)
        roles, last, completed = entry.roles, float(event_time), completed+1
    current = _anchors(state, roles, anchors, target.state.metric)
    state = _bridge(state, current, last, time)
    return PathSample(state, time, *current, len(program)-completed,
                      program[completed].action if completed < len(program) else None,
                      target.properties, target.identity)
