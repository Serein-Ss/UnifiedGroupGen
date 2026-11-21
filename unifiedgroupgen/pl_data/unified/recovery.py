"""Auxiliary recovery states, independent of the generator-matching teacher path."""
from collections import Counter
from dataclasses import dataclass
from random import Random

import torch

from unifiedgroupgen.group_constraints.flow_state import FlowState, actions, apply
from unifiedgroupgen.group_constraints.state import JointAction


@dataclass(frozen=True)
class RecoveryExample:
    state: FlowState
    time: float
    properties: dict
    correction: JointAction
    support: tuple[JointAction, ...]


def recovery_example(sample, endpoint, seed, allowed, max_atoms, max_orbits, birth_scale):
    """Add a legal off-target birth and label its inverse if the inverse is legal."""
    state = sample.state
    if state.frozen or len(state.letters) == 0:
        return None
    if state.chemistry.counts:
        expected = Counter(zip(endpoint.letters, endpoint.elements))
        current = Counter(zip(state.letters, state.elements))
        def is_extra(action):
            key = (action.letter, action.element)
            return current[key] >= expected[key]
    else:
        expected, current = Counter(endpoint.letters), Counter(state.letters)
        def is_extra(action):
            return current[action.letter] >= expected[action.letter]

    choices = [a for a in actions(state, allowed, max_atoms, max_orbits)
               if a.kind == 'birth' and is_extra(a)]
    rng = Random(seed ^ 0x79C4CDD9)
    rng.shuffle(choices)
    generator = torch.Generator(device=state.metric.device).manual_seed(rng.getrandbits(63))
    correction = JointAction('delete', orbit=len(state.letters))
    for birth in choices[:16]:
        trial = apply(state, birth, generator, birth_scale)
        support = actions(trial, allowed, max_atoms, max_orbits)
        if correction in support:
            return RecoveryExample(trial, sample.time, sample.properties, correction, support)
    return None
