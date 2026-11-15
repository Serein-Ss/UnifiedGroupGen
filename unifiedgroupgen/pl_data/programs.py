"""One shared teacher-program implementation for training paths."""
from dataclasses import dataclass, replace
import numpy as np
import torch
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
from unifiedgroupgen.group_constraints.unified import catalog_spec, SymmetrySpec
from unifiedgroupgen.group_constraints.state import JointState, JointAction, initial_state, apply_action

from unifiedgroupgen.group_constraints.chemistry import edit_signature

@dataclass(frozen=True)
class PathStep:
    action: JointAction
    roles: tuple


def _program(target, rng, allowed, max_atoms, max_orbits, budget, augment):
    state = target.state
    if state.num_atoms > max_atoms or len(state.letters) > max_orbits:
        raise ValueError('Target exceeds the configured atom/orbit limits')
    if set(state.elements)-set(allowed):
        raise ValueError('Target element is outside the TRAIN-only element domain')
    if state.chemistry.counts:
        return inventory_program(target, rng, allowed, max_atoms, max_orbits, budget, augment)
    real_x = state.coordinates.split([w.multiplicity for w in state.compiled.templates])
    real_m = None if state.moments is None else state.moments.split([w.multiplicity for w in state.compiled.templates])
    anchors = {j: (x, None if real_m is None else real_m[j]) for j, x in enumerate(real_x)}
    roles, letters, program = [], [], []
    ghost = -1

    def append(action, role=None):
        if action.kind in ('birth', 'split'):
            roles.append(role); letters.append(action.letter)
        elif action.kind == 'replace':
            roles[action.orbit], letters[action.orbit] = role, action.letter
        elif action.kind == 'delete':
            roles.pop(action.orbit); letters.pop(action.orbit)
        program.append(PathStep(action, tuple(roles)))

    def temporary(w):
        nonlocal ghost
        a, b = w.arrays()
        x = state.coordinates.new_tensor(b + rng.normal(0, .2, b.shape))
        m = None if real_m is None else state.moments.new_tensor(rng.normal(size=b.shape))
        role = ghost; ghost -= 1
        anchors[role] = (x, m)
        return role

    order = rng.permutation(len(state.letters))
    for j in order:
        letter, z = state.letters[j], state.elements[j]
        current = sum(state.spec.template(w).multiplicity for w in letters)
        candidates = [w for w in state.spec.templates if current+w.multiplicity <= max_atoms
                      and (w.dof or w.letter not in letters)]
        if augment and rng.random() < .25 and len(letters) < max_orbits and candidates:
            w = candidates[int(rng.integers(len(candidates)))]
            append(JointAction('birth', letter=w.letter), temporary(w))
            append(JointAction('element', orbit=len(letters)-1, element=int(rng.choice(allowed))))
            append(JointAction('delete', orbit=len(letters)-1))
        wrong = [w for w in candidates if w.letter != letter]
        if augment and wrong and rng.random() < .25:
            w = wrong[int(rng.integers(len(wrong)))]
            append(JointAction('birth', letter=w.letter), temporary(w))
            append(JointAction('element', orbit=len(letters)-1, element=int(rng.choice(allowed))))
            append(JointAction('replace', len(letters)-1, letter, z), int(j))
        else:
            parents = [i for i, old in enumerate(letters) if old == letter and state.spec.template(old).dof]
            if parents and rng.random() < .5:
                append(JointAction('split', orbit=int(rng.choice(parents)), letter=letter), int(j))
            else:
                append(JointAction('birth', letter=letter), int(j))
            alternatives = [a for a in allowed if a != z]
            if augment and alternatives and rng.random() < .2:
                append(JointAction('element', orbit=len(letters)-1, element=int(rng.choice(alternatives))))
            append(JointAction('element', orbit=len(letters)-1, element=z))
    if augment and rng.random() < .25:
        pairs = [(i,j) for i in range(len(roles)) for j in range(i+1,len(roles))
                 if state.elements[roles[i]] != state.elements[roles[j]]]
        if pairs:
            i,j = pairs[int(rng.integers(len(pairs)))]
            append(JointAction('swap',orbit=i,partner=j))
            append(JointAction('swap',orbit=i,partner=j))
    append(JointAction('end'))
    if len(program) >= budget:
        if augment:
            return _program(target, rng, allowed, max_atoms, max_orbits, budget, False)
        raise ValueError('Event budget cannot represent the target path')
    return program, anchors


def inventory_program(target, rng, allowed, max_atoms, max_orbits, budget, augment):
    from unifiedgroupgen.group_constraints.state import JointState, JointAction, legal_actions
    endpoint = target.state
    real_x = endpoint.coordinates.split([w.multiplicity for w in endpoint.compiled.templates])
    real_m = None if endpoint.moments is None else endpoint.moments.split([w.multiplicity for w in endpoint.compiled.templates])
    anchors = {j: (x, None if real_m is None else real_m[j]) for j, x in enumerate(real_x)}
    letters, elements, roles, program = (), (), [], []
    ghost = -1

    def visible():
        n = sum(endpoint.spec.template(w).multiplicity for w in letters)
        return JointState(endpoint.spec, letters, elements, endpoint.metric.new_zeros(endpoint.metric.shape),
                          endpoint.coordinates.new_zeros((n, 3)), step=len(program), budget=budget,
                          chemistry=endpoint.chemistry)

    def support():
        return legal_actions(visible(), allowed, max_atoms, max_orbits)

    def append(action, role=None):
        nonlocal letters, elements
        if action not in support():
            raise ValueError('Inventory teacher action is outside inference support')
        letters, elements = edit_signature(letters, elements, action)
        if action.kind in ('birth', 'split'):
            roles.append(role)
        elif action.kind == 'replace':
            roles[action.orbit] = role
        elif action.kind == 'delete':
            roles.pop(action.orbit)
        program.append(PathStep(action, tuple(roles)))

    def temporary(letter):
        nonlocal ghost
        w = endpoint.spec.template(letter)
        _, b = w.arrays()
        x = endpoint.coordinates.new_tensor(b+rng.normal(0, .2, b.shape))
        m = None if real_m is None else endpoint.moments.new_tensor(rng.normal(size=b.shape))
        role = ghost; ghost -= 1
        anchors[role] = (x, m)
        return role

    order = rng.permutation(len(endpoint.letters))
    for position, j in enumerate(order):
        # A two-action reversible detour teaches corrections without changing
        # the final inventory or conditioning the birth kernel on its noise.
        if augment and rng.random() < .3 and len(program)+len(order)-position+3 < budget:
            candidates = [a for a in support() if a.kind in ('birth', 'element', 'replace', 'swap')]
            rng.shuffle(candidates)
            for action in candidates[:32]:
                old_role = roles[action.orbit] if action.orbit >= 0 else None
                inverse = (JointAction('delete', orbit=len(letters)) if action.kind == 'birth' else
                           JointAction('element', orbit=action.orbit, element=elements[action.orbit]) if action.kind == 'element' else
                           JointAction('replace', action.orbit, letters[action.orbit], elements[action.orbit]) if action.kind == 'replace' else action)
                next_letters, next_elements = edit_signature(letters, elements, action)
                n = sum(endpoint.spec.template(w).multiplicity for w in next_letters)
                proposed = replace(visible(), letters=next_letters, elements=next_elements,
                                   coordinates=endpoint.coordinates.new_zeros((n, 3)), step=len(program)+1)
                if inverse not in legal_actions(proposed, allowed, max_atoms, max_orbits):
                    continue
                role = temporary(action.letter) if action.kind in ('birth', 'replace') else None
                append(action, role)
                append(inverse, old_role)
                break
        letter, z = endpoint.letters[j], endpoint.elements[j]
        parents = [i for i, w in enumerate(letters) if w == letter and endpoint.spec.template(w).dof]
        action = (JointAction('split', orbit=int(rng.choice(parents)), letter=letter, element=z)
                  if parents and rng.random() < .5 else JointAction('birth', letter=letter, element=z))
        append(action, int(j))
    append(JointAction('end'))
    return program, anchors
