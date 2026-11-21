"""Exact-inventory teacher program shared by current CSP training paths."""
from dataclasses import dataclass, replace
import torch
from unifiedgroupgen.group_constraints.state import JointState, JointAction, legal_actions, inventory_teacher_allows
from unifiedgroupgen.group_constraints.chemistry import edit_signature
from unifiedgroupgen.pl_data.orbit_order import endpoint_orbit_order

@dataclass(frozen=True)
class PathStep:
    action: JointAction
    roles: tuple

def inventory_program(target, rng, allowed, max_atoms, max_orbits, budget, augment, orbit_order='random'):
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
        valid = (action in support() if augment else inventory_teacher_allows(
            endpoint.spec, letters, elements, endpoint.chemistry, allowed, max_atoms, max_orbits,
            budget-len(program), action))
        if not valid:
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

    order = endpoint_orbit_order(endpoint, rng, orbit_order)
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
