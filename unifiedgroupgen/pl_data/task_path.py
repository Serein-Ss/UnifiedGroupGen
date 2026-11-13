"""Independent task visibility and inventory-constrained teacher programs."""
from dataclasses import replace
import math
import numpy as np
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl, orbit_counts, edit_signature

TASKS = ('dng', 'property', 'csp')
RESERVED = {'source_domain', 'space_group', 'layer_group', 'plane_group', 'group',
            'symmetry', 'chemical_system', 'composition', 'formula', 'kind', 'dimension'}


def validate_task_config(config):
    task = config.get('generation_task')
    if task is None:
        return
    if task not in TASKS or not config.get('model', {}).get('task_controls'):
        raise ValueError('Select one independent task and enable model.task_controls')
    properties = config.get('properties', {})
    if (task == 'property') != bool(properties):
        raise ValueError('Only the property task requires a nonempty target-property schema')
    if config.get('dataset_domains') or RESERVED & set(properties):
        raise ValueError('Group, composition and source labels are not target properties')
    if any(spec.get('type', 'scalar') != 'scalar' or spec.get('context')
           or spec.get('domain_scalers') or spec.get('per_domain') for spec in properties.values()):
        raise ValueError('Independent property generation accepts scalar target properties only')
    probability = config.get('chemical_set_probability', .5)
    if not 0 <= probability <= 1:
        raise ValueError('chemical_set_probability must be in [0, 1]')


def has_property_target(record, names):
    return any(record.get('properties', {}).get(name) is not None
               and math.isfinite(float(record['properties'][name])) for name in names)


def task_target(target, config, seed):
    task = config.get('generation_task')
    if task is None:
        return target
    validate_task_config(config)
    rng = np.random.default_rng(seed)
    vocabulary = tuple(config['allowed_elements'])
    properties = {}
    if task == 'property':
        properties = {name: target.properties[name] for name in config['properties']
                      if target.properties.get(name) is not None and math.isfinite(float(target.properties[name]))}
        if not properties:
            raise ValueError('Property-task record has no observed target label')
    if task == 'csp':
        control = ChemicalControl.from_inputs(counts=orbit_counts(target.state.spec, target.state.letters,
                                                                 target.state.elements))
    elif rng.random() < config.get('chemical_set_probability', .5):
        present = set(target.state.elements)
        extra = sorted(set(vocabulary)-present)
        count = int(rng.integers(0, min(4, len(extra))+1))
        added = rng.choice(extra, size=count, replace=False).tolist() if count else []
        control = ChemicalControl.from_inputs(allowed=sorted(present | set(added)))
    else:
        control = ChemicalControl()
    control.domain(vocabulary)
    return replace(target, state=replace(target.state, chemistry=control), properties=properties)


def inventory_program(target, rng, allowed, max_atoms, max_orbits, budget, augment):
    from unifiedgroupgen.pl_data.joint_path import PathStep
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
