"""Observable joint state and fixed, fully specified symmetry-preserving edit kernels."""
from dataclasses import dataclass, replace
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl, can_complete, edit_signature
from functools import lru_cache
import torch
from unifiedgroupgen.group_constraints.unified import compile_constraints


@dataclass
class JointState:
    spec: object
    letters: tuple
    elements: tuple
    metric: torch.Tensor
    coordinates: torch.Tensor
    moments: torch.Tensor | None = None
    step: int = 0
    budget: int = 128
    status: str = 'active'
    chemistry: ChemicalControl = ChemicalControl()

    def __post_init__(self):
        c = self.compiled
        self.chemistry.validate(self.spec, self.letters, self.elements, self.status == 'done')
        if len(self.letters) != len(self.elements) or self.coordinates.shape != (c.num_atoms, 3):
            raise ValueError('Joint state orbit labels, elements and coordinates disagree')
        if self.metric.shape != (c.metric_dimension,):
            raise ValueError('Joint state metric has wrong dimension')
        if any(not 0 <= z <= 118 for z in self.elements):
            raise ValueError('Elements must be atomic numbers or MASK=0')
        if any(self.letters.count(w.letter) > 1 for w in c.templates if w.dof == 0):
            raise ValueError('A zero-dimensional orbit cannot be occupied twice')
        if self.moments is not None and self.moments.shape != self.coordinates.shape:
            raise ValueError('Moment ordering/shape differs from the atom graph')
        if self.status not in ('active', 'done', 'failed') or not 0 <= self.step <= self.budget:
            raise ValueError('Invalid edit status or event budget')

    @property
    def compiled(self):
        return compile_constraints(self.spec, self.letters)

    @property
    def num_atoms(self):
        return self.compiled.num_atoms

    def to(self, device=None, dtype=None):
        return replace(self, metric=self.metric.to(device=device, dtype=dtype),
                       coordinates=self.coordinates.to(device=device, dtype=dtype),
                       moments=None if self.moments is None else self.moments.to(device=device, dtype=dtype))

    def decoded(self, wrap=True):
        c = self.compiled
        lattice = c.lattice(self.metric)
        coords = c.project_coordinates(self.coordinates, wrap=wrap)
        numbers = torch.tensor(self.elements, device=self.metric.device, dtype=torch.long)
        orbit = torch.tensor(c.atom_orbit, device=self.metric.device, dtype=torch.long)
        return dict(coordinates=coords, lattice=lattice, atomic_numbers=numbers[orbit], orbit_index=orbit,
                    moments=c.project_moments(self.moments, lattice), periodic=self.spec.periodic)


@dataclass(frozen=True)
class JointAction:
    kind: str
    orbit: int = -1
    letter: str = ''
    element: int = 0
    partner: int = -1

    def to_dict(self):
        return dict(kind=self.kind, orbit=self.orbit, letter=self.letter,
                    element=self.element, partner=self.partner)


def initial_state(spec, device='cpu', dtype=torch.float32, generator=None, budget=128, magnetic=False, chemistry=None):
    c = compile_constraints(spec, ())
    return JointState(spec, (), (), torch.randn(c.metric_dimension, device=device, dtype=dtype, generator=generator),
                      torch.zeros((0, 3), device=device, dtype=dtype),
                      torch.zeros((0, 3), device=device, dtype=dtype) if magnetic else None, budget=budget, chemistry=chemistry or ChemicalControl())


@lru_cache(maxsize=64)
def _actions(spec, letters, elements, allowed_elements, max_atoms, max_orbits, last_event):
    c = compile_constraints(spec, letters)
    actions = []
    if c.num_atoms and all(elements):
        actions.append(JointAction('end'))
    if last_event:
        return (JointAction('fail'),)
    for w in spec.templates:
        if (len(letters) < max_orbits and c.num_atoms+w.multiplicity <= max_atoms
                and (w.dof or w.letter not in letters)):
            actions.append(JointAction('birth', letter=w.letter))
    for j, (letter, element) in enumerate(zip(letters, elements)):
        actions.append(JointAction('delete', orbit=j))
        actions.extend(JointAction('element', orbit=j, element=z) for z in allowed_elements if z != element)
        old = spec.template(letter)
        if old.dof and len(letters) < max_orbits and c.num_atoms+old.multiplicity <= max_atoms:
            actions.append(JointAction('split', orbit=j, letter=letter))
        for w in spec.templates:
            if w.letter == letter or c.num_atoms-old.multiplicity+w.multiplicity > max_atoms:
                continue
            if not w.dof and w.letter in letters[:j]+letters[j+1:]:
                continue
            # Equal-size replacements keep the raw state; transdimensional
            # replacements use the fixed birth kernel, never a target location.
            for z in (0, *allowed_elements):
                actions.append(JointAction('replace', j, w.letter, z))
        for other in range(j+1, len(letters)):
            if element != elements[other]:
                actions.append(JointAction('swap', orbit=j, partner=other))
    return tuple(actions) or (JointAction('fail'),)


def legal_actions(state, allowed_elements, max_atoms=192, max_orbits=52):
    if state.status != 'active' or state.step >= state.budget:
        return ()
    allowed = tuple(sorted(set(int(z) for z in allowed_elements)))
    if not allowed or any(not 1 <= z <= 118 for z in allowed):
        raise ValueError('The allowed element domain must be nonempty and exclude MASK')
    allowed = state.chemistry.domain(allowed)
    if state.chemistry.counts:
        allowed = tuple(z for z in allowed if z in dict(state.chemistry.counts))
        return _inventory_actions(state.spec, state.letters, state.elements, state.chemistry,
                                  allowed, max_atoms, max_orbits, state.budget-state.step)
    return _actions(state.spec, state.letters, state.elements, allowed, max_atoms, max_orbits,
                    state.step == state.budget-1)


@lru_cache(maxsize=256)
def _inventory_actions(spec, letters, elements, control, allowed, max_atoms, max_orbits, remaining_events):
    if remaining_events <= 1:
        return (JointAction('fail'),)
    options = _actions(spec, letters, elements, allowed, max_atoms, max_orbits, False)
    legal = []
    for action in options:
        if action.kind == 'end':
            if not any(n for _, n in control.remaining(spec, letters, elements)):
                legal.append(action)
            continue
        candidates = (tuple(replace(action, element=z) for z in allowed)
                      if action.kind in ('birth', 'split') else (action,))
        for candidate in candidates:
            if candidate.kind in ('replace', 'element') and candidate.element == 0:
                continue
            new_letters, new_elements = edit_signature(letters, elements, candidate)
            if can_complete(spec, new_letters, new_elements, control, remaining_events-3,
                            max_orbits, max_atoms):
                legal.append(candidate)
    return tuple(legal) or (JointAction('fail'),)



def apply_action(state, action, generator=None, birth_scale=.2, check=None):
    """K_a = pushforward of a fixed Gaussian kernel; survivors keep their raw state."""
    if state.status != 'active' or state.step >= state.budget:
        raise ValueError('Cannot edit a stopped state')
    if state.step == state.budget-1 and action.kind != 'fail':
        raise ValueError('The last event slot is reserved for failure, not forced completion')
    if check is not None and action not in check:
        raise ValueError('Action is outside the shared training/inference support')
    step = state.step+1
    if action.kind in ('end', 'fail'):
        if action.kind == 'end' and (not state.num_atoms or not all(state.elements)):
            raise ValueError('END requires a nonempty, fully assigned structure')
        return replace(state, step=step, status='done' if action.kind == 'end' else 'failed')
    letters, elements = list(state.letters), list(state.elements)
    coords = list(state.coordinates.split([w.multiplicity for w in state.compiled.templates]))
    moments = None if state.moments is None else list(state.moments.split([w.multiplicity for w in state.compiled.templates]))

    def born(letter, parent=None):
        w = state.spec.template(letter)
        _, b = w.arrays()
        x = state.coordinates.new_tensor(b) if parent is None else coords[parent]
        x = x+birth_scale*torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        m = None if moments is None else torch.randn(x.shape, dtype=x.dtype, device=x.device, generator=generator)
        if m is not None and parent is not None:
            m = m+moments[parent]
        return x, m

    if action.kind in ('birth', 'split'):
        x, m = born(action.letter, action.orbit if action.kind == 'split' else None)
        letters.append(action.letter); elements.append(action.element); coords.append(x)
        if moments is not None:
            moments.append(m)
    elif action.kind == 'delete':
        for values in (letters, elements, coords):
            values.pop(action.orbit)
        if moments is not None:
            moments.pop(action.orbit)
    elif action.kind == 'element':
        elements[action.orbit] = action.element
    elif action.kind == 'swap':
        i, j = action.orbit, action.partner
        elements[i], elements[j] = elements[j], elements[i]
    elif action.kind == 'replace':
        if state.spec.template(action.letter).multiplicity == len(coords[action.orbit]):
            x = coords[action.orbit]
            m = None if moments is None else moments[action.orbit]
        else:
            x, m = born(action.letter)
        letters[action.orbit], elements[action.orbit], coords[action.orbit] = action.letter, action.element, x
        if moments is not None:
            moments[action.orbit] = m
    else:
        raise ValueError(f'Unknown joint action {action.kind}')
    return replace(state, letters=tuple(letters), elements=tuple(elements), step=step,
                   coordinates=torch.cat(coords) if coords else state.coordinates.new_zeros((0, 3)),
                   moments=None if moments is None else (torch.cat(moments) if moments else state.moments.new_zeros((0, 3))),
                   status='failed' if step == state.budget else 'active')
