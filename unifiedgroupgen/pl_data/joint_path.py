"""Legacy-record adapter and exactly simulated mixed, variable-dimensional teacher paths."""
from dataclasses import dataclass, replace
import numpy as np
import torch
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
from unifiedgroupgen.group_constraints.unified import catalog_spec, SymmetrySpec
from unifiedgroupgen.group_constraints.state import JointState, JointAction, initial_state, apply_action


@dataclass
class JointTarget:
    state: JointState
    identity: str
    properties: dict


def target_from_record(record, dtype=torch.float32, operation_compiler=False):
    target = _target_from_record(record, dtype)
    if operation_compiler:
        from unifiedgroupgen.pl_data.operation_path import recompile_target
        return recompile_target(target)
    return target


def _target_from_record(record, dtype=torch.float32):
    """Decode verified existing records without rewriting or re-standardizing the data."""
    if 'public_state' in record:
        spec = SymmetrySpec.from_dict(record['symmetry'])
        value = record['public_state']
        moments = record.get('magnetic_moments')
        if moments is not None and record.get('moment_order') != 'compiled':
            raise ValueError('Moment labels require moment_order=compiled')
        state = JointState(spec, tuple(value['letters']), tuple(value['elements']),
                           torch.tensor(value['metric'], dtype=dtype),
                           torch.tensor(value['coordinates'], dtype=dtype).reshape(-1, 3),
                           None if moments is None else torch.tensor(moments, dtype=dtype), status='done')
        if not state.num_atoms or not all(state.elements):
            raise ValueError('Training endpoints must be nonempty and contain no MASK elements')
        if not all(torch.isfinite(v).all() for v in (state.metric, state.coordinates)):
            raise ValueError('Nonfinite explicit target')
        if moments is not None:
            if not torch.isfinite(state.moments).all() or not torch.allclose(state.decoded()['moments'], state.moments, atol=2e-5, rtol=2e-5):
                raise ValueError('Moment labels violate the supplied representation')
        return JointTarget(state, str(record.get('id', 'structure')), dict(record.get('properties', {})))
    descriptor = Descriptor.from_dict(record['descriptor'])
    old = compile_descriptor(descriptor)
    endpoint = torch.tensor(record['state'], dtype=dtype)
    if endpoint.shape != (old.dof,) or not torch.isfinite(endpoint).all():
        raise ValueError(f'Invalid endpoint: {record.get("id")}')
    spec = SymmetrySpec.from_dict(record['symmetry']) if 'symmetry' in record else catalog_spec(descriptor.kind, descriptor.number)
    letters = tuple(w.letter for w in descriptor.orbits)
    elements = tuple(w.atomic_number for w in descriptor.orbits)
    from unifiedgroupgen.group_constraints.unified import compile_constraints
    new = compile_constraints(spec, letters)
    symmetric = torch.einsum('k,kij->ij', endpoint[:old.lattice_dof], torch.tensor(old.metric_basis, dtype=dtype))
    metric = torch.einsum('kij,ij->k', torch.tensor(new.standard, dtype=dtype), symmetric)
    coordinates = old.expand(endpoint, wrap=False)['coordinates']
    moments = record.get('magnetic_moments')
    if moments is not None:
        if 'symmetry' not in record or record.get('moment_order') != 'compiled':
            raise ValueError('Moment supervision needs explicit symmetry and moment_order=compiled')
        moments = torch.tensor(moments, dtype=dtype)
        if moments.shape != coordinates.shape or not torch.isfinite(moments).all():
            raise ValueError('Invalid or unaligned moment labels')
    state = JointState(spec, letters, elements, metric, coordinates, moments, status='done')
    decoded = state.decoded(wrap=False)
    if not torch.allclose(decoded['coordinates'], coordinates, atol=2e-5, rtol=2e-5):
        raise ValueError('Record does not lie on the declared coordinate chart')
    if moments is not None and not torch.allclose(decoded['moments'], moments, atol=2e-5, rtol=2e-5):
        raise ValueError('Moment labels violate their declared symmetry')
    return JointTarget(state, str(record.get('id', 'structure')), dict(record.get('properties', {})))


@dataclass(frozen=True)
class PathStep:
    action: JointAction
    roles: tuple


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


def _program(target, rng, allowed, max_atoms, max_orbits, budget, augment):
    state = target.state
    if state.num_atoms > max_atoms or len(state.letters) > max_orbits:
        raise ValueError('Target exceeds the configured atom/orbit limits')
    if set(state.elements)-set(allowed):
        raise ValueError('Target element is outside the TRAIN-only element domain')
    if state.chemistry.counts:
        from unifiedgroupgen.pl_data.task_path import inventory_program
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
