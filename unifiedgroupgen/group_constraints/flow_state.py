"""Hard-constrained public state and shared, target-independent orbit kernels."""
from dataclasses import dataclass, replace
import torch
from unifiedgroupgen.group_constraints.state import JointState, JointAction, initial_state, legal_actions as inventory_actions
from unifiedgroupgen.group_constraints.chemistry import can_complete

@dataclass
class FlowState(JointState):
    frozen: bool = False
    frame: torch.Tensor | None = None
    origin: torch.Tensor | None = None

    def to(self, device=None, dtype=None):
        value = super().to(device, dtype)
        return replace(value, frame=None if self.frame is None else self.frame.to(device=device, dtype=dtype),
                       origin=None if self.origin is None else self.origin.to(device=device,dtype=dtype))

    def decoded(self, wrap=True):
        with torch.autocast(device_type=self.metric.device.type,enabled=False):
            return self._decoded(wrap)

    def _decoded(self, wrap=True):
        value = super().decoded(wrap)
        if self.frame is not None:
            value['lattice'] = self.frame @ value['lattice']
        value['cartesian_coordinates']=value['coordinates']@value['lattice'].T
        if self.origin is not None:value['cartesian_coordinates']=value['cartesian_coordinates']+self.origin
        return value

def from_state(state, frozen=False, frame=None):
    return FlowState(**vars(state), frozen=frozen, frame=frame)

def metric_tangent(state, value):
    with torch.autocast(device_type=state.metric.device.type,enabled=False):
        return _metric_tangent(state,value.to(state.metric.dtype))


def _metric_tangent(state, value):
    c = state.compiled
    basis = value.new_tensor(c.metric_basis)
    standard = value.new_tensor(c.standard)
    matrix = torch.einsum('k,kij->ij', value, standard)
    matrix = torch.einsum('a,aij->ij', torch.einsum('aij,ij->a', basis, matrix), basis)
    return torch.einsum('kij,ij->k', standard, matrix)

def coordinate_tangent(state, value, cartesian=False, *, reference=None):
    """Project a velocity, optionally relative to a fractional reference velocity."""
    with torch.autocast(device_type=state.metric.device.type,enabled=False):
        return _coordinate_tangent(state,value.to(state.metric.dtype),cartesian,reference)


def _coordinate_tangent(state, value, cartesian, reference):
    chart = state.compiled
    if not chart.slices:
        return value
    lattice = chart.lattice(state.metric)
    frame = getattr(state, 'frame', None)
    if frame is not None:
        lattice = frame @ lattice
    blocks = [None] * len(chart.slices)
    systems = {}
    for i, (section, (basis, _, _)) in enumerate(zip(chart.slices, chart.blocks)):
        dof = basis.shape[-1]
        if not dof:
            blocks[i] = value[section]*0
            continue
        a = value.new_tensor(basis)
        j = torch.einsum('ab,nbq->naq', lattice, a).reshape(-1, dof)
        v = value[section] if cartesian else value[section] @ lattice.T
        if reference is not None:
            v = v - reference[section] @ lattice.T
        systems.setdefault(dof, []).append((i, a, j.T @ j, j.T @ v.reshape(-1)))
    # Orbit multiplicities may differ, but their Gram systems share the free dimension.
    for group in systems.values():
        indices, bases, matrices, rhs = zip(*group)
        q = torch.linalg.solve(torch.stack(matrices), torch.stack(rhs).unsqueeze(-1)).squeeze(-1)
        for i, a, free in zip(indices, bases, q):
            blocks[i] = torch.einsum('nbq,q->nb', a, free)
    return torch.cat(blocks)

def initial(spec, generator=None, budget=96, chemistry=None, device='cpu', dtype=torch.float32, frame=None):
    s = initial_state(spec, device, dtype, generator, budget, False, chemistry)
    if spec.magnetic or spec.spin_generators:
        raise ValueError('joint_flow_v2 is a nonmagnetic crystal model')
    return from_state(replace(s, metric=metric_tangent(s, s.metric)), frame=frame)

def actions(state, allowed, max_atoms=192, max_orbits=52):
    if state.frozen or state.status != 'active':
        return ()
    # Existing exact-inventory task uses typed whole-orbit edits, not independent MASK draws.
    if state.chemistry.counts:
        support = inventory_actions(state, allowed, max_atoms, max_orbits)
        return tuple(replace(a, kind='freeze') if a.kind == 'end' else a for a in support)
    if state.step >= state.budget-1:
        return (JointAction('freeze'),) if state.num_atoms else (JointAction('fail'),)
    support = []
    if state.num_atoms:
        support.append(JointAction('freeze'))
    for w in state.spec.templates:
        if (len(state.letters) < max_orbits and state.num_atoms+w.multiplicity <= max_atoms
                and (w.dof or w.letter not in state.letters)):
            support.append(JointAction('birth', letter=w.letter))
    for j, letter in enumerate(state.letters):
        old = state.spec.template(letter)
        if len(state.letters) > 1:
            support.append(JointAction('delete', orbit=j))
        if old.dof and len(state.letters) < max_orbits and state.num_atoms+old.multiplicity <= max_atoms:
            support.append(JointAction('split', orbit=j, letter=letter))
        for w in state.spec.templates:
            if (w.letter != letter and state.num_atoms-old.multiplicity+w.multiplicity <= max_atoms
                    and (w.dof or w.letter not in state.letters)):
                support.append(JointAction('replace', j, w.letter))
    return tuple(support) or (JointAction('fail'),)

def apply(state, action, generator=None, birth_scale=.2):
    if state.frozen or state.status != 'active':
        raise ValueError('Cannot edit a frozen/failed state')
    if action.kind == 'freeze':
        if not state.num_atoms or (state.chemistry.counts and any(n for _,n in state.chemistry.remaining(state.spec,state.letters,state.elements))):
            raise ValueError('Cannot freeze an empty or incomplete-inventory configuration')
        return replace(state, frozen=True, step=state.step+1)
    if action.kind == 'fail':
        return replace(state, status='failed', step=min(state.budget,state.step+1))
    letters, elements = list(state.letters), list(state.elements)
    coords = list(state.coordinates.split([w.multiplicity for w in state.compiled.templates]))
    if action.kind in ('birth','split','replace'):
        w = state.spec.template(action.letter)
        a,b = w.arrays()
        a,b = state.metric.new_tensor(a),state.metric.new_tensor(b)
        # q Gaussian is a fixed chart prior; its covariance is independent of Cartesian frame.
        q = torch.randn(w.dof, device=b.device, dtype=b.dtype, generator=generator)*birth_scale
        if action.kind == 'split':
            _,_,inverse = state.compiled.blocks[action.orbit]
            parent = coords[action.orbit]
            q = q+state.metric.new_tensor(inverse)@(parent-b).reshape(-1)
        # Replacement is explicitly death+birth; equal multiplicity is not a correspondence.
        x = b+torch.einsum('nbq,q->nb',a,q)
        if action.kind == 'replace':
            letters[action.orbit],elements[action.orbit],coords[action.orbit]=action.letter,action.element,x
        else:
            letters.append(action.letter);elements.append(action.element);coords.append(x)
    elif action.kind == 'delete':
        for values in (letters,elements,coords): values.pop(action.orbit)
    elif action.kind == 'element':
        elements[action.orbit]=action.element
    elif action.kind == 'swap':
        elements[action.orbit],elements[action.partner]=elements[action.partner],elements[action.orbit]
    else:
        raise ValueError('Unknown orbit action')
    return replace(state,letters=tuple(letters),elements=tuple(elements),
                   coordinates=torch.cat(coords) if coords else state.coordinates.new_zeros((0,3)),
                   step=state.step+1)

def assign(state, orbit, element):
    if state.elements[orbit] or not element:
        raise ValueError('DFM only unmasks an existing MASK orbit')
    values=list(state.elements);values[orbit]=int(element)
    return replace(state,elements=tuple(values))

def advance(state, metric, coordinate, dt):
    return replace(state,metric=state.metric+dt*metric_tangent(state,metric),
                   coordinates=state.coordinates+dt*coordinate_tangent(state,coordinate))
