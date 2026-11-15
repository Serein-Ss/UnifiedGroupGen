"""Decode existing records into verified target states without rewriting data."""
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
