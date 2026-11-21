"""Separate operation satisfaction from maximal-group recovery; isolate native calls."""
import json
import os
import subprocess
import sys
import numpy as np


def _cell(lattice, coordinates, numbers, kind):
    lattice = np.asarray(lattice, dtype=np.float64).copy()
    coordinates = np.asarray(coordinates, dtype=np.float64).copy()
    if not np.isfinite(lattice).all() or not np.isfinite(coordinates).all() or len(coordinates) == 0:
        raise ValueError('Only finite nonempty structures can be identified')
    if kind in ('layer', 'plane'):
        height = max(20., float(np.ptp(coordinates[:, 2]))+10.)
        lattice[2, 2] = height
        coordinates[:, 2] = (coordinates[:, 2]-coordinates[:, 2].mean())/height+.5
    return lattice.T, coordinates, np.asarray(numbers, dtype=np.int32)


def _plane_to_flat_layer():
    """Classify plane groups through the flat-layer product with normal reflection."""
    import spglib
    import torch
    from unifiedgroupgen.group_constraints.compiler import Descriptor, OrbitSpec, compile_descriptor, group_table
    mapping = {}
    generator = torch.Generator().manual_seed(9318)
    for number in range(1, 18):
        general = group_table('plane', number)[0].letter
        compiled = compile_descriptor(Descriptor('plane', number, tuple(OrbitSpec(general, z) for z in (6, 8, 14))))
        value = compiled.expand(compiled.prior(dtype=torch.float64, generator=generator))
        cell = _cell(value['lattice'].numpy(), value['coordinates'].numpy(), compiled.elements, 'plane')
        data = spglib.get_layergroup(cell, aperiodic_dir=2, symprec=1e-5)
        if data is None or int(data.number) in mapping:
            raise ValueError('Plane-to-flat-layer classification was not unique')
        mapping[int(data.number)] = number
    return mapping


def identify_maximal_group(decoded, spec, tolerance=1e-3, timeout=40):
    if spec.kind not in ('space', 'layer', 'plane', 'magnetic'):
        return dict(number=None, exact_target_group=None, error='No maximal-group identifier for this explicit group family')
    request = dict(kind=spec.kind, number=spec.number, tolerance=tolerance,
                   lattice=decoded['lattice'].detach().cpu().tolist(),
                   coordinates=decoded['coordinates'].detach().cpu().tolist(),
                   atomic_numbers=decoded['atomic_numbers'].detach().cpu().tolist(),
                   moments=None if decoded['moments'] is None else decoded['moments'].detach().cpu().tolist())
    from unifiedgroupgen.common.configuration import runtime_pythonpath
    env = {**os.environ, 'PYTHONPATH': runtime_pythonpath()}
    try:
        result = subprocess.run([sys.executable, '-m', 'unifiedgroupgen.group_constraints.identification'],
                                input=json.dumps(request), text=True, capture_output=True, timeout=timeout, env=env,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
        if result.returncode:
            raise ValueError(f'Native identifier exited {result.returncode}: {result.stderr[-500:]}')
        return json.loads(result.stdout)
    except (ValueError, subprocess.TimeoutExpired) as error:
        return dict(number=None, exact_target_group=None, error=str(error))


def validate_joint_state(state, exact=False, tolerance=1e-3, minimum_distance=.6):
    import torch
    decoded = state.decoded()
    residual = state.compiled.residuals(decoded['coordinates'], decoded['atomic_numbers'], decoded['lattice'], decoded['moments'])
    metric = decoded['lattice'].T@decoded['lattice']
    metric_error = metric.new_zeros(())
    for op in state.spec.operations:
        r = metric.new_tensor(op.rotation)
        metric_error = torch.maximum(metric_error, (r.T@metric@r-metric).abs().max())
    valid = bool(residual['elements'] and residual['coordinates'] < tolerance and metric_error < tolerance)
    if (state.spec.magnetic or state.spec.spin_generators or any(op.spin is not None for op in state.spec.operations)) and state.moments is None:
        valid = False
    if 'moments' in residual:
        valid = valid and bool(residual['moments'] < tolerance)
    closest, regular = None, True
    if state.num_atoms and any(state.spec.periodic):
        from ase.geometry import minkowski_reduce
        cell = decoded['lattice'].detach().cpu().double().numpy().T
        reduced, _ = minkowski_reduce(cell, pbc=state.spec.periodic)
        closest = float(np.linalg.norm(np.asarray(reduced)[np.asarray(state.spec.periodic)], axis=1).min())
    if state.num_atoms > 1:
        from ase.geometry import find_mic
        x = decoded['coordinates'].detach().cpu().double().numpy()
        lattice = decoded['lattice'].detach().cpu().double().numpy()
        difference = (x[None]-x[:, None])@lattice.T
        ids = np.triu_indices(state.num_atoms, k=1)
        _, distances = find_mic(difference[ids], lattice.T, pbc=state.spec.periodic)
        pair_distance = float(distances.min())
        closest = pair_distance if closest is None else min(closest, pair_distance)
        orbit = decoded['orbit_index'].detach().cpu().numpy()
        within = orbit[ids[0]] == orbit[ids[1]]
        regular = bool(np.all(distances[within] > 1e-5))
    report = dict(satisfies_requested_operations=valid, metric_residual=float(metric_error),
                  coordinate_residual=float(residual['coordinates']), elements_compatible=bool(residual['elements']),
                  minimum_distance_angstrom=closest, distance_valid=closest is None or closest >= minimum_distance,
                  orbit_multiplicity_regular=regular, no_coincident_orbits=closest is None or closest > 1e-5,
                  maximal_group_checked=exact, exact_target_group=None,
                  magnetic_validation_status='not_tested' if state.moments is not None else 'not_applicable',
                  stability_validated=False)
    if exact:
        report.update(identify_maximal_group(decoded, state.spec, tolerance))
        if state.spec.kind == 'rod':
            report['maximal_group_checked'] = False
    return report


def main():
    import spglib
    request = json.load(sys.stdin)
    try:
        kind = request['kind']
        cell = _cell(request['lattice'], request['coordinates'], request['atomic_numbers'], kind)
        if kind == 'space':
            data = spglib.get_symmetry_dataset(cell, symprec=request['tolerance'])
            number = None if data is None else int(data.number)
        elif kind in ('layer', 'plane'):
            data = spglib.get_layergroup(cell, aperiodic_dir=2, symprec=request['tolerance'])
            number = None if data is None else int(data.number)
            if kind == 'plane' and number is not None:
                number = _plane_to_flat_layer().get(number)
        elif kind == 'magnetic' and request['moments'] is not None:
            data = spglib.get_magnetic_symmetry_dataset((*cell, request['moments']), symprec=request['tolerance'], is_axial=True)
            number = None if data is None else int(data.uni_number)
        else:
            raise ValueError('Magnetic identification requires aligned moment values')
        output = dict(number=number, exact_target_group=None if number is None else number == request['number'],
                      error=None if number is not None else 'No maximal group recovered')
    except (ValueError, RuntimeError, AttributeError) as error:
        output = dict(number=None, exact_target_group=None, error=str(error))
    print(json.dumps(output), flush=True)


if __name__ == '__main__':
    main()
