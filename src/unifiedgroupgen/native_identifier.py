"""Isolate native symmetry identification so a spglib abort cannot kill sampling."""

import json
from pathlib import Path
import subprocess
import sys


def identify_isolated(structure, kind, timeout=20):
    request = {'kind': kind, 'cell': [structure.lattice.matrix.tolist(),
                                     structure.frac_coords.tolist(), list(structure.atomic_numbers)]}
    try:
        result = subprocess.run([sys.executable, str(Path(__file__).resolve())],
                                input=json.dumps(request), capture_output=True, text=True, timeout=timeout,
                                creationflags=subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0)
    except subprocess.TimeoutExpired:
        return {'number': None, 'error': f'Native symmetry identification timed out after {timeout}s'}
    if result.returncode:
        return {'number': None, 'error': f'Native symmetry worker exited with {result.returncode}: {result.stderr[-1000:]}'}
    try:
        output = json.loads(result.stdout)
    except ValueError:
        return {'number': None, 'error': 'Native symmetry worker returned no valid result'}
    if output.get('number') is not None and not isinstance(output['number'], int):
        return {'number': None, 'error': 'Native symmetry worker returned an invalid group number'}
    return output


def main():
    import spglib
    request = json.load(sys.stdin)
    try:
        if request['kind'] == 'space':
            dataset = spglib.get_symmetry_dataset(tuple(request['cell']), symprec=1e-3)
        elif request['kind'] == 'layer':
            dataset = spglib.get_layergroup(tuple(request['cell']), aperiodic_dir=2, symprec=1e-3)
        else:
            raise ValueError('Native identification supports space and layer groups')
        result = {'number': int(dataset.number) if dataset is not None else None,
                  'error': None if dataset is not None else 'spglib returned no group'}
    except (ValueError, RuntimeError, AttributeError) as error:
        result = {'number': None, 'error': str(error)}
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
