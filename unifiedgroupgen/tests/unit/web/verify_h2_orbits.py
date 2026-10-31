"""Verify and export the H.2 examples from the Site's existing material fixture."""

import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[4]/'web/symmetry-lab'


def main():
    source = (ROOT / 'dist/data.js').read_text(encoding='utf-8')
    data = json.loads(source[source.index('{'):].rstrip(';\r\n'))
    specs = {
        'tio2': [
            ('Ti 2a', 0, [1, 3], -2*np.eye(3), [0, 0, 0], [0, 0, 0], np.zeros((3, 0))),
            ('O 4f', 2, [1, 2, 3, 4], [[-1, 1, 0], [0, 0, -2]], [0, 0], [0, 0, 0], [[1], [1], [0]]),
        ],
        'mos2': [
            ('Mo 1a', 0, [1], [[-1, -1, 0], [1, -2, 0], [0, 0, -2]], [0, 0, 0], [0, 0, 0], np.zeros((3, 0))),
            ('S 2f', 1, [1, 4], [[-1, -1, 0], [1, -2, 0]], [-1, 0], [2/3, 1/3, 0], [[0], [0], [1]]),
        ],
    }
    result = {'passed': True, 'operation_indices': '1-based Site operation order', 'materials': []}
    maxima = []
    for m in data['materials']:
        xyz = np.asarray(m['coordinates'])
        periodic = np.asarray(m['periodic'])
        operations = m['operations']
        rows = []
        all_b, all_B = [], []
        for name, representative, representatives, reduced, rhs, particular, basis in specs[m['key']]:
            f = xyz[representative]
            reduced, rhs, particular, basis = [np.asarray(x, dtype=float) for x in (reduced, rhs, particular, basis)]
            stabilizer, lifts, blocks, constants = [], [], [], []
            for j, op in enumerate(operations):
                r, t = np.asarray(op['R']), np.asarray(op['t'])
                delta = r@f+t-f
                lift = np.where(periodic, np.rint(delta), 0)
                if np.max(np.abs(delta-lift)) < 1e-8:
                    stabilizer.append(j+1)
                    lifts.append(lift[:m['p']].astype(int).tolist())
                    blocks.append(r-np.eye(3))
                    constants.append(-t+lift)
            a, c = np.vstack(blocks), np.concatenate(constants)
            rank = int(np.linalg.matrix_rank(a))
            assert rank == np.linalg.matrix_rank(reduced)
            assert basis.shape == (3, 3-rank)
            residuals = [np.max(np.abs(a@particular-c)), np.max(np.abs(reduced@particular-rhs))]
            if basis.size:
                residuals += [np.max(np.abs(a@basis)), np.max(np.abs(reduced@basis))]
                assert np.linalg.matrix_rank(basis) == 3-rank
            orbit_b, orbit_B, cosets = [], [], []
            q = np.asarray(m['state'][m['k_dof']:]) if basis.size else np.zeros(0)
            for index in representatives:
                op = operations[index-1]
                r, t = np.asarray(op['R']), np.asarray(op['t'])
                b, B = r@particular+t, r@basis
                position = b+B@q
                members = []
                for j, other in enumerate(operations):
                    delta = np.asarray(other['R'])@f+other['t']-position
                    delta[periodic] -= np.rint(delta[periodic])
                    if np.max(np.abs(delta)) < 1e-8:
                        members.append(j+1)
                assert len(members) == len(stabilizer)
                cosets.append(members)
                orbit_b.append(b.tolist())
                orbit_B.append(B.tolist())
            assert sorted(j for coset in cosets for j in coset) == list(range(1, len(operations)+1))
            all_b.extend(orbit_b)
            all_B.extend([b if basis.size else [[0.], [0.], [0.]] for b in orbit_B])
            maxima.extend(float(x) for x in residuals)
            rows.append({'name': name, 'stabilizer_indices': stabilizer, 'lifts': lifts,
                         'A_all_shape': list(a.shape), 'A_all': a.tolist(), 'c_all': c.tolist(),
                         'A_reduced': reduced.tolist(), 'c_reduced': rhs.tolist(), 'rank': rank,
                         'particular': particular.tolist(), 'basis_shape': list(basis.shape),
                         'basis': basis.tolist(), 'representatives': representatives, 'cosets': cosets,
                         'expanded_b': orbit_b, 'expanded_B': orbit_B})
        b, B = np.asarray(all_b), np.asarray(all_B)
        assert np.allclose(b, m['b']) and np.allclose(B, m['B'])
        flat_B = B.reshape(-1, 1)
        n = len(xyz)
        constraints = []
        for op in operations:
            for i, target in enumerate(op['permutation']):
                block = np.zeros((3, 3*n))
                block[:, 3*target:3*target+3] += np.eye(3)
                block[:, 3*i:3*i+3] -= np.asarray(op['R'])
                constraints.append(block)
        C = np.vstack(constraints)
        assert np.max(np.abs(C@flat_B)) < 1e-10
        assert np.linalg.matrix_rank(C)+np.linalg.matrix_rank(flat_B) == 3*n
        for q in [m['state'][-1], m['state'][-1]+.037, m['state'][-1]-.029]:
            points = b+B[:, :, 0]*q
            for op in operations:
                transformed = points@np.asarray(op['R']).T+op['t']
                delta = transformed-points[op['permutation']]
                delta[:, periodic] -= np.rint(delta[:, periodic])
                maxima.append(float(np.max(np.abs(delta))))
        result['materials'].append({'key': m['key'], 'id': m['id'], 'orbits': rows,
                                    'flat_b_shape': [3*n, 1], 'flat_B_shape': list(flat_B.shape),
                                    'C_shape': list(C.shape), 'C_rank': int(np.linalg.matrix_rank(C)),
                                    'b': b.tolist(), 'B': B.tolist(), 'coordinates': xyz.tolist()})
    result['maximum_residual'] = max(maxima)
    assert result['maximum_residual'] < 1e-10
    out = ROOT / 'dist/tables/h2_orbit_derivations.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'materials': len(result['materials']), 'orbits': 4,
                      'maximum_residual': result['maximum_residual'], 'output': str(out)}))


if __name__ == '__main__':
    main()
