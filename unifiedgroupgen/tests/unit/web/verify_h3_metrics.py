"""Audit H.3 metric charts using the Site's saved TiO2 and MoS2 fixtures."""

import json
from pathlib import Path

import numpy as np
from scipy.linalg import null_space


ROOT = Path(__file__).resolve().parents[4]/'web/symmetry-lab'


def symmetric_function(a, fn):
    values, vectors = np.linalg.eigh((a+a.T)/2)
    return (vectors*fn(values))@vectors.T


def main():
    text = (ROOT/'dist/data.js').read_text(encoding='utf-8')
    data = json.loads(text[text.index('{'):].rstrip(';\r\n'))
    result = {'passed': True, 'length_scale_angstrom': 3., 'materials': []}
    all_errors = []
    for m in data['materials']:
        p = m['p']
        rotations = np.array([o['R'] for o in m['operations']])[:, :p, :p]
        average = np.mean(rotations.transpose(0, 2, 1)@rotations, axis=0)
        h0 = 9*average/np.linalg.det(average)**(1/p)
        root = symmetric_function(h0, np.sqrt)
        inverse = symmetric_function(h0, lambda x: 1/np.sqrt(x))
        orthogonal = root[None]@rotations@inverse[None]
        standard = []
        for i in range(p):
            for j in range(i, p):
                a = np.zeros((p, p))
                a[i, j] = a[j, i] = 1 if i == j else 1/np.sqrt(2)
                standard.append(a)
        standard = np.array(standard)
        constraints = np.concatenate([
            np.stack([(u.T@b@u-b).reshape(-1) for b in standard], axis=1)
            for u in orthogonal
        ])
        coefficients = null_space(constraints, rcond=1e-8)
        projector = coefficients@coefficients.T
        canonical = []
        for column in projector.T:
            v = column.copy()
            for previous in canonical:
                v -= previous*(previous@v)
            norm = np.linalg.norm(v)
            if norm > 1e-7:
                v /= norm
                if v[np.flatnonzero(np.abs(v)>1e-7)[0]] < 0:
                    v = -v
                canonical.append(v)
        basis = np.einsum('ik,ide->kde', np.stack(canonical, axis=1), standard)
        assert np.allclose(h0, m['H0']) and np.allclose(basis, m['basis'])
        lattice = np.array(m['lattice'])[:p, :p]
        h = lattice.T@lattice
        a = inverse@h@inverse
        s = symmetric_function(a, np.log)
        k = np.einsum('ijk,jk->i', basis, s)
        errors = {
            'reference_metric': np.max(np.abs(h0-m['H0'])),
            'basis': np.max(np.abs(basis-m['basis'])),
            'encoded_k': np.max(np.abs(k-np.array(m['state'][:m['k_dof']]))),
            'orthogonality': np.max(np.abs(orthogonal.transpose(0, 2, 1)@orthogonal-np.eye(p))),
        }
        for values in [np.zeros(len(k)), k, k+.07*np.arange(1, len(k)+1)]:
            log_metric = np.einsum('i,ijk->jk', values, basis)
            metric = root@symmetric_function(log_metric, np.exp)@root
            assert np.linalg.eigvalsh(metric).min() > 0
            recovered = np.einsum('ijk,jk->i', basis, symmetric_function(inverse@metric@inverse, np.log))
            all_errors.append(float(np.max(np.abs(recovered-values))))
            for r in rotations:
                all_errors.append(float(np.max(np.abs(r.T@metric@r-metric))))
            chol = np.linalg.cholesky(metric).T
            frame = symmetric_function(log_metric, lambda x: np.exp(x/2))@root
            all_errors.append(float(np.max(np.abs(frame.T@frame-metric))))
            all_errors.append(float(np.max(np.abs(chol.T@chol-metric))))
            for r, u in zip(rotations, orthogonal):
                all_errors.append(float(np.max(np.abs(frame@r@np.linalg.inv(frame)-u))))
        new_root = 2*root
        shifted = s-np.log(4)*np.eye(p)
        errors['reference_scale_change'] = np.max(np.abs(new_root@symmetric_function(shifted, np.exp)@new_root-h))
        frame = symmetric_function(s, lambda x: np.exp(x/2))@root
        all_errors.extend(float(v) for v in errors.values())
        result['materials'].append({
            'key': m['key'], 'p': p, 'operation_count': len(rotations),
            'periodic_operations': rotations.tolist(), 'average_metric': average.tolist(),
            'average_metric_determinant': float(np.linalg.det(average)),
            'H0': h0.tolist(), 'root': root.tolist(), 'inverse_root': inverse.tolist(),
            'orthogonal_operations': orthogonal.tolist(),
            'standard_basis': standard.tolist(), 'full_constraint_shape': list(constraints.shape),
            'full_constraints': constraints.tolist(), 'constraint_rank': int(np.linalg.matrix_rank(constraints, tol=1e-8)),
            'basis': basis.tolist(), 'k': k.tolist(), 'A': a.tolist(), 'S': s.tolist(),
            'H': h.tolist(), 'lattice': lattice.tolist(), 'fixed_action_frame': frame.tolist(),
            'frame_rotation': (lattice@np.linalg.inv(frame)).tolist(),
            'cell_measure': float(np.sqrt(np.linalg.det(h))),
            'reference_edge_angstrom': float(np.sqrt(h0[0, 0])),
            'checks': {key: float(value) for key, value in errors.items()},
        })
    result['maximum_residual'] = max(all_errors)
    assert result['maximum_residual'] < 1e-10
    out = ROOT/'dist/tables/h3_metric_derivations.json'
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'materials': 2, 'maximum_residual': result['maximum_residual'],
                      'basis_dimensions': [len(m['k']) for m in result['materials']], 'output': str(out)}))


if __name__ == '__main__':
    main()
