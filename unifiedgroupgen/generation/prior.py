"""Configurable priors on legal chart coordinates, never on unconstrained cells."""
import math
import torch


def sample_prior(compiled, device='cpu', dtype=torch.float32, generator=None, specification=None):
    spec = specification or {}
    if not spec or spec.get('name', 'legacy') == 'legacy':
        return compiled.prior(device, dtype, generator)
    for key in ('lattice_scale','height_scale','periodic_std','length_median','length_log_std','fallback_scale'):
        if key in spec and (not math.isfinite(spec[key]) or spec[key] <= 0):
            raise ValueError(f'Prior {key} must be finite and positive')
    z = compiled.prior(device, dtype, generator)
    width = compiled.lattice_dof
    name = spec['name']
    if name == 'physical_metric':
        # FlowMM-inspired physical cell base, pushed into this group's log-metric chart.
        d = compiled.root.shape[0]
        length = torch.exp(z.new_tensor(math.log(spec.get('length_median', 3.))) +
                           spec.get('length_log_std', .25)*torch.randn(d, device=device, dtype=dtype, generator=generator))
        for _ in range(64):
            angles = (60+60*torch.rand(3, device=device, dtype=dtype, generator=generator))*math.pi/180
            raw = torch.diag(length.square())
            if d == 2:
                raw[0, 1] = raw[1, 0] = length.prod()*angles[0].cos()
            else:
                for (i, j), angle in zip(((1, 2), (0, 2), (0, 1)), angles):
                    raw[i, j] = raw[j, i] = length[i]*length[j]*angle.cos()
            root = compiled.tensor(compiled.invroot, z)
            values, vectors = torch.linalg.eigh(root @ raw @ root)
            if values.min() > 1e-7:
                log_metric = (vectors*values.log()) @ vectors.T
                z[:width] = torch.einsum('kij,ij->k', compiled.tensor(compiled.metric_basis, z), log_metric)
                break
        else:
            raise ValueError('Physical-cell prior failed to draw a positive-definite metric')
    elif name == 'train_gaussian':
        key = f'{compiled.descriptor.kind}:{compiled.descriptor.number}'
        stats = spec.get('lattice_statistics', {}).get(key)
        if stats:
            z[:width] = z.new_tensor(stats['mean']) + z[:width]*z.new_tensor(stats['std'])
        else:
            # Explicit documented fallback permits groups absent from training.
            z[:width] *= spec.get('fallback_scale', 1.)
    elif name in ('scaled_normal', 'wrapped_normal'):
        z[:width] *= spec.get('lattice_scale', 1.)
    else:
        raise ValueError(f'Unknown chart prior: {name}')
    periodic = compiled.tensor(compiled.state_periodic, z, torch.bool)
    if name == 'wrapped_normal':
        z[periodic] = torch.remainder(spec.get('periodic_mean', 0.) + spec.get('periodic_std', .5)*
            torch.randn(int(periodic.sum()), device=device, dtype=dtype, generator=generator), 1.)
    nonperiodic = ~periodic
    nonperiodic[:width] = False
    z[nonperiodic] *= spec.get('height_scale', 1.)
    if not torch.isfinite(z).all():
        raise FloatingPointError('Nonfinite chart prior; no silent repair')
    return z


def fit_prior(records, specification):
    """Fit group-wise lattice statistics from the supplied TRAIN records only."""
    from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
    spec = dict(specification or {})
    if spec.get('name') != 'train_gaussian':
        return spec
    grouped = {}
    for record in records:
        chart = compile_descriptor(Descriptor.from_dict(record['descriptor']))
        key = f'{chart.descriptor.kind}:{chart.descriptor.number}'
        grouped.setdefault(key, []).append(record['state'][:chart.lattice_dof])
    spec['lattice_statistics'] = {}
    for key, rows in grouped.items():
        values = torch.tensor(rows, dtype=torch.float64)
        spec['lattice_statistics'][key] = {'mean': values.mean(0).tolist(),
            'std': values.std(0, unbiased=False).clamp_min(spec.get('minimum_std', .1)).tolist(), 'records': len(rows)}
    spec['fitted_split'] = 'train'
    return spec
