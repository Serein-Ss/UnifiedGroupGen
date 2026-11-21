"""One packed forward and generator-matching losses on a shared conditional path."""
import torch

def geometry_errors(state, metric_error, position_error):
    """Squared errors of the existing scaled field targets in physical coordinates."""
    with torch.autocast(device_type=state.metric.device.type, enabled=False):
        return _geometry_errors(state, metric_error.to(state.metric.dtype),
                                position_error.to(state.metric.dtype))


def _geometry_errors(state, metric_error, position_error):
    chart = state.compiled
    rank = state.spec.rank
    zero = metric_error.sum() * 0
    shape, log_measure, isotropic = zero, zero, zero
    if rank:
        basis = metric_error.new_tensor(chart.standard)
        error = torch.einsum('k,kij->ij', metric_error, basis)
        trace = error.trace()
        log_measure = (.5 * trace).square()
        isotropic = trace.square() / rank
        shape_dof = len(chart.metric_basis) - 1
        if shape_dof:
            detraced = error - trace / rank * torch.eye(rank, device=error.device, dtype=error.dtype)
            shape = detraced.square().sum() / shape_dof

    active = [(section, template) for section, template in zip(chart.slices, chart.templates) if template.dof]
    site = zero
    if active:
        lattice = chart.lattice(state.metric)
        length_sq = (lattice[:, :rank].square().sum()/rank).clamp_min(1e-8) if rank else 1.
        physical = position_error @ lattice.T
        site = torch.stack([physical[section].square().sum()/(template.dof * template.multiplicity)
                            for section, template in active]).mean() / length_sq
    return shape, log_measure, site, isotropic


def layer_geometry_errors(state, metric_error, position_error, reference_orbits):
    """Squared log-area and centered nonperiodic height residuals for layer states."""
    with torch.autocast(device_type=state.metric.device.type, enabled=False):
        metric_error = metric_error.to(state.metric)
        basis = metric_error.new_tensor(state.compiled.standard)
        trace = basis.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
        log_area = (.5 * (metric_error @ trace)).square()
        height = position_error[:, 2].to(state.coordinates)
        if reference_orbits is not None and len(height):
            atoms = torch.as_tensor(state.compiled.atom_orbit, device=height.device)
            reference = torch.tensor(reference_orbits, dtype=torch.bool, device=height.device)
            height = height[reference[atoms]]
        height_error = (height - height.mean()).square().mean() if len(height) else height.sum()
        return log_area, height_error


def loss(model,samples,jump_weight=1., *, recovery=(), recovery_weight=0., geometry_objective=None,
         short_contact_weight=None, layer_log_area_weight=None, layer_height_weight=None):
    if short_contact_weight is not None:
        from .short_contacts import endpoint_short_contacts
    outputs=model.forward_batch([s.state for s in samples],[s.time for s in samples],[s.properties for s in samples])
    geometry=[];elements=[];orbits=[]
    components=[]
    contacts=[]
    layer_errors=[]
    log_layer = layer_log_area_weight is not None or layer_height_weight is not None
    for sample,out in zip(samples,outputs):
        s=sample.state
        metric_error=out['metric_velocity_scaled']-(sample.metric_anchor-s.metric)
        metric=metric_error.square().mean()
        delta=out['coordinate_velocity_scaled']-(sample.coordinate_anchor-s.coordinates)
        position=delta.square().mean() if delta.numel() else metric*0
        legacy = metric + position
        if geometry_objective is not None:
            shape, size, site, isotropic = geometry_errors(s, metric_error, delta)
            components.append(torch.stack((shape.detach(), size.detach(), site.detach(), legacy.detach())))
            geometry.append(shape + isotropic + site if geometry_objective == 'decomposed' else legacy)
        else:
            geometry.append(legacy)
        selected=[i for i,z in enumerate(s.elements) if z==0]
        if selected:
            target=torch.tensor([sample.element_targets[i] for i in selected],device=s.metric.device)
            elements.append(torch.nn.functional.cross_entropy(out['element_logits'][selected],target,reduction='sum'))
        else:elements.append(metric*0)
        if short_contact_weight is not None and s.frozen and s.num_atoms:
            with torch.set_grad_enabled(torch.is_grad_enabled() and short_contact_weight > 0):
                contacts.append(endpoint_short_contacts(s, out['metric_velocity_scaled'],
                                                        out['coordinate_velocity_scaled']))
        if log_layer and s.spec.kind == 'layer':
            layer_errors.append(torch.stack(layer_geometry_errors(s, metric_error, delta,
                                                                   sample.reference_orbits)))
        mu=out['remaining'].float()
        if sample.remaining:
            index=out['actions'].index(sample.next_action)
            orbits.append(mu-sample.remaining*(mu.log()+out['action_logits'].log_softmax(0)[index]))
        else:orbits.append(mu)
    result=dict(continuous=torch.stack(geometry).mean(),element=torch.stack(elements).mean(),jump=torch.stack(orbits).mean())
    result['loss']=result['continuous']+result['element']+jump_weight*result['jump']
    if components:
        names = ('lattice_shape', 'lattice_log_measure', 'orbit_site', 'geometry_legacy')
        result.update(zip(names, torch.stack(components).mean(0).unbind()))
    if log_layer:
        area, height = (torch.stack(layer_errors).sum(0) / len(samples)).unbind() if layer_errors else (
            result['loss'].new_zeros(()), result['loss'].new_zeros(()))
        result['layer_log_area'] = area.detach()
        result['layer_height'] = height.detach()
        if layer_log_area_weight:
            result['loss'] = result['loss'] + layer_log_area_weight * area
        if layer_height_weight:
            result['loss'] = result['loss'] + layer_height_weight * height
    if short_contact_weight is not None:
        short = torch.stack(contacts) if contacts else result['loss'].new_empty(0)
        result['short_contact'] = short.detach().sum() / len(samples)
        result['short_contact_eligible'] = short.new_tensor(len(contacts) / len(samples))
        result['short_contact_violation'] = (short.detach() > 0).sum().to(short.dtype) / len(samples)
        if short_contact_weight:
            result['loss'] = result['loss'] + short_contact_weight * short.sum() / len(samples)
    if recovery_weight:
        recovery_loss = result['loss'].new_zeros(())
        if recovery:
            predictions = model.forward_batch([r.state for r in recovery], [r.time for r in recovery],
                                              [r.properties for r in recovery],
                                              actions=[r.support for r in recovery])
            recovery_loss = torch.stack([
                -out['action_logits'].float().log_softmax(0)[out['actions'].index(r.correction)]
                for r, out in zip(recovery, predictions)
            ]).mean()
        result['orbit_recovery'] = recovery_loss
        result['loss'] = result['loss'] + recovery_weight * recovery_loss
    if not torch.isfinite(torch.stack(tuple(result.values()))).all():raise FloatingPointError('Nonfinite mixed-flow loss')
    return result
