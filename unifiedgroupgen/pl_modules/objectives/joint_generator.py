"""Conditional generator matching, with rate-weighted structured-action supervision."""
import torch


def joint_generator_loss(model, samples, jump_weight=1.):
    if not samples:
        raise ValueError('A nonempty conditional-path batch is required')
    continuous, jumps, lattice, coordinates, moments = [], [], [], [], []
    for sample in samples:
        output = model(sample.state, sample.time, sample.properties)
        lat = (output['metric_anchor']-sample.metric_anchor).square().sum()
        pos = (output['coordinate_anchor']-sample.coordinate_anchor).square().sum()
        mag = lat.new_zeros(())
        dimension = sample.metric_anchor.numel()+sample.coordinate_anchor.numel()
        if sample.moment_anchor is not None:
            mag = (output['moment_anchor']-sample.moment_anchor).square().sum()
            dimension += sample.moment_anchor.numel()
        cont = (lat+pos+mag)/max(1, dimension)
        mu = output['remaining']
        if sample.remaining:
            try:
                index = output['actions'].index(sample.next_action)
            except ValueError as error:
                raise ValueError(f'Teacher action is outside inference support: {sample.identity}: {sample.next_action}') from error
            logpi = torch.log_softmax(output['action_logits'], 0)[index]
            jump = mu-sample.remaining*(torch.log(mu.clamp_min(1e-12))+logpi)
        else:
            jump = mu
        continuous.append(cont); jumps.append(jump)
        lattice.append(lat/max(1, sample.metric_anchor.numel()))
        coordinates.append(pos/max(1, sample.coordinate_anchor.numel()))
        moments.append(mag/max(1, 0 if sample.moment_anchor is None else sample.moment_anchor.numel()))
    result = {name: torch.stack(values).mean() for name, values in (
        ('continuous', continuous), ('jump', jumps), ('lattice', lattice),
        ('coordinates', coordinates), ('moments', moments))}
    result['loss'] = result['continuous']+jump_weight*result['jump']
    if not all(torch.isfinite(value) for value in result.values()):
        raise FloatingPointError('Nonfinite joint generator loss')
    return result
