"""In-memory orbit and geometry sampling; no configuration, files or task loops."""
from unifiedgroupgen.common.structures import composition_of
from unifiedgroupgen.group_constraints.compiler import compile_descriptor
from unifiedgroupgen.common.precision import inference_autocast


def sample_orbits(proposal, *, kind, properties=None, composition=None, fixed=None,
                  chemical=None, prefix=None, guidance=0., temperature=1., generator=None,
                  condition_composition=True):
    """Choose/accept a descriptor, compile its chart and enforce hard chemistry."""
    with inference_autocast(proposal):
        descriptor=fixed or proposal.sample(properties,kind,composition,guidance,temperature,generator,
            chemical=chemical,prefix=prefix,condition_composition=condition_composition)
    compiled=compile_descriptor(descriptor)
    counts=composition_of(compiled)
    if composition is not None and counts!=composition:
        raise ValueError('Fixed descriptor conflicts with requested composition')
    if chemical is not None and not chemical.matches(counts):
        raise ValueError('Generated descriptor violates requested element sets')
    return compiled


def sample_geometry(flow, compiled, *, properties=None, steps=64, guidance=0.,
                    observation=None, composition=None, generator=None, prior=False,
                    condition_observation=True, solver='heun'):
    """Generate legal continuous parameters and expand them to physical geometry."""
    if prior:
        parameter=next(flow.parameters())
        from unifiedgroupgen.generation.prior import sample_prior
        state=sample_prior(compiled,parameter.device,parameter.dtype,generator,getattr(flow,'prior',None))
        if observation is not None: state=observation.retract(state)
        generated=compiled.expand(state)
        generated['state']=state
        return generated
    from unifiedgroupgen.generation.flow import sample_flow
    return sample_flow(flow,compiled,properties,steps,guidance,observation,composition,generator,
                       condition_observation=condition_observation,solver=solver)
