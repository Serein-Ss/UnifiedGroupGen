import random
import numpy as np
import torch
from unifiedgroupgen.group_constraints.projections import AffineObservation
from unifiedgroupgen.pl_modules.objectives.flow_matching import flow_loss_batch

def joint_loss(proposal, flow, records, composition_probability=0.5, observation_probability=0.25,
               coordinate_length_scale=1.0):
    like = next(flow.parameters())
    descriptors, compiled, cpu_endpoints = zip(*[_training_entry(r, like.dtype) for r in records])
    endpoints = torch.cat(cpu_endpoints).to(like.device).split([z.numel() for z in cpu_endpoints])
    compositions, observations = [], []
    for r, c, endpoint in zip(records, compiled, cpu_endpoints):
        composition = None
        # Keep the RNG draw even at probability zero for resume reproducibility.
        if random.random() < composition_probability:
            elements, counts = np.unique(c.elements, return_counts=True)
            composition = dict(zip(elements.tolist(), counts.tolist()))
        observation = None
        if random.random() < observation_probability:
            policy = getattr(flow, 'observation_policy', None)
            if policy:
                from unifiedgroupgen.conditional.structure import training_observation
                observation = training_observation(c, endpoint, policy)
            else:
                indices = (list(range(c.lattice_dof)) if random.random() < 0.5
                           else random.sample(range(c.dof), max(1, c.dof//3)))
                observation = AffineObservation.fixed(c.dof, indices, endpoint[indices].detach().cpu())
        compositions.append(composition)
        observations.append(observation)
    properties = [r['properties'] for r in records]
    visibility = None
    if proposal.training and getattr(flow, 'condition_mask_policy', 'independent') == 'shared':
        from unifiedgroupgen.conditional.properties import sample_visibility
        visibility = sample_visibility(proposal.condition, properties, endpoints[0])
    descriptor = proposal.loss_batch(descriptors, properties, compositions, visibility=visibility)
    continuous = flow_loss_batch(flow, compiled, endpoints, properties, observations, compositions, visibility=visibility)
    if not np.isfinite(coordinate_length_scale) or coordinate_length_scale <= 0:
        raise ValueError('coordinate_length_scale must be finite and positive (angstrom)')
    objective = (continuous['loss'] if coordinate_length_scale == 1.0 else
                 continuous['lattice']+continuous['coordinates']/coordinate_length_scale**2)
    return {'descriptor': descriptor, 'flow': objective,
            'lattice': continuous['lattice'], 'coordinates': continuous['coordinates']}
from unifiedgroupgen.pl_data.dataset import _training_entry
