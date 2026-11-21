"""Attach chemical controls and optional target properties to a training endpoint."""
from dataclasses import replace
import numpy as np
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl, orbit_counts
from unifiedgroupgen.conditional.tasks import generation_mode, observed_properties, validate_task_config

def task_target(target, config, seed):
    task = generation_mode(config)
    if task is None:
        return target
    validate_task_config(config)
    schema = config.get('properties', {})
    properties = observed_properties(target.properties, schema)
    if schema and not properties:
        raise ValueError('Record has no observed label for property conditioning')
    control = chemical_control(target.state, task, config, seed)
    return replace(target, state=replace(target.state, chemistry=control), properties=properties)


def chemical_control(state, task, config, seed):
    vocabulary = tuple(config['allowed_elements'])
    if task == 'csp':
        control = ChemicalControl.from_inputs(counts=orbit_counts(state.spec, state.letters, state.elements))
        control.domain(vocabulary)
        return control
    rng = np.random.default_rng(seed)
    if rng.random() < config.get('chemical_set_probability', .5):
        present = set(state.elements)
        extra = sorted(set(vocabulary)-present)
        count = int(rng.integers(0, min(4, len(extra))+1))
        added = rng.choice(extra, size=count, replace=False).tolist() if count else []
        control = ChemicalControl.from_inputs(allowed=sorted(present | set(added)))
    else:
        control = ChemicalControl()
    control.domain(vocabulary)
    return control
