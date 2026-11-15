"""Apply task visibility to endpoints before constructing network inputs."""
from dataclasses import replace
import math
import numpy as np
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl, orbit_counts, edit_signature
from unifiedgroupgen.conditional.tasks import TASKS, RESERVED, validate_task_config, has_property_target
from unifiedgroupgen.pl_data.programs import inventory_program

def task_target(target, config, seed):
    task = config.get('generation_task')
    if task is None:
        return target
    validate_task_config(config)
    rng = np.random.default_rng(seed)
    vocabulary = tuple(config['allowed_elements'])
    properties = {}
    if task == 'property':
        properties = {name: target.properties[name] for name in config['properties']
                      if target.properties.get(name) is not None and math.isfinite(float(target.properties[name]))}
        if not properties:
            raise ValueError('Property-task record has no observed target label')
    if task == 'csp':
        control = ChemicalControl.from_inputs(counts=orbit_counts(target.state.spec, target.state.letters,
                                                                 target.state.elements))
    elif rng.random() < config.get('chemical_set_probability', .5):
        present = set(target.state.elements)
        extra = sorted(set(vocabulary)-present)
        count = int(rng.integers(0, min(4, len(extra))+1))
        added = rng.choice(extra, size=count, replace=False).tolist() if count else []
        control = ChemicalControl.from_inputs(allowed=sorted(present | set(added)))
    else:
        control = ChemicalControl()
    control.domain(vocabulary)
    return replace(target, state=replace(target.state, chemistry=control), properties=properties)
