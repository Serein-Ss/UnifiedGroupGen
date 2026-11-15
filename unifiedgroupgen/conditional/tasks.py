"""Task visibility rules; composition restrictions remain hard task constraints."""
import math

TASKS = ('dng', 'property', 'csp')
RESERVED = {'source_domain', 'space_group', 'layer_group', 'plane_group', 'group',
            'symmetry', 'chemical_system', 'composition', 'formula', 'kind', 'dimension'}

def validate_task_config(config):
    task = config.get('generation_task')
    if task is None:
        return
    if task not in TASKS or not config.get('model', {}).get('task_controls'):
        raise ValueError('Select one independent task and enable model.task_controls')
    properties = config.get('properties', {})
    if (task == 'property') != bool(properties):
        raise ValueError('Only the property task requires a nonempty target-property schema')
    if config.get('dataset_domains') or RESERVED & set(properties):
        raise ValueError('Group, composition and source labels are not target properties')
    if any(spec.get('type', 'scalar') != 'scalar' or spec.get('context')
           or spec.get('domain_scalers') or spec.get('per_domain') for spec in properties.values()):
        raise ValueError('Independent property generation accepts scalar target properties only')
    probability = config.get('chemical_set_probability', .5)
    if not 0 <= probability <= 1:
        raise ValueError('chemical_set_probability must be in [0, 1]')


def has_property_target(record, names):
    return any(record.get('properties', {}).get(name) is not None
               and math.isfinite(float(record['properties'][name])) for name in names)
