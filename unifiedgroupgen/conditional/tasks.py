"""Task visibility rules; composition restrictions remain hard task constraints."""
import math

TASKS = ('dng', 'csp')
RESERVED = {'source_domain', 'space_group', 'layer_group', 'plane_group', 'rod_group', 'group',
            'symmetry', 'chemical_system', 'composition', 'formula', 'kind', 'dimension'}

def generation_mode(config):
    return config.get('generation_task')


def validate_task_config(config):
    from unifiedgroupgen.pl_data.orbit_order import ORBIT_ORDERS
    if config.get('orbit_order', 'random') not in ORBIT_ORDERS:
        raise ValueError('orbit_order must be random or site_symmetry')
    if config.get('geometry_objective') not in (None, 'legacy', 'decomposed'):
        raise ValueError('geometry_objective must be legacy or decomposed')
    for key in ('short_contact_weight', 'layer_log_area_weight', 'layer_height_weight'):
        weight = config.get(key)
        if weight is not None and (not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0):
            raise ValueError(f'{key} must be finite and nonnegative')
    recovery_weight = config.get('orbit_recovery_weight', 0.)
    if not isinstance(recovery_weight, (int, float)) or not math.isfinite(recovery_weight) or recovery_weight < 0:
        raise ValueError('orbit_recovery_weight must be finite and nonnegative')
    task = generation_mode(config)
    from unifiedgroupgen.group_constraints.group_prior import GROUP_POLICIES
    group_policy = config.get('csp_group_policy', 'prior')
    if group_policy not in GROUP_POLICIES or (group_policy != 'prior' and task != 'csp'):
        raise ValueError('Composition-ranked group selection requires CSP')
    if task is None:
        return
    if task not in TASKS or not config.get('model', {}).get('task_controls'):
        raise ValueError('Select one independent task and enable model.task_controls')
    properties = config.get('properties', {})
    if not 0 <= config.get('cond_dropout', 0.) < 1:
        raise ValueError('cond_dropout must be in [0, 1)')
    if config.get('dataset_domains') or RESERVED & set(properties):
        raise ValueError('Group, composition and source labels are not target properties')
    if any(spec.get('type', 'scalar') != 'scalar' or spec.get('context')
           or spec.get('domain_scalers') or spec.get('per_domain') for spec in properties.values()):
        raise ValueError('Property conditioning accepts scalar target properties only')
    probability = config.get('chemical_set_probability', .5)
    if not 0 <= probability <= 1:
        raise ValueError('chemical_set_probability must be in [0, 1]')


def observed_properties(values, schema):
    return {name: values[name] for name in schema
            if values.get(name) is not None and math.isfinite(float(values[name]))}


def eligible_records(records, config):
    schema = config.get('properties', {})
    if generation_mode(config) is None or not schema:
        return records
    return [record for record in records if observed_properties(record.get('properties', {}), schema)]
