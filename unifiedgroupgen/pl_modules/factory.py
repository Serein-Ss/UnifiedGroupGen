"""Construction of the current joint-flow model only."""
from unifiedgroupgen.conditional.tasks import validate_task_config
from unifiedgroupgen.pl_modules.unified.model import JointFlow


def build_joint_model(config, device='cpu'):
    if config.get('model', {}).get('family') != 'joint_flow_v2':
        raise ValueError('Only model.family=joint_flow_v2 is supported by the production library')
    validate_task_config(config)
    return JointFlow(config).to(device)
