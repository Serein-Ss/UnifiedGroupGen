import torch
from unifiedgroupgen.common.configuration import resolve_path
from unifiedgroupgen.pl_modules.orbitflow import OrbitFlow
from unifiedgroupgen.pl_modules.descriptor import DescriptorTransformer


def build_models(config, device):
    from unifiedgroupgen.tests.catalog import check_layer_catalog
    check_layer_catalog()
    groups = config.get("groups")
    if groups is None:
        limits = {"space": 230, "layer": 80, "plane": 17}
        groups = [(kind, number) for kind in config.get("kinds", [config["kind"]]) for number in range(1, limits[kind]+1)]
    properties = config.get("properties", {})
    architecture = config.get("model", {})
    def backbone(name):
        specification = architecture.get(name)
        if specification is None:
            return None
        from hydra.utils import instantiate
        return instantiate(specification, _partial_=True)
    proposal = DescriptorTransformer(groups, properties, hidden=architecture.get("proposal_hidden", architecture.get("hidden", 128)),
                                     layers=architecture.get("proposal_layers", 3), heads=architecture.get("heads", 4),
                                     max_atoms=config.get("max_atoms", 192), max_orbits=config.get("max_orbits", 32),
                                     cond_dropout=config.get("cond_dropout", 0.1), allowed_elements=config.get('allowed_elements'),
                                     backbone_factory=backbone('descriptor_backbone')).to(device)
    flow = OrbitFlow(properties, hidden=architecture.get("flow_hidden", architecture.get("hidden", 128)), layers=architecture.get("flow_layers", 4),
                     frequencies=architecture.get("frequencies", 8), cond_dropout=config.get("cond_dropout", 0.1),
                     checkpoint_layers=architecture.get('checkpoint_layers', False),
                     edge_chunk_size=architecture.get('edge_chunk_size', 8192),
                     backbone_factory=backbone('flow_backbone'),
                     candidate_backbone_factory=backbone('candidate_backbone')).to(device)
    proposal.condition.subset_training = config.get('property_subset_training', False)
    flow.property_encoder.subset_training = config.get('property_subset_training', False)
    if config.get('input_contract') == 'source_free_v2':
        if ('source_domain' in properties or config.get('dataset_domains')
                or any(spec.get('per_domain') or spec.get('domain_scalers') for spec in properties.values())):
            raise ValueError('New models must not consume source labels or source-wise property scalers')
    flow.condition_mask_policy = config.get('condition_mask_policy', 'independent')
    if flow.condition_mask_policy not in ('shared', 'independent'):
        raise ValueError('condition_mask_policy must be shared or independent')
    flow.prior = config.get('prior')
    flow.flow_path = config.get('flow_path')
    flow.observation_policy = config.get('observation_policy')
    if architecture.get('candidate_backbone'):
        from unifiedgroupgen.pl_modules.backbones.provenance import backbone_identity
        flow.backbone_identity = backbone_identity(flow.candidate_backbone)
        if config.get('backbone_identity') is not None and config['backbone_identity'] != flow.backbone_identity:
            raise ValueError('Installed backbone source or dependency versions differ from the checkpoint')
    return proposal, flow


def load_model_checkpoint(path, map_location='cpu', weights_only=False):
    """Read existing inference .pt and native Lightning .ckpt without rewriting either."""
    saved = torch.load(resolve_path(path), map_location=map_location, weights_only=weights_only)
    if 'proposal' in saved and 'flow' in saved:
        return saved
    if 'state_dict' not in saved or 'ugg_state' not in saved:
        raise ValueError('Not a UnifiedGroupGen inference or Lightning checkpoint')
    state = saved['ugg_state']
    return {'config': state['config'], 'epoch': saved['epoch'], 'runtime_options': state['runtime'],
            **state['hashes'], 'trainer_backend': 'lightning',
            'proposal': {k.removeprefix('proposal.'):v for k,v in saved['state_dict'].items() if k.startswith('proposal.')},
            'flow': {k.removeprefix('flow.'):v for k,v in saved['state_dict'].items() if k.startswith('flow.')}}
