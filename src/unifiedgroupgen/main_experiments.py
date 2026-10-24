"""The approved 3 datasets x 5 property-training regimes. No control fine-tuning."""

import copy
import yaml

REGIMES = {'u': (), 'ef': ('formation_energy',), 'eg': ('band_gap',),
           'eh': ('energy_above_hull',),
           'all': ('formation_energy', 'band_gap', 'energy_above_hull')}
SOURCES = {'mp20': ('mp_20', 'configs/mp20_verified_v1.yaml'),
           'c2db51': ('c2db_51', 'configs/c2db_verified_v1.yaml')}
TRAINING_ORDER = (
    'mp20_u', 'c2db51_u', 'joint_u',
    'mp20_all', 'c2db51_all', 'joint_all',
    'mp20_ef', 'mp20_eg', 'mp20_eh',
    'c2db51_ef', 'c2db51_eg', 'c2db51_eh',
    'joint_ef', 'joint_eg', 'joint_eh',
)
RUNTIME_BY_DATASET = {
    'mp20': {'batch_size': 256, 'edge_budget': 400000},
    'c2db51': {'batch_size': 768, 'edge_budget': 200000},
    'joint': {'batch_size': 256, 'edge_budget': 400000},
}


def configurations(root):
    parents = {key: yaml.safe_load((root/path).read_text(encoding='utf-8'))
               for key, (_, path) in SOURCES.items()}
    result = {}
    for dataset in ('mp20', 'c2db51', 'joint'):
        source_keys = tuple(SOURCES) if dataset == 'joint' else (dataset,)
        base = copy.deepcopy(parents[source_keys[0]])
        base.update(dataset='mp20_c2db51' if dataset == 'joint' else SOURCES[dataset][0],
                    max_atoms=192, max_orbits=52, composition_probability=0.0,
                    observation_probability=0.0, validation_interval=5,
                    source_configs={SOURCES[key][0]: SOURCES[key][1] for key in source_keys})
        for split in ('train', 'val', 'test'):
            base[split+'_files'] = [f'{parents[key]["prepared_root"]}/{split}.jsonl' for key in source_keys]
        if dataset == 'joint':
            base.update(kinds=['space', 'layer'], dataset_domains={'mp_20': 0, 'c2db_51': 1},
                        prepared_root='artifacts/main15/joint')
            for key in ('data_root', 'coordinate_mode', 'symmetry_tolerance', 'property_columns'):
                base.pop(key, None)
        for regime, properties in REGIMES.items():
            name = f'{dataset}_{regime}'
            config = copy.deepcopy(base)
            config.update(experiment_id=name, experiment_family='main15_source_integrity_v1',
                          properties={p: {'type': 'scalar', **({'per_domain': True} if dataset == 'joint' else {})}
                                      for p in properties},
                          cond_dropout=.1 if properties else 0.0,
                          property_subset_training=regime == 'all',
                          data_audit=f'artifacts/main15/audits/{name}.json')
            if dataset == 'joint':
                config['properties']['source_domain'] = {'type': 'categorical', 'num_classes': 2, 'context': True}
            result[name] = config
    return {name: result[name] for name in TRAINING_ORDER}


def paths(config, root):
    return {split: [root/p for p in config[split+'_files']] for split in ('train', 'val', 'test')}
