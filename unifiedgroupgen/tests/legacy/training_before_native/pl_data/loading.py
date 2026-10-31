"""Read and audit train/validation splits without constructing networks or loggers."""
from unifiedgroupgen.common.configuration import resolve_path
from unifiedgroupgen.conditional.properties import attach_dataset_domains
from unifiedgroupgen.pl_data.encoding import read_jsonl
from unifiedgroupgen.tests.preflight import check_training_gate


def load_training_splits(config, root, train=None, val=None):
    prepared = config.get('prepared_root', f"data/dataset/{config['dataset']}")
    paths = {split: [resolve_path(p, root) for p in ({'train': train, 'val': val}[split]
             or config.get(split+'_files') or [f'{prepared}/{split}.jsonl'])] for split in ('train', 'val')}
    check_training_gate(config, paths['train'], paths['val'], root)
    records = attach_dataset_domains([r for p in paths['train'] for r in read_jsonl(p)], config)
    validation = attach_dataset_domains([r for p in paths['val'] for r in read_jsonl(p)], config)
    if not records or not validation:
        raise ValueError('Training and validation records must both be nonempty')
    if {(r['dataset'], r['id']) for r in records} & {(r['dataset'], r['id']) for r in validation}:
        raise ValueError('Training/validation identifier overlap')
    return records, validation, paths
