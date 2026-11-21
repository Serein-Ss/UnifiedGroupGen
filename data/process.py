"""Prepare crystal CSV splits using a source YAML and command-line overrides."""
from argparse import ArgumentParser, Namespace
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(args: Namespace) -> None:
    import torch
    from unifiedgroupgen.common.configuration import package_parent, project_root, load_yaml, resolve_path
    from unifiedgroupgen.pl_data.cache import ensure_tensor_cache, target_from_cache
    from unifiedgroupgen.group_constraints.wyckoff import record_orbit_labels

    root = args.root.resolve() if args.root else project_root()
    torch.set_num_threads(1)
    for config_path in args.config:
        config = load_yaml(config_path, root)
        if args.directory is not None:
            config['data_root'] = str(resolve_path(args.directory, root))
        model_path = args.model or package_parent()/'configs/model/joint_flow_equiformer.yaml'
        model = load_yaml(model_path, root)
        settings = {**model, **config.get('preprocessing', {})}
        splits = args.split if args.split is not None else settings.get('splits', ['train', 'val', 'test'])
        workers = args.workers if args.workers is not None else settings.get('workers', 0)
        graph_options = {
            key: getattr(args, key) if getattr(args, key) is not None else settings[key]
            for key in ('cutoff', 'max_neighbors', 'max_images')
        }
        if not splits or any(split not in ('train', 'val', 'test') for split in splits):
            raise ValueError('preprocessing.splits must select train, val or test')
        if workers < 0:
            raise ValueError('workers must be nonnegative')
        if graph_options['cutoff'] <= 0 or graph_options['max_neighbors'] < 1 or graph_options['max_images'] < 0:
            raise ValueError('cutoff/max_neighbors must be positive and max_images nonnegative')

        for split in splits:
            csv = resolve_path(Path(config['data_root']) / (split + '.csv'), root)
            print(f'Working on {csv}', flush=True)
            path, payload, hit = ensure_tensor_cache(
                config, split, root, operation_compiler=model['operation_compiler'],
                graph_options=graph_options, workers=workers,
            )
            record = payload['records'][0]
            endpoint = target_from_cache(payload, 0, record).state
            example = dict(id=record['id'], orbit_labels=record_orbit_labels(record, endpoint))
            print(json.dumps(dict(dataset=config['dataset'], split=split, state='cached' if hit else 'created',
                                  records=len(payload['records']), cache=str(path),
                                  cache_config=str(path.with_suffix('.config.json')),
                                  orbit_example=example), ensure_ascii=False), flush=True)


def parse_args(argv=None) -> Namespace:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument('directory', nargs='?', type=Path, help='Override the CSV directory for a single config')
    parser.add_argument('--config', required=True, nargs='+', type=Path, help='Source dataset YAML files')
    parser.add_argument('--model', type=Path, help='Model graph settings YAML')
    parser.add_argument('--root', type=Path, help='Workspace root; otherwise read .env')
    parser.add_argument('--split', nargs='+', choices=['train', 'val', 'test'])
    parser.add_argument('--workers', type=int)
    parser.add_argument('--cutoff', type=float)
    parser.add_argument('--max-neighbors', type=int)
    parser.add_argument('--max-images', type=int)
    args = parser.parse_args(argv)
    if args.directory is not None and len(args.config) != 1:
        parser.error('directory overrides require exactly one --config')
    return args


if __name__ == '__main__':
    main(parse_args())
