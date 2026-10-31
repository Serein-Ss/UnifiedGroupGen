"""Small research CLI: preprocessing, joint training, inverse sampling, and validation."""

from unifiedgroupgen.tests.legacy.paths import project_root, resolve_path
import argparse
import json
import os
import random
import time
from pathlib import Path
import numpy as np
import torch
import yaml
from unifiedgroupgen.conditional.properties import fit_property_scalers, attach_dataset_domains
from unifiedgroupgen.pl_data.encoding import prepare_csv, read_jsonl, write_jsonl
from unifiedgroupgen.group_constraints.validation import to_structure, validate, summarize_attempts
from unifiedgroupgen.pl_modules.orbitflow import OrbitFlow
from unifiedgroupgen.pl_modules.descriptor import DescriptorTransformer
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor


ROOT = project_root()


def root_path(path):
    return resolve_path(path, ROOT)


def artifact_path(path):
    from unifiedgroupgen.tests.legacy.paths import artifact_path as checked_artifact_path
    return checked_artifact_path(path, ROOT)


def audit(args):
    records = read_jsonl(root_path(args.records))
    reports = []
    for record in records:
        compiled = compile_descriptor(Descriptor.from_dict(record['descriptor']))
        generated = compiled.expand(torch.tensor(record['state'], dtype=torch.float64))
        reports.append({'id': record['id'], **validate(compiled, generated, isolate_identifier=True)})
    write_jsonl(artifact_path(args.output), reports)
    print(json.dumps({'records': len(reports), 'distance_valid': sum(r['distance_valid'] for r in reports), 'output': args.output}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--seed', type=int, default=42)
    subparsers = parser.add_subparsers(dest='command', required=True)
    prepare = subparsers.add_parser('prepare')
    prepare.add_argument('--config', required=True)
    prepare.add_argument('--split', choices=['train', 'val', 'test'], required=True)
    prepare.add_argument('--limit', type=int)
    prepare.add_argument('--output')
    prepare.add_argument('--workers', type=int, default=0)
    prepare.add_argument('--resume', action='store_true')
    training = subparsers.add_parser('train')
    training.add_argument('--config', required=True)
    training.add_argument('--trainer', choices=['lightning', 'legacy'], default=os.environ.get('UGG_TRAINER', 'lightning'))
    training.add_argument('--experiment', help='Experiment directory name under HYDRA_JOBS')
    training.add_argument('--wandb-mode', choices=['online', 'offline', 'disabled'],
                          help='Default from USE_WANDB_LOGGING: 1=online, 0=offline')
    training.add_argument('--wandb-project', default='UnifiedGroupGen')
    training.add_argument('--initialize-from', help='Load legacy inference weights into a new Lightning run; not optimizer resume')
    training.add_argument('--train', nargs='+')
    training.add_argument('--val', nargs='+')
    training.add_argument('--device', default='cpu')
    training.add_argument('--epochs', type=int)
    training.add_argument('--batch-size', type=int)
    training.add_argument('--edge-budget', type=int, help='Runtime edge cap without editing the audited configuration')
    training.add_argument('--coordinate-length-scale', type=float,
                          help='Coordinate loss reference length in angstrom; runtime override recorded in checkpoints')
    training.add_argument('--output')
    training.add_argument('--resume')
    training.add_argument('--checkpoint', action=argparse.BooleanOptionalAction, default=None,
                          help='Override activation checkpointing without changing model weights or the data configuration')
    training.add_argument('--edge-chunk-size', type=int)
    training.add_argument('--validation-interval', type=int, help='Validate every N epochs (default: 5); also validate the final epoch')
    sampling = subparsers.add_parser('sample')
    sampling.add_argument('--checkpoint', required=True)
    sampling.add_argument('--device', default='cpu')
    sampling.add_argument('--kind', choices=['space', 'layer', 'plane'], default='space')
    sampling.add_argument('--properties', default='{}')
    sampling.add_argument('--composition')
    sampling.add_argument('--formula', help='Reduced empirical formula; choose feasible formula-unit counts')
    sampling.add_argument('--formula-units', nargs='+', type=int, default=[1, 2, 3, 4])
    sampling.add_argument('--chemical-system', help='Exact element set, e.g. Li-Fe-O; all must occur')
    sampling.add_argument('--allowed-elements', help='Only these elements may occur')
    sampling.add_argument('--required-elements', help='All these elements must occur; extras allowed')
    sampling.add_argument('--group', type=int, help='Fix group token while generating remaining orbits')
    sampling.add_argument('--scaffold', help='Fixed-cell partial CIF in the supplied group setting')
    sampling.add_argument('--atom-map', help='JSON observed-CIF index to full-descriptor atom index')
    sampling.add_argument('--scaffold-tolerance', type=float, default=1e-4)
    sampling.add_argument('--descriptor')
    sampling.add_argument('--observations')
    sampling.add_argument('--prior', action='store_true', help='Legal random chart prior; bypass the trained continuous flow')
    sampling.add_argument('--guidance', type=float, default=1.0)
    sampling.add_argument('--temperature', type=float, default=1.0)
    sampling.add_argument('--steps', type=int, default=64)
    sampling.add_argument('--count', type=int, default=10)
    sampling.add_argument('--output', default='artifacts/samples')
    checking = subparsers.add_parser('audit')
    checking.add_argument('--records', required=True)
    checking.add_argument('--output', default='artifacts/audit.jsonl')
    preflight = subparsers.add_parser('preflight')
    preflight.add_argument('--config', required=True)
    preflight.add_argument('--train', nargs='+')
    preflight.add_argument('--val', nargs='+')
    preflight.add_argument('--test', nargs='+')
    preflight.add_argument('--output', required=True)
    parameters = subparsers.add_parser('parameters')
    parameters.add_argument('--config', required=True)
    summary = subparsers.add_parser('summarize')
    summary.add_argument('--samples', required=True)
    summary.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command in ('prepare', 'audit', 'preflight'):
        from unifiedgroupgen.tests.catalog import check_layer_catalog
        check_layer_catalog()
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    if args.command == 'prepare':
        config = yaml.safe_load(root_path(args.config).read_text(encoding='utf-8'))
        source = root_path(config['data_root'])/f'{args.split}.csv'
        prepared = config.get('prepared_root', f"artifacts/{config['dataset']}")
        output = artifact_path(args.output or f"{prepared}/{args.split}.jsonl")
        print(json.dumps(prepare_csv(source, output, config, args.limit, args.workers, args.resume)))
    elif args.command == 'train':
        from unifiedgroupgen.tests.legacy.training_entry import train_args
        train_args(args)
    elif args.command == 'sample':
        sample(args)
    elif args.command == 'audit':
        audit(args)
    elif args.command == 'preflight':
        from unifiedgroupgen.tests.preflight import audit_files
        config = yaml.safe_load(root_path(args.config).read_text(encoding='utf-8'))
        prepared = config.get('prepared_root', f"artifacts/{config['dataset']}")
        paths = {'train': [root_path(p) for p in (args.train or config.get('train_files') or [f"{prepared}/train.jsonl"])],
                 'val': [root_path(p) for p in (args.val or config.get('val_files') or [f"{prepared}/val.jsonl"])]}
        if args.test or config.get('test_files'):
            paths['test'] = [root_path(p) for p in (args.test or config['test_files'])]
        report = audit_files(paths, config, artifact_path(args.output))
        print(json.dumps({'passed': report['passed'], 'errors': len(report['errors']), 'output': args.output}))
        if not report['passed']:
            raise SystemExit(2)
    elif args.command == 'summarize':
        report = summarize_attempts(read_jsonl(root_path(args.samples)))
        artifact_path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(json.dumps(report))
    else:
        config = yaml.safe_load(root_path(args.config).read_text(encoding='utf-8'))
        proposal, flow = build_models(config, 'cpu')
        print(json.dumps({'proposal': sum(p.numel() for p in proposal.parameters()),
                          'flow': sum(p.numel() for p in flow.parameters()),
                          'total': sum(p.numel() for model in (proposal, flow) for p in model.parameters())}))


from unifiedgroupgen.pl_modules.factory import build_models
from unifiedgroupgen.run import fit_training_config
def sample(args):
    from unifiedgroupgen.tests.legacy.generation_before_entry.pipeline import sample as generate
    return generate(args, root=ROOT)

if __name__ == '__main__':
    main()
