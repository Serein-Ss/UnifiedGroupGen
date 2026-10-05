"""Full held-out loss evaluation; unsupported labels remain in the coverage denominator."""

import argparse
import json
import random
import numpy as np
import torch
from unifiedgroupgen.cli import artifact_path, build_models, root_path
from unifiedgroupgen.data import read_jsonl
from unifiedgroupgen.conditioning import attach_dataset_domains
from unifiedgroupgen.training import batch_indices, file_hash, joint_loss


def check_heldout_inputs(checkpoint, paths):
    """Formal evaluation must cover the audited test files and the matching training run."""
    config = checkpoint['config']
    if not config.get('require_data_audit'):
        return None
    audit_path = root_path(config['data_audit'])
    report = json.loads(audit_path.read_text(encoding='utf-8'))
    if not report.get('passed'):
        raise ValueError('Held-out data audit has not passed')
    for split in ('train', 'val'):
        if checkpoint.get(f'{split}_split_sha256') != [item['sha256'] for item in report['inputs'][split]]:
            raise ValueError('Checkpoint training/validation files differ from the held-out audit')
    actual = sorted((str(path.resolve()), file_hash(path)) for path in paths)
    expected = sorted((str(root_path(item['path'])), item['sha256']) for item in report['inputs']['test'])
    if actual != expected:
        raise ValueError('Formal held-out evaluation requires exactly the complete audited test files')
    return {'path': str(audit_path), 'sha256': file_hash(audit_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--records', required=True, nargs='+')
    parser.add_argument('--output', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    torch.set_num_threads(4)
    random.seed(918); np.random.seed(918); torch.manual_seed(918)
    checkpoint_path = root_path(args.checkpoint)
    records_paths = [root_path(path) for path in args.records]
    checkpoint = torch.load(checkpoint_path, map_location=args.device, weights_only=False)
    heldout_audit = check_heldout_inputs(checkpoint, records_paths)
    config = checkpoint['config']
    proposal, flow = build_models(config, args.device)
    proposal.load_state_dict(checkpoint['proposal']); flow.load_state_dict(checkpoint['flow'])
    proposal.eval(); flow.eval()
    records = attach_dataset_domains([r for path in records_paths for r in read_jsonl(path)], config)
    allowed = set(config['allowed_elements'])
    supported, unsupported, sources = [], [], {}
    for record in records:
        dataset = record.get('dataset', 'unspecified')
        source = sources.setdefault(dataset, {'requested_records': 0, 'evaluated_records': 0,
                                             'unsupported_records': 0, 'property_label_counts': {},
                                             'loss_totals': {key: 0.0 for key in ('descriptor', 'flow', 'lattice', 'coordinates')}})
        source['requested_records'] += 1
        for name, specification in config.get('properties', {}).items():
            if not specification.get('context'):
                source['property_label_counts'][name] = source['property_label_counts'].get(name, 0)+int(record['properties'].get(name) is not None)
        unseen = sorted({o['atomic_number'] for o in record['descriptor']['orbits']}-allowed)
        if unseen:
            unsupported.append({'id': record['id'], 'unseen_elements': unseen,
                                **({'dataset': dataset} if 'dataset' in record else {})})
            source['unsupported_records'] += 1
        else:
            supported.append(record)
            source['evaluated_records'] += 1
    totals = {key: 0.0 for key in ('descriptor', 'flow', 'lattice', 'coordinates')}
    batches = batch_indices(supported, config['batch_size'], config['edge_budget'], shuffle=False)
    with torch.no_grad():
        for step, indices in enumerate(batches, 1):
            losses = joint_loss(proposal, flow, [supported[i] for i in indices], 0, 0)
            for key in totals:
                if not torch.isfinite(losses[key]).all():
                    raise FloatingPointError(f'Nonfinite held-out {key} loss at step {step}')
                totals[key] += float(losses[key].sum())
            positions = {}
            for position, index in enumerate(indices):
                positions.setdefault(supported[index].get('dataset', 'unspecified'), []).append(position)
            for dataset, selected in positions.items():
                for key in totals:
                    sources[dataset]['loss_totals'][key] += float(losses[key][selected].sum())
            if step % 25 == 0 or step == len(batches):
                print(json.dumps({'event': 'test_progress', 'step': step, 'steps': len(batches)}), flush=True)
    for source in sources.values():
        source['label_coverage'] = source['evaluated_records']/source['requested_records']
        source['loss_on_supported_records'] = {
            key: value/source['evaluated_records'] if source['evaluated_records'] else None
            for key, value in source.pop('loss_totals').items()}
    report = {'requested_records': len(records), 'evaluated_records': len(supported),
              'label_coverage': len(supported)/len(records) if records else 0,
              'unsupported_records': unsupported,
              'loss_on_supported_records': {key: value/len(supported) if supported else None for key, value in totals.items()},
              'checkpoint_epoch': checkpoint['epoch'], 'checkpoint_sha256': file_hash(checkpoint_path),
              'test_split_sha256': [file_hash(path) for path in records_paths], 'seed': 918,
              'heldout_audit': heldout_audit, 'source_breakdown': sources,
              'scope': 'Held-out conditional flow/descriptor losses; not property accuracy or stability certification'}
    artifact_path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
