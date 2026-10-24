"""CPU held-out descriptor-token and fixed-description flow diagnostics."""

import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
import argparse
import json
from pathlib import Path
import random
import sys


def descriptor_batch(model, descriptors, properties):
    import torch
    from torch.nn import functional as F
    sequences = [model.vocab.encode(d) for d in descriptors]
    width = max(len(s)-1 for s in sequences)
    ids = torch.full((len(sequences), width), model.vocab.index['PAD'], dtype=torch.long)
    targets = torch.full_like(ids, -100)
    legal = torch.ones(len(sequences), width, len(model.vocab.tokens), dtype=torch.bool)
    for i, (seq, descriptor) in enumerate(zip(sequences, descriptors)):
        n = len(seq)-1
        ids[i, :n], targets[i, :n] = torch.tensor(seq[:-1]), torch.tensor(seq[1:])
        legal[i, :n] = torch.stack([model.legality.allowed(seq[:j], None, descriptor.kind)
                                    for j in range(1, len(seq))])
    h = model.embedding(ids)+model.position(torch.arange(width))[None]
    h = h+model.condition.forward_batch(properties, h)[:, None]
    causal = torch.ones(width, width, dtype=torch.bool).triu(1)
    logits = model.head(model.transformer(h, mask=causal,
                                         src_key_padding_mask=ids.eq(model.vocab.index['PAD'])))
    logits = logits.masked_fill(~legal, -torch.inf)
    loss = F.cross_entropy(logits.transpose(1, 2), targets, reduction='none')
    predictions = logits.argmax(-1)
    rows = []
    for i, seq in enumerate(sequences):
        n = len(seq)-1
        parts = {k: {'nll_sum': 0., 'tokens': 0, 'correct': 0, 'contribution': 0.}
                 for k in ('G', 'W', 'Z', 'STOP')}
        for j, token in enumerate(seq[1:]):
            token_type = model.vocab.tokens[token].split(':')[0]
            p = parts[token_type]
            p['nll_sum'] += float(loss[i, j]); p['tokens'] += 1
            p['correct'] += int(predictions[i, j] == token)
            p['contribution'] += float(loss[i, j])/n
        rows.append({'descriptor_loss': float(loss[i, :n].mean()),
                     'teacher_forced_sequence_correct': bool(predictions[i, :n].eq(targets[i, :n]).all()),
                     'parts': parts})
    return rows


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--threads', type=int, default=1)
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1:
        parser.error('Batch size and threads must be positive')
    root = args.project_root.resolve()
    sys.path[:0] = [str(root/'src'), str(root/'scripts')]
    import numpy as np
    import torch
    from evaluate_checkpoint import check_heldout_inputs
    from unifiedgroupgen.cli import build_models
    from unifiedgroupgen.conditioning import attach_dataset_domains
    from unifiedgroupgen.data import read_jsonl
    from unifiedgroupgen.flow import flow_loss_batch
    from unifiedgroupgen.symmetry import Descriptor, compile_descriptor
    from unifiedgroupgen.training import file_hash
    torch.set_num_threads(args.threads)
    random.seed(918); np.random.seed(918); torch.manual_seed(918)
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    config = saved['config']
    paths = [root/p for p in config.get('test_files', [config['prepared_root']+'/test.jsonl'])]
    audit = check_heldout_inputs(saved, paths)
    proposal, flow = build_models(config, 'cpu')
    proposal.load_state_dict(saved['proposal']); flow.load_state_dict(saved['flow'])
    proposal.eval(); flow.eval()
    scale = saved.get('runtime_options', {}).get('coordinate_length_scale', 1.)
    records = attach_dataset_domains([r for p in paths for r in read_jsonl(p)], config)
    allowed = set(config['allowed_elements'])
    args.output.mkdir(parents=True, exist_ok=True)
    rows, unsupported = [], []
    supported = []
    for record in records:
        unseen = sorted({o['atomic_number'] for o in record['descriptor']['orbits']}-allowed)
        if unseen:
            unsupported.append({'dataset': record['dataset'], 'id': record['id'], 'unseen_elements': unseen})
        else:
            supported.append(record)
    with (args.output/'records.jsonl').open('w', encoding='utf-8') as stream, torch.inference_mode():
        for start in range(0, len(supported), args.batch_size):
            batch = supported[start:start+args.batch_size]
            ds = [Descriptor.from_dict(r['descriptor']) for r in batch]
            cs = [compile_descriptor(d) for d in ds]
            endpoints = [torch.tensor(r['state'], dtype=torch.float32) for r in batch]
            props = [r['properties'] for r in batch]
            discrete = descriptor_batch(proposal, ds, props)
            continuous = flow_loss_batch(flow, cs, endpoints, props)
            for i, record in enumerate(batch):
                row = {'dataset': record['dataset'], 'id': record['id'], **discrete[i],
                       'lattice': float(continuous['lattice'][i]),
                       'coordinates': float(continuous['coordinates'][i])}
                row['coordinates_dimensionless'] = row['coordinates']/scale**2
                row['flow'] = row['lattice']+row['coordinates_dimensionless']
                stream.write(json.dumps(row, allow_nan=False)+'\n'); rows.append(row)
            print(json.dumps({'phase': 'heldout_components', 'completed': start+len(batch),
                              'supported': len(supported), 'requested': len(records)}), flush=True)
    def aggregate(selected):
        result = {'evaluated': len(selected)}
        for key in ('descriptor_loss', 'lattice', 'coordinates', 'coordinates_dimensionless', 'flow'):
            result[key] = sum(r[key] for r in selected)/len(selected) if selected else None
        result['teacher_forced_sequence_accuracy'] = (sum(r['teacher_forced_sequence_correct'] for r in selected)/len(selected)
                                                       if selected else None)
        result['tokens'] = {}
        for key in ('G', 'W', 'Z', 'STOP'):
            totals = {field: sum(r['parts'][key][field] for r in selected)
                      for field in ('nll_sum', 'tokens', 'correct', 'contribution')}
            result['tokens'][key] = {**totals,
                'mean_nll': totals['nll_sum']/totals['tokens'] if totals['tokens'] else None,
                'top1': totals['correct']/totals['tokens'] if totals['tokens'] else None}
        return result
    report = {'state': 'complete', 'requested': len(records), 'unsupported': unsupported,
              'checkpoint_sha256': file_hash(args.checkpoint), 'checkpoint_epoch_1based': saved['epoch']+1,
              'test_sha256': [file_hash(p) for p in paths], 'heldout_audit': audit,
              'coordinate_length_scale': scale, 'seed': 918, 'overall': aggregate(rows),
              'sources': {name: aggregate([r for r in rows if r['dataset'] == name])
                          for name in sorted({r['dataset'] for r in records})},
              'protocol': 'Full available properties; no composition/observations; teacher-forced token metrics; '
                          'flow uses ground-truth descriptions and one random path/time per record. '
                          'Not reconstruction success, property accuracy or stability.'}
    (args.output/'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    main()
