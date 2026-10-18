"""Verify the layer catalog and optionally existing checkpoint/data compatibility on CPU."""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))

import numpy as np
import torch
from unifiedgroupgen.cli import build_models, root_path
from unifiedgroupgen.conditioning import attach_dataset_domains
from unifiedgroupgen.representation_audit import compiled_chart
from unifiedgroupgen.symmetry import Descriptor
from unifiedgroupgen.symmetry_preflight import check_layer_catalog
from unifiedgroupgen.training import file_hash, joint_loss


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint')
    parser.add_argument('--records', nargs='+')
    parser.add_argument('--output', default='artifacts/symmetry_runtime.json')
    args = parser.parse_args()
    torch.set_num_threads(4)
    report = {'layer_catalog': check_layer_catalog(), 'records': [], 'checkpoint': None}
    print('[PASS] layer catalog: 80 groups / 477 single orbits', flush=True)
    saved = torch.load(root_path(args.checkpoint), map_location='cpu', weights_only=False) if args.checkpoint else None
    config = saved['config'] if saved else {}
    paths = args.records
    if paths is None:
        paths = []
        for split in ('train', 'val', 'test'):
            files = config.get(split+'_files')
            if files is None and config.get('prepared_root'):
                files = [str(Path(config['prepared_root'])/f'{split}.jsonl')]
            paths.extend(files or [])
    if saved is not None and not paths:
        raise ValueError('Checkpoint has no declared data paths; supply --records for forward compatibility checks')
    probes = {}
    for name in dict.fromkeys(paths):
        path = root_path(name)
        digest = file_hash(path)
        count = lg52c = 0
        with path.open(encoding='utf-8') as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                descriptor = Descriptor.from_dict(record['descriptor'])
                c = compiled_chart(descriptor)
                state = np.asarray(record['state'], dtype=float)
                if state.shape != (c.dof,) or not np.isfinite(state).all():
                    raise ValueError(f'Incompatible state: {path}, {record["id"]}')
                lg52c += descriptor.kind == 'layer' and descriptor.number == 52 and any(
                    orbit.letter == 'c' for orbit in descriptor.orbits)
                probes.setdefault(record['dataset'], record)
                count += 1
        if file_hash(path) != digest:
            raise ValueError(f'Data changed during verification: {path}')
        if not count:
            raise ValueError(f'Cannot verify an empty data file: {path}')
        report['records'].append({'path': str(path), 'count': count, 'sha256': digest, 'lg52_4c_records': lg52c})
        print(f'[PASS] {path.name}: {count} compatible states ({path.parent.name})', flush=True)
    if saved is not None:
        checkpoint_path = root_path(args.checkpoint)
        for split in ('train', 'val'):
            declared = config.get(split+'_files')
            if declared and saved.get(split+'_split_sha256') is not None:
                if [file_hash(root_path(p)) for p in declared] != saved[split+'_split_sha256']:
                    raise ValueError(f'Checkpoint {split} split hashes differ from existing data')
        proposal, flow = build_models(config, 'cpu')
        proposal.load_state_dict(saved['proposal'], strict=True)
        flow.load_state_dict(saved['flow'], strict=True)
        proposal.eval(); flow.eval()
        torch.manual_seed(513)
        losses = {}
        with torch.no_grad():
            for dataset, record in probes.items():
                values = joint_loss(proposal, flow, attach_dataset_domains([record], config), 0, 0)
                if any(not torch.isfinite(value).all() for value in values.values()):
                    raise ValueError(f'Nonfinite checkpoint forward for {dataset}')
                losses[dataset] = {key: float(value[0]) for key, value in values.items()}
        report['checkpoint'] = {'path': str(checkpoint_path), 'sha256': file_hash(checkpoint_path),
                                'epoch_index': saved['epoch'], 'strict_state_dict_load': True,
                                'proposal_parameters': sum(p.numel() for p in proposal.parameters()),
                                'flow_parameters': sum(p.numel() for p in flow.parameters()), 'forward_probes': losses}
        print('[PASS] checkpoint strict load and source forward probes', flush=True)
    report.update(passed=True, scope='Runtime catalog, finite state/schema and strict checkpoint/forward compatibility; '
                  'no data rewrite, retraining, full held-out loss or physical certification. '
                  'Historical checkpoints do not contain an operation-table fingerprint; '
                  'this does not establish bitwise reproduction of old sampling.')
    output = root_path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps({'passed': True, 'records': sum(r['count'] for r in report['records']), 'output': str(output)}))


if __name__ == '__main__':
    main()
