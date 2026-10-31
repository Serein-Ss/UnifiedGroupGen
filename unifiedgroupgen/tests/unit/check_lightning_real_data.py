"""Bounded real MP20/C2DB engineering check; never a research training run."""
import hashlib
from itertools import islice
import json
from pathlib import Path
import random
from types import SimpleNamespace

import numpy as np
import torch

from unifiedgroupgen.pl_modules.factory import build_models

from unifiedgroupgen.conditional.properties import attach_dataset_domains
from unifiedgroupgen.group_constraints.validation import to_structure, validate
from unifiedgroupgen.generation.flow import sample_flow
from unifiedgroupgen.pl_data.datamodule import CrystalDataModule
from unifiedgroupgen.run import run_lightning
from unifiedgroupgen.tests.legacy.paths import project_root, resolve_path
from unifiedgroupgen.common.configuration import RunSettings
from unifiedgroupgen.run import training_output
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor


def main():
    root = project_root()
    torch.set_num_threads(2)
    random.seed(42); np.random.seed(42); torch.manual_seed(42)
    source = root/'outputs/joint_all/26-10-30-12-50-31/model.best.pt'
    config = torch.load(source, map_location='cpu', weights_only=False)['config']
    config.update(precision='bf16' if torch.cuda.is_available() else 'fp32', early_stopping_patience=0,
                  engineering_scope='One-epoch Lightning integration diagnostic; fresh weights; not a research model')
    rows = {}
    for split, count in [('train', 2), ('val', 1), ('test', 1)]:
        selected = []
        for dataset in ('mp_20', 'c2db_51'):
            with resolve_path(f'artifacts/full_verified_v1/{dataset}/{split}.jsonl').open(encoding='utf-8') as stream:
                selected += [json.loads(line) for line in islice(stream, count)]
        rows[split] = attach_dataset_domains(selected, config)
    base = RunSettings.load(root, mode='offline')
    settings = RunSettings(root, root/'outputs/tests', base.wandb_dir, 'offline')
    output = training_output(settings, 'lightning_real_data')
    for split, selected in rows.items():
        (output.parent/(split+'_subset.jsonl')).write_text(
            ''.join(json.dumps(row)+'\n' for row in selected), encoding='utf-8')
    proposal, flow = build_models(config, 'cpu')
    args = SimpleNamespace(device='cuda' if torch.cuda.is_available() else 'cpu', resume=None, epochs=1, seed=42,
                           initialize_from=None, checkpoint=False, edge_chunk_size=32768,
                           batch_size=8, edge_budget=400000, coordinate_length_scale=3., validation_interval=5)
    trainer = run_lightning(args, config, rows['train'], rows['val'], proposal, flow,
                            [output.parent/'train_subset.jsonl'], [output.parent/'val_subset.jsonl'], output, settings,
                            finish_logging=False)
    data = CrystalDataModule(rows['train'], rows['val'], trainer.datamodule.runtime, test_records=rows['test'])
    test = trainer.test(trainer.lightning_module, datamodule=data, verbose=False)
    import wandb
    wandb.finish()
    flow.eval()
    samples = []
    for index, record in enumerate(rows['val']):
        chart = compile_descriptor(Descriptor.from_dict(record['descriptor']))
        with torch.inference_mode():
            generated = sample_flow(flow, chart, record['properties'], steps=2)
        cif = to_structure(chart, generated).to(fmt='cif')
        samples.append({'index': index, 'status': 'generated', 'descriptor': record['descriptor'],
                        'cif': cif, 'cif_sha256': hashlib.sha256(cif.encode()).hexdigest(),
                        'validation': validate(chart, generated, isolate_identifier=True)})
    (output.parent/'samples.jsonl').write_text(''.join(json.dumps(s)+'\n' for s in samples), encoding='utf-8')
    report = {'scope': config['engineering_scope'], 'torch': torch.__version__,
              'training_device': args.device, 'sampling_device': str(next(flow.parameters()).device),
              'precision': config['precision'],
              'train_records': len(rows['train']), 'validation_records': len(rows['val']), 'test_records': len(rows['test']),
              'parameters': sum(p.numel() for model in (proposal, flow) for p in model.parameters()),
              'test_loss': test, 'generated_fixed_descriptors': len(samples), 'sampling_steps': 2,
              'group_operation_closure': [s['validation']['group_operation_closure'] for s in samples],
              'output': str(output), 'wandb_mode': 'offline'}
    (output.parent/'engineering_report.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
