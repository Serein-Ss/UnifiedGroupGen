"""Compare the current implementation with the downloaded main15 server source.

Only bounded software checks are run. No formal campaign or model training is resumed.
"""
import argparse
import gc
import hashlib
import importlib
import json
import random
import sys
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import yaml

from unifiedgroupgen.common.configuration import load_experiments, project_root
from unifiedgroupgen.common.source import file_hash
from unifiedgroupgen.pl_modules.factory import build_models, load_model_checkpoint
from unifiedgroupgen.pl_modules.objectives.joint import joint_loss
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor
from unifiedgroupgen.group_constraints.validation import to_structure
from unifiedgroupgen.generation.flow import sample_flow


def seed(value=142):
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)


def tensor_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def archive_package(archive, output):
    target = output/'server_source'/'server_original_ugg'
    target.mkdir(parents=True, exist_ok=True)
    hashes = {}
    with ZipFile(archive) as source:
        prefix = 'UnifiedGroupGen/src/unifiedgroupgen/'
        for name in source.namelist():
            relative = name.removeprefix(prefix)
            if name.startswith(prefix) and '/' not in relative and name.endswith(('.py', '.json')):
                payload = source.read(name)
                (target/relative).write_bytes(payload)
                hashes[name] = hashlib.sha256(payload).hexdigest()
        configs = {name.stem: yaml.safe_load(source.read(str(name).replace('\\', '/')))
                   for name in map(Path, source.namelist())
                   if str(name).replace('\\', '/').startswith('UnifiedGroupGen/configs/main15/')
                   and name.suffix == '.yaml'}
    sys.path.insert(0, str(target.parent))
    return importlib.import_module('server_original_ugg.cli'), configs, hashes


def compare(archive, output, steps):
    root = project_root()
    output.mkdir(parents=True, exist_ok=True)
    original, definitions, source_hashes = archive_package(archive, output)
    old_training = importlib.import_module('server_original_ugg.training')
    old_symmetry = importlib.import_module('server_original_ugg.symmetry')
    old_flow = importlib.import_module('server_original_ugg.flow')
    old_evaluation = importlib.import_module('server_original_ugg.evaluation')
    records = {}
    for dataset in ('mp_20', 'c2db_51'):
        with (root/'data/dataset/full_verified_v1'/dataset/'test.jsonl').open(encoding='utf-8') as stream:
            records[dataset] = json.loads(next(stream))
    inventory = json.loads((root/'outputs/model_inventory.json').read_text())
    catalog = load_experiments('configs/experiments.yaml', root)
    before = json.loads((root/'old_results/_audit/retained_models_before.json').read_text())
    changed = [name for name, digest in before['files'].items() if file_hash(root/name) != digest]
    assert not changed, changed
    probes, configurations, strict_loads = [], {}, []
    gradients = []
    full_sampling = {'mp20_u', 'c2db51_u', 'joint_u', 'joint_all'}
    gradient_models = full_sampling | {'mp20_all', 'c2db51_all'}
    for entry in inventory['models']:
        name = entry['name']
        configurations[name] = catalog[name]['config'] == definitions[name]
        assert configurations[name], name
        checkpoint = root/entry['directory']/'model.best.pt'
        saved = load_model_checkpoint(checkpoint)
        old_models = original.build_models(saved['config'], 'cpu')
        new_models = build_models(saved['config'], 'cpu')
        for models in (old_models, new_models):
            for model, key in zip(models, ('proposal', 'flow')):
                model.load_state_dict(saved[key], strict=True)
                model.eval()
            models[1].checkpoint_layers = saved['runtime_options']['checkpoint_layers']
            models[1].edge_chunk_size = saved['runtime_options']['edge_chunk_size']
        strict_loads.append(name)
        domains = (list(saved['config']['dataset_domains']) if saved['config'].get('dataset_domains')
                   else [saved['config']['dataset']])
        for dataset in domains:
            record = records[dataset]
            properties = dict(record['properties'])
            if saved['config'].get('dataset_domains'):
                properties['source_domain'] = saved['config']['dataset_domains'][dataset]
            descriptor = Descriptor.from_dict(record['descriptor'])
            old_descriptor = old_symmetry.Descriptor.from_dict(record['descriptor'])
            chart, old_chart = compile_descriptor(descriptor), old_symmetry.compile_descriptor(old_descriptor)
            assert (chart.lattice_dof, chart.coordinate_dof, chart.num_atoms) == (
                old_chart.lattice_dof, old_chart.coordinate_dof, old_chart.num_atoms)
            state = torch.tensor(record['state'], dtype=torch.float32)
            row = {'model': name, 'dataset': dataset, 'record_id': record['id'],
                   'lattice_dof': chart.lattice_dof, 'coordinate_dof': chart.coordinate_dof}
            with torch.no_grad():
                seed(); old_loss = old_models[0].loss(old_descriptor, properties)
                seed(); new_loss = new_models[0].loss(descriptor, properties)
                torch.testing.assert_close(new_loss, old_loss, rtol=0, atol=0)
                old_velocity = old_models[1](old_chart, state, .4, properties)
                new_velocity = new_models[1](chart, state, .4, properties)
                torch.testing.assert_close(new_velocity, old_velocity, rtol=0, atol=0)
                row.update(descriptor_loss=float(new_loss), descriptor_max_difference=float((new_loss-old_loss).abs().max()),
                           velocity_max_difference=float((new_velocity-old_velocity).abs().max()))
                if name in full_sampling:
                    guidance = 2. if name.endswith('_all') else 0.
                    before_sample = old_flow.sample_flow(old_models[1], old_chart, properties, steps=steps,
                        guidance=guidance, generator=torch.Generator().manual_seed(142))
                    after_sample = sample_flow(new_models[1], chart, properties, steps=steps,
                        guidance=guidance, generator=torch.Generator().manual_seed(142))
                    differences = {}
                    for key in ('state', 'lattice', 'coordinates'):
                        torch.testing.assert_close(after_sample[key], before_sample[key], rtol=0, atol=0)
                        differences[key] = float((after_sample[key]-before_sample[key]).abs().max())
                    old_structure = old_evaluation.to_structure(old_chart, before_sample)
                    new_structure = to_structure(chart, after_sample)
                    old_cif, new_cif = old_structure.to(fmt='cif'), new_structure.to(fmt='cif')
                    assert old_cif == new_cif
                    row.update(sampling_steps=steps, guidance=guidance, generation_max_differences=differences,
                               cif_identical=True, cif_sha256=hashlib.sha256(new_cif.encode()).hexdigest())
            if name in gradient_models:
                batch = [{**record, 'properties': properties}]
                for model in (*old_models, *new_models): model.train()
                seed(); old_parts = old_training.joint_loss(*old_models, batch, 0., 0., 3.)
                old_total = old_parts['descriptor'].mean()+old_parts['flow'].mean()
                old_total.backward()
                expected = {prefix+'.'+key: tensor_hash(parameter.grad) if parameter.grad is not None else None
                            for prefix, model in zip(('proposal', 'flow'), old_models)
                            for key, parameter in model.named_parameters()}
                seed(); new_parts = joint_loss(*new_models, batch, 0., 0., 3.)
                new_total = new_parts['descriptor'].mean()+new_parts['flow'].mean()
                new_total.backward()
                for key in old_parts:
                    torch.testing.assert_close(new_parts[key], old_parts[key], rtol=0, atol=0)
                actual = {prefix+'.'+key: tensor_hash(parameter.grad) if parameter.grad is not None else None
                          for prefix, model in zip(('proposal', 'flow'), new_models)
                          for key, parameter in model.named_parameters()}
                assert actual == expected, name
                gradients.append({'model': name, 'dataset': dataset, 'losses_identical': True,
                                  'gradient_tensors_identical': sum(v is not None for v in actual.values())})
                for model in (*old_models, *new_models): model.zero_grad(set_to_none=True); model.eval()
            probes.append(row)
        print(json.dumps({'model': name, 'strict_load': True, 'paired_checks': 'passed'}), flush=True)
        del old_models, new_models, saved
        gc.collect()
    return {'passed': True, 'scope': 'Current code versus original downloaded server implementation; bounded FP32 CPU checks',
            'archive': str(archive), 'archive_bytes': archive.stat().st_size, 'original_source_sha256': source_hashes,
            'preserved_files': len(before['files']), 'changed_files': changed,
            'original_configurations_equal': configurations, 'strictly_loaded_models': strict_loads,
            'paired_real_record_probes': probes, 'loss_and_gradient_probes': gradients,
            'training_trajectory_identity': 'Not guaranteed: batch RNG, validation precision and checkpoint protocol differ',
            'limits': ['No full retraining', 'Bounded structure probes do not establish generation benchmark quality',
                       'Full sampling paired only for three bases and joint multi-property model',
                       'Historical .pt optimizer state is not a native Lightning resume checkpoint']}


def descriptor_probes(archive, output):
    """Free rule selection with real trained bases and a multi-property model."""
    root = project_root()
    original, _, _ = archive_package(archive, output)
    inventory = json.loads((root/'outputs/model_inventory.json').read_text())
    results = []
    for entry in inventory['models']:
        if entry['name'] not in {'mp20_u', 'c2db51_u', 'joint_u', 'joint_all'}:
            continue
        saved = load_model_checkpoint(root/entry['directory']/'model.best.pt')
        old_models, new_models = original.build_models(saved['config'], 'cpu'), build_models(saved['config'], 'cpu')
        for models in (old_models, new_models):
            models[0].load_state_dict(saved['proposal'], strict=True)
            models[0].eval()
        domains = saved['config'].get('dataset_domains') or {saved['config']['dataset']: None}
        for dataset, domain in domains.items():
            with (root/'data/dataset/full_verified_v1'/dataset/'test.jsonl').open(encoding='utf-8') as stream:
                record = json.loads(next(stream))
            properties = dict(record['properties'])
            if domain is not None: properties['source_domain'] = domain
            guidance = 2. if entry['name'].endswith('_all') else 0.
            descriptors = []
            for model in (old_models[0], new_models[0]):
                seed()
                descriptor = model.sample(properties=properties, kind=record['descriptor']['kind'],
                    guidance=guidance, generator=torch.Generator().manual_seed(142))
                descriptors.append(descriptor.to_dict())
            assert descriptors[0] == descriptors[1], (entry['name'], dataset)
            results.append({'model': entry['name'], 'dataset': dataset, 'guidance': guidance,
                            'identical': True, 'sampled_descriptor': descriptors[1]})
        del old_models, new_models, saved
        gc.collect()
    return {'passed': True, 'free_descriptor_sampling': results,
            'scope': 'Free group/element/orbit sampling; fixed kind and existing source context retained'}


def training_protocol_probes(archive, output):
    """Measure protocol differences without fitting or modifying any model."""
    from unifiedgroupgen.pl_data.dataset import batch_indices
    root = project_root()
    archive_package(archive, output)
    old_batches = importlib.import_module('server_original_ugg.training').batch_indices
    records = []
    for domain in ('mp_20', 'c2db_51'):
        with (root/'data/dataset/full_verified_v1'/domain/'test.jsonl').open(encoding='utf-8') as stream:
            records.append(json.loads(next(stream)))
    rows = [{**records[i % 2], 'id': 'probe'+str(i)} for i in range(40)]
    random.seed(42)
    old_first = old_batches(rows, 4, 400000)
    for _ in range(80): random.random()  # joint_loss keeps two draws per record even with probability zero.
    old_second = old_batches(rows, 4, 400000)
    new_first = batch_indices(rows, 4, 400000, rng=random.Random(42))
    new_second = batch_indices(rows, 4, 400000, rng=random.Random(43))
    result = {'batch_order': {'epoch0_equal': old_first == new_first, 'epoch1_equal': old_second == new_second,
        'old_epoch1_first_batches': old_second[:3], 'native_epoch1_first_batches': new_second[:3],
        'scope': '40 real-record replicas and probability-zero RNG consumption; not full training'}}
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        from pytorch_lightning.plugins.precision import MixedPrecision
        saved = load_model_checkpoint(root/'outputs/mp20_u/26-10-30-12-50-31/model.best.pt')
        proposal, flow = build_models(saved['config'], 'cuda')
        proposal.load_state_dict(saved['proposal']); flow.load_state_dict(saved['flow'])
        proposal.eval(); flow.eval()
        flow.checkpoint_layers, flow.edge_chunk_size = False, 32768
        def objective():
            seed(918)
            return joint_loss(proposal, flow, [records[0]], 0., 0., 3.)
        with torch.no_grad():
            before = objective()
            with MixedPrecision('bf16-mixed', 'cuda').forward_context():
                after = objective()
        result['validation_precision'] = {'device': torch.cuda.get_device_name(0), 'record': records[0]['id'],
            'old_policy': 'FP32 validation', 'new_policy': 'Lightning BF16 mixed validation',
            'old_losses': {k: float(v.mean()) for k, v in before.items()},
            'new_losses': {k: float(v.mean()) for k, v in after.items()},
            'max_absolute_differences': {k: float((before[k]-after[k]).abs().max()) for k in before},
            'scope': 'Same trained model, record and RNG; only validation autocast context differs'}
    else:
        result['validation_precision'] = {'verified_from_code': True, 'empirical_check': 'BF16 GPU unavailable'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--descriptors-only', action='store_true')
    parser.add_argument('--training-protocol-only', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.training_protocol_only:
        result = training_protocol_probes(args.archive.resolve(), args.output.resolve())
        (args.output/'training_protocol_differences.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps(result, indent=2))
        return
    if args.descriptors_only:
        result = descriptor_probes(args.archive.resolve(), args.output.resolve())
        (args.output/'descriptors_comparison.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
        print(json.dumps({'passed': True, 'free_descriptor_probes': len(result['free_descriptor_sampling'])}))
        return
    result = compare(args.archive.resolve(), args.output.resolve(), args.steps)
    (args.output/'server_comparison.json').write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'passed': result['passed'], 'models': len(result['strictly_loaded_models']),
                      'probes': len(result['paired_real_record_probes'])}), flush=True)


if __name__ == '__main__':
    main()
