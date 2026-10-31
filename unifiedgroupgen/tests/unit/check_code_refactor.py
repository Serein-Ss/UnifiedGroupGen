"""Paired SG/LG probe for code-only refactors; no material-quality claim."""
import argparse
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


def normalize(value):
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in value.items() if k not in ('identity', 'efficiency')}
    if isinstance(value, list):
        return [normalize(v) for v in value]
    return value


def run(output, fixtures):
    from unifiedgroupgen.pl_modules.factory import build_models

    from unifiedgroupgen.pl_modules.objectives.joint import joint_loss
    from unifiedgroupgen.generation.flow import sample_flow
    from unifiedgroupgen.group_constraints.compiler import Descriptor, OrbitSpec, compile_descriptor
    from unifiedgroupgen.tests.legacy.evaluation_before_tasks import generation_cli as cli
    from unifiedgroupgen.evaluate import evaluate

    random.seed(142); np.random.seed(142); torch.manual_seed(142); torch.set_num_threads(1)
    output.mkdir(parents=True, exist_ok=True); fixtures.mkdir(parents=True, exist_ok=True)
    descriptors = [Descriptor('space', 2, (OrbitSpec('i', 6),)),
                   Descriptor('layer', 78, (OrbitSpec('a', 42), OrbitSpec('f', 16)))]
    config = {'groups': [['space', 2], ['layer', 78]], 'kind': 'space',
              'properties': {'band_gap': {'type': 'scalar'}}, 'allowed_elements': [6, 16, 42],
              'max_atoms': 16, 'max_orbits': 3, 'composition_probability': 0., 'observation_probability': 0.,
              'model': {'hidden': 16, 'proposal_layers': 1, 'flow_layers': 1, 'heads': 4, 'frequencies': 2}}
    records = [{'id': str(i), 'dataset': d.kind, 'descriptor': d.to_dict(), 'properties': {'band_gap': 1.},
                'state': ([.1]*6+[.2, .3, .4] if i == 0 else [-.035, 1.56356])}
               for i, d in enumerate(descriptors)]
    proposal, flow = build_models(config, 'cpu')
    initial = {'config': config, 'proposal': proposal.state_dict(), 'flow': flow.state_dict()}
    torch.save(initial, output/'initial.pt')
    losses = joint_loss(proposal, flow, records, .5, .25, 3.)
    total = losses['descriptor'].mean()+losses['flow'].mean(); total.backward()
    tensors = {'loss/'+k: v.detach() for k, v in losses.items()}
    for prefix, model in (('proposal', proposal), ('flow', flow)):
        tensors.update({prefix+'/'+k: p.grad.detach().clone() if p.grad is not None else None
                        for k, p in model.named_parameters()})
    proposal.eval(); flow.eval()
    semantics = {}
    for i, descriptor in enumerate(descriptors):
        compiled = compile_descriptor(descriptor)
        with torch.no_grad():
            result = sample_flow(flow, compiled, {'band_gap': 1.}, 3, 2., None, None,
                                 torch.Generator().manual_seed(142))
        tensors['sample/'+descriptor.kind] = result['state'].detach()
        ref = fixtures/(descriptor.kind+'.jsonl')
        content = json.dumps(records[i])+'\n'
        if ref.exists():
            assert ref.read_text(encoding='utf-8') == content
        else:
            ref.write_text(content, encoding='utf-8')
        desc = fixtures/(descriptor.kind+'.descriptor.json')
        desc.write_text(json.dumps(descriptor.to_dict()), encoding='utf-8')
        folder = output/descriptor.kind
        sys.argv = ['ugg-generate', '--checkpoint', str(output/'initial.pt'), '--descriptor', str(desc),
                    '--kind', descriptor.kind, '--count', '2', '--steps', '3', '--seed', '142',
                    '--threads', '1', '--output', str(folder)]
        cli.main()
        sample_file = folder/'samples.jsonl'
        generated = [json.loads(line) for line in sample_file.read_text(encoding='utf-8').splitlines()]
        for row in generated:
            row['case_id'] = str(i)
        generated.append({'index': 2, 'case_id': str(i), 'status': 'failed', 'reason': 'Explicit fixture failure'})
        sample_file.write_text(''.join(json.dumps(row)+'\n' for row in generated), encoding='utf-8')
        for task in ('csp', 'dng'):
            rows, report, reference = evaluate(sample_file, task, descriptor.kind, folder/task, [ref], [ref],
                                               use_fingerprints=False, reidentify=False)
            semantics[descriptor.kind+'/'+task] = normalize({'rows': rows, 'report': report, 'reference': reference})
    torch.save(tensors, output/'tensors.pt')
    (output/'semantics.json').write_text(json.dumps(semantics, sort_keys=True, allow_nan=False), encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, required=True)
    parser.add_argument('--compare', type=Path)
    args = parser.parse_args()
    run(args.output, args.fixtures)
    if args.compare:
        before = torch.load(args.compare/'tensors.pt', weights_only=False)
        after = torch.load(args.output/'tensors.pt', weights_only=False)
        assert before.keys() == after.keys()
        for key in before:
            if before[key] is None:
                assert after[key] is None, key
            else:
                torch.testing.assert_close(before[key], after[key], rtol=0, atol=0, msg=key)
        assert json.loads((args.compare/'semantics.json').read_text(encoding='utf-8')) == json.loads((args.output/'semantics.json').read_text(encoding='utf-8'))
        print(json.dumps({'passed': True, 'tensor_checks': len(before), 'evaluation_cases': 4}))


if __name__ == '__main__':
    main()
