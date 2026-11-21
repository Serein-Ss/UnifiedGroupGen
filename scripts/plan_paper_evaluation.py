"""Print cache readiness and full-model or paired evaluation commands."""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from submit_files.joint_flow5090 import catalog
from unifiedgroupgen.common.configuration import configuration_root, load_yaml, project_root
from unifiedgroupgen.experiment_config import compose_experiment, training_definition
from unifiedgroupgen.pl_data.cache import expected_tensor_cache


def cache_status(entry, root, domain=None):
    config = training_definition(compose_experiment(entry['experiment']))
    source = load_yaml(config['source_configs'][domain or entry['dataset']], configuration_root())
    graph = {key: config['model'][key] for key in ('cutoff', 'max_neighbors', 'max_images')}
    result = {}
    for split in ('train', 'val', 'test'):
        path, identity, _ = expected_tensor_cache(source, split, root, config['model']['operation_compiler'], graph)
        sidecar = path.with_suffix('.config.json')
        metadata = json.loads(sidecar.read_text(encoding='utf-8')) if sidecar.is_file() else {}
        result[split] = dict(path=path.relative_to(root).as_posix(),
                             ready=path.is_file() and metadata.get('identity') == identity)
    return result


def evaluation_plan(data, name, model, seed, caches, candidates=None, group=None, *, main=False, domain=None, split=None):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', name) or not 0 <= seed < 2**32:
        raise ValueError('Pass a campaign name and an unsigned 32-bit training seed')
    paper = data['paper']
    if model not in paper['main_models' if main else 'models']:
        raise ValueError('Select an experiment from the corresponding full-model or paired matrix')
    if main and seed != paper['seeds'][0]:
        raise ValueError('The first full-model campaign uses seed 42')
    entry = data['experiments'][model]
    training_dataset, task = entry['dataset'], entry['task']
    dataset = domain or training_dataset
    if dataset not in ('mp_20', 'c2db_51') or training_dataset not in (dataset, 'mp20_c2db51'):
        raise ValueError('Select a held-out domain from the experiment training datasets')
    kind = {'mp_20': 'space', 'c2db_51': 'layer'}[dataset]
    options = paper['evaluation']
    attempts = options['dng_attempts']
    if candidates is not None and (task != 'csp' or candidates < 1):
        raise ValueError('--candidates is a positive CSP-only budget')
    if group is not None and (task != 'dng' or not 1 <= group <= {'space': 230, 'layer': 80}[kind]):
        raise ValueError('--group requires DNG and a valid group number for the model family')
    k = options['csp_candidates'] if candidates is None else candidates
    split = split or ('val' if main else 'test')
    if split not in ('val','test') or (split == 'val' and not main):
        raise ValueError('Validation previews require the full-model campaign')
    heldout, training = caches[split]['path'], caches['train']['path']
    manifest_label = f'{dataset}_{split}' if main else dataset
    manifest = f'intermediate/hpc_runs/{name}/evaluation/{manifest_label}_k{k}_cases.json'
    prepare = None
    if task == 'csp':
        prepare = ['python','-m','unifiedgroupgen.joint','prepare-csp',
                   '--references',heldout,'--candidates',str(k),'--output',manifest]
        label = f'paper_csp_k{k}'
    else:
        label = 'paper_dng_free' if group is None else f'paper_dng_group{group}'
    runs = []
    if main:
        label = f'{label}_{dataset}_{split}'
    for mode in (('joint',) if main else paper['ablations']):
        directory = f'outputs/{model}/{name}' if main else f'outputs/{model}_{mode}_s{seed}/{name}'
        checkpoint = f"{directory}/{'best.ckpt' if main else 'model.pt'}"
        output = f'{directory}/generate/{label}/samples.jsonl'
        metric = f'{directory}/evaluate/{label}'
        sample = ['python','-m','unifiedgroupgen.generate','--checkpoint',checkpoint,'--output',output,
                  '--seed',str(seed+options['seed_offset']),'--steps',str(options['sampling_steps']),
                  '--batch-size',str(options['sampling_batch_size']),'--device','cuda:0']
        if task == 'csp':
            sample += ['--case-manifest',manifest]
            ks = sorted({v for v in (1,5,k) if v <= k})
            assess = ['python','-m','unifiedgroupgen.joint','evaluate-csp',
                      '--samples',output,'--references',heldout,'--case-manifest',manifest,
                      '--output',f'{metric}/metrics.json','--ks',*map(str,ks)]
        else:
            sample += ['--kind',kind,'--count',str(attempts)]
            if group is not None:
                sample += ['--group',str(group)]
            assess = ['python','-m','unifiedgroupgen.evaluate',
                      f'evaluation.samples={output}',f'evaluation.kind={kind}',f'evaluation.dataset={dataset}',
                      f'evaluation.reference_files=[{heldout}]',f'evaluation.novelty_files=[{training}]',
                      f'evaluation.expected_attempts={attempts}',f'evaluation.output={metric}']
        runs.append(dict(ablation=mode,completion_status=f'{directory}/model.status.json',
                         sample=sample,evaluate=assess))
    return dict(model=model,train_seed=seed,cache_ready=all(item['ready'] for item in caches.values()),
                caches=caches,prepare_once=prepare,runs=runs,
                training_dataset=training_dataset,evaluation_dataset=dataset,evaluation_split=split)


def main():
    data = catalog()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--name',required=True,help='Paper training campaign name')
    parser.add_argument('--main',action='store_true',help='Use the first full-model campaign checkpoint layout')
    parser.add_argument('--model',required=True,choices=data['paper']['main_models'])
    parser.add_argument('--domain',choices=['mp_20','c2db_51'],help='Preview one held-out domain of a mixed model')
    parser.add_argument('--seed',type=int,default=data['paper']['seeds'][0])
    parser.add_argument('--split',choices=['val','test'],help='Main defaults to VAL for architecture screening, paired defaults to TEST')
    parser.add_argument('--candidates',type=int)
    parser.add_argument('--group',type=int)
    args = parser.parse_args()
    if args.model not in data['paper']['main_models' if args.main else 'models']:
        parser.error('Mixed experiment previews require --main')
    entry = data['experiments'][args.model]
    mixed = entry['dataset'] == 'mp20_c2db51'
    if args.domain and not (mixed or args.domain == entry['dataset']):
        parser.error('Domain must be among the training datasets')
    if mixed and args.group is not None and args.domain is None:
        parser.error('Provide --domain when specifying a group for a mixed DNG model')
    domains = [args.domain] if args.domain else (['mp_20','c2db_51'] if mixed else [entry['dataset']])
    root = project_root()
    plans = {}
    for domain in domains:
        caches = cache_status(entry,root,domain)
        plans[domain] = evaluation_plan(data,args.name,args.model,args.seed,caches,args.candidates,args.group,
                                        main=args.main,domain=domain,split=args.split)
    result = next(iter(plans.values())) if len(plans) == 1 else dict(
        model=args.model,cache_ready=all(plan['cache_ready'] for plan in plans.values()),by_domain=plans)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
