"""Paired held-out CSP diagnostics for the frozen MP-20 EquiformerV2 run."""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import importlib.util
from itertools import islice
import json
import os
from pathlib import Path, PurePosixPath
import random
import sys
from time import perf_counter

import yaml

BASE = Path(__file__).resolve().parents[1]
CONFIG = BASE / 'configs/evaluation/mp20_eqv2_csp_diagnosis.yaml'
SAMPLER = BASE / 'unifiedgroupgen/generation/unified/sampling.py'
ARMS = ('blind_wide', 'blind_top4_repeat', 'reference_group',
        'reference_orbits', 'reference_lattice')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def save(path, value):
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def digest(path):
    sha = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            sha.update(block)
    return sha.hexdigest()


def audited_inputs(root, cfg):
    campaign = root / 'intermediate/hpc_runs' / cfg['campaign']
    checkpoint = root / 'outputs' / cfg['model'] / cfg['campaign'] / 'best.ckpt'
    if (read(campaign / 'campaign.json')['revision'] != cfg['revision']
            or read(checkpoint.parent / 'model.status.json')['state'] != 'complete'):
        raise ValueError('Expected the completed frozen training campaign')
    source = campaign / 'source'
    frozen_sampler = source / SAMPLER.relative_to(BASE)
    text = frozen_sampler.read_text(encoding='utf-8')
    if hashlib.sha256(text.encode()).hexdigest() != cfg['frozen_sampler_text_sha256']:
        raise ValueError('Unexpected frozen sampler source')
    audit = read(campaign / 'audits' / (cfg['model'] + '.json'))
    if not audit['passed'] or audit['records']['test'] != 9046 or len(audit['inputs']['test']) != 1:
        raise ValueError('Expected the audited MP-20 TEST split')
    entry = audit['inputs']['test'][0]
    cache = root / 'data/dataset/mp_20' / PurePosixPath(entry['path']).name
    for path, expected in ((checkpoint, cfg['checkpoint_sha256']), (cache, entry['sha256'])):
        if digest(path) != expected:
            raise ValueError(f'Unexpected artifact hash: {path}')
    return source, checkpoint, cache, entry['sha256']


def load_conditioned_sampler():
    # Import only this file from the diagnostics bundle. Its model and state imports
    # resolve against the verified frozen training source.
    spec = importlib.util.spec_from_file_location('mp20_csp_diagnostic_sampler', SAMPLER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.sample_batch


def draw_ranked(case, settings, model, prior):
    import torch
    from unifiedgroupgen.group_constraints.group_prior import NoFeasibleGroup, sample_group

    result = {}
    generator = torch.Generator(device='cpu').manual_seed(settings['seed'])
    for rank in range(settings['candidates']):
        try:
            group = sample_group(prior, generator, kind=case.kind, chemistry=case.chemistry,
                                 budget=model.event_budget, max_atoms=model.max_atoms,
                                 max_orbits=model.max_orbits, composition_rank=rank)
        except NoFeasibleGroup:
            break
        if group.number in result:  # The prior wraps once all feasible ranks are exhausted.
            break
        result[group.number] = group
    return tuple(result.values())


def select_group(arm, ranked, candidate, top_groups):
    if not ranked:
        return None
    if arm == 'blind_wide':
        return ranked[candidate] if candidate < len(ranked) else None
    return ranked[candidate % min(top_groups, len(ranked))]


def initialized_orbits(target, chemistry, model, generator, lock_metric):
    import torch
    from unifiedgroupgen.group_constraints.flow_state import apply, initial
    from unifiedgroupgen.group_constraints.state import JointAction

    like = next(model.parameters())
    state = initial(target.spec, generator, model.event_budget, chemistry, like.device, like.dtype)
    for letter, element in zip(target.letters, target.elements):
        state = apply(state, JointAction('birth', letter=letter, element=element), generator, model.birth_scale)
    state = apply(state, JointAction('freeze'))
    if lock_metric:
        state = replace(state, metric=target.metric.to(device=like.device, dtype=like.dtype))
    expected = dict(chemistry.counts)
    if any(n != 0 for _, n in chemistry.remaining(state.spec, state.letters, state.elements)):
        raise ValueError(f'Reference orbit template does not reconstruct inventory {expected}')
    if not torch.isfinite(state.metric).all():
        raise ValueError('Nonfinite diagnostic lattice')
    return state


def prepare_cases(cache, cfg):
    from unifiedgroupgen.evaluation.joint_csp import prepare_csp_manifest
    from unifiedgroupgen.pl_data.cache import records_from_cache

    pool = sorted(records_from_cache(cache), key=lambda row: row['id'])
    if (len(pool) != 9046 or len({row['id'] for row in pool}) != 9046
            or any(row.get('dataset') != 'mp_20' or row['descriptor']['kind'] != 'space' for row in pool)):
        raise ValueError('Unexpected TEST cache contents')
    selected = sorted(random.Random(cfg['seed']).sample(pool, cfg['cases']), key=lambda row: row['id'])
    return selected, prepare_csp_manifest(selected, cfg['candidates'])


def sample_arm(model, sample_batch, arm, items, manifest, cfg, path):
    import torch
    from unifiedgroupgen.joint import evaluation_record

    n = len(items)
    manifest_sha = manifest.digest
    like = next(model.parameters())
    duration, failures = {}, Counter()
    lattice_locked = arm == 'reference_lattice'
    orbit_fixed = arm in ('reference_orbits', 'reference_lattice')

    def sync():
        if like.device.type == 'cuda':
            torch.cuda.synchronize(like.device)

    with path.open('x', encoding='utf-8') as stream:
        sync()
        started = perf_counter()
        for candidate in range(cfg['candidates']):
            for begin in range(0, n, cfg['batch_size']):
                requests, rows = [], {}
                for i in range(begin, min(begin + cfg['batch_size'], n)):
                    case, label, ranked, target = items[i]
                    seed = cfg['seed'] + cfg['candidates'] * i + candidate
                    index = candidate * n + i
                    if arm.startswith('blind_'):
                        choice = select_group(arm, ranked, candidate, cfg['top_groups'])
                        spec, number = choice if choice is not None else (None, None)
                    else:
                        spec, number = target.spec, label
                    missing = set(dict(case.chemistry.counts)) - set(model.allowed_elements)
                    if missing:
                        reason = 'elements_outside_training_vocabulary'
                    elif sum(dict(case.chemistry.counts).values()) > model.max_atoms:
                        reason = 'inventory_exceeds_max_atoms'
                    elif spec is None:
                        reason = 'no_feasible_group' if not ranked else 'insufficient_distinct_ranked_groups'
                    elif orbit_fixed and (len(target.letters) > model.max_orbits
                                          or len(target.letters) + 1 > model.event_budget):
                        reason = 'reference_orbits_exceed_sampler_budget'
                    else:
                        reason = None
                    identity = dict(index=index, candidate=candidate, case_id=case.case_id,
                                    dataset=case.dataset, kind=case.kind, seed=seed,
                                    case_manifest_sha256=manifest_sha, requested_group=number,
                                    reference_group=label, diagnostic_arm=arm)
                    if reason:
                        rows[i] = dict(identity, status='failed', reason=reason, accepted=False)
                        continue
                    generator = torch.Generator(device=like.device).manual_seed(seed)
                    start = initialized_orbits(target, case.chemistry, model, generator, lattice_locked) if orbit_fixed else None
                    requests.append((i, spec, generator, case.chemistry, start, identity))
                if requests:
                    samples = sample_batch(model, [r[1] for r in requests], steps=cfg['steps'],
                                           generators=[r[2] for r in requests], chemistries=[r[3] for r in requests],
                                           initial_states=[r[4] for r in requests] if orbit_fixed else None,
                                           lock_metric=lattice_locked)
                    if len(samples) != len(requests):
                        raise RuntimeError('Sampler did not return every requested candidate')
                    for (i, _, _, _, _, identity), sample in zip(requests, samples):
                        row = evaluation_record(sample, identity['index'], identity['seed'],
                                                requested_group=identity['requested_group'])
                        rows[i] = {**row, **identity}
                for i in sorted(rows):
                    row = rows[i]
                    stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
                    if row['status'] != 'generated':
                        reason = row.get('reason', '').split(':', 1)[0]
                        if row['status'] == 'rejected' and reason == 'complete':
                            check = row.get('validation', {})
                            reason = ('short_contact' if check.get('distance_valid') is False else 'output_rejected')
                        failures[reason or row['status']] += 1
            stream.flush()
            if candidate == 0 or candidate + 1 == cfg['candidates']:
                sync()
                duration[candidate + 1] = perf_counter() - started
                print(arm, f'K={candidate + 1}', 'sampling_seconds', round(duration[candidate + 1], 2), flush=True)
    return duration, dict(failures)


def evaluate_arm(path, references, manifest, duration):
    from unifiedgroupgen.common.structures import iter_records
    from unifiedgroupgen.evaluation.joint_csp import evaluate_csp

    summary, hits = {}, {}
    for k in (1, 20):
        started = perf_counter()
        ledger = islice(iter_records(path), len(references)*k)
        metrics = evaluate_csp(ledger, references, (k,), manifest)
        score = metrics['summary'][str(k)]
        if score['attempted_candidates'] != score['planned_candidates']:
            raise ValueError(f'Incomplete K={k} candidate ledger')
        unsuccessful = score['planned_candidates'] - score['candidate_outcomes'].get('generated', 0)
        no_output = sum(not any(c['status'] == 'generated' for c in case['candidates'])
                        for case in metrics['cases'])
        summary[str(k)] = dict(score, candidate_failure_rate=unsuccessful / score['planned_candidates'],
                               case_no_output_rate=no_output / score['cases'],
                               sampling_seconds=duration[k], sampling_seconds_per_case=duration[k] / score['cases'],
                               matching_seconds=perf_counter() - started)
        hits[k] = {case['case_id'] for case in metrics['cases'] if any(c['matched'] for c in case['candidates'])}
        save(path.parent / f'metrics_k{k:02d}.json', dict(metrics, summary={str(k): summary[str(k)]}))
    return summary, hits


def run(cfg, source, checkpoint_path, cache, cache_sha, out):
    import torch
    from unifiedgroupgen.common.data_utils import target_from_record
    from unifiedgroupgen.group_constraints.operations import compile_operations
    from unifiedgroupgen.pl_modules.factory import build_joint_model
    from unifiedgroupgen.pl_modules.training_utils import strict_fp32
    from unifiedgroupgen.pl_modules.unified.module import load_joint_checkpoint

    if not torch.cuda.is_available():
        raise RuntimeError('Run the diagnostic on a scheduler-allocated CUDA GPU')
    torch.set_num_threads(1)
    strict_fp32(dict(strict_fp32=True, precision='fp32'))
    checkpoint = load_joint_checkpoint(checkpoint_path, 'cpu')
    config = checkpoint['config']
    if (config['generation_task'] != 'csp' or config['model']['backbone'] != 'equiformer_v2'
            or config.get('dataset') != 'mp_20'):
        raise ValueError('Expected an MP-20 EquiformerV2 CSP checkpoint')
    model = build_joint_model(config, 'cuda:0').eval()
    model.load_state_dict(checkpoint['model'], strict=True)
    refs, manifest = prepare_cases(cache, cfg)
    sample_batch = load_conditioned_sampler()
    compiled = {}
    items = []
    for case, record in zip(manifest.cases, refs):
        ranked = draw_ranked(case, cfg, model, config['group_prior'])
        choices = []
        for group in ranked:
            number = group.number
            if number not in compiled:
                compiled[number] = compile_operations(group.operations, group.kind) if config['model'].get('operation_compiler') else group
            choices.append((compiled[number], number))
        target = target_from_record(record, operation_compiler=config['model'].get('operation_compiler', False)).state
        label = record['descriptor']['number']
        items.append((case, label, tuple(choices), target))

    out.mkdir(parents=True, exist_ok=False)
    save(out / 'cases.json', manifest.to_dict())
    save(out / 'inputs.json', dict(settings=cfg, split='test', test_sha256=cache_sha, source=str(source),
                                   sampler_sha256=digest(SAMPLER), launcher_sha256=digest(Path(__file__)),
                                   gpu=torch.cuda.get_device_name(0), torch_version=torch.__version__, manifest_sha256=manifest.digest,
                                   ranked_groups={case.case_id: [number for _, number in choices]
                                                  for case, _, choices, _ in items}))
    report, matched = {}, {}
    for arm in ARMS:
        folder = out / arm
        folder.mkdir()
        ledger = folder / 'samples.jsonl'
        timings, reasons = sample_arm(model, sample_batch, arm, items, manifest, cfg, ledger)
        summary, matched[arm] = evaluate_arm(ledger, refs, manifest, timings)
        report[arm] = dict(metrics=summary, rejected_or_failed=reasons)
        save(folder / 'summary.json', report[arm])
    save(out / 'summary.json', report)
    comparisons = {}
    for left, right in (('blind_wide', 'blind_top4_repeat'), ('blind_wide', 'reference_group'),
                        ('blind_top4_repeat', 'reference_group'), ('reference_group', 'reference_orbits'),
                        ('reference_orbits', 'reference_lattice')):
        comparisons[left + '_to_' + right] = {str(k): dict(
            new_matches=len(matched[right][k] - matched[left][k]),
            lost_matches=len(matched[left][k] - matched[right][k])) for k in (1, 20)}
    save(out / 'paired_changes.json', comparisons)
    print('CSP_DIAGNOSTIC_COMPLETE', out, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('preflight', 'run'))
    parser.add_argument('--root', type=Path, default=Path.cwd())
    parser.add_argument('--config', type=Path, default=CONFIG)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    if not (cfg['candidates'] == 20 and cfg['top_groups'] == 4 and 0 < cfg['cases'] <= 9046
            and cfg['batch_size'] > 0 and cfg['steps'] > 0):
        parser.error('Expected 20 candidates, four repeated groups, and positive sampling controls')
    root = args.root.resolve()
    source, checkpoint, cache, cache_sha = audited_inputs(root, cfg)
    os.environ['UGG_PROJECT_ROOT'] = str(root)
    sys.path.insert(0, str(source))
    if args.action == 'preflight':
        refs, manifest = prepare_cases(cache, cfg)
        load_conditioned_sampler()
        print('CSP_DIAGNOSTIC_PREFLIGHT_OK', len(refs), manifest.digest)
        return
    if not args.output:
        parser.error('run requires --output')
    out = args.output.resolve()
    if not out.is_relative_to(root / 'intermediate') or out.exists():
        parser.error('--output must be a new directory inside the project intermediate directory')
    run(cfg, source, checkpoint, cache, cache_sha, out)


if __name__ == '__main__':
    main()
