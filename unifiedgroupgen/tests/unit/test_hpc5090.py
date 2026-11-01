"""CPU tests for matched controls, queue dependencies, and lossless candidate reuse."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from submit_files.hpc5090 import TRAIN_MODELS, job_plan
from unifiedgroupgen import generate
from unifiedgroupgen.common.configuration import load_experiments
from unifiedgroupgen.common.structures import read_records, sha256, verified_prior_records


def test_matched_source_control_and_composition_training():
    entries = load_experiments('configs/experiments_hpc5090.yaml')
    for name in TRAIN_MODELS:
        config = entries[name]['config']
        assert config['composition_probability'] == .5
        assert config['observation_probability'] == 0
        assert config['precision'] == 'bf16'
        assert not any(not p.get('context') for p in config['properties'].values())
    first = dict(entries['joint_u_comp_v2']['config'])
    second = dict(entries['joint_u_comp_source_control_v2']['config'])
    assert first['properties'] == {} and not first.get('dataset_domains')
    assert second['dataset_domains'] == {'mp_20': 0, 'c2db_51': 1}
    for key in ('experiment_id', 'data_audit', 'input_contract', 'properties', 'dataset_domains', 'evaluation_domains'):
        first.pop(key, None); second.pop(key, None)
    assert first == second


def test_exact_composition_accepts_hydra_json_without_reducing_counts():
    from unifiedgroupgen.conditional.chemistry import integer_composition
    assert integer_composition('{"Si":2,"O":4}') == {14: 2, 8: 4}
    assert integer_composition('Si2O4') == {14: 2, 8: 4}
    with pytest.raises(ValueError, match='integer'):
        integer_composition('{"Si":1.5,"O":3}')


def test_design_script_composition_argument_passes_hydra_parser():
    import re
    from hydra import compose, initialize_config_dir
    from unifiedgroupgen.common.configuration import project_root
    from unifiedgroupgen.conditional.chemistry import integer_composition
    root = project_root()
    script = (root/'submit_files/design_composition.sbatch').read_text()
    template = re.search(r'"(generation\.composition=[^"]+)"', script).group(1)
    override = template.replace('$2', '{"Si":2,"O":4}')
    with initialize_config_dir(config_dir=str(root/'configs'), version_base='1.3'):
        cfg = compose(config_name='generate', overrides=[override])
    assert integer_composition(cfg.generation.composition) == {14: 2, 8: 4}


def test_frozen_sources_only_write_to_declared_data_and_result_mounts(tmp_path):
    from unifiedgroupgen.common.configuration import artifact_path
    frozen = tmp_path/'frozen'; frozen.mkdir()
    live = tmp_path/'live'; live.mkdir()
    for name in ('data', 'outputs', 'intermediate', 'wandb'):
        (live/name).mkdir()
        try:
            (frozen/name).symlink_to(live/name, target_is_directory=True)
        except OSError as error:
            pytest.skip('The local OS cannot create the Linux job symlinks: '+str(error))
        result = artifact_path(frozen/name/'new/result.json', frozen)
        assert result == live/name/'new/result.json'
    with pytest.raises(ValueError, match='inside UnifiedGroupGen'):
        artifact_path(live/'unrelated/result.json', frozen)


def test_queue_has_independent_legacy_and_four_training_pairs():
    plan = job_plan(['old'+str(i) for i in range(15)])
    assert len(plan) == 25
    ids = {j[0] for j in plan}
    assert all(set(j[3]) <= ids for j in plan)
    assert all(j[3] == ['prepare'] for j in plan if j[1] == 'legacy')
    assert all(j[3] == ['profile'] for j in plan if j[1] == 'train')
    assert all(j[3] == ['train:'+j[2]] for j in plan if j[1] == 'new')


def test_submit_queues_all_dependencies_once_without_real_scheduler(tmp_path, monkeypatch):
    from submit_files import hpc5090 as submitter
    import yaml
    config = tmp_path/'configs/evaluation_suite/main15_local.yaml'
    config.parent.mkdir(parents=True)
    legacy = ['old'+str(i) for i in range(15)]
    config.write_text(yaml.safe_dump({'models': legacy}))
    monkeypatch.setattr(submitter, '__file__', str(tmp_path/'submit_files/hpc5090.py'))
    monkeypatch.setattr(submitter.subprocess, 'run', lambda *a, **kw: None)
    def freeze(root, folder, revision):
        path = folder/'source'; path.mkdir(); return path
    monkeypatch.setattr(submitter, 'snapshot', freeze)
    queued = []
    def command(args, **kwargs):
        if args[0] == 'git':
            return 'a'*40
        assert args[0] == 'yhbatch'
        queued.append(args)
        return str(1000+len(queued)-1)+';cluster\n'
    monkeypatch.setattr(submitter.subprocess, 'check_output', command)
    submitter.submit('queue_fixture')
    assert len(queued) == 25
    assigned = {job[0]: str(1000+i) for i, job in enumerate(job_plan(legacy))}
    for args, job in zip(queued, job_plan(legacy)):
        if job[3]:
            assert '--dependency=afterok:'+':'.join(assigned[key] for key in job[3]) in args
        assert '--kill-on-invalid-dep=yes' in args and '--export=ALL' in args
    submitter.submit('queue_fixture')
    assert len(queued) == 25


def test_calibration_rejects_oom_and_preserves_completed_parameters(tmp_path, monkeypatch):
    from submit_files.hpc5090_profile import CANDIDATES, profile, select_trial
    from submit_files.hpc5090 import write
    valid = {'state': 'pass', 'records_per_second': 10.}
    assert select_trial([{'state': 'oom'}, valid]) == valid
    with pytest.raises(RuntimeError):
        select_trial([{'state': 'memory_margin_failed'}])
    selected = {k: {'batch_size': 256} for k in CANDIDATES}
    write(tmp_path/'profile.json', {'revision': 'fixed', 'selected': selected})
    monkeypatch.setattr('torch.cuda.is_available', lambda: pytest.fail('Completed calibration must not restart GPU work'))
    profile({'folder': str(tmp_path), 'revision': 'fixed'})


def test_fully_reused_legacy_model_uses_more_cpu_metric_workers(tmp_path):
    from submit_files.hpc5090_eval import arguments
    from submit_files.hpc5090 import write
    campaign = {'folder': str(tmp_path), 'name': 'fixture', 'metric_workers': 3}
    before = arguments(campaign, 'mp20_u', 'legacy')
    assert (before.threads, before.metric_workers) == (3, 3)
    write(tmp_path/'legacy_budget.json', {'models': {'mp20_u': {'new_attempts': 0}}})
    after = arguments(campaign, 'mp20_u', 'legacy')
    assert (after.threads, after.metric_workers) == (1, 6)


@pytest.mark.parametrize('batch_size', [1, 3])
@pytest.mark.parametrize('partial', [False, True])
def test_csp_reuses_exact_candidate_slots_including_failures(tmp_path, monkeypatch, batch_size, partial):
    descriptor = {'kind': 'space', 'number': 1, 'orbits': [{'letter': 'a', 'atomic_number': 6}]}
    reference = tmp_path/'reference.jsonl'
    reference.write_text(json.dumps({'id': 'case', 'dataset': 'fixture', 'descriptor': descriptor})+'\n')
    prior = tmp_path/'prior.jsonl'
    candidates = [0, 2] if partial else [0, 1, 2]
    prior.write_text(''.join(json.dumps({'index': i, 'case_id': 'case', 'candidate': k,
        'status': 'failed', 'reason': 'original-'+str(k), 'seed': 100+k, 'wall_seconds': 4.})+'\n'
        for i, k in enumerate(candidates)))
    spec = {'samples': str(prior), 'samples_sha256': sha256(prior), 'records': len(candidates)}
    job = {'task': 'csp', 'dataset': 'fixture', 'case_ids': ['case'], 'k': 3, 'kind': 'space',
           'properties': {}, 'prior_samples': spec}
    output = tmp_path/'output'; output.mkdir()
    calls = []
    def sample(*args, **kwargs):
        calls.append(True)
        raise ValueError('new-slot')
    monkeypatch.setattr(generate, 'sample_orbits', sample)
    flow = SimpleNamespace(inference_precision='fp32')
    generate.generate_records(None, flow, {'test_files': [str(reference)]}, job, output,
                              'cpu', 142, 2, 0., batch_size=batch_size)
    rows = read_records(output/'samples.jsonl')
    assert [r['candidate'] for r in rows] == [0, 1, 2]
    assert len(calls) == int(partial)
    for k in candidates:
        assert rows[k]['reason'] == 'original-'+str(k)
        assert rows[k]['seed'] == 100+k
        assert rows[k]['source_wall_seconds'] == 4.
    assert json.loads((output/'sampling.status.json').read_text())['reused'] == len(candidates)
    before = (output/'samples.jsonl').read_bytes()
    generate.generate_records(None, flow, {'test_files': [str(reference)]}, job, output,
                              'cpu', 142, 2, 0., batch_size=batch_size)
    assert (output/'samples.jsonl').read_bytes() == before


def test_duplicate_prior_slots_rejected(tmp_path):
    path = tmp_path/'prior.jsonl'
    path.write_text(''.join(json.dumps({'index': i, 'case_id': 'r', 'candidate': 0, 'status': 'failed'})+'\n'
                            for i in range(2)))
    job = {'task': 'csp', 'prior_samples': {'samples': str(path), 'samples_sha256': sha256(path), 'records': 2}}
    with pytest.raises(ValueError, match='duplicate'):
        generate.prior_candidates(job, SimpleNamespace(inference_precision='fp32'))


def test_importing_an_already_imported_record_preserves_source_timing(tmp_path):
    path = tmp_path/'previous.jsonl'
    path.write_text(json.dumps({'index': 0, 'status': 'failed', 'source_wall_seconds': 3.5})+'\n')
    rows = verified_prior_records({'samples': str(path), 'samples_sha256': sha256(path), 'records': 1})
    assert rows[0]['source_wall_seconds'] == 3.5


def test_shared_reference_cache_builds_once_for_concurrent_workers(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from unifiedgroupgen.evaluate import reference_metrics
    class Bank:
        builds = 0
        def identity(self):
            return {'fixture': 'immutable'}
        @property
        def rows(self):
            self.builds += 1
            return []
    bank = Bank(); barrier = Barrier(2)
    def call():
        barrier.wait()
        return reference_metrics(bank, 'space', False, tmp_path/'cache')
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(call); second = pool.submit(call)
        assert first.result() == second.result()
    assert bank.builds == 1


def test_streaming_controller_overlaps_metrics_and_skips_verified_resume(tmp_path, monkeypatch):
    from submit_files import hpc5090_eval as controller
    from unifiedgroupgen.common.structures import write_json
    root = tmp_path/'evaluation'
    jobs = [{'name': 'dng_'+str(i), 'task': 'dng', 'model': 'fixture', 'checkpoint_sha256': 'fixture'}
            for i in range(2)]
    monkeypatch.setattr(controller, 'arguments', lambda *a: SimpleNamespace(output=root, threads=3))
    monkeypatch.setattr(controller.suite, 'build_plan', lambda *a: (jobs, 2, {'fixture': True}))
    monkeypatch.setattr(controller.suite, 'job_directory', lambda j: tmp_path/'generate'/j['name'])
    events = []
    def generate_job(job, args, target, entry, state, pipeline, env):
        events.append(('generate', job['name']))
        (target/'samples.jsonl').write_text('{"index":0,"status":"failed"}\n')
        entry.update(state='generated', generation={'seconds': 1., 'outputs': controller.suite.output_hashes(target)})
        write_json(pipeline, state)
    monkeypatch.setattr(controller.suite, 'run_job', generate_job)
    class Worker:
        def __init__(self, command, **kwargs):
            self.spec = command[-1]; self.finished = False; self.checks = 0
            spec = controller.read(self.spec)
            events.append(('metrics_start', spec['job']['name']))
            directory = tmp_path/'metrics'/spec['job']['name']; directory.mkdir(parents=True)
            (directory/'summary.json').write_text('{}')
            controller.write(controller.Path(self.spec).with_suffix('.done.json'), {
                'metric_directory': str(directory), 'outputs': controller.suite.output_hashes(directory),
                'evaluation_seconds': 2.})
        def poll(self):
            self.checks += 1
            if self.checks >= 2 and not self.finished:
                self.wait()
            return 0 if self.finished else None
        def wait(self):
            self.finished = True
            events.append(('metrics_finish', self.spec))
            return 0
    monkeypatch.setattr(controller.subprocess, 'Popen', Worker)
    monkeypatch.setattr('unifiedgroupgen.common.reports.campaign_summary', lambda *a: None)
    campaign = {'name': 'fixture', 'metric_workers': 3}
    controller.evaluate_model(campaign, 'fixture', 'legacy')
    assert events.index(('generate', 'dng_1')) < next(i for i, e in enumerate(events) if e[0] == 'metrics_finish')
    assert controller.read(root/'pipeline.json')['state'] == 'complete'
    before = list(events)
    controller.evaluate_model(campaign, 'fixture', 'legacy')
    assert events == before
    (tmp_path/'metrics/dng_0/summary.json').write_text('{"changed":true}')
    with pytest.raises(ValueError, match='missing or modified'):
        controller.evaluate_model(campaign, 'fixture', 'legacy')


def test_streaming_real_cpu_generation_metrics_and_resume(tmp_path, monkeypatch):
    """Exercise real subprocess entry points with tiny weights and no external predictor."""
    import sys
    import torch
    from omegaconf import OmegaConf
    from submit_files import hpc5090_eval as controller
    from unifiedgroupgen.pl_modules.factory import build_models
    config = {'experiment_id': 'fixture', 'dataset': 'fixture', 'kind': 'space', 'groups': [['space', 2]],
              'properties': {}, 'composition_probability': .5, 'observation_probability': 0.,
              'allowed_elements': [6], 'max_atoms': 4, 'max_orbits': 2, 'require_data_audit': False,
              'model': {'hidden': 16, 'proposal_layers': 1, 'flow_layers': 1, 'heads': 2, 'frequencies': 2}}
    record = {'id': 'r', 'dataset': 'fixture', 'descriptor': {'kind': 'space', 'number': 2,
              'orbits': [{'letter': 'i', 'atomic_number': 6}]}, 'state': [0.]*6+[.2, .3, .4], 'properties': {}}
    reference = tmp_path/'reference.jsonl'; reference.write_text(json.dumps(record)+'\n')
    for split in ('train', 'val', 'test'):
        config[split+'_files'] = [str(reference)]
    proposal, flow = build_models(config, 'cpu')
    weights = tmp_path/'weights/model.best.pt'; weights.parent.mkdir()
    repo = Path(__file__).resolve().parents[3]
    frozen = tmp_path/'frozen_source'; frozen.mkdir()
    try:
        for name, target in {'configs': repo/'configs', 'outputs': weights.parent, 'intermediate': tmp_path}.items():
            (frozen/name).symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip('The local OS cannot create the Linux job symlinks: '+str(error))
    monkeypatch.setenv('UGG_PROJECT_ROOT', str(frozen))
    torch.save({'config': config, 'proposal': proposal.state_dict(), 'flow': flow.state_dict()}, weights)
    config_path = tmp_path/'config.yaml'; OmegaConf.save(OmegaConf.create(config), config_path)
    catalog = tmp_path/'legacy_catalog.yaml'
    OmegaConf.save(OmegaConf.create({'experiments': {'fixture': {'config': str(config_path), 'checkpoint': str(weights)}}}), catalog)
    campaign = {'folder': str(tmp_path), 'name': 'fixture', 'metric_workers': 2}
    original_arguments = controller.arguments
    def options(*args):
        result = original_arguments(*args)
        result.device = 'cpu'; result.threads = 1
        result.reuse_manifest = None; result.conditional_enabled = False; result.no_fingerprints = True
        result.count = result.csp_cases = result.fixed_descriptor_k = 1
        result.csp_k = result.steps = 2; result.per_group = 0
        result.tasks = ['components', 'dng', 'csp', 'fixed_descriptor']
        return result
    monkeypatch.setattr(controller, 'arguments', options)
    worker = tmp_path/'fixture_worker.py'
    worker.write_text('import sys\nfrom pathlib import Path\n'+f'sys.path.insert(0, {str(repo)!r})\n'
        'from submit_files import hpc5090_eval as c\noriginal=c.arguments\n'
        'def options(*args):\n a=original(*args)\n a.conditional_enabled=False\n a.no_fingerprints=True\n return a\n'
        'c.arguments=options\nc.metric_worker(Path(sys.argv[1]))\n')
    original_popen = controller.subprocess.Popen
    def launch(command, **kwargs):
        if '-m' in command and 'submit_files.hpc5090_eval' in command:
            command = [sys.executable, '-u', str(worker), command[-1]]
        return original_popen(command, **kwargs)
    monkeypatch.setattr(controller.subprocess, 'Popen', launch)
    controller.evaluate_model(campaign, 'fixture', 'legacy')
    pipeline = tmp_path/'evaluation/fixture/pipeline.json'
    state = controller.read(pipeline)
    assert state['state'] == 'complete' and len(state['jobs']) == 4
    ledgers = {str(p): sha256(p) for p in weights.parent.rglob('samples.jsonl')}
    assert len(ledgers) == 3
    controller.evaluate_model(campaign, 'fixture', 'legacy')
    assert all(sha256(p) == digest for p, digest in ledgers.items())
