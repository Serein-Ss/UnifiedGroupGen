"""Scheduler-only recovery tests: no GPU work or real job submission."""
import pytest

from submit_files.hpc5090 import job_plan, read, write
from submit_files import recover_hpc5090 as recovery


@pytest.fixture
def cluster(tmp_path, monkeypatch):
    source = tmp_path/'source'
    (source/'submit_files').mkdir(parents=True)
    script = source/'submit_files/hpc5090.sbatch'; script.write_text('frozen source')
    campaign = {'name': 'fixture', 'folder': str(tmp_path), 'source': str(source),
                'revision': 'original-revision', 'legacy_models': ['joint_eg', 'joint_eh', 'mp20_u']}
    manifest = tmp_path/'campaign.json'; write(manifest, campaign)
    submitted = {row[0]: str(1000+i) for i, row in enumerate(job_plan(campaign['legacy_models']))}
    write(tmp_path/'submitted.json', submitted)
    for phase, model in [('train', 'joint_u_comp_source_control_v2'), ('legacy', 'joint_eg'),
                         ('legacy', 'joint_eh'), ('new', 'c2db51_u_comp_v2')]:
        write(tmp_path/'state'/f'{phase}_{model}.json', {'state': 'failed'})
    write(tmp_path/'state/train_c2db51_u_comp_v2.json', {'state': 'complete'})
    write(tmp_path/'state/legacy_mp20_u.json', {'state': 'complete'})
    live = {submitted['train:mp20_u_comp_v2']: 'RUNNING', submitted['train:joint_u_comp_v2']: 'RUNNING',
            submitted['new:mp20_u_comp_v2']: 'PENDING', submitted['new:joint_u_comp_v2']: 'PENDING'}
    queued, updated, exclusions = [], [], {}
    monkeypatch.setattr(recovery, 'active_jobs', lambda: dict(live))
    monkeypatch.setattr(recovery.shutil, 'which', lambda _: 'control')
    def output(command, **kwargs):
        if command[0] == 'control':
            return 'JobId='+command[4]+' ExcNodeList='+exclusions.get(command[4], 'node-5090-5')
        assert command[0] == 'yhbatch'
        assert kwargs['env']['UGG_GIT_COMMIT'] == campaign['revision']
        queued.append(command)
        job = str(3000+len(queued)); live[job] = 'PENDING'
        return job+';cluster\n'
    def update(command, **kwargs):
        assert command[:2] == ['control', 'update']
        job = command[2].split('=', 1)[1]
        assert live[job] == 'PENDING'
        exclusions[job] = command[3].split('=', 1)[1]
        updated.append(job)
    monkeypatch.setattr(recovery.subprocess, 'check_output', output)
    monkeypatch.setattr(recovery.subprocess, 'run', update)
    return manifest, submitted, live, queued, updated, exclusions, script


def test_recovery_preview_never_submits_or_updates_jobs(cluster):
    manifest, submitted, _, queued, updated, _, _ = cluster
    recovery.recover(manifest, 'node-5090-2')
    assert not queued and not updated
    assert read(manifest.parent/'submitted.json') == submitted


def test_recovery_preserves_running_work_and_restores_dependency(cluster):
    manifest, original, _, queued, updated, exclusions, script = cluster
    recovery.recover(manifest, 'node-5090-2', submit=True)
    assert len(queued) == 5
    assert set(updated) == {original['new:mp20_u_comp_v2'], original['new:joint_u_comp_v2']}
    assert all(nodes == 'node-5090-5,node-5090-2' for nodes in exclusions.values())
    current = read(manifest.parent/'submitted.json')
    for key in ('train:mp20_u_comp_v2', 'train:joint_u_comp_v2', 'train:c2db51_u_comp_v2', 'legacy:mp20_u'):
        assert current[key] == original[key]
    assert read(manifest.parent/'submitted.before_recovery.json') == original
    for command in queued:
        assert '--exclude=node-5090-2' in command
        assert str(script) in command
        assert '--chdir='+str(script.parents[1]) in command
    child = next(c for c in queued if c[-1] == 'joint_u_comp_source_control_v2' and c[c.index('--phase')+1] == 'new')
    assert '--dependency=afterok:'+current['train:joint_u_comp_source_control_v2'] in child
    assert script.read_text() == 'frozen source'
    # Recover the receipt->submitted update window without duplicating jobs.
    write(manifest.parent/'submitted.json', original)
    recovery.recover(manifest, 'node-5090-2', submit=True)
    assert len(queued) == 5
    assert read(manifest.parent/'submitted.json') == current


def test_recovery_does_not_duplicate_a_phase_that_is_still_active(cluster):
    manifest, original, live, queued, updated, _, _ = cluster
    live[original['legacy:joint_eg']] = 'RUNNING'
    recovery.recover(manifest, 'node-5090-2', submit=True)
    assert len(queued) == 4
    assert all(c[-1] != 'joint_eg' for c in queued)
    assert original['legacy:joint_eg'] not in updated


@pytest.fixture
def interrupted_evaluations(cluster):
    manifest, original, live, queued, updated, exclusions, script = cluster
    recovery.recover(manifest, 'node-5090-2', submit=True)
    folder = manifest.parent
    current = read(folder/'submitted.json')
    failed = [('legacy', 'joint_eg'), ('legacy', 'joint_eh'), ('new', 'c2db51_u_comp_v2')]
    for phase, model in failed:
        live.pop(current[phase+':'+model])
        write(folder/'state'/f'{phase}_{model}.json', {'state': 'failed', 'error': "'state'"})
        write(folder/'evaluation'/model/'pipeline.json', {
            'identity': {'source': 'original', 'checkpoint': model}, 'state': 'failed',
            'jobs': [{'name': 'finished', 'state': 'complete', 'outputs': {'sample': 'digest'}},
                     {'name': 'components', 'directory': 'original-output'}]})
    write(folder/'state/train_joint_u_comp_source_control_v2.json', {'state': 'running'})
    live[current['train:joint_u_comp_source_control_v2']] = 'RUNNING'
    return cluster, current, failed


def test_state_repair_only_requeues_failed_evaluations_and_preserves_identity(interrupted_evaluations):
    cluster, previous, failed = interrupted_evaluations
    manifest, _, live, queued, updated, _, source = cluster
    folder = manifest.parent
    pipelines = {model: (folder/'evaluation'/model/'pipeline.json').read_bytes() for _, model in failed}
    original_receipt = read(folder/'recovery.json')
    before_updates = list(updated)
    recovery.recover(manifest, 'node-5090-2', repair_state=True)
    assert len(queued) == 5 and updated == before_updates
    assert all((folder/'evaluation'/model/'pipeline.json').read_bytes() == data for model, data in pipelines.items())
    recovery.recover(manifest, 'node-5090-2', submit=True, repair_state=True)
    assert len(queued) == 8 and updated == before_updates
    current = read(folder/'submitted.json')
    changed = {key for key in current if current[key] != previous[key]}
    assert changed == {phase+':'+model for phase, model in failed}
    assert read(folder/'submitted.before_state_repair.json') == previous
    assert source.read_text() == 'frozen source'
    for phase, model in failed:
        pipeline = folder/'evaluation'/model/'pipeline.json'
        assert pipeline.with_name('pipeline.before_state_repair.json').read_bytes() == pipelines[model]
        expected = read(pipeline.with_name('pipeline.before_state_repair.json'))
        expected['jobs'][1]['state'] = 'pending'
        assert read(pipeline) == expected
        record = read(folder/'recovery.json')['jobs'][phase+':'+model]
        assert record['state_repair']
        assert record['previous_recovery'] == original_receipt['jobs'][phase+':'+model]
        live.pop(current[phase+':'+model])  # Even if another retry fails, do not loop automatically.
    write(folder/'submitted.json', previous)  # Receipt/map crash-window reconciliation.
    recovery.recover(manifest, 'node-5090-2', submit=True, repair_state=True)
    assert len(queued) == 8 and read(folder/'submitted.json') == current
    recovery.recover(manifest, 'node-5090-2', submit=True)
    assert len(queued) == 8


def test_state_repair_skips_active_evaluation(interrupted_evaluations):
    cluster, current, _ = interrupted_evaluations
    manifest, _, live, queued, _, _, _ = cluster
    live[current['legacy:joint_eg']] = 'RUNNING'
    pipeline = manifest.parent/'evaluation/joint_eg/pipeline.json'
    before = pipeline.read_bytes()
    recovery.recover(manifest, 'node-5090-2', submit=True, repair_state=True)
    assert len(queued) == 7 and pipeline.read_bytes() == before
    assert not pipeline.with_name('pipeline.before_state_repair.json').exists()


def test_state_repair_respects_worker_phase_lock(interrupted_evaluations):
    from filelock import FileLock, Timeout
    cluster, _, _ = interrupted_evaluations
    manifest, _, _, queued, _, _, _ = cluster
    folder = manifest.parent
    pipeline = folder/'evaluation/joint_eg/pipeline.json'
    before = pipeline.read_bytes()
    with FileLock(str(folder/'state/legacy_joint_eg.json')+'.lock', timeout=0):
        with pytest.raises(Timeout):
            recovery.recover(manifest, 'node-5090-2', submit=True, repair_state=True)
    assert len(queued) == 5 and pipeline.read_bytes() == before
