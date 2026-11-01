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
