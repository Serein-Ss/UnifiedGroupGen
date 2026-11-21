"""Validate catalog labels and public output without fitting a model."""
import csv
import json
from types import SimpleNamespace

import pytest
import torch

from unifiedgroupgen.common.data_utils import target_from_record
from unifiedgroupgen.generation.trajectory import TrajectoryWriter
from unifiedgroupgen.generation.types import JointSample
from unifiedgroupgen.group_constraints.flow_state import from_state
from unifiedgroupgen.group_constraints.wyckoff import TABLE, orbit_labels
from unifiedgroupgen.joint import evaluation_record
from unifiedgroupgen.pl_data.datamodule import _collate
from unifiedgroupgen.pl_data.dataset import CrystalDataset
from unifiedgroupgen.pl_modules.unified.module import JointTrainingArtifacts


def test_catalog_csv_covers_each_setting_without_duplicate_labels():
    with TABLE.open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    assert {(r['kind'], int(r['group'])) for r in rows} == {
        (kind, group) for kind, count in (('space', 230), ('layer', 80), ('rod', 75))
        for group in range(1, count+1)
    }
    for field in ('wyckoff_letter', 'template_id'):
        assert len({(r['kind'], r['group'], r[field]) for r in rows}) == len(rows)


@pytest.mark.parametrize('identity,kind,group,orbits,compact,labels', [
    ('mp-2657', 'space', 136, [('a',22),('f',8)], [1.2415616737794481,-0.020637444068401813,0.69542], ['2a','4f']),
    ('1MoS2-2', 'layer', 72, [('a',42),('c',16)], [-0.0314347550712232,1.5872887295579794], ['1a','2c']),
])
def test_standard_labels_on_cache_records_logs_and_generation(tmp_path, identity, kind, group, orbits, compact, labels):
    record = dict(id=identity, descriptor=dict(kind=kind,number=group,
        orbits=[dict(letter=l,atomic_number=z) for l,z in orbits]), state=compact)
    target = target_from_record(record, operation_compiler=True)
    state = from_state(target.state, frozen=True)
    assert list(state.letters) == ['o000', 'o004']
    assert orbit_labels(state.spec, state.letters) == ['o000', 'o004']
    graph = CrystalDataset([{**record, '_joint_target': target}], operation_compiler=True)[0]
    assert graph.orbit_labels == labels
    assert 'orbit_labels' not in graph.to_dict()

    output = tmp_path/'model.pt'
    callback = JointTrainingArtifacts(output)
    trainer = SimpleNamespace(current_epoch=0,global_step=1,callback_metrics={},optimizers=[],callbacks=[])
    module = SimpleNamespace(config={'model': {'family': 'joint_flow_v2'}})
    callback.on_train_batch_end(trainer, module, torch.tensor(1.), _collate([graph]), 0)
    callback.on_train_epoch_end(trainer, module)
    for suffix in ('.status.json', '.history.jsonl'):
        log = json.loads(output.with_suffix(suffix).read_text(encoding='utf-8'))
        assert log['target_orbit_example']['orbit_labels'] == labels

    birth = {'action': {'kind': 'birth', 'letter': 'o004'}}
    sample = JointSample(state, 1., [birth], 'complete', 0, 1e-4)
    row = evaluation_record(sample, 0, 42, requested_group=group)
    assert row['accepted'] and row['orbit_labels'] == labels
    assert row['letters'] == ['o000', 'o004']
    assert row['events'] == [birth]
    custom = evaluation_record(sample, 0, 42, requested_group=group, label_group=0)
    assert custom['orbit_labels'] == custom['letters']

    with TrajectoryWriter(tmp_path/'trajectory', 1) as writer:
        writer.start(0, 42, state.spec, group, state.chemistry, {})
        writer.record(0, 'event', state, 1., 0, birth)
        writer.finish(0, row)
    frame = json.loads(writer.path(0).read_text(encoding='utf-8').splitlines()[1])
    assert frame['orbit_labels'] == labels
    assert frame['event'] == birth
    assert frame['event_orbit_label'] == labels[1]
