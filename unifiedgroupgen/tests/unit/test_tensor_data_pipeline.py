"""CSV cache and PyG batching contracts; never fit or update parameters."""
import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch.utils.data import RandomSampler, SequentialSampler
from pymatgen.core import Lattice,Structure
from torch_geometric.data import Data,Batch

from unifiedgroupgen.common import data_utils as utils
from unifiedgroupgen.pl_data import cache
from unifiedgroupgen.pl_data.dataset import CrystalDataset,load_training_splits
from unifiedgroupgen.pl_data.datamodule import JointDataModule,_collate


@pytest.fixture
def source(tmp_path):
    torch.set_num_threads(1)
    folder=tmp_path/'data/src/fixture';folder.mkdir(parents=True)
    structure=Structure(Lattice.from_parameters(4.,4.5,5.,80.,87.,73.),['Si','O'],[[.11,.12,.13],[.37,.52,.61]])
    for split in ('train','val','test'):
        rows=[dict(material_id=split+str(i),cif=structure.to(fmt='cif'),gap=.5+i) for i in range(2)]
        pd.DataFrame(rows).to_csv(folder/(split+'.csv'),index=False)
    config=dict(dataset='fixture',kind='space',data_root='data/src/fixture',coordinate_mode='fractional',
        symmetry_tolerance=.001,symmetry_protocol=utils.SYMMETRY_PROTOCOL,max_atoms=192,max_orbits=32,property_columns={'band_gap':'gap'})
    yield tmp_path,config
    cache._load_tensor_cache.cache_clear()


def cached(source,split='train'):
    root,config=source
    path,payload,hit=cache.ensure_tensor_cache(config,split,root,graph_options={'cutoff':3.})
    return path,payload,cache.records_from_cache(path)


def test_csv_to_safe_tensor_cache_and_hit_skips_all_processing(source,monkeypatch):
    root,config=source;path,payload,records=cached(source)
    assert len(records)==2 and path.parent==root/'data/dataset/fixture'
    raw=torch.load(path,weights_only=True)
    assert isinstance(raw['target_tensors']['metric']['values'],torch.Tensor)
    assert isinstance(payload['targets'][0]['metric'],torch.Tensor)
    assert records[0]['dataset']==config['dataset'] and 'provenance' not in records[0]
    assert json.loads(path.with_suffix('.config.json').read_text())['identity']==raw['identity']
    def forbidden(*a,**k):pytest.fail('Cache hit repeated preprocessing')
    for name in ('convert_row','target_from_record','crystal_graph_arrays'):
        monkeypatch.setattr(cache,name,forbidden)
    second,_,hit=cache.ensure_tensor_cache(config,'train',root,graph_options={'cutoff':3.})
    assert second==path and hit
    data=CrystalDataset(records,operation_compiler=True)
    assert isinstance(data[0],Data) and data[0].pos.shape==(2,3)
    assert torch.isfinite(data[0].pos).all()


def test_source_or_options_change_selects_a_different_cache(source):
    root,config=source;first,_,_=cached(source)
    changed,_,_=cache.ensure_tensor_cache(config,'train',root,graph_options={'cutoff':2.})
    assert changed!=first and first.exists()
    frame=pd.read_csv(root/'data/src/fixture/train.csv');frame.loc[0,'gap']=9.
    frame.to_csv(root/'data/src/fixture/train.csv',index=False)
    revised,payload,_=cache.ensure_tensor_cache(config,'train',root,graph_options={'cutoff':3.})
    assert revised!=first and payload['records'][0]['properties']['band_gap']==9.


def test_pyg_batch_offsets_orbits_without_cross_graph_edges(source):
    _,_,records=cached(source);data=CrystalDataset(records,True)
    batch=_collate([data[0],data[1]])
    assert isinstance(batch,Batch) and batch.num_graphs==2
    assert batch.pos.shape==(4,3) and batch.cell.shape==(2,3,3)
    assert torch.equal(batch.batch[batch.edge_index[0]],batch.batch[batch.edge_index[1]])
    assert batch.orbit_index[2:].min()==data[0].num_orbits
    assert [r['id'] for r in batch.records]==['train0','train1']
    assert batch.property_values.shape==(2,1) and batch.property_mask.all()


def test_fixed_batches_keep_all_graphs_and_resume_shuffled_order(source, monkeypatch):
    _, _, records = cached(source)
    template = CrystalDataset(records, True)[0]
    graphs = [template.clone() for _ in range(7)]
    for i, graph in enumerate(graphs):
        object.__setattr__(graph, 'record', {'id': i})
    monkeypatch.setattr('unifiedgroupgen.pl_data.datamodule.CrystalDataset', lambda *_: graphs)
    data = JointDataModule(records, records, {'batch_size': 3})
    data.trainer = SimpleNamespace(current_epoch=0)

    def ids(batch):
        return [row['id'] for row in batch.records]

    train = data.train_dataloader()
    val = data.val_dataloader()
    assert isinstance(train.sampler, RandomSampler) and isinstance(val.sampler, SequentialSampler)
    assert len(train) == len(val) == 3
    expected = [ids(batch) for batch in train]
    assert [len(batch) for batch in expected] == [3, 3, 1]
    assert sorted(i for batch in expected for i in batch) == list(range(7))
    assert [ids(batch) for batch in val] == [[0, 1, 2], [3, 4, 5], [6]]

    iterator = iter(train)
    assert ids(next(iterator)) == expected[0]
    state = train.state_dict()
    resumed = data.train_dataloader()
    resumed.load_state_dict(state)
    assert [ids(batch) for batch in resumed] == expected[1:]

    data.trainer.current_epoch = 1
    next_epoch = data.train_dataloader()
    next_epoch.load_state_dict(state)
    next_order = [ids(batch) for batch in next_epoch]
    assert next_order != expected and len(next_order) == 3
    assert [ids(batch) for batch in data.train_dataloader()] == next_order


def test_training_loads_from_csv_then_cache_without_any_jsonl(source,monkeypatch):
    root, config = source
    profile = root/'fixture.yaml'
    profile.write_text(json.dumps(config), encoding='utf-8')
    definition = {
        'dataset': config['dataset'], 'source_configs': {'fixture': str(profile)},
        'model': {'operation_compiler': True, 'cutoff': 6., 'max_neighbors': 64, 'max_images': 16},
    }
    records,validation,paths=load_training_splits(definition,root)
    assert len(records)==len(validation)==2
    assert all(p.suffix=='.pt' for values in paths.values() for p in values)
    assert not list(root.rglob('*.jsonl'))
    monkeypatch.setattr(cache,'convert_row',lambda *a:pytest.fail('Repeated CSV conversion'))
    again,_,_=load_training_splits(definition,root)
    assert again[0]['_tensor_cache']==records[0]['_tensor_cache']


def test_cache_failure_leaves_no_partial_pt(source,monkeypatch):
    root,config=source
    monkeypatch.setattr(cache,'crystal_graph_arrays',lambda *a,**k:(_ for _ in ()).throw(RuntimeError('fixture failure')))
    with pytest.raises(RuntimeError,match='fixture failure'):cache.ensure_tensor_cache(config,'train',root)
    assert not list((root/'data/dataset/fixture').glob('*.pt'))
    assert not list((root/'data/dataset/fixture').glob('*.tmp'))


def test_dataset_loads_each_cache_once_for_a_consistent_snapshot(source,monkeypatch):
    from unifiedgroupgen.pl_data import dataset as dataset_module
    _,_,records=cached(source);calls=[];load=dataset_module.load_tensor_cache
    def counted(path):
        calls.append(path)
        return load(path)
    monkeypatch.setattr(dataset_module,'load_tensor_cache',counted)
    data=CrystalDataset(records,True)
    assert len(data)==2 and calls==[records[0]['_tensor_cache']['path']]


def test_cached_and_direct_targets_match_paths_losses_and_gradients(source):
    from unifiedgroupgen.pl_data.unified.path import sample_path
    from unifiedgroupgen.pl_modules.factory import build_joint_model
    from unifiedgroupgen.pl_modules.unified.objectives import loss
    _,payload,records=cached(source)
    direct=utils.target_from_record(payload['records'][0],operation_compiler=True)
    restored=cache.target_from_cache(payload,0)
    assert direct.state.spec==restored.state.spec
    torch.testing.assert_close(direct.state.metric,restored.state.metric,atol=0,rtol=0)
    torch.testing.assert_close(direct.state.coordinates,restored.state.coordinates,atol=0,rtol=0)
    a=sample_path(direct,31,time=.7,allowed_elements=[8,14]);b=sample_path(restored,31,time=.7,allowed_elements=[8,14])
    torch.testing.assert_close(a.state.metric,b.state.metric,atol=0,rtol=0)
    torch.testing.assert_close(a.state.coordinates,b.state.coordinates,atol=0,rtol=0)
    cfg=dict(model=dict(family='joint_flow_v2',backbone='equiformer_v2',hidden=16,layers=1,channels=16,lmax=2,mmax=1,
             cutoff=3.,max_neighbors=32,use_gate_act=True),allowed_elements=[8,14],properties={},max_atoms=192,max_orbits=32)
    torch.manual_seed(1);model=build_joint_model(cfg).eval()
    rng=torch.get_rng_state().clone()
    first=loss(model,[a]);first['loss'].backward()
    gradients={k:p.grad.clone() for k,p in model.named_parameters() if p.grad is not None}
    model.zero_grad();torch.set_rng_state(rng);second=loss(model,[b]);second['loss'].backward()
    for key in first:torch.testing.assert_close(first[key],second[key],atol=2e-6,rtol=2e-6)
    for key,p in model.named_parameters():
        if key in gradients:torch.testing.assert_close(p.grad,gradients[key],atol=2e-5,rtol=2e-5)


def test_reference_bank_reads_tensor_cache(source):
    from unifiedgroupgen.common.structures import ReferenceBank, iter_records, read_records
    path,payload,records=cached(source)
    bank=ReferenceBank([path],dataset='fixture')
    assert len(bank.rows)==2 and len(bank.structure('train0'))==2
    assert len(read_records(path)) == 2 and next(iter_records(path))['id'] == 'train0'
    assert list(bank.identity())==[str(path)]


def test_batch_enters_shared_training_objective_without_training(source,monkeypatch):
    from unifiedgroupgen.pl_modules.unified.module import JointGeneratorModule
    _,_,records=cached(source)
    data=CrystalDataset(records,True);batch=_collate([data[0],data[1]])
    cfg=dict(model=dict(family='joint_flow_v2',backbone='equiformer_v2',hidden=16,layers=1,channels=16,lmax=2,mmax=1,
             cutoff=3.,max_neighbors=32,use_gate_act=True,operation_compiler=True,task_controls=True),allowed_elements=[8,14],
             properties={},max_atoms=192,max_orbits=32,generation_task='dng',chemical_set_probability=0.,event_budget=24)
    module=JointGeneratorModule(cfg).eval();logged=[]
    monkeypatch.setattr(module,'log_dict',lambda values,**kwargs:logged.append(kwargs))
    rng=torch.get_rng_state().clone()
    actual=module.training_step(batch,0)
    torch.set_rng_state(rng);expected=module.objective(batch.records)['loss']
    torch.testing.assert_close(actual,expected,atol=2e-6,rtol=2e-6)
    assert logged[0]['batch_size']==2


def test_cache_audit_uses_pt_without_jsonl(source):
    from unifiedgroupgen.pl_data.validation import audit_files, verify_training_audit
    root,config=source
    paths={split:[cache.ensure_tensor_cache(config,split,root)[0]] for split in ('train','val','test')}
    report=audit_files(paths,config,root/'audit.json')
    assert report['passed'] and report['records']==dict(train=2,val=2,test=2)
    verify_training_audit(root/'audit.json',config,paths)
    with pytest.raises(ValueError,match='different experiment'):
        verify_training_audit(root/'audit.json',dict(config, symmetry_tolerance=.1),paths)
    assert not list(root.rglob('*.jsonl'))


def test_fp32_metric_round_trip_uses_double_intermediate_and_keeps_fp32():
    root=Path(__file__).resolve().parents[3]
    row=json.loads((root/'unifiedgroupgen/tests/fixtures/operation_metric_fp32_regression.json').read_text(encoding='utf-8'))
    original=utils.target_from_record(row)
    converted=utils.target_from_record(row,operation_compiler=True)
    assert converted.state.metric.dtype==converted.state.coordinates.dtype==torch.float32
    torch.testing.assert_close(converted.state.decoded(wrap=False)['lattice'],original.state.decoded(wrap=False)['lattice'],
                               atol=2e-5,rtol=2e-5)


def test_packed_layout_uses_few_storages_and_zero_copy_views(source):
    path,payload,_=cached(source);raw=torch.load(path,weights_only=True,mmap=True)
    assert raw['version']==3
    assert all(set(row)=={'id','descriptor','state','properties'} for row in json.loads(raw['records_json']))
    first=payload['graphs'][0]['pos'];second=payload['graphs'][1]['pos']
    assert first.untyped_storage().data_ptr()==second.untyped_storage().data_ptr()
    assert first.storage_offset()!=second.storage_offset()
    assert len(raw['graph_tensors'])<20


def test_cached_group_checksum_does_not_disable_raw_group_validation(source):
    _,payload,_=cached(source)
    payload['specs'][0]+=' '
    with pytest.raises(ValueError,match='metadata changed'):cache.target_from_cache(payload,0)
    raw=json.loads(payload['specs'][0]);raw['operations'][0]['rotation']=[[0,0,0]]*3
    with pytest.raises(ValueError):utils.SymmetrySpec.from_dict(raw)


