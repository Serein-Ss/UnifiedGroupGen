"""The new source-to-standard-cell protocol, without fitting a network."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from pymatgen.core import Lattice, Structure

from unifiedgroupgen.common import data_utils as data
from unifiedgroupgen.group_constraints.compiler import compile_descriptor
from unifiedgroupgen.group_constraints.encoding import decompose_orbits, operation_residual


ROOT = Path(__file__).resolve().parents[3]


def config(kind='space'):
    return dict(dataset='fixture', kind=kind, symmetry_protocol=data.SYMMETRY_PROTOCOL,
                coordinate_mode='fractional', symmetry_tolerance=.1 if kind=='space' else .01,
                angle_tolerance=5., cif_fractional_tolerance=0, niggli=False,
                layer_group_policy='identify', max_atoms=192, max_orbits=52)


def test_identification_uses_source_once_and_never_calls_pyxtal_or_niggli(monkeypatch):
    from pyxtal import pyxtal
    calls=[];native=data.spglib.get_symmetry_dataset
    structure=Structure(Lattice.cubic(4), ['Na','Cl'], [[0,0,0],[.5,.5,.5]])
    def identify(cell, **kwargs):
        calls.append((cell,kwargs));return native(cell,**kwargs)
    def forbidden(*args,**kwargs):pytest.fail('Repeated refinement or lattice reduction')
    monkeypatch.setattr(data.spglib,'get_symmetry_dataset',identify)
    monkeypatch.setattr(pyxtal,'from_seed',forbidden)
    monkeypatch.setattr(Structure,'get_reduced_structure',forbidden)
    descriptor,state,provenance=data.standard_record(structure,'space',.1,5.)
    assert len(calls)==1 and calls[0][1]==dict(symprec=.1,angle_tolerance=5.)
    assert descriptor.number==221 and provenance['identified_orbits']==2
    assert provenance['symmetry_protocol']==data.SYMMETRY_PROTOCOL
    compiled=compile_descriptor(descriptor)
    assert compiled.num_atoms==2 and np.isfinite(state).all()


def test_hall_selection_reuses_original_cell_not_refined_positions(monkeypatch):
    from types import SimpleNamespace
    calls=[];native=data.spglib.get_symmetry_dataset
    structure=Structure(Lattice.cubic(4), ['Na','Cl'], [[.001,0,0],[.5,.5,.5]])
    def identify(cell, **kwargs):
        calls.append((cell,kwargs));result=native(cell,**kwargs)
        if len(calls)==1:
            return SimpleNamespace(number=result.number,hall_number=-1)
        return result
    monkeypatch.setattr(data.spglib,'get_symmetry_dataset',identify)
    data.standard_record(structure,'space',.1,5.)
    assert len(calls)==2 and calls[0][0] is calls[1][0]
    assert calls[1][1]['angle_tolerance']==5.


def test_standard_orbit_mapping_covers_expanded_cell():
    from types import SimpleNamespace
    result=SimpleNamespace(mapping_to_primitive=[0,1,2],crystallographic_orbits=[0,1,1],
                           std_mapping_to_primitive=[0,1,2,0,1,2])
    assert data._standard_orbit_labels(result).tolist()==[0,1,1,0,1,1]


def test_native_orbits_cannot_be_merged_or_split_by_loose_chart_tolerance():
    coords=np.array([[.001,0,0],[.002,0,0]])
    descriptor,_=decompose_orbits('space',1,coords,np.array([14,14]),.1,np.eye(3),orbit_labels=[0,1])
    assert len(descriptor.orbits)==2
    with pytest.raises(ValueError,match='Cannot match'):
        decompose_orbits('space',1,coords,np.array([14,14]),.1,np.eye(3),orbit_labels=[0,0])


def test_layer_uses_native_2d_api_and_fixed_orbits(monkeypatch):
    calls=[];native=data.spglib.get_layergroup
    structure=Structure(Lattice.hexagonal(3.2,30),['Mo','S','S'],
                        [[0,0,.5],[1/3,2/3,.45],[1/3,2/3,.55]])
    def identify(cell, **kwargs):
        calls.append(kwargs);return native(cell,**kwargs)
    monkeypatch.setattr(data.spglib,'get_layergroup',identify)
    descriptor,state,provenance=data.standard_record(structure,'layer',.01)
    assert calls==[dict(aperiodic_dir=0,symprec=.01)]
    assert provenance['native_axis_order']==[2,0,1]
    assert provenance['angle_tolerance_degrees'] is None
    assert len(descriptor.orbits)==provenance['identified_orbits']==2
    compiled=compile_descriptor(descriptor)
    expanded=compiled.expand(torch.tensor(state,dtype=torch.float64))
    assert operation_residual(expanded['coordinates'].numpy(),compiled.elements,
                              data.group_table('layer',descriptor.number)[0].ops,
                              compiled.periodic,expanded['lattice'].numpy()) < 1e-5


def test_cif_parser_keeps_near_rational_coordinates(monkeypatch):
    structure=Structure(Lattice.from_parameters(4,4.5,5,80,87,73),['Si','O'],
                        [[.66667968,.12,.13],[.37,.52,.61]])
    seen=[];native=data.standard_record
    def convert(s,*args,**kwargs):
        seen.append(s.frac_coords.copy());return native(s,*args,**kwargs)
    monkeypatch.setattr(data,'standard_record',convert)
    result=data.convert_row((0,dict(material_id='rounding',cif=structure.to(fmt='cif')),config(),'fixture.csv'))
    assert 'record' in result,result
    assert seen[0][0,0]==pytest.approx(.66667968,abs=1e-9)


@pytest.mark.parametrize('change',[{'niggli':True},{'cif_fractional_tolerance':1e-4}])
def test_conflicting_protocol_options_are_rejected(change):
    c=config();c.update(change)
    s=Structure(Lattice.cubic(4),['Si'],[[0,0,0]])
    result=data.convert_row((0,dict(material_id='bad',cif=s.to(fmt='cif')),c,'fixture.csv'))
    assert 'rejection' in result and 'unsnapped' in result['rejection']['reason']


def test_mp1640048_reproduces_recommended_group_and_cell():
    frame=pd.read_csv(ROOT/'data/src/mp_20/train.csv',nrows=616)
    row=frame[frame.material_id=='mp-1640048'].iloc[0].to_dict()
    result=data.convert_row((615,row,config(),'fixture.csv'))
    assert 'record' in result,result
    record=result['record']
    assert record['descriptor']['number']==204
    assert len(record['descriptor']['orbits'])==4
    assert record['provenance']['source_atoms']==20 and record['provenance']['expanded_atoms']==40


@pytest.mark.parametrize('identifier',['mp-1205336','mp-1252764','mp-1078209','mp-1281596',
                                       'mp-1095141','mp-1304047','mp-1289597','mp-850083'])
def test_nonstandard_input_orbits_use_crystallographic_partition(identifier):
    frame=pd.read_csv(ROOT/'data/src/mp_20/train.csv',nrows=4700)
    row=frame[frame.material_id==identifier].iloc[0].to_dict()
    result=data.convert_row((0,row,config(),'fixture.csv'))
    assert 'record' in result,result
    record=result['record']
    assert len(record['descriptor']['orbits'])==record['provenance']['identified_orbits']


def test_exact_cubic_primitive_matches_ideal_conventional_cell():
    row=pd.read_csv(ROOT/'data/src/mp_20/val.csv',nrows=8661).iloc[8660].to_dict()
    assert row['material_id']=='mp-1063914'
    first=data.convert_row((8660,row,config(),'fixture.csv'))
    assert 'record' in first,first
    assert first['record']['descriptor']['number']==216
    check=first['record']['provenance']['source_integrity']
    assert check['geometry_equivalent'] and check['max_displacement_angstrom']<1e-6


def test_near_square_layer_uses_equivalent_native_basis_without_hanging():
    row=pd.read_csv(ROOT/'data/src/c2db_51/train.csv',nrows=4025).iloc[4024].to_dict()
    assert row['uid']=='2HKO-2'
    result=data.convert_row((4024,row,config('layer'),'fixture.csv'))
    assert 'record' in result,result
    assert result['record']['descriptor']['number']==9
    assert result['record']['provenance']['expanded_atoms']==6


def test_mixed_space_and_layer_batch_keeps_pbc_and_atom_offsets():
    from unifiedgroupgen.pl_data.dataset import CrystalDataset
    from unifiedgroupgen.pl_data.datamodule import _collate
    records=[]
    for kind,structure in [
        ('space',Structure(Lattice.cubic(4),['Na','Cl'],[[0,0,0],[.5,.5,.5]])),
        ('layer',Structure(Lattice.hexagonal(3.2,30),['Mo','S','S'],
                           [[0,0,.5],[1/3,2/3,.45],[1/3,2/3,.55]]))]:
        row=dict(material_id=kind,cif=structure.to(fmt='cif'),lgnum=-1,official_structure_sha256='fixture')
        result=data.convert_row((0,row,config(kind),'fixture.csv'))
        assert 'record' in result,result
        records.append(result['record'])
    dataset=CrystalDataset(records,operation_compiler=True)
    batch=_collate([dataset[0],dataset[1]])
    assert batch.pbc.tolist()==[[True,True,True],[True,True,False]]
    assert batch.num_nodes==5 and torch.isfinite(batch.pos).all()
    assert torch.equal(batch.batch[batch.edge_index[0]],batch.batch[batch.edge_index[1]])


def test_native_correspondence_accepts_shared_origin_but_rejects_atomic_distortion():
    s=Structure(Lattice.cubic(4),['Na','Cl'],[[0,0,0],[.5,.5,.5]])
    native=data.spglib.get_symmetry_dataset((s.lattice.matrix,s.frac_coords,s.atomic_numbers),
                                           symprec=.1,angle_tolerance=5.)
    descriptor,state,_=data.standard_record(s,'space',.1,5.)
    compiled=compile_descriptor(descriptor)
    shifted=s.copy();shifted.translate_sites(range(len(s)),[.9,0,0],frac_coords=False)
    check=data._standard_space_integrity(shifted,native,compiled,state,.1,5.)
    assert check['geometry_equivalent'] and check['max_displacement_angstrom']<1e-8
    shifted.translate_sites([1],[.4,0,0],frac_coords=False)
    with pytest.raises(ValueError,match='Source integrity failed'):
        data._standard_space_integrity(shifted,native,compiled,state,.1,5.)


def test_preparation_worker_limits_both_tensor_and_blas_threads(monkeypatch):
    from threadpoolctl import threadpool_info,threadpool_limits
    from unifiedgroupgen.pl_data import cache
    monkeypatch.setattr(cache,'_TRANSFER_ARRAYS',False)
    with threadpool_limits(limits=None):
        cache._prepare_worker()
        assert torch.get_num_threads()==1
        assert all(pool['num_threads']==1 for pool in threadpool_info() if pool['user_api']=='blas')


@pytest.mark.parametrize('identifier',['mp-30545','mp-548149','mp-19127','mp-1218148','mp-1187854','mp-1218740'])
def test_conventional_cell_length_change_is_not_a_second_symprec_filter(identifier):
    frame=pd.read_csv(ROOT/'data/src/mp_20/train.csv',nrows=19200)
    row=frame[frame.material_id==identifier].iloc[0].to_dict()
    result=data.convert_row((0,row,config(),'fixture.csv'))
    assert 'record' in result,result
    check=result['record']['provenance']['source_integrity']
    assert check['geometry_equivalent'] and check['max_displacement_angstrom']<=.1
    assert check['lattice_length_relative_change']>check['lattice_relative_tolerance']


def test_numeric_failure_needs_two_agreeing_valid_passes(monkeypatch):
    from copy import deepcopy
    s=Structure(Lattice.cubic(4),['Na','Cl'],[[0,0,0],[.5,.5,.5]])
    item=(0,dict(material_id='numeric',cif=s.to(fmt='cif')),config(),'fixture.csv')
    good=data._convert_row_once(item)
    bad={'row':0,'rejection':dict(reason='matrix contains invalid numeric entries')}
    calls=[]
    def attempt(value):
        calls.append(value);return deepcopy(bad if len(calls)==1 else good)
    monkeypatch.setattr(data,'_convert_row_once',attempt)
    result=data.convert_row(item)
    assert len(calls)==3 and all(value is item for value in calls)
    assert result['record']['provenance']['numerical_recheck']['valid_agreeing_checks']==2


def test_numeric_rechecks_cannot_accept_disagreement_or_change_tolerance(monkeypatch):
    from copy import deepcopy
    s=Structure(Lattice.cubic(4),['Na','Cl'],[[0,0,0],[.5,.5,.5]])
    item=(0,dict(material_id='numeric',cif=s.to(fmt='cif')),config(),'fixture.csv')
    good=data._convert_row_once(item);changed=deepcopy(good);changed['record']['state'][0]+=.01
    bad={'row':0,'rejection':dict(reason='matrix contains invalid numeric entries')}
    attempts=iter([bad,good,changed])
    monkeypatch.setattr(data,'_convert_row_once',lambda value:next(attempts))
    assert data.convert_row(item) is bad and item[2]['symmetry_tolerance']==.1
