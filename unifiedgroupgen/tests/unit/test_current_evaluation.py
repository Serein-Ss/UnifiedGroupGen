"""Current evaluation APIs directly, without historical CLI imports or training."""
import copy
import pytest
from pymatgen.core import Lattice, Structure

from unifiedgroupgen.common.structures import ReferenceBank
from unifiedgroupgen.evaluation.common import EvaluationContext, evaluate_records
from unifiedgroupgen.evaluation.joint_csp import match_csp, target_view
from unifiedgroupgen.evaluation.utils import geometry_metrics
from unifiedgroupgen.tests.unit.test_joint_flow_v2 import record


def test_current_dng_retains_failed_attempts_and_duplicate_labels(tmp_path):
    structure=Structure(Lattice.cubic(5.6),['Na','Cl'],[[0,0,0],[.5,.5,.5]])
    cif=structure.to(fmt='cif')
    rows=[dict(index=0,status='generated',cif=cif),dict(index=1,status='generated',cif=cif),
          dict(index=2,status='failed',reason='sampler fixture'),dict(index=3,status='generated',cif='invalid CIF')]
    refs=ReferenceBank([])
    context=EvaluationContext('dng','space',None,{},refs,refs,False,False,[],tmp_path/'samples.jsonl')
    before=copy.deepcopy(rows)
    result,features=evaluate_records(rows,context)
    assert rows==before and [r['index'] for r in result]==[0,1,2,3]
    assert result[0]['basic_valid'] and result[0]['unique']
    assert result[1]['basic_valid'] and not result[1]['unique']
    assert result[2]['failure_reason']=='sampler fixture' and not result[2]['generation_success']
    assert result[3]['parse_success'] is False and not result[3]['basic_valid']
    assert features==[]


def test_dng_chemistry_does_not_read_unrelated_smact_hhi(monkeypatch):
    from smact import data_loader
    from unifiedgroupgen.evaluation.utils import composition_validity

    def unexpected_hhi(symbol):
        pytest.fail(f'Chemical validity must not read HHI for {symbol}')

    monkeypatch.setattr(data_loader, 'lookup_element_hhis', unexpected_hhi)
    nacl = Structure(Lattice.cubic(5.6), ['Na', 'Cl'], [[0, 0, 0], [.5, .5, .5]])
    assert composition_validity(nacl) == (True, None)


@pytest.mark.parametrize('dataset',['mp_20','c2db_51'])
def test_current_csp_matches_real_endpoint_and_rejects_wrong_composition(dataset):
    view=target_view(record(dataset))
    result=match_csp(view,view)
    assert result['composition_match'] and result['matched'] and result['normalized_rmse']<1e-5
    wrong={**view,'atomic_numbers':[118]*len(view['atomic_numbers'])}
    result=match_csp(view,wrong)
    assert not result['composition_match'] and not result['matched']


def test_current_layer_geometry_excludes_artificial_vacuum():
    def slab(height):
        return Structure(Lattice.hexagonal(3.2,height),['Mo','S','S'],
                         [[0,0,height/2],[1.6,.7,height/2-1.5],[1.6,.7,height/2+1.5]],coords_are_cartesian=True)
    first,second=(geometry_metrics(slab(h),'layer') for h in (20,50))
    assert first['layer_thickness_angstrom']==pytest.approx(second['layer_thickness_angstrom'])
    assert first['area_per_atom_angstrom2']==pytest.approx(second['area_per_atom_angstrom2'])
