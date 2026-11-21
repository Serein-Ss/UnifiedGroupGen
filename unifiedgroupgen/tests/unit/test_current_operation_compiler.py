"""Current operation compiler crosschecks; no model training or old adapters."""
from collections import Counter
import json
import numpy as np
import pytest
import torch
from unifiedgroupgen.group_constraints.operations import compile_operations, operation_spec
from unifiedgroupgen.group_constraints.unified import catalog_spec, SymmetrySpec
from unifiedgroupgen.common.data_utils import match_orbit

GROUPS = [(kind,n) for kind,count in [('space',230),('layer',80),('plane',17),('rod',75)] for n in range(1,count+1)]

@pytest.mark.parametrize('kind,number', GROUPS)
def test_computed_atlas_matches_every_catalog_orbit(kind, number):
    spec = operation_spec(kind, number)
    reference = catalog_spec(kind, number)
    assert spec.number == 0
    assert Counter((w.multiplicity, w.dof) for w in spec.templates) == Counter(
        (w.multiplicity, w.dof) for w in reference.templates)
    # Match actual generic coordinates, not only counts/histograms.
    matches = {}
    for w in reference.templates:
        a, b = w.arrays()
        x = b+np.einsum('niq,q->ni', a, np.array([.137031, .293117, 1.419071])[:w.dof])
        letter, fitted, order = match_orbit(spec, x, tolerance=1e-7)
        matches[letter] = w.letter
        assert spec.template(letter).dof == w.dof
        delta = fitted-x[order]
        delta[:, spec.periodic] -= np.round(delta[:, spec.periodic])
        assert np.max(np.abs(delta)) < 1e-7
    assert len(matches) == len(reference.templates)
    if kind == 'rod':
        from unifiedgroupgen.group_constraints.wyckoff import orbit_labels
        assert spec.dimension == 3 and spec.periodic == (True, False, False)
        assert orbit_labels(spec, tuple(matches), number) == [
            f'{spec.template(letter).multiplicity}{standard}' for letter, standard in matches.items()]


def test_compilation_uses_no_special_position_table(monkeypatch):
    from unifiedgroupgen.group_constraints import operations as module
    from unifiedgroupgen.group_constraints import unified
    from unifiedgroupgen.group_constraints import compiler
    module._atlas.cache_clear()
    def forbidden(*args, **kwargs):
        raise AssertionError('A special-position table was consulted')
    monkeypatch.setattr(unified, 'wp_arrays', forbidden)
    monkeypatch.setattr(unified, 'group_table', forbidden)
    monkeypatch.setattr(unified, 'catalog_spec', forbidden)
    monkeypatch.setattr(compiler, 'group_table', forbidden)
    mirror = dict(rotation=[[-1,0,0],[0,1,0],[0,0,1]], translation=[0,0,0])
    spec = compile_operations([mirror], 'space')
    assert Counter((w.multiplicity,w.dof) for w in spec.templates) == Counter({(1,2):2,(2,3):1})


def test_equivalent_generators_and_serialization_are_identical():
    x = dict(rotation=[[-1,0,0],[0,1,0],[0,0,1]], translation=[0,0,0])
    y = dict(rotation=[[1,0,0],[0,-1,0],[0,0,1]], translation=[0,0,0])
    first = compile_operations([x,y], 'plane')
    second = compile_operations([y,x,x]+[o.to_dict() for o in first.operations], 'plane')
    assert first == second == SymmetrySpec.from_dict(first.to_dict())
    assert first == SymmetrySpec.from_dict(dict(kind='plane',generators=[x,y]))


@pytest.mark.parametrize('extra', [dict(forbidden=['inversion']), dict(periodic=[True,True,True]),
                                  dict(dimension=3), dict(magnetic=True)])
def test_unsupported_or_inconsistent_requests_are_not_ignored(extra):
    with pytest.raises(ValueError):
        SymmetrySpec.from_dict(dict(kind='plane', generators=[], **extra))


@pytest.mark.parametrize('operation,kind,message', [
    (dict(rotation=[[1,1,0],[0,1,0],[0,0,1]],translation=[0,0,0]),'space','close'),
    (dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation=[0,0,1]),'layer','close'),
    (dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation=[0,0,1]),'plane','z=0'),
    (dict(rotation=[[2,0,0],[0,1,0],[0,0,1]],translation=[0,0,0]),'space','adapted'),
    (dict(rotation=[[1,0,0],[0,1,0],[0,0,1]],translation=[2**.5,0,0]),'space','rational'),
])
def test_invalid_operation_requests_fail_explicitly(operation, kind, message):
    with pytest.raises(ValueError, match=message):
        compile_operations([operation],kind)


