import copy
import importlib.util
from pathlib import Path


def module():
    path = Path(__file__).resolve().parents[1]/'scripts/audit_runtime_threads.py'
    spec = importlib.util.spec_from_file_location('audit_runtime_threads', path)
    value = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(value)
    return value


def report():
    m = module()
    value = {key: 'unchanged' for key in m.IDENTITY}
    value.update(fitted_config_sha256='original', properties={'gap': {'per_domain': True, 'shift': 1., 'scale': 2.,
                 'domain_scalers': {'0': {'shift': 1., 'scale': 2.}, '1': {'shift': 2., 'scale': 3.}}}})
    return value


def test_unused_global_roundoff_explains_hash_without_changing_active_inputs():
    m = module()
    base = report(); other = copy.deepcopy(base)
    other['properties']['gap']['shift'] += 1e-15
    other['fitted_config_sha256'] = 'different'
    result = m.compare(base, other)
    assert result['compatible'] and len(result['unused_global_scaler_roundoff']) == 1
    assert other['properties']['gap']['shift'] != base['properties']['gap']['shift']


def test_data_change_and_active_scaler_change_still_fail():
    m = module(); base = report()
    for key in ('record_ids', 'train_split_sha256', 'allowed_elements'):
        other = copy.deepcopy(base); other[key] = 'changed'
        assert not m.compare(base, other)['compatible']
    other = copy.deepcopy(base)
    other['properties']['gap']['domain_scalers']['0']['shift'] += 1e-15
    assert not m.compare(base, other)['compatible']
    other = copy.deepcopy(base)
    del other['properties']['gap']['domain_scalers']
    assert not m.compare(base, other)['compatible']


def test_unexplained_hash_or_large_global_change_fails():
    m = module(); base = report(); other = copy.deepcopy(base)
    other['fitted_config_sha256'] = 'different'
    assert not m.compare(base, other)['compatible']
    other['properties']['gap']['scale'] = 20.
    assert not m.compare(base, other)['compatible']
