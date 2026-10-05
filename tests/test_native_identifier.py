from types import SimpleNamespace
import subprocess
import pytest
import torch
from unifiedgroupgen.symmetry import Descriptor, OrbitSpec, compile_descriptor
from unifiedgroupgen.evaluation import to_structure, validate, summarize_attempts
from unifiedgroupgen.native_identifier import identify_isolated


@pytest.mark.parametrize('kind,number,letter', [('space', 2, 'i'), ('layer', 37, 'r')])
def test_isolated_identifier_matches_direct_native_result(kind, number, letter):
    torch.manual_seed(3)
    descriptor = Descriptor(kind, number, (OrbitSpec(letter, 14), OrbitSpec(letter, 8)))
    compiled = compile_descriptor(descriptor)
    generated = compiled.expand(compiled.prior(dtype=torch.float64))
    direct = validate(compiled, generated)
    isolated = validate(compiled, generated, isolate_identifier=True)
    assert isolated[f'recovered_{kind}_group'] == direct[f'recovered_{kind}_group']
    assert isolated['exact_target_group'] == direct['exact_target_group']
    assert isolated['group_identification_error'] is None


def test_native_exit_and_timeout_are_unknown_and_remain_in_sampling_denominator(monkeypatch):
    import unifiedgroupgen.native_identifier as native
    compiled = compile_descriptor(Descriptor('space', 2, (OrbitSpec('i', 14),)))
    generated = compiled.expand(compiled.prior())
    monkeypatch.setattr(native.subprocess, 'run', lambda *a, **kw: SimpleNamespace(returncode=-11, stderr='native abort', stdout=''))
    report = validate(compiled, generated, isolate_identifier=True)
    assert report['recovered_space_group'] is None and report['exact_target_group'] is None
    assert 'exited with -11' in report['group_identification_error']
    summary = summarize_attempts([{'status': 'generated', 'validation': report}])
    assert summary['attempted'] == 1 and summary['exact_group_unidentified_generated'] == 1
    assert summary['exact_group_success'] == 0
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired('worker', 20)
    monkeypatch.setattr(native.subprocess, 'run', timeout)
    assert 'timed out' in identify_isolated(to_structure(compiled, generated), 'space')['error']
