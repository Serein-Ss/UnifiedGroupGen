"""Validate the single environment specification and actual installed GPU runtime."""
import argparse
import importlib
import importlib.metadata
import json
import locale
from pathlib import Path
import platform
import subprocess
import sys

from packaging.requirements import Requirement
from packaging.utils import parse_wheel_filename
from urllib.parse import urlparse, unquote
import yaml


def inspect_environment(require_cuda=False):
    import torch
    from unifiedgroupgen.tests.catalog import check_layer_catalog

    package = Path(__file__).resolve().parents[1]
    specification = yaml.safe_load((package/'environment.yml').read_text(encoding='utf-8'))
    versions, failures = {}, []
    if platform.python_version() != '3.10.18':
        failures.append('Python differs from the 3.10.18 reference')
    for item in specification['dependencies']:
        requirements = item['pip'] if isinstance(item, dict) else [item.replace('=', '==', 1)]
        for text in requirements:
            if text.startswith('--') or text.startswith('python=='):
                continue
            req = Requirement(text)
            if req.marker and not req.marker.evaluate():
                continue
            try:
                actual = importlib.metadata.version(req.name)
                versions[req.name] = actual
                expected_url_version = str(parse_wheel_filename(Path(unquote(urlparse(req.url).path)).name)[1]) if req.url else None
                if actual not in req.specifier or (expected_url_version and actual != expected_url_version):
                    failures.append(f'{req.name}: expected {req.specifier}, found {actual}')
            except importlib.metadata.PackageNotFoundError:
                failures.append('Missing distribution: '+req.name)
    result = subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True)
    if result.returncode:
        failures.append('pip check failed: '+result.stdout+result.stderr)
    cuda = {'available': torch.cuda.is_available(), 'runtime_version': torch.__version__,
            'distribution_version': importlib.metadata.version('torch'), 'build': torch.version.cuda}
    if torch.__version__ != cuda['distribution_version']:
        failures.append('Torch runtime/distribution versions differ')
    if require_cuda and not cuda['available']:
        failures.append('CUDA is required but unavailable')
    if cuda['available']:
        if cuda['build'] != '11.8':
            failures.append('Expected the official CUDA 11.8 build')
        value = torch.randn(4, 4, device='cuda', requires_grad=True)
        (value @ value).square().mean().backward()
        torch.cuda.synchronize()
        if not torch.isfinite(value.grad).all():
            failures.append('Nonfinite CUDA gradient')
        from torch_scatter import scatter
        scatter(torch.ones(2, device='cuda'), torch.tensor([0, 0], device='cuda'))
        cuda['device'] = torch.cuda.get_device_name(0)
        cuda['tensor_and_backward'] = True
    for name in ('torch_sparse', 'torch_cluster', 'torch_spline_conv', 'torch_geometric',
                 'pytorch_lightning', 'hydra', 'wandb', 'smact', 'matminer', 'ase',
                 'einops', 'e3nn', 'matgl', 'dgl'):
        try:
            importlib.import_module(name)
        except Exception as error:
            failures.append(name+': '+str(error))
    try:
        catalog = check_layer_catalog()
    except Exception as error:
        catalog = None
        failures.append('Group catalog: '+str(error))
    chemistry = None
    try:
        from pymatgen.core import Lattice, Structure
        from unifiedgroupgen.evaluation.utils import composition_validity
        chemistry, reason = composition_validity(Structure(Lattice.cubic(5.64), ['Na', 'Cl'], [[0, 0, 0], [.5, .5, .5]]))
        if chemistry is not True:
            failures.append('NaCl chemistry check failed: '+str(reason))
    except Exception as error:
        failures.append('SMACT data read failed; activate the UTF-8 configured environment: '+str(error))
    return {'text_encoding': locale.getpreferredencoding(False), 'smact_nacl_valid': chemistry, 'passed': not failures, 'python': platform.python_version(), 'prefix': sys.prefix,
            'versions': versions, 'cuda': cuda, 'catalog': catalog, 'failures': failures,
            'external_predictors': 'Public MEGNet adapter uses matgl/DGL; weights fetched or cached separately. CGDiT source/checkpoints remain separate explicit inputs.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--require-cuda', action='store_true')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = inspect_environment(args.require_cuda)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({'passed': report['passed'], 'cuda': report['cuda'], 'failures': report['failures']}, indent=2))
    if not report['passed']:
        raise SystemExit(1)


if __name__ == '__main__':
    main()
