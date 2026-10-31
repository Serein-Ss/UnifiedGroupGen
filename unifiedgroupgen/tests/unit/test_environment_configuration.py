"""Installation and legacy-command boundaries for the maintained package."""
import ast
from pathlib import Path
import sys

import yaml

PKG = Path(__file__).resolve().parents[2]


def test_one_environment_file_and_one_distribution_configuration():
    assert not (PKG/'environment').exists()
    assert (PKG/'environment.yml').is_file()
    assert not (PKG.parent/'pyproject.toml').exists()
    spec = yaml.safe_load((PKG/'environment.yml').read_text(encoding='utf-8'))
    assert 'variables' not in spec
    pip = next(item['pip'] for item in spec['dependencies'] if isinstance(item, dict))
    assert 'torch==2.7.1+cu118' in pip and 'pyxtal==1.1.3' in pip
    assert not any('win_amd64' in item or 'sys_platform' in item for item in pip)
    assert not any('file://' in s or 'D:\\' in s for s in pip)


def test_new_runtime_has_no_archived_command_imports():
    violations=[]
    for p in PKG.rglob('*.py'):
        if p.is_relative_to(PKG/'tests') or any(part in p.relative_to(PKG).parts for part in ('__pycache__', 'build')) or p.name=='setup.py':
            continue
        tree=ast.parse(p.read_text(encoding='utf-8'))
        for node in ast.walk(tree):
            if isinstance(node,ast.ImportFrom):
                names=[node.module or '']
            elif isinstance(node,ast.Import):
                names=[a.name for a in node.names]
            else:
                continue
            for name in names:
                if name.startswith(('unifiedgroupgen.tests.legacy','unifiedgroupgen.common.cli',
                                    'unifiedgroupgen.common.compat','unifiedgroupgen.common.workflows')):
                    violations.append((str(p),name))
    assert not violations


def test_only_new_console_commands_are_advertised():
    import tomli
    config=tomli.loads((PKG/'pyproject.toml').read_text(encoding='utf-8'))
    assert set(config['project']['scripts'])=={'ugg'}
    assert 'packages' not in config['tool']['setuptools']


def test_source_identity_excludes_archived_commands():
    from unifiedgroupgen.common.source import package_sources
    assert package_sources()
    assert not any(p.is_relative_to(PKG/'tests/legacy') for p in package_sources())


def test_archived_aliases_only_reference_existing_implementations():
    from unifiedgroupgen.tests.legacy.layout import MODULE_PATHS
    from unifiedgroupgen.tests.legacy.workflow_registry import WORKFLOWS
    assert all((PKG/path).is_file() for path in MODULE_PATHS.values())
    assert all((PKG/(module.replace('.', '/')+'.py')).is_file() for module in WORKFLOWS.values())
