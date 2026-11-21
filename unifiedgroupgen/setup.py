"""Discover subpackages; distribution metadata is in pyproject.toml."""
from pathlib import Path
from setuptools import find_packages, setup

root = Path(__file__).resolve().parent
packages = find_packages(where=str(root), exclude=('tests', 'tests.*'))
setup(packages=['unifiedgroupgen', *(f'unifiedgroupgen.{name}' for name in packages)])
