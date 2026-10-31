"""Discover subpackages; distribution metadata is in pyproject.toml."""
from pathlib import Path
from setuptools import find_packages, setup

root = Path(__file__).resolve().parent
setup(packages=['unifiedgroupgen', *('unifiedgroupgen.'+name for name in find_packages(where=str(root)))])
