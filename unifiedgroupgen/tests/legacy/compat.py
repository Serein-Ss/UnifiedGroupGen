"""Thin legacy entry points execute the single packaged workflow implementation."""
from pathlib import Path
from importlib.util import find_spec
from unifiedgroupgen.tests.legacy.workflow_registry import WORKFLOWS


def legacy_entry(name,namespace):
    path=Path(find_spec('unifiedgroupgen.'+WORKFLOWS[name]).origin)
    namespace['__file__']=str(path)
    exec(compile(path.read_text(encoding='utf-8'),str(path),'exec'),namespace)
