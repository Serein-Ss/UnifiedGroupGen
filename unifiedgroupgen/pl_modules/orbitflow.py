"""Compatibility import; implementation lives in unifiedgroupgen.pl_modules.two_stage.orbitflow."""
import importlib as _importlib
import sys as _sys

if __name__ == "__main__":
    import runpy as _runpy
    _runpy.run_module("unifiedgroupgen.pl_modules.two_stage.orbitflow", run_name="__main__")
else:
    _sys.modules[__name__] = _importlib.import_module("unifiedgroupgen.pl_modules.two_stage.orbitflow")
