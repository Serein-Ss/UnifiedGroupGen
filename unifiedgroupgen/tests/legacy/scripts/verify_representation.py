"""Compatibility entry; implementation lives in the installed package."""
import sys
from pathlib import Path
from unifiedgroupgen.tests.legacy.compat import legacy_entry
legacy_entry('verify_representation',globals())
