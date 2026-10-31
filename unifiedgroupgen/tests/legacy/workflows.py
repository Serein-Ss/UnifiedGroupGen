"""Run a packaged workflow: ugg-run <workflow> [its original arguments]."""
import argparse
import importlib
from pathlib import Path
import sys
from unifiedgroupgen.tests.legacy.workflow_registry import WORKFLOWS


def main():
    choices=sorted(WORKFLOWS)
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('workflow',choices=choices)
    args=parser.parse_args(sys.argv[1:2])
    remaining=sys.argv[2:]
    sys.argv=[args.workflow,*remaining]
    importlib.import_module('unifiedgroupgen.'+WORKFLOWS[args.workflow]).main()


if __name__=='__main__': main()
