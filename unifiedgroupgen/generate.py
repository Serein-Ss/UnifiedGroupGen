"""Generate structures with a current joint_flow_v2 checkpoint."""
import sys
from unifiedgroupgen.joint import main as joint_main


def main(argv=None):
    return joint_main(['sample', *(sys.argv[1:] if argv is None else argv)])


if __name__ == '__main__':
    main()
