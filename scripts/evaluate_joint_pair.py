"""Two-checkpoint evaluation: discovery first, complete-test CSP separately. Default: plan."""
import argparse
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]


def command(args):
    output=args.output or ROOT/f'artifacts/evaluations/joint_pair_{args.suite}_seed{args.seed}'
    result=['scripts/evaluate_main15_unified.py','--models','joint_u','joint_all',
        '--checkpoint-root',str(args.checkpoint_root),'--output',str(output),
        '--stage',args.stage,'--count',str(args.count),'--conditional-count',str(args.conditional_count),
        '--csp-cases','0','--csp-k','20','--steps',str(args.steps),'--seed',str(args.seed),
        '--guidance',str(args.guidance),'--device',args.device,'--threads',str(args.threads)]
    if args.suite=='discovery':
        result+=['--tasks','components','dng','group','--per-group','5',
                 '--target-properties','formation_energy','band_gap','--conditional-tasks','dng']
    else:
        result+=['--tasks','csp','fixed_descriptor','--conditional-tasks']
    for name in ('cgdit_root','predictor_root'):
        value=getattr(args,name)
        if value is not None: result+=['--'+name.replace('_','-'),str(value)]
    result+=['--predictor-seed',str(args.predictor_seed)]
    for name in ('run','plan','resume','status','verify_generated','allow_proxy_transfer'):
        if getattr(args,name): result+=['--'+name.replace('_','-')]
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite',choices=['discovery','reconstruction'],default='discovery')
    parser.add_argument('--stage',choices=['generate','metrics','all'],default='generate')
    parser.add_argument('--checkpoint-root',type=Path,default=ROOT/'artifacts/runs/main15_seed42')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--count',type=int,default=1024)
    parser.add_argument('--conditional-count',type=int,default=256)
    parser.add_argument('--steps',type=int,default=64)
    parser.add_argument('--seed',type=int,default=142)
    parser.add_argument('--guidance',type=float,default=2.)
    parser.add_argument('--device',default='cuda')
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--cgdit-root',type=Path)
    parser.add_argument('--predictor-root',type=Path)
    parser.add_argument('--predictor-seed',type=int,default=123)
    for name in ('run','plan','resume','status','verify-generated','allow-proxy-transfer'):
        parser.add_argument('--'+name,action='store_true')
    args=parser.parse_args()
    subprocess.run([sys.executable,*command(args)],cwd=ROOT,check=True)


if __name__=='__main__': main()
