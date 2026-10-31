"""Generate structures using a trained checkpoint and explicit task constraints."""
import argparse
import random
import numpy as np
import torch
from unifiedgroupgen.tests.legacy.generation_before_entry.pipeline import sample


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--kind', choices=['space', 'layer', 'plane'], default='space')
    parser.add_argument('--properties', default='{}')
    parser.add_argument('--composition')
    parser.add_argument('--formula', help='Reduced empirical formula; choose feasible formula-unit counts')
    parser.add_argument('--formula-units', nargs='+', type=int, default=[1, 2, 3, 4])
    parser.add_argument('--chemical-system', help='Exact element set, e.g. Li-Fe-O; all must occur')
    parser.add_argument('--allowed-elements', help='Only these elements may occur')
    parser.add_argument('--required-elements', help='All these elements must occur; extras allowed')
    parser.add_argument('--group', type=int, help='Fix group token while generating remaining orbits')
    parser.add_argument('--scaffold', help='Fixed-cell partial CIF in the supplied group setting')
    parser.add_argument('--atom-map', help='JSON observed-CIF index to full-descriptor atom index')
    parser.add_argument('--scaffold-tolerance', type=float, default=1e-4)
    parser.add_argument('--descriptor')
    parser.add_argument('--observations')
    parser.add_argument('--prior', action='store_true', help='Legal random chart prior; bypass the trained continuous flow')
    parser.add_argument('--guidance', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--count', type=int, default=10)
    parser.add_argument('--output', default='outputs/generation/samples')
    args=parser.parse_args()
    torch.set_num_threads(args.threads)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    sample(args)


if __name__=='__main__': main()
