"""Export one successful sample from a JSONL ledger without rewriting the ledger."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch  # initialize the project's OpenMP runtime before pymatgen/numpy

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from unifiedgroupgen.evaluation_io import inline_cif_bytes, load_structure


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples',type=Path,required=True)
    parser.add_argument('--index',type=int,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.index<0: parser.error('index must be nonnegative')
    record=None
    with args.samples.open(encoding='utf-8-sig') as stream:
        for line in stream:
            if not line.strip(): continue
            item=json.loads(line)
            if item.get('index')==args.index:
                record=item;break
    if record is None: raise ValueError(f'No sample with index {args.index}')
    if record.get('status','generated')!='generated':
        raise ValueError('Requested sample failed generation; its failure record was preserved')
    structure=load_structure(record,args.samples.parent)
    if 'cif' in record:
        content=inline_cif_bytes(record)
    elif record.get('cif_path'):
        path=Path(record['cif_path'])
        content=(path if path.is_absolute() else args.samples.parent/path).read_bytes()
    else:
        content=structure.to(fmt='cif').encode('utf-8')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('xb') as stream: stream.write(content)
    print(json.dumps({'index':args.index,'output':str(args.output),'cif_sha256':hashlib.sha256(content).hexdigest()},ensure_ascii=False))


if __name__=='__main__': main()
