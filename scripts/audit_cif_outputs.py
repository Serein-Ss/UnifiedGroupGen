"""Read-only CIF round-trip audit; no generation, repair or occupancy rescaling."""
import argparse
import csv
import json
from pathlib import Path
import sys
import time
import warnings
import torch
import numpy as np
from pymatgen.io.cif import CifParser
from pymatgen.core import Lattice,Structure

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from unifiedgroupgen.evaluation_io import sha256,write_json


def audit(root,output):
    root,output=Path(root).resolve(),Path(output).resolve()
    if not root.is_dir(): raise FileNotFoundError(root)
    files=sorted(root.rglob('*.cif'));failures=[];started=time.perf_counter()
    if not files: raise ValueError('No CIF files in the supplied directory')
    for i,path in enumerate(files,1):
        with warnings.catch_warnings(record=True) as messages:
            warnings.simplefilter('always')
            try:
                parser=CifParser(path,frac_tolerance=0)
                structure=parser.parse_structures(primitive=False)[0]
                if not np.isfinite(structure.lattice.matrix).all() or not np.isfinite(structure.frac_coords).all():
                    raise FloatingPointError('Nonfinite lattice or coordinates')
            except (ValueError,KeyError,IndexError,OSError,RuntimeError,FloatingPointError) as error:
                row={'file':str(path.relative_to(root)),'sha256':sha256(path),'error_type':type(error).__name__,
                     'reason':str(error),'parser_warnings':[str(m.message) for m in messages]}
                # Raw atom-table diagnostic bypasses site merging ONLY to diagnose failure.
                # It never replaces the failed structure in evaluation or writes repaired CIFs.
                try:
                    cell=next(iter(CifParser(path,frac_tolerance=0).as_dict().values()))
                    lattice=Lattice.from_parameters(*[float(cell['_cell_length_'+a]) for a in 'abc'],
                        *[float(cell['_cell_angle_'+a]) for a in ('alpha','beta','gamma')])
                    xyz=np.column_stack([[float(v) for v in cell['_atom_site_fract_'+a]] for a in 'xyz'])
                    raw=Structure(lattice,cell['_atom_site_type_symbol'],xyz)
                    distance=raw.distance_matrix.copy();np.fill_diagonal(distance,np.inf)
                    row['raw_sites']=len(raw)
                    value=float(distance.min()) if len(raw)>1 else None
                    row['minimum_raw_pair_distance_angstrom']=value if value is not None and np.isfinite(value) else None
                except (ValueError,KeyError,IndexError,TypeError,StopIteration):
                    pass
                failures.append(row)
        if i%1000==0: print(json.dumps({'checked':i,'files':len(files),'failed':len(failures)}),flush=True)
    report={'root':str(root),'files':len(files),'readback_succeeded':len(files)-len(failures),
            'failed':failures,'seconds':time.perf_counter()-started,
            'scope':'Parsing and finite-value audit; raw periodic pair distances diagnose failed CIFs, not physical validity'}
    output.mkdir(parents=True,exist_ok=True);write_json(output/'audit.json',report)
    fields=sorted({k for r in failures for k in r}) or ['file','reason']
    with (output/'failures.csv').open('w',newline='',encoding='utf-8-sig') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        writer.writerows({k:json.dumps(v) if isinstance(v,list) else v for k,v in r.items()} for r in failures)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();result=audit(args.root,args.output)
    print(json.dumps({k:v for k,v in result.items() if k!='failed'}),flush=True)
