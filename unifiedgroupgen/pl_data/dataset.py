"""Single-crystal PyG objects and automatic CSV-to-Tensor cache loading."""
from pathlib import Path
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data
from unifiedgroupgen.common.configuration import resolve_path
from unifiedgroupgen.common.data_utils import crystal_graph_arrays, target_from_record
from unifiedgroupgen.pl_data.cache import cache_paths, load_tensor_cache, records_from_cache, target_from_cache


class CrystalData(Data):
    """One endpoint graph; record/target metadata never enter neural features."""
    def __inc__(self,key,value,*args,**kwargs):
        if key=='orbit_index':return self.num_orbits
        return super().__inc__(key,value,*args,**kwargs)

    @property
    def orbit_labels(self):
        from unifiedgroupgen.group_constraints.wyckoff import record_orbit_labels
        return record_orbit_labels(self.record, self.record['_joint_target'].state)


class CrystalDataset(Dataset):
    def __init__(self,records,operation_compiler=False):
        self.records=[];self.graphs=[];payloads={}
        names=sorted({key for row in records for key in row.get('properties',{})})
        for row in records:
            pointer=row.get('_tensor_cache')
            if pointer:
                path=pointer['path']
                if path not in payloads:payloads[path]=load_tensor_cache(path)
                payload=payloads[path];index=pointer['index']
                if payload['identity']['operation_compiler']!=operation_compiler:
                    raise ValueError('Cached endpoint compiler differs from the model')
                target=target_from_cache(payload,index,row)
                arrays=payload['graphs'][index]
            else:
                target=row.get('_joint_target') or target_from_record(row,operation_compiler=operation_compiler)
                arrays=crystal_graph_arrays(target)
            record={**row,'_joint_target':target}
            graph=CrystalData(**arrays)
            values=[row.get('properties',{}).get(name) for name in names]
            mask=torch.tensor([[v is not None for v in values]],dtype=torch.bool)
            graph.property_values=torch.tensor([[0. if v is None else float(v) for v in values]],dtype=torch.float32)
            graph.property_mask=mask
            object.__setattr__(graph,'record',record)
            self.records.append(record);self.graphs.append(graph)

    def __len__(self):return len(self.graphs)

    def __getitem__(self,index):return self.graphs[index]

def load_training_splits(config, root, train=None, val=None):
    root = Path(root).resolve()
    paths = {}
    splits = {}
    for split, explicit in [('train', train), ('val', val)]:
        files = [resolve_path(p, root) for p in explicit] if explicit else cache_paths(config, split, root)
        rows = []
        for path in files:
            payload = load_tensor_cache(path)
            if payload['identity']['split'] != split:
                raise ValueError(f'Expected a {split} cache: {path}')
            rows.extend(records_from_cache(path))
        paths[split] = files
        splits[split] = rows
    records,validation=splits['train'],splits['val']
    if not records or not validation:
        raise ValueError('Training and validation records must both be nonempty')
    if {(r['dataset'], r['id']) for r in records} & {(r['dataset'], r['id']) for r in validation}:
        raise ValueError('Training/validation identifier overlap')
    return records, validation, paths
