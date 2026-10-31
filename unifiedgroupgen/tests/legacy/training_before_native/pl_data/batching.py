import random
import torch
from unifiedgroupgen.group_constraints.compiler import Descriptor, compile_descriptor

def _training_entry(record, dtype):
    if '_training' in record:
        return record['_training']
    descriptor = Descriptor.from_dict(record['descriptor'])
    compiled = compile_descriptor(descriptor)
    endpoint = torch.tensor(record['state'], dtype=dtype)
    if endpoint.shape != (compiled.dof,) or not torch.isfinite(endpoint).all():
        raise ValueError(f"Invalid training state in {record['id']}")
    return descriptor, compiled, endpoint

def prepare_training_records(records, dtype=torch.float32):
    """Validate/cache immutable chart metadata and CPU endpoints once per run."""
    return [{**record, '_training': _training_entry(record, dtype)} for record in records]

def batch_indices(records, batch_size, edge_budget, shuffle=True, rng=None):
    """Size buckets reduce variance while bounding the total full-graph edge count."""
    if batch_size < 1 or edge_budget < 1:
        raise ValueError('Batch size and edge budget must be positive')
    rng = random if rng is None else rng
    sizes = [(r['_training'][1] if '_training' in r else compile_descriptor(
        Descriptor.from_dict(r['descriptor']))).num_atoms for r in records]
    order = list(range(len(records)))
    if shuffle:
        rng.shuffle(order)
        order = [j for start in range(0, len(order), batch_size*32)
                 for j in sorted(order[start:start+batch_size*32], key=lambda i: sizes[i])]
    batches, current, edges = [], [], 0
    for index in order:
        required = sizes[index]*(sizes[index]-1)
        if required > edge_budget:
            raise ValueError(f"Single structure {records[index]['id']} needs {required} edges, budget={edge_budget}")
        if current and (len(current) >= batch_size or edges+required > edge_budget):
            batches.append(current)
            current, edges = [], 0
        current.append(index)
        edges += required
    if current:
        batches.append(current)
    if shuffle:
        rng.shuffle(batches)
    return batches
