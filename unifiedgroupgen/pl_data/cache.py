"""Current CSV-to-PyTorch cache preparation, storage and identity checks."""
from collections import deque
from functools import lru_cache
import hashlib
import json
import multiprocessing
from pathlib import Path

import pandas as pd
import torch

from unifiedgroupgen.common.configuration import configuration_root, load_yaml, resolve_path
from unifiedgroupgen.common.data_utils import (
    JointTarget, convert_row, crystal_graph_arrays, target_from_record,
)
from unifiedgroupgen.common.source import file_hash
from unifiedgroupgen.group_constraints.state import JointState
from unifiedgroupgen.group_constraints.unified import SymmetrySpec


# Persistent, task-independent CPU Tensor caches. Change the version when a
# conversion or serialization rule changes; network weights are never cached.
TENSOR_CACHE_VERSION = 3
_TRANSFER_ARRAYS = False
_CONVERSION_KEYS = ('dataset','kind','coordinate_mode','symmetry_tolerance','group_column',
                    'property_columns','max_atoms','max_orbits','layer_group_policy',
                    'symmetry_protocol','angle_tolerance','cif_fractional_tolerance','niggli')


def symmetry_protocol_identity(config):
    """Version the numerical protocol independently of the Tensor storage format."""
    if not config.get('symmetry_protocol'):
        return None
    from importlib.metadata import version
    core = Path(__file__).parents[1]
    files = [core/'common/data_utils.py', core/'group_constraints/encoding.py',
             core/'group_constraints/compiler.py',
             core/'group_constraints/operations.py', core/'group_constraints/unified.py']
    return dict(protocol=config['symmetry_protocol'],
                versions={name:version(name) for name in ('numpy','pymatgen','spglib','pyxtal')},
                implementation={p.name:hashlib.sha256(p.read_bytes().replace(b'\r\n', b'\n')).hexdigest() for p in files})


def tensor_cache_identity(config, split, csv, operation_compiler, graph_options):
    identity = dict(version=TENSOR_CACHE_VERSION,dataset=config['dataset'],split=split,
                    source_sha256=file_hash(csv),conversion={k:config.get(k) for k in _CONVERSION_KEYS},
                    operation_compiler=operation_compiler,graph_options=graph_options)
    if config.get('symmetry_protocol'):
        identity['symmetry_protocol'] = symmetry_protocol_identity(config)
    return identity


@lru_cache(maxsize=512)
def _cached_spec(text,digest):
    if hashlib.sha256(text.encode()).hexdigest()!=digest:
        raise ValueError('Cached compiled symmetry metadata changed')
    return SymmetrySpec.from_dict(json.loads(text),_compiled=True)


def target_from_cache(payload, index, record=None):
    item=payload['targets'][index];record=record or payload['records'][index]
    digest=payload['spec_hashes'][item['spec']]
    spec=_cached_spec(payload['specs'][item['spec']],digest)
    state=JointState(spec,tuple(item['letters']),tuple(item['elements']),item['metric'],item['coordinates'],
                     item.get('moments'),status='done')
    return JointTarget(state,str(record['id']),dict(record.get('properties',{})))


@lru_cache(maxsize=8)
def _load_tensor_cache(path, modified, size):
    payload=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    if payload.get('version') != TENSOR_CACHE_VERSION:
        raise ValueError('Expected the current Tensor cache format')
    return _unpack_tensor_payload(payload)


def load_tensor_cache(path):
    path=Path(path).resolve();stat=path.stat()
    payload=_load_tensor_cache(str(path),stat.st_mtime_ns,stat.st_size)
    if not len(payload['records'])==len(payload['targets'])==len(payload['graphs']):
        raise ValueError('Incomplete Tensor cache: '+str(path))
    return payload


class _TensorRows:
    """Zero-copy per-structure views into a small number of packed storages."""
    def __init__(self,fields,count):
        self.fields=fields;self.count=count
        self.offsets={k:v['offsets'].tolist() for k,v in fields.items() if v is not None and 'offsets' in v}
        self.scalars={k:v['values'].tolist() for k,v in fields.items() if v is not None and 'offsets' not in v}

    def __len__(self):return self.count

    def __getitem__(self,index):
        if not 0<=index<self.count:raise IndexError(index)
        result={}
        for key,field in self.fields.items():
            if field is None:result[key]=None
            elif key in self.scalars:result[key]=self.scalars[key][index]
            else:
                offsets=self.offsets[key]
                result[key]=field['values'].narrow(field['dim'],offsets[index],offsets[index+1]-offsets[index])
        return result


class _TargetRows(_TensorRows):
    def __init__(self,fields,metadata):
        super().__init__(fields,len(metadata));self.metadata=metadata

    def __getitem__(self,index):return {**self.metadata[index],**super().__getitem__(index)}


def _pack_tensor_rows(rows):
    result={}
    for key in rows[0]:
        values=[row[key] for row in rows]
        if all(value is None for value in values):result[key]=None
        elif torch.is_tensor(values[0]):
            dim=1 if key=='edge_index' else 0
            sizes=torch.tensor([value.shape[dim] for value in values],dtype=torch.long)
            result[key]=dict(values=torch.cat(values,dim=dim),dim=dim,
                offsets=torch.cat((torch.zeros(1,dtype=torch.long),sizes.cumsum(0))))
        else:result[key]=dict(values=torch.tensor(values,dtype=torch.long))
    return result


def _cache_records_json(records, dataset):
    fields = ('id', 'descriptor', 'state', 'properties', 'symmetry', 'public_state',
              'magnetic_moments', 'moment_order')
    if any(record['dataset'] != dataset for record in records):
        raise ValueError('Training cache contains a record from a different dataset')
    return json.dumps([{key: record[key] for key in fields if key in record} for record in records],
                      ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _pack_tensor_payload(payload,identity):
    metadata=[{key:row[key] for key in ('spec','letters','elements')} for row in payload['targets']]
    tensors=[{key:row[key] for key in ('metric','coordinates','moments')} for row in payload['targets']]
    return dict(version=TENSOR_CACHE_VERSION,identity=identity,
        records_json=_cache_records_json(payload['records'], identity['dataset']),
        target_meta_json=json.dumps(metadata,separators=(',',':')),
        specs=payload['specs'],spec_hashes=[hashlib.sha256(s.encode()).hexdigest() for s in payload['specs']],
        target_tensors=_pack_tensor_rows(tensors),graph_tensors=_pack_tensor_rows(payload['graphs']))


def _unpack_tensor_payload(payload):
    records=json.loads(payload['records_json']);metadata=json.loads(payload['target_meta_json'])
    for record in records:
        record['dataset'] = payload['identity']['dataset']
    return dict(version=payload['version'],identity=payload['identity'],
                records=records,specs=payload['specs'],spec_hashes=payload['spec_hashes'],
                targets=_TargetRows(payload['target_tensors'],metadata),
                graphs=_TensorRows(payload['graph_tensors'],len(records)))


def _publish_tensor_cache(path,payload):
    from uuid import uuid4
    temporary=path.with_name(path.name+'.'+uuid4().hex+'.tmp')
    try:
        torch.save(payload,temporary)
        temporary.replace(path)
    finally:temporary.unlink(missing_ok=True)


def write_tensor_cache_config(path, identity, source, count, **build):
    """Record the effective settings once, beside the cache they produced."""
    from unifiedgroupgen.common.structures import write_json

    path = Path(path)
    config_path = path.with_suffix('.config.json')
    write_json(config_path, dict(schema='ugg_tensor_cache_config_v1', cache=path.name,
        source_csv=Path(source).as_posix(), records=count, identity=identity,
        graph_policy=dict(keep_equal_distance_shells=True, allow_large_image_search=True),
        build=build))
    return config_path


def records_from_cache(path):
    payload=load_tensor_cache(path)
    resolved=str(Path(path).resolve())
    return [{**r,'_tensor_cache':dict(path=resolved,index=i)}
            for i,r in enumerate(payload['records'])]


class _TensorTransfer:
    """Transfer small CPU tensors as bytes, avoiding per-tensor shared handles."""
    def __init__(self,tensor):self.array=tensor.detach().numpy()

    def __reduce__(self):return torch.from_numpy,(self.array,)


def _prepare_cached_graph(item):
    record,operation_compiler,graph_options=item
    target=target_from_record(record,operation_compiler=operation_compiler)
    state=target.state
    values=dict(letters=state.letters,elements=state.elements,metric=state.metric,
                coordinates=state.coordinates,moments=state.moments)
    graph=crystal_graph_arrays(target,**graph_options)
    if _TRANSFER_ARRAYS:
        values={k:_TensorTransfer(v) if torch.is_tensor(v) else v for k,v in values.items()}
        graph={k:_TensorTransfer(v) if torch.is_tensor(v) else v for k,v in graph.items()}
    return json.dumps(state.spec.to_dict(),sort_keys=True),values,graph


def _build_tensor_payload(records, identity, operation_compiler, graph_options, workers=0):
    from unifiedgroupgen.common.progress import track
    from concurrent.futures import ProcessPoolExecutor
    from contextlib import nullcontext
    specs=[];lookup={};targets=[];graphs=[]
    pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
                             initializer=_prepare_worker) if workers else nullcontext(None)
    with pool as executor:
        items=((record,operation_compiler,graph_options) for record in records)
        prepared=executor.map(_prepare_cached_graph,items,chunksize=16) if executor else map(_prepare_cached_graph,items)
        for key,values,graph in track(prepared,'Cache graphs',total=len(records),unit='structure'):
            if key not in lookup:
                lookup[key]=len(specs);specs.append(key)
            targets.append(dict(spec=lookup[key],**values));graphs.append(graph)
    return _pack_tensor_payload(dict(records=records,specs=specs,targets=targets,graphs=graphs),identity)


def _prepare_worker():
    import torch
    from threadpoolctl import threadpool_limits
    torch.set_num_threads(1)
    # Small per-crystal matrices must not start a full BLAS thread team in
    # every preparation process. This changes worker resources, not training.
    threadpool_limits(limits=1, user_api='blas')
    global _TRANSFER_ARRAYS
    _TRANSFER_ARRAYS=True


def _parallel_rows(items, workers, timeout):
    """Bound queued work and fail visibly if a native-library worker stops responding."""
    pool = multiprocessing.get_context('spawn').Pool(workers, initializer=_prepare_worker)
    pending = deque()
    source = iter(items)
    try:
        for _ in range(workers*2):
            item = next(source, None)
            if item is not None:
                pending.append((item[0], pool.apply_async(convert_row, (item,))))
        while pending:
            index, result = pending.popleft()
            try:
                yield result.get(timeout=timeout)
            except multiprocessing.TimeoutError as error:
                raise RuntimeError(f'Preparation worker timed out at source row {index}. '
                                   'Inspect that row or retry with fewer workers.') from error
            item = next(source, None)
            if item is not None:
                pending.append((item[0], pool.apply_async(convert_row, (item,))))
        pool.close()
    except BaseException:
        pool.terminate()
        raise
    finally:
        pool.join()


def expected_tensor_cache(config, split, root, operation_compiler, graph_options):
    """Resolve a cache identity without creating files or building targets."""
    root = Path(root).resolve()
    if split not in ('train','val','test'):raise ValueError('Unknown dataset split')
    name=config['dataset']
    if Path(name).name!=name or any(c in name for c in '/\\:'):
        raise ValueError('Dataset name must be one path component')
    csv=resolve_path(Path(config['data_root'])/(split+'.csv'),root)
    source = csv.relative_to(root) if csv.is_relative_to(root) else csv
    graph_options={**dict(cutoff=6.,max_neighbors=64,max_images=16),**(graph_options or {})}
    identity=tensor_cache_identity(config,split,csv,operation_compiler,graph_options)
    fingerprint=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:16]
    path=root/'data/dataset'/name/(split+'.'+fingerprint+'.pt')
    return path, identity, source


def ensure_tensor_cache(config, split, root=None, operation_compiler=True, graph_options=None,
                        workers=0):
    """Build or reuse the content-addressed cache for the current CSV."""
    from unifiedgroupgen.common.configuration import project_root
    from filelock import FileLock
    root = Path(root).resolve() if root is not None else project_root()
    path, identity, source = expected_tensor_cache(config, split, root, operation_compiler, graph_options)
    csv = resolve_path(source, root)
    graph_options = identity['graph_options']
    path.parent.mkdir(parents=True,exist_ok=True)
    with FileLock(str(path)+'.lock',timeout=3600):
        if path.is_file():
            payload=load_tensor_cache(path)
            if payload['identity']!=identity:raise ValueError('Tensor cache identity mismatch')
            if not path.with_suffix('.config.json').is_file():
                write_tensor_cache_config(path, identity, source, len(payload['records']), mode='csv')
            return path,payload,True
        frame = pd.read_csv(csv)
        records = []
        from unifiedgroupgen.common.progress import track
        conversion = dict(config, _training_cache=True)
        items = ((int(index), row.to_dict(), conversion, str(csv)) for index, row in frame.iterrows())
        converted = _parallel_rows(items, workers, config.get('preparation_row_timeout', 120)) if workers else map(convert_row, items)
        for result in track(converted, 'Process CSV', total=len(frame), unit='structure'):
            if 'rejection' in result:
                raise ValueError('CSV preprocessing failed: '+json.dumps(result['rejection']))
            records.append(result['record'])
        if not records:raise ValueError('Dataset split is empty')
        if len({(r['dataset'],r['id']) for r in records})!=len(records):
            raise ValueError('Duplicate structure identities in split')
        payload=_build_tensor_payload(records,identity,operation_compiler,graph_options,workers)
        _publish_tensor_cache(path, payload)
        write_tensor_cache_config(path, identity, source, len(records), mode='csv', workers=workers,
                                  row_wait_timeout_seconds=config.get('preparation_row_timeout', 120) if workers else None)
    return path, load_tensor_cache(path), False


def cache_paths(config, split, root):
    """Resolve source profiles and graph settings identically for train and audit."""
    graph = {key: config['model'][key] for key in ('cutoff', 'max_neighbors', 'max_images')}
    paths = []
    for name, source in config['source_configs'].items():
        profile = load_yaml(source, configuration_root())
        if profile['dataset'] != name:
            raise ValueError(f'Source profile {source} declares {profile["dataset"]}, expected {name}')
        workers = profile.get('preprocessing', {}).get('workers', 0)
        paths.append(ensure_tensor_cache(profile, split, root,
                     operation_compiler=config['model']['operation_compiler'],
                     graph_options=graph, workers=workers)[0])
    return paths


def training_split_hashes(paths):
    """Bind current checkpoints to the actual Tensor caches used."""
    return {split: {str(path): file_hash(path) for path in files} for split, files in paths.items()}
