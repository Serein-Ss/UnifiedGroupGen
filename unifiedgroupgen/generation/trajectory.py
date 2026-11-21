"""Streaming, opt-in structure trajectories; no random draws or model changes."""
import json
import math
from pathlib import Path
import torch
from unifiedgroupgen.group_constraints.wyckoff import orbit_labels, event_orbit_label


def _plain(value):
    if torch.is_tensor(value):
        return _plain(value.detach().cpu().tolist())
    if isinstance(value,dict):
        return {str(k):_plain(v) for k,v in value.items()}
    if isinstance(value,(tuple,list)):
        return [_plain(v) for v in value]
    if isinstance(value,float) and not math.isfinite(value):
        return None
    return value


def snapshot(state, requested_group=None):
    """Keep raw state even if a failed geometry can no longer be decoded."""
    result=dict(num_atoms=state.num_atoms,orbit_labels=orbit_labels(state.spec, state.letters, requested_group),
                letters=state.letters,orbit_elements=state.elements,
                metric_state=state.metric,coordinate_state=state.coordinates,
                frozen=state.frozen,state_status=state.status,orbit_step=state.step,
                budget=state.budget,periodic=state.spec.periodic,
                frame=state.frame,origin=state.origin,
                chemical_control=state.chemistry.to_dict())
    try:
        decoded=state.decoded()
        unwrapped=state.decoded(wrap=False)
        result.update({name:decoded[name] for name in
                       ('lattice','coordinates','cartesian_coordinates','atomic_numbers','orbit_index')})
        result['coordinates_unwrapped']=unwrapped['coordinates']
        result['remaining_inventory']=dict(state.chemistry.remaining(state.spec,state.letters,state.elements))
        result['geometry_finite']=bool(torch.isfinite(decoded['lattice']).all() and
                                       torch.isfinite(decoded['coordinates']).all())
    except (ValueError,RuntimeError,FloatingPointError) as error:
        result.update(geometry_finite=False,decode_error=str(error))
    return _plain(result)


class TrajectoryWriter:
    """One JSONL per global sample index, at most one batch of open streams.

    start() writes metadata, record() streams sampler frames, finish() appends
    the final ledger result and closes the stream. Existing files are preserved.
    """
    def __init__(self,directory,count,indices=None,metadata=None):
        self.directory=Path(directory)
        self.indices=None if indices is None else set(indices)
        if self.indices is not None and any(not 0<=i<count for i in self.indices):
            raise ValueError('Trajectory indices must lie in [0, count)')
        if self.directory.exists() and not self.directory.is_dir():
            raise NotADirectoryError(str(self.directory))
        for i in range(count) if self.indices is None else sorted(self.indices):
            if self.path(i).exists():raise FileExistsError(str(self.path(i)))
        self.metadata=metadata or {}
        self.streams={};self.frames={};self.groups={};self.completed=0

    def path(self,index):
        return self.directory/f'sample_{index:06d}.jsonl'

    def selected(self,index):
        return self.indices is None or index in self.indices

    def _write(self,index,value):
        stream=self.streams[index]
        stream.write(json.dumps(_plain(value),ensure_ascii=False,allow_nan=False)+'\n')
        stream.flush()

    def start(self,index,seed,spec,requested_group,chemistry,properties,*,label_group=None):
        if not self.selected(index):return
        self.directory.mkdir(parents=True,exist_ok=True)
        self.streams[index]=self.path(index).open('x',encoding='utf-8')
        self.frames[index]=0
        self.groups[index]=requested_group if label_group is None else label_group
        self._write(index,dict(record_type='metadata',schema='ugg_joint_flow_trajectory_v1',
                    index=index,seed=seed,requested_group=requested_group,
                    symmetry=None if spec is None else spec.to_dict(),
                    chemical_control=chemistry.to_dict(),properties_requested=properties,
                    coordinate_convention='fractional periodic components; angstrom nonperiodic components; lattice vectors are columns; MASK=0',
                    frame_policy='initial + every committed integration step + post-event + final + optional contact repair; no predictor/root-search states',
                    **self.metadata))

    def record(self,index,stage,state,time,integration_step,event=None,reason=None):
        if not self.selected(index):return
        group=self.groups[index]
        frame=dict(record_type='frame',frame_index=self.frames[index],stage=stage,
                   time=time,integration_step=integration_step,event=event,reason=reason,
                   **snapshot(state, group))
        if (label := event_orbit_label(state.spec, event, group)) is not None:
            frame['event_orbit_label']=label
        self._write(index,frame)
        self.frames[index]+=1

    def finish(self,index,row):
        if not self.selected(index):return
        row.update(trajectory_path=str(self.path(index)),trajectory_frames=self.frames[index])
        self._write(index,dict(record_type='result',frame_count=self.frames[index],sample=row))
        self.streams.pop(index).close();self.frames.pop(index);self.completed+=1
        self.groups.pop(index)

    def __enter__(self):
        return self

    def __exit__(self,kind,error,traceback):
        try:
            if error is not None:
                for index in self.streams:
                    self._write(index,dict(record_type='interrupted',reason=str(error),frame_count=self.frames[index]))
        finally:
            for stream in self.streams.values():stream.close()
            self.streams.clear();self.frames.clear();self.groups.clear()
