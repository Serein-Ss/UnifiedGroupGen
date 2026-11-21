"""Before/after CPU fingerprints for the current model; no fit/optimizer steps."""
import hashlib
import json
from pathlib import Path
import random
import sys

import numpy as np
import torch


def digest(tensor):
    value = tensor.detach().cpu().contiguous()
    return dict(shape=list(value.shape), dtype=str(value.dtype),
                sha256=hashlib.sha256(value.numpy().tobytes()).hexdigest())


def capture():
    from unifiedgroupgen.pl_modules.factory import build_joint_model
    from unifiedgroupgen.common.data_utils import target_from_record
    from unifiedgroupgen.pl_data.unified.tasks import task_target
    from unifiedgroupgen.pl_data.unified.path import sample_path
    from unifiedgroupgen.pl_modules.unified.objectives import loss
    from unifiedgroupgen.generation.unified.sampling import sample_batch
    from unifiedgroupgen.group_constraints.unified import catalog_spec
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parents[3]
    result = {}
    for backend in ('equiformer_v2', 'joint_atom_graph'):
        for dataset in ('mp_20', 'c2db_51'):
            record=json.loads((root/'unifiedgroupgen/tests/fixtures/current_training_records.json').read_text(encoding='utf-8'))[dataset]['train']
            for task in ('dng', 'csp', 'property'):
                torch.manual_seed(91); random.seed(91); np.random.seed(91)
                config = dict(model=dict(family='joint_flow_v2', backbone=backend, hidden=16, layers=1,
                    channels=16, heads=4, lmax=2, mmax=1, cutoff=4., max_neighbors=32,
                    use_gate_act=True, task_controls=True), allowed_elements=list(range(1,119)),
                    properties={'band_gap':dict(type='scalar')} if task=='property' else {},
                    generation_task=task, chemical_set_probability=0., cond_dropout=0.,
                    max_atoms=192, max_orbits=52, event_budget=96, birth_scale=.2)
                model = build_joint_model(config).eval()
                target = task_target(target_from_record(record), config, 12)
                samples = [sample_path(target, seed, time=t, allowed_elements=config['allowed_elements'])
                           for seed,t in ((3,.2),(4,.67),(5,.92))]
                values = loss(model, samples)
                values['loss'].backward()
                name = backend+'/'+dataset+'/'+task
                result[name] = dict(losses={k:digest(v) for k,v in values.items()},
                    parameters={k:digest(v) for k,v in model.state_dict().items()},
                    gradients={k:digest(p.grad) for k,p in model.named_parameters() if p.grad is not None})
                if task == 'dng':
                    with torch.no_grad():
                        outputs = model.forward_batch([s.state for s in samples], [s.time for s in samples])
                    result[name]['forward'] = [{k:digest(v) for k,v in output.items() if torch.is_tensor(v)}
                                               for output in outputs]
                    generated = sample_batch(model, [catalog_spec('space',1),catalog_spec('layer',1),catalog_spec('plane',1)],
                                             seeds=[7,8,9], steps=3, budget=10, epsilon=.005)
                    result[name]['sampling'] = [dict(status=s.state.status, events=s.events,
                        metric=digest(s.state.metric), coordinates=digest(s.state.coordinates)) for s in generated]
    return result


if __name__ == '__main__':
    Path(sys.argv[1]).write_text(json.dumps(capture(),sort_keys=True,indent=2),encoding='utf-8')
