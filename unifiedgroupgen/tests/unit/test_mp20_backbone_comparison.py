"""Release checks for the paired MP-20 backbone and group-selection paths."""
import json
from dataclasses import replace

import pytest
import torch

from submit_files.joint_flow5090 import catalog, plan
from submit_files.joint_flow5090_calibrate import model_policy
from unifiedgroupgen.experiment_config import compose_experiment, training_definition
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl
from unifiedgroupgen.group_constraints.compiler import group_table
from unifiedgroupgen.group_constraints.flow_state import JointAction, apply, initial
from unifiedgroupgen.group_constraints.group_prior import CompositionNeighbors, fit_group_prior, sample_group
from unifiedgroupgen.group_constraints.operations import operation_spec
from unifiedgroupgen.pl_data.unified.path import FlowSample
from unifiedgroupgen.pl_modules.factory import build_joint_model
from unifiedgroupgen.pl_modules.unified.objectives import loss


def test_composition_ranking_uses_train_counts_and_keeps_prior_rng():
    def row(group, element):
        letter = next(w.letter for w in group_table('space', group) if w.multiplicity == 1)
        orbit = dict(letter=letter, atomic_number=element)
        return dict(descriptor=dict(kind='space', number=group, orbits=[orbit]))

    training = [row(1, 8), row(2, 14), row(2, 14), row(2, 14)]
    legacy = fit_group_prior(training)
    prior = json.loads(json.dumps(fit_group_prior(training, composition_ranked=True)))
    assert {k: v for k, v in prior.items() if k != 'composition_counts'} == legacy
    chemistry = ChemicalControl.from_inputs(counts={8: 1})
    first, other = [torch.Generator().manual_seed(81) for _ in range(2)]
    args = dict(kind='space', chemistry=chemistry, budget=8)
    legacy_number = sample_group(legacy, first, **args).number
    assert sample_group(prior, other, **args, composition_rank=0).number == 1
    assert torch.equal(first.get_state(), other.get_state())
    first.manual_seed(81)
    assert sample_group(prior, first, **args).number == legacy_number
    assert sample_group(prior, other, **args, composition_rank=1).number == 2
    with pytest.raises(ValueError, match='fitted TRAIN composition counts'):
        sample_group(legacy, other, **args, composition_rank=0)


def test_backbone_plan_preserves_each_models_fixed_batch_size():
    names = catalog()['backbone_comparison']['models']
    result = plan(names)
    assert result['jobs'] == 4
    assert result['calibration']['candidates'] == [8, 32]
    sizes = {'equiformer_v2': 8, 'cspnet': 32}
    for backbone, batch_size in sizes.items():
        selected = [name for name in names if result['entries'][name]['backbone'] == backbone]
        assert len(selected) == 2
        assert plan(selected)['calibration']['candidates'] == [batch_size]
        for name in selected:
            assert model_policy(result, name)['candidates'] == [batch_size]
    assert len(plan()['selected']) == 2
    assert {(result['available'][n]['backbone'], result['available'][n]['task']) for n in names} == {
        (backbone, task) for backbone in ('equiformer_v2', 'cspnet') for task in ('dng', 'csp')
    }


def test_neighbor_prior_keeps_kinds_and_sampler_rng_separate():
    prior = dict(source_split='train', train_records=410,
                 entries=[dict(kind='space', number=1, count=10),
                          dict(kind='space', number=2, count=200),
                          dict(kind='layer', number=1, count=200)],
                 composition_counts={
                     'space': [{}, {}, {'8:2': {'1': 10}, '14:1': {'2': 200}}],
                     'layer': [{}, {}, {'8:2': {'1': 200}}],
                 })
    neighbors = CompositionNeighbors(json.loads(json.dumps(prior)))
    chemistry = ChemicalControl.from_inputs(counts={8: 2})
    assert neighbors.votes('space', chemistry.counts)[1] > neighbors.votes('space', chemistry.counts)[2]
    assert set(neighbors.votes('layer', chemistry.counts)) == {1}
    plain, reranked = [torch.Generator().manual_seed(199) for _ in range(2)]
    args = dict(kind='space', chemistry=chemistry, budget=8, composition_rank=0)
    sample_group(prior, plain, **args)
    sample_group(prior, reranked, neighbors=neighbors, **args)
    assert torch.equal(plain.get_state(), reranked.get_state())


@pytest.mark.parametrize('name', catalog()['backbone_comparison']['models'])
def test_real_backbone_joint_geometry_gradients(name):
    torch.set_num_threads(1)
    cfg = compose_experiment(name)
    definition = training_definition(cfg)
    assert cfg.initialize_from is None and cfg.resume is None
    assert cfg.train.early_stopping.patience == 30
    batch_size = {'equiformer_v2': 8, 'cspnet': 32}[definition['model']['backbone']]
    assert cfg.runtime.batch_size == batch_size and cfg.runtime.strict_fp32
    task = definition['generation_task']
    assert definition.get('csp_group_policy', 'prior') == ('composition_ranked' if task == 'csp' else 'prior')
    model = build_joint_model({**definition, 'allowed_elements': [8, 14]}, 'cpu')
    spec = operation_spec('space', 1)
    chemistry = ChemicalControl.from_inputs(counts={8: 1}) if task == 'csp' else None
    rng = torch.Generator().manual_seed(11)
    state = initial(spec, rng, budget=96, chemistry=chemistry)
    fields = 3.2 * torch.eye(3)
    metric = torch.einsum('kij,ij->k', fields.new_tensor(state.compiled.standard), fields)
    state = replace(state, metric=metric)
    state = apply(state, JointAction('birth', letter=spec.templates[0].letter, element=8), rng)
    sample = FlowSample(state, .6, state.metric+0.1, state.coordinates+0.02, (), 1, JointAction('freeze'), {}, 'probe')
    losses = loss(model, [sample], geometry_objective=definition['geometry_objective'])
    assert {'lattice_shape', 'lattice_log_measure', 'orbit_site', 'jump', 'loss'} <= losses.keys()
    assert all(torch.isfinite(value) for value in losses.values())
    losses['loss'].backward()
    assert model.backbone.context.weight.grad is not None
    assert torch.isfinite(model.backbone.context.weight.grad).all()
