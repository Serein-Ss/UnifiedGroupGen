"""TRAIN-only empirical group initialization, with deterministic chemical support."""
from collections import Counter
import torch
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl, can_complete


class NoFeasibleGroup(ValueError):
    pass


def fit_group_prior(train_records):
    counts = Counter((r['descriptor']['kind'], int(r['descriptor']['number'])) for r in train_records)
    if not counts:
        raise ValueError('A group prior needs nonempty TRAIN records')
    return dict(source_split='train', train_records=sum(counts.values()),
                entries=[dict(kind=kind, number=number, count=count)
                         for (kind, number), count in sorted(counts.items())])


def sample_group(prior, generator, kind=None, chemistry=None, budget=256, max_atoms=192, max_orbits=52):
    if prior.get('source_split') != 'train':
        raise ValueError('Group frequencies must be explicitly fitted on TRAIN')
    chemistry = chemistry or ChemicalControl()
    candidates, weights = [], []
    for entry in prior['entries']:
        if kind is not None and entry['kind'] != kind:
            continue
        if not isinstance(entry['count'], int) or entry['count'] <= 0:
            raise ValueError('Empirical group weights must be positive integer counts')
        spec = catalog_spec(entry['kind'], int(entry['number']))
        if chemistry.counts and not can_complete(spec, (), (), chemistry, budget-2, max_orbits, max_atoms):
            continue
        candidates.append(spec); weights.append(entry['count'])
    if not candidates:
        raise NoFeasibleGroup('No TRAIN-frequency group has a completion for this type/inventory/budget')
    probabilities = torch.tensor(weights, dtype=torch.float64, device=generator.device)
    index = int(torch.multinomial(probabilities, 1, generator=generator))
    return candidates[index]
