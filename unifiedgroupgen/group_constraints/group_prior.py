"""TRAIN-only empirical group initialization, with deterministic chemical support."""
from collections import Counter, defaultdict
import torch
from unifiedgroupgen.common.structures import formula_key, record_counts
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl, can_complete

GROUP_POLICIES = ('prior', 'composition_ranked')


class NoFeasibleGroup(ValueError):
    pass


def _composition_keys(counts):
    return (
        ','.join(str(z) for z in sorted(counts)),
        ','.join(f'{z}:{n}' for z, n in formula_key(counts)),
        ','.join(f'{z}:{n}' for z, n in sorted(counts.items())),
    )


def fit_group_prior(train_records, *, composition_ranked=False):
    counts = Counter()
    tables = defaultdict(lambda: [defaultdict(Counter) for _ in range(3)])
    for row in train_records:
        kind, number = row['descriptor']['kind'], int(row['descriptor']['number'])
        counts[kind, number] += 1
        if composition_ranked:
            composition = record_counts(row, '.')
            for table, key in zip(tables[kind], _composition_keys(composition)):
                table[key][str(number)] += 1
    if not counts:
        raise ValueError('A group prior needs nonempty TRAIN records')
    prior = dict(source_split='train', train_records=sum(counts.values()),
                 entries=[dict(kind=kind, number=number, count=count)
                          for (kind, number), count in sorted(counts.items())])
    if composition_ranked:
        prior['composition_counts'] = {
            kind: [{key: dict(values) for key, values in table.items()} for table in levels]
            for kind, levels in tables.items()
        }
    return prior


def _rank_candidates(prior, candidates, weights, chemistry):
    total = sum(weights)
    scores = [weight / total for weight in weights]
    keys = _composition_keys(dict(chemistry.counts))
    tables = prior.get('composition_counts', {})
    for level, (key, strength) in enumerate(zip(keys, (10, 5, 5))):
        evidence = [tables[c.kind][level].get(key, {}).get(str(c.number), 0)
                    if c.kind in tables else 0 for c in candidates]
        count = sum(evidence)
        scores = [(n + strength * score) / (count + strength)
                  for n, score in zip(evidence, scores)]
    return [candidate for _, candidate in sorted(zip(scores, candidates),
            key=lambda item: (-item[0], item[1].kind, item[1].number))]


def sample_group(prior, generator, kind=None, chemistry=None, budget=256, max_atoms=192, max_orbits=52,
                 *, composition_rank=None):
    if prior.get('source_split') != 'train':
        raise ValueError('Group frequencies must be explicitly fitted on TRAIN')
    chemistry = chemistry or ChemicalControl()
    if composition_rank is not None and (composition_rank < 0 or not chemistry.counts
                                        or not prior.get('composition_counts')):
        raise ValueError('Composition ranking requires exact counts and fitted TRAIN composition counts')
    candidates, weights = [], []
    for entry in prior['entries']:
        if kind is not None and entry['kind'] != kind:
            continue
        if not isinstance(entry['count'], int) or entry['count'] <= 0:
            raise ValueError('Empirical group weights must be positive integer counts')
        spec = catalog_spec(entry['kind'], int(entry['number']))
        if chemistry.counts and not can_complete(spec, (), (), chemistry, budget-2, max_orbits, max_atoms):
            continue
        candidates.append(spec)
        weights.append(entry['count'])
    if not candidates:
        raise NoFeasibleGroup('No TRAIN-frequency group has a completion for this type/inventory/budget')
    probabilities = torch.tensor(weights, dtype=torch.float64, device=generator.device)
    index = int(torch.multinomial(probabilities, 1, generator=generator))
    if composition_rank is None:
        return candidates[index]
    # Consume the same prior draw in both modes, so downstream birth noise is paired.
    ranked = _rank_candidates(prior, candidates, weights, chemistry)
    return ranked[composition_rank % len(ranked)]
