"""TRAIN-only empirical group initialization, with deterministic chemical support."""
from collections import Counter, defaultdict
from functools import lru_cache
import numpy as np
import torch
from unifiedgroupgen.common.structures import formula_key, record_counts
from unifiedgroupgen.group_constraints.unified import catalog_spec
from unifiedgroupgen.group_constraints.chemistry import ChemicalControl, can_complete

GROUP_POLICIES = ('prior', 'composition_ranked', 'composition_neighbors')


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


def _composition_features(counts):
    features = np.zeros(121, dtype=np.float32)
    total = sum(counts.values())
    for element, count in counts.items():
        features[element] = (count / total) ** .5
    features[-2] = .1 * len(counts)
    features[-1] = .2 * np.log1p(total)
    return features


class CompositionNeighbors:
    """Use only the checkpoint's exact TRAIN composition histogram within each group kind."""

    def __init__(self, prior):
        self.banks = {}
        for kind, (_, _, exact) in prior['composition_counts'].items():
            keys = sorted(exact)
            counts = [{int(z): int(n) for z, n in (item.split(':') for item in key.split(','))}
                      for key in keys]
            labels = sorted({int(number) for group_counts in exact.values() for number in group_counts})
            votes = np.array([[exact[key].get(str(group), 0) for group in labels]
                              for key in keys], dtype=np.int32)
            features = np.stack([_composition_features(count) for count in counts])
            self.banks[kind] = features, votes, votes.sum(axis=1), labels
        self.votes = lru_cache(maxsize=4096)(self._votes)

    def _votes(self, kind, counts):
        features, votes, frequency, labels = self.banks[kind]
        difference = features - _composition_features(dict(counts))
        distance = (difference * difference).sum(axis=1)
        order = np.argsort(distance, kind='stable')
        cumulative = np.cumsum(frequency[order])
        limit = min(128, int(cumulative[-1]))
        boundary = distance[order[np.searchsorted(cumulative, limit)]]
        inside, tied = distance < boundary, distance == boundary
        remaining = limit - frequency[inside].sum()
        weights = np.exp(-12 * distance[inside])
        scores = (weights @ votes[inside]
                  + np.exp(-12 * boundary) * remaining / frequency[tied].sum() * votes[tied].sum(axis=0))
        return dict(zip(labels, scores))


def _rank_candidates(prior, candidates, weights, chemistry, neighbors=None):
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
    if neighbors is not None:
        kinds = {c.kind for c in candidates}
        for kind in kinds & neighbors.banks.keys():
            indices = [i for i, c in enumerate(candidates) if c.kind == kind]
            local = neighbors.votes(kind, chemistry.counts)
            votes = [local.get(candidates[i].number, 0.) for i in indices]
            mass = sum(scores[i] for i in indices)
            if sum(votes):
                scale = .25 * mass / sum(votes)
                for i, vote in zip(indices, votes):
                    scores[i] = .75 * scores[i] + scale * vote
    return [candidate for _, candidate in sorted(zip(scores, candidates),
            key=lambda item: (-item[0], item[1].kind, item[1].number))]


def sample_group(prior, generator, kind=None, chemistry=None, budget=256, max_atoms=192, max_orbits=52,
                 *, composition_rank=None, neighbors=None):
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
    ranked = _rank_candidates(prior, candidates, weights, chemistry, neighbors)
    return ranked[composition_rank % len(ranked)]
