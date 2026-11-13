"""Immutable chemical task state and cached, finite-budget orbit completion."""
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from math import gcd
import torch


def atomic_number(value):
    if isinstance(value, bool):
        raise ValueError('Boolean is not an atomic number')
    if isinstance(value, str) and not value.isdigit():
        from pymatgen.core import Element
        return Element(value).Z
    number = int(value)
    if number != float(value) or not 1 <= number <= 118:
        raise ValueError('Expected an element symbol or atomic number in 1..118')
    return number


@dataclass(frozen=True)
class ChemicalControl:
    allowed: tuple = ()
    counts: tuple = ()

    def __post_init__(self):
        if tuple(sorted(set(self.allowed))) != self.allowed or any(not 1 <= z <= 118 for z in self.allowed):
            raise ValueError('Allowed elements must be a sorted, unique atomic-number tuple')
        if tuple(sorted(dict(self.counts).items())) != self.counts:
            raise ValueError('Counts must be sorted and have unique element keys')
        if any(not 1 <= z <= 118 or int(n) != n or n <= 0 for z, n in self.counts):
            raise ValueError('Fixed counts must be positive integers')
        if self.counts and self.allowed and set(dict(self.counts))-set(self.allowed):
            raise ValueError('Fixed composition contains a forbidden element')

    @classmethod
    def from_inputs(cls, allowed=None, counts=None):
        domain = tuple(sorted(set(atomic_number(z) for z in (allowed or ()))))
        if counts is not None:
            if not isinstance(counts, dict) or not counts:
                raise ValueError('A fixed composition must be nonempty')
            converted = {}
            for z, n in counts.items():
                z = atomic_number(z)
                if isinstance(n, bool) or int(n) != n or n <= 0 or z in converted:
                    raise ValueError('Fixed composition must have unique elements and positive integer counts')
                converted[z] = int(n)
            return cls(domain, tuple(sorted(converted.items())))
        return cls(domain)

    @property
    def mode(self):
        return 'fixed_counts' if self.counts else 'allowed_set' if self.allowed else 'free'

    def domain(self, vocabulary):
        supplied = set(self.allowed) | set(dict(self.counts))
        if supplied-set(vocabulary):
            raise ValueError('Chemical request contains elements outside the TRAIN vocabulary')
        return tuple(z for z in vocabulary if (not self.allowed or z in self.allowed)
                     and (not self.counts or z in dict(self.counts)))

    def to_dict(self):
        return dict(mode=self.mode, allowed_elements=list(self.allowed), counts=dict(self.counts))

    def remaining(self, spec, letters, elements):
        used = orbit_counts(spec, letters, elements)
        return tuple((z, n-used.get(z, 0)) for z, n in self.counts)

    def validate(self, spec, letters, elements, finished=False):
        used = orbit_counts(spec, letters, elements)
        if self.allowed and set(used)-set(self.allowed):
            raise ValueError('State contains a forbidden chemical element')
        if self.counts:
            limits = dict(self.counts)
            if 0 in elements or set(used)-set(limits) or any(n > limits[z] for z, n in used.items()):
                raise ValueError('State exceeds the fixed inventory or contains unassigned elements')
            if finished and used != limits:
                raise ValueError('END requires the exact fixed composition')

    def features(self, state, vocabulary):
        values = state.metric.new_zeros(357)
        domain = self.domain(vocabulary)
        values[[z-1 for z in domain]] = 1
        for z, n in self.counts:
            values[118+z-1] = n
        for z, n in self.remaining(state.spec, state.letters, state.elements):
            values[236+z-1] = n
        values[118:354] = torch.log1p(values[118:354])
        values[354+('free', 'allowed_set', 'fixed_counts').index(self.mode)] = 1
        return values


def orbit_counts(spec, letters, elements):
    result = Counter()
    for letter, z in zip(letters, elements):
        if z:
            result[int(z)] += spec.template(letter).multiplicity
    return dict(sorted(result.items()))


@lru_cache(maxsize=131072)
def _fill(general, fixed, amounts, slots):
    if slots < 0:
        return False
    if not any(amounts):
        return True
    if slots <= 0:
        return False
    sizes = tuple(sorted(set(general) | {m for m, n in fixed if n}, reverse=True))
    if not sizes or sum(amounts) > slots*max(sizes):
        return False
    divisor = 0
    for size in sizes:
        divisor = gcd(divisor, size)
    if any(n % divisor for n in amounts):
        return False
    index = min((i for i, n in enumerate(amounts) if n), key=lambda i: amounts[i])
    count = amounts[index]
    for size in sizes:
        if size > count:
            continue
        next_amounts = list(amounts); next_amounts[index] -= size
        if size in general and _fill(general, fixed, tuple(next_amounts), slots-1):
            return True
        for j, (m, capacity) in enumerate(fixed):
            if m == size and capacity:
                next_fixed = list(fixed); next_fixed[j] = (m, capacity-1)
                if _fill(general, tuple(next_fixed), tuple(next_amounts), slots-1):
                    return True
    return False


def can_complete(spec, letters, elements, control, slots, max_orbits=52, max_atoms=192):
    """Conservative witness: complete using whole-orbit births, then END."""
    if not control.counts:
        return True
    try:
        control.validate(spec, letters, elements)
    except ValueError:
        return False
    if sum(dict(control.counts).values()) > max_atoms:
        return False
    remaining = tuple(n for _, n in control.remaining(spec, letters, elements))
    general = tuple(sorted({w.multiplicity for w in spec.templates if w.dof}))
    fixed = Counter(w.multiplicity for w in spec.templates if not w.dof and w.letter not in letters)
    return _fill(general, tuple(sorted(fixed.items())), remaining,
                 min(slots, max_orbits-len(letters)))


def edit_signature(letters, elements, action):
    letters, elements = list(letters), list(elements)
    if action.kind in ('birth', 'split'):
        letters.append(action.letter); elements.append(action.element)
    elif action.kind == 'delete':
        letters.pop(action.orbit); elements.pop(action.orbit)
    elif action.kind == 'replace':
        letters[action.orbit], elements[action.orbit] = action.letter, action.element
    elif action.kind == 'element':
        elements[action.orbit] = action.element
    elif action.kind == 'swap':
        i, j = action.orbit, action.partner
        elements[i], elements[j] = elements[j], elements[i]
    return tuple(letters), tuple(elements)
