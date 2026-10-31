"""Inference-time chemical rules; these do not estimate a conditional density."""

from functools import lru_cache
from pymatgen.core import Composition, Element


def element_numbers(values):
    if isinstance(values, str):
        values = values.replace(',', '-').split('-')
    return frozenset(Element(v).Z if isinstance(v, str) and not v.isdigit() else int(v)
                     for v in values if v != '')


def integer_composition(formula, empirical=False):
    composition = Composition(formula)
    if empirical:
        composition = composition.reduced_composition
    result = {}
    for element, count in composition.items():
        if not float(count).is_integer() or count <= 0:
            raise ValueError('Composition requires positive integer counts')
        result[element.Z] = int(count)
    if not result:
        raise ValueError('Empty composition')
    return result


class ChemicalRules:
    def __init__(self, allowed=None, required=None):
        self.allowed = element_numbers(allowed) if allowed is not None else None
        self.required = element_numbers(required or [])
        if self.allowed is not None and (not self.allowed or not self.required <= self.allowed):
            raise ValueError('Required elements must belong to the nonempty allowed set')
        if any(z < 1 or z > 118 for z in (self.allowed or frozenset()) | self.required):
            raise ValueError('Invalid atomic number')

    def validate_domain(self, legality, composition=None):
        requested = (self.allowed or frozenset()) | self.required | set(composition or {})
        if not requested <= set(legality.elements):
            raise ValueError('Chemical request contains elements outside the trained model domain')
        if composition is not None and not self.matches(composition):
            raise ValueError('Exact composition conflicts with chemical-system requirements')

    def matches(self, counts):
        present = {int(z) for z, n in counts.items() if n > 0}
        return self.required <= present and (self.allowed is None or present <= self.allowed)

    def mask(self, legality, prefix, composition=None, kind=None):
        self.validate_domain(legality, composition)
        base = legality.allowed(prefix, composition, kind)
        tokens = [legality.vocab.tokens[i] for i in prefix]
        if len(tokens) == 1:
            for index in base.nonzero().flatten().tolist():
                if not self._feasible(legality, tokens + [legality.vocab.tokens[index]]):
                    base[index] = False
            return base
        if len(tokens) % 2 == 0:  # choose an orbit, or finish
            for index in base.nonzero().flatten().tolist():
                token = legality.vocab.tokens[index]
                if token == 'STOP':
                    base[index] = self.matches(self._counts(legality, tokens))
                else:
                    base[index] = bool(self.mask(legality, prefix+[index], composition, kind).any())
            return base
        for index in base.nonzero().flatten().tolist():
            z = int(legality.vocab.tokens[index][2:])
            if (self.allowed is not None and z not in self.allowed
                    or not self._feasible(legality, tokens+[legality.vocab.tokens[index]])):
                base[index] = False
        return base

    @staticmethod
    def _counts(legality, tokens):
        _, kind, number = tokens[1].split(':')
        rows, counts = legality.rows[kind, int(number)], {}
        for i in range(2, len(tokens)-1, 2):
            z = int(tokens[i+1][2:])
            counts[z] = counts.get(z, 0)+rows[tokens[i][2:]][0]
        return counts

    def _feasible(self, legality, tokens):
        _, kind, number = tokens[1].split(':')
        rows = legality.rows[kind, int(number)]
        counts = self._counts(legality, tokens)
        missing = len(self.required-set(counts))
        slots = legality.max_orbits-(len(tokens)-2)//2
        atoms = legality.max_atoms-sum(counts.values())
        occupied = {tokens[i][2:] for i in range(2, len(tokens)-1, 2) if rows[tokens[i][2:]][1]}
        @lru_cache(None)
        def solve(left, budget, remaining, used):
            if not left:
                return True
            if left > budget:
                return False
            for letter, (multiplicity, fixed) in rows.items():
                if multiplicity <= remaining and (not fixed or letter not in used):
                    new_used = tuple(sorted((*used, letter))) if fixed else used
                    if solve(left-1, budget-1, remaining-multiplicity, new_used):
                        return True
            return False
        return solve(missing, slots, atoms, tuple(sorted(occupied)))


def feasible_formula_counts(formula, units, legality, kind, chemical=None, prefix=None):
    reduced = integer_composition(formula, empirical=True)
    result = []
    start = prefix or [legality.vocab.index['BOS']]
    for multiplier in units:
        if int(multiplier) != multiplier or multiplier < 1:
            raise ValueError('Formula units must be positive integers')
        counts = {z: n*int(multiplier) for z, n in reduced.items()}
        if sum(counts.values()) > legality.max_atoms:
            continue
        try:
            allowed = (chemical.mask(legality, start, counts, kind) if chemical
                       else legality.allowed(start, counts, kind))
        except ValueError:
            continue
        if allowed.any():
            result.append(counts)
    if not result:
        raise ValueError('No requested formula-unit count fits the group/orbit/atom budgets')
    return result
