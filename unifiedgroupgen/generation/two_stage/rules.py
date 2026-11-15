from functools import lru_cache
import math
import torch
from unifiedgroupgen.group_constraints.compiler import Descriptor, OrbitSpec, group_table, wp_arrays

class Vocabulary:
    def __init__(self, groups):
        self.groups = [tuple(g) for g in groups]
        tokens = ["PAD", "BOS", "STOP"]
        tokens += [f"G:{kind}:{number}" for kind, number in self.groups]
        letters = sorted({w.letter for kind, number in self.groups for w in group_table(kind, number)})
        tokens += [f"W:{w}" for w in letters]
        tokens += [f"Z:{z}" for z in range(1, 119)]
        self.tokens, self.index = tokens, {token: i for i, token in enumerate(tokens)}

    def encode(self, descriptor):
        result = [self.index["BOS"], self.index[f"G:{descriptor.kind}:{descriptor.number}"]]
        for orbit in descriptor.orbits:
            result.extend([self.index[f"W:{orbit.letter}"], self.index[f"Z:{orbit.atomic_number}"]])
        result.append(self.index["STOP"])
        return result

    def decode(self, tokens):
        text = [self.tokens[i] for i in tokens]
        if text[0] != "BOS" or text[-1] != "STOP":
            raise ValueError("Incomplete descriptor token sequence")
        _, kind, number = text[1].split(":")
        orbits = tuple(OrbitSpec(text[i][2:], int(text[i+1][2:])) for i in range(2, len(text)-1, 2))
        return Descriptor(kind, int(number), orbits)

class Legality:
    def __init__(self, vocabulary, max_atoms=192, max_orbits=32, allowed_elements=None):
        self.vocab, self.max_atoms, self.max_orbits = vocabulary, max_atoms, max_orbits
        self.elements = tuple(sorted(set(range(1, 119) if allowed_elements is None else allowed_elements)))
        if not self.elements or any(not 1 <= z <= 118 for z in self.elements):
            raise ValueError("Allowed elements must be atomic numbers in 1..118")
        self.rows = {}
        for kind, number in vocabulary.groups:
            self.rows[kind, number] = {
                w.letter: (w.multiplicity, wp_arrays(kind, number, w.letter)[0].shape[-1] == 0)
                for w in group_table(kind, number)
            }

    def _completion(self, group, remaining, occupied, slots):
        """Bounded symbolic search, including single occupancy of zero-DOF positions."""
        return self._cached_completion(group, tuple(sorted((n for n in remaining if n), reverse=True)),
                                       tuple(sorted(occupied)), slots)

    @lru_cache(maxsize=8192)
    def _cached_completion(self, group, remaining, occupied, slots):
        rows = tuple(self.rows[group].items())
        @lru_cache(maxsize=None)
        def solve(counts, used, budget):
            if not any(counts):
                return True
            if budget <= 0:
                return False
            available = [m for letter, (m, fixed) in rows if not fixed or letter not in used]
            if not available:
                return False
            if sum(math.ceil(n/max(available)) for n in counts) > budget:
                return False
            divisor = math.gcd(*available)
            if any(n % divisor for n in counts):
                return False
            # Element order does not affect feasibility; consume the first remaining element.
            e = next(i for i, count in enumerate(counts) if count)
            for letter, (multiplicity, fixed) in rows:
                if counts[e] < multiplicity or (fixed and letter in used):
                    continue
                updated = list(counts)
                updated[e] -= multiplicity
                next_used = tuple(sorted((*used, letter))) if fixed else used
                if solve(tuple(updated), next_used, budget-1):
                    return True
            return False
        return solve(tuple(remaining), tuple(sorted(occupied)), slots)

    def allowed(self, prefix, composition=None, kind=None):
        tokens = [self.vocab.tokens[i] for i in prefix]
        result = torch.zeros(len(self.vocab.tokens), dtype=torch.bool)
        composition = {int(z): int(n) for z, n in (composition or {}).items()}
        if set(composition)-set(self.elements):
            raise ValueError("Requested composition contains elements outside the model domain")
        if any(n <= 0 for n in composition.values()) or sum(composition.values()) > self.max_atoms:
            raise ValueError("Composition must contain positive counts within max_atoms")
        if len(tokens) == 1:
            for group in self.vocab.groups:
                if kind is not None and group[0] != kind:
                    continue
                rows = self.rows[group]
                legal = (self._completion(group, list(composition.values()), (), self.max_orbits)
                         if composition else min(v[0] for v in rows.values()) <= self.max_atoms)
                if legal:
                    result[self.vocab.index[f"G:{group[0]}:{group[1]}"]] = True
            return result
        _, group_kind, group_number = tokens[1].split(":")
        group = (group_kind, int(group_number))
        rows = self.rows[group]
        completed = (len(tokens)-2)//2
        occupied, n_atoms = set(), 0
        for i in range(2, 2+completed*2, 2):
            letter, z = tokens[i][2:], int(tokens[i+1][2:])
            multiplicity, fixed = rows[letter]
            n_atoms += multiplicity
            if fixed:
                occupied.add(letter)
            if composition:
                if z not in composition:
                    raise ValueError("Prefix violates composition")
                composition[z] -= multiplicity
                if composition[z] < 0:
                    raise ValueError("Prefix exceeds a composition count")
        if len(tokens) % 2 == 0:  # choose W or STOP
            if completed and (not composition or not any(composition.values())):
                result[self.vocab.index["STOP"]] = True
            if completed >= self.max_orbits:
                return result
            for letter, (multiplicity, fixed) in rows.items():
                if n_atoms + multiplicity > self.max_atoms or (fixed and letter in occupied):
                    continue
                candidate = prefix + [self.vocab.index[f"W:{letter}"]]
                if self.allowed(candidate, composition=None if not composition else self._original_composition(prefix, composition), kind=kind).any():
                    result[self.vocab.index[f"W:{letter}"]] = True
        else:  # choose element of a pending W
            letter = tokens[-1][2:]
            multiplicity, fixed = rows[letter]
            if n_atoms + multiplicity > self.max_atoms or (fixed and letter in occupied):
                return result
            new_occupied = occupied | ({letter} if fixed else set())
            candidates = composition.keys() if composition else self.elements
            for z in candidates:
                legal = True
                if composition:
                    rem = dict(composition)
                    rem[z] -= multiplicity
                    legal = rem[z] >= 0 and self._completion(group, list(rem.values()), new_occupied, self.max_orbits-completed-1)
                if legal:
                    result[self.vocab.index[f"Z:{z}"]] = True
        return result

    def _original_composition(self, prefix, remaining):
        result = dict(remaining)
        tokens = [self.vocab.tokens[i] for i in prefix]
        _, kind, number = tokens[1].split(":")
        rows = self.rows[kind, int(number)]
        for i in range(2, len(tokens)-1, 2):
            result[int(tokens[i+1][2:])] += rows[tokens[i][2:]][0]
        return result
