"""Property-conditioned autoregressive G -> (Wyckoff, element)* -> STOP.

Masking enforces symbolic completion; it is not exact rejection sampling from the
unmasked Transformer conditioned on legality. No empirical SG sampling is used.
"""

from functools import lru_cache
import math
import torch
from torch import nn
from torch.nn import functional as F
from .conditioning import PropertyEncoder
from .symmetry import Descriptor, OrbitSpec, group_table, wp_arrays


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


class DescriptorTransformer(nn.Module):
    def __init__(self, groups, properties, hidden=128, layers=3, heads=4, max_atoms=192, max_orbits=32, cond_dropout=0.1, allowed_elements=None):
        super().__init__()
        self.vocab = Vocabulary(groups)
        self.legality = Legality(self.vocab, max_atoms, max_orbits, allowed_elements)
        self._mask_cache = {}
        self.max_orbits = max_orbits
        self.embedding = nn.Embedding(len(self.vocab.tokens), hidden)
        self.position = nn.Embedding(2*max_orbits+3, hidden)
        layer = nn.TransformerEncoderLayer(hidden, heads, 4*hidden, dropout=0.0, batch_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.condition = PropertyEncoder(properties, hidden, cond_dropout)
        self.head = nn.Linear(hidden, len(self.vocab.tokens))

    def forward(self, tokens, properties, force_null=False, composition=None):
        ids = torch.as_tensor(tokens, device=self.embedding.weight.device, dtype=torch.long)
        positions = torch.arange(len(tokens), device=ids.device)
        h = self.embedding(ids) + self.position(positions)
        h = h + self.condition(properties, h, force_null, composition)
        mask = torch.ones(len(tokens), len(tokens), device=ids.device, dtype=torch.bool).triu(1)
        return self.head(self.transformer(h[None], mask=mask)[0])

    def loss(self, descriptor, properties, composition=None):
        return self.loss_batch([descriptor], [properties], [composition]).mean()

    def loss_batch(self, descriptors, properties, compositions=None):
        """One padded causal Transformer call; return an equal-weight loss per structure."""
        compositions = compositions or [None] * len(descriptors)
        sequences = [self.vocab.encode(d) for d in descriptors]
        device = self.embedding.weight.device
        width = max(len(s)-1 for s in sequences)
        if width >= self.position.num_embeddings:
            raise ValueError("Descriptor exceeds the configured orbit budget")
        ids = torch.full((len(sequences), width), self.vocab.index['PAD'], dtype=torch.long)
        targets = torch.full_like(ids, -100)
        legal = torch.ones(len(sequences), width, len(self.vocab.tokens), dtype=torch.bool)
        for row, (d, sequence, composition) in enumerate(zip(descriptors, sequences, compositions)):
            length = len(sequence)-1
            ids[row, :length] = torch.tensor(sequence[:-1])
            targets[row, :length] = torch.tensor(sequence[1:])
            key = (d, tuple(sorted((composition or {}).items())))
            if key not in self._mask_cache:
                masks = torch.stack([self.legality.allowed(sequence[:i], composition, d.kind)
                                     for i in range(1, len(sequence))])
                if not masks[torch.arange(length), torch.tensor(sequence[1:])].all():
                    raise ValueError("Training descriptor violates configured symbolic constraints")
                if len(self._mask_cache) >= 32768:
                    self._mask_cache.pop(next(iter(self._mask_cache)))
                self._mask_cache[key] = masks
            legal[row, :length] = self._mask_cache[key]
        ids, targets = ids.to(device), targets.to(device)
        h = self.embedding(ids) + self.position(torch.arange(width, device=device))[None]
        h = h + self.condition.forward_batch(properties, h, compositions=compositions)[:, None]
        causal = torch.ones(width, width, device=device, dtype=torch.bool).triu(1)
        logits = self.head(self.transformer(h, mask=causal, src_key_padding_mask=ids.eq(self.vocab.index['PAD'])))
        losses = F.cross_entropy(logits.masked_fill(~legal.to(device), -torch.inf).transpose(1, 2),
                                 targets, reduction='none')
        lengths = targets.ne(-100).sum(1)
        return losses.sum(1)/lengths

    @torch.no_grad()
    def sample(self, properties=None, kind="space", composition=None, guidance=1.0, temperature=1.0, generator=None):
        if self.training:
            raise RuntimeError("Call eval() before sampling to disable condition dropout")
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        tokens = [self.vocab.index["BOS"]]
        for _ in range(2*self.max_orbits+2):
            if guidance == 0:
                logits = self(tokens, properties or {}, force_null=True, composition=composition)[-1]
            elif guidance == 1:
                logits = self(tokens, properties or {}, composition=composition)[-1]
            else:
                conditional = self(tokens, properties or {}, composition=composition)[-1]
                unconditional = self(tokens, properties or {}, force_null=True, composition=composition)[-1]
                logits = unconditional + guidance*(conditional-unconditional)
            allowed = self.legality.allowed(tokens, composition, kind).to(logits.device)
            if not allowed.any():
                raise ValueError("No descriptor completes the requested constraints")
            probabilities = torch.softmax(logits.masked_fill(~allowed, -torch.inf)/temperature, dim=-1)
            token = int(torch.multinomial(probabilities, 1, generator=generator))
            tokens.append(token)
            if token == self.vocab.index["STOP"]:
                return self.vocab.decode(tokens)
        raise RuntimeError("Descriptor did not terminate within the configured orbit budget")
