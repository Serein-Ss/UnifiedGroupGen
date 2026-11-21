"""Scalar target-property embeddings with explicit missingness and CFG."""

import math
import torch
from torch import nn


def sample_visibility(encoder, properties, like):
    """One property task/dropout decision shared across the joint heads."""
    subsets = [encoder._subset(row, like.device) for row in properties]
    dropped = (torch.rand(len(properties), device=like.device) < encoder.dropout).tolist()
    return [{name: bool(not drop and name in selected)
             for name in encoder.configs}
            for selected, drop in zip(subsets, dropped)]


class PropertyEncoder(nn.Module):
    def __init__(self, configs, hidden, dropout=0.1, subset_training=False):
        super().__init__()
        self.configs, self.dropout, self.hidden = configs, dropout, hidden
        self.subset_training = subset_training
        self.encoders, self.null = nn.ModuleDict(), nn.ParameterDict()
        for name in configs:
            self.encoders[name] = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.null[name] = nn.Parameter(torch.zeros(hidden))
        self.composition = nn.Sequential(nn.Linear(118, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def _subset(self, properties, device):
        names = [name for name in self.configs
                 if properties.get(name) is not None and math.isfinite(float(properties[name]))]
        if not self.training or not self.subset_training or len(names) < 2:
            return set(names)
        # Sample task metadata on CPU; do not synchronize CUDA once per record.
        bits = int(torch.randint(1, 2**len(names), ()))
        return {name for index, name in enumerate(names) if bits & (1 << index)}

    def _normalized(self, name, value):
        config = self.configs[name]
        return (float(value)-config.get('shift', 0))/config.get('scale', 1)

    def forward(self, properties, like, force_null=False, composition=None):
        result = like.new_zeros(self.hidden)
        # Whole-property dropout trains the exact null branch used by CFG.
        drop_all = force_null or (self.training and torch.rand((), device=like.device) < self.dropout)
        selected = self._subset(properties, like.device)
        for name in self.configs:
            value = properties.get(name)
            if drop_all or name not in selected:
                result = result + self.null[name]
            else:
                result = result + self.encoders[name](like.new_tensor([self._normalized(name, value)]))
        # Composition is a separate hard condition; property CFG never erases it.
        counts = like.new_zeros(118)
        if composition is not None:
            for atomic_number, count in composition.items():
                counts[int(atomic_number) - 1] = int(count)
        result = result + self.composition(torch.log1p(counts))
        return result

    def forward_batch(self, properties, like, force_null=False, compositions=None, visibility=None):
        compositions = compositions or [None] * len(properties)
        count = len(properties)
        subsets = ([self._subset(p, like.device) for p in properties] if visibility is None else
                   [{name for name, visible in row.items() if visible} for row in visibility])
        result = like.new_zeros((count, self.hidden))
        drop = (torch.rand(count, device=like.device) < self.dropout if self.training and not force_null and visibility is None
                else torch.full((count,), force_null, device=like.device, dtype=torch.bool))
        for name in self.configs:
            values = [p.get(name) for p in properties]
            known = [v is not None and math.isfinite(float(v)) for v in values]
            if not any(known) or force_null or (self.training and self.dropout == 1):
                result = result + self.null[name]
                continue
            inputs = like.new_tensor([[self._normalized(name, v)] if k else [0]
                                      for v, k in zip(values, known)])
            use = torch.tensor([k and name in subset for k, subset in zip(known, subsets)], device=like.device) & ~drop
            encoded = self.encoders[name](inputs)
            result = result + torch.where(use[:, None], encoded, self.null[name])
        if all(composition is None for composition in compositions):
            return result + self.composition(like.new_zeros(118))
        counts = [[0]*118 for _ in compositions]
        for row, composition in zip(counts, compositions):
            for atomic_number, number in (composition or {}).items():
                row[int(atomic_number)-1] = int(number)
        return result + self.composition(torch.log1p(like.new_tensor(counts)))


def fit_property_scalers(records, configs):
    """Fit continuous scalers using TRAINING records only, without replacing missing labels."""
    fitted = {name: dict(config) for name, config in configs.items()}
    for name, config in fitted.items():
        values = [r["properties"][name] for r in records
                  if r["properties"].get(name) is not None
                  and bool(torch.isfinite(torch.tensor(float(r["properties"][name]))))]
        if not values:
            raise ValueError(f"No finite training labels for configured property {name}")
        tensor = torch.tensor(values, dtype=torch.float64)
        config["shift"] = tensor.mean().item()
        config["scale"] = max(tensor.std(unbiased=False).item(), 1e-6)
    return fitted
