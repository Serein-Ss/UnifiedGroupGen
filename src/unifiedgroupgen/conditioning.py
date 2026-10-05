"""CGDiT-style scalar/categorical property embeddings with explicit missingness and CFG."""

import torch
from torch import nn


class PropertyEncoder(nn.Module):
    def __init__(self, configs, hidden, dropout=0.1):
        super().__init__()
        self.configs, self.dropout, self.hidden = configs, dropout, hidden
        self.encoders, self.null = nn.ModuleDict(), nn.ParameterDict()
        for name, config in configs.items():
            if config.get("type", "scalar") == "categorical":
                self.encoders[name] = nn.Embedding(config["num_classes"], hidden)
            else:
                self.encoders[name] = nn.Sequential(nn.Linear(1, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
            self.null[name] = nn.Parameter(torch.zeros(hidden))
        self.composition = nn.Sequential(nn.Linear(118, hidden), nn.SiLU(), nn.Linear(hidden, hidden))

    def forward(self, properties, like, force_null=False, composition=None):
        result = like.new_zeros(self.hidden)
        # Whole-property dropout trains the exact null branch used by CFG.
        drop_all = force_null or (self.training and torch.rand((), device=like.device) < self.dropout)
        for name, config in self.configs.items():
            value = properties.get(name)
            known = value is not None and bool(torch.isfinite(torch.as_tensor(value)))
            context = config.get('context', False)
            if context and not known:
                raise ValueError(f'Missing required context: {name}')
            if (drop_all and not context) or not known:
                result = result + self.null[name]
            else:
                if config.get("type", "scalar") == "categorical":
                    if int(value) != value or not 0 <= int(value) < config['num_classes']:
                        raise ValueError(f'Invalid category for {name}: {value}')
                    value = torch.as_tensor(int(value), device=like.device)
                else:
                    value = like.new_tensor([(float(value) - config.get("shift", 0)) / config.get("scale", 1)])
                result = result + self.encoders[name](value)
        # Composition is a separate hard condition; property CFG never erases it.
        counts = like.new_zeros(118)
        if composition is not None:
            for atomic_number, count in composition.items():
                counts[int(atomic_number) - 1] = int(count)
        result = result + self.composition(torch.log1p(counts))
        return result

    def forward_batch(self, properties, like, force_null=False, compositions=None):
        compositions = compositions or [None] * len(properties)
        return torch.stack([self(p, like, force_null, c) for p, c in zip(properties, compositions)])


def attach_dataset_domains(records, config):
    """Keep calculation/source domain as mandatory context when mixing property-labelled datasets."""
    domains = config.get('dataset_domains')
    if not domains:
        return records
    specification = config.get('properties', {}).get('source_domain', {})
    if specification.get('type') != 'categorical' or not specification.get('context'):
        raise ValueError('dataset_domains requires categorical source_domain with context=true')
    updated = []
    for record in records:
        dataset = record['dataset']
        if dataset not in domains:
            raise ValueError(f'No calculation domain for dataset {dataset}')
        domain = domains[dataset]
        if int(domain) != domain or not 0 <= domain < specification['num_classes']:
            raise ValueError(f'Invalid calculation domain for dataset {dataset}')
        existing = record['properties'].get('source_domain', domain)
        if existing != domain:
            raise ValueError(f'Conflicting source domain in {record["id"]}')
        updated.append({**record, 'properties': {**record['properties'], 'source_domain': domain}})
    return updated


def fit_property_scalers(records, configs):
    """Fit continuous scalers using TRAINING records only, without replacing missing labels."""
    fitted = {name: dict(config) for name, config in configs.items()}
    for name, config in fitted.items():
        if config.get("type", "scalar") != "scalar":
            continue
        values = [r["properties"][name] for r in records
                  if r["properties"].get(name) is not None
                  and bool(torch.isfinite(torch.tensor(float(r["properties"][name]))))]
        if not values:
            raise ValueError(f"No finite training labels for configured property {name}")
        tensor = torch.tensor(values, dtype=torch.float64)
        config["shift"] = tensor.mean().item()
        config["scale"] = max(tensor.std(unbiased=False).item(), 1e-6)
    return fitted
