import torch
from torch import nn
from unifiedgroupgen.conditional.properties import PropertyEncoder

from unifiedgroupgen.generation.rules import Vocabulary, Legality
from unifiedgroupgen.pl_modules.backbones.transformer import build_transformer

class DescriptorTransformer(nn.Module):
    def __init__(self, groups, properties, hidden=128, layers=3, heads=4, max_atoms=192, max_orbits=32, cond_dropout=0.1, allowed_elements=None, backbone_factory=None):
        super().__init__()
        self.vocab = Vocabulary(groups)
        self.legality = Legality(self.vocab, max_atoms, max_orbits, allowed_elements)
        self._mask_cache = {}
        self.max_orbits = max_orbits
        self.embedding = nn.Embedding(len(self.vocab.tokens), hidden)
        self.position = nn.Embedding(2*max_orbits+3, hidden)
        self.transformer = (backbone_factory or build_transformer)(hidden=hidden, heads=heads, layers=layers)
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

    def loss_batch(self, *args, **kwargs):
        from unifiedgroupgen.pl_modules.objectives.descriptor import descriptor_loss
        return descriptor_loss(self, *args, **kwargs)

    def sample(self, *args, **kwargs):
        from unifiedgroupgen.generation.descriptor import sample_descriptor
        return sample_descriptor(self, *args, **kwargs)
