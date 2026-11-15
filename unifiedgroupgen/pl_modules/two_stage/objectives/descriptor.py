import torch
from torch.nn import functional as F

def descriptor_loss(self, descriptors, properties, compositions=None, visibility=None):
    """One padded causal Transformer call; return an equal-weight loss per structure."""
    compositions = compositions or [None] * len(descriptors)
    rows = []
    for descriptor, composition in zip(descriptors, compositions):
        key = (descriptor, tuple(sorted((composition or {}).items())))
        if key not in self._mask_cache:
            sequence = self.vocab.encode(descriptor)
            masks = torch.stack([self.legality.allowed(sequence[:i], composition, descriptor.kind)
                                 for i in range(1, len(sequence))])
            if not masks[torch.arange(len(sequence)-1), torch.tensor(sequence[1:])].all():
                raise ValueError('Training descriptor violates configured symbolic constraints')
            if len(self._mask_cache) >= 32768:
                self._mask_cache.pop(next(iter(self._mask_cache)))
            self._mask_cache[key] = (torch.tensor(sequence), masks)
        rows.append(self._mask_cache[key])
    device = self.embedding.weight.device
    width = max(len(sequence)-1 for sequence, _ in rows)
    if width >= self.position.num_embeddings:
        raise ValueError("Descriptor exceeds the configured orbit budget")
    ids = torch.full((len(rows), width), self.vocab.index['PAD'], dtype=torch.long)
    targets = torch.full_like(ids, -100)
    legal = torch.ones(len(rows), width, len(self.vocab.tokens), dtype=torch.bool)
    for row, (sequence, masks) in enumerate(rows):
        length = len(sequence)-1
        ids[row, :length] = sequence[:-1]
        targets[row, :length] = sequence[1:]
        legal[row, :length] = masks
    ids, targets = ids.to(device), targets.to(device)
    h = self.embedding(ids) + self.position(torch.arange(width, device=device))[None]
    h = h + self.condition.forward_batch(properties, h, compositions=compositions, visibility=visibility)[:, None]
    causal = torch.ones(width, width, device=device, dtype=torch.bool).triu(1)
    logits = self.head(self.transformer(h, mask=causal, src_key_padding_mask=ids.eq(self.vocab.index['PAD'])))
    losses = F.cross_entropy(logits.masked_fill(~legal.to(device), -torch.inf).transpose(1, 2),
                             targets, reduction='none')
    lengths = targets.ne(-100).sum(1)
    return losses.sum(1)/lengths
