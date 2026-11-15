import torch

@torch.no_grad()
def sample_descriptor(self, properties=None, kind="space", composition=None, guidance=1.0, temperature=1.0, generator=None,
           chemical=None, prefix=None, condition_composition=True):
    if self.training:
        raise RuntimeError("Call eval() before sampling to disable condition dropout")
    if temperature <= 0:
        raise ValueError("Temperature must be positive")
    tokens = list(prefix) if prefix is not None else [self.vocab.index["BOS"]]
    learned_composition = composition if condition_composition else None
    for i in range(1, len(tokens)):
        legal = (chemical.mask(self.legality, tokens[:i], composition, kind) if chemical
                 else self.legality.allowed(tokens[:i], composition, kind))
        if not legal[tokens[i]]:
            raise ValueError('Known descriptor prefix conflicts with the requested constraints')
    for _ in range(2*self.max_orbits+3-len(tokens)):
        if guidance == 0:
            logits = self(tokens, properties or {}, force_null=True, composition=learned_composition)[-1]
        elif guidance == 1:
            logits = self(tokens, properties or {}, composition=learned_composition)[-1]
        else:
            conditional = self(tokens, properties or {}, composition=learned_composition)[-1]
            unconditional = self(tokens, properties or {}, force_null=True, composition=learned_composition)[-1]
            logits = unconditional + guidance*(conditional-unconditional)
        allowed = (chemical.mask(self.legality, tokens, composition, kind) if chemical
                   else self.legality.allowed(tokens, composition, kind)).to(logits.device)
        if not allowed.any():
            raise ValueError("No descriptor completes the requested constraints")
        probabilities = torch.softmax(logits.masked_fill(~allowed, -torch.inf)/temperature, dim=-1)
        token = int(torch.multinomial(probabilities, 1, generator=generator))
        tokens.append(token)
        if token == self.vocab.index["STOP"]:
            return self.vocab.decode(tokens)
    raise RuntimeError("Descriptor did not terminate within the configured orbit budget")
