"""Operation-compiled scalar and Cartesian tangent-vector feature fields.

The packed output is a proposal context, not a claim that arbitrary off-manifold
inputs or its raw anchor heads are equivariant. The returned typed fields satisfy
the compiled constraints. Physical samples still use the joint hard decoder.
"""
import torch
from torch import nn


def project_scalars(values, compiled):
    return torch.cat([values[section].mean(0, keepdim=True).expand_as(values[section])
                      for section in compiled.slices]) if compiled.num_atoms else values


def project_vectors(values, compiled, lattice):
    """Project N x channels x 3 vectors onto each orbit's allowed tangent field.

    The Cartesian basis depends on the current lattice; no state-derived tensor
    is cached or detached. Each solve is at most 3 x 3.
    """
    blocks = []
    for section, (a, _, _) in zip(compiled.slices, compiled.blocks):
        raw = values[section]
        basis = torch.as_tensor(a, device=values.device, dtype=values.dtype)
        if not basis.shape[-1]:
            blocks.append(raw*0)
            continue
        cartesian = torch.einsum('ab,nbq->naq', lattice, basis).reshape(-1, basis.shape[-1])
        matrix = raw.transpose(1, 2).reshape(-1, raw.shape[1])
        coefficients = torch.linalg.solve(cartesian.T@cartesian, cartesian.T@matrix)
        blocks.append((cartesian@coefficients).reshape(len(raw), 3, raw.shape[1]).transpose(1, 2))
    return torch.cat(blocks) if blocks else values


class OperationConstraintLayer(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.channels = max(1, hidden//16)
        self.vector_source = nn.Linear(hidden, 3*self.channels)
        self.pack = nn.Linear(3*self.channels, hidden, bias=False)

    def forward(self, nodes, state):
        scalar = project_scalars(nodes[1:], state.compiled)
        raw = self.vector_source(nodes[1:]).reshape(state.num_atoms, self.channels, 3)
        vector = project_vectors(raw, state.compiled, state.compiled.lattice(state.metric))
        # The public teacher path includes unconstrained normal residuals, even
        # on fixed sites. Retain its atomwise proposal context as well as the
        # constrained fields; otherwise ambient anchor matching is inexpressive.
        atom_context = nodes[1:]+scalar+self.pack(vector.flatten(1))
        return torch.cat((nodes[:1], atom_context)), dict(scalars=scalar, vectors=vector)
