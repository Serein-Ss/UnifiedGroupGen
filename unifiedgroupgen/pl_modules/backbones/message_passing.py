import torch
from torch import nn

def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))

class MessageLayer(nn.Module):
    def __init__(self, hidden, edge_dim):
        super().__init__()
        self.edge = mlp(2*hidden+edge_dim, hidden, hidden)
        self.node = mlp(2*hidden, hidden, hidden)
        self.norm = nn.LayerNorm(hidden)

    def forward(self, h, edges, edge_index, degree, chunk_size=8192):
        # Split the SAME affine map; do not repeat node projections per edge.
        hidden = h.shape[-1]
        linear = self.edge[0]
        source_h = nn.functional.linear(h, linear.weight[:, :hidden])
        target_h = nn.functional.linear(h, linear.weight[:, hidden:2*hidden])
        # AMP does not cache non-leaf weight slices. Cast this shared slice
        # once per layer, rather than once per edge chunk.
        edge_weight = linear.weight[:, 2*hidden:].to(source_h.dtype)
        edge_bias = linear.bias.to(source_h.dtype)
        aggregate = torch.zeros_like(h)
        for start in range(0, edges.shape[0], chunk_size):
            source, target = edge_index[:, start:start+chunk_size]
            edge_h = nn.functional.linear(edges[start:start+chunk_size], edge_weight, edge_bias)
            messages = self.edge[2](self.edge[1](source_h[source] + target_h[target] + edge_h))
            aggregate = aggregate.index_add(0, source, messages.to(aggregate.dtype))
        aggregate = aggregate / degree[:, None].clamp_min(1)
        return self.norm(h + self.node(torch.cat((h, aggregate), dim=-1)))
