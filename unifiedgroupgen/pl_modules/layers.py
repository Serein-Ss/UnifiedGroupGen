"""Shared graph layers; parameter names and forward operations are unchanged."""
import math
import torch
from torch import nn

class AtomGraphBlock(nn.Module):
    def __init__(self, hidden, heads, edge_dim=33):
        super().__init__()
        self.heads, self.width = heads, hidden//heads
        self.norm1, self.norm2 = nn.LayerNorm(hidden), nn.LayerNorm(hidden)
        self.qkv, self.output = nn.Linear(hidden, 3*hidden), nn.Linear(hidden, hidden)
        self.edge = nn.Sequential(nn.Linear(edge_dim, hidden), nn.SiLU(), nn.Linear(hidden, heads))
        self.feedforward = nn.Sequential(nn.Linear(hidden, 4*hidden), nn.SiLU(), nn.Linear(4*hidden, hidden))

    def forward(self, nodes, edges):
        count = len(nodes)
        q, k, v = self.qkv(self.norm1(nodes)).reshape(count, 3, self.heads, self.width).unbind(1)
        q, k, v = (x.transpose(0, 1) for x in (q, k, v))
        bias = self.edge(edges).permute(2, 0, 1)
        weights = torch.softmax(q @ k.transpose(-1, -2)/math.sqrt(self.width)+bias, dim=-1)
        update = (weights @ v).transpose(0, 1).reshape_as(nodes)
        nodes = nodes+self.output(update)
        return nodes+self.feedforward(self.norm2(nodes))
