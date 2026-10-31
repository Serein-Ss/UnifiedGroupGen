"""Causal descriptor backbone; tokenization, objectives and sampling live elsewhere."""
from torch import nn


def build_transformer(hidden, heads, layers):
    layer = nn.TransformerEncoderLayer(hidden, heads, 4*hidden, dropout=0.0, batch_first=True)
    return nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
