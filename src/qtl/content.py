"""Shared fusion and topology layout for query-tree content encoders.

ContextFusion is promoted unchanged from the initial inside-outside prototype.
That running prototype keeps its frozen definition until its search terminates.
"""
from collections import OrderedDict

import torch
from torch import nn

from . import ctree


class ContextFusion(nn.Module):
    def __init__(self, hidden, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(6 * hidden, hidden), nn.GELU(),
                                 nn.Dropout(dropout), nn.Linear(hidden, hidden))
        self.norm = nn.LayerNorm(hidden)
        self.drop = nn.Dropout(dropout)

    def forward(self, inside, outside):
        other_i, other_o = inside.flip(1), outside.flip(1)
        x = torch.cat((inside, outside, inside - outside, inside * outside,
                       other_i, other_o), dim=-1)
        return self.norm(inside + self.drop(self.net(x)))


class ContentLayouts:
    """Bounded layout cache; no model outputs or autograd graphs are cached."""
    def __init__(self, limit=32):
        self.limit = limit
        self.cache = OrderedDict()

    def get(self, mask):
        lengths = tuple(int(n) for n in mask.sum(1).tolist())
        key = (lengths, mask.shape[1], str(mask.device))
        if key not in self.cache:
            forest = ctree.Forest(lengths, mask.shape[1])
            levels = [{k: v.to(mask.device) if isinstance(v, torch.Tensor) else v
                       for k, v in lev.items()} for lev in forest.levels]
            self.cache[key] = (forest, levels, forest.pos_root.to(mask.device))
            if len(self.cache) > self.limit:
                self.cache.popitem(last=False)
        self.cache.move_to_end(key)
        return key, self.cache[key]
