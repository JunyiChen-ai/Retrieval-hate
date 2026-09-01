"""Single-student 1fps temporal scorer; no teacher is used at inference."""
from __future__ import annotations

import torch
from torch import nn


DIMS = {"audio": 128, "visual": 1024, "text": 768}


class ResidualTemporalBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.conv = nn.Conv1d(width, width, 3, padding=dilation,
                              dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        residual = x
        x = self.norm(x)
        x = self.conv(x.transpose(1, 2)).transpose(1, 2)
        x = self.dropout(self.activation(x))
        return (residual + x) * mask.unsqueeze(-1)


class SequenceCrowdStudent(nn.Module):
    def __init__(self, width=128, dropout=.1):
        super().__init__()
        branch = width // 2
        self.project = nn.ModuleDict({
            name: nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, branch),
                                nn.GELU())
            for name, dim in DIMS.items()
        })
        self.fuse = nn.Sequential(nn.Linear(branch * 3, width), nn.GELU(),
                                  nn.Dropout(dropout))
        self.temporal = nn.ModuleList([
            ResidualTemporalBlock(width, 1, dropout),
            ResidualTemporalBlock(width, 2, dropout),
            ResidualTemporalBlock(width, 4, dropout),
        ])
        self.head = nn.Linear(width, 1)

    def forward(self, features, mask):
        x = torch.cat([self.project[name](features[name])
                       for name in ("audio", "visual", "text")], dim=-1)
        x = self.fuse(x) * mask.unsqueeze(-1)
        for block in self.temporal:
            x = block(x, mask)
        return self.head(x).squeeze(-1)


def topk_bag_probability(logits, lengths, proportion=3):
    values = []
    for row, length in zip(logits, lengths):
        length = int(length)
        k = max(1, (length + proportion - 1) // proportion)
        values.append(torch.sigmoid(row[:length]).topk(k).values.mean())
    return torch.stack(values)
