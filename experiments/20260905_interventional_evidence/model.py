"""Candidate 5: aligned evidence association and train-time Yager fusion.

Prototype, not yet code-reviewed or trained. Inputs after A_EXT_DIM are
two groups of eight channels [L0,dv,da,dav,H_av,H_v,H_a,H_0]. Old scaffold
channels are never fed into either head; retained only for train block loss.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from hier_evidence_common import A_EXT_DIM, SCAF_OFFSET


class Candidate(nn.Module):
    def __init__(self, dropout=0.2, ablation='full'):
        super().__init__()
        self.ablation = ablation
        self.content = nn.Sequential(nn.Linear(1024 + SCAF_OFFSET, 128), nn.GELU(),
                                     nn.Dropout(dropout))
        self.evidence = nn.Linear(16, 128)
        self.query = nn.Linear(128, 128, bias=False)
        self.key = nn.Linear(128, 128, bias=False)
        self.value = nn.Linear(128, 128, bias=False)
        self.context = nn.Sequential(nn.Linear(512, 128), nn.GELU(), nn.Dropout(dropout))
        self.content_head = nn.Linear(128, 2)
        self.evidence_head = nn.Linear(128, 2)
        self.fusion_scale = nn.Parameter(torch.tensor(1.0))

    @staticmethod
    def opinion(e):
        total = e.sum(-1, keepdim=True) + 2
        return e / total, 2 / total

    def forward(self, f_a, f_v, seq_len=None):
        bsz, length, _ = f_v.shape
        if f_a.shape[-1] != A_EXT_DIM + 16:
            raise ValueError('expected old scaffold bookkeeping + sixteen evidence channels')
        x = self.content(torch.cat([f_v, f_a[..., :SCAF_OFFSET]], -1))
        ev = self.evidence(f_a[..., A_EXT_DIM:])
        q = self.query(x).reshape(bsz, length, 4, 32).transpose(1, 2)
        k = self.key(ev).reshape(bsz, length, 4, 32).transpose(1, 2)
        value = self.value(x).reshape(bsz, length, 4, 32).transpose(1, 2)
        affinity = q @ k.transpose(-1, -2) / math.sqrt(32)
        if seq_len is None:
            seq_len = torch.full((bsz,), length, device=x.device, dtype=torch.long)
        else:
            seq_len = seq_len.to(x.device)
        invalid = torch.arange(length, device=x.device)[None, :] >= seq_len[:, None]
        if (seq_len <= 0).any():
            raise ValueError('empty input sequence')
        mask = invalid[:, None, None, :]
        positive = affinity.masked_fill(mask, -torch.inf).softmax(-1) @ value
        negative = (-affinity).masked_fill(mask, -torch.inf).softmax(-1) @ value
        positive = positive.transpose(1, 2).reshape(bsz, length, 128)
        negative = negative.transpose(1, 2).reshape(bsz, length, 128)
        if self.ablation == 'ordinary_attention':
            negative = torch.zeros_like(negative)
            difference = torch.zeros_like(positive)
        else:
            difference = positive - negative
        hidden = self.context(torch.cat([x, positive, negative, difference], -1))
        e1, e2 = F.softplus(self.content_head(hidden)), F.softplus(self.evidence_head(ev))
        b1, u1 = self.opinion(e1)
        b2, u2 = self.opinion(e2)
        conflict = (b1[..., 0:1] * b2[..., 1:2] + b1[..., 1:2] * b2[..., 0:1])
        belief = b1 * b2 + b1 * u2 + u1 * b2
        unknown = u1 * u2 + conflict
        if self.ablation == 'dempster_fusion':
            belief = belief / (1 - conflict).clamp_min(1e-7)
            unknown = u1 * u2 / (1 - conflict).clamp_min(1e-7)
        probability = (belief + unknown / 2)[..., 1:2]
        if self.ablation == 'additive_fusion':
            p1, p2 = (b1 + u1 / 2)[..., 1:2], (b2 + u2 / 2)[..., 1:2]
            probability = torch.sigmoid(torch.logit(p1.clamp(1e-7, 1-1e-7)) +
                                        self.fusion_scale * torch.logit(p2.clamp(1e-7, 1-1e-7)))
        logits = torch.logit(probability.clamp(1e-7, 1-1e-7))
        self.last_content_logit = torch.logit((b1 + u1 / 2)[..., 1:2].clamp(1e-7, 1-1e-7))
        bags = torch.stack([probability[i, :int(n), 0].topk(max(1, math.ceil(int(n)/16))).values.mean()
                            for i, n in enumerate(seq_len)])
        return bags, logits, logits, logits, hidden, ev
