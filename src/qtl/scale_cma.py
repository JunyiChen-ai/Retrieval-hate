"""Scale-aware cross-modal attention prior (`experiments/20261031_scale_aware_cma/README.md`, section 2).

r5's content prior (qtl.model.PriorNet, MACIL-SD's cross-modal layer) with one change: the attention score between
query second t and key second t' gets a learned bias b[h, k(|t - t'|)] per head, k = 0 for the same second and
1 + floor(log2 d) for distance d >= 1, capped at N_SCALE - 1 (the dyadic scales of the question tree's nodes).
The table starts at 0, so at initialisation the network is exactly r5; `scale_bias=false` freezes it at 0 (ablation).
The bias is a few scalars, so it trains at the small-parameter rate (lr_answer), like the chain transition.
"""
import math

import torch
from torch import nn
import torch.nn.functional as F

from qtl.model import MultiHeadAttention, PriorNet

N_SCALE = 11            # distances 0, 1, 2-3, ..., 256-511, >= 512 seconds
EXTRA_DEFAULTS = {"backbone": "scale_cma", "answer_source": "soft_both", "soft_levels": 8, "node_prior": True,
                  "scale_bias": True}


def scale_buckets(T, device):
    """(T, T) int64: 0 on the diagonal, 1 + floor(log2 |i - j|) elsewhere, capped at N_SCALE - 1."""
    d = torch.arange(T, device=device)
    d = (d[:, None] - d[None, :]).abs()
    # floor(log2 d) for integers d >= 1 = bit length - 1, computed exactly on integers
    k = torch.zeros_like(d)
    m = d.clone()
    while bool((m > 1).any()):
        k = k + (m > 1).long()
        m = m // 2
    return torch.where(d > 0, (k + 1).clamp(max=N_SCALE - 1), torch.zeros_like(d))


class _BucketBias(torch.autograd.Function):
    """table[:, buckets] with the table gradient computed as a dense (heads x T*T) by (T*T x scales) product.
    Same forward values as the index lookup; the lookup's own backward is an index scatter-add of 4*T*T values into 44
    slots, which is slow on GPU for long videos (experiment README section 6, run-time note)."""

    @staticmethod
    def forward(ctx, table, buckets):
        ctx.save_for_backward(buckets)
        ctx.n_scale = table.size(1)
        return table[:, buckets]

    @staticmethod
    def backward(ctx, grad):
        buckets, = ctx.saved_tensors
        onehot = F.one_hot(buckets.reshape(-1), ctx.n_scale).to(grad.dtype)        # (T*T, K)
        return torch.matmul(grad.reshape(grad.size(0), -1), onehot), None


class ScaleBiasAttention(MultiHeadAttention):
    def __init__(self, nhead, hid, learn=True, dropout=0.1):
        super().__init__(nhead, hid, dropout)
        self.scale_bias = nn.Parameter(torch.zeros(nhead, N_SCALE), requires_grad=learn)
        self._buckets = {}

    def buckets(self, T, device):
        key = (T, str(device))
        if key not in self._buckets:
            if len(self._buckets) > 64:
                self._buckets.clear()
            self._buckets[key] = scale_buckets(T, device)
        return self._buckets[key]

    def forward(self, q, k, v, key_mask=None):
        B, T = q.size(0), q.size(1)
        assert k.size(1) == T, "cross-modal attention over the same seconds"
        sp = lambda lin, x: lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)  # noqa: E731
        q, k, v = sp(self.lin_q, q), sp(self.lin_k, k), sp(self.lin_v, v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores + _BucketBias.apply(self.scale_bias, self.buckets(T, scores.device))[None]
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], -1e9)
        p = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(p, v).transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.lin_o(out)


class ScaleAwarePrior(PriorNet):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.cma.attn = ScaleBiasAttention(int(cfg["nhead"]), int(cfg["hid_dim"]), bool(cfg["scale_bias"]))

    def small_parameters(self):
        """Few-scalar structural parameters, trained at lr_answer (qtl.train optimizer groups)."""
        return [self.cma.attn.scale_bias] if self.cma.attn.scale_bias.requires_grad else []


def make_model(cfg):
    assert cfg["backbone"] == "scale_cma"
    return ScaleAwarePrior(cfg)
