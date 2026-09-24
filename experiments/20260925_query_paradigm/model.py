"""Content prior network of the query-tree method (README section 2.3).

Encoder: MACIL-SD's cross-modal attention layer (one pre-norm transformer layer shared by the visual->audio+text
and audio+text->visual directions; the plain layer of experiments/20260910_online_query_within/model.py with no
evidence code, no key bias, no context term), on the 1-second grid.
Heads: per-second logit s_t = fc(a_out_t) + fc(v_out_t) (MACIL-SD's shared head), pi_t = sigmoid(s_t); video logit
g = w . sum_t softmax(att(h_t)) h_t + b with h_t = a_out_t + v_out_t (attention pooling).
The network never reads a VLM answer."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from macilsd import align                 # noqa: F401  (V_DIM)
import data as qdata


class MultiHeadAttention(nn.Module):
    """MACIL-SD's multi-head attention (4 linears, dropout on the attention probabilities), key padding mask."""

    def __init__(self, nhead, hid, dropout=0.1):
        super().__init__()
        assert hid % nhead == 0
        self.h, self.d_k = nhead, hid // nhead
        self.lin_q = nn.Linear(hid, hid)
        self.lin_k = nn.Linear(hid, hid)
        self.lin_v = nn.Linear(hid, hid)
        self.lin_o = nn.Linear(hid, hid)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, key_mask=None):
        B = q.size(0)
        sp = lambda lin, x: lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)  # noqa: E731
        q, k, v = sp(self.lin_q, q), sp(self.lin_k, k), sp(self.lin_v, v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if key_mask is not None:
            scores = scores.masked_fill(~key_mask[:, None, None, :], -1e9)
        p = self.dropout(F.softmax(scores, dim=-1))
        out = torch.matmul(p, v).transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.lin_o(out)


class CrossModalLayer(nn.Module):
    def __init__(self, hid, nhead, ffn, dropout):
        super().__init__()
        self.attn = MultiHeadAttention(nhead, hid)
        self.ff = nn.Sequential(nn.Linear(hid, ffn), nn.ReLU(), nn.Dropout(0.1), nn.Linear(ffn, hid))
        self.norm_attn = nn.LayerNorm(hid)
        self.norm_ff = nn.LayerNorm(hid)
        self.drop = nn.Dropout(dropout)

    def one(self, x, y, mask):
        h = x + self.drop(self.attn(self.norm_attn(x), y, y, key_mask=mask))
        return h + self.drop(self.ff(self.norm_ff(h)))

    def forward(self, v, a, mask):
        return self.one(v, a, mask), self.one(a, v, mask)


class PriorNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        hid = int(cfg["hid_dim"])
        self.fc_v = nn.Linear(align.V_DIM, hid)
        self.fc_a = nn.Linear(qdata.A_IN, hid)
        self.cma = CrossModalLayer(hid, int(cfg["nhead"]), int(cfg["ffn_dim"]), float(cfg["dropout"]))
        self.fc = nn.Linear(hid, 1)
        self.att = nn.Linear(hid, 1)
        self.vid = nn.Linear(hid, 1)
        # g_head false (README section 8): no video head; g is one learned constant, so only the per-second
        # logits can separate positive from negative videos
        self.g_head = bool(cfg.get("g_head", True))
        self.g0 = nn.Parameter(torch.zeros(()))

    def forward(self, f_a, f_v, mask):
        """f_a (B, T, A_IN), f_v (B, T, V_DIM), mask (B, T) bool. Returns s (B, T), g_logit (B,), a_log (B, T),
        v_log (B, T), v_out, a_out."""
        v_out, a_out = self.cma(self.fc_v(f_v), self.fc_a(f_a), mask)
        a_log = self.fc(a_out).squeeze(-1)
        v_log = self.fc(v_out).squeeze(-1)
        s = a_log + v_log
        h = a_out + v_out
        w = self.att(h).squeeze(-1).masked_fill(~mask, -1e9)
        pooled = torch.einsum("bt,bth->bh", torch.softmax(w, dim=1), h)
        g_logit = self.vid(pooled).squeeze(-1) if self.g_head else self.g0.expand(s.shape[0])
        return s, g_logit, a_log, v_log, v_out, a_out
