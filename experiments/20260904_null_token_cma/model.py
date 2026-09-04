"""Null-token cross-modal attention backbone (candidate 4, 2026-09-04).

Starting point: candidate 1's MACIL-SD AVCE backbone (experiments/
20260903_hier_evidence_mil): fc_v / fc_a projections (the four verdict columns
concatenated into the audio+text vector, the proven evidence path), ONE
transformer layer shared by both cross-modal directions, head logit
fc(a_out) + fc(v_out). Observation that motivates this candidate (candidate 3
README 7.1): MACIL-SD never masks padded rows in the attention; at training
time the zero rows (all mapped to the same bias vector by the projection)
receive on average .254 of the attention mass of valid queries (HateMM, 744
train videos, .324 of the keys are padding), i.e. every second can partly
"attend to nothing"; at test time sequences are not truncated, there is no
padding and that mass falls back onto real seconds. Masking the padding
(candidate 3) costs HateMM ROC -.017 / within -.011. So the ability to decline
cross-modal context is useful, but the current form is an accident that is
absent at test time.

This backbone replaces the accidental sink by an explicit, train/test
consistent one: each key modality gets ONE extra key/value token
    n_m = b_m + W_m c,      c = mean_t x_t   (video-level summary of the four
                                              verdict columns, valid rows only)
appended to its sequence; padded keys are masked. A query that finds no
relevant second in the other modality attends to n_m instead; because n_m
carries the video's verdict summary, "attending to nothing" adds the
video-level evidence context (candidate 1 README 9.5: video-level density
is the backbone's main mechanism) without pushing every row towards the
same evidence seconds (which erased within-video ordering in candidate 3).

Arms (train.py --ablation):
  full               evidence-conditioned null token per key modality, padding masked
  const_token        null token = b_m only (pure learnable sink)
  shared_token       one null token for both key modalities
  no_token_masked    padding masked, no token (candidate 3's setting)
  no_token_unmasked  candidate 1 / MACIL-SD exactly (padding as accidental sink)
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import hier_evidence_common as hc

N_EVID = 4
STRUCT_ARMS = ("full", "const_token", "shared_token", "no_token_masked",
               "no_token_unmasked")


class MultiHeadAttention(nn.Module):
    """MACIL-SD's multi-head attention (four linears, dropout .1 on the
    attention probabilities) with an optional key padding mask."""

    def __init__(self, nhead, hid, dropout=0.1):
        super().__init__()
        assert hid % nhead == 0
        self.h, self.d_k = nhead, hid // nhead
        self.lin_q = nn.Linear(hid, hid)
        self.lin_k = nn.Linear(hid, hid)
        self.lin_v = nn.Linear(hid, hid)
        self.lin_o = nn.Linear(hid, hid)
        self.dropout = nn.Dropout(dropout)
        self.attn = None

    def forward(self, q, k, v, key_mask=None):
        B = q.size(0)
        sp = lambda lin, x: lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)  # noqa: E731
        q, k, v = sp(self.lin_q, q), sp(self.lin_k, k), sp(self.lin_v, v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        if key_mask is not None:                       # (B, Tk) bool, True = valid
            scores = scores.masked_fill(~key_mask[:, None, None, :], -1e9)
        p = self.dropout(F.softmax(scores, dim=-1))
        self.attn = p
        out = torch.matmul(p, v).transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.lin_o(out)


class NullTokenCMA(nn.Module):
    """One pre-norm transformer layer (MACIL-SD's TransformerLayer: LayerNorm on
    the query only, residual + dropout, position-wise FFN) shared by both
    cross-modal directions. The key sequence of modality m is extended by the
    null token n_m (prepended at position 0); padded keys are masked."""

    def __init__(self, hid, nhead, ffn, dropout, token="evidence", mask=True):
        super().__init__()
        assert token in ("evidence", "const", "shared", "none")
        self.attn = MultiHeadAttention(nhead, hid)
        self.ff = nn.Sequential(nn.Linear(hid, ffn), nn.ReLU(), nn.Dropout(0.1),
                                nn.Linear(ffn, hid))
        self.norm_attn = nn.LayerNorm(hid)
        self.norm_ff = nn.LayerNorm(hid)
        self.drop = nn.Dropout(dropout)
        self.token = token
        self.mask = bool(mask)
        if token == "none":
            self.base = None
        elif token == "shared":
            self.base = nn.Parameter(torch.zeros(1, hid))
            self.cond = nn.Linear(N_EVID, hid)
        else:
            self.base = nn.Parameter(torch.zeros(2, hid))        # [visual, audio]
            self.cond = nn.Linear(N_EVID, hid) if token == "evidence" else None

    def null_token(self, which, c):
        """(B, 1, hid) null token of key modality `which` (0 visual, 1 audio)."""
        if self.base is None:
            return None
        b = self.base[0] if self.token == "shared" else self.base[which]
        n = b[None, :].expand(c.shape[0], -1)
        if self.cond is not None:
            n = n + self.cond(c)
        return n[:, None, :]

    def one(self, x, y, which_y, c, mask):
        n = self.null_token(which_y, c)
        keys = y if n is None else torch.cat([n, y], dim=1)
        km = None
        if self.mask:
            km = mask if n is None else torch.cat(
                [torch.ones(mask.shape[0], 1, dtype=torch.bool, device=mask.device), mask], dim=1)
        h = x + self.drop(self.attn(self.norm_attn(x), keys, keys, key_mask=km))
        return h + self.drop(self.ff(self.norm_ff(h)))

    def forward(self, v, a, c, mask):
        return self.one(v, a, 1, c, mask), self.one(a, v, 0, c, mask)


class NTCA(nn.Module):
    """Same forward signature / return tuple as candidate 1's Candidate:
    (mmil, sigmoid a_log, sigmoid v_log, av_log with prior, v_out, a_out);
    `last_content_logit` = av_log without the prior (verdict-block MIL reads it)."""

    def __init__(self, cfg, prior_scale, arm="full", no_verdict=False, hide_input=False):
        super().__init__()
        assert arm in STRUCT_ARMS, arm
        hid, nhead, ffn, dropout = cfg.hid_dim, cfg.nhead, cfg.ffn_dim, cfg.dropout
        self.arm = arm
        self.no_verdict = bool(no_verdict)
        self.hide_input = bool(hide_input)
        self.prior_scale = float(prior_scale)
        self.topk_div = int(cfg.topk_div)
        self.fc_v = nn.Linear(hc.align.V_DIM, hid)
        self.fc_a = nn.Linear(hc.SCAF_OFFSET + N_EVID, hid)     # audio ⊕ text ⊕ 4 verdict columns
        token = {"full": "evidence", "const_token": "const", "shared_token": "shared",
                 "no_token_masked": "none", "no_token_unmasked": "none"}[arm]
        self.cma = NullTokenCMA(hid, nhead, ffn, dropout, token=token,
                                mask=(arm != "no_token_unmasked"))
        self.fc = nn.Linear(hid, cfg.num_classes)               # shared head (Att_MMIL)
        self.last_content_logit = None

    def bag(self, logits, seq_len):
        logits = logits.squeeze(-1)
        out = []
        for i in range(logits.shape[0]):
            if seq_len is None:
                out.append(logits[i].mean().view(1))
            else:
                t = int(seq_len[i])
                k = max(1, int(-(-t // self.topk_div)))
                out.append(torch.topk(logits[i][:t], k=k).values.mean().view(1))
        return torch.sigmoid(torch.cat(out))

    def forward(self, f_a, f_v, seq_len):
        B, T, _ = f_a.shape
        if seq_len is None:
            mask = torch.ones(B, T, dtype=torch.bool, device=f_a.device)
        else:
            mask = torch.arange(T, device=f_a.device)[None, :] < seq_len.to(f_a.device)[:, None]
        evid = f_a[..., hc.SCAF_OFFSET:hc.SCAF_OFFSET + N_EVID].clone()
        evid[..., hc.COL_ELL] = evid[..., hc.COL_ELL] / hc.ELL_SCALE     # in [-1, 1]
        ell = f_a[..., hc.SCAF_OFFSET + hc.COL_ELL:hc.SCAF_OFFSET + hc.COL_ELL + 1]
        if self.no_verdict or self.hide_input:
            evid = torch.zeros_like(evid)
        m = mask[..., None].float()
        c = (evid * m).sum(1) / m.sum(1).clamp(min=1.0)                  # (B, 4) video summary
        h_a = self.fc_a(torch.cat([f_a[..., :hc.SCAF_OFFSET], evid], dim=-1))
        h_v = self.fc_v(f_v)
        v_out, a_out = self.cma(h_v, h_a, c, mask)
        a_log = self.fc(a_out)
        v_log = self.fc(v_out)
        av_log = a_log + v_log
        self.last_content_logit = av_log
        if not self.no_verdict:
            av_log = av_log + self.prior_scale * ell / hc.ELL_SCALE
        mmil = self.bag(av_log, seq_len)
        return mmil, torch.sigmoid(a_log), torch.sigmoid(v_log), av_log, v_out, a_out
