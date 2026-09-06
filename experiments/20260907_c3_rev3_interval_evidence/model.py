"""Evidence-routed cross-modal attention backbone (candidate 3 revision 3, 2026-09-07).

Revision 2 (experiments/20260904_evidence_guided_attention/model.py) put the
frozen-VLM evidence into MACIL-SD's shared cross-modal attention layer in three
places: (A) added to the attention queries and keys, (B) a per-head additive
bias on every key second computed from that second's evidence, (C) a
video-level context vector added to both modality streams before the head.
The external review (REVIEW_NOVELTY_GPT6ASTRA.md of that experiment) showed
that (C) contradicts the stated mechanism -- the context enters the
representations that CMAL contrasts and the block MIL reads -- and that (B)
does not depend on the querying second, so every second is pulled towards the
same evidence seconds (within-video ROC falls on HateMM).

Revision 3 keeps the content path of MACIL-SD untouched and makes the
mechanism literally true in code: evidence decides WHERE each second
aggregates content from and, separately, a video-level calibration of the
logit; it never enters a representation.

  A  evidence encoding e_t = Emb[cell(b_fine_t, b_coarse_t)] + W [ell_t /
     ELL_SCALE, P(s_t)], added to the attention queries and keys only
     (values, residual streams unchanged);
  B' query-conditioned evidence routing: score_h(i, j) += g_h(e_i) * beta_h(e_j),
     beta_h = Linear(hid -> nhead) on the key second, g_h = 2 sigmoid(Linear(hid
     -> nhead)(e_i)) on the query second (zero-initialised: g = 1, i.e. the
     revision-2 key bias is the starting point). A second whose own verdicts
     are decisive can take less from the evidence seconds;
  C' video-level evidence calibration: c = Linear(hid -> 1)(mean_t e_t), added to
     the per-second logit only. CMAL and the verdict-block MIL see fc(a_out) +
     fc(v_out) without c and without the prior.

Arms (train.py --ablation, structure):
  full        A + B' + C'
  avce        candidate 1 backbone: verdict columns concatenated into the audio
              stream through fc_a, no A / B' / C'
  no_qk_enc   e_t not added to q/k (B', C' kept)
  no_cell     Emb[cell] replaced by a linear map of all four columns
  no_bias     no evidence routing term at all
  key_bias    g == 1: revision-2 key-only bias (is the query conditioning needed?)
  shared_bias g == 1 and beta = Linear(hid -> 1) shared by all heads (is the
              per-head bias needed? -- the isolating control the review asked for)
  no_context  no C'
  ctx_in_rep  revision-2 placement: c (hid-dim) added to both streams before
              the head (does moving it out of the representation cost anything?)
Key padding is masked in the attention.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import hier_evidence_common as hc

N_EVID = 4     # ell, p_s, b_fine, b_coarse
STRUCT_ARMS = ("full", "avce", "no_qk_enc", "no_cell", "no_bias", "key_bias",
               "shared_bias", "no_context", "ctx_in_rep")


class EvidenceEncoder(nn.Module):
    """e_t = Emb[2*b_fine + b_coarse] + W [ell/ELL_SCALE, p_s]  (cell=True), or a
    linear map of all four columns (cell=False, the `no_cell` arm)."""

    def __init__(self, hid, cell=True):
        super().__init__()
        self.cell = nn.Embedding(4, hid) if cell else None
        if cell:
            nn.init.zeros_(self.cell.weight)      # starts as the linear map
        self.lin = nn.Linear(2 if cell else N_EVID, hid)

    def forward(self, evid):                       # evid: (B, T, 4), ell already / ELL_SCALE
        if self.cell is None:
            return self.lin(evid)
        idx = (2 * (evid[..., 2] > 0.5).long() + (evid[..., 3] > 0.5).long())
        return self.cell(idx) + self.lin(evid[..., :2])


class BiasedMultiHeadAttention(nn.Module):
    """MACIL-SD's multi-head attention (4 linears, dropout on the attention
    probabilities) plus an additive bias on the scores: (B, H, Tq, Tk)."""

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

    def forward(self, q, k, v, score_bias=None, key_mask=None):
        B = q.size(0)
        sp = lambda lin, x: lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)  # noqa: E731
        q, k, v = sp(self.lin_q, q), sp(self.lin_k, k), sp(self.lin_v, v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)   # B,H,Tq,Tk
        if score_bias is not None:
            scores = scores + score_bias
        if key_mask is not None:                       # (B, Tk) bool, True = valid
            scores = scores.masked_fill(~key_mask[:, None, None, :], -1e9)
        p = self.dropout(F.softmax(scores, dim=-1))
        self.attn = p
        out = torch.matmul(p, v).transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.lin_o(out)


class EvidenceRoutedCMA(nn.Module):
    """One pre-norm transformer layer (MACIL-SD's TransformerLayer: LayerNorm on
    the query, residual + dropout, position-wise FFN) shared by both cross-modal
    directions, with the evidence routing term (part B')."""

    def __init__(self, hid, nhead, ffn, dropout, bias_mode="gated", qk_enc=True):
        super().__init__()
        assert bias_mode in ("gated", "key", "shared", "none")
        self.nhead = nhead
        self.qk_enc = bool(qk_enc)
        self.bias_mode = bias_mode
        self.attn = BiasedMultiHeadAttention(nhead, hid)
        self.ff = nn.Sequential(nn.Linear(hid, ffn), nn.ReLU(), nn.Dropout(0.1),
                                nn.Linear(ffn, hid))
        self.norm_attn = nn.LayerNorm(hid)
        self.norm_ff = nn.LayerNorm(hid)
        self.drop = nn.Dropout(dropout)
        self.beta = None
        self.gate = None
        if bias_mode in ("gated", "key"):
            self.beta = nn.Linear(hid, nhead)
        elif bias_mode == "shared":
            self.beta = nn.Linear(hid, 1)
        if self.beta is not None:
            nn.init.zeros_(self.beta.weight)          # starts as plain shared CMA
            nn.init.zeros_(self.beta.bias)
        if bias_mode == "gated":
            self.gate = nn.Linear(hid, nhead)
            nn.init.zeros_(self.gate.weight)          # g = 2 sigmoid(0) = 1: starts as the key bias
            nn.init.zeros_(self.gate.bias)

    def routing(self, e):
        """(B, H, Tq, Tk) additive score term from the evidence codes, or None."""
        if self.beta is None or e is None:
            return None
        kb = self.beta(e)                                        # (B, Tk, H or 1)
        if self.bias_mode == "shared":
            kb = kb.expand(-1, -1, self.nhead)
        kb = kb.transpose(1, 2)                                  # (B, H, Tk)
        if self.gate is None:
            return kb[:, :, None, :]
        g = 2.0 * torch.sigmoid(self.gate(e)).transpose(1, 2)    # (B, H, Tq)
        return g[:, :, :, None] * kb[:, :, None, :]

    def one(self, x, y, e, mask):
        q_in, k_in = self.norm_attn(x), y
        if self.qk_enc and e is not None:            # evidence in q/k only (A)
            q_in = q_in + e
            k_in = y + e
        h = x + self.drop(self.attn(q_in, k_in, y, score_bias=self.routing(e), key_mask=mask))
        return h + self.drop(self.ff(self.norm_ff(h)))

    def forward(self, v, a, e, mask):
        return self.one(v, a, e, mask), self.one(a, v, e, mask)


class ERCA(nn.Module):
    """Evidence-routed cross-modal attention model. Same forward signature and
    return tuple as candidate 1's Candidate (mmil, sigmoid a_log, sigmoid v_log,
    av_log with calibration and prior, v_out, a_out); `last_content_logit` =
    fc(a_out) + fc(v_out) without the video-level calibration and without the
    prior (CMAL uses a_out / v_out; the verdict-block MIL reads this logit)."""

    def __init__(self, cfg, prior_scale, arm="full", no_verdict=False):
        super().__init__()
        assert arm in STRUCT_ARMS, arm
        hid, nhead, ffn, dropout = cfg.hid_dim, cfg.nhead, cfg.ffn_dim, cfg.dropout
        self.arm = arm
        self.no_verdict = bool(no_verdict)
        self.prior_scale = float(prior_scale)
        self.topk_div = int(cfg.topk_div)
        self.concat = arm == "avce"
        a_in = hc.SCAF_OFFSET + (N_EVID if self.concat else 0)
        self.fc_v = nn.Linear(hc.align.V_DIM, hid)
        self.fc_a = nn.Linear(a_in, hid)
        self.enc = None if self.concat else EvidenceEncoder(hid, cell=(arm != "no_cell"))
        bias_mode = {"avce": "none", "no_bias": "none", "key_bias": "key",
                     "shared_bias": "shared"}.get(arm, "gated")
        self.cma = EvidenceRoutedCMA(hid, nhead, ffn, dropout, bias_mode=bias_mode,
                                     qk_enc=(arm not in ("avce", "no_qk_enc")))
        self.ctx_mode = {"avce": None, "no_context": None, "ctx_in_rep": "rep"}.get(arm, "logit")
        if self.ctx_mode == "logit":
            self.ctx = nn.Linear(hid, cfg.num_classes)
        elif self.ctx_mode == "rep":
            self.ctx = nn.Linear(hid, hid)
        else:
            self.ctx = None
        self.fc = nn.Linear(hid, cfg.num_classes)      # shared head (Att_MMIL)
        self.last_content_logit = None
        self.last_calibration = None

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
        if self.no_verdict:
            evid = torch.zeros_like(evid)
        content_a = f_a[..., :hc.SCAF_OFFSET]
        h_v = self.fc_v(f_v)
        if self.concat:
            h_a = self.fc_a(torch.cat([content_a, evid], dim=-1))
            e = None
        else:
            h_a = self.fc_a(content_a)
            e = self.enc(evid) * mask[..., None].float()
        v_out, a_out = self.cma(h_v, h_a, e, mask)
        c = None
        if self.ctx is not None and e is not None:
            m = mask[..., None].float()
            c = self.ctx((e * m).sum(1) / m.sum(1).clamp(min=1.0))      # (B, hid) or (B, 1)
            if self.ctx_mode == "rep":                                 # revision-2 placement
                v_out = v_out + c[:, None, :]
                a_out = a_out + c[:, None, :]
                c = None
        a_log = self.fc(a_out)
        v_log = self.fc(v_out)
        av_log = a_log + v_log
        self.last_content_logit = av_log
        self.last_calibration = c
        if c is not None:                                              # C': logit only
            av_log = av_log + c[:, None, :]
        if not self.no_verdict:
            av_log = av_log + self.prior_scale * ell / hc.ELL_SCALE
        mmil = self.bag(av_log, seq_len)
        return mmil, torch.sigmoid(a_log), torch.sigmoid(v_log), av_log, v_out, a_out
