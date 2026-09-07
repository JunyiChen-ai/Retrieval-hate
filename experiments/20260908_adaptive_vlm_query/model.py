"""Backbone for the adaptive VLM query module (2026-09-08).

Same network as candidate 3 (revision-2 backbone by default: evidence e_t in
q/k, per-head KEY bias, video-level context c added to both streams; the
variant is a config: cfg.bias_mode in {key, gated}, cfg.ctx_mode in {rep,
logit}) with ONE change: the evidence cell embedding has a third fine-verdict
value "not asked" (b_fine = -1), so e_t = Emb[3 * b_coarse + (b_fine + 1)] +
W [ell / ELL_SCALE, P(s)]; ell and P(s) come from the interval HMM run with
missing emissions. Arms: full | no_missing_state (-1 mapped to 0, four cells).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import hier_evidence_common as hc

N_EVID = 4     # ell, p_s, b_fine, b_coarse
STRUCT_ARMS = ("full", "no_missing_state")


class EvidenceEncoder(nn.Module):
    """e_t = Emb[cell(b_fine, b_coarse)] + W [ell/ELL_SCALE, p_s].
    missing_state=True: b_fine in {-1 (not asked), 0, 1} -> 6 cells;
    False: -1 treated as 0 -> 4 cells (revision 2/3 encoder)."""

    def __init__(self, hid, missing_state=True):
        super().__init__()
        self.missing_state = missing_state
        self.cell = nn.Embedding(6 if missing_state else 4, hid)
        nn.init.zeros_(self.cell.weight)      # starts as the linear map
        self.lin = nn.Linear(2, hid)

    def forward(self, evid):                       # evid: (B, T, 4), ell already / ELL_SCALE
        bf = evid[..., 2]
        bc = (evid[..., 3] > 0.5).long()
        if self.missing_state:
            fi = torch.where(bf < -0.5, torch.zeros_like(bc),
                             torch.where(bf > 0.5, torch.full_like(bc, 2), torch.ones_like(bc)))
            idx = 3 * bc + fi
        else:
            idx = 2 * (bf > 0.5).long() + bc
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
        self.concat = False
        a_in = hc.SCAF_OFFSET
        self.fc_v = nn.Linear(hc.align.V_DIM, hid)
        self.fc_a = nn.Linear(a_in, hid)
        self.enc = EvidenceEncoder(hid, missing_state=(arm != "no_missing_state"))
        bias_mode = str(getattr(cfg, "bias_mode", "key"))               # key (rev 2) | gated (rev 3)
        self.cma = EvidenceRoutedCMA(hid, nhead, ffn, dropout, bias_mode=bias_mode,
                                     qk_enc=(arm not in ("avce", "no_qk_enc")))
        self.ctx_mode = str(getattr(cfg, "ctx_mode", "rep"))              # rep (rev 2) | logit (rev 3)
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
            if self.ctx_mode == "rep":                                 # revision-2 placement (full)
                v_out = v_out + c[:, None, :]
                a_out = a_out + c[:, None, :]
                c = None
        a_log = self.fc(a_out)
        v_log = self.fc(v_out)
        av_log = a_log + v_log
        self.last_content_logit = av_log
        self.last_calibration = c
        if c is not None:                                              # ctx_on_logit arm only
            av_log = av_log + c[:, None, :]
        if not self.no_verdict:
            av_log = av_log + self.prior_scale * ell / hc.ELL_SCALE
        mmil = self.bag(av_log, seq_len)
        return mmil, torch.sigmoid(a_log), torch.sigmoid(v_log), av_log, v_out, a_out
