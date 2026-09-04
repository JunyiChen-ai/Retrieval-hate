"""Evidence-guided cross-modal attention backbone (candidate 3, 2026-09-04).

Starting point: the MACIL-SD AVCE backbone of experiments/20260903_hier_evidence_mil
(candidate 1): fc_v / fc_a projections, ONE transformer layer shared by both
cross-modal directions, a head whose per-row logit is fc(a_out) + fc(v_out).
Candidate 1's structure ablations (its README 9.7) show the shared
cross-modal layer is the only AVCE part with a confirmed effect on both
corpora, and its mechanism analysis (README 9.2-9.5) shows the backbone works
by (i) a video-level hate-density estimate read off the verdict distribution,
(ii) a learned correction of the relative reliability of the two verdict
granularities, (iii) within-video ordering from the HMM posterior. In
candidate 1 all of this is implicit: the four verdict columns are concatenated
into the 896-dim audio+text vector and pass through fc_a.

This backbone makes the three mechanisms explicit, ablatable parts:

  A  evidence encoding shared by both modalities: e_t = Emb[cell(b_fine_t,
     b_coarse_t)] + W [ell_t / ELL_SCALE, P(s_t)]. Revision 1 added it to both
     residual streams (like a positional encoding) -- that made the bag loss
     collapse onto the evidence within three epochs (README section 7).
     Revision 2 adds it to the attention queries and keys only: the residual
     streams and the attention values stay content-only, evidence decides
     WHERE content is aggregated from, never replaces it. The 2x2 agreement
     cell embedding is the reliability correction of README 9.2 made explicit;
  B  evidence-biased attention: the shared cross-modal layer adds, for every
     query of either modality, a per-head bias beta_h(e_j) on key second j,
     so both modalities attend to evidence-bearing seconds;
  C  video-level evidence context: c = mean_t e_t (valid rows) -> Linear, added
     to every row of both streams before the head (the density term of 9.5).

Arms (train.py --ablation): avce = candidate 1's backbone (verdicts
concatenated into the audio stream, no A/B/C) under this file's training;
no_qk_enc / no_cell / no_bias / no_context switch A, D, B, C; stream_enc is
revision 1's residual-stream encoding (record only);
scalar_bias (rule-4 review requirement) replaces beta_h(e_j) by one scalar
gamma * ell_j / ELL_SCALE shared by all heads.
Key padding is masked in the attention (candidate 1 / MACIL-SD did not).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import hier_evidence_common as hc

N_EVID = 4     # ell, p_s, b_fine, b_coarse
# revision 2 (2026-09-04 18:40, README section 7): the evidence encoding enters
# the attention QUERIES and KEYS only; the residual streams and the attention
# values stay content-only. `stream_enc` = revision 1's full (e_t added to both
# residual streams), `no_qk_enc` = revision 1's no_enc (e_t only via bias + context).
STRUCT_ARMS = ("full", "avce", "stream_enc", "no_qk_enc", "no_cell",
               "no_bias", "scalar_bias", "no_context")


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
    probabilities) plus an additive per-head bias on the keys."""

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

    def forward(self, q, k, v, key_bias=None, key_mask=None):
        B = q.size(0)
        sp = lambda lin, x: lin(x).view(B, -1, self.h, self.d_k).transpose(1, 2)  # noqa: E731
        q, k, v = sp(self.lin_q, q), sp(self.lin_k, k), sp(self.lin_v, v)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)   # B,H,Tq,Tk
        if key_bias is not None:                       # (B, H, Tk)
            scores = scores + key_bias[:, :, None, :]
        if key_mask is not None:                       # (B, Tk) bool, True = valid
            scores = scores.masked_fill(~key_mask[:, None, None, :], -1e9)
        p = self.dropout(F.softmax(scores, dim=-1))
        self.attn = p
        out = torch.matmul(p, v).transpose(1, 2).contiguous().view(B, -1, self.h * self.d_k)
        return self.lin_o(out)


class EvidenceGuidedCMA(nn.Module):
    """One pre-norm transformer layer (MACIL-SD's TransformerLayer: LayerNorm on
    the query, residual + dropout, position-wise FFN) shared by both cross-modal
    directions, with the evidence bias on the key seconds (part B)."""

    def __init__(self, hid, nhead, ffn, dropout, use_bias=True, scalar_bias=False,
                 qk_enc=True):
        super().__init__()
        self.nhead = nhead
        self.qk_enc = bool(qk_enc)
        self.attn = BiasedMultiHeadAttention(nhead, hid)
        self.ff = nn.Sequential(nn.Linear(hid, ffn), nn.ReLU(), nn.Dropout(0.1),
                                nn.Linear(ffn, hid))
        self.norm_attn = nn.LayerNorm(hid)
        self.norm_ff = nn.LayerNorm(hid)
        self.drop = nn.Dropout(dropout)
        # rule-4 review arm `scalar_bias`: one learnable gamma shared by all heads,
        # bias = gamma * ell_j / ELL_SCALE (no learned function of the evidence encoding)
        self.gamma = nn.Parameter(torch.zeros(1)) if scalar_bias else None
        self.bias = nn.Linear(hid, nhead) if (use_bias and not scalar_bias) else None
        if self.bias is not None:
            nn.init.zeros_(self.bias.weight)          # starts as plain shared CMA
            nn.init.zeros_(self.bias.bias)

    def one(self, x, y, e, mask, ell=None):
        kb = None
        if self.gamma is not None and ell is not None:
            kb = (self.gamma * ell)[:, None, :].expand(-1, self.nhead, -1)
        elif self.bias is not None and e is not None:
            kb = self.bias(e).transpose(1, 2)
        q_in, k_in = self.norm_attn(x), y
        if self.qk_enc and e is not None:            # revision 2: evidence in q/k only
            q_in = q_in + e
            k_in = y + e
        h = x + self.drop(self.attn(q_in, k_in, y, key_bias=kb, key_mask=mask))
        return h + self.drop(self.ff(self.norm_ff(h)))

    def forward(self, v, a, e, mask, ell=None):
        return self.one(v, a, e, mask, ell), self.one(a, v, e, mask, ell)


class EGCA(nn.Module):
    """Evidence-guided cross-modal attention model. Same forward signature and
    return tuple as candidate 1's Candidate (mmil, sigmoid a_log, sigmoid v_log,
    av_log with prior, v_out, a_out); `last_content_logit` = av_log without the
    prior (verdict-block MIL reads it)."""

    def __init__(self, cfg, prior_scale, arm="full", no_verdict=False):
        super().__init__()
        assert arm in STRUCT_ARMS, arm
        hid, nhead, ffn, dropout = cfg.hid_dim, cfg.nhead, cfg.ffn_dim, cfg.dropout
        self.arm = arm
        self.no_verdict = bool(no_verdict)
        self.prior_scale = float(prior_scale)
        self.topk_div = int(cfg.topk_div)
        self.key_mask = bool(cfg.get("key_mask", True))   # diagnostic: False = MACIL-SD's unmasked padded keys
        self.concat = arm == "avce"
        a_in = hc.SCAF_OFFSET + (N_EVID if self.concat else 0)
        self.fc_v = nn.Linear(hc.align.V_DIM, hid)
        self.fc_a = nn.Linear(a_in, hid)
        self.enc = None if self.concat else EvidenceEncoder(hid, cell=(arm != "no_cell"))
        self.cma = EvidenceGuidedCMA(hid, nhead, ffn, dropout,
                                     use_bias=(arm not in ("avce", "no_bias")),
                                     scalar_bias=(arm == "scalar_bias"),
                                     qk_enc=(arm not in ("avce", "stream_enc", "no_qk_enc")))
        self.ctx = nn.Linear(hid, hid) if arm not in ("avce", "no_context") else None
        self.fc = nn.Linear(hid, cfg.num_classes)      # shared head (Att_MMIL)
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
            if self.arm == "stream_enc":               # revision 1 design (record only)
                h_a = h_a + e
                h_v = h_v + e
        v_out, a_out = self.cma(h_v, h_a, e, mask if self.key_mask else None,
                                ell=evid[..., hc.COL_ELL] * mask.float())
        if self.ctx is not None and e is not None:
            m = mask[..., None].float()
            c = self.ctx((e * m).sum(1) / m.sum(1).clamp(min=1.0))      # (B, hid)
            v_out = v_out + c[:, None, :]
            a_out = a_out + c[:, None, :]
        a_log = self.fc(a_out)
        v_log = self.fc(v_out)
        av_log = a_log + v_log
        self.last_content_logit = av_log
        if not self.no_verdict:
            av_log = av_log + self.prior_scale * ell / hc.ELL_SCALE
        mmil = self.bag(av_log, seq_len)
        return mmil, torch.sigmoid(a_log), torch.sigmoid(v_log), av_log, v_out, a_out
