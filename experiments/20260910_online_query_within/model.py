"""Backbone of the online-query module (2026-09-10; verbatim copy of
experiments/20260908_adaptive_vlm_query/model.py, experiments may not import
each other).

Module-1 iteration 2 (2026-09-10, "text evidence"): the evidence encoder's linear
map also reads the per-second text log-likelihood ratio (scaffold column
hc.COL_TEXT, divided by hc.LLR_SCALE and clipped to [-1, 1]) when cfg.text_input;
arm no_text_input switches it off. The prior term stays alpha * ell, where ell now
comes from the HMM posterior that includes the text families.

Module-1 iteration 5 (2026-09-17, reviewer round): cfg.prior_mode = "split" replaces the
single prior term alpha * E_t by three learned scalars on the components of the
decomposed evidence, a_f * ell_fine + a_x * x_t + a_v * v (all divided by ELL_SCALE),
each initialised to alpha (prior_scale); ell_fine = COL_ELL - COL_TEXT - COL_V,
x_t = COL_TEXT, v = COL_V. "single" = the iteration 1-4 term (arm single_prior).

Same network as candidate 3 (revision-2 backbone by default: evidence e_t in
q/k, per-head KEY bias, video-level context c added to both streams; the
variant is a config: cfg.bias_mode in {key, gated}, cfg.ctx_mode in {rep,
logit}) with ONE change: the evidence cell embedding has a third fine-verdict
value "not asked" (b_fine = -1), so e_t = Emb[3 * b_coarse + (b_fine + 1)] +
W [ell / ELL_SCALE, P(s)]; ell and P(s) come from the interval HMM run with
missing emissions. Arms: full | no_missing_state (-1 mapped to 0, four cells).

Iteration-5 ablation table (README section 13, 2026-09-23): structure arms ported from
revision 4 -- avce (candidate-1 backbone: the four evidence columns concatenated into the
audio stream, no evidence code, no bias, no context), no_qk_enc (e_t not added to q/k),
no_cell (cell embedding replaced by a linear map of all four columns); ERCA flags
no_prior (no prior term) and no_verdict (evidence columns zeroed; the prior keeps only
the text term a_x * x_t).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

import hier_evidence_common as hc

N_EVID = 4     # ell, p_s, b_fine, b_coarse (+ the per-second text LLR column hc.COL_TEXT when text_input)
STRUCT_ARMS = ("full", "no_missing_state", "no_text_input", "single_prior", "avce", "no_qk_enc", "no_cell")


class EvidenceEncoder(nn.Module):
    """e_t = Emb[cell(b_fine, b_coarse)] + W [ell/ELL_SCALE, p_s].
    missing_state=True: b_fine in {-1 (not asked), 0, 1} -> 6 cells;
    False: -1 treated as 0 -> 4 cells (revision 2/3 encoder).
    cell=False (arm no_cell): no cell embedding, a linear map of all four columns
    [ell/ELL_SCALE, p_s, b_fine (-1 / 0 / 1), b_coarse] (revision-4 no_cell)."""

    def __init__(self, hid, missing_state=True, text_input=False, cell=True):
        super().__init__()
        self.missing_state = missing_state
        self.text_input = bool(text_input)
        self.use_cell = bool(cell)
        if self.use_cell:
            self.cell = nn.Embedding(6 if missing_state else 4, hid)
            nn.init.zeros_(self.cell.weight)      # starts as the linear map
            self.lin = nn.Linear(3 if self.text_input else 2, hid)
        else:
            self.cell = None
            self.lin = nn.Linear((N_EVID + 1) if self.text_input else N_EVID, hid)

    def forward(self, evid, text_llr=None):        # evid: (B, T, 4), ell already / ELL_SCALE; text_llr (B, T) already / LLR_SCALE
        if not self.use_cell:
            lin_in = evid
            if self.text_input:
                lin_in = torch.cat([lin_in, text_llr[..., None]], dim=-1)
            return self.lin(lin_in)
        bf = evid[..., 2]
        bc = (evid[..., 3] > 0.5).long()
        if self.missing_state:
            fi = torch.where(bf < -0.5, torch.zeros_like(bc),
                             torch.where(bf > 0.5, torch.full_like(bc, 2), torch.ones_like(bc)))
            idx = 3 * bc + fi
        else:
            idx = 2 * (bf > 0.5).long() + bc
        lin_in = evid[..., :2]
        if self.text_input:
            lin_in = torch.cat([lin_in, text_llr[..., None]], dim=-1)
        return self.cell(idx) + self.lin(lin_in)


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

    def __init__(self, cfg, prior_scale, arm="full", no_verdict=False, no_prior=False):
        super().__init__()
        assert arm in STRUCT_ARMS, arm
        hid, nhead, ffn, dropout = cfg.hid_dim, cfg.nhead, cfg.ffn_dim, cfg.dropout
        self.arm = arm
        self.no_verdict = bool(no_verdict)
        self.no_prior = bool(no_prior)
        self.prior_scale = float(prior_scale)
        self.prior_mode = "single" if arm == "single_prior" else str(getattr(cfg, "prior_mode", "single"))
        assert self.prior_mode in ("single", "split"), self.prior_mode
        if self.prior_mode == "split":            # iteration 5: learned per-component fusion scalars, init = alpha
            self.prior_w = nn.Parameter(torch.full((3,), float(prior_scale)))
        # whether COL_ELL includes x_t (train.py: text_term); when it does not (arms no_text_term / text_prior_off /
        # evidence_hmm) the split prior has no text term and ell_fine = COL_ELL - COL_V
        self.text_in_ell = bool(getattr(cfg, "text_in_ell", True))
        self.topk_div = int(cfg.topk_div)
        self.concat = arm == "avce"                    # candidate-1 backbone: evidence columns into the audio stream
        a_in = hc.SCAF_OFFSET + (N_EVID if self.concat else 0)
        self.fc_v = nn.Linear(hc.align.V_DIM, hid)
        self.fc_a = nn.Linear(a_in, hid)
        self.text_input = bool(getattr(cfg, "text_input", False)) and arm != "no_text_input"
        assert not (self.concat and self.text_input), "avce concatenates the four evidence columns only"
        self.enc = None if self.concat else EvidenceEncoder(hid, missing_state=(arm != "no_missing_state"),
                                                            text_input=self.text_input, cell=(arm != "no_cell"))
        bias_mode = "none" if self.concat else str(getattr(cfg, "bias_mode", "key"))   # key (rev 2) | gated (rev 3) | shared | none
        self.cma = EvidenceRoutedCMA(hid, nhead, ffn, dropout, bias_mode=bias_mode,
                                     qk_enc=(arm not in ("avce", "no_qk_enc")))
        self.ctx_mode = "none" if self.concat else str(getattr(cfg, "ctx_mode", "rep"))   # rep (rev 2) | logit (rev 3) | none
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
        evid[..., hc.COL_ELL] = torch.clamp(evid[..., hc.COL_ELL] / hc.ELL_SCALE, -1.0, 1.0)   # encoder input in [-1, 1]; the prior term uses the raw ell
        ell = f_a[..., hc.SCAF_OFFSET + hc.COL_ELL:hc.SCAF_OFFSET + hc.COL_ELL + 1]
        x_text = f_a[..., hc.SCAF_OFFSET + hc.COL_TEXT:hc.SCAF_OFFSET + hc.COL_TEXT + 1]
        v_video = f_a[..., hc.SCAF_OFFSET + hc.COL_V:hc.SCAF_OFFSET + hc.COL_V + 1]
        text_llr = torch.clamp(f_a[..., hc.SCAF_OFFSET + hc.COL_TEXT] / hc.LLR_SCALE, -1.0, 1.0)
        if self.no_verdict:
            evid = torch.zeros_like(evid)
            text_llr = torch.zeros_like(text_llr)
        content_a = f_a[..., :hc.SCAF_OFFSET]
        h_v = self.fc_v(f_v)
        if self.concat:
            h_a = self.fc_a(torch.cat([content_a, evid], dim=-1))
            e = None
        else:
            h_a = self.fc_a(content_a)
            e = self.enc(evid, text_llr) * mask[..., None].float()
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
        if self.no_verdict and not self.no_prior:
            # arm no_verdict: every VLM-derived column is zeroed above; the prior keeps only the
            # per-second text term (the text classifier is not a VLM verdict)
            if self.prior_mode == "split" and self.text_in_ell:
                av_log = av_log + self.prior_w[1] * x_text / hc.ELL_SCALE
        elif not self.no_prior:
            if self.prior_mode == "split":
                if self.text_in_ell:
                    ell_fine = ell - x_text - v_video
                    prior = (self.prior_w[0] * ell_fine + self.prior_w[1] * x_text + self.prior_w[2] * v_video) / hc.ELL_SCALE
                else:
                    ell_fine = ell - v_video
                    prior = (self.prior_w[0] * ell_fine + self.prior_w[2] * v_video) / hc.ELL_SCALE
            else:
                prior = self.prior_scale * ell / hc.ELL_SCALE
            av_log = av_log + prior
        mmil = self.bag(av_log, seq_len)
        return mmil, torch.sigmoid(a_log), torch.sigmoid(v_log), av_log, v_out, a_out
