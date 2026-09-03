"""Evidence-chain network (README section 1).

P1 content encoder + content evidence u_t
P2 video-level evidence-profile encoder -> density d_v (chain prior)
P3 granularity reliability gates gf (per window), gc (per block)
P4 differentiable evidence chain head (src/evidence_chain) -> posterior = score

Ablations are switches on the same module (README section 2):
  full, no_density, density_bias, no_gate, gate_content_only, no_content,
  no_vlm, indep, flat_coarse, topk_head, macilsd_encoder
(the contrastive-loss ablations live in train.py).
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import evidence_chain as ec                    # noqa: E402
import dataset as ds                           # noqa: E402

ABLATIONS = ("full", "no_density", "density_bias", "no_gate", "gate_content_only",
             "no_content", "no_vlm", "indep", "flat_coarse", "topk_head",
             "macilsd_encoder",
             # train-time only (model = full)
             "no_contrast", "contrast_self_topk", "contrast_vlm_thresh")
GATE_INIT_BIAS = 4.0          # sigmoid(4) = .982: gates start (almost) open
EPS = 1e-6


def masked_mean(x, mask):
    """x [B,T,d], mask [B,T] -> [B,d]."""
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


def group_mean(x, idx, n_groups, mask):
    """Mean of x [B,T,d] over rows sharing idx [B,T] (values in [0, n_groups)),
    returned per row: [B,T,d] (each row gets its group's mean). Padded rows
    (mask False) carry idx == n_groups and are excluded."""
    B, T, d = x.shape
    flat_idx = (idx + torch.arange(B, device=x.device)[:, None] * (n_groups + 1)).reshape(-1)
    xm = (x * mask.float().unsqueeze(-1)).reshape(B * T, d)
    sums = torch.zeros(B * (n_groups + 1), d, device=x.device, dtype=x.dtype)
    cnt = torch.zeros(B * (n_groups + 1), 1, device=x.device, dtype=x.dtype)
    sums.index_add_(0, flat_idx, xm)
    cnt.index_add_(0, flat_idx, mask.float().reshape(-1, 1))
    means = sums / cnt.clamp(min=1.0)
    return means[flat_idx].reshape(B, T, d)


class MLP(nn.Sequential):
    def __init__(self, n_in, n_hidden, n_out, dropout, out_bias=0.0):
        super().__init__(nn.Linear(n_in, n_hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(n_hidden, n_out))
        nn.init.constant_(self[-1].bias, out_bias)


class EvidenceChainNet(nn.Module):
    def __init__(self, cfg, pot, ablation="full"):
        super().__init__()
        assert ablation in ABLATIONS, ablation
        self.ablation = ablation
        d = int(cfg.hid_dim)
        p = float(cfg.dropout)
        self.d = d
        self.a_switch = float(pot.a)
        self.p0_hate = float(pot.p0_hate)
        self.topk_div = int(cfg.topk_div)
        # ---- P1 encoder
        if ablation == "macilsd_encoder":
            from macilsd.avce_network import AVCE_Model
            self.av = AVCE_Model(cfg)
            self.fuse = nn.Linear(2 * d, d)
        else:
            self.proj_v = nn.Linear(ds.V_DIM, d)
            self.proj_a = nn.Linear(ds.A_DIM, d)
            self.proj_x = nn.Linear(ds.TEXT_DIM, d)
            self.fuse = nn.Linear(3 * d, d)
            self.enc = nn.TransformerEncoderLayer(d, nhead=int(cfg.nhead),
                                                  dim_feedforward=int(cfg.ffn_dim),
                                                  dropout=p, batch_first=True)
        self.drop = nn.Dropout(p)
        self.u_head = MLP(d, d, 1, p)
        # ---- P2 density
        self.dens_head = MLP(ds.PROFILE_DIM + d, d, 1, p)
        # ---- P3 gates
        self.gate_f = MLP(d + 4 + 1, d, 1, p, out_bias=GATE_INIT_BIAS)
        self.gate_c = MLP(d + 1 + 1, d, 1, p, out_bias=GATE_INIT_BIAS)

    # ------------------------------------------------------------------ parts
    def encode(self, f_v, f_a, mask):
        if self.ablation == "macilsd_encoder":
            _, _, _, _, v_out, a_out = self.av(f_a, f_v, None)
            h = self.fuse(torch.cat([v_out, a_out], -1))
            return self.drop(h)
        hv = self.proj_v(f_v)
        ha = self.proj_a(f_a[..., :ds.A_DIM])
        hx = self.proj_x(f_a[..., ds.A_DIM:ds.A_DIM + ds.TEXT_DIM])
        h = self.fuse(torch.cat([hv, ha, hx], -1))
        h = self.enc(h, src_key_padding_mask=~mask)
        return self.drop(h)

    def forward(self, batch):
        f_v, f_a, mask = batch["f_v"], batch["f_a"], batch["mask"]
        w, j = batch["w"], batch["j"]
        B, T = mask.shape
        abl = self.ablation
        h = self.encode(f_v, f_a, mask)                        # [B,T,d]
        # P1 content evidence
        u = self.u_head(h).squeeze(-1)
        if abl == "no_content":
            u = torch.zeros_like(u)
        # P2 density
        pooled = masked_mean(h, mask)
        d_v = torch.sigmoid(self.dens_head(torch.cat([batch["profile"], pooled], -1))).squeeze(-1)
        d_chain = d_v
        if abl in ("no_density", "density_bias"):
            d_chain = torch.full_like(d_v, self.p0_hate)
        if abl == "density_bias":
            u = u + torch.logit(d_v.clamp(EPS, 1 - EPS))[:, None]
        # P3 gates (window / block level: per-row logits averaged within the group)
        vcols = torch.stack([batch["bf"], batch["bc"], batch["bfp"], batch["bfn"]], -1)
        if abl == "gate_content_only":
            vcols = torch.zeros_like(vcols)
        dv_rows = d_v.detach()[:, None, None].expand(B, T, 1)
        gf_logit = self.gate_f(torch.cat([h, vcols, dv_rows], -1))          # [B,T,1]
        gc_logit = self.gate_c(torch.cat([h, vcols[..., 1:2], dv_rows], -1))
        gf = torch.sigmoid(group_mean(gf_logit, w, ds.K, mask)).squeeze(-1)
        gc = torch.sigmoid(group_mean(gc_logit, j, ds.J, mask)).squeeze(-1)
        if abl == "no_gate":
            gf, gc = torch.ones_like(gf), torch.ones_like(gc)
        # VLM potentials
        phi_f, phi_c = batch["phi_f"], batch["phi_c"]
        if abl == "no_vlm":
            phi_f, phi_c = torch.zeros_like(phi_f), torch.zeros_like(phi_c)
        gf_chain, gc_chain = gf, gc
        if abl == "flat_coarse":
            # coarse verdict spread over the rows of its block, on s (no OR variable);
            # gates folded into the potentials, chain gates set to 1
            phi_f = gf * phi_f + gc * phi_c / batch["n_j"].clamp(min=1.0)
            phi_c = torch.zeros_like(phi_c)
            gf_chain, gc_chain = torch.ones_like(gf), torch.ones_like(gc)
        a_eff = 1.0 if abl == "indep" else self.a_switch
        out = {"h": h, "u": u, "d_v": d_v, "gf": gf, "gc": gc}
        if abl == "topk_head":
            score = u + gf * phi_f * batch["n_w"] + gc * phi_c \
                + torch.logit(d_v.clamp(EPS, 1 - EPS))[:, None]
            score = torch.where(mask, score, torch.full_like(score, -1e4))
            bags = []
            for i in range(B):
                t = int(mask[i].sum())
                k = max(1, int(math.ceil(t / self.topk_div)))
                bags.append(torch.topk(score[i, :t], k=k).values.mean().view(1))
            out["p_video"] = torch.sigmoid(torch.cat(bags))
            out["score"] = score
            out["post"] = torch.sigmoid(score)
            return out
        ch = ec.chain(u, phi_f, gf_chain, phi_c, gc_chain, d_chain, a_eff, j, mask)
        post = ch["post_s1"].clamp(EPS, 1 - EPS)
        out["p_video"] = ch["p_video"].clamp(EPS, 1 - EPS)
        out["post"] = post
        out["score"] = torch.logit(post)
        out["log_Z"], out["log_Z0"] = ch["log_Z"], ch["log_Z0"]
        return out
