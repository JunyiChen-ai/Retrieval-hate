"""Evidence-chain network (README section 1; revision 2 after the seed-234 screening, 2026-09-04).

Revision-2 changes (README section 5): (i) the encoder also reads the verdict
context columns [bf, bc, bfp, bfn, phi_f*n_w/3, phi_c/3] (the previous
candidate showed the network needs the raw verdict sequence: revision 2 of
hier_evidence_mil lost .036 without it); (ii) the density head is trained with
BCE on the video label (its logit is returned as `d_logit`); (iii) reliability
gates act only on fired verdicts (b = 1), so a gate can never delete the
VLM's negative evidence.

P1 content encoder (visual / audio-text projections, one temporal self-attention
   layer) + content evidence u_t
P2 video-level evidence-profile encoder -> density d_v (chain prior); reads the
   verdict profile only (no content), bounded to [D_LO, D_HI]
P3 granularity reliability gates gf (per window), gc (per block); read the
   verdict context (own / neighbour verdicts, block verdict) and d_v only
P4 differentiable evidence chain head (src/evidence_chain) -> log-odds = score

Switching rate: the EM value a is estimated on the K=30 window grid; per video
it is converted to the chain's step grid, a_step = 1 - (1 - a)^(K / L) with L
the number of valid steps, so the expected number of switches is preserved.

Model-level ablations (README section 2): full, no_density, density_bias,
density_content, no_gate, gate_with_content, no_content, no_vlm, indep,
flat_coarse, topk_head, macilsd_encoder, no_text.  Loss-level ablations
(train.py): no_block_mil, no_contrast, contrast_self_topk, contrast_vlm_thresh.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

import evidence_chain as ec                    # noqa: E402
import dataset as ds                           # noqa: E402

MODEL_ABLATIONS = ("full", "no_density", "density_bias", "density_content", "no_gate",
                   "gate_with_content", "no_content", "no_vlm", "indep", "flat_coarse",
                   "topk_head", "macilsd_encoder", "no_text", "no_vctx")
LOSS_ABLATIONS = ("no_block_mil", "no_contrast", "contrast_self_topk", "contrast_vlm_thresh",
                  "no_density_loss")
ABLATIONS = MODEL_ABLATIONS + LOSS_ABLATIONS
N_VCTX = 6                    # verdict-context columns fed to the encoder
GATE_INIT_BIAS = 3.0          # sigmoid(3) = .953: gates start open (README constants table)
D_LO, D_HI = 0.01, 0.99       # density bounds
EPS = 1e-6


def masked_mean(x, mask):
    m = mask.float().unsqueeze(-1)
    return (x * m).sum(1) / m.sum(1).clamp(min=1.0)


def group_mean(x, idx, n_groups, mask):
    """Mean of x [B,T,d] over rows sharing idx [B,T] (values in [0, n_groups]),
    returned per row [B,T,d]; padded rows carry idx == n_groups and mask False."""
    B, T, d = x.shape
    flat_idx = (idx + torch.arange(B, device=x.device)[:, None] * (n_groups + 1)).reshape(-1)
    xm = (x * mask.float().unsqueeze(-1)).reshape(B * T, d)
    sums = torch.zeros(B * (n_groups + 1), d, device=x.device, dtype=x.dtype)
    cnt = torch.zeros(B * (n_groups + 1), 1, device=x.device, dtype=x.dtype)
    sums.index_add_(0, flat_idx, xm)
    cnt.index_add_(0, flat_idx, mask.float().reshape(-1, 1))
    return (sums / cnt.clamp(min=1.0))[flat_idx].reshape(B, T, d)


class MLP(nn.Sequential):
    def __init__(self, n_in, n_hidden, n_out, dropout, out_bias=0.0):
        super().__init__(nn.Linear(n_in, n_hidden), nn.ReLU(), nn.Dropout(dropout),
                         nn.Linear(n_hidden, n_out))
        nn.init.constant_(self[-1].bias, out_bias)


class EvidenceChainNet(nn.Module):
    def __init__(self, cfg, pot, ablation="full"):
        super().__init__()
        assert ablation in MODEL_ABLATIONS, ablation
        self.ablation = ablation
        d = int(cfg.hid_dim)
        p = float(cfg.dropout)
        self.d = d
        self.a_window = float(pot.a)
        assert 0.0 < self.a_window <= 1.0
        self.p0_hate = float(pot.p0_hate)
        self.topk_div = int(cfg.topk_div)
        # ---- P1 encoder
        if ablation == "macilsd_encoder":
            from macilsd.avce_network import AVCE_Model
            self.av = AVCE_Model(cfg)
            self.fuse = nn.Linear(2 * d, d)
        else:
            self.proj_v = nn.Linear(ds.V_DIM, d)
            self.proj_at = nn.Linear(ds.F_A_DIM, d)
            self.proj_ctx = nn.Linear(N_VCTX, d)
            self.fuse = nn.Linear(3 * d, d)
            self.enc = nn.TransformerEncoderLayer(d, nhead=int(cfg.nhead),
                                                  dim_feedforward=int(cfg.ffn_dim),
                                                  dropout=p, batch_first=True)
        self.drop = nn.Dropout(p)
        self.u_head = MLP(d, d, 1, p)
        # ---- P2 density (verdict profile only; + pooled content in the density_content arm)
        n_dens_in = ds.PROFILE_DIM + (d if ablation == "density_content" else 0)
        self.dens_head = MLP(n_dens_in, d, 1, p)
        # ---- P3 gates (verdict context + d_v; + content in the gate_with_content arm)
        n_ctx = d if ablation == "gate_with_content" else 0
        self.gate_f = MLP(4 + 1 + n_ctx, d, 1, p, out_bias=GATE_INIT_BIAS)
        self.gate_c = MLP(1 + 1 + n_ctx, d, 1, p, out_bias=GATE_INIT_BIAS)

    # ------------------------------------------------------------------ parts
    def encode(self, f_v, f_a, mask, vctx):
        """returns fused h [B,T,d] and the two modality embeddings (for the contrast)."""
        if self.ablation == "no_text":
            f_a = torch.cat([f_a[..., :ds.A_DIM], torch.zeros_like(f_a[..., ds.A_DIM:])], -1)
        if self.ablation == "macilsd_encoder":
            _, _, _, _, v_out, a_out = self.av(f_a, f_v, None)
            h = self.drop(self.fuse(torch.cat([v_out, a_out], -1)))
            return h, v_out, a_out
        hv = self.proj_v(f_v)
        hat = self.proj_at(f_a)
        hc = self.proj_ctx(vctx)
        h = self.fuse(torch.cat([hv, hat, hc], -1))
        h = self.enc(h, src_key_padding_mask=~mask)
        return self.drop(h), hv, hat

    def step_switching_rate(self, mask):
        L = mask.float().sum(1).clamp(min=1.0)
        return 1.0 - (1.0 - self.a_window) ** (ds.K / L)

    def forward(self, batch):
        f_v, f_a, mask = batch["f_v"], batch["f_a"], batch["mask"]
        w, j = batch["w"], batch["j"]
        B, T = mask.shape
        abl = self.ablation
        vctx = torch.stack([batch["bf"], batch["bc"], batch["bfp"], batch["bfn"],
                            batch["phi_f"] * batch["n_w"] / 3.0, batch["phi_c"] / 3.0], -1)
        vctx = vctx * mask.unsqueeze(-1).float()
        if abl == "no_vctx":
            vctx = torch.zeros_like(vctx)
        h, hv, hat = self.encode(f_v, f_a, mask, vctx)
        # P1 content evidence
        u = self.u_head(h).squeeze(-1)
        if abl == "no_content":
            u = torch.zeros_like(u)
        # P2 density
        dens_in = batch["profile"]
        if abl == "density_content":
            dens_in = torch.cat([dens_in, masked_mean(h, mask)], -1)
        d_logit = self.dens_head(dens_in).squeeze(-1)
        d_v = D_LO + (D_HI - D_LO) * torch.sigmoid(d_logit)
        d_chain = d_v
        if abl in ("no_density", "density_bias"):
            d_chain = torch.full_like(d_v, self.p0_hate)
        if abl == "density_bias":
            u = u + torch.logit(d_v)[:, None]
        # P3 gates: per-row logits from verdict context, averaged within window / block
        vcols = torch.stack([batch["bf"], batch["bc"], batch["bfp"], batch["bfn"]], -1)
        dv_rows = d_v.detach()[:, None, None].expand(B, T, 1)
        gin_f = [vcols, dv_rows]
        gin_c = [vcols[..., 1:2], dv_rows]
        if abl == "gate_with_content":
            gin_f.append(h)
            gin_c.append(h)
        gf = torch.sigmoid(group_mean(self.gate_f(torch.cat(gin_f, -1)), w, ds.K, mask)).squeeze(-1)
        gc = torch.sigmoid(group_mean(self.gate_c(torch.cat(gin_c, -1)), j, ds.J, mask)).squeeze(-1)
        if abl == "no_gate":
            gf, gc = torch.ones_like(gf), torch.ones_like(gc)
        # gates act on fired verdicts only: negative evidence (b = 0) is never scaled
        gf = torch.where(batch["bf"] > 0.5, gf, torch.ones_like(gf))
        gc = torch.where(batch["bc"] > 0.5, gc, torch.ones_like(gc))
        # VLM potentials
        phi_f, phi_c = batch["phi_f"], batch["phi_c"]
        if abl == "no_vlm":
            phi_f, phi_c = torch.zeros_like(phi_f), torch.zeros_like(phi_c)
        gf_chain, gc_chain = gf, gc
        if abl == "flat_coarse":
            phi_f = gf * phi_f + gc * phi_c / batch["n_j"].clamp(min=1.0)
            phi_c = torch.zeros_like(phi_c)
            gf_chain, gc_chain = torch.ones_like(gf), torch.ones_like(gc)
        a_step = torch.ones(B, device=u.device, dtype=u.dtype) if abl == "indep" \
            else self.step_switching_rate(mask)
        out = {"h": h, "hv": hv, "hat": hat, "u": u, "d_v": d_v, "d_logit": d_logit,
               "gf": gf, "gc": gc, "a_step": a_step}
        if abl == "topk_head":
            score = u + gf * phi_f * batch["n_w"] + gc * phi_c + torch.logit(d_v)[:, None]
            score = torch.where(mask, score, torch.full_like(score, -1e4))
            bags = []
            for i in range(B):
                t = int(mask[i].sum())
                k = max(1, int(math.ceil(t / self.topk_div)))
                bags.append(torch.topk(score[i, :t], k=k).values.mean().view(1))
            logit_video = torch.cat(bags)
            out["log_p_video"] = nn.functional.logsigmoid(logit_video)
            out["log_rho"] = nn.functional.logsigmoid(-logit_video)
            out["score"] = score
            out["post"] = torch.sigmoid(score)
            return out
        ch = ec.chain(u, phi_f, gf_chain, phi_c, gc_chain, d_chain, a_step, j, mask)
        out["log_p_video"] = ch["log_p_video"]
        out["log_rho"] = ch["log_rho"]
        out["post"] = ch["post_s1"]
        out["score"] = ch["logodds_s1"]
        out["log_Z"], out["log_Z0"] = ch["log_Z"], ch["log_Z0"]
        return out
