#!/usr/bin/env python
"""TERA Gate-0 — exact arm instantiations A0-A4 and B0-B5 (appendix sec 4, 6).

Every head emits one logit; loss is `BCEWithLogitsLoss()` with no pos_weight.
Only the final scalar output layer uses `normal_(0, 0.01)/zeros_`; hidden layers
use PyTorch's default `nn.Linear` init (appendix sec 4).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .common import (A2_K_GRID, A4_TAU_GRID, K_WINDOWS, LR_GRID, LR_TAG, TAU_TAG,
                     TeraHalt, WD_GRID, WD_TAG)

R_PROJ = 128          # Gate-B projection width (appendix sec 6.3)
H_MLP = 64            # Gate-B trunk hidden width
H_ATT = 128           # A3 attention hidden width
PHI_DIM = 6

A_ARMS = ("A0", "A1", "A2", "A3", "A4")
B_ARMS = ("B0", "B1", "B2", "B3", "B4", "B5")


def _init_output(linear: nn.Linear):
    nn.init.normal_(linear.weight, 0.0, 0.01)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


# --------------------------------------------------------------- Gate-A arms --
class A0(nn.Module):
    """Whole-video/global mean representation + linear head."""

    takes = "whole"

    def __init__(self, d):
        super().__init__()
        self.head = nn.Linear(d, 1)
        _init_output(self.head)

    def forward(self, x_whole):
        return self.head(x_whole).squeeze(-1)


class _SegmentHeadArm(nn.Module):
    takes = "seg"

    def segment_logits(self, x_seg):
        """Registered per-segment score operand (appendix sec 3.1)."""
        return self.head(x_seg).squeeze(-1)


class A1(_SegmentHeadArm):
    """Mean of the K=30 segment representations + same head form."""

    def __init__(self, d):
        super().__init__()
        self.head = nn.Linear(d, 1)
        _init_output(self.head)

    def forward(self, x_seg):
        return self.head(x_seg.mean(dim=1)).squeeze(-1)


class A2(_SegmentHeadArm):
    """max / top-k MIL on SCALAR SEGMENT LOGITS; ties -> smaller window index."""

    def __init__(self, d, k):
        super().__init__()
        self.k = int(k)
        self.head = nn.Linear(d, 1)
        _init_output(self.head)

    def forward(self, x_seg):
        z = self.segment_logits(x_seg)
        vals, _ = torch.sort(z, dim=1, descending=True, stable=True)
        return vals[:, :self.k].mean(dim=1)


class A3(_SegmentHeadArm):
    """Non-gated single-hidden-layer tanh attention pooling, H = 128."""

    def __init__(self, d, hidden=H_ATT):
        super().__init__()
        self.att = nn.Linear(d, hidden)
        self.u = nn.Linear(hidden, 1, bias=False)
        self.head = nn.Linear(d, 1)
        _init_output(self.head)

    def attention(self, x_seg):
        a = self.u(torch.tanh(self.att(x_seg))).squeeze(-1)
        return torch.softmax(a, dim=1)

    def forward(self, x_seg):
        alpha = self.attention(x_seg)
        pooled = (alpha.unsqueeze(-1) * x_seg).sum(dim=1)
        return self.head(pooled).squeeze(-1)


class A4(_SegmentHeadArm):
    """Log-sum-exp pooling on the same scalar segment logits as A2."""

    def __init__(self, d, tau):
        super().__init__()
        self.tau = float(tau)
        self.head = nn.Linear(d, 1)
        _init_output(self.head)

    def forward(self, x_seg):
        z = self.segment_logits(x_seg)
        return self.tau * (torch.logsumexp(z / self.tau, dim=1) - math.log(z.shape[1]))


# --------------------------------------------------------------- Gate-B arms --
class _BTrunk(nn.Module):
    def __init__(self, d, in_dim, hidden, r=R_PROJ):
        super().__init__()
        self.P = nn.Linear(d, r)
        self.g1 = nn.Linear(in_dim, hidden)
        self.g2 = nn.Linear(hidden, 1)
        _init_output(self.g2)

    def mlp(self, x):
        return self.g2(torch.relu(self.g1(x))).squeeze(-1)


class B0(_BTrunk):
    """Strongest single selected segment."""

    takes = "top"

    def __init__(self, d, r=R_PROJ, hidden=H_MLP):
        super().__init__(d, r, hidden, r)

    def forward(self, e_top):
        return self.mlp(self.P(e_top))


class B1(_BTrunk):
    """Non-interactive mean of the top-two segment representations."""

    takes = "pair"

    def __init__(self, d, r=R_PROJ, hidden=H_MLP):
        super().__init__(d, r, hidden, r)

    def forward(self, e_first, e_second, phi=None):
        return self.mlp((self.P(e_first) + self.P(e_second)) / 2.0)


class B2(_BTrunk):
    """Ordered pair + interaction term + relative-time encoding."""

    takes = "pair_phi"

    def __init__(self, d, r=R_PROJ, hidden=H_MLP):
        super().__init__(d, 3 * r + PHI_DIM, hidden, r)

    def forward(self, e_first, e_second, phi):
        p = self.P(e_first)
        q = self.P(e_second)
        return self.mlp(torch.cat([p, q, p * q, phi], dim=-1))


class B3(_BTrunk):
    """Width-matched single-segment capacity control."""

    takes = "top"

    def __init__(self, d, hidden, r=R_PROJ):
        super().__init__(d, r, hidden, r)

    def forward(self, e_top):
        return self.mlp(self.P(e_top))


# ------------------------------------------------------- capacity arithmetic --
def params_projection(d, r=R_PROJ):
    return d * r + r


def params_b2(d, r=R_PROJ, h=H_MLP):
    return params_projection(d, r) + ((3 * r + PHI_DIM) * h + h) + (h + 1)


def params_b3(d, h3, r=R_PROJ):
    return params_projection(d, r) + (r * h3 + h3) + (h3 + 1)


def solve_h3(d, r=R_PROJ, h=H_MLP):
    """H3* = argmin_{H3 >= 1} |params(B3) - params(B2)|; ties -> smaller H3."""
    target = params_b2(d, r, h)
    base = params_projection(d, r) + 1
    step = r + 2
    approx = max(1, int((target - base) // step))
    best = None
    for cand in range(max(1, approx - 3), approx + 4):
        diff = abs(params_b3(d, cand, r) - target)
        if best is None or diff < best[0]:
            best = (diff, cand)
    h3 = best[1]
    rel = abs(params_b3(d, h3, r) - target) / target
    if rel > 0.05:
        raise TeraHalt("HALT_CAPACITY_MATCH", "relative difference %.4g > 0.05" % rel)
    return h3, rel


def head_capacity_check(d, r=R_PROJ, h=H_MLP):
    """Reproduces the registered head-capacity numbers (appendix sec 6.5)."""
    h3, _ = solve_h3(d, r, h)
    head_b2 = (3 * r + PHI_DIM) * h + h                 # 390 -> 64 layer, with bias
    head_b3 = params_b3(d, h3, r) - params_projection(d, r) - 1
    rel = abs(head_b3 - head_b2) / head_b2
    return {"h3": h3, "head_b2": head_b2, "head_b3": head_b3, "relative": rel}


# ----------------------------------------------------------------- configs ---
def config_list(arm):
    """Registered fixed enumeration order: arm_local -> lr -> weight_decay."""
    if arm == "A2":
        local = [("k", k) for k in A2_K_GRID]
    elif arm == "A4":
        local = [("tau", t) for t in A4_TAU_GRID]
    else:
        local = [(None, None)]
    out = []
    for name, value in local:
        for lr in LR_GRID:
            for wd in WD_GRID:
                cfg = {"lr": lr, "weight_decay": wd}
                if name == "k":
                    cfg["k"] = value
                    cid = "cfg_k%d_lr%s_wd%s" % (value, LR_TAG[lr], WD_TAG[wd])
                elif name == "tau":
                    cfg["tau"] = value
                    cid = "cfg_tau%s_lr%s_wd%s" % (TAU_TAG[value], LR_TAG[lr], WD_TAG[wd])
                else:
                    cid = "cfg_lr%s_wd%s" % (LR_TAG[lr], WD_TAG[wd])
                cfg["config_id"] = cid
                out.append(cfg)
    return out


def make_model(arm, cfg, d, seed, h3=None):
    torch.manual_seed(int(seed))
    if arm == "A0":
        return A0(d)
    if arm == "A1":
        return A1(d)
    if arm == "A2":
        return A2(d, cfg["k"])
    if arm == "A3":
        return A3(d)
    if arm == "A4":
        return A4(d, cfg["tau"])
    if arm == "B0":
        return B0(d)
    if arm == "B1":
        return B1(d)
    if arm in ("B2", "B4", "B5"):
        return B2(d)
    if arm == "B3":
        if h3 is None:
            h3, _ = solve_h3(d)
        return B3(d, h3)
    raise TeraHalt("HALT_UNKNOWN_ARM", arm)


def param_count(model):
    return int(sum(p.numel() for p in model.parameters()))


def relative_time_encoding(i_a, i_b, k=K_WINDOWS):
    """phi = [tA, tB, delta, |delta|, sin(pi delta), cos(pi delta)] (sec 6.4)."""
    ta = i_a / float(k - 1)
    tb = i_b / float(k - 1)
    delta = tb - ta
    return [ta, tb, delta, abs(delta), math.sin(math.pi * delta), math.cos(math.pi * delta)]


def select_pair(seg_scores, k=K_WINDOWS, min_sep=2):
    """Top-two selection with minimum separation 2 (appendix sec 6.2)."""
    scores = list(seg_scores)
    i1 = max(range(k), key=lambda idx: (scores[idx], -idx))
    cand = [idx for idx in range(k) if abs(idx - i1) >= min_sep]
    if not cand:
        raise TeraHalt("HALT_PAIR_CANDIDATES", "empty candidate set at K=%d" % k)
    i2 = max(cand, key=lambda idx: (scores[idx], -idx))
    a, b = min(i1, i2), max(i1, i2)
    return i1, (a, b)
