"""Prior weights (revision 7; experiments/20261110_prior_weight/README.md).

The prior network is fitted on the training videos; on videos it has not seen its video logit g and its per-second
logits s_t / node potentials phi_n are over-confident (experiments/20261110_answer_likelihood/README.md section 2:
the calibrated weight of g on test is .24-.29), so the tree inference under-weights the answers. Three scalars
re-weight the prior inside the model:

    P(G = 1 | x) = sigmoid(a g + b),      per-second logits c s_t,      node potentials c phi_n.

They are estimated WITHOUT labels on held-out (validation) videos by maximum marginal likelihood of the videos' cached
answers under the model,

    P(o | x) = P(G = 1 | x) P(o | G = 1, x) + P(G = 0 | x) P(o | G = 0),

the answer model and the chain fixed:
  (a, b)  from the root's answer alone (one answer per video; log P(o_root | G) is the answer table's row, no tree);
  c       from all cached answers of the video with (a, b) fixed (ctree.log_evidence under c s, c phi).
mode "joint" estimates the three from all cached answers at once; "root" leaves c = 1; "loc" estimates c alone from
all cached answers with (a, b) = (1, 0).
No video label and no per-second label enters. The weights then apply wherever the prior is used at validation and
test: the posterior, the choice of questions (EIG) and the stopping statistics (cpolicy.run_batch prior_weight).

Search for c (modes "loc" and "root_all"). The likelihood is not concave in c, and it is flat where the weighted
prior is saturated: a network whose per-second logits are all large (e.g. mean 5) puts the whole prior mass on one
configuration for every c above about .2, the gradient at c = 1 is 0 to machine precision and L-BFGS started there
does not move, although c near 0 has a much higher likelihood (HateClipSeg seed 3407, revision-6 trials 8, 9, 14,
17: -198.05 per video for every c >= .2 against -189.4 at c = .003). The search therefore starts from the point of
C_GRID with the highest likelihood (ties: the point closest to 1) and refines it by L-BFGS; start="one" is the
search from c = 1 used before this correction (2026-11-10), kept for comparison only.

Answer trust (modes "trust" / "trust_len"). The anchored answer table knows two states, fitted where the video label
fixes the state: "no hate" on every node of the negative training videos, "hate" on the roots of the positive ones.
Inside a hateful video it is over-trusted (experiments/20261109_tree_within/README.md section 11): clean nodes of
hateful videos answer higher than nodes of negative videos, and short hateful nodes answer lower than the roots. Two
more label-free scalars describe this, estimated together with c by the same marginal likelihood:
    leak rho:  a clean node of a hateful video answers like a hateful node with probability rho,
               P(o | clean, G = 1) = (1 - rho) P(o | no hate) + rho P(o | hate);
    miss eta:  a hateful node answers like a clean node of a hateful video with probability eta_n,
               P(o | hateful) = (1 - eta_n) P(o | hate) + eta_n P(o | clean, G = 1),
               eta_n = sigmoid(logit(eta) + k u_n), u_n the node's standardised log length (k = 0 in mode "trust").
TrustTable is the resulting three-state table (the answer model of the policy and the posterior at validation / test).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import ctree
from . import data as qdata
from . import policy

IDENTITY = (1.0, 0.0, 1.0)
C_GRID = (0.0, 0.003, 0.01, 0.03, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)      # starting points of the search for c


@torch.no_grad()
def collect(model, store, vids, device):
    """{video: (s (T,), g, phi or None)}: the prior network's outputs (five-crop means, policy.video_prior_nodes)."""
    was_training = model.training
    model.eval()
    out = {v: policy.video_prior_nodes(model, store, v, device) for v in vids}
    if was_training:
        model.train()
    return out


def _chunks(pri, store, vids, am, answers, device, chunk):
    """Per chunk of videos (sorted by length): forest, S (B, Tmax), G (B,), PHI (N,) or None, A3 (N, 3) the
    log-likelihoods of all cached answers, the root rows ll_root (B, 3) and has_root (B,) bool. Double precision."""
    vids = sorted(vids, key=lambda v: store.T[v])
    out = []
    for i in range(0, len(vids), chunk):
        vs = vids[i:i + chunk]
        Ts = [store.T[v] for v in vs]
        Tm = max(Ts)
        fo = ctree.Forest(Ts, Tm)
        S = torch.zeros(len(vs), Tm, dtype=torch.float64)
        G = torch.zeros(len(vs), dtype=torch.float64)
        PHI = None
        A3 = torch.zeros(fo.N, 3, dtype=torch.float64)
        U = torch.zeros(fo.N, dtype=torch.float64)       # standardised log length of the answered nodes
        ll_root = torch.zeros(len(vs), 3, dtype=torch.float64)
        has_root = torch.zeros(len(vs), dtype=torch.bool)
        for b, v in enumerate(vs):
            s, g, phi = pri[v]
            S[b, :len(s)] = torch.as_tensor(s, dtype=torch.float64)
            G[b] = float(g)
            if phi is not None:
                if PHI is None:
                    PHI = torch.zeros(fo.N, dtype=torch.float64)
                PHI[int(fo.offs[b]):int(fo.offs[b]) + len(phi)] = torch.as_tensor(phi, dtype=torch.float64)
            nid, ans, lens = qdata.observed(answers[v], Ts[b])
            if len(nid):
                with torch.no_grad():
                    ll = am.loglik(torch.as_tensor(ans), torch.as_tensor(lens)).double().cpu()
                A3[torch.as_tensor(nid + int(fo.offs[b]))] = ll
                U[torch.as_tensor(nid + int(fo.offs[b]))] = torch.as_tensor((np.log(lens) - am.len_mu) / am.len_sd)
                if nid[0] == 0:                  # node 0 is the root (observed() sorts by node id)
                    ll_root[b] = ll[0]
                    has_root[b] = True
        out.append({"fo": fo, "S": S.to(device), "G": G.to(device), "PHI": None if PHI is None else PHI.to(device),
                    "A3": A3.to(device), "U": U.to(device), "ll_root": ll_root.to(device),
                    "has_root": has_root.to(device)})
    return out


def _trust_rows(A3, U, trust):
    """Answer log-likelihoods (N, 3) under the leak / miss scalars; trust = (rho, logit eta, k) tensors or None.
    A3 holds the two-state table's rows (columns 0 and 1 = "no hate", 2 = "hate"); rows of unanswered nodes are 0
    and stay 0."""
    if trust is None:
        return A3
    rho, h, k = trust
    ll0, ll2 = A3[:, 0], A3[:, 2]
    ll1 = torch.logaddexp(torch.log1p(-rho) + ll0, torch.log(rho) + ll2)
    eta = torch.sigmoid(h + k * U)
    return torch.stack([ll0, ll1, torch.logaddexp(torch.log1p(-eta) + ll2, torch.log(eta) + ll1)], dim=1)


def _nll_root(chunks, a, b):
    """Negative log marginal likelihood of the root answers (sum over videos with a root answer) and their number."""
    tot, n = 0.0, 0
    for c in chunks:
        gp = (a * c["G"] + b)[c["has_root"]]
        ll = c["ll_root"][c["has_root"]]
        tot = tot - torch.logaddexp(F.logsigmoid(gp) + ll[:, 2], F.logsigmoid(-gp) + ll[:, 0]).sum()
        n += int(c["has_root"].sum())
    return tot, n


def _nll_all(chunks, chain_d, a, b, c_loc, trust=None):
    """Negative log marginal likelihood of all cached answers (sum over videos)."""
    tot = 0.0
    for c in chunks:
        phi = None if c["PHI"] is None else c_loc * c["PHI"]
        l1, l0, lo1, lo0 = ctree.log_evidence(c["fo"], c_loc * c["S"], a * c["G"] + b,
                                              _trust_rows(c["A3"], c["U"], trust), chain_d, phi)
        tot = tot - torch.logaddexp(l1 + lo1, l0 + lo0).sum()
    return tot


def _lbfgs(params, closure_value, max_iter=50):
    opt = torch.optim.LBFGS(params, max_iter=max_iter, tolerance_grad=1e-8, tolerance_change=1e-10,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        nll = closure_value()
        nll.backward()
        return nll

    opt.step(closure)


def _fit_c(c, nll_of_c, start):
    """Maximum-likelihood c: L-BFGS from the best point of C_GRID (start "grid") or from c = 1 (start "one").
    Returns the starting point."""
    c0 = 1.0
    if start == "grid":
        with torch.no_grad():
            vals = [float(nll_of_c(torch.full_like(c, g))) for g in C_GRID]
        vals = [v if np.isfinite(v) else float("inf") for v in vals]
        c0 = min(zip(vals, [abs(g - 1.0) for g in C_GRID], C_GRID))[2]
        with torch.no_grad():
            c.fill_(c0)
    _lbfgs([c], lambda: nll_of_c(c))
    return c0


def fit(pri, store, vids, am, chain, answers, device, mode="root_all", chunk=64, start="grid"):
    """Label-free prior weights from the cached answers of `vids` (module docstring). pri: collect(...).
    mode: "root_all", "root" (c = 1), "joint" (a, b, c from all answers at once), "loc" (c only, a = 1, b = 0),
    "trust" (c, leak rho, miss eta from all answers; a = 1, b = 0), "trust_len" (trust with the miss length slope k).
    start: where the search for c begins in the modes "loc" and "root_all" ("grid" or "one", module docstring).
    Returns {"a", "b", "c", "n_videos", "n_root", "nll_root_before / after" (per video with a root answer),
    "nll_all_before / after" (per video; before = at a = 1, b = 0, c = 1)} and, in the trust modes, "rho", "eta",
    "eta_slope"; in the modes "loc" and "root_all" also "c_start"."""
    assert mode in ("root_all", "root", "joint", "loc", "trust", "trust_len") and start in ("grid", "one")
    assert not mode.startswith("trust") or (getattr(am, "n_state", 3) == 2 and len(am.cats) == 1), \
        "the trust scalars extend the two-state table of one answer category"
    chunks = _chunks(pri, store, vids, am, answers, device, chunk)
    chain_d = ctree._DoubleChain(chain)
    a = torch.ones((), dtype=torch.float64, device=device, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    c = torch.ones((), dtype=torch.float64, device=device, requires_grad=True)
    with torch.no_grad():
        r0, n_root = _nll_root(chunks, a, b)
        a0 = _nll_all(chunks, chain_d, a, b, c)
    trust, c_start = None, None
    if mode.startswith("trust"):
        r = torch.full((), -3.0, dtype=torch.float64, device=device, requires_grad=True)     # rho = sigmoid(r)
        h = torch.full((), -3.0, dtype=torch.float64, device=device, requires_grad=True)     # eta at u = 0
        k = torch.zeros((), dtype=torch.float64, device=device, requires_grad=mode == "trust_len")
        a_, b_ = a.detach(), b.detach()
        _lbfgs([c, r, h] + ([k] if mode == "trust_len" else []),
               lambda: _nll_all(chunks, chain_d, a_, b_, c, (torch.sigmoid(r), h, k)), max_iter=100)
        trust = (torch.sigmoid(r).detach(), h.detach(), k.detach())
    elif mode == "joint":
        _lbfgs([a, b, c], lambda: _nll_all(chunks, chain_d, a, b, c))
    elif mode == "loc":
        a_, b_ = a.detach(), b.detach()
        c_start = _fit_c(c, lambda cc: _nll_all(chunks, chain_d, a_, b_, cc), start)
    else:
        _lbfgs([a, b], lambda: _nll_root(chunks, a, b)[0])
        if mode == "root_all":
            a_, b_ = a.detach(), b.detach()
            c_start = _fit_c(c, lambda cc: _nll_all(chunks, chain_d, a_, b_, cc), start)
    with torch.no_grad():
        r1, _ = _nll_root(chunks, a, b)
        a1 = _nll_all(chunks, chain_d, a, b, c, trust)
    n = len(vids)
    out = {"a": float(a), "b": float(b), "c": float(c), "n_videos": n, "n_root": n_root, "mode": mode,
           "nll_root_before": float(r0) / max(n_root, 1), "nll_root_after": float(r1) / max(n_root, 1),
           "nll_all_before": float(a0) / max(n, 1), "nll_all_after": float(a1) / max(n, 1)}
    if trust is not None:
        out.update(rho=float(trust[0]), eta=float(torch.sigmoid(trust[1])), eta_slope=float(trust[2]))
    if c_start is not None:
        out["c_start"] = float(c_start)
    return out


def as_tuple(w):
    return IDENTITY if w is None else (float(w["a"]), float(w["b"]), float(w["c"]))


class TrustTable(torch.nn.Module):
    """The anchored two-state answer table with the leak / miss scalars (module docstring) as a three-state answer
    model: states 0 = no hate in the video, 1 = clean node of a hateful video, 2 = hateful node. Same interface as
    qtree.AnswerModel where the policy and the posterior use it (level_logp, loglik, cats, n_state)."""

    def __init__(self, base, rho, eta, eta_slope=0.0):
        super().__init__()
        assert base.n_state == 2 and len(base.cats) == 1
        self.base, self.cats, self.n_state = base, list(base.cats), 3
        self.len_mu, self.len_sd = base.len_mu, base.len_sd
        self.rho = float(min(max(rho, 1e-9), 1 - 1e-9))
        self.eta = float(min(max(eta, 1e-9), 1 - 1e-9))
        self.eta_slope = float(eta_slope)

    def level_logp(self, length):
        lp = self.base.level_logp(length)                                 # N, C, 2, L
        lp0, lp2 = lp[:, :, 0], lp[:, :, 1]
        lp1 = torch.logaddexp(np.log1p(-self.rho) + lp0, np.log(self.rho) + lp2)
        u = self.base.u(length).to(lp.device)
        eta = torch.sigmoid(np.log(self.eta) - np.log1p(-self.eta) + self.eta_slope * u)[:, None, None]
        lp2 = torch.logaddexp(torch.log1p(-eta) + lp2, torch.log(eta) + lp1)
        return torch.stack([lp0, lp1, lp2], dim=2)                        # N, C, 3, L

    def loglik(self, answers, length):
        lp = self.level_logp(length)
        idx = answers.long().to(lp.device)[:, :, None, None].expand(-1, -1, 3, 1)
        return torch.gather(lp, 3, idx).squeeze(-1)[:, self.cats].sum(1)  # N, 3


def table(am, w):
    """The answer model to use with the weights w: the TrustTable when w carries the trust scalars, else am."""
    if w is None or "rho" not in w:
        return am
    return TrustTable(am, w["rho"], w["eta"], w.get("eta_slope", 0.0))


def label_reference(pri, vids, labels):
    """Diagnostic only (not the method): the labelled logistic regression of the video label on g, (slope, intercept).
    Shows how far the label-free (a, b) is from what the labels of the same videos would give."""
    from sklearn.linear_model import LogisticRegression
    g = np.array([[float(pri[v][1])] for v in vids])
    y = np.array([int(labels[v]) for v in vids])
    if y.min() == y.max():
        return None
    lr = LogisticRegression(C=1e6, max_iter=2000).fit(g, y)
    return {"a": float(lr.coef_[0, 0]), "b": float(lr.intercept_[0])}
