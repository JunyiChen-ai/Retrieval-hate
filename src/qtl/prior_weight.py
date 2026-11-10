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
mode "joint" (diagnostic) estimates the three from all cached answers at once; "root" leaves c = 1.
No video label and no per-second label enters. The weights then apply wherever the prior is used at validation and
test: the posterior, the choice of questions (EIG) and the stopping statistics (cpolicy.run_batch prior_weight).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import ctree
from . import data as qdata
from . import policy

IDENTITY = (1.0, 0.0, 1.0)


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
                if nid[0] == 0:                  # node 0 is the root (observed() sorts by node id)
                    ll_root[b] = ll[0]
                    has_root[b] = True
        out.append({"fo": fo, "S": S.to(device), "G": G.to(device), "PHI": None if PHI is None else PHI.to(device),
                    "A3": A3.to(device), "ll_root": ll_root.to(device), "has_root": has_root.to(device)})
    return out


def _nll_root(chunks, a, b):
    """Negative log marginal likelihood of the root answers (sum over videos with a root answer) and their number."""
    tot, n = 0.0, 0
    for c in chunks:
        gp = (a * c["G"] + b)[c["has_root"]]
        ll = c["ll_root"][c["has_root"]]
        tot = tot - torch.logaddexp(F.logsigmoid(gp) + ll[:, 2], F.logsigmoid(-gp) + ll[:, 0]).sum()
        n += int(c["has_root"].sum())
    return tot, n


def _nll_all(chunks, chain_d, a, b, c_loc):
    """Negative log marginal likelihood of all cached answers (sum over videos)."""
    tot = 0.0
    for c in chunks:
        phi = None if c["PHI"] is None else c_loc * c["PHI"]
        l1, l0, lo1, lo0 = ctree.log_evidence(c["fo"], c_loc * c["S"], a * c["G"] + b, c["A3"], chain_d, phi)
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


def fit(pri, store, vids, am, chain, answers, device, mode="root_all", chunk=64):
    """Label-free prior weights from the cached answers of `vids` (module docstring). pri: collect(...).
    mode: "root_all" (the method), "root" (c = 1), "joint" (diagnostic: a, b, c from all answers at once).
    Returns {"a", "b", "c", "n_videos", "n_root", "nll_root_before / after" (per video with a root answer),
    "nll_all_before / after" (per video)}."""
    assert mode in ("root_all", "root", "joint")
    chunks = _chunks(pri, store, vids, am, answers, device, chunk)
    chain_d = ctree._DoubleChain(chain)
    a = torch.ones((), dtype=torch.float64, device=device, requires_grad=True)
    b = torch.zeros((), dtype=torch.float64, device=device, requires_grad=True)
    c = torch.ones((), dtype=torch.float64, device=device, requires_grad=True)
    with torch.no_grad():
        r0, n_root = _nll_root(chunks, a, b)
        a0 = _nll_all(chunks, chain_d, a, b, c)
    if mode == "joint":
        _lbfgs([a, b, c], lambda: _nll_all(chunks, chain_d, a, b, c))
    else:
        _lbfgs([a, b], lambda: _nll_root(chunks, a, b)[0])
        if mode == "root_all":
            a_, b_ = a.detach(), b.detach()
            _lbfgs([c], lambda: _nll_all(chunks, chain_d, a_, b_, c))
    with torch.no_grad():
        r1, _ = _nll_root(chunks, a, b)
        a1 = _nll_all(chunks, chain_d, a, b, c)
    n = len(vids)
    return {"a": float(a), "b": float(b), "c": float(c), "n_videos": n, "n_root": n_root, "mode": mode,
            "nll_root_before": float(r0) / max(n_root, 1), "nll_root_after": float(r1) / max(n_root, 1),
            "nll_all_before": float(a0) / max(n, 1), "nll_all_after": float(a1) / max(n, 1)}


def as_tuple(w):
    return IDENTITY if w is None else (float(w["a"]), float(w["b"]), float(w["c"]))


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
