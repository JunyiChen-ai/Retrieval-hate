"""Answer models read on the logit scale of the soft answer (experiments/20261110_answer_likelihood/README.md).

Revision 6 bins P(Yes) into 8 quantile levels of the training answers and fits a free table per state. Most training
answers are near 0, so the quantile edges sit at small P(Yes) and the top level spans logit P(Yes) from about 2 to 9
(HateMM: 76% of the positive roots fall into it); a strong Yes and a weak Yes then carry the same evidence.
Here the answer level is a bin of a uniform grid on logit P(Yes) (`logit_grid`), and the table of the hateful state is
tied to the table of the no-hate state by a smooth term, so the few positive roots estimate one or two numbers:

  fit_tilted   state 0 as revision 6 (free table and per-level log-length slope on the nodes of negative training
               videos, qtree.fit_anchored_shared); state 1: log P(l | 1, u) = log P(l | 0, u) + sum_k beta_k x_l^k + c,
               x_l the bin centre in logit units / 5, beta by maximum likelihood on the roots of the positive training
               videos. degree 1: the log-likelihood ratio of an answer is linear in logit P(Yes), beta_1 / 5 per logit
               (1 = the VLM's own probability taken as calibrated evidence).
  fit_smooth   both states smooth: log P(l | s, u) = sum_k (a_{s,k} + b_k u) x_l^k + c (degree 2: a discretised
               Gaussian on the logit scale whose mean and spread move with the log length; b shared, fitted with a_0
               on the negative videos, a_1 on the positive roots).
All fits use video labels only (as the anchored model) and return theta, omega of shape (1, 2, levels) for
qtree.AnswerModel(theta, len_mu, len_sd, True, 2, [0], omega).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import qtree

X_SCALE = 5.0


def logit_grid(levels, lo=-10.0, hi=10.0):
    """levels - 1 P(Yes) edges uniform on the logit scale over [lo, hi] (answers outside fall into the end bins) and
    the bin centres in logit units."""
    z = np.linspace(lo, hi, int(levels) + 1)
    return 1.0 / (1.0 + np.exp(-z[1:-1])), 0.5 * (z[:-1] + z[1:])


def anchor_rows(train_answers, len_mu, len_sd):
    """The training answers whose state the video label fixes: state 0 = every node of a negative video, state 1 =
    the root of a positive video. Per state (levels (n,), standardised log lengths (n,))."""
    rows = {0: ([], []), 1: ([], [])}
    for label, nodes in train_answers:
        for a, b, T, o in nodes:
            if o is None:
                continue
            if label == 0:
                s = 0
            elif a == 0 and b == T:
                s = 1
            else:
                continue
            rows[s][0].append(int(np.asarray(o).reshape(-1)[0]))
            rows[s][1].append((np.log(b - a) - len_mu) / len_sd)
    return {s: (torch.as_tensor(np.asarray(r[0], dtype=np.int64)), torch.as_tensor(np.asarray(r[1], dtype=np.float64)))
            for s, r in rows.items()}


def _features(centers, degree):
    x = torch.as_tensor(np.asarray(centers, dtype=np.float64)) / X_SCALE
    return torch.stack([x ** k for k in range(1, int(degree) + 1)], dim=1)       # L, d


def _lbfgs(params, nll_fn):
    opt = torch.optim.LBFGS(params, max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        nll = nll_fn()
        nll.backward()
        return nll

    opt.step(closure)
    return float(nll_fn().detach())


def fit_tilted(train_answers, len_mu, len_sd, centers, degree=1, fixed=None):
    """See the module docstring. fixed: beta given instead of fitted (diagnostic). Returns theta, omega (1, 2, L)
    and {"n": answers per state, "beta": ..., "nll_root": ...}."""
    assert qtree.N_CAT == 1 and qtree.N_LEV == len(centers)
    theta, omega, n_fit = qtree.fit_anchored_shared(train_answers, len_mu, len_sd)
    O, U = anchor_rows(train_answers, len_mu, len_sd)[1]
    feats = _features(centers, degree)
    base = torch.as_tensor(theta[0, 0])[None] + torch.as_tensor(omega[0, 0])[None] * U[:, None]      # n, L
    beta = torch.zeros(feats.shape[1], dtype=torch.float64, requires_grad=fixed is None)
    if fixed is not None:
        beta = torch.as_tensor(np.asarray(fixed, dtype=np.float64))

    def nll_fn():
        return -F.log_softmax(base + (feats @ beta)[None], dim=-1).gather(1, O[:, None]).sum()

    nll = _lbfgs([beta], nll_fn) if fixed is None else float(nll_fn())
    theta[0, 1] = theta[0, 0] + (feats @ beta).detach().numpy()
    omega[0, 1] = omega[0, 0]
    return theta, omega, {"n": n_fit, "beta": beta.detach().numpy().tolist(), "nll_root": nll}


def fit_smooth(train_answers, len_mu, len_sd, centers, degree=2):
    """See the module docstring. Returns theta, omega (1, 2, L) and the fitted coefficients."""
    assert qtree.N_CAT == 1 and qtree.N_LEV == len(centers)
    rows = anchor_rows(train_answers, len_mu, len_sd)
    feats = _features(centers, degree)
    d = feats.shape[1]
    a0 = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    b = torch.zeros(d, dtype=torch.float64, requires_grad=True)
    O0, U0 = rows[0]

    def nll0():
        return -F.log_softmax((feats @ a0)[None] + U0[:, None] * (feats @ b)[None], dim=-1).gather(1, O0[:, None]).sum()

    _lbfgs([a0, b], nll0)
    a0, b = a0.detach(), b.detach()
    a1 = a0.clone().requires_grad_(True)
    O1, U1 = rows[1]

    def nll1():
        return -F.log_softmax((feats @ a1)[None] + U1[:, None] * (feats @ b)[None], dim=-1).gather(1, O1[:, None]).sum()

    nll = _lbfgs([a1], nll1)
    theta = np.zeros((1, 2, len(centers)))
    omega = np.zeros((1, 2, len(centers)))
    theta[0, 0], theta[0, 1] = (feats @ a0).numpy(), (feats @ a1.detach()).numpy()
    omega[0, 0] = omega[0, 1] = (feats @ b).numpy()
    return theta, omega, {"n": {0: len(O0), 1: len(O1)}, "a0": a0.numpy().tolist(),
                          "a1": a1.detach().numpy().tolist(), "b": b.numpy().tolist(), "nll_root": nll}
