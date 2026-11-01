"""Per-video answer offset (revision 6, experiments/20261101_r5_completion/README.md section 3).

The VLM's answers within one video are correlated: some videos get a high P(Yes) on most nodes, others a low one
(answer_stats.py: the between-video share of the answer-level residual is .43-.59 on HateMM negative videos), and
with the answers of a video shuffled across videos of the same node states the r5 model improves steadily up to 32
calls while with the real answers it stalls (sim_monotone.py). Treating the answers of a video as independent makes
every extra answer count as new evidence. Model: each video has one offset b shared by all its answers; given the
node state s and b,
    P(o = l | s, b) = softmax_l(theta_{s, l} + b c_l),   c_l = l / (L - 1) - 1/2   (L answer levels),
i.e. b tilts the answer distribution towards higher or lower P(Yes) levels in every node of the video. b has a
discrete distribution on a fixed fine grid whose weights are fitted by maximum likelihood (the nonparametric maximum
likelihood estimate of a mixing distribution, Laird 1978, JASA): no number of offset classes is set by hand; after
the fit the points carrying almost no weight are dropped (PRUNE, a numerical tolerance) so inference runs over the
few points that remain. Fitted once before training, label-free like the anchored model (qtree.fit_anchored): the
state-0 table and the offset weights on the negative training videos (every node in state 0, all answers of a video
share its offset); the state-1 table on the roots of the positive training videos (state 1 by the label), by its
marginal likelihood under the fitted offset distribution. The offset distribution is shared by both states.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from . import qtree

GRID = np.arange(-12.0, 12.0 + 1e-9, 0.5)       # offset grid (tilt units over the full level range)
PRUNE = 1e-4                                     # drop grid points whose fitted weight is below this


def level_scores(L):
    return np.arange(L, dtype=np.float64) / max(L - 1, 1) - 0.5


def _lbfgs(params, closure_fn):
    opt = torch.optim.LBFGS(params, max_iter=1000, tolerance_grad=1e-10, tolerance_change=1e-13,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = closure_fn()
        loss.backward()
        return loss
    opt.step(closure)


def fit_offset(train_answers, grid=GRID):
    """train_answers: [(label, [(a, b, T, o), ...]), ...] as for qtree.fit_anchored (o: answer vector (N_CAT,) or
    None). Returns theta (N_CAT, 2, N_LEV), support (K,), log weights (K,), and fit details."""
    C, L = qtree.N_CAT, qtree.N_LEV
    c = torch.as_tensor(level_scores(L))
    b = torch.as_tensor(np.asarray(grid, dtype=np.float64))
    neg, roots = [], []
    for label, nodes in train_answers:
        obs = [o for (a_, b_, T, o) in nodes if o is not None]
        if label == 0 and obs:
            neg.append(np.stack(obs))
        elif label == 1:
            roots += [o for (a_, b_, T, o) in nodes if o is not None and a_ == 0 and b_ == T]
    # state 0 and the offset weights: counts per (video, category, level) are sufficient statistics
    N = torch.as_tensor(np.stack([np.stack([np.bincount(x[:, k], minlength=L) for k in range(C)]) for x in neg]),
                        dtype=torch.float64)                                     # V, C, L
    th0 = torch.zeros(C, L, dtype=torch.float64, requires_grad=True)
    zeta = torch.zeros(len(b), dtype=torch.float64, requires_grad=True)

    def nll0():
        lp = F.log_softmax(th0[None, :, :] + b[:, None, None] * c[None, None, :], dim=-1)   # K, C, L
        per = torch.einsum("vcl,kcl->vk", N, lp)                                            # V, K
        lw = F.log_softmax(zeta, dim=0)
        return -torch.logsumexp(per + lw[None], dim=1).sum() - F.log_softmax(th0, dim=-1).sum()
    _lbfgs([th0, zeta], nll0)
    w = torch.softmax(zeta, 0).detach().numpy()
    keep = w >= PRUNE
    support = np.asarray(grid, dtype=np.float64)[keep]
    w = w[keep] / w[keep].sum()
    mean = float(np.sum(w * support))                    # centre: a shift of all offsets equals a shift of theta
    support = support - mean
    theta0 = th0.detach().numpy() + mean * level_scores(L)[None, :]
    # state 1 (positive roots): one answer per video, its offset integrated out under the fitted distribution
    Rt = torch.as_tensor(np.stack(roots), dtype=torch.long)                              # R, C
    bs, lws = torch.as_tensor(support), torch.as_tensor(np.log(w))
    th1 = torch.zeros(C, L, dtype=torch.float64, requires_grad=True)

    def nll1():
        lp = F.log_softmax(th1[None, :, :] + bs[:, None, None] * c[None, None, :], dim=-1)  # K, C, L
        per = lp[:, torch.arange(C)[None, :], Rt].sum(-1)                                   # K, R
        return -torch.logsumexp(per + lws[:, None], dim=0).sum() - F.log_softmax(th1, dim=-1).sum()
    _lbfgs([th1], nll1)
    theta = np.stack([theta0, th1.detach().numpy()], axis=1)                             # C, 2, L
    info = {"n_negative_videos": len(neg), "n_negative_answers": int(N.sum() / C), "n_roots": len(roots),
            "support": support.tolist(), "weights": w.tolist(), "offset_sd": float(np.sqrt(np.sum(w * support ** 2)))}
    return theta, support, np.log(w), info


class OffsetAnswerModel:
    """Anchored two-state answer table with the per-video offset; tables per offset value."""

    def __init__(self, theta, support, logw, categories):
        self.theta = np.asarray(theta, dtype=np.float64)                  # C, 2, L
        self.support = np.asarray(support, dtype=np.float64)              # K
        self.logw = np.asarray(logw, dtype=np.float64)                    # K
        self.cats = list(categories)
        self.K = len(self.support)

    def level_logp(self):
        """(K, C, 3, L) log P(o_c = l | s, b_k); states 0 and 1 share the state-0 table (anchored two states)."""
        c = level_scores(self.theta.shape[-1])
        x = self.theta[None] + self.support[:, None, None, None] * c[None, None, None, :]
        lp = x - np.logaddexp.reduce(x, axis=-1, keepdims=True)
        return lp[:, :, [0, 0, 1], :]

    def outcome_tables(self, n):
        """(K, n, 3, O) log P(o | s, b_k) over all answer vectors of the used categories, for n nodes."""
        lp = self.level_logp()[:, self.cats]                              # K, C', 3, L
        return np.stack([qtree.outcome_table(np.repeat(lp[k][None], n, axis=0)) for k in range(self.K)])

    def loglik(self, answers):
        """(K, N, 3) torch log P(o_n | s, b_k) for integer answers (N, N_CAT)."""
        lp = torch.as_tensor(self.level_logp())                           # K, C, 3, L
        o = torch.as_tensor(np.asarray(answers), dtype=torch.long)        # N, C
        out = torch.zeros(self.K, o.shape[0], 3, dtype=torch.float64)
        for k in self.cats:
            out = out + lp[:, k][:, None, :, :].expand(-1, o.shape[0], -1, -1).gather(
                3, o[:, k][None, :, None, None].expand(self.K, -1, 3, 1)).squeeze(-1)
        return out
