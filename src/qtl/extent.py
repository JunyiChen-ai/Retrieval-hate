"""Extent question on the query tree (experiments/20261109_tree_within README section 8).

The VLM is asked "No / Some / All": whether the node holds harmful content and whether it fills the whole node. The
node state then has four values, s_n in {0: G = 0; 1: G = 1 and none; 2: G = 1 and some; 3: G = 1 and all}
(revision 6: three, with some and all merged).

Inference (exact): the chain-tree upward pass of ctree._up with the node's "any" flag replaced by its extent
e in {0 none, 1 some, 2 all}: per node a table over (first second's state, last second's state, e), 12 log-weights;
two children merge by none + none = none, all + all = all, otherwise some. A single second is none (y = 0) or all
(y = 1). Node potentials phi (revision 4) act on "any" = e in {some, all}. Marginals by automatic differentiation:
E[y_t | G = 1, o] = d log V1 / d s_t, P(e_n = some / all | G = 1, o) = d log V1 / d eta_n (one eta per value).

Answer: the first-token probabilities (p_no, p_some, p_all); two coordinates, presence 1 - p_no and fullness
p_all / (p_some + p_all), each binned into training-quantile levels (label-free). Answer table P(o | s, length):
the two coordinates independent given the extent state, each a multinomial logistic regression of the level on the
standardised log length (revision-6 form); states G = 0 and "G = 1, none" share one table.
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as F

from . import ctree, qtree

BIG = ctree.BIG
N_EXT = 3                                   # none, some, all
SOURCE = "ext_both_p3"


# ------------------------------------------------------------------ answers
def load_extent(corpus, source=SOURCE):
    """video id -> {(a, b): (presence, fullness) or None}, T, split (raw values before binning)."""
    from .data import CORPUS_DIR, ROOT
    base = os.path.join(ROOT, "data", "vlm_tree", CORPUS_DIR[corpus])
    out, T, split = {}, {}, {}
    for path in sorted(glob.glob(os.path.join(base, source + "[.]*jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if r["id"] in out:
                continue
            d = {}
            for a, b, o, _raw in r["nodes"]:
                if o is None:
                    d[(int(a), int(b))] = None
                    continue
                pn, ps, pa = float(o[0]), float(o[1]), float(o[2])
                d[(int(a), int(b))] = (1.0 - pn, pa / (ps + pa) if ps + pa > 0 else 0.5)
            out[r["id"]] = d
            T[r["id"]] = int(r["T"])
            split[r["id"]] = r["split"]
    return out, T, split


def bin_extent(raw, split, levels=8):
    """Binned answers {video: {(a, b): int array (2,) or None}} and the edges (training quantiles per coordinate)."""
    edges = []
    for c in range(2):
        vals = np.array([o[c] for v, d in raw.items() if split.get(v) == "train" for o in d.values() if o is not None])
        assert len(vals) > 0, "no training answers to set the levels"
        edges.append(np.quantile(vals, [k / levels for k in range(1, levels)]))
    out = {v: {k: (None if o is None else np.array([int(np.searchsorted(edges[c], o[c], side="right"))
                                                     for c in range(2)], dtype=np.int64))
               for k, o in d.items()} for v, d in raw.items()}
    return out, edges


def node_extent(y, a, b):
    """Extent state of [a, b) from per-second labels y: 0 none, 1 some, 2 all."""
    s = int(np.sum(y[a:b]))
    return 0 if s == 0 else (2 if s == b - a else 1)


class ExtentAnswerModel:
    """theta, omega (2 coordinates, 3 extent states, L levels); len_mu, len_sd standardise the log length."""

    def __init__(self, theta, omega, len_mu, len_sd):
        self.theta = np.asarray(theta, dtype=np.float64)
        self.omega = np.asarray(omega, dtype=np.float64)
        self.len_mu, self.len_sd = float(len_mu), float(len_sd)
        self.L = self.theta.shape[-1]

    def level_logp(self, length):
        """(n, 2, 3, L) log P(level | extent state, length)."""
        u = (np.log(np.asarray(length, dtype=np.float64)) - self.len_mu) / self.len_sd
        z = self.theta[None] + self.omega[None] * u[:, None, None, None]
        return z - np.logaddexp.reduce(z, axis=-1, keepdims=True)

    def table(self, length):
        """(n, 4, L * L) log P(o | s) over the four node states (G = 0 uses the none table)."""
        lp = self.level_logp(length)                                     # n, 2, 3, L
        t = (lp[:, 0, :, :, None] + lp[:, 1, :, None, :]).reshape(len(lp), N_EXT, -1)
        return t[:, [0, 0, 1, 2]]

    def to_dict(self):
        return {"theta": self.theta.tolist(), "omega": self.omega.tolist(), "len_mu": self.len_mu,
                "len_sd": self.len_sd}


def _fit_state(O, U, L):
    """Per coordinate a multinomial logistic regression of the level on U (one Laplace pseudo-count per level at
    u = 0), maximum likelihood. O (n, 2) int, U (n,). Returns theta, omega (2, L)."""
    O = torch.as_tensor(O, dtype=torch.long)
    U = torch.as_tensor(U, dtype=torch.float64)
    th = torch.zeros(2, L, dtype=torch.float64, requires_grad=True)
    om = torch.zeros(2, L, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([th, om], max_iter=500, tolerance_grad=1e-10, tolerance_change=1e-12,
                            line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lp = F.log_softmax(th[None] + om[None] * U[:, None, None], dim=-1)
        nll = -lp.gather(2, O[:, :, None]).sum() - F.log_softmax(th, dim=-1).sum()
        nll.backward()
        return nll

    opt.step(closure)
    return th.detach().numpy(), om.detach().numpy()


def fit_extent(answers, labels, train_ids, val_ids, val_gt, len_mu, len_sd, levels=8):
    """State none: every node of the negative training videos (label-only). States some / all: the nodes of the
    validation videos with per-second GT whose extent is some / all (validation labels, no gradient; user ruling
    2026-09-29). Returns the model and the answer counts per state."""
    rows = {s: ([], []) for s in range(N_EXT)}
    for v in train_ids:
        if labels[v] != 0 or v not in answers:
            continue
        for (a, b), o in answers[v].items():
            if o is not None:
                rows[0][0].append(o)
                rows[0][1].append((np.log(b - a) - len_mu) / len_sd)
    for v in val_ids:
        if labels[v] != 1 or v not in val_gt or v not in answers:
            continue
        y = np.asarray(val_gt[v]).astype(int)
        for (a, b), o in answers[v].items():
            if o is None or b > len(y):
                continue
            s = node_extent(y, a, b)
            if s > 0:
                rows[s][0].append(o)
                rows[s][1].append((np.log(b - a) - len_mu) / len_sd)
    theta, omega = np.zeros((2, N_EXT, levels)), np.zeros((2, N_EXT, levels))
    for s in range(N_EXT):
        th, om = _fit_state(np.array(rows[s][0]), np.array(rows[s][1]), levels)
        theta[:, s], omega[:, s] = th, om
    return ExtentAnswerModel(theta, omega, len_mu, len_sd), {s: len(rows[s][0]) for s in range(N_EXT)}


# ------------------------------------------------------------------ inference
def _merge(L, R, logA):
    """L, R: (n, f, l, e) log tables of two adjacent intervals, e in {none, some, all} -> (n, f, l, e) of their
    union. logA (2, 2)."""
    X = L[:, :, :, :, None, None, None] + logA[None, None, :, None, :, None, None] + R[:, None, None, None, :, :, :]
    X = torch.logsumexp(X, dim=(2, 4))                                   # n, f, eL, l, eR
    e0 = X[:, :, 0, :, 0]
    e2 = X[:, :, 2, :, 2]
    e1 = torch.logsumexp(torch.stack([X[:, :, 0, :, 1], X[:, :, 0, :, 2], X[:, :, 1, :, 0], X[:, :, 1, :, 1],
                                      X[:, :, 1, :, 2], X[:, :, 2, :, 0], X[:, :, 2, :, 1]]), dim=0)
    return torch.stack([e0, e1, e2], dim=-1)


def _up(forest, s, A4, chain, eta=None, phi=None):
    """log V1 (B,): weight of the paths with at least one harmful second, answers under states 1-3, without g; and
    lp0 (B,): the answers' log-likelihood under state 0. A4 (N, 4); eta (N, 2) on some / all; phi (N,) on any."""
    assert not getattr(chain, "normalized", False)
    dev = s.device
    flat = s.reshape(-1)
    logA = chain.logA_for(forest).to(dev).to(s.dtype)
    assert logA.dim() == 2
    a1 = A4[:, 1:].clone()
    if eta is not None:
        a1 = a1 + torch.cat([torch.zeros_like(eta[:, :1]), eta], dim=1)
    if phi is not None:
        a1 = a1 + phi.to(a1.dtype)[:, None] * torch.tensor([0.0, 1.0, 1.0], dtype=a1.dtype, device=dev)
    prev = None
    for d in range(len(forest.levels) - 1, -1, -1):
        lev = forest.levels[d]
        nodes = lev["nodes"].to(dev)
        n_leaf = lev["n_leaf"]
        tabs, lzs = [], []
        if n_leaf:
            sl = flat[lev["leaf_row"].to(dev)]
            t = torch.full((n_leaf, 2, 2, N_EXT), -BIG, device=dev, dtype=s.dtype)
            ar = torch.arange(n_leaf, device=dev)
            z = torch.zeros(n_leaf, dtype=torch.long, device=dev)
            t = t.index_put((ar, z, z, z), torch.zeros_like(sl))
            t = t.index_put((ar, z + 1, z + 1, z + 2), sl)
            tabs.append(t)
            lzs.append(torch.zeros(n_leaf, device=dev, dtype=s.dtype))
        if len(nodes) > n_leaf:
            L, R = lev["L"].to(dev), lev["R"].to(dev)
            tabs.append(_merge(prev[0][L], prev[0][R], logA))
            lzs.append(prev[1][L] + prev[1][R])
        tab = torch.cat(tabs)
        lz = torch.cat(lzs)
        tab = tab + a1[nodes][:, None, None, :]
        c = torch.logsumexp(tab.reshape(len(nodes), -1), dim=1)
        prev = (tab - c[:, None, None, None], lz + c)
    pr = forest.pos_root.to(dev)
    root = torch.logsumexp(prev[0][pr][..., 1:], dim=-1)                 # B, f, l: at least one harmful second
    lp0 = torch.zeros(forest.B, device=dev, dtype=s.dtype).index_add(0, forest.node_video.to(dev), A4[:, 0])
    if chain.closed:
        enter = logA[0, :][None].expand(forest.B, -1)
        leave = logA[:, 0][None].expand(forest.B, -1)
        v1 = prev[1][pr] + torch.logsumexp(enter[:, :, None] + root + leave[:, None, :], dim=(1, 2))
    else:
        v1 = prev[1][pr] + torch.logsumexp(chain.logpi()[None, :, None] + root, dim=(1, 2))
    return v1, lp0


def up(forest, s, g, A4, chain, phi=None):
    """(logW1, logW0) as ctree.up, zero-inflated chain: logW1 = g + log V1(answers) - log V1(no answers)."""
    assert getattr(chain, "zero_inflated", False)
    v1, lp0 = _up(forest, s, A4, chain, None, phi)
    v1_prior, _ = _up(forest, s, torch.zeros_like(A4), chain, None, phi)
    return g + v1 - v1_prior, lp0


def marginals(forest, s, g, A4, chain, phi=None):
    """p_G (B,), P(e_n = some | G = 1, o) and P(e_n = all | G = 1, o) (N,) each, second posteriors
    p (B, Tmax) = P(G = 1 | o) P(y_t = 1 | G = 1, o). No gradient to the caller."""
    with torch.enable_grad():
        s = s.detach().double().requires_grad_(True)
        eta = torch.zeros(forest.N, 2, dtype=torch.float64, device=s.device, requires_grad=True)
        g = g.detach().double()
        phi = None if phi is None else phi.detach().double()
        chain_d = ctree._DoubleChain(chain)
        w1, w0 = up(forest, s, g, A4.double(), chain_d, phi=phi)
        v1, _ = _up(forest, s, A4.double(), chain_d, eta, phi)
        gs, ge = torch.autograd.grad(v1.sum(), (s, eta))
    pG = torch.sigmoid(w1 - w0).detach()
    return pG, ge[:, 0].detach(), ge[:, 1].detach(), (pG[:, None] * gs).detach()


# ------------------------------------------------------------------ questions
class ExtentAsker:
    """Outcome tables (nq, 4, O) of one video's queryable nodes under an ExtentAnswerModel."""

    def __init__(self, vt, eam):
        self.vt = vt
        self.q = vt.q_ids
        self.qpos = {int(n): i for i, n in enumerate(self.q)}
        self.L = eam.L
        self.table = eam.table(vt.length[self.q]) if len(self.q) else np.zeros((0, 4, eam.L ** 2))
        self.po_s = np.exp(self.table)
        self.h_s = -np.sum(self.po_s * self.table, axis=2)

    def index(self, o):
        return int(o[0]) * self.L + int(o[1])

    def loglik(self, node, o):
        return self.table[self.qpos[int(node)], :, self.index(o)]

    def eig(self, cand_pos, w):
        po = np.einsum("nso,ns->no", self.po_s[cand_pos], w)
        h = -np.sum(po * np.log(np.clip(po, 1e-300, None)), axis=1)
        return (h - np.sum(w * self.h_s[cand_pos], axis=1)) / np.log(2.0)


def run_batch(model, store, vids, eam, chain, answers, max_calls, device, allowed=None, answer_ll=None):
    """The revision-6 question loop (cpolicy.run_batch, order eig, tree fusion) with the extent question: each step
    one batched exact inference, each video asks its EIG-best unasked node (EIG over the four node states), reads
    its answer. allowed: per video the node ids that may be asked (None = every queryable node; E3 single levels).
    answer_ll (diagnostic): function (video index, node id) -> (4,) log-likelihood to use instead of the cached
    answer. Returns per video: scores (after 0..n calls), eig, asked, p_G."""
    from . import cpolicy, policy
    Ts = [store.T[v] for v in vids]
    Tm = max(Ts)
    S = torch.zeros(len(vids), Tm, dtype=torch.float64)
    G = torch.zeros(len(vids), dtype=torch.float64)
    fo = ctree.Forest(Ts, Tm)
    PHI = None
    for b, v in enumerate(vids):
        s, g, phi = policy.video_prior_nodes(model, store, v, device)
        S[b, :len(s)] = torch.from_numpy(s)
        G[b] = g
        if phi is not None:
            if PHI is None:
                PHI = torch.zeros(fo.N, dtype=torch.float64)
            PHI[int(fo.offs[b]):int(fo.offs[b]) + len(phi)] = torch.from_numpy(phi)
    S, G = S.to(device), G.to(device)
    PHI = None if PHI is None else PHI.to(device)
    vts = [cpolicy._vt(T) for T in Ts]
    askers = [ExtentAsker(vt, eam) for vt in vts]
    A4 = torch.zeros(fo.N, 4, dtype=torch.float64, device=device)
    out = [{"scores": [], "eig": [], "asked": [], "p_G": []} for _ in vids]
    rem = [np.ones(len(a.q), dtype=bool) if allowed is None else np.isin(a.q, np.asarray(sorted(allowed[b])))
           for b, a in enumerate(askers)]
    for step in range(max_calls + 1):
        pG, ms, ma, p = marginals(fo, S, G, A4, chain, PHI)
        pG, ms, ma, p = pG.cpu().numpy(), ms.cpu().numpy(), ma.cpu().numpy(), p.cpu().numpy()
        for b, T in enumerate(Ts):
            if step <= len(askers[b].q):
                out[b]["scores"].append(p[b, :T].copy())
                out[b]["p_G"].append(float(pG[b]))
        if step == max_calls:
            break
        chosen = []
        for b, (asker, off) in enumerate(zip(askers, fo.offs)):
            cand = np.where(rem[b])[0]
            if len(cand) == 0:
                continue
            nodes = asker.q[cand]
            s_, a_ = ms[off + nodes], ma[off + nodes]
            w = np.stack([np.full(len(cand), 1.0 - pG[b]), pG[b] * np.clip(1.0 - s_ - a_, 0.0, None),
                          pG[b] * s_, pG[b] * a_], axis=1)
            e = asker.eig(cand, w)
            j = int(np.argmax(e))
            chosen.append((b, int(cand[j]), int(nodes[j]), float(e[j])))
        if not chosen:
            break
        for b, pos, node, e in chosen:
            rem[b][pos] = False
            tr = vts[b].tr
            if answer_ll is None:
                o = answers[vids[b]].get((int(tr["a"][node]), int(tr["b"][node])))
                ll = None if o is None else askers[b].loglik(node, o)
            else:
                ll = answer_ll(b, node)
            if ll is not None:
                A4[fo.offs[b] + node] = torch.as_tensor(ll, dtype=torch.float64, device=device)
            out[b]["eig"].append(e)
            out[b]["asked"].append(node)
    return {v: r for v, r in zip(vids, out)}
