"""Question policy and evaluation-time inference (README section 2.4).

For one video: the prior network gives s (T,) and g; each step computes the exact posterior on the tree, the
expected information gain of every unasked queryable node, asks the best one (reads its cached answer), and
records the posterior after the answer. `order="bfs"` asks the queryable nodes top-down, left to right (ablation
d). Parse failures count as a call with no observation."""
from __future__ import annotations

import numpy as np
import torch

import qtree


@torch.no_grad()
def video_prior(model, store, v, device):
    """Five-crop mean of the per-second logit s and of the video logit g."""
    T = store.T[v]
    f_v = torch.from_numpy(np.stack([store.visual(v, c) for c in range(5)])).to(device)
    f_a = torch.from_numpy(np.repeat(store.at[v][None], 5, axis=0)).to(device)
    mask = torch.ones(5, T, dtype=torch.bool, device=device)
    s, g, *_ = model(f_a, f_v, mask)
    return s.mean(0).double().cpu().numpy(), float(g.mean().item())


class Asker:
    """Outcome tables of one video's queryable nodes under the current answer model."""

    def __init__(self, vt, answer_model, categories):
        self.vt = vt
        self.q = vt.q_ids
        self.qpos = {int(n): i for i, n in enumerate(self.q)}
        self.cats = list(categories)
        with torch.no_grad():
            lp = answer_model.level_logp(torch.as_tensor(vt.length[self.q])).double().cpu().numpy()
        lp = lp[:, self.cats]                                            # nq, C', S, L
        if lp.shape[2] == 2:                                             # two-state arm: s=1 shares s=0
            lp = lp[:, :, [0, 0, 1]]
        self.table = qtree.outcome_table(lp)                             # nq, 3, O
        self.po_s = np.exp(self.table)
        self.h_s = -np.sum(self.po_s * self.table, axis=2)               # nq, 3

    def loglik(self, node, o):
        return self.table[self.qpos[int(node)], :, qtree.answer_index(np.asarray(o)[self.cats])]

    def eig(self, cand_pos, w):
        po = np.einsum("nso,ns->no", self.po_s[cand_pos], w)
        h = -np.sum(po * np.log(np.clip(po, 1e-300, None)), axis=1)
        return (h - np.sum(w * self.h_s[cand_pos], axis=1)) / np.log(2.0)


def flat_scores(vt, s, g, logA, asked):
    """Ablation (b): no tree structure. logit p_t = logit(g pi_t) + mean over the asked nodes covering t of
    [log P(o | s=2) - log P(o | s=0)] (same model, same asked nodes)."""
    T = vt.T
    log_p = -np.logaddexp(0.0, -g) - np.logaddexp(0.0, -s)                # log(g pi_t)
    prior = log_p - np.log(-np.expm1(np.minimum(log_p, -1e-12)))         # logit(g pi_t)
    llr, cnt = np.zeros(T), np.zeros(T)
    for n in asked:
        a, b = vt.tr["a"][n], vt.tr["b"][n]
        llr[a:b] += logA[n, 2] - logA[n, 0]
        cnt[a:b] += 1
    return 1.0 / (1.0 + np.exp(-(prior + llr / np.maximum(cnt, 1))))


def run_video(vt, s, g, asker, answers_v, max_calls, order="eig", fusion="tree"):
    """Returns dict: scores (list of (T,) arrays after 0..n calls; the tree posterior, or the flat score of
    ablation b when fusion == "flat"), eig (EIG of each asked question), asked (node ids), p_G (after each number
    of calls)."""
    logA = np.zeros((vt.N, 3))
    pG, m, p = vt.infer(s, g, logA)
    first = p if fusion == "tree" else flat_scores(vt, s, g, logA, [])
    scores, eigs, asked, pgs = [first], [], [], [pG]
    remaining = list(asker.q) if order == "eig" else list(asker.q)      # q ids are in breadth-first order
    rem_mask = np.ones(len(asker.q), dtype=bool)
    tr = vt.tr
    for _ in range(min(max_calls, len(asker.q))):
        cand = np.where(rem_mask)[0]
        if order == "eig":
            nodes = asker.q[cand]
            w = np.stack([np.full(len(cand), 1.0 - pG), pG * (1.0 - m[nodes]), pG * m[nodes]], axis=1)
            e = asker.eig(cand, w)
            j = int(np.argmax(e))
            e_best = float(e[j])
        else:
            j = 0
            n0 = asker.q[cand[0]]
            w = np.array([[1.0 - pG, pG * (1.0 - m[n0]), pG * m[n0]]])
            e_best = float(asker.eig(cand[:1], w)[0])
        pos = int(cand[j])
        node = int(asker.q[pos])
        rem_mask[pos] = False
        o = answers_v.get((int(tr["a"][node]), int(tr["b"][node])))
        if o is not None:
            logA[node] = asker.loglik(node, o)
        pG, m, p = vt.infer(s, g, logA)
        eigs.append(e_best)
        asked.append(node)
        scores.append(p if fusion == "tree" else flat_scores(vt, s, g, logA, asked))
        pgs.append(pG)
    del remaining
    return {"scores": scores, "eig": eigs, "asked": asked, "p_G": pgs}


def stop_calls(eigs, c):
    """Number of calls under the threshold rule: ask while the best question's EIG >= c."""
    for k, e in enumerate(eigs):
        if e < c:
            return k
    return len(eigs)


def calibrate(runs, budget):
    """Smallest threshold c whose mean number of calls over `runs` is <= budget (runs: list of eig lists)."""
    cands = np.unique(np.concatenate([np.asarray(r, dtype=np.float64) for r in runs if len(r)] + [np.zeros(1)]))
    best = None
    for c in cands[::-1]:                    # descending: fewer calls first
        mean = float(np.mean([stop_calls(r, c) for r in runs]))
        if mean <= budget + 1e-9:
            best = (float(c), mean)
        else:
            break
    if best is None:                         # even the largest threshold exceeds the budget
        c = float(cands[-1]) + 1e-9
        best = (c, float(np.mean([stop_calls(r, c) for r in runs])))
    return best
