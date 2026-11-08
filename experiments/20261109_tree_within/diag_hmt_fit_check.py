"""Diagnostic (README section 6): does the fitted HMT reproduce the answer correlations it was built for?
Negative training videos (the fit data): Pearson correlation of the answer levels (0..L-1) of node pairs in the same
video, by tree distance (1 = parent-child, 2, 3, 4, >= 5) among queryable nodes; the same statistic simulated from the
fitted model (runs/20261109_tree_within/stage_a/<corpus>/hmt_fit.json) on the same trees. Prints both."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import data as qdata, hmt, qtree                               # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402


def dist(tr, i, j):
    ai, aj = [], []
    while i >= 0:
        ai.append(i); i = tr["parent"][i]
    while j >= 0:
        aj.append(j); j = tr["parent"][j]
    common = set(ai) & set(aj)
    return min(ai.index(c) + aj.index(c) for c in common)


def pair_corr(videos, max_pairs=200000, seed=0):
    rng = np.random.default_rng(seed)
    acc = {}
    for tr, q, lev in videos:
        if len(q) < 2:
            continue
        for _ in range(min(60, len(q) * (len(q) - 1) // 2)):
            i, j = rng.choice(len(q), 2, replace=False)
            d = min(dist(tr, q[i], q[j]), 5)
            acc.setdefault(d, []).append((lev[i], lev[j]))
    return {d: (float(np.corrcoef(np.array(v).T)[0, 1]), len(v)) for d, v in sorted(acc.items())}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True)
    a = ap.parse_args()
    labels, ids, _, _ = hc.load_fixed_cohort(a.corpus)
    qdata.configure_source("soft_both", 8)
    answers, _ = qdata.load_answers(a.corpus, "soft_both")
    fit = json.load(open(ROOT / f"runs/20261109_tree_within/stage_a/{a.corpus}/hmt_fit.json"))
    ham = hmt.HMTAnswerModel.from_dict(fit["model"])
    rng = np.random.default_rng(1)
    emp, sim = [], []
    pi = np.exp(ham.logw)
    pih = np.exp(hmt.normal_logw(hmt.Z_H))
    KH = len(pih)
    for v in ids["train"]:
        if labels[v] != 0 or not answers[v]:
            continue
        T = max(b for (_, b) in answers[v])
        tr = qtree.tree(T)
        q = [int(n) for n in np.where(tr["queryable"])[0] if answers[v].get((int(tr["a"][n]), int(tr["b"][n]))) is not None]
        if len(q) < 2:
            continue
        emp.append((tr, q, [int(answers[v][(int(tr["a"][n]), int(tr["b"][n]))][0]) for n in q]))
        # simulate biases down the queryable tree, then levels from the state-0 table
        b = {}
        for n in np.where(tr["queryable"])[0]:                       # breadth-first order: parents first
            p = tr["parent"][n]
            if p < 0 or p not in b:
                b[n] = rng.choice(ham.K, p=pi)
            else:                                                    # same video part, tree part copied or redrawn
                h = b[p] % KH if rng.random() < ham.rho else rng.choice(KH, p=pih)
                b[n] = (b[p] // KH) * KH + h
        lens = [float(tr["b"][n] - tr["a"][n]) for n in q]
        lp = ham.level_logp(lens)[:, :, 0, 0, :]                      # n, K, L
        pr = np.exp(lp)
        pr = pr / pr.sum(-1, keepdims=True)
        lev = [int(rng.choice(pr.shape[-1], p=pr[i, b[n]])) for i, n in enumerate(q)]
        sim.append((tr, q, lev))
    ce, cs = pair_corr(emp), pair_corr(sim)
    print(a.corpus, "rho %.3f sigma_v %.2f sigma_h %.2f" % (ham.rho, ham.sigma_v, ham.sigma_h))
    for d in ce:
        print("  distance %s: data %.3f (%d pairs)  model %.3f" % (d if d < 5 else ">=5", ce[d][0], ce[d][1], cs[d][0]))


if __name__ == "__main__":
    main()
