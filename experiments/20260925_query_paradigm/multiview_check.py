"""Revision 5 step 3, offline check (README section 17.3; development evidence on test under rule 10; nothing here
trains or selects): can the answer reliability of the SHORT nodes inside positive videos be estimated WITHOUT
per-second labels from three views of the same VLM on the same node (both = frames + transcript, frames only,
transcript only; extract_tree_soft.py --view)?

Model (Pepe & Janes 2007; Fu et al. 2020, "triplet" method of moments; Dawid & Skene 1979 EM as a cross-check):
inside positive videos, a short node is in state 2 (contains hate) with probability pi, else state 1; given the
state the three binary answers y_i = 1[p_i > .5] are conditionally independent with P(y_i = 1 | state 2) = a_i and
P(y_i = 1 | state 1) = c_i.  Moments (closed form):
    Cov(y_i, y_j) = pi (1 - pi) d_i d_j,  d_i = a_i - c_i          -> d_i sqrt(pi (1 - pi)) = sqrt(C_ij C_ik / C_jk)
    E[prod (y_i - m_i)] = pi (1 - pi) (1 - 2 pi) d_1 d_2 d_3        -> pi
    a_i = m_i + (1 - pi) d_i,  c_i = m_i - pi d_i
with the sign convention d_i > 0 (yes is more likely under hate).  State 0 (nodes of negative videos) is observed
directly, and the conditional independence given the state is TESTED there (negative videos: the state is known):
pairwise Pearson correlations of the three binary answers and a G-test of mutual independence on the 2x2x2 table.

Population for the label-free estimate: TRAIN split, queryable nodes of length <= 16 s (also 4-8 and 8-16 s
separately), positive videos (video label only).  Development check on TEST (per-second labels, rule 10): the
estimated (c_both, a_both) against the measured yes-rates of state-1 and state-2 nodes; usable if both differ by
<= .10 (README 17.3).  The same estimator run on the test positives separates estimator error from train/test shift.
Bootstrap over videos (200 resamples) gives the spread.  With the fitted binary model, the per-node posterior of
state 2 gives the soft-level (data.soft_edges, train quantiles) distributions of state 1 and 2 for the both view
(the table the anchored answer model would use), compared to the ground-truth-state histograms on test.

    python experiments/20260925_query_paradigm/multiview_check.py --corpus hatemm [--selftest]
Writes runs/20260929_query_paradigm_r5/multiview/<corpus>.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys

import numpy as np
from scipy.stats import chi2

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402
import qtree                                    # noqa: E402

OUT = os.path.join(ROOT, "runs", "20260929_query_paradigm_r5", "multiview")
VIEWS = ("both", "frames", "text")
BUCKETS = {"4-16": (4, 17), "4-8": (4, 9), "8-16": (9, 17)}


def triplet(Y):
    """Y: (n, 3) binary. Closed-form moment estimate -> dict(pi, a (3,), c (3,)) or None if not identifiable."""
    Y = np.asarray(Y, float)
    m = Y.mean(0)
    R = Y - m
    C = (R.T @ R) / len(Y)                        # covariance
    if min(C[0, 1], C[0, 2], C[1, 2]) <= 1e-9:
        return None
    g = np.array([np.sqrt(C[0, 1] * C[0, 2] / C[1, 2]), np.sqrt(C[0, 1] * C[1, 2] / C[0, 2]),
                  np.sqrt(C[0, 2] * C[1, 2] / C[0, 1])])                     # d_i sqrt(pi (1 - pi))
    k3 = float(np.mean(R[:, 0] * R[:, 1] * R[:, 2]))                       # pi (1 - pi) (1 - 2 pi) d1 d2 d3
    r = k3 / float(np.prod(g))                                             # (1 - 2 pi) / sqrt(pi (1 - pi))
    u = np.sign(r) * abs(r) / np.sqrt(4.0 + r * r)                         # 1 - 2 pi
    pi = float((1.0 - u) / 2.0)
    s = np.sqrt(pi * (1.0 - pi))
    d = g / s
    a = np.clip(m + (1.0 - pi) * d, 0.0, 1.0)
    c = np.clip(m - pi * d, 0.0, 1.0)
    return {"pi": pi, "a": a.tolist(), "c": c.tolist(), "cov": C.tolist(), "third_central": k3}


def dawid_skene(Y, init, iters=500):
    """Two-state EM with conditionally independent binary raters, from the triplet solution."""
    Y = np.asarray(Y, float)
    pi, a, c = init["pi"], np.array(init["a"]), np.array(init["c"])
    for _ in range(iters):
        l2 = np.log(pi) + (Y * np.log(np.clip(a, 1e-6, 1)) + (1 - Y) * np.log(np.clip(1 - a, 1e-6, 1))).sum(1)
        l1 = np.log(1 - pi) + (Y * np.log(np.clip(c, 1e-6, 1)) + (1 - Y) * np.log(np.clip(1 - c, 1e-6, 1))).sum(1)
        r = 1.0 / (1.0 + np.exp(l1 - l2))
        pi_n = float(np.clip(r.mean(), 1e-4, 1 - 1e-4))
        a_n = (r[:, None] * Y).sum(0) / r.sum()
        c_n = ((1 - r)[:, None] * Y).sum(0) / (1 - r).sum()
        done = abs(pi_n - pi) < 1e-7 and np.abs(a_n - a).max() < 1e-7 and np.abs(c_n - c).max() < 1e-7
        pi, a, c = pi_n, a_n, c_n
        if done:
            break
    ll = float(np.logaddexp(l1, l2).sum())
    return {"pi": pi, "a": a.tolist(), "c": c.tolist(), "loglik": ll}


def responsibilities(Y, fit):
    Y = np.asarray(Y, float)
    pi, a, c = fit["pi"], np.array(fit["a"]), np.array(fit["c"])
    l2 = np.log(pi) + (Y * np.log(np.clip(a, 1e-6, 1)) + (1 - Y) * np.log(np.clip(1 - a, 1e-6, 1))).sum(1)
    l1 = np.log(1 - pi) + (Y * np.log(np.clip(c, 1e-6, 1)) + (1 - Y) * np.log(np.clip(1 - c, 1e-6, 1))).sum(1)
    return 1.0 / (1.0 + np.exp(l1 - l2))


def independence_tests(Y):
    """Pairwise Pearson correlations and the G-test of mutual independence of three binary variables (df 4)."""
    Y = np.asarray(Y, float)
    n = len(Y)
    corr = np.corrcoef(Y.T) if n > 2 else np.full((3, 3), np.nan)
    m = Y.mean(0)
    G = 0.0
    for cell in itertools.product((0, 1), repeat=3):
        obs = float(np.all(Y == np.array(cell), axis=1).sum())
        exp = n * float(np.prod([m[i] if cell[i] else 1 - m[i] for i in range(3)]))
        if obs > 0 and exp > 0:
            G += 2 * obs * np.log(obs / exp)
    return {"n": int(n), "pearson": [[float(corr[i, j]) for j in range(3)] for i in range(3)],
            "G": float(G), "df": 4, "p_value": float(chi2.sf(G, 4)), "yes_rate": m.tolist()}


def selftest(seed=0):
    rng = np.random.default_rng(seed)
    pi, a, c = .35, np.array([.55, .40, .50]), np.array([.15, .20, .10])
    z = rng.random(20000) < pi
    Y = np.where(z[:, None], rng.random((20000, 3)) < a, rng.random((20000, 3)) < c).astype(float)
    t = triplet(Y)
    e = dawid_skene(Y, t)
    print("selftest true pi %.3f a %s c %s" % (pi, a.tolist(), c.tolist()))
    print("  triplet pi %.3f a %s c %s" % (t["pi"], np.round(t["a"], 3).tolist(), np.round(t["c"], 3).tolist()))
    print("  EM      pi %.3f a %s c %s" % (e["pi"], np.round(e["a"], 3).tolist(), np.round(e["c"], 3).tolist()))
    assert abs(t["pi"] - pi) < .03 and np.abs(np.array(t["a"]) - a).max() < .04 and np.abs(np.array(t["c"]) - c).max() < .04
    print("  independence test on the state-1 subset (independent by construction) p = %.3f; on the mixture p = %.2e" % (
          independence_tests(Y[~z])["p_value"], independence_tests(Y)["p_value"]))


def backbone_scores(corpus, trial, ids, device="cpu", score="smax"):
    """README 17.3 variant: the prior network of a finished trial as the third rater. Per video, the node potential
    phi (revision-4 node prior, qtree.tree order) of every queryable node, or the maximum of the per-second prior
    over the node when the model has no node prior. Returns {video: {(a, b): score}}."""
    import torch
    import concern_diagnostics as cd
    import policy
    cfg0 = json.load(open(os.path.join(trial, "config.json")))
    qdata.configure_source(cfg0["answer_source"], int(cfg0.get("soft_levels", qdata.SOFT_LEVELS)))
    answers, _T = qdata.load_answers(corpus, cfg0["answer_source"])
    _summ, cfg, model, _am, _chain = cd.load_trial(trial, device, answers, ids)
    vids = [v for sp in ("train", "val", "test") for v in ids[sp]]
    store = qdata.Store(corpus, vids, cfg.get("text_sources", ["bert"]))
    out = {}
    with torch.no_grad():
        for v in vids:
            s_, _g, phi = policy.video_prior_nodes(model, store, v, device)
            tr = qtree.tree(store.T[v])
            d = {}
            for n in np.where(tr["queryable"])[0]:
                a_, b_ = int(tr["a"][n]), int(tr["b"][n])
                use_phi = score == "phi" and phi is not None and len(phi) == len(tr["a"])
                d[(a_, b_)] = float(phi[n]) if use_phi else float(np.max(s_[a_:b_]))
            out[v] = d
    return out


def collect(corpus, split, labels, ids, gt, P, Tp, levels=8):
    """Rows (video, state, length, p_both, p_frames, p_text) for the queryable nodes of `split` with all three views;
    state 0 / 1 / 2 as in soft_answer_check (2 needs per-second labels: test / val only, else 1 for every positive)."""
    rows = []
    for v in ids[split]:
        if any(v not in P[w] for w in VIEWS):
            continue
        y = np.asarray(gt[split][v]) if split in gt and v in gt[split] else None
        tr = qtree.tree(Tp[v])
        for n in np.where(tr["queryable"])[0]:
            a_, b_ = int(tr["a"][n]), int(tr["b"][n])
            ps = [P[w][v].get((a_, b_)) for w in VIEWS]
            if any(p is None for p in ps):
                continue
            s = 0 if labels[v] == 0 else (2 if (y is not None and y[a_:b_].any()) else 1)
            rows.append((v, s, b_ - a_, *ps, a_, b_))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--levels", type=int, default=8)
    ap.add_argument("--third", default="both", help="the third rater of the triplet: both (pre-registered) or "
                    "backbone (README 17.3 variant; needs --trial); frames and text are always the other two")
    ap.add_argument("--trial", default=None)
    ap.add_argument("--backbone-score", default="smax", choices=("smax", "phi"),
                    help="smax = maximum of the per-second prior over the node; phi = the node potential (relative)")
    ap.add_argument("--backbone-yes-rate", type=float, default=0.2,
                    help="the backbone rater says yes above the (1 - rate) train quantile of its node scores")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        if not a.corpus:
            return
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    P, Tp, split_of = {}, None, None
    for w in VIEWS:
        P[w], Tp, split_of = qdata.load_soft_p(a.corpus, "soft_" + w)
    edges = qdata.soft_edges(P["both"], split_of, a.levels)
    res = {"corpus": a.corpus, "views": list(VIEWS), "threshold": .5, "levels": a.levels, "edges": edges.tolist(),
           "third": a.third, "trial": a.trial}
    bb = None
    if a.third == "backbone":
        assert a.trial, "--third backbone needs --trial"
        bb = backbone_scores(a.corpus, a.trial, ids, score=a.backbone_score)
        tr_scores = np.array([sc for v in ids["train"] if v in bb for sc in bb[v].values()])
        bb_thr = float(np.quantile(tr_scores, 1.0 - a.backbone_yes_rate))
        res["backbone_threshold"] = bb_thr
        print("backbone rater: %d train node scores, threshold %.4f (yes rate %.2f)" % (len(tr_scores), bb_thr, a.backbone_yes_rate))
    rng = np.random.default_rng(0)
    for split in ("train", "test", "val"):
        if split not in ids:
            continue
        rows = collect(a.corpus, split, labels, ids, gt, P, Tp)
        if not rows:
            continue
        V = np.array([r[0] for r in rows]); S = np.array([r[1] for r in rows]); L = np.array([r[2] for r in rows])
        node_ab = [(r[-2], r[-1]) for r in rows]
        Pm = np.array([r[3:6] for r in rows], float)                         # n, 3 soft p (both, frames, text)
        Y = (Pm > .5).astype(float)
        if bb is not None:                       # triplet = (backbone, frames, text); column 0 replaced
            Y[:, 0] = np.array([float(bb[r[0]][(int(a_), int(b_))] > bb_thr) if r[0] in bb else np.nan
                                for r, (a_, b_) in zip(rows, node_ab)], float)
            keep = ~np.isnan(Y[:, 0])
            V, S, L, Pm, Y = V[keep], S[keep], L[keep], Pm[keep], Y[keep]
        out = {"n_nodes": len(rows), "n_videos": len(set(V))}
        for bname, (lo, hi) in BUCKETS.items():
            k = (L >= lo) & (L < hi)
            neg, pos = k & (S == 0), k & (S > 0)
            b = {"n_neg": int(neg.sum()), "n_pos": int(pos.sum())}
            if neg.sum() > 10:
                b["neg_independence"] = independence_tests(Y[neg])
            if pos.sum() > 30:
                t = triplet(Y[pos])
                b["triplet"] = t
                if t is not None:
                    b["em"] = dawid_skene(Y[pos], t)
                    b["pos_marginal_yes"] = Y[pos].mean(0).tolist()
                    # bootstrap over positive videos
                    vp = np.array(sorted(set(V[pos])))
                    bs = []
                    for _ in range(a.boot):
                        pick = rng.choice(vp, len(vp), replace=True)
                        idx = np.concatenate([np.where(pos & (V == v))[0] for v in pick])
                        tb = triplet(Y[idx])
                        if tb is not None:
                            bs.append([tb["pi"]] + tb["a"] + tb["c"])
                    if bs:
                        bs = np.array(bs)
                        b["triplet_boot"] = {"n_ok": len(bs), "q05": np.quantile(bs, .05, 0).round(3).tolist(),
                                             "q95": np.quantile(bs, .95, 0).round(3).tolist(),
                                             "order": ["pi", "a_both", "a_frames", "a_text", "c_both", "c_frames", "c_text"]}
                    # soft-level tables for the both view from the responsibilities
                    r = responsibilities(Y[pos], b["em"])
                    lev = np.searchsorted(edges, Pm[pos, 0], side="right")
                    h2 = np.array([(r * (lev == l)).sum() for l in range(a.levels)]) / r.sum()
                    h1 = np.array([((1 - r) * (lev == l)).sum() for l in range(a.levels)]) / (1 - r).sum()
                    b["level_hist_est"] = {"state1": h1.round(4).tolist(), "state2": h2.round(4).tolist()}
                    yb = (Pm[pos, 0] > .5).astype(float)               # the both view's binary answer
                    b["both_implied"] = {"c": float((yb * (1 - r)).sum() / (1 - r).sum()),
                                         "a": float((yb * r).sum() / r.sum())}
                if (S[pos] == 2).any():                                  # per-second labels: the measured rates
                    b["gt"] = {"pi": float((S[pos] == 2).mean()),
                               "a": Y[k & (S == 2)].mean(0).tolist(), "c": Y[k & (S == 1)].mean(0).tolist(),
                               "state2_independence": independence_tests(Y[k & (S == 2)]) if (k & (S == 2)).sum() > 10 else None,
                               "state1_independence": independence_tests(Y[k & (S == 1)]) if (k & (S == 1)).sum() > 10 else None}
                    lev = np.searchsorted(edges, Pm[:, 0], side="right")
                    b["level_hist_gt"] = {"state%d" % s: np.bincount(lev[k & (S == s)], minlength=a.levels).astype(float)
                                          .__truediv__(max(1, (k & (S == s)).sum())).round(4).tolist() for s in (1, 2)}
                    if t is not None:
                        gb = {"c": float(Y[k & (S == 1)][:, 0].mean()), "a": float(Y[k & (S == 2)][:, 0].mean())}
                        yb_all = (Pm[:, 0] > .5).astype(float)
                        gt_both = {"c": float(yb_all[k & (S == 1)].mean()), "a": float(yb_all[k & (S == 2)].mean())}
                        b["gt_both"] = gt_both
                        b["abs_err_both"] = {"a": abs(b["both_implied"]["a"] - gt_both["a"]),
                                             "c": abs(b["both_implied"]["c"] - gt_both["c"]),
                                             "pi": abs(t["pi"] - b["gt"]["pi"])}
            if neg.sum() > 10:
                b["neg_yes_rate"] = Y[neg].mean(0).tolist()
            out[bname] = b
        res[split] = out
        print("== %s %s: %d nodes / %d videos" % (a.corpus, split, len(rows), len(set(V))))
        for bname, b in out.items():
            if not isinstance(b, dict):
                continue
            t = b.get("triplet")
            print("  %-5s n_neg %5d n_pos %5d | neg yes %s indep p %s | triplet pi %s a %s c %s | EM pi %s a %s c %s%s" % (
                bname, b["n_neg"], b["n_pos"],
                None if "neg_yes_rate" not in b else np.round(b["neg_yes_rate"], 3).tolist(),
                None if "neg_independence" not in b else round(b["neg_independence"]["p_value"], 4),
                None if not t else round(t["pi"], 3), None if not t else np.round(t["a"], 3).tolist(),
                None if not t else np.round(t["c"], 3).tolist(),
                None if "em" not in b else round(b["em"]["pi"], 3), None if "em" not in b else np.round(b["em"]["a"], 3).tolist(),
                None if "em" not in b else np.round(b["em"]["c"], 3).tolist(),
                "" if "gt" not in b else " | GT pi %.3f a %s c %s" % (b["gt"]["pi"], np.round(b["gt"]["a"], 3).tolist(),
                                                                       np.round(b["gt"]["c"], 3).tolist())))
            if "both_implied" in b:
                print("        both view implied (c, a) = (%.3f, %.3f)%s" % (
                    b["both_implied"]["c"], b["both_implied"]["a"],
                    "" if "gt_both" not in b else "  GT (%.3f, %.3f)  |err| c %.3f a %.3f" % (
                        b["gt_both"]["c"], b["gt_both"]["a"], b["abs_err_both"]["c"], b["abs_err_both"]["a"])))
    os.makedirs(OUT, exist_ok=True)
    if "val" in res and "level_hist_gt" in res["val"].get("4-16", {}):   # the fallback table (validation labels)
        vt = res["val"]["4-16"]
        json.dump({"corpus": a.corpus, "source": "validation per-second labels (fallback), 4-16 s nodes, both view",
                   "levels": a.levels, "edges": edges.tolist(), "n_pos": vt["n_pos"], "pi": vt["gt"]["pi"],
                   "p_state1": vt["level_hist_gt"]["state1"], "p_state2": vt["level_hist_gt"]["state2"]},
                  open(os.path.join(OUT, "%s_val_state1_table.json" % a.corpus), "w"), indent=1)
    suffix = "" if a.third == "both" else "_" + a.third + ("" if a.third != "backbone" else "_" + a.backbone_score)
    json.dump(res, open(os.path.join(OUT, "%s%s.json" % (a.corpus, suffix)), "w"), indent=1, default=float)
    # the state-1 table for train.py answer_model "anchored_ext" (README 17.3): the both-view level distribution
    # of the positive-video nodes WITHOUT hate, estimated on the TRAIN split (4-16 s nodes) without labels
    tr = res.get("train", {}).get("4-16", {})
    if "level_hist_est" in tr:
        tab = {"corpus": a.corpus, "source": "multiview triplet + EM, train split, 4-16 s nodes, both view",
               "levels": a.levels, "edges": edges.tolist(), "n_pos": tr["n_pos"], "pi": tr["em"]["pi"],
               "p_state1": tr["level_hist_est"]["state1"], "p_state2": tr["level_hist_est"]["state2"]}
        tab["source"] += "; third rater %s" % a.third
        json.dump(tab, open(os.path.join(OUT, "%s%s_state1_table.json" % (a.corpus, suffix)), "w"), indent=1)
        print("state-1 table written: p_state1 %s" % np.round(tab["p_state1"], 3).tolist())


if __name__ == "__main__":
    main()
