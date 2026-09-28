"""Revision 5 step 1, offline check (README section 17.1; development evidence on test under rule 10; nothing here
trains or selects): do the soft first-token answers (extract_tree_soft.py, P(Yes) under the per-dataset definition)
carry more information about WHICH nodes contain hate than the decoded 0-3 answers (answer_source "words")?

Per test node (queryable nodes of every test video), the state from the ground truth: 0 = node of a negative video,
1 = node of a positive video without a hateful second, 2 = node with a hateful second. Reported, for the soft p and
for the decoded answer (max category level; "any category >= 2"):
  - node ROC-AUC of state 2 vs {0, 1} and of 2 vs 1 (inside positive videos), overall and per length bucket;
  - mean p / yes-rate per state x length bucket (the 47% / 57% table of README 10.4);
  - within-video AUC: per positive video with both states among its nodes <= 16 s, AUC of 2 vs 1, averaged;
  - Spearman of the per-video mean p (nodes <= 8 s) with the GT hate fraction over positive videos (README 16.2);
  - video-level ROC of the root answer;
  - agreement between soft (p > .5) and decoded (any >= 2).

    python experiments/20260925_query_paradigm/soft_answer_check.py --corpus hatemm [--view both] [--split test]
Writes runs/20260929_query_paradigm_r5/soft_check/<corpus>_<view>_<split>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src"))
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402
import qtree                                    # noqa: E402

OUT = os.path.join(ROOT, "runs", "20260929_query_paradigm_r5", "soft_check")
BUCKETS = [(4, 8), (8, 16), (16, 32), (32, 64), (64, 128), (128, 100000)]


def auc(y, x):
    y, x = np.asarray(y), np.asarray(x)
    return float(roc_auc_score(y, x)) if len(y) > 1 and y.min() != y.max() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--view", default="both")
    ap.add_argument("--split", default="test")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    P, Tp, _ = qdata.load_soft_p(a.corpus, "soft_" + a.view)
    D, Td = qdata.load_answers(a.corpus, "words")
    vids = [v for v in ids[a.split] if v in P and v in D]
    print("%s %s: %d videos with soft answers (of %d)" % (a.corpus, a.split, len(vids), len(ids[a.split])), flush=True)
    rows = []                                   # (video, state, length, p, dec_max, dec_yes)
    for v in vids:
        y = np.asarray(gt[a.split][v])
        tr = qtree.tree(Tp[v])
        for n in np.where(tr["queryable"])[0]:
            a_, b_ = int(tr["a"][n]), int(tr["b"][n])
            p = P[v].get((a_, b_))
            d = D[v].get((a_, b_))
            if p is None or d is None:
                continue
            s = 0 if labels[v] == 0 else (2 if y[a_:b_].any() else 1)
            rows.append((v, s, b_ - a_, p, int(np.max(d)), int(np.any(d >= 2))))
    V = np.array([r[0] for r in rows]); S = np.array([r[1] for r in rows]); L = np.array([r[2] for r in rows])
    p = np.array([r[3] for r in rows]); dm = np.array([r[4] for r in rows]); dy = np.array([r[5] for r in rows])
    res = {"corpus": a.corpus, "view": a.view, "split": a.split, "n_nodes": len(rows), "n_videos": len(vids)}
    res["node_auc_2_vs_01"] = {"soft": auc(S == 2, p), "dec_max": auc(S == 2, dm), "dec_yes": auc(S == 2, dy)}
    m = S > 0
    res["node_auc_2_vs_1"] = {"soft": auc(S[m] == 2, p[m]), "dec_max": auc(S[m] == 2, dm[m]), "dec_yes": auc(S[m] == 2, dy[m])}
    res["by_length"] = {}
    for lo, hi in BUCKETS:
        k = (L >= lo) & (L < hi)
        if k.sum() == 0:
            continue
        r = {"n": int(k.sum())}
        for s in (0, 1, 2):
            ks = k & (S == s)
            r["state%d" % s] = {"n": int(ks.sum()), "mean_p": float(p[ks].mean()) if ks.any() else None,
                                "p_gt_half": float((p[ks] > .5).mean()) if ks.any() else None,
                                "dec_yes_rate": float(dy[ks].mean()) if ks.any() else None}
        km = k & (S > 0)
        r["auc_2_vs_1"] = {"soft": auc(S[km] == 2, p[km]), "dec_max": auc(S[km] == 2, dm[km])}
        r["auc_2_vs_01"] = {"soft": auc(S[k] == 2, p[k]), "dec_max": auc(S[k] == 2, dm[k])}
        res["by_length"]["%d-%d" % (lo, min(hi, 999))] = r
    # within-video AUC over short nodes of positive videos
    w_soft, w_dec = [], []
    for v in set(V[S > 0]):
        k = (V == v) & (L <= 16)
        if k.sum() > 1 and S[k].min() == 1 and S[k].max() == 2:
            w_soft.append(auc(S[k] == 2, p[k])); w_dec.append(auc(S[k] == 2, dm[k]))
    res["within_video_auc_le16s"] = {"n_videos": len(w_soft), "soft": float(np.mean(w_soft)) if w_soft else None,
                                     "dec_max": float(np.mean(w_dec)) if w_dec else None}
    # fraction signal
    pos = sorted(set(V[S > 0]))
    frac = [float(np.mean(gt[a.split][v])) for v in pos]
    for name, arr in (("soft", p), ("dec_max", dm)):
        mean_short = [float(arr[(V == v) & (L <= 8)].mean()) if ((V == v) & (L <= 8)).any() else np.nan for v in pos]
        mean_all = [float(arr[V == v].mean()) for v in pos]
        ok = ~np.isnan(mean_short)
        res["fraction_spearman_%s" % name] = {"nodes_le8s": float(spearmanr(np.array(frac)[ok], np.array(mean_short)[ok]).correlation),
                                              "all_nodes": float(spearmanr(frac, mean_all).correlation)}
    # video-level from the root answer
    root_p, root_d, lab = [], [], []
    for v in vids:
        T = Tp[v]
        if (0, T) in P[v] and P[v][(0, T)] is not None and D[v].get((0, T)) is not None:
            root_p.append(P[v][(0, T)]); root_d.append(int(np.max(D[v][(0, T)]))); lab.append(labels[v])
    res["video_auc_root"] = {"soft": auc(lab, root_p), "dec_max": auc(lab, root_d), "n": len(lab)}
    res["agreement_soft_vs_dec"] = float(np.mean((p > .5) == (dy == 1)))
    res["soft_p_quantiles_all"] = np.quantile(p, [.1, .25, .5, .75, .9]).round(4).tolist()
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "%s_%s_%s.json" % (a.corpus, a.view, a.split)), "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("node_auc_2_vs_01", "node_auc_2_vs_1", "within_video_auc_le16s",
                                           "fraction_spearman_soft", "fraction_spearman_dec_max", "video_auc_root",
                                           "agreement_soft_vs_dec", "soft_p_quantiles_all")}, indent=1))
    for k, r in res["by_length"].items():
        print("len %-8s n %5d | mean_p s0/s1/s2 %s | p>.5 %s | dec_yes %s | auc 2v1 soft %s dec %s" % (
            k, r["n"], [None if r["state%d" % s]["mean_p"] is None else round(r["state%d" % s]["mean_p"], 3) for s in (0, 1, 2)],
            [None if r["state%d" % s]["p_gt_half"] is None else round(r["state%d" % s]["p_gt_half"], 3) for s in (0, 1, 2)],
            [None if r["state%d" % s]["dec_yes_rate"] is None else round(r["state%d" % s]["dec_yes_rate"], 3) for s in (0, 1, 2)],
            None if r["auc_2_vs_1"]["soft"] is None else round(r["auc_2_vs_1"]["soft"], 3),
            None if r["auc_2_vs_1"]["dec_max"] is None else round(r["auc_2_vs_1"]["dec_max"], 3)))


if __name__ == "__main__":
    main()
