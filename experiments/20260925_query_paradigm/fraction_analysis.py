"""Revision-5 error analysis (README section 16; development evidence on test under rule 10: reads test scores, test
answers and test GT; nothing here trains or selects).

Question: where does pooled AP come from, and how well does each signal rank the positive videos by how much of
the video is hateful (GT hate fraction f_v)?
  headroom   pooled AP / ROC of oracle scores that keep one part of the ranking perfect and the rest from the model:
             (1) "video oracle": every second of a video gets its GT fraction f_v (negatives 0): perfect ranking of
             videos by amount of hate, no within-video ranking; (2) "fraction oracle + model within": model scores
             inside each video, shifted so that the video mean equals f_v; (3) "model video + perfect within": each
             video keeps its model mean, seconds inside ranked by GT (hateful seconds get mean + delta).
  per-video  Spearman over positive test videos between f_v and: the model's mean score (B = 0 / 8 / 32), the
             backbone prior only (B = 0), and raw VLM signals with every queryable node answered: root max level,
             share of nodes with some category >= 2 (all nodes / nodes <= 16 s), length-weighted share of seconds
             covered by "yes" leaves-of-answered-nodes.
             Also AUC positive vs negative of the same signals.

    python experiments/20260925_query_paradigm/fraction_analysis.py --corpus hatemm --runs name=dir ... [--out f.json]
Each run dir holds scores_test_fixed{0,8,32}.jsonl (a trial or a control arm).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE)
import hier_evidence_common as hc               # noqa: E402
import data as qdata                            # noqa: E402


def read_scores(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["video_id"]] = np.asarray(r["score_av"], dtype=np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--runs", nargs="*", default=[])
    ap.add_argument("--source", default="words")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    vids = [v for v in ids["test"]]
    G = {v: np.asarray(gt["test"][v], dtype=np.float64) for v in vids}
    frac = {v: float(G[v].mean()) for v in vids}
    pos = [v for v in vids if labels[v] == 1 and frac[v] > 0]
    y_all = np.concatenate([G[v] for v in vids])

    def pooled(S):
        s = np.concatenate([S[v] for v in vids])
        return float(average_precision_score(y_all, s)), float(roc_auc_score(y_all, s))

    rng = np.random.RandomState(0)
    jit = {v: rng.rand(len(G[v])) * 1e-9 for v in vids}
    res = {"corpus": a.corpus, "n_test": len(vids), "n_pos_with_spans": len(pos),
           "mean_fraction_pos": float(np.mean([frac[v] for v in pos])),
           "share_hateful_seconds": float(y_all.mean())}
    res["video_oracle"] = pooled({v: np.full(len(G[v]), frac[v]) + jit[v] for v in vids})
    print("%s: %d test videos, %d positive with spans, mean fraction %.3f, hateful seconds %.3f" % (
        a.corpus, len(vids), len(pos), res["mean_fraction_pos"], res["share_hateful_seconds"]))
    print("  video oracle (every second = GT fraction of its video): AP %.4f ROC %.4f" % res["video_oracle"])

    # raw VLM signals with every queryable node answered
    A, _ = qdata.load_answers(a.corpus, a.source)
    raw = {}
    for v in vids:
        T = len(G[v])
        nodes = [(s, e, o) for (s, e), o in A[v].items() if o is not None]
        root = [max(o) for s, e, o in nodes if s == 0 and e >= T]
        yes_all = np.mean([max(o) >= 2 for s, e, o in nodes]) if nodes else 0.0
        short = [max(o) >= 2 for s, e, o in nodes if e - s <= 16]
        best = np.full(T, np.inf)
        leaf = np.zeros(T)
        for s, e, o in nodes:
            m = (e - s) < best[s:e]
            idx = np.arange(s, e)[m]
            best[idx] = e - s
            leaf[idx] = float(max(o) >= 2)
        raw[v] = {"root_max": float(root[0]) if root else 0.0, "yes_share_all": float(yes_all),
                  "yes_share_short": float(np.mean(short)) if short else 0.0, "yes_seconds_shortest": float(leaf.mean())}

    def rank_stats(sig):
        rho = spearmanr([frac[v] for v in pos], [sig[v] for v in pos]).correlation
        auc = roc_auc_score([labels[v] for v in vids], [sig[v] for v in vids])
        return float(rho), float(auc)

    res["raw_signals"] = {}
    for k in ("root_max", "yes_share_all", "yes_share_short", "yes_seconds_shortest"):
        res["raw_signals"][k] = rank_stats({v: raw[v][k] for v in vids})
        print("  raw VLM, all nodes answered, %-22s Spearman with fraction (positives) %.3f | video AUC %.3f" % (
            k, *res["raw_signals"][k]))
    res["runs"] = {}
    for spec in a.runs:
        name, d = spec.split("=", 1)
        r = {}
        for B in (0, 8, 32):
            p = os.path.join(d, "scores_test_fixed%d.jsonl" % B)
            if not os.path.exists(p):
                continue
            S = read_scores(p)
            mean = {v: float(S[v].mean()) for v in vids}
            shift = {v: S[v] - mean[v] + frac[v] for v in vids}
            delta = 1e-6
            perfect_within = {v: np.full(len(G[v]), mean[v]) + delta * G[v] + jit[v] for v in vids}
            r[B] = {"pooled": pooled(S), "constant_per_video": pooled({v: np.full(len(G[v]), mean[v]) + jit[v]
                                                                        for v in vids}),
                    "fraction_oracle_model_within": pooled(shift), "model_video_perfect_within": pooled(perfect_within),
                    "rank_mean": rank_stats(mean)}
            print("  %-14s B=%-2d pooled %.4f/%.4f | const/video %.4f/%.4f | GT fraction + model within %.4f/%.4f | "
                  "model video + GT within %.4f/%.4f | mean score: Spearman %.3f video AUC %.3f" % (
                      name, B, *r[B]["pooled"], *r[B]["constant_per_video"], *r[B]["fraction_oracle_model_within"],
                      *r[B]["model_video_perfect_within"], *r[B]["rank_mean"]))
        res["runs"][name] = r
    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=2)


if __name__ == "__main__":
    main()
