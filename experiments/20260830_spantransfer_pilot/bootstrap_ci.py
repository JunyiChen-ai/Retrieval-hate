#!/usr/bin/env python3
"""Paired per-video bootstrap CIs for within-ROC (SCALE_PLAN item 5).

For each corpus: per-video within-AUC averaged over seeds for each arm, then
10k bootstrap resamples over the both-class hate test videos of the paired
difference. Comparisons: valsel vs best reproduced baseline, valsel vs
loo_zero, valsel vs loo_naive.
"""
import glob
import json
import os
import sys

import numpy as np

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
from hate_common import data as hdata  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

SCORE_DIR = os.path.join(REPO, "runs", "20260830_spantransfer_pilot", "scores")
BASE_ROOT = os.path.join(REPO, "results", "reproduction", "official_val", "final")
OUT = os.path.join(REPO, "runs", "20260830_spantransfer_pilot", "bootstrap_ci.md")
BEST_BASELINE = {"hatemm": ("multihateloc", "score_fused"),
                 "mhclip_en": ("cmhkf", "score_align"),
                 "mhclip_zh": ("multihateloc", "score_union"),
                 "hateclipseg": ("vera", "score_official_postprocessed")}
CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")
NBOOT, SEED = 10000, 20260830


def rank_auc(s, y):
    y = np.asarray(y).astype(bool)
    if y.all() or not y.any():
        return None
    r = rankdata(s)
    n1, n0 = int(y.sum()), int((~y).sum())
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def per_video_auc_ours(corpus, arm):
    """video -> mean-over-seeds within AUC from our dense score files."""
    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    per = {}
    for path in sorted(glob.glob(os.path.join(SCORE_DIR, corpus,
                                              arm + "_seed*.jsonl"))):
        for line in open(path):
            d = json.loads(line)
            v = d["video_id"]
            if labels.get(v) != 1 or v not in gt:
                continue
            a = rank_auc(np.asarray(d["score"]), gt[v])
            if a is not None:
                per.setdefault(v, []).append(a)
    return {v: float(np.mean(a)) for v, a in per.items()}


def per_video_auc_baseline(corpus, method, branch):
    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    per = {}
    paths = sorted(glob.glob(os.path.join(BASE_ROOT, method, corpus,
                                          "seed_*", "scores.jsonl")) +
                   glob.glob(os.path.join(BASE_ROOT, method, corpus,
                                          "seed_*", corpus, "scores.jsonl")))
    for path in paths:
        for line in open(path):
            d = json.loads(line)
            v = d["video_id"]
            if labels.get(v) != 1 or v not in gt:
                continue
            s = np.asarray(d[branch], dtype=float)
            y = np.asarray(gt[v])
            L = min(len(s), len(y))
            a = rank_auc(s[:L], y[:L])
            if a is not None:
                per.setdefault(v, []).append(a)
    return {v: float(np.mean(a)) for v, a in per.items()}


def boot_ci(a, b):
    """Paired bootstrap CI of mean(a - b) over shared videos."""
    vids = sorted(set(a) & set(b))
    if len(vids) < 3:
        return None
    d = np.asarray([a[v] - b[v] for v in vids])
    rng = np.random.default_rng(SEED)
    idx = rng.integers(len(d), size=(NBOOT, len(d)))
    means = d[idx].mean(axis=1)
    return {"n": len(d), "mean_diff": float(d.mean()),
            "ci_lo": float(np.quantile(means, 0.025)),
            "ci_hi": float(np.quantile(means, 0.975))}


def main():
    lines = ["# Paired per-video bootstrap (within-AUC, 10k resamples)", "",
             "| corpus | comparison | n videos | mean diff | 95% CI |",
             "|---|---|---:|---:|---|"]
    for corpus in CORPORA:
        ours = per_video_auc_ours(corpus, "valsel")
        method, branch = BEST_BASELINE[corpus]
        comps = [("valsel - %s/%s" % (method, branch),
                  per_video_auc_baseline(corpus, method, branch)),
                 ("valsel - loo_zero", per_video_auc_ours(corpus, "loo_zero")),
                 ("valsel - loo_naive", per_video_auc_ours(corpus, "loo_naive"))]
        for name, other in comps:
            r = boot_ci(ours, other)
            if r is None:
                lines.append("| %s | %s | - | - | insufficient |" % (corpus, name))
                continue
            sig = "" if r["ci_lo"] <= 0 <= r["ci_hi"] else " *"
            lines.append("| %s | %s | %d | %+.4f | [%+.4f, %+.4f]%s |" % (
                corpus, name, r["n"], r["mean_diff"], r["ci_lo"], r["ci_hi"], sig))
    with open(OUT, "w") as fh:
        fh.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
