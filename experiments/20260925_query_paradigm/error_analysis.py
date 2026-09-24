"""Error analysis for the 2026-09-25 redesign (README section 1). Test labels are read for developmental
analysis only (rule 10); nothing here trains or selects a model.

Q1  VLM-silent positive test videos (fine rate of level >= 2 below .1): which HCS harm dimensions they carry
    (hateful / insulting / sexual / violence / harm), vs the non-silent positives.
Q2  Per-window level distribution (0-3) for fine windows inside GT hate, outside GT in positive videos, and in
    negative videos; ROC of the raw level vs the >= 2 binary as a window score.
Q3  Silent positives: fraction of the video covered by GT hate; whether any coarse block says >= 2 or >= 1.

    python experiments/20260925_query_paradigm/error_analysis.py
Output: runs/20260925_query_paradigm/error_analysis/summary.json
"""
from __future__ import annotations

import ast
import csv
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from hate_common import data as hdata  # noqa: E402
import vlm_verdict                     # noqa: E402

OUT = os.path.join(ROOT, "runs", "20260925_query_paradigm", "error_analysis")
DIMS = ("hateful", "insulting", "sexual", "violence", "harm")


def hcs_dims():
    csv.field_size_limit(1 << 30)
    path = next(p for p in hdata.HCS_SEGMENT_CSV_CANDIDATES if os.path.isfile(p))
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            labels = ast.literal_eval(row["Segment-Level Label"])
            spans = ast.literal_eval(row["Segment Timestamp"])
            if len(labels) != len(spans):
                continue
            out[row["Video Id"].strip()] = [int(any(int(l[1 + d]) == 1 for l in labels)) for d in range(5)]
    return out


def window_gt(gt, k):
    """Fraction of GT hate seconds in each of k equal windows (the verdict windows' convention)."""
    n = len(gt)
    b = np.linspace(0, n, k + 1)
    return np.array([gt[int(np.floor(b[i])):max(int(np.ceil(b[i + 1])), int(np.floor(b[i])) + 1)].mean()
                     for i in range(k)])


def main():
    os.makedirs(OUT, exist_ok=True)
    res = {}
    for corpus in ("hatemm", "hateclipseg"):
        labels = hdata.load_labels(corpus)
        gt = hdata.gt_arrays(corpus, "test")
        F = vlm_verdict.load_verdicts(corpus, k=30, tag="qwen")
        C = vlm_verdict.load_verdicts(corpus, k=4, tag="qwen")
        ids = [v for v in hdata.load_split(corpus, "test") if v in gt and v in F and v in C]
        pos = [v for v in ids if labels[v] == 1]
        silent = [v for v in pos if np.mean(F[v] >= 2) < 0.1]
        loud = [v for v in pos if v not in silent]
        r = {"n_test": len(ids), "n_pos": len(pos), "n_silent_pos": len(silent)}
        # Q1
        if corpus == "hateclipseg":
            D = hcs_dims()
            for name, grp in (("silent_pos", silent), ("loud_pos", loud)):
                m = np.array([D[v] for v in grp])
                r["dims_" + name] = {d: int(m[:, i].sum()) for i, d in enumerate(DIMS)}
                r["only_nonhateful_" + name] = int(sum(1 for row in m if row[0] == 0))
        # Q2
        lev_in, lev_out, lev_neg = [], [], []
        y, s_raw = [], []
        for v in ids:
            wg = window_gt(np.asarray(gt[v], float), 30)
            for w in range(30):
                lv = int(F[v][w])
                if labels[v] == 0:
                    lev_neg.append(lv)
                    y.append(0)
                elif wg[w] > 0.5:
                    lev_in.append(lv)
                    y.append(1)
                else:
                    lev_out.append(lv)
                    y.append(0)
                s_raw.append(lv)
        y, s_raw = np.array(y), np.array(s_raw)
        dist = lambda a: [round(float(np.mean(np.array(a) == l)), 4) for l in range(4)]
        r["level_dist"] = {"in_gt": dist(lev_in), "pos_outside_gt": dist(lev_out), "neg_video": dist(lev_neg),
                           "n": [len(lev_in), len(lev_out), len(lev_neg)]}
        r["window_roc_raw_level"] = round(float(roc_auc_score(y, s_raw)), 4)
        r["window_roc_binary_ge2"] = round(float(roc_auc_score(y, (s_raw >= 2).astype(float))), 4)
        r["window_roc_binary_ge1"] = round(float(roc_auc_score(y, (s_raw >= 1).astype(float))), 4)
        # Q2 restricted to silent positives: level 1 inside vs outside GT
        li, lo = [], []
        for v in silent:
            wg = window_gt(np.asarray(gt[v], float), 30)
            for w in range(30):
                (li if wg[w] > 0.5 else lo).append(int(F[v][w]))
        r["silent_level_dist"] = {"in_gt": dist(li) if li else None, "outside_gt": dist(lo) if lo else None,
                                  "n": [len(li), len(lo)]}
        # Q3
        cov = [float(np.mean(gt[v])) for v in silent]
        r["silent_gt_coverage"] = {"mean": round(float(np.mean(cov)), 3),
                                   "n_ge_.9": int(sum(c >= .9 for c in cov)), "n_le_.5": int(sum(c <= .5 for c in cov))}
        r["silent_coarse_any_ge2"] = int(sum(np.any(C[v] >= 2) for v in silent))
        r["silent_coarse_any_ge1"] = int(sum(np.any(C[v] >= 1) for v in silent))
        r["silent_fine_any_ge1"] = int(sum(np.any(F[v] >= 1) for v in silent))
        negs = [v for v in ids if labels[v] == 0]
        r["neg_fine_any_ge1"] = int(sum(np.any(F[v] >= 1) for v in negs))
        r["neg_fine_any_ge2"] = int(sum(np.any(F[v] >= 2) for v in negs))
        r["neg_coarse_any_ge2"] = int(sum(np.any(C[v] >= 2) for v in negs))
        r["n_neg"] = len(negs)
        r["loud_coarse_any_ge2"] = int(sum(np.any(C[v] >= 2) for v in loud))
        res[corpus] = r
        print(corpus, json.dumps(r, indent=1))
    json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
