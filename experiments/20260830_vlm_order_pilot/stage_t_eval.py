#!/usr/bin/env python3
"""Stage T evaluation: teacher window scores -> per-second scores -> within-hate
macro ROC/AP via the standard evaluator. Kill gate: within-ROC < .60 on BOTH
hatemm and mhclip_en kills the candidate.

Seconds covered by several windows take the mean of covering window scores
(fixed choice; max reported as a secondary column).
"""
import json
import os
import sys

import numpy as np

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

OUT_DIR = os.path.join(REPO, "runs", "20260830_vlm_order_pilot")
CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")


def densify(rows, how="mean"):
    scores = {}
    for r in rows:
        T = r["T"]
        acc = np.zeros(T)
        cnt = np.zeros(T)
        mx = np.full(T, -np.inf)
        for (a, b), s in zip(r["spans"], r["scores"]):
            acc[a:b] += s
            cnt[a:b] += 1
            mx[a:b] = np.maximum(mx[a:b], s)
        dense = np.where(cnt > 0, acc / np.maximum(cnt, 1), 0.0) \
            if how == "mean" else np.where(np.isfinite(mx), mx, 0.0)
        scores[r["video_id"]] = dense.astype(float)
    return scores


def main():
    lines = ["# Stage T: teacher (Qwen2.5-VL-7B window scores) on TEST hate videos",
             "", "| corpus | pool | n videos | within-ROC macro (n) | within-AP macro |",
             "|---|---|---:|---:|---:|"]
    summary = {}
    for corpus in CORPORA:
        path = os.path.join(OUT_DIR, "teacher_%s.jsonl" % corpus)
        if not os.path.exists(path):
            continue
        rows = [json.loads(l) for l in open(path)]
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        hate_ids = {v for v in gt if labels.get(v) == 1}
        for how in ("mean", "max"):
            sc = densify(rows, how)
            sc = {v: s for v, s in sc.items() if v in gt}
            m = evaluate_scores(sc, {v: gt[v] for v in sc}, hate_ids)
            aps = []
            for v in sc:
                y = np.asarray(gt[v]).astype(int)
                if v in hate_ids and 0 < y.sum() < len(y):
                    aps.append(average_precision_score(y, sc[v]))
            lines.append("| %s | %s | %d | %.4f (%d) | %.4f |" % (
                corpus, how, len(sc), m["per_video"]["macro_auc"],
                m["per_video"]["n_videos_both_classes"],
                float(np.mean(aps)) if aps else float("nan")))
            summary.setdefault(corpus, {})[how] = {
                "within_roc": m["per_video"]["macro_auc"],
                "within_ap": float(np.mean(aps)) if aps else None,
                "n": m["per_video"]["n_videos_both_classes"]}
    with open(os.path.join(OUT_DIR, "stage_t_eval.md"), "w") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(OUT_DIR, "stage_t_eval.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
