"""Paired-by-video bootstrap intervals for ablation differences (full - arm).

For each corpus, seed and arm: resample test videos with replacement (the
same resample for full and arm), recompute pooled frame AP / ROC on the
resampled pool, and report the 95% interval of the difference; the three
seeds are combined by averaging the difference over seeds inside each
resample (same video resample for all seeds). Reads the evaluator inputs
(scores_test.jsonl) and the fixed GT; no frame metric is redefined.

    python scripts/analysis/paired_bootstrap_ablations.py \
        --search-root runs/<exp>/<corpus> --ablation-root runs/<exp>/ablations/<corpus> \
        --corpus <corpus> --arms a b c --out runs/<exp>/ablations/<corpus>/paired_bootstrap.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
from hate_common import data as hdata          # noqa: E402

SEEDS = (234, 2025, 3407)


def read_scores(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["video_id"]] = np.asarray(r["score_av"], dtype=np.float64)
    return out


def pooled(scores, gt, vids):
    y = np.concatenate([np.asarray(gt[v]) for v in vids])
    s = np.concatenate([scores[v][:len(gt[v])] for v in vids])
    return average_precision_score(y, s), roc_auc_score(y, s)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--search-root", required=True, help="runs/<exp>/<corpus>")
    ap.add_argument("--ablation-root", required=True, help="runs/<exp>/ablations/<corpus>")
    ap.add_argument("--arms", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    gt = hdata.gt_arrays(a.corpus, "test")
    full, arms = {}, {arm: {} for arm in a.arms}
    for s in a.seeds:
        summ = json.load(open(os.path.join(a.search_root, "seed%d" % s, "study_summary.json")))
        b = summ["best"]["number"]
        full[s] = read_scores(os.path.join(a.search_root, "seed%d" % s, "trial%d" % b, "scores_test.jsonl"))
        for arm in a.arms:
            arms[arm][s] = read_scores(os.path.join(a.ablation_root, "seed%d" % s, arm, "scores_test.jsonl"))
    vids = sorted(v for v in gt if all(v in full[s] for s in a.seeds))
    rng = np.random.RandomState(0)
    boots = [rng.choice(len(vids), len(vids), replace=True) for _ in range(a.n_boot)]
    report = {"corpus": a.corpus, "seeds": list(a.seeds), "n_videos": len(vids),
              "n_boot": a.n_boot, "arms": {}}
    point_full = {s: pooled(full[s], gt, vids) for s in a.seeds}
    for arm in a.arms:
        diffs = {"pooled_ap": [], "pooled_roc": []}
        per_seed = {s: {"pooled_ap": [], "pooled_roc": []} for s in a.seeds}
        for idx in boots:
            sub = [vids[i] for i in idx]
            d_ap, d_roc = [], []
            for s in a.seeds:
                f_ap, f_roc = pooled(full[s], gt, sub)
                m_ap, m_roc = pooled(arms[arm][s], gt, sub)
                d_ap.append(f_ap - m_ap)
                d_roc.append(f_roc - m_roc)
                per_seed[s]["pooled_ap"].append(f_ap - m_ap)
                per_seed[s]["pooled_roc"].append(f_roc - m_roc)
            diffs["pooled_ap"].append(np.mean(d_ap))
            diffs["pooled_roc"].append(np.mean(d_roc))
        pt = {k: float(np.mean([point_full[s][i] - pooled(arms[arm][s], gt, vids)[i] for s in a.seeds]))
              for i, k in enumerate(("pooled_ap", "pooled_roc"))}
        report["arms"][arm] = {
            "mean_drop_point": pt,
            "ci95_three_seed_mean": {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                     for k, v in diffs.items()},
            "ci95_by_seed": {str(s): {k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                      for k, v in per_seed[s].items()} for s in a.seeds},
            "p_drop_positive": {k: float(np.mean(np.asarray(v) > 0)) for k, v in diffs.items()}}
        print("%-14s dAP %+.3f [%+.3f, %+.3f]  dROC %+.3f [%+.3f, %+.3f]" % (
            arm, pt["pooled_ap"], *report["arms"][arm]["ci95_three_seed_mean"]["pooled_ap"],
            pt["pooled_roc"], *report["arms"][arm]["ci95_three_seed_mean"]["pooled_roc"]))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(report, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
