#!/usr/bin/env python3
"""Step-4 diagnosis of POWA-MACIL: within-video localization on TEST.

Computes, per corpus and seed, POWA's pooled / within-hate-macro / video-level
test metrics from the frozen dense-score artifacts, adds a per-hate-video AP,
and assembles the comparison table against every reproduced baseline's
frame_eval.json (all branches). Everything is the 1 fps test split.

Output: runs/20260830_powa_within_diagnosis/{summary.json,summary.md,per_video.csv}
"""
import csv
import glob
import json
import os
import sys

import numpy as np

REPO = "/home/jehc223/Retrieval-hate"
sys.path.insert(0, os.path.join(REPO, "scripts", "reproduction_baselines"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

POWA_ROOT = os.path.join(REPO, "results", "reproduction", "powa_macil")
BASE_ROOT = os.path.join(REPO, "results", "reproduction", "official_val", "final")
OUT_DIR = os.path.join(REPO, "runs", "20260830_powa_within_diagnosis")

SEEDS = (234, 2025, 3407)
POWA_PATTERNS = {
    "hatemm": "final_maskfix_finetune_hatemm_seed{seed}_e5",
    "mhclip_en": "final_maskfix_finetune_mhclip_en_seed{seed}_e5",
    "mhclip_zh": "final_maskfix_frozen_positive_mhclip_zh_seed{seed}_e5",
    "hateclipseg": "final_maskfix_joint_w48_seed{seed}_e5",
}
CORPORA = ("hatemm", "mhclip_en", "mhclip_zh", "hateclipseg")


def load_scores(path, branch="score_powa"):
    scores = {}
    with open(path) as fh:
        for line in fh:
            row = json.loads(line)
            scores[row["video_id"]] = np.asarray(row[branch], dtype=float)
    return scores


def within_ap(scores, gt, hate_ids):
    """Per-video AP over hate videos carrying both classes."""
    per = {}
    for vid in sorted(hate_ids):
        if vid not in scores or vid not in gt:
            continue
        y = np.asarray(gt[vid]).astype(int)
        if y.min() == y.max():
            continue
        per[vid] = float(average_precision_score(y, np.asarray(scores[vid])))
    return per


def powa_block():
    out = {}
    per_video_rows = []
    for corpus in CORPORA:
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        hate_ids = {v for v in gt if labels.get(v) == 1}
        seed_stats = []
        for seed in SEEDS:
            path = os.path.join(POWA_ROOT, POWA_PATTERNS[corpus].format(seed=seed),
                                corpus, "scores.jsonl")
            scores = load_scores(path)
            m = evaluate_scores(scores, gt, hate_ids)
            w_ap = within_ap(scores, gt, hate_ids)
            pv = m["per_video"]["per_video_auc"]
            seed_stats.append({
                "seed": seed,
                "frame_ap": m["pr_auc"], "frame_roc": m["roc_auc"],
                "within_roc_macro": m["per_video"]["macro_auc"],
                "within_ap_macro": float(np.mean(list(w_ap.values()))) if w_ap else None,
                "within_n": m["per_video"]["n_videos_both_classes"],
                "video_roc_max": m["video_level"]["max_roc_auc"],
            })
            for vid in pv:
                y = np.asarray(gt[vid]).astype(int)
                per_video_rows.append({
                    "corpus": corpus, "seed": seed, "video_id": vid,
                    "within_roc": pv[vid], "within_ap": w_ap.get(vid),
                    "duration_s": int(len(y)), "pos_frac": float(y.mean()),
                    "n_pos_seg": int(np.diff(np.pad(y, 1)).clip(0).sum()),
                })
        agg = {}
        for key in ("frame_ap", "frame_roc", "within_roc_macro", "within_ap_macro",
                    "video_roc_max"):
            vals = [s[key] for s in seed_stats if s[key] is not None]
            agg[key] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals))}
        agg["within_n"] = seed_stats[0]["within_n"]
        out[corpus] = {"seeds": seed_stats, "agg": agg}
    return out, per_video_rows


def baseline_block():
    """method -> corpus -> branch -> seed-aggregated test metrics."""
    out = {}
    for fe_path in sorted(glob.glob(os.path.join(BASE_ROOT, "*", "*", "seed_*",
                                                 "frame_eval.json"))):
        parts = fe_path.split(os.sep)
        method, corpus, seed = parts[-4], parts[-3], parts[-2]
        fe = json.load(open(fe_path))
        assert fe.get("split") == "test", fe_path
        for branch, r in fe["results"].items():
            if not isinstance(r, dict) or "roc_auc" not in r:
                continue
            d = out.setdefault(method, {}).setdefault(corpus, {}).setdefault(
                branch, {"frame_ap": [], "frame_roc": [], "within_roc_macro": [],
                         "within_n": r["per_video"]["n_videos_both_classes"]})
            d["frame_ap"].append(r["pr_auc"])
            d["frame_roc"].append(r["roc_auc"])
            d["within_roc_macro"].append(r["per_video"]["macro_auc"])
    for method in out:
        for corpus in out[method]:
            for branch, d in out[method][corpus].items():
                for key in ("frame_ap", "frame_roc", "within_roc_macro"):
                    vals = [v for v in d[key] if v is not None]
                    d[key] = ({"mean": float(np.mean(vals)),
                               "sd": float(np.std(vals)), "n_seeds": len(vals)}
                              if vals else None)
    return out


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    powa, per_video_rows = powa_block()
    base = baseline_block()
    summary = {"split": "test", "grid": "1fps", "powa": powa, "baselines": base}
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=1)
    with open(os.path.join(OUT_DIR, "per_video.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_video_rows[0].keys()))
        w.writeheader()
        w.writerows(per_video_rows)

    lines = ["# POWA within-video diagnosis (TEST, 1 fps)", ""]
    for corpus in CORPORA:
        lines.append(f"## {corpus}")
        a = powa["powa" in powa and corpus][ "agg"] if False else powa[corpus]["agg"]
        lines.append("| method/branch | frame AP | frame ROC | within-ROC macro | n |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append("| **POWA (score_powa)** | %.4f±%.3f | %.4f±%.3f | %.4f±%.3f | %d |" % (
            a["frame_ap"]["mean"], a["frame_ap"]["sd"],
            a["frame_roc"]["mean"], a["frame_roc"]["sd"],
            a["within_roc_macro"]["mean"], a["within_roc_macro"]["sd"],
            a["within_n"]))
        lines.append("| POWA within-AP macro | %.4f±%.3f | | | |" % (
            a["within_ap_macro"]["mean"], a["within_ap_macro"]["sd"]))
        rows = []
        for method in sorted(base):
            for branch, d in base[method].get(corpus, {}).items():
                if d["within_roc_macro"] is None:
                    continue
                rows.append((d["within_roc_macro"]["mean"], method, branch, d))
        for wr, method, branch, d in sorted(rows, reverse=True):
            lines.append("| %s/%s | %.4f±%.3f | %.4f±%.3f | %.4f±%.3f | %d |" % (
                method, branch,
                d["frame_ap"]["mean"], d["frame_ap"]["sd"],
                d["frame_roc"]["mean"], d["frame_roc"]["sd"],
                d["within_roc_macro"]["mean"], d["within_roc_macro"]["sd"],
                d["within_n"]))
        lines.append("")
    with open(os.path.join(OUT_DIR, "summary.md"), "w") as fh:
        fh.write("\n".join(lines))
    print("wrote", OUT_DIR)


if __name__ == "__main__":
    main()
