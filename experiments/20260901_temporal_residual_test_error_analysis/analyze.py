#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "reproduction_baselines"))
sys.path.insert(0, str(ROOT / "scripts" / "duplex"))
from hate_common import data as hdata  # noqa: E402
from frame_eval_common import evaluate, rank_roc_auc  # noqa: E402

RUN = ROOT / "runs/20260901_temporal_residual_reconcilement/formal_val_selected_seed234"
OUT = ROOT / "runs/20260901_temporal_residual_test_error_analysis/main/metrics.json"
CORPORA = ("hatemm", "hateclipseg")
ARMS = ("cyclic_control", "temporal_residual")
BRANCHES = ("visual", "audio", "text", "fused")


def load_scores(path):
    out = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            video_id = row.pop("video_id")
            out[video_id] = {
                key.removeprefix("score_"): np.asarray(value, float)
                for key, value in row.items()}
    return out


def finite_spearman(a, b):
    value = spearmanr(a, b).statistic
    return float(value) if np.isfinite(value) else None


def main():
    payload = {"split": "test", "test_predictions_and_gt_used": True,
               "test_labels_used_for_training_or_selection": False,
               "corpora": {}}
    for corpus in CORPORA:
        gold = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        positive_ids = {video_id for video_id in gold if labels[video_id] == 1}
        scores = {arm: load_scores(RUN / corpus / arm / "scores.jsonl")
                  for arm in ARMS}
        corpus_out = {"arms": {}, "core_vs_control": {}}
        for arm in ARMS:
            arm_out = {"branches": {}}
            for branch in BRANCHES:
                per_video = {video_id: (scores[arm][video_id][branch], target)
                             for video_id, target in gold.items()}
                arm_out["branches"][branch] = evaluate(
                    per_video, macro_over=positive_ids)
            logits = {branch: [] for branch in BRANCHES}
            for video_id in gold:
                for branch in BRANCHES:
                    prob = np.clip(scores[arm][video_id][branch], 1e-6, 1 - 1e-6)
                    logits[branch].append(np.log(prob / (1 - prob)))
            arm_out["absolute_logit"] = {
                branch: {
                    "mean": float(np.mean(np.abs(np.concatenate(values)))),
                    "p95": float(np.quantile(np.abs(np.concatenate(values)), .95)),
                    "p99": float(np.quantile(np.abs(np.concatenate(values)), .99)),
                } for branch, values in logits.items()
            }
            corpus_out["arms"][arm] = arm_out

        per_video_delta, per_video_corr = {}, {}
        for video_id in sorted(positive_ids):
            y = gold[video_id]
            if len(np.unique(y)) < 2:
                continue
            control = scores["cyclic_control"][video_id]["fused"]
            core = scores["temporal_residual"][video_id]["fused"]
            per_video_delta[video_id] = rank_roc_auc(core, y) - rank_roc_auc(control, y)
            per_video_corr[video_id] = finite_spearman(core, control)
        deltas = np.asarray(list(per_video_delta.values()))
        correlations = [value for value in per_video_corr.values() if value is not None]
        corpus_out["core_vs_control"] = {
            "mean_within_delta": float(deltas.mean()),
            "fraction_videos_improved": float((deltas > 0).mean()),
            "fraction_videos_delta_ge_020": float((deltas >= .020).mean()),
            "fraction_videos_delta_le_minus020": float((deltas <= -.020).mean()),
            "mean_within_score_spearman": float(np.mean(correlations)),
            "per_video_within_delta": per_video_delta,
        }
        payload["corpora"][corpus] = corpus_out
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(OUT)


if __name__ == "__main__":
    main()
