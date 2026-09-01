#!/usr/bin/env python3
"""Read-only TEST error analysis over compliant corpus-specific POWA scores."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASELINES = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASELINES))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402


SOURCES = {
    "hatemm": Path(
        "/home/jehc223/Hate-follow-up/results/reproduction/powa_macil/"
        "final_maskfix_finetune_hatemm_seed234_e5/hatemm/scores.jsonl"
    ),
    "mhclip_en": Path(
        "/home/jehc223/Hate-follow-up/results/reproduction/powa_macil/"
        "final_maskfix_finetune_mhclip_en_seed234_e5/mhclip_en/scores.jsonl"
    ),
    "mhclip_zh": Path(
        "/home/jehc223/Hate-follow-up/results/reproduction/powa_macil/"
        "final_maskfix_frozen_positive_mhclip_zh_seed234_e5/"
        "mhclip_zh/scores.jsonl"
    ),
    "hateclipseg": REPO / (
        "runs/20260831_powa_starting_point/hcs_maskfix_seed234/scores.jsonl"
    ),
}
WINDOWS = (3, 5, 9, 15, 31)


def smooth(values, width):
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 1:
        return values.copy()
    width = min(int(width), len(values))
    left = width // 2
    right = width - 1 - left
    padded = np.pad(values, (left, right), mode="edge")
    return np.convolve(padded, np.ones(width) / width, mode="valid")


def metric_summary(report):
    return {
        "pooled_ap": report["pr_auc"],
        "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


def strata(report, gt):
    per_video = report["per_video"]["per_video_auc"]
    groups = {
        "pos_fraction_le_0.2": [],
        "pos_fraction_0.2_to_0.6": [],
        "pos_fraction_gt_0.6": [],
    }
    for video_id, auc in per_video.items():
        fraction = float(np.asarray(gt[video_id]).mean())
        if fraction <= 0.2:
            key = "pos_fraction_le_0.2"
        elif fraction <= 0.6:
            key = "pos_fraction_0.2_to_0.6"
        else:
            key = "pos_fraction_gt_0.6"
        groups[key].append(float(auc))
    return {
        key: {"n": len(values), "within_roc": (
            float(np.mean(values)) if values else None
        )}
        for key, values in groups.items()
    }


def main():
    payload = {
        "date": "2026-08-31",
        "split": "test",
        "seed": 234,
        "purpose": "error analysis informing later method development",
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "future_test_evidence_status": "iterative/developmental",
        "shared_evaluator": str(
            (BASELINES / "eval_baseline_scores.py").resolve()
        ),
        "corpora": {},
    }
    for corpus, source in SOURCES.items():
        rows = [json.loads(line) for line in source.read_text().splitlines()]
        scores = {
            branch: {
                row["video_id"]: np.asarray(row[branch], dtype=np.float64)
                for row in rows
            }
            for branch in ("score_powa", "score_base", "score_audio", "score_visual")
        }
        probes = dict(scores)
        probes["policy_delta_powa_minus_base"] = {
            video_id: scores["score_powa"][video_id] - scores["score_base"][video_id]
            for video_id in scores["score_powa"]
        }
        probes["policy_ratio_powa_over_base"] = {
            video_id: scores["score_powa"][video_id]
            / (scores["score_base"][video_id] + 1e-4)
            for video_id in scores["score_powa"]
        }
        for width in WINDOWS:
            probes[f"powa_smooth_{width}"] = {
                video_id: smooth(values, width)
                for video_id, values in scores["score_powa"].items()
            }
            probes[f"powa_innovation_{width}"] = {
                video_id: values - smooth(values, width)
                for video_id, values in scores["score_powa"].items()
            }
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        positive_ids = {
            video_id for video_id in gt if labels.get(video_id) == 1
        }
        reports = {
            name: evaluate_scores(branch, gt, positive_ids)
            for name, branch in probes.items()
        }
        payload["corpora"][corpus] = {
            "prediction_artifact": str(source.resolve()),
            "probes": {
                name: {
                    "metrics": metric_summary(report),
                    "strata": strata(report, gt),
                }
                for name, report in reports.items()
            },
        }
    out = REPO / "runs/20260831_powa_error_structure/analysis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    temporary = out.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(out)
    print(out)


if __name__ == "__main__":
    main()
