#!/usr/bin/env python
"""Evaluate frozen probe scores and circular-shift controls on test GT."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "reproduction_baselines"))
from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402

CORPORA = ("hatemm", "hateclipseg")
MAX_SHIFTS = 32


def shift_values(length: int) -> list[int]:
    if length <= 1:
        return []
    values = np.linspace(1, length - 1, num=min(MAX_SHIFTS, length - 1), dtype=int)
    return sorted(set(int(x) for x in values if 0 < int(x) < length))


def summarize_controls(rows: list[dict]) -> dict:
    out = {"n_controls": len(rows)}
    for key in ("pr_auc", "roc_auc", "within_video_roc"):
        vals = np.asarray([r[key] for r in rows], dtype=float)
        out[key] = {
            "mean": float(vals.mean()),
            "min": float(vals.min()),
            "max": float(vals.max()),
        }
    return out


def equal_video_shift_summary(scores: dict[str, np.ndarray],
                              gt: dict[str, np.ndarray],
                              hate_ids: set[str]) -> dict:
    """Average unique nonzero shifts within video, then average videos.

    This keeps a short video from giving extra weight to the first few shifts
    merely because a 32-arm corpus-level schedule has to repeat its smaller
    set of possible integer offsets.  Per-video AUC still comes exclusively
    from the shared evaluator.
    """
    video_means = []
    n_evaluations = 0
    for vid in sorted(scores):
        y = np.asarray(gt[vid])
        if vid not in hate_ids or len(np.unique(y)) != 2:
            continue
        values = []
        for shift in shift_values(len(scores[vid])):
            res = evaluate_scores(
                {vid: np.roll(scores[vid], shift)}, {vid: y}, {vid})
            values.append(float(res["per_video"]["macro_auc"]))
        if values:
            video_means.append(float(np.mean(values)))
            n_evaluations += len(values)
    if not video_means:
        raise RuntimeError("no both-class videos available for shift control")
    return {
        "aggregation": "equal unique shifts within video, then equal videos",
        "mean": float(np.mean(video_means)),
        "min_video_mean": float(np.min(video_means)),
        "max_video_mean": float(np.max(video_means)),
        "n_videos": len(video_means),
        "n_video_shift_evaluations": n_evaluations,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    final = {"developmental_test_evidence": True, "corpora": {}}
    for corpus in CORPORA:
        with np.load(run_dir / f"{corpus}_scores.npz") as z:
            scores = {v: np.asarray(z[v], dtype=float) for v in z.files}
        gt = hdata.gt_arrays(corpus, "test")
        hate_ids = {v for v, y in gt.items() if np.asarray(y).max() > 0}
        original = evaluate_scores(scores, gt, hate_ids)
        controls = []
        for j in range(MAX_SHIFTS):
            shifted = {}
            for vid, s in scores.items():
                choices = shift_values(len(s))
                # Spread any necessary repetitions evenly across a short
                # video's unique shifts instead of overweighting early shifts.
                q = min(len(choices) - 1, j * len(choices) // MAX_SHIFTS)
                k = choices[q] if choices else 0
                shifted[vid] = np.roll(s, k)
            res = evaluate_scores(shifted, gt, hate_ids)
            controls.append({
                "control_index": j,
                "pr_auc": res["pr_auc"],
                "roc_auc": res["roc_auc"],
                "within_video_roc": res["per_video"]["macro_auc"],
            })
        control_summary = summarize_controls(controls)
        equal_shift = equal_video_shift_summary(scores, gt, hate_ids)
        control_summary["within_video_equal_shift"] = equal_shift
        within = float(original["per_video"]["macro_auc"])
        shift_mean = float(equal_shift["mean"])
        gate = {
            "within_above_052": within > 0.52,
            "within_minus_shift_mean": within - shift_mean,
            "alignment_gain_at_least_020": within - shift_mean >= 0.020,
            "exact_score_coverage": (
                original["n_videos_missing_from_scores"] == 0
                and original["n_videos_not_in_gold"] == 0
            ),
        }
        gate["pass"] = bool(all([
            gate["within_above_052"], gate["alignment_gain_at_least_020"],
            gate["exact_score_coverage"],
        ]))
        final["corpora"][corpus] = {
            "original": original,
            "circular_shift_summary": control_summary,
            "circular_shift_controls": controls,
            "gate": gate,
        }
        print(corpus, "AP", original["pr_auc"], "ROC", original["roc_auc"],
              "within", within, "shift_mean", shift_mean, "PASS", gate["pass"], flush=True)
    final["joint_pass"] = bool(all(x["gate"]["pass"] for x in final["corpora"].values()))
    final["decision"] = "PROCEED_TO_NOVELTY" if final["joint_pass"] else "STOP_DIRECTION"
    (run_dir / "metrics.json").write_text(json.dumps(final, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
