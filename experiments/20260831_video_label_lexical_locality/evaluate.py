#!/usr/bin/env python
"""Evaluate lexical premise and frozen controls on test GT."""
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
MAX_SHIFTS = 16


def load_scores(path: Path) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            vid = str(row.pop("video_id"))
            if vid in out:
                raise ValueError(f"duplicate score record: {vid}")
            out[vid] = {key: np.asarray(value, dtype=np.float64)
                        for key, value in row.items()}
    return out


def shift_values(length: int) -> list[int]:
    if length <= 1:
        return []
    values = np.linspace(1, length - 1, num=min(MAX_SHIFTS, length - 1),
                         dtype=int)
    return sorted(set(int(x) for x in values if 0 < int(x) < length))


def equal_video_unique_shift(scores: dict[str, np.ndarray],
                             gt: dict[str, np.ndarray],
                             hate_ids: set[str]) -> dict:
    video_means = []
    n_evaluations = 0
    for vid in sorted(scores):
        y = np.asarray(gt[vid])
        if vid not in hate_ids or len(np.unique(y)) != 2:
            continue
        aucs = []
        for offset in shift_values(len(scores[vid])):
            result = evaluate_scores(
                {vid: np.roll(scores[vid], offset)}, {vid: y}, {vid})
            aucs.append(float(result["per_video"]["macro_auc"]))
        if aucs:
            video_means.append(float(np.mean(aucs)))
            n_evaluations += len(aucs)
    if not video_means:
        raise RuntimeError("no eligible videos for circular-shift control")
    return {
        "aggregation": (
            "at most 16 uniformly spaced unique nonzero shifts within video, "
            "then equal videos"),
        "within_video_roc_mean": float(np.mean(video_means)),
        "within_video_roc_min_video_mean": float(np.min(video_means)),
        "within_video_roc_max_video_mean": float(np.max(video_means)),
        "n_videos": len(video_means),
        "n_video_shift_evaluations": n_evaluations,
    }


def exact_coverage(result: dict) -> bool:
    return bool(result["n_videos_missing_from_scores"] == 0 and
                result["n_videos_not_in_gold"] == 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", required=True)
    args = ap.parse_args()
    run_dir = Path(args.run_dir).resolve()
    final = {"developmental_test_evidence": True, "corpora": {}}
    for corpus in CORPORA:
        records = load_scores(run_dir / corpus / "scores.jsonl")
        gt = hdata.gt_arrays(corpus, "test")
        labels = hdata.load_labels(corpus)
        hate_ids = {vid for vid in gt if labels.get(vid) == 1}
        if set(records) != set(gt):
            raise ValueError(f"{corpus}: scores do not exactly cover test GT")
        lexical_scores = {v: row["score_lexical"] for v, row in records.items()}
        speech_scores = {v: row["score_speech"] for v, row in records.items()}
        lexical = evaluate_scores(lexical_scores, gt, hate_ids)
        speech = evaluate_scores(speech_scores, gt, hate_ids)
        shifted = equal_video_unique_shift(lexical_scores, gt, hate_ids)
        lexical_within = float(lexical["per_video"]["macro_auc"])
        speech_within = float(speech["per_video"]["macro_auc"])
        shift_within = float(shifted["within_video_roc_mean"])
        gate = {
            "lexical_within_at_least_052": lexical_within >= 0.52,
            "lexical_minus_speech_within": lexical_within - speech_within,
            "lexical_minus_speech_at_least_010":
                lexical_within - speech_within >= 0.010,
            "lexical_minus_shift_within": lexical_within - shift_within,
            "lexical_minus_shift_at_least_020":
                lexical_within - shift_within >= 0.020,
            "exact_coverage": exact_coverage(lexical) and exact_coverage(speech),
        }
        gate["pass"] = bool(
            gate["lexical_within_at_least_052"] and
            gate["lexical_minus_speech_at_least_010"] and
            gate["lexical_minus_shift_at_least_020"] and
            gate["exact_coverage"])
        final["corpora"][corpus] = {
            "lexical": lexical,
            "speech_presence_control": speech,
            "circular_shift_control": shifted,
            "gate": gate,
        }
        print(corpus, "AP", lexical["pr_auc"], "ROC", lexical["roc_auc"],
              "within", lexical_within, "speech", speech_within,
              "shift", shift_within, "PASS", gate["pass"], flush=True)
    final["joint_pass"] = bool(
        all(row["gate"]["pass"] for row in final["corpora"].values()))
    final["decision"] = (
        "PROCEED_TO_NOVELTY" if final["joint_pass"] else "STOP_DIRECTION")
    (run_dir / "metrics.json").write_text(json.dumps(final, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
