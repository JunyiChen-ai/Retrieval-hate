#!/usr/bin/env python3
"""Evaluate frozen OmniVTG reopening controls on developmental test GT."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/reproduction_baselines"))
sys.path.insert(0, str(ROOT / "scripts/duplex"))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
import frame_eval_common as fec  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")
RUN_ROOT = ROOT / "runs/20260831_omnivtg_reopening_controls/main"
MAX_SHIFTS = 16
MIN_STRATUM_VIDEOS = 5
MARGIN = 0.020


def load_score_rows(path: Path) -> dict[str, dict[str, np.ndarray | list[str]]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = str(row.pop("video_id"))
            if video_id in rows:
                raise RuntimeError(f"duplicate score row: {video_id}")
            donors = list(row.pop("donor_ids"))
            parsed = {key: np.asarray(value, dtype=np.float64)
                      for key, value in row.items()}
            parsed["donor_ids"] = donors
            rows[video_id] = parsed
    return rows


def shift_values(length: int) -> list[int]:
    if length <= 1:
        return []
    values = np.linspace(1, length - 1, num=min(MAX_SHIFTS, length - 1), dtype=int)
    return sorted(set(int(value) for value in values if 0 < int(value) < length))


def per_video_diagnostics(scores: dict[str, np.ndarray],
                          gt: dict[str, np.ndarray]) -> dict[str, dict]:
    diagnostics = {}
    for video_id in sorted(scores):
        raw_auc = fec.rank_roc_auc(scores[video_id], gt[video_id])
        if raw_auc is None:
            diagnostics[video_id] = {
                "eligible_both_classes": False, "raw_auc": None,
                "shift_mean_auc": None, "raw_minus_shift": None,
                "n_shift_evaluations": 0,
            }
            continue
        shifted = [
            fec.rank_roc_auc(np.roll(scores[video_id], offset), gt[video_id])
            for offset in shift_values(len(scores[video_id]))
        ]
        if not shifted or any(value is None for value in shifted):
            raise RuntimeError(f"invalid shift evaluation: {video_id}")
        shift_mean = float(np.mean(shifted))
        diagnostics[video_id] = {
            "eligible_both_classes": True,
            "raw_auc": float(raw_auc),
            "shift_mean_auc": shift_mean,
            "raw_minus_shift": float(raw_auc - shift_mean),
            "n_shift_evaluations": len(shifted),
        }
    return diagnostics


def aggregate_ids(ids: list[str], per_video: dict[str, dict]) -> dict:
    eligible = [video_id for video_id in ids
                if per_video[video_id]["eligible_both_classes"]]
    raw_values = [per_video[video_id]["raw_auc"] for video_id in eligible]
    shift_values_ = [per_video[video_id]["shift_mean_auc"] for video_id in eligible]
    raw = float(np.mean(raw_values)) if raw_values else None
    shifted = float(np.mean(shift_values_)) if shift_values_ else None
    margin = (float(raw - shifted) if raw is not None and shifted is not None else None)
    return {
        "n_videos_total": len(ids),
        "n_videos_both_classes": len(eligible),
        "video_ids_both_classes": eligible,
        "raw_macro_auc": raw,
        "shift_macro_auc": shifted,
        "raw_minus_shift": margin,
        "n_shift_evaluations": int(sum(
            per_video[video_id]["n_shift_evaluations"] for video_id in eligible
        )),
        "minimum_n_pass": len(eligible) >= MIN_STRATUM_VIDEOS,
        "margin_pass": margin is not None and margin >= MARGIN,
        "pass": bool(len(eligible) >= MIN_STRATUM_VIDEOS
                     and margin is not None and margin >= MARGIN),
    }


def evaluate_corpus(corpus: str) -> dict:
    out_dir = RUN_ROOT / corpus
    rows = load_score_rows(out_dir / "scores.jsonl")
    report = json.loads((out_dir / "producer_report.json").read_text())
    gt_full = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    expected = {video_id for video_id in gt_full if labels.get(video_id) == 1}
    if set(rows) != expected:
        raise RuntimeError(f"{corpus}: score cohort is not exact positive test GT cohort")
    gt = {video_id: gt_full[video_id] for video_id in sorted(expected)}
    branch_names = (
        "score_raw", "score_mean_repeated", "score_temporal_corruption",
        "score_donor_average",
    )
    branches = {}
    for branch in branch_names:
        score_map = {video_id: rows[video_id][branch] for video_id in rows}
        for video_id, score in score_map.items():
            if score.shape != gt[video_id].shape or not np.isfinite(score).all():
                raise RuntimeError(f"{corpus}/{video_id}: invalid {branch} score")
        branches[branch] = evaluate_scores(score_map, gt, expected)
        if (branches[branch]["n_videos_missing_from_scores"] != 0
                or branches[branch]["n_videos_not_in_gold"] != 0):
            raise RuntimeError(f"{corpus}: canonical evaluator coverage failure")

    raw_map = {video_id: rows[video_id]["score_raw"] for video_id in rows}
    per_video = per_video_diagnostics(raw_map, gt)
    aggregate = aggregate_ids(sorted(rows), per_video)
    raw_within = float(branches["score_raw"]["per_video"]["macro_auc"])
    mean_within = float(branches["score_mean_repeated"]["per_video"]["macro_auc"])
    corruption_within = float(
        branches["score_temporal_corruption"]["per_video"]["macro_auc"]
    )
    if abs(raw_within - float(aggregate["raw_macro_auc"])) > 1e-12:
        raise RuntimeError("aggregate raw macro disagrees with canonical evaluator")

    strata = {}
    for carrier, values in {
        "asr": ("sparse", "mixed", "dense"),
        "ocr": ("sparse", "mixed", "dense"),
        "visual": ("static", "dynamic"),
    }.items():
        key = f"{carrier}_band"
        strata[carrier] = {}
        for value in values:
            ids = sorted(video_id for video_id in rows
                         if report["strata"][video_id][key] == value)
            strata[carrier][value] = aggregate_ids(ids, per_video)

    raw_success_corrupted_failure = list(
        report["raw_success_corrupted_failure_ids"]
    )
    integrity = {
        "exact_positive_test_cohort": set(rows) == expected,
        "all_score_lengths_and_finite": True,
        "missing_ocr_upstream_ids": report["missing_ocr_upstream_ids"],
        "no_missing_ocr_upstream": not report["missing_ocr_upstream_ids"],
        "raw_success_corrupted_failure_ids": raw_success_corrupted_failure,
        "no_raw_success_corrupted_failure": not raw_success_corrupted_failure,
    }
    gates = {
        "aggregate_time_shift_margin": aggregate["raw_minus_shift"],
        "aggregate_time_shift_pass": bool(
            aggregate["raw_minus_shift"] is not None
            and aggregate["raw_minus_shift"] >= MARGIN
        ),
        "mean_repeated_margin": raw_within - mean_within,
        "mean_repeated_pass": raw_within - mean_within >= MARGIN,
        "temporal_corruption_margin": raw_within - corruption_within,
        "temporal_corruption_pass": raw_within - corruption_within >= MARGIN,
        "all_carrier_strata_pass": all(
            row["pass"] for carrier in strata.values() for row in carrier.values()
        ),
        "integrity_pass": bool(all([
            integrity["exact_positive_test_cohort"],
            integrity["all_score_lengths_and_finite"],
            integrity["no_missing_ocr_upstream"],
            integrity["no_raw_success_corrupted_failure"],
        ])),
    }
    gates["pass"] = bool(all([
        gates["aggregate_time_shift_pass"], gates["mean_repeated_pass"],
        gates["temporal_corruption_pass"], gates["all_carrier_strata_pass"],
        gates["integrity_pass"],
    ]))
    return {
        "corpus": corpus,
        "split": "test",
        "evidence_status": "iterative/developmental identifiability audit",
        "test_gt_used_for_gradient_or_checkpoint_selection": False,
        "fixed_metrics_positive_test_cohort": branches,
        "aggregate_time_shift": aggregate,
        "carrier_strata": strata,
        "per_video": per_video,
        "integrity": integrity,
        "gates": gates,
    }


def main() -> None:
    corpora = {corpus: evaluate_corpus(corpus) for corpus in CORPORA}
    joint_pass = all(row["gates"]["pass"] for row in corpora.values())
    payload = {
        "split": "test",
        "evidence_status": "iterative/developmental identifiability audit",
        "old_omnivtg_stop_before_student_remains_effective": True,
        "corpora": corpora,
        "joint_pass": joint_pass,
        "verdict": ("REOPENING_EVIDENCE_PASS_ONLY" if joint_pass
                    else "KEEP_CANDIDATE_FREEZE"),
    }
    (RUN_ROOT / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({
        "verdict": payload["verdict"],
        "corpora": {corpus: row["gates"] for corpus, row in corpora.items()},
    }, indent=2))


if __name__ == "__main__":
    main()
