#!/usr/bin/env python3
"""Evaluate the frozen CLIP policy reopening controls on test GT."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / "scripts/duplex"))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
import frame_eval_common as fec  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")
RUN_ROOT = ROOT / "runs/20260831_clip_policy_reopening/main"
MARGIN = 0.020
MAX_SHIFTS = 16
MIN_STRATUM = 5


def load_records(path: Path) -> dict[str, dict[str, np.ndarray]]:
    records = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = str(row.pop("video_id"))
            if video_id in records:
                raise RuntimeError(f"duplicate score row: {video_id}")
            records[video_id] = {
                key: np.asarray(value, dtype=np.float64) for key, value in row.items()
            }
    return records


def shift_offsets(length: int) -> list[int]:
    if length <= 1:
        return []
    values = np.linspace(1, length - 1, num=min(MAX_SHIFTS, length - 1), dtype=int)
    return sorted(set(int(value) for value in values if 0 < int(value) < length))


def diagnostics(records: dict[str, dict[str, np.ndarray]], gt: dict[str, np.ndarray],
                positive_ids: set[str]) -> dict[str, dict]:
    output = {}
    for video_id in sorted(records):
        raw = records[video_id]["score_raw"]
        target = gt[video_id]
        raw_auc = fec.rank_roc_auc(raw, target) if video_id in positive_ids else None
        if raw_auc is None:
            output[video_id] = {"eligible": False, "raw_auc": None,
                                "shift_auc": None, "margin": None, "n_shifts": 0}
            continue
        shifted = [fec.rank_roc_auc(np.roll(raw, offset), target)
                   for offset in shift_offsets(len(raw))]
        if not shifted or any(value is None for value in shifted):
            raise RuntimeError(f"invalid shift control: {video_id}")
        shift_auc = float(np.mean(shifted))
        output[video_id] = {"eligible": True, "raw_auc": float(raw_auc),
                            "shift_auc": shift_auc,
                            "margin": float(raw_auc - shift_auc),
                            "n_shifts": len(shifted)}
    return output


def aggregate(video_ids: list[str], rows: dict[str, dict]) -> dict:
    eligible = [video_id for video_id in video_ids if rows[video_id]["eligible"]]
    raw = [rows[v]["raw_auc"] for v in eligible]
    shift = [rows[v]["shift_auc"] for v in eligible]
    margin = float(np.mean(raw) - np.mean(shift)) if raw else None
    return {
        "n_videos_total": len(video_ids),
        "n_videos_both_classes": len(eligible),
        "raw_macro_auc": float(np.mean(raw)) if raw else None,
        "shift_macro_auc": float(np.mean(shift)) if shift else None,
        "raw_minus_shift": margin,
        "n_shift_evaluations": int(sum(rows[v]["n_shifts"] for v in eligible)),
        "minimum_n_pass": len(eligible) >= MIN_STRATUM,
        "margin_pass": margin is not None and margin >= MARGIN,
        "pass": bool(len(eligible) >= MIN_STRATUM and margin is not None and margin >= MARGIN),
    }


def evaluate_corpus(corpus: str) -> dict:
    out_dir = RUN_ROOT / corpus
    records = load_records(out_dir / "controls.jsonl")
    report = json.loads((out_dir / "producer_report.json").read_text())
    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    positive_ids = {video_id for video_id in gt if labels.get(video_id) == 1}
    if set(records) != set(gt) or set(report["strata"]) != set(gt):
        raise RuntimeError(f"{corpus}: exact test cohort mismatch")
    metrics = {}
    for branch in ("score_raw", "score_mean_repeated", "score_position_only"):
        scores = {video_id: records[video_id][branch] for video_id in records}
        for video_id, score in scores.items():
            if score.shape != gt[video_id].shape or not np.isfinite(score).all():
                raise RuntimeError(f"invalid {branch}: {corpus}/{video_id}")
        metrics[branch] = evaluate_scores(scores, gt, positive_ids)
        if (metrics[branch]["n_videos_missing_from_scores"]
                or metrics[branch]["n_videos_not_in_gold"]):
            raise RuntimeError("canonical evaluator coverage mismatch")

    rows = diagnostics(records, gt, positive_ids)
    overall = aggregate(sorted(records), rows)
    raw_within = float(metrics["score_raw"]["per_video"]["macro_auc"])
    mean_within = float(metrics["score_mean_repeated"]["per_video"]["macro_auc"])
    position_within = float(metrics["score_position_only"]["per_video"]["macro_auc"])
    if abs(raw_within - float(overall["raw_macro_auc"])) > 1e-12:
        raise RuntimeError("canonical/raw diagnostic disagreement")

    strata = {}
    for carrier in ("asr", "ocr", "visual"):
        strata[carrier] = {}
        for band in ("low", "high"):
            ids = sorted(video_id for video_id in records
                         if report["strata"][video_id][f"{carrier}_band"] == band)
            strata[carrier][band] = aggregate(ids, rows)
    gates = {
        "time_shift_margin": overall["raw_minus_shift"],
        "time_shift_pass": bool(overall["pass"]),
        "mean_repeated_margin": raw_within - mean_within,
        "mean_repeated_pass": raw_within - mean_within >= MARGIN,
        "position_only_margin": raw_within - position_within,
        "position_only_pass": raw_within - position_within >= MARGIN,
        "all_carrier_strata_pass": all(
            value["pass"] for carrier in strata.values() for value in carrier.values()
        ),
        "integrity_pass": True,
    }
    gates["pass"] = bool(
        gates["time_shift_pass"] and gates["mean_repeated_pass"]
        and gates["position_only_pass"] and gates["all_carrier_strata_pass"]
        and gates["integrity_pass"]
    )
    return {
        "corpus": corpus,
        "split": "test",
        "evidence_status": "iterative/developmental reopening premise",
        "test_gt_used_for_gradient_or_checkpoint_selection": False,
        "metrics": metrics,
        "time_shift": overall,
        "carrier_strata": strata,
        "per_video": rows,
        "gates": gates,
    }


def main() -> None:
    corpora = {corpus: evaluate_corpus(corpus) for corpus in CORPORA}
    joint_pass = all(row["gates"]["pass"] for row in corpora.values())
    payload = {
        "split": "test",
        "evidence_status": "iterative/developmental reopening premise",
        "corpora": corpora,
        "joint_pass": joint_pass,
        "verdict": "REOPENING_EVIDENCE_PASS_ONLY" if joint_pass else "KEEP_CANDIDATE_FREEZE",
    }
    (RUN_ROOT / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"verdict": payload["verdict"],
                      "gates": {c: r["gates"] for c, r in corpora.items()}}, indent=2))


if __name__ == "__main__":
    main()
