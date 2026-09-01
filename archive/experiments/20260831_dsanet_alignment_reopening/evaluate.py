#!/usr/bin/env python3
"""Evaluate DSANet alignment reopening controls on the fixed test sets."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / "scripts/duplex"))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
import frame_eval_common as fec  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")
RUN_ROOT = ROOT / "runs/20260831_dsanet_alignment_reopening/main"
MARGIN = 0.020
MIN_STRATUM = 5
MAX_SHIFTS = 16


def load_controls(path: Path) -> dict[str, dict[str, np.ndarray]]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = str(row.pop("video_id"))
            if video_id in rows:
                raise RuntimeError(f"duplicate control row: {video_id}")
            rows[video_id] = {
                key: np.asarray(value, dtype=np.float64) for key, value in row.items()
            }
    return rows


def shift_offsets(length: int) -> list[int]:
    if length <= 1:
        return []
    values = np.linspace(1, length - 1, num=min(MAX_SHIFTS, length - 1), dtype=int)
    return sorted(set(int(value) for value in values if 0 < int(value) < length))


def per_video_rows(records: dict[str, dict[str, np.ndarray]],
                   gt: dict[str, np.ndarray], hate_ids: set[str]) -> dict[str, dict]:
    output = {}
    for video_id in sorted(records):
        raw = records[video_id]["score_raw"]
        target = gt[video_id]
        raw_auc = fec.rank_roc_auc(raw, target) if video_id in hate_ids else None
        if raw_auc is None:
            output[video_id] = {"eligible": False, "raw_auc": None,
                                "shift_auc": None, "margin": None,
                                "n_shifts": 0, "reverse_spearman": None}
            continue
        shifted = [fec.rank_roc_auc(np.roll(raw, offset), target)
                   for offset in shift_offsets(len(raw))]
        if not shifted or any(value is None for value in shifted):
            raise RuntimeError(f"invalid shift control: {video_id}")
        correlation = spearmanr(raw, records[video_id]["score_reverse_inverse"]).statistic
        correlation = float(correlation) if np.isfinite(correlation) else None
        shifted_mean = float(np.mean(shifted))
        output[video_id] = {
            "eligible": True, "raw_auc": float(raw_auc),
            "shift_auc": shifted_mean, "margin": float(raw_auc - shifted_mean),
            "n_shifts": len(shifted), "reverse_spearman": correlation,
        }
    return output


def aggregate(video_ids: list[str], per_video: dict[str, dict]) -> dict:
    eligible = [video_id for video_id in video_ids if per_video[video_id]["eligible"]]
    raw = ([per_video[video_id]["raw_auc"] for video_id in eligible])
    shifted = ([per_video[video_id]["shift_auc"] for video_id in eligible])
    raw_mean = float(np.mean(raw)) if raw else None
    shift_mean = float(np.mean(shifted)) if shifted else None
    margin = (float(raw_mean - shift_mean)
              if raw_mean is not None and shift_mean is not None else None)
    return {
        "n_videos_total": len(video_ids),
        "n_videos_both_classes": len(eligible),
        "video_ids_both_classes": eligible,
        "raw_macro_auc": raw_mean,
        "shift_macro_auc": shift_mean,
        "raw_minus_shift": margin,
        "n_shift_evaluations": int(sum(per_video[v]["n_shifts"] for v in eligible)),
        "minimum_n_pass": len(eligible) >= MIN_STRATUM,
        "margin_pass": margin is not None and margin >= MARGIN,
        "pass": bool(len(eligible) >= MIN_STRATUM
                     and margin is not None and margin >= MARGIN),
    }


def evaluate_corpus(corpus: str) -> dict:
    out_dir = RUN_ROOT / corpus
    records = load_controls(out_dir / "controls.jsonl")
    report = json.loads((out_dir / "control_report.json").read_text())
    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    hate_ids = {video_id for video_id in gt if labels.get(video_id) == 1}
    if set(records) != set(gt):
        raise RuntimeError(f"{corpus}: controls do not exactly cover test GT")
    branches = {}
    for branch in (
        "score_raw", "score_mean_repeated", "score_position_only",
        "score_reverse_inverse",
    ):
        score_map = {video_id: records[video_id][branch] for video_id in records}
        for video_id, score in score_map.items():
            if score.shape != gt[video_id].shape or not np.isfinite(score).all():
                raise RuntimeError(f"{corpus}/{video_id}: invalid {branch}")
        branches[branch] = evaluate_scores(score_map, gt, hate_ids)
        if (branches[branch]["n_videos_missing_from_scores"]
                or branches[branch]["n_videos_not_in_gold"]):
            raise RuntimeError("canonical evaluator coverage mismatch")
    diagnostics = per_video_rows(records, gt, hate_ids)
    overall = aggregate(sorted(records), diagnostics)
    raw_within = float(branches["score_raw"]["per_video"]["macro_auc"])
    if abs(raw_within - float(overall["raw_macro_auc"])) > 1e-12:
        raise RuntimeError("raw within disagreement with canonical evaluator")
    mean_within = float(branches["score_mean_repeated"]["per_video"]["macro_auc"])
    position_within = float(branches["score_position_only"]["per_video"]["macro_auc"])
    reverse_within = float(branches["score_reverse_inverse"]["per_video"]["macro_auc"])
    correlations = [row["reverse_spearman"] for row in diagnostics.values()
                    if row["eligible"] and row["reverse_spearman"] is not None]
    n_eligible = sum(row["eligible"] for row in diagnostics.values())
    correlation_summary = {
        "n_eligible": int(n_eligible), "n_defined": len(correlations),
        "n_undefined": int(n_eligible - len(correlations)),
        "equal_video_median": float(np.median(correlations)) if correlations else None,
    }
    strata = {}
    for carrier, values in {
        "asr": ("sparse", "mixed", "dense"),
        "ocr": ("sparse", "mixed", "dense"),
        "visual": ("static", "dynamic"),
    }.items():
        key = f"{carrier}_band"
        strata[carrier] = {}
        for value in values:
            ids = sorted(video_id for video_id in records
                         if report["strata"][video_id][key] == value)
            strata[carrier][value] = aggregate(ids, diagnostics)
    gates = {
        "time_shift_margin": overall["raw_minus_shift"],
        "time_shift_pass": bool(overall["raw_minus_shift"] is not None
                                and overall["raw_minus_shift"] >= MARGIN),
        "mean_repeated_margin": raw_within - mean_within,
        "mean_repeated_pass": raw_within - mean_within >= MARGIN,
        "position_only_margin": raw_within - position_within,
        "position_only_pass": raw_within - position_within >= MARGIN,
        "reverse_within_drop": raw_within - reverse_within,
        "reverse_within_pass": reverse_within >= raw_within - MARGIN,
        "reverse_spearman_pass": bool(
            correlation_summary["n_defined"] == correlation_summary["n_eligible"]
            and correlation_summary["equal_video_median"] is not None
            and correlation_summary["equal_video_median"] >= 0.50
        ),
        "all_carrier_strata_pass": all(
            row["pass"] for carrier in strata.values() for row in carrier.values()
        ),
        "integrity_pass": bool(
            not report["missing_ocr_upstream_ids"] and set(records) == set(gt)
        ),
    }
    gates["pass"] = bool(all([
        gates["time_shift_pass"], gates["mean_repeated_pass"],
        gates["position_only_pass"], gates["reverse_within_pass"],
        gates["reverse_spearman_pass"], gates["all_carrier_strata_pass"],
        gates["integrity_pass"],
    ]))
    return {
        "corpus": corpus, "split": "test",
        "evidence_status": "iterative/developmental reopening premise",
        "test_gt_used_for_gradient_or_checkpoint_selection": False,
        "metrics": branches, "time_shift": overall,
        "reverse_correlation": correlation_summary,
        "carrier_strata": strata, "per_video": diagnostics, "gates": gates,
    }


def main() -> None:
    corpora = {corpus: evaluate_corpus(corpus) for corpus in CORPORA}
    joint_pass = all(row["gates"]["pass"] for row in corpora.values())
    payload = {
        "split": "test",
        "evidence_status": "iterative/developmental reopening premise",
        "corpora": corpora, "joint_pass": joint_pass,
        "verdict": ("REOPENING_EVIDENCE_PASS_ONLY" if joint_pass
                    else "KEEP_CANDIDATE_FREEZE"),
    }
    (RUN_ROOT / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps({"verdict": payload["verdict"],
                      "gates": {c: r["gates"] for c, r in corpora.items()}}, indent=2))


if __name__ == "__main__":
    main()
