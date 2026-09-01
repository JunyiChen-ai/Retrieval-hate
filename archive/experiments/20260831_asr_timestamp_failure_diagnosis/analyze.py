#!/usr/bin/env python3
"""Relate frozen Qwen test localization errors to ASR timestamp fallback mode."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts/reproduction_baselines"))

from hate_common import data as hdata  # noqa: E402


CONFIG = {
    "hatemm": {
        "predictions": REPO / "runs/20260831_qwen3_test_teacher_diagnostic/formal/hatemm/predictions.jsonl",
        "asr": REPO / "data/ASR/HateMM/test_seen_asrK4_whisper-large-v3.jsonl",
    },
    "hateclipseg": {
        "predictions": REPO / "runs/20260831_qwen3_test_teacher_diagnostic/formal/hateclipseg/predictions.jsonl",
        "asr": REPO / "data/ASR/HateClipSeg/test_seen_asrK4_whisper-large-v3.jsonl",
    },
}
RUN_DIR = REPO / "runs/20260831_asr_timestamp_failure_diagnosis/main"


def load_rows(path: Path, id_key: str) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = str(row[id_key])
            if video_id in rows:
                raise RuntimeError(f"duplicate row: {video_id}")
            rows[video_id] = row
    return rows


def densify(row: dict) -> np.ndarray:
    length = int(row["length"])
    total = np.zeros(length, dtype=np.float64)
    count = np.zeros(length, dtype=np.float64)
    for window in row["windows"]:
        start, end = (int(value) for value in window["span"])
        score = float(window["parsed_score"]) / 10.0
        if not 0 <= start < end <= length or not np.isfinite(score):
            raise RuntimeError(f"invalid prediction window in {row['video_id']}")
        total[start:end] += score
        count[start:end] += 1.0
    if np.any(count == 0):
        raise RuntimeError(f"uncovered score seconds in {row['video_id']}")
    return total / count


def finite_chunk_lengths(row: dict) -> list[float]:
    values = []
    for chunk in row.get("chunks", []):
        if len(chunk) < 3 or not str(chunk[2]).strip():
            continue
        start = float(chunk[0])
        end = float(chunk[1])
        if np.isfinite(start) and np.isfinite(end) and end >= start:
            values.append(end - start)
    return values


def mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def analyze_corpus(corpus: str, paths: dict[str, Path]) -> dict:
    predictions = load_rows(paths["predictions"], "video_id")
    asr = load_rows(paths["asr"], "id")
    gt = hdata.gt_arrays(corpus, "test")
    eligible = sorted(
        video_id for video_id in predictions
        if video_id in gt and len(np.unique(gt[video_id])) == 2
    )
    rows = []
    for video_id in eligible:
        if video_id not in asr:
            raise RuntimeError(f"missing ASR row: {corpus}/{video_id}")
        score = densify(predictions[video_id])
        gold = np.asarray(gt[video_id], dtype=np.int8)
        if score.shape != gold.shape or not np.isfinite(score).all():
            raise RuntimeError(f"score/GT mismatch: {corpus}/{video_id}")
        lengths = finite_chunk_lengths(asr[video_id])
        if not lengths:
            raise RuntimeError(f"no finite nonempty ASR chunks: {corpus}/{video_id}")
        rows.append({
            "video_id": video_id,
            "timestamp_mode": str(asr[video_id].get("timestamps", "unknown")),
            "within_roc": float(roc_auc_score(gold, score)),
            "max_nonempty_chunk_seconds": float(max(lengths)),
            "median_nonempty_chunk_seconds": float(np.median(lengths)),
            "n_nonempty_chunks": len(lengths),
        })
    modes = sorted(set(row["timestamp_mode"] for row in rows))
    by_mode = {
        mode: {
            "n_videos": sum(row["timestamp_mode"] == mode for row in rows),
            "mean_within_roc": mean_or_none([
                row["within_roc"] for row in rows if row["timestamp_mode"] == mode
            ]),
            "mean_max_chunk_seconds": mean_or_none([
                row["max_nonempty_chunk_seconds"]
                for row in rows if row["timestamp_mode"] == mode
            ]),
        }
        for mode in modes
    }
    correlation = spearmanr(
        [row["max_nonempty_chunk_seconds"] for row in rows],
        [row["within_roc"] for row in rows],
    )
    if "word" not in by_mode or "chunk" not in by_mode:
        word_minus_chunk = None
    else:
        word_minus_chunk = (
            by_mode["word"]["mean_within_roc"] - by_mode["chunk"]["mean_within_roc"]
        )
    return {
        "prediction_path": str(paths["predictions"]),
        "asr_path": str(paths["asr"]),
        "n_eligible_videos": len(rows),
        "coverage_exact": len(rows) == len(eligible),
        "by_timestamp_mode": by_mode,
        "word_minus_chunk_mean_within_roc": word_minus_chunk,
        "within_roc_vs_max_chunk_seconds_spearman": {
            "rho": float(correlation.statistic),
            "pvalue": float(correlation.pvalue),
            "n": len(rows),
        },
        "per_video": rows,
    }


def main() -> None:
    corpora = {corpus: analyze_corpus(corpus, paths) for corpus, paths in CONFIG.items()}
    gates = {}
    for corpus, result in corpora.items():
        difference = result["word_minus_chunk_mean_within_roc"]
        rho = result["within_roc_vs_max_chunk_seconds_spearman"]["rho"]
        gates[corpus] = {
            "word_minus_chunk_at_least_0_03": difference is not None and difference >= 0.03,
            "longer_chunk_negative_correlation": rho < 0.0,
        }
    joint_pass = all(all(values.values()) for values in gates.values())
    output = {
        "date": "2026-08-31",
        "split": "test",
        "analysis_only": True,
        "test_predictions_and_gt_used_for_error_analysis": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "future_test_evidence_status": "iterative/developmental",
        "fixed_gate": {
            "requirements": (
                "both corpora: word-minus-chunk mean within ROC >= .03 and "
                "within ROC vs max chunk duration Spearman rho < 0"
            ),
            "per_corpus": gates,
            "joint_pass": joint_pass,
            "verdict": "GO_UNIFIED_WORD_ALIGNMENT_PREMISE" if joint_pass else "STOP_AS_COMMON_BOTTLENECK",
        },
        "corpora": corpora,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    path = RUN_DIR / "analysis.json"
    path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(path),
        "verdict": output["fixed_gate"]["verdict"],
        "summary": {
            corpus: {
                "n": result["n_eligible_videos"],
                "by_mode": result["by_timestamp_mode"],
                "word_minus_chunk": result["word_minus_chunk_mean_within_roc"],
                "rho": result["within_roc_vs_max_chunk_seconds_spearman"]["rho"],
            }
            for corpus, result in corpora.items()
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

