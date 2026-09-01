#!/usr/bin/env python3
"""Test-only structural premise for ASR utterance-boundary alignment."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO / "scripts/reproduction_baselines"))

from hate_common import data as hdata  # noqa: E402


CORPORA = {
    "hatemm": REPO / "data/ASR/HateMM/test_seen_asrK4_whisper-large-v3.jsonl",
    "hateclipseg": REPO / "data/ASR/HateClipSeg/test_seen_asrK4_whisper-large-v3.jsonl",
}
N_SHIFTS = 30
RECALL_RADII = (1.0, 2.0, 4.0)
MAX_MERGE_GAP_SECONDS = 0.8
TERMINAL_PUNCTUATION = (".", "?", "!", "。", "？", "！")
RUN_DIR = REPO / "runs/20260831_utterance_boundary_premise/main"


def load_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = str(row["id"])
            if video_id in rows:
                raise RuntimeError(f"duplicate ASR row: {video_id}")
            rows[video_id] = row
    return rows


def linear_nearest(query: np.ndarray, references: np.ndarray) -> np.ndarray:
    delta = np.abs(query[:, None] - references[None, :])
    return np.min(delta, axis=1)


def internal_utterance_boundaries(row: dict, duration: float) -> np.ndarray:
    entries = []
    for chunk in row.get("chunks", []):
        if len(chunk) < 3 or not str(chunk[2]).strip():
            continue
        raw_start = float(chunk[0])
        raw_end = float(chunk[1])
        if not np.isfinite(raw_start) or not np.isfinite(raw_end):
            continue
        if raw_end < raw_start or raw_end < 0.0 or raw_start > duration:
            continue
        start = min(duration, max(0.0, raw_start))
        end = min(duration, max(0.0, raw_end))
        entries.append((start, end, str(chunk[2]).strip()))
    entries.sort(key=lambda item: (item[0], item[1]))
    utterances = []
    for start, end, text in entries:
        if not utterances:
            utterances.append([start, end, text])
            continue
        previous = utterances[-1]
        gap = start - float(previous[1])
        previous_terminal = str(previous[2]).rstrip().endswith(TERMINAL_PUNCTUATION)
        if gap < MAX_MERGE_GAP_SECONDS and not previous_terminal:
            previous[1] = max(float(previous[1]), end)
            previous[2] = f"{previous[2]} {text}".strip()
        else:
            utterances.append([start, end, text])
    points = []
    for start, end, _ in utterances:
        if 0.0 < start < duration:
            points.append(start)
        if 0.0 < end < duration:
            points.append(end)
    if not points:
        return np.empty(0, dtype=float)
    return np.unique(np.asarray(points, dtype=float))


def self_check() -> None:
    point_punctuation = {
        "chunks": [
            [0.1, 0.5, "hello"],
            [0.5, 0.5, "."],
            [0.6, 1.0, "next"],
        ]
    }
    actual = internal_utterance_boundaries(point_punctuation, 2.0)
    expected = np.asarray([0.1, 0.5, 0.6, 1.0])
    if not np.allclose(actual, expected):
        raise RuntimeError(
            f"zero-duration punctuation grouping self-check failed: {actual.tolist()}"
        )


def macro_mean(rows: list[dict], key: str) -> float | None:
    values = [float(row[key]) for row in rows if row.get(key) is not None]
    return float(np.mean(values)) if values else None


def analyze_corpus(corpus: str, asr_path: Path) -> dict:
    gt = hdata.gt_arrays(corpus, "test")
    labels = hdata.load_labels(corpus)
    asr = load_jsonl(asr_path)
    eligible_ids = sorted(
        video_id
        for video_id, gold in gt.items()
        if labels.get(video_id) == 1 and len(np.unique(gold)) == 2
    )

    rows = []
    missing_asr = []
    no_boundary = []
    for video_id in eligible_ids:
        if video_id not in asr:
            missing_asr.append(video_id)
            continue
        gold = np.asarray(gt[video_id], dtype=np.int8)
        duration = float(len(gold))
        transition = np.flatnonzero(gold[1:] != gold[:-1]).astype(float) + 1.0
        boundaries = internal_utterance_boundaries(asr[video_id], duration)
        if not bool(asr[video_id].get("audio_ok", False)) or len(boundaries) == 0:
            no_boundary.append(video_id)
            continue

        observed_distance = linear_nearest(transition, boundaries)
        shifted_distances = []
        for shift_index in range(1, N_SHIFTS + 1):
            offset = duration * shift_index / (N_SHIFTS + 1)
            shifted = np.mod(boundaries + offset, duration)
            shifted_distances.append(linear_nearest(transition, shifted))
        shifted_distances = np.stack(shifted_distances, axis=0)

        row = {
            "video_id": video_id,
            "duration_seconds": int(len(gold)),
            "asr_metadata_duration_seconds": float(asr[video_id].get("duration", duration)),
            "asr_minus_gt_duration_seconds": float(
                asr[video_id].get("duration", duration)
            ) - duration,
            "source_timestamp_granularity": str(asr[video_id].get("timestamps", "unknown")),
            "n_gt_transitions": int(len(transition)),
            "n_grouped_utterance_internal_boundaries": int(len(boundaries)),
            "observed_mean_nearest_seconds": float(observed_distance.mean()),
            "shift_mean_nearest_seconds": float(shifted_distances.mean(axis=1).mean()),
        }
        for radius in RECALL_RADII:
            suffix = str(int(radius))
            row[f"observed_recall_at_{suffix}s"] = float(np.mean(observed_distance <= radius))
            row[f"shift_recall_at_{suffix}s"] = float(
                np.mean(shifted_distances <= radius, axis=1).mean()
            )
        rows.append(row)

    summary = {
        "n_test_videos": len(gt),
        "n_eligible_positive_videos": len(eligible_ids),
        "n_analyzed_videos": len(rows),
        "n_missing_asr": len(missing_asr),
        "n_audio_failed_or_no_internal_boundary": len(no_boundary),
        "observed_mean_nearest_seconds_macro": macro_mean(
            rows, "observed_mean_nearest_seconds"
        ),
        "shift_mean_nearest_seconds_macro": macro_mean(
            rows, "shift_mean_nearest_seconds"
        ),
    }
    for radius in RECALL_RADII:
        suffix = str(int(radius))
        observed = macro_mean(rows, f"observed_recall_at_{suffix}s")
        shifted = macro_mean(rows, f"shift_recall_at_{suffix}s")
        summary[f"observed_recall_at_{suffix}s_macro"] = observed
        summary[f"shift_recall_at_{suffix}s_macro"] = shifted
        summary[f"recall_gain_at_{suffix}s"] = (
            None if observed is None or shifted is None else observed - shifted
        )
    return {
        "asr_path": str(asr_path),
        "coverage": summary,
        "per_video": rows,
    }


def main() -> None:
    self_check()
    results = {corpus: analyze_corpus(corpus, path) for corpus, path in CORPORA.items()}
    gates = {}
    for corpus, result in results.items():
        summary = result["coverage"]
        gates[corpus] = {
            "observed_distance_better_than_shift": (
                summary["observed_mean_nearest_seconds_macro"]
                < summary["shift_mean_nearest_seconds_macro"]
            ),
            "recall_gain_at_2s_at_least_0_05": summary["recall_gain_at_2s"] >= 0.05,
        }
    joint_pass = all(all(values.values()) for values in gates.values())
    output = {
        "date": "2026-08-31",
        "split": "test",
        "analysis_only": True,
        "test_predictions_used": False,
        "test_gt_used_for_error_analysis": True,
        "test_labels_used_for_gradient_or_checkpoint_selection": False,
        "future_test_evidence_status": "iterative/developmental",
        "control": {
            "kind": "modulo shift of complete ASR boundary set with linear nearest distance",
            "n_nonzero_shifts": N_SHIFTS,
            "aggregation": "equal transitions within video, then equal videos",
        },
        "utterance_grouping": {
            "merge_condition": (
                "previous text has no terminal punctuation and next-start minus "
                "previous-end is less than 0.8 seconds"
            ),
            "terminal_punctuation": list(TERMINAL_PUNCTUATION),
            "max_merge_gap_seconds_exclusive": MAX_MERGE_GAP_SECONDS,
            "time_domain": "GT 1fps interval [0,T); timestamps outside are clipped/dropped",
        },
        "fixed_gate": {
            "requirements": (
                "both corpora: observed mean distance < shift mean and "
                "recall@2s gain >= 0.05"
            ),
            "per_corpus": gates,
            "joint_pass": joint_pass,
            "verdict": "GO_NOVELTY_REVIEW" if joint_pass else "STOP_DIRECTION",
        },
        "corpora": results,
    }
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    output_path = RUN_DIR / "analysis.json"
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print(json.dumps({
        "output": str(output_path),
        "verdict": output["fixed_gate"]["verdict"],
        "summary": {corpus: result["coverage"] for corpus, result in results.items()},
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
