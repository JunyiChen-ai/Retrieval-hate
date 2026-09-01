#!/usr/bin/env python3
"""Produce label-isolated score maps and carrier strata for the audit."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/reproduction_baselines"))
sys.path.insert(0, str(ROOT / "scripts/duplex"))

from hate_common import data as hdata  # noqa: E402
import frame_eval_common as fec  # noqa: E402
from omnivtg_protocol import (  # noqa: E402
    interval_score, inverse_mapped_interval_score,
    load_corrupted_rows, load_raw_rows,
)


CORPORA = ("hatemm", "hateclipseg")
RUN_ROOT = ROOT / "runs/20260831_omnivtg_reopening_controls/main"
RAW_ROOT = ROOT / "runs/20260831_omnivtg_grounder_diagnostic/formal"
FEATURE_ROOT = ROOT / "results/reproduction/features"
LEXICAL_ROOT = ROOT / "runs/20260831_video_label_lexical_locality/premise"
OCR_SOURCES = {
    "hatemm": ROOT / "data/OCR/HateMM/ocr_windows_K30_test.jsonl",
    "hateclipseg": ROOT / "data/OCR/HateClipSeg/ocr_windows_K30.jsonl",
}
SCOPED_TRAIN_LABELS = {
    corpus: ROOT / "results/reproduction/splits/scoped_labels" / f"{corpus}_train.json"
    for corpus in CORPORA
}
MAX_DONORS = 16


def load_lexical_speech(corpus: str) -> dict[str, np.ndarray]:
    path = LEXICAL_ROOT / corpus / "scores.jsonl"
    rows = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = str(row["video_id"])
            if video_id in rows:
                raise RuntimeError(f"duplicate lexical row: {video_id}")
            values = np.asarray(row["score_speech"], dtype=np.float64)
            if values.ndim != 1 or not np.isin(values, [0.0, 1.0]).all():
                raise RuntimeError(f"invalid speech availability: {corpus}/{video_id}")
            rows[video_id] = values
    return rows


def load_ocr_source_ids(path: Path) -> set[str]:
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = row.get("video_id")
            if not isinstance(video_id, str) or not video_id:
                raise RuntimeError(f"invalid OCR source row at {path}:{line_number}")
            ids.add(video_id)
    return ids


def visual_change_stat(features: np.ndarray) -> float:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or x.shape[0] < 2:
        raise RuntimeError("visual static statistic requires T>=2 matrix")
    norms = np.linalg.norm(x, axis=1)
    if not np.isfinite(x).all() or not np.isfinite(norms).all() or (norms <= 0).any():
        raise RuntimeError("visual static statistic encountered nonfinite/zero-norm row")
    normalized = x / norms[:, None]
    distances = 1.0 - np.sum(normalized[1:] * normalized[:-1], axis=1)
    return float(np.median(distances))


def load_scoped_train_labels(corpus: str) -> dict[str, int]:
    payload = json.loads(SCOPED_TRAIN_LABELS[corpus].read_text())
    if payload.get("corpus") != corpus or payload.get("split") != "train":
        raise RuntimeError(f"{corpus}: scoped train-label identity mismatch")
    labels = {str(key): int(value) for key, value in payload["labels"].items()}
    train_ids = hdata.load_split(corpus, "train")
    if set(labels) != set(train_ids) or set(labels.values()) != {0, 1}:
        raise RuntimeError(f"{corpus}: scoped train labels do not exactly cover split")
    return labels


def train_static_threshold(corpus: str) -> tuple[float, int]:
    labels = load_scoped_train_labels(corpus)
    train_ids = hdata.load_split(corpus, "train")
    positive_ids = [video_id for video_id in train_ids if labels.get(video_id) == 1]
    if not positive_ids:
        raise RuntimeError(f"{corpus}: no positive train videos")
    values = []
    for video_id in positive_ids:
        path = FEATURE_ROOT / "clip_b16_1fps" / corpus / f"{video_id}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing positive-train visual feature: {path}")
        values.append(visual_change_stat(np.load(path, mmap_mode="r")))
    return float(np.median(values)), len(values)


def coverage_band(value: float) -> str:
    if value <= 1.0 / 3.0:
        return "sparse"
    if value < 2.0 / 3.0:
        return "mixed"
    return "dense"


def donor_indices(size: int) -> list[int]:
    if size <= 0:
        return []
    values = np.linspace(0, size - 1, num=min(MAX_DONORS, size), dtype=int)
    return sorted(set(int(value) for value in values))


def rescale_score(score: np.ndarray, target_length: int) -> np.ndarray:
    source = np.asarray(score, dtype=np.float64)
    indices = np.minimum(
        (np.arange(target_length, dtype=np.float64) * len(source) / target_length).astype(int),
        len(source) - 1,
    )
    return source[indices]


def produce_corpus(corpus: str) -> dict:
    raw_path = RAW_ROOT / corpus / "predictions.jsonl"
    corrupted_path = RUN_ROOT / corpus / "corrupted_predictions.jsonl"
    raw_rows = load_raw_rows(raw_path, corpus)
    corrupted_rows = load_corrupted_rows(corrupted_path, corpus, raw_rows)
    speech_rows = load_lexical_speech(corpus)
    ocr_source_ids = load_ocr_source_ids(OCR_SOURCES[corpus])
    static_threshold, n_positive_train = train_static_threshold(corpus)
    out_dir = RUN_ROOT / corpus
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_scores = {}
    lengths = {}
    strata = {}
    missing_ocr_upstream = []
    ocr_empty_videos = []
    ocr_nonempty_videos = []
    raw_success_corrupted_failure = []
    for video_id in sorted(raw_rows):
        visual_path = FEATURE_ROOT / "clip_b16_1fps" / corpus / f"{video_id}.npy"
        ocr_path = FEATURE_ROOT / "ocr_bert_1fps" / corpus / f"{video_id}.npy"
        if not visual_path.is_file():
            raise FileNotFoundError(f"missing test visual feature: {visual_path}")
        visual = np.load(visual_path, mmap_mode="r")
        length = int(visual.shape[0])
        if length <= 0:
            raise RuntimeError(f"invalid visual grid: {visual_path}")
        lengths[video_id] = length
        if video_id not in speech_rows or len(speech_rows[video_id]) != length:
            raise RuntimeError(f"ASR availability length/coverage mismatch: {corpus}/{video_id}")
        if video_id not in ocr_source_ids:
            missing_ocr_upstream.append(video_id)
            continue
        if not ocr_path.is_file():
            raise FileNotFoundError(f"OCR upstream exists but feature is missing: {ocr_path}")
        ocr = np.load(ocr_path, mmap_mode="r")
        if ocr.ndim != 2 or ocr.shape[0] != length or not np.isfinite(ocr).all():
            raise RuntimeError(f"invalid OCR feature: {ocr_path}")
        ocr_present = np.linalg.norm(ocr, axis=1) > 0
        if ocr_present.any():
            ocr_nonempty_videos.append(video_id)
        else:
            ocr_empty_videos.append(video_id)
        raw = interval_score(raw_rows[video_id]["interval_seconds"], length, fec)
        corrupted = inverse_mapped_interval_score(
            corrupted_rows[video_id]["interval_seconds"], length,
            corrupted_rows[video_id]["corruption"],
        )
        if raw_rows[video_id]["parse_ok"] and not corrupted_rows[video_id]["parse_ok"]:
            raw_success_corrupted_failure.append(video_id)
        raw_scores[video_id] = raw
        visual_stat = visual_change_stat(visual)
        strata[video_id] = {
            "score_length": length,
            "asr_coverage": float(speech_rows[video_id].mean()),
            "asr_band": coverage_band(float(speech_rows[video_id].mean())),
            "ocr_coverage": float(ocr_present.mean()),
            "ocr_band": coverage_band(float(ocr_present.mean())),
            "visual_change_median": visual_stat,
            "visual_band": "static" if visual_stat <= static_threshold else "dynamic",
            "raw_parse_ok": bool(raw_rows[video_id]["parse_ok"]),
            "corrupted_parse_ok": bool(corrupted_rows[video_id]["parse_ok"]),
            "corruption_n_blocks": int(corrupted_rows[video_id]["corruption"]["n_blocks"]),
        }

    if missing_ocr_upstream:
        raise RuntimeError(
            f"{corpus}: required cohort missing OCR upstream: {missing_ocr_upstream}"
        )
    if set(raw_scores) != set(raw_rows) or set(strata) != set(raw_rows):
        raise RuntimeError(f"{corpus}: producer did not retain exact cohort")

    sorted_ids = sorted(raw_scores)
    with (out_dir / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for video_id in sorted_ids:
            length = lengths[video_id]
            candidates = [other for other in sorted_ids if other != video_id]
            selected = [candidates[index] for index in donor_indices(len(candidates))]
            if not selected:
                raise RuntimeError("donor diagnostic requires at least two videos")
            donor_average = np.mean(
                [rescale_score(raw_scores[other], length) for other in selected], axis=0
            )
            raw = raw_scores[video_id]
            corrupted = inverse_mapped_interval_score(
                corrupted_rows[video_id]["interval_seconds"], length,
                corrupted_rows[video_id]["corruption"],
            )
            row = {
                "video_id": video_id,
                "score_raw": raw.tolist(),
                "score_mean_repeated": np.full(length, float(raw.mean())).tolist(),
                "score_temporal_corruption": corrupted.tolist(),
                "score_donor_average": donor_average.tolist(),
                "donor_ids": selected,
            }
            handle.write(json.dumps(row) + "\n")

    report = {
        "corpus": corpus,
        "cohort_source": "frozen OmniVTG positive-test prediction IDs",
        "reads_test_video_labels": False,
        "reads_frame_or_span_ground_truth": False,
        "n_videos": len(raw_rows),
        "static_threshold_source": "positive train videos only",
        "static_threshold": static_threshold,
        "n_positive_train_for_static_threshold": n_positive_train,
        "missing_ocr_upstream_ids": missing_ocr_upstream,
        "n_ocr_observed_empty_videos": len(ocr_empty_videos),
        "n_ocr_nonempty_videos": len(ocr_nonempty_videos),
        "raw_success_corrupted_failure_ids": raw_success_corrupted_failure,
        "donor_pool_source": "frozen positive test cohort",
        "transductive_diagnostic_only": True,
        "used_for_score_generation_after_old_test_video_label_selection": True,
        "strata": strata,
    }
    (out_dir / "producer_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    config = {
        "corpora": list(CORPORA),
        "raw_statistic": "frozen OmniVTG binary interval",
        "main_position_control": "8-second block half-rotation then inverse map",
        "donor_control_is_transductive_diagnostic_only": True,
        "carrier_bands": {"sparse": "<=1/3", "mixed": ">1/3 and <2/3",
                          "dense": ">=2/3"},
        "test_gt_read_by_producer": False,
    }
    (RUN_ROOT / "controls_config.json").write_text(json.dumps(config, indent=2) + "\n")
    reports = {corpus: produce_corpus(corpus) for corpus in CORPORA}
    (RUN_ROOT / "producer_summary.json").write_text(json.dumps(
        {corpus: {key: value for key, value in report.items() if key != "strata"}
         for corpus, report in reports.items()}, indent=2) + "\n")


if __name__ == "__main__":
    main()
