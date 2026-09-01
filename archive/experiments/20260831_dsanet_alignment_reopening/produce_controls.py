#!/usr/bin/env python3
"""Build position, mean and carrier controls without reading test labels/GT."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / "src"))

from hate_common import data as hdata  # noqa: E402
from scoped_video_protocol import evaluator_test_ids, scoped_video_labels  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")
RUN_ROOT = ROOT / "runs/20260831_dsanet_alignment_reopening/main"
FEATURE_ROOT = ROOT / "results/reproduction/features"
LEXICAL_ROOT = ROOT / "runs/20260831_video_label_lexical_locality/premise"
OCR_SOURCES = {
    "hatemm": ROOT / "data/OCR/HateMM/ocr_windows_K30_test.jsonl",
    "hateclipseg": ROOT / "data/OCR/HateClipSeg/ocr_windows_K30.jsonl",
}
N_POSITION_BINS = 20


def load_produced(path: Path, corpus: str) -> tuple[dict[str, np.ndarray], dict[str, dict]]:
    train, test = {}, {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            if row.get("corpus") != corpus or row.get("score_branch") != "score_align":
                raise RuntimeError(f"producer row identity mismatch at {path}:{line_number}")
            video_id = str(row["video_id"])
            raw = np.asarray(row["score_raw"], dtype=np.float64)
            if raw.ndim != 1 or not np.isfinite(raw).all() or len(raw) != row["n_frames"]:
                raise RuntimeError(f"invalid raw score at {path}:{line_number}")
            if row["split"] == "train":
                if video_id in train or row["score_reverse_inverse"] is not None:
                    raise RuntimeError("invalid/duplicate train score row")
                train[video_id] = raw
            elif row["split"] == "test":
                if video_id in test:
                    raise RuntimeError("duplicate test score row")
                reverse = np.asarray(row["score_reverse_inverse"], dtype=np.float64)
                if reverse.shape != raw.shape or not np.isfinite(reverse).all():
                    raise RuntimeError("invalid reverse score row")
                test[video_id] = {"raw": raw, "reverse_inverse": reverse}
            else:
                raise RuntimeError("unsupported producer split")
    return train, test


def position_template(train_scores: dict[str, np.ndarray]) -> np.ndarray:
    per_bin = [[] for _ in range(N_POSITION_BINS)]
    for video_id in sorted(train_scores):
        score = train_scores[video_id]
        bins = np.minimum(
            (N_POSITION_BINS * np.arange(len(score)) // len(score)).astype(int),
            N_POSITION_BINS - 1,
        )
        for bin_index in range(N_POSITION_BINS):
            selected = score[bins == bin_index]
            if selected.size:
                per_bin[bin_index].append(float(selected.mean()))
    if any(not values for values in per_bin):
        raise RuntimeError("positive-train position template has an empty bin")
    template = np.asarray([np.mean(values) for values in per_bin], dtype=np.float64)
    if not np.isfinite(template).all():
        raise RuntimeError("position template is nonfinite")
    return template


def apply_position_template(template: np.ndarray, length: int) -> np.ndarray:
    bins = np.minimum(
        (len(template) * np.arange(length) // length).astype(int), len(template) - 1
    )
    return template[bins]


def load_speech(corpus: str) -> dict[str, np.ndarray]:
    rows = {}
    path = LEXICAL_ROOT / corpus / "scores.jsonl"
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            values = np.asarray(row["score_speech"], dtype=np.float64)
            if values.ndim != 1 or not np.isin(values, [0.0, 1.0]).all():
                raise RuntimeError("invalid frozen speech availability")
            rows[str(row["video_id"])] = values
    return rows


def load_ocr_ids(path: Path) -> set[str]:
    ids = set()
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            video_id = row.get("video_id")
            if not isinstance(video_id, str) or not video_id:
                raise RuntimeError("invalid OCR upstream row")
            ids.add(video_id)
    return ids


def visual_change(features: np.ndarray) -> float:
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2 or len(x) < 2 or not np.isfinite(x).all():
        raise RuntimeError("invalid visual sequence for static statistic")
    norms = np.linalg.norm(x, axis=1)
    if (norms <= 0).any() or not np.isfinite(norms).all():
        raise RuntimeError("zero/nonfinite visual row")
    x = x / norms[:, None]
    return float(np.median(1.0 - np.sum(x[1:] * x[:-1], axis=1)))


def coverage_band(value: float) -> str:
    if value <= 1.0 / 3.0:
        return "sparse"
    if value < 2.0 / 3.0:
        return "mixed"
    return "dense"


def produce_corpus(corpus: str) -> dict:
    out_dir = RUN_ROOT / corpus
    train_scores, test_scores = load_produced(out_dir / "scores.jsonl", corpus)
    train_ids = hdata.load_split(corpus, "train")
    train_labels = scoped_video_labels(corpus, "train", train_ids)
    expected_train = {video_id for video_id in train_ids if train_labels[video_id] == 1}
    expected_test = set(evaluator_test_ids(corpus, hdata.load_split(corpus, "test")))
    if set(train_scores) != expected_train or set(test_scores) != expected_test:
        raise RuntimeError(f"{corpus}: produced score cohort mismatch")
    template = position_template(train_scores)
    train_visual = [
        visual_change(np.load(hdata.feature_path(corpus, video_id), mmap_mode="r"))
        for video_id in sorted(expected_train)
    ]
    static_threshold = float(np.median(train_visual))
    speech = load_speech(corpus)
    ocr_upstream = load_ocr_ids(OCR_SOURCES[corpus])
    missing_ocr = sorted(expected_test - ocr_upstream)
    if missing_ocr:
        raise RuntimeError(f"{corpus}: missing OCR upstream: {missing_ocr}")
    strata = {}
    controls_path = out_dir / "controls.jsonl"
    with controls_path.open("w", encoding="utf-8") as handle:
        for video_id in sorted(expected_test):
            raw = test_scores[video_id]["raw"]
            reverse = test_scores[video_id]["reverse_inverse"]
            length = len(raw)
            if video_id not in speech or len(speech[video_id]) != length:
                raise RuntimeError(f"{corpus}/{video_id}: speech length mismatch")
            ocr_path = FEATURE_ROOT / "ocr_bert_1fps" / corpus / f"{video_id}.npy"
            visual_path = Path(hdata.feature_path(corpus, video_id))
            if not ocr_path.is_file() or not visual_path.is_file():
                raise FileNotFoundError(f"missing feature for {corpus}/{video_id}")
            ocr = np.load(ocr_path, mmap_mode="r")
            visual = np.load(visual_path, mmap_mode="r")
            if (ocr.ndim != 2 or len(ocr) != length or not np.isfinite(ocr).all()
                    or visual.ndim != 2 or len(visual) != length):
                raise RuntimeError(f"{corpus}/{video_id}: feature grid mismatch")
            ocr_presence = np.linalg.norm(ocr, axis=1) > 0
            visual_stat = visual_change(visual)
            strata[video_id] = {
                "asr_coverage": float(speech[video_id].mean()),
                "asr_band": coverage_band(float(speech[video_id].mean())),
                "ocr_coverage": float(ocr_presence.mean()),
                "ocr_band": coverage_band(float(ocr_presence.mean())),
                "visual_change_median": visual_stat,
                "visual_band": "static" if visual_stat <= static_threshold else "dynamic",
            }
            row = {
                "video_id": video_id,
                "score_raw": raw.tolist(),
                "score_mean_repeated": np.full(length, float(raw.mean())).tolist(),
                "score_position_only": apply_position_template(template, length).tolist(),
                "score_reverse_inverse": reverse.tolist(),
            }
            handle.write(json.dumps(row) + "\n")
    report = {
        "corpus": corpus,
        "score_branch": "score_align",
        "n_positive_train": len(expected_train),
        "n_test": len(expected_test),
        "position_bins": N_POSITION_BINS,
        "position_template": template.tolist(),
        "static_threshold_source": "positive-train videos only",
        "static_threshold": static_threshold,
        "missing_ocr_upstream_ids": missing_ocr,
        "test_labels_read_by_producer": False,
        "frame_or_span_gt_read_by_producer": False,
        "strata": strata,
    }
    (out_dir / "control_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {key: value for key, value in report.items() if key != "strata"}


def main() -> None:
    reports = {corpus: produce_corpus(corpus) for corpus in CORPORA}
    (RUN_ROOT / "producer_summary.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
