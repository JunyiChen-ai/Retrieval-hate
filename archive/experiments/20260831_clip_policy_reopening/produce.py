#!/usr/bin/env python3
"""Produce frozen CLIP policy scores and label-blind controls."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import CLIPModel, CLIPTokenizer

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / "src"))

from hate_common import data as hdata  # noqa: E402
from lexical_locality import load_asr, local_texts  # noqa: E402
from scoped_video_protocol import evaluator_test_ids, scoped_video_labels  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")
RUN_ROOT = ROOT / "runs/20260831_clip_policy_reopening/main"
FEATURE_ROOT = ROOT / "results/reproduction/features"
ASR_PATHS = {
    corpus: ROOT / "results/reproduction/asr" / f"{corpus}_all" /
    "timestamped_chunks.jsonl" for corpus in CORPORA
}
MODEL_ID = "openai/clip-vit-base-patch16"
N_POSITION_BINS = 20
POLICY_PROMPTS = (
    "a hateful or degrading image targeting a person or group",
    "an image displaying abusive, insulting, or threatening content",
    "an image showing violence, sexual abuse, or encouragement of harm",
    "an image containing offensive symbols, gestures, or text",
)
BENIGN_PROMPTS = (
    "an ordinary benign image of people or everyday life",
    "a respectful or neutral image without abuse",
    "a peaceful safe scene without violence or threats",
    "an image without offensive symbols, gestures, or text",
)


def unit_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise RuntimeError("invalid embedding matrix")
    norms = np.linalg.norm(values, axis=1)
    if not np.isfinite(values).all() or (norms <= 0).any():
        raise RuntimeError("invalid embedding matrix")
    return values / norms[:, None]


def text_direction() -> np.ndarray:
    tokenizer = CLIPTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
    model = CLIPModel.from_pretrained(MODEL_ID, local_files_only=True)
    model.eval()
    prompts = list(POLICY_PROMPTS + BENIGN_PROMPTS)
    batch = tokenizer(prompts, padding=True, truncation=True, return_tensors="pt")
    with torch.inference_mode():
        features = model.get_text_features(**batch).detach().cpu().numpy()
    features = unit_rows(features)
    n_policy = len(POLICY_PROMPTS)
    policy = features[:n_policy].mean(axis=0)
    benign = features[n_policy:].mean(axis=0)
    policy /= np.linalg.norm(policy)
    benign /= np.linalg.norm(benign)
    direction = policy - benign
    if direction.shape != (512,) or not np.isfinite(direction).all():
        raise RuntimeError("invalid CLIP text direction")
    return direction


def score_video(corpus: str, video_id: str, direction: np.ndarray) -> np.ndarray:
    path = Path(hdata.feature_path(corpus, video_id))
    if not path.is_file():
        raise FileNotFoundError(path)
    image = unit_rows(np.load(path, mmap_mode="r"))
    score = image @ direction
    if score.ndim != 1 or len(score) == 0 or not np.isfinite(score).all():
        raise RuntimeError(f"invalid score: {corpus}/{video_id}")
    return score


def position_template(scores: dict[str, np.ndarray]) -> np.ndarray:
    bins = [[] for _ in range(N_POSITION_BINS)]
    for video_id in sorted(scores):
        score = scores[video_id]
        index = np.minimum(N_POSITION_BINS * np.arange(len(score)) // len(score),
                           N_POSITION_BINS - 1)
        for bin_index in range(N_POSITION_BINS):
            selected = score[index == bin_index]
            if selected.size:
                bins[bin_index].append(float(selected.mean()))
    if any(not values for values in bins):
        raise RuntimeError("empty positive-train position bin")
    return np.asarray([np.mean(values) for values in bins], dtype=np.float64)


def apply_template(template: np.ndarray, length: int) -> np.ndarray:
    index = np.minimum(len(template) * np.arange(length) // length,
                       len(template) - 1)
    return template[index]


def visual_change(features: np.ndarray) -> float:
    x = unit_rows(features)
    if len(x) < 2:
        return 0.0
    return float(np.median(1.0 - np.sum(x[1:] * x[:-1], axis=1)))


def carrier_stats(corpus: str, video_id: str, length: int,
                  asr: dict[str, dict]) -> dict[str, float]:
    if video_id not in asr:
        raise RuntimeError(f"missing ASR row: {corpus}/{video_id}")
    _, speech = local_texts(asr[video_id]["chunks"], length, 2.0, 3.0)
    ocr_path = FEATURE_ROOT / "ocr_bert_1fps" / corpus / f"{video_id}.npy"
    visual_path = Path(hdata.feature_path(corpus, video_id))
    if not ocr_path.is_file():
        raise FileNotFoundError(ocr_path)
    ocr = np.load(ocr_path, mmap_mode="r")
    visual = np.load(visual_path, mmap_mode="r")
    if (ocr.ndim != 2 or len(ocr) != length or not np.isfinite(ocr).all()
            or visual.ndim != 2 or len(visual) != length):
        raise RuntimeError(f"carrier grid mismatch: {corpus}/{video_id}")
    return {
        "asr_coverage": float(speech.mean()),
        "ocr_coverage": float((np.linalg.norm(ocr, axis=1) > 0).mean()),
        "visual_change": visual_change(visual),
    }


def produce_corpus(corpus: str, direction: np.ndarray) -> dict:
    train_ids = hdata.load_split(corpus, "train")
    labels = scoped_video_labels(corpus, "train", train_ids)
    positive_train = sorted(v for v in train_ids if labels[v] == 1)
    test_ids = evaluator_test_ids(corpus, hdata.load_split(corpus, "test"))
    asr, asr_filtering = load_asr(ASR_PATHS[corpus])
    expected_asr = set(positive_train) | set(test_ids)
    if not expected_asr.issubset(asr):
        raise RuntimeError(f"{corpus}: incomplete ASR coverage")

    train_scores = {v: score_video(corpus, v, direction) for v in positive_train}
    template = position_template(train_scores)
    train_carriers = {
        v: carrier_stats(corpus, v, len(train_scores[v]), asr)
        for v in positive_train
    }
    thresholds = {
        key: float(np.median([row[key] for row in train_carriers.values()]))
        for key in ("asr_coverage", "ocr_coverage", "visual_change")
    }

    out_dir = RUN_ROOT / corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    strata = {}
    with (out_dir / "controls.jsonl").open("w", encoding="utf-8") as handle:
        for video_id in test_ids:
            raw = score_video(corpus, video_id, direction)
            carrier = carrier_stats(corpus, video_id, len(raw), asr)
            strata[video_id] = {
                **carrier,
                "asr_band": "low" if carrier["asr_coverage"] <= thresholds["asr_coverage"] else "high",
                "ocr_band": "low" if carrier["ocr_coverage"] <= thresholds["ocr_coverage"] else "high",
                "visual_band": "low" if carrier["visual_change"] <= thresholds["visual_change"] else "high",
            }
            row = {
                "video_id": video_id,
                "score_raw": raw.tolist(),
                "score_mean_repeated": np.full(len(raw), float(raw.mean())).tolist(),
                "score_position_only": apply_template(template, len(raw)).tolist(),
            }
            handle.write(json.dumps(row) + "\n")
    report = {
        "corpus": corpus,
        "n_positive_train": len(positive_train),
        "n_test": len(test_ids),
        "position_bins": N_POSITION_BINS,
        "position_template": template.tolist(),
        "carrier_threshold_source": "positive-train median only",
        "carrier_thresholds": thresholds,
        "asr_filtering": asr_filtering,
        "test_labels_read_by_producer": False,
        "frame_or_span_gt_read_by_producer": False,
        "strata": strata,
    }
    (out_dir / "producer_report.json").write_text(json.dumps(report, indent=2) + "\n")
    return {key: value for key, value in report.items() if key != "strata"}


def main() -> None:
    RUN_ROOT.mkdir(parents=True, exist_ok=True)
    config = {
        "corpora": list(CORPORA),
        "model": MODEL_ID,
        "policy_prompts": list(POLICY_PROMPTS),
        "benign_prompts": list(BENIGN_PROMPTS),
        "statistic": "cosine(image, unit(mean(policy))-unit(mean(benign)))",
        "training": "none; positive train labels only build controls",
        "test_labels_or_temporal_gt_read_by_producer": False,
    }
    (RUN_ROOT / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    direction = text_direction()
    reports = {corpus: produce_corpus(corpus, direction) for corpus in CORPORA}
    (RUN_ROOT / "producer_summary.json").write_text(json.dumps(reports, indent=2) + "\n")


if __name__ == "__main__":
    main()
