#!/usr/bin/env python3
"""Produce natural and input-reversal DSANet alignment scores without GT."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(ROOT / "src"))

from dsanet.model import DSANet  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from hate_common import runtime  # noqa: E402
from scoped_video_protocol import evaluator_test_ids, scoped_video_labels  # noqa: E402
from score_diagnostics import load_score_branch  # noqa: E402


CORPORA = ("hatemm", "hateclipseg")
RUN_ROOT = ROOT / "runs/20260831_dsanet_alignment_reopening/main"
OFFICIAL_ROOT = ROOT / "results/reproduction/official_val/final/dsanet"
CONFIG_PATH = Path(__file__).with_name("config.json")


def model_args(config: dict, corpus: str) -> SimpleNamespace:
    values = dict(config["model_common"])
    values.update(config["corpora"][corpus])
    values.update({"corpus": corpus, "device": "cuda", "seed": config["seed"]})
    return SimpleNamespace(**values)


def build_model(config: dict, corpus: str) -> tuple[DSANet, SimpleNamespace]:
    args = model_args(config, corpus)
    model = DSANet(
        args.classes_num, args.embed_dim, args.visual_length,
        args.visual_width, args.visual_head, args.visual_layers,
        args.attn_window, args.prompt_prefix, args.prompt_postfix,
        args, args.device, clip_download_root=None,
    )
    checkpoint = OFFICIAL_ROOT / corpus / "seed_234/model.pth"
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    model.load_state_dict(torch.load(checkpoint, map_location=args.device))
    model.to(args.device).eval()
    return model, args


def alignment_score(model: DSANet, args: SimpleNamespace,
                    features: np.ndarray) -> np.ndarray:
    array = np.asarray(features, dtype=np.float32)
    if array.ndim != 2 or array.shape[0] <= 0 or not np.isfinite(array).all():
        raise RuntimeError("invalid input feature sequence")
    processed, total_length = hdata.tools.process_split(array, args.visual_length)
    visual = torch.from_numpy(np.ascontiguousarray(processed))
    if total_length < args.visual_length:
        visual = visual.unsqueeze(0)
    visual = visual.to(args.device)
    lengths = runtime.chunk_lengths(total_length, args.visual_length).to(args.device)
    with torch.no_grad():
        output = model(
            visual, None, list(hdata.PROMPT_TEXT), lengths, bool(args.DNP_use)
        )
        logits = output[2].reshape(-1, output[2].shape[2])[:total_length]
        score = 1.0 - F.softmax(logits, dim=-1)[:, 0]
    values = score.float().cpu().numpy().astype(np.float64)
    if values.shape != (total_length,) or not np.isfinite(values).all():
        raise RuntimeError("invalid alignment score")
    return values


def load_completed(path: Path, corpus: str,
                   allowed: set[tuple[str, str]]) -> set[tuple[str, str]]:
    completed = set()
    if not path.exists():
        return completed
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            row = json.loads(line)
            key = (str(row.get("split")), str(row.get("video_id")))
            if (
                key not in allowed or key in completed
                or row.get("corpus") != corpus or row.get("seed") != 234
                or row.get("score_branch") != "score_align"
            ):
                raise RuntimeError(f"invalid resumed row at {path}:{line_number}")
            raw = np.asarray(row.get("score_raw"), dtype=np.float64)
            reverse = row.get("score_reverse_inverse")
            if raw.ndim != 1 or not np.isfinite(raw).all():
                raise RuntimeError(f"invalid resumed raw score at {path}:{line_number}")
            if key[0] == "test":
                reverse = np.asarray(reverse, dtype=np.float64)
                if reverse.shape != raw.shape or not np.isfinite(reverse).all():
                    raise RuntimeError(f"invalid resumed reverse score at {path}:{line_number}")
            elif reverse is not None:
                raise RuntimeError("train row unexpectedly carries reverse score")
            completed.add(key)
    return completed


def append(path: Path, row: dict) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", choices=CORPORA, required=True)
    args_cli = parser.parse_args()
    config = json.loads(CONFIG_PATH.read_text())
    corpus = args_cli.corpus
    train_ids = hdata.load_split(corpus, "train")
    train_labels = scoped_video_labels(corpus, "train", train_ids)
    positive_train_ids = sorted(video_id for video_id in train_ids
                                if train_labels[video_id] == 1)
    test_ids = evaluator_test_ids(corpus, hdata.load_split(corpus, "test"))
    official_scores = load_score_branch(
        OFFICIAL_ROOT / corpus / "seed_234/scores.jsonl", "score_align"
    )
    if set(official_scores) != set(test_ids):
        raise RuntimeError(f"{corpus}: official score cohort mismatch")
    ordered = ([('train', video_id) for video_id in positive_train_ids]
               + [('test', video_id) for video_id in test_ids])
    allowed = set(ordered)
    out_dir = RUN_ROOT / corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    output = out_dir / "scores.jsonl"
    completed = load_completed(output, corpus, allowed)
    pending = [key for key in ordered if key not in completed]
    metadata = {
        "corpus": corpus, "seed": config["seed"],
        "checkpoint": str((OFFICIAL_ROOT / corpus / "seed_234/model.pth").resolve()),
        "checkpoint_selection": "validation-selected fixed DSANet method",
        "score_branch": "score_align",
        "n_positive_train": len(positive_train_ids),
        "n_test": len(test_ids),
        "test_labels_read_by_producer": False,
        "frame_or_span_gt_read_by_producer": False,
        "model_parameters": {**config["model_common"], **config["corpora"][corpus]},
    }
    metadata_path = out_dir / "producer_config.json"
    if metadata_path.exists() and json.loads(metadata_path.read_text()) != metadata:
        raise RuntimeError("existing producer config mismatch")
    if not metadata_path.exists():
        metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"event": "cohort", "corpus": corpus,
                      "total": len(ordered), "pending": len(pending)}), flush=True)
    if not pending:
        return
    model, args = build_model(config, corpus)
    for index, (split, video_id) in enumerate(pending, 1):
        features = np.load(hdata.feature_path(corpus, video_id)).astype(np.float32)
        raw = alignment_score(model, args, features)
        reverse_inverse = None
        max_saved_difference = None
        if split == "test":
            if raw.shape != official_scores[video_id].shape:
                raise RuntimeError(f"{corpus}/{video_id}: official score shape mismatch")
            max_saved_difference = float(np.max(np.abs(raw - official_scores[video_id])))
            if max_saved_difference > 2e-5:
                raise RuntimeError(
                    f"{corpus}/{video_id}: checkpoint/config does not reproduce official score"
                )
            reversed_score = alignment_score(model, args, features[::-1].copy())
            reverse_inverse = reversed_score[::-1].copy()
        row = {
            "corpus": corpus, "split": split, "video_id": video_id,
            "seed": config["seed"], "score_branch": "score_align",
            "n_frames": int(len(raw)), "score_raw": raw.tolist(),
            "score_reverse_inverse": (
                None if reverse_inverse is None else reverse_inverse.tolist()
            ),
            "max_abs_difference_from_saved_test_score": max_saved_difference,
        }
        append(output, row)
        print(json.dumps({"event": "video_complete", "corpus": corpus,
                          "index": len(completed) + index, "total": len(ordered),
                          "split": split, "video_id": video_id}), flush=True)
    if load_completed(output, corpus, allowed) != allowed:
        raise RuntimeError(f"{corpus}: producer ended without exact cohort")


if __name__ == "__main__":
    main()
