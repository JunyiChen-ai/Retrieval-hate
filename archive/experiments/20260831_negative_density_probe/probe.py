#!/usr/bin/env python3
"""Validation-only linear probe for same-corpus local content evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import SGDClassifier


HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
BASE = REPO / "scripts/reproduction_baselines"
sys.path.insert(0, str(BASE))

from eval_baseline_scores import evaluate_scores  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from macilsd import align  # noqa: E402
from powa_macil.dataset import aligned_text, usable_text_ids  # noqa: E402


CHANNELS = ("audio", "visual", "text", "concat")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    p.add_argument("--anchor-scores", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--seed", type=int, default=234)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max-rows", type=int, default=200)
    p.add_argument("--alpha", type=float, default=1e-4)
    p.add_argument("--limit-train-videos", type=int, default=0)
    p.add_argument("--limit-val-videos", type=int, default=0)
    return p


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize(rows):
    rows = np.asarray(rows, dtype=np.float32)
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norm, 1e-6)


def features(corpus, video_id, max_rows=None):
    audio, n_seconds, snippets = align.aligned_audio(
        corpus, video_id, "snippet"
    )
    if max_rows and len(audio) > max_rows:
        selected = np.linspace(
            0, len(audio) - 1, max_rows, dtype=np.uint16
        ).astype(np.int64)
        audio = audio[selected]
    else:
        selected = np.arange(len(audio), dtype=np.int64)
    visual_file = np.load(align.visual_path(corpus, video_id), mmap_mode="r")
    visual = np.asarray(visual_file[selected], dtype=np.float32).mean(axis=1)
    text = aligned_text(corpus, video_id, "snippet", n_seconds, snippets)
    text = text[selected]
    parts = {
        "audio": normalize(audio),
        "visual": normalize(visual),
        "text": normalize(text),
    }
    return parts, n_seconds, snippets


def channel_rows(parts, channel):
    if channel != "concat":
        return parts[channel]
    return np.concatenate(
        [parts[name] for name in ("audio", "visual", "text")], axis=1
    ) / np.sqrt(3.0)


def stratified_limit(ids, labels, limit):
    if not limit or len(ids) <= limit:
        return ids
    positive = [v for v in ids if labels[v] == 1]
    negative = [v for v in ids if labels[v] == 0]
    n_positive = min(len(positive), (limit + 1) // 2)
    n_negative = min(len(negative), limit - n_positive)
    if n_positive + n_negative < limit:
        n_positive = min(len(positive), limit - n_negative)
    output = positive[:n_positive] + negative[:n_negative]
    return sorted(output, key=ids.index)


def train_models(args, train_ids, labels):
    models = {
        channel: SGDClassifier(
            loss="log_loss", penalty="l2", alpha=args.alpha,
            random_state=args.seed, average=True,
        )
        for channel in CHANNELS
    }
    counts = {label: sum(labels[v] == label for v in train_ids)
              for label in (0, 1)}
    rng = np.random.default_rng(args.seed)
    updates = {channel: 0 for channel in CHANNELS}
    cache = {
        video_id: features(args.corpus, video_id, args.max_rows)[0]
        for video_id in train_ids
    }
    for epoch in range(args.epochs):
        order = np.asarray(train_ids, dtype=object)
        rng.shuffle(order)
        for video_id in order:
            parts = cache[str(video_id)]
            y_value = int(labels[str(video_id)])
            for channel in CHANNELS:
                rows = channel_rows(parts, channel)
                target = np.full(len(rows), y_value, dtype=np.int64)
                weight = np.full(
                    len(rows), len(train_ids) / (2.0 * counts[y_value]),
                    dtype=np.float64,
                )
                models[channel].partial_fit(
                    rows, target, classes=np.asarray([0, 1]),
                    sample_weight=weight,
                )
                updates[channel] += 1
        print(json.dumps({"epoch": epoch + 1, "updates": updates}), flush=True)
    return models


def load_anchor(path):
    output = {}
    with path.open() as handle:
        for line in handle:
            row = json.loads(line)
            output[row["video_id"]] = np.asarray(row["score_powa"], dtype=float)
    return output


def assign_multiset(anchor, order_score):
    output = np.empty_like(anchor)
    output[np.argsort(order_score, kind="stable")] = np.sort(anchor, kind="stable")
    return output


def metric_summary(report):
    return {
        "pooled_ap": report["pr_auc"], "pooled_roc": report["roc_auc"],
        "within_roc": report["per_video"]["macro_auc"],
        "within_n": report["per_video"]["n_videos_both_classes"],
    }


def main(argv=None):
    args = parser().parse_args(argv)
    args.anchor_scores = args.anchor_scores.resolve()
    args.out_dir = args.out_dir.resolve()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    labels = hdata.load_labels(args.corpus)
    train_ids, val_ids = hdata.load_train_val(args.corpus, labels)
    train_ids = usable_text_ids(args.corpus, train_ids)
    val_ids = usable_text_ids(args.corpus, val_ids)
    train_ids = stratified_limit(train_ids, labels, args.limit_train_videos)
    val_ids = stratified_limit(val_ids, labels, args.limit_val_videos)
    models = train_models(args, train_ids, labels)
    anchors = load_anchor(args.anchor_scores)
    branches = {"score_powa": {}}
    for channel in CHANNELS:
        branches[f"score_probe_{channel}"] = {}
        branches[f"score_transport_{channel}"] = {}
    branches["score_position_center"] = {}
    multiset_errors = []
    for video_id in val_ids:
        parts, n_seconds, snippets = features(args.corpus, video_id)
        index = align.snippet_index_for_seconds(snippets, n_seconds)
        anchor = anchors[video_id]
        if len(anchor) != n_seconds:
            raise RuntimeError(f"anchor alignment mismatch: {video_id}")
        branches["score_powa"][video_id] = anchor
        for channel in CHANNELS:
            raw = models[channel].decision_function(
                channel_rows(parts, channel)
            )[index]
            branches[f"score_probe_{channel}"][video_id] = raw
            transported = assign_multiset(anchor, raw)
            branches[f"score_transport_{channel}"][video_id] = transported
            multiset_errors.append(float(np.max(
                np.abs(np.sort(anchor) - np.sort(transported))
            )))
        position = -np.abs(np.arange(n_seconds) - (n_seconds - 1) / 2)
        branches["score_position_center"][video_id] = assign_multiset(
            anchor, position
        )
    gt = hdata.gt_arrays(args.corpus, "val")
    positives = {v for v in gt if labels[v] == 1}
    reports = {
        name: evaluate_scores(values, gt, positives)
        for name, values in branches.items()
    }
    result = {
        "date": "2026-08-31", "split": "val", "corpus": args.corpus,
        "method_status": "diagnostic_only", "test_used": False,
        "args": {key: str(value) if isinstance(value, Path) else value
                 for key, value in vars(args).items()},
        "anchor_scores_sha256": sha256(args.anchor_scores),
        "train_videos": len(train_ids), "val_videos": len(val_ids),
        "train_positive": sum(labels[v] == 1 for v in train_ids),
        "train_negative": sum(labels[v] == 0 for v in train_ids),
        "max_multiset_error": max(multiset_errors),
        "metrics": {name: metric_summary(report)
                    for name, report in reports.items()},
    }
    temporary = args.out_dir / "analysis.json.tmp"
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(args.out_dir / "analysis.json")
    with (args.out_dir / "scores.jsonl").open("w") as handle:
        for video_id in val_ids:
            handle.write(json.dumps({
                "video_id": video_id,
                **{name: values[video_id].tolist()
                   for name, values in branches.items()},
            }) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
