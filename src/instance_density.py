"""Shared same-corpus bag-label instance-density utilities."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import SGDClassifier

from macilsd import align
from powa_macil.dataset import aligned_text


CHANNELS = ("audio", "visual", "text", "concat")


def normalize(rows):
    rows = np.asarray(rows, dtype=np.float32)
    return rows / np.maximum(np.linalg.norm(rows, axis=1, keepdims=True), 1e-6)


def features(corpus, video_id, max_rows=None):
    audio, n_seconds, snippets = align.aligned_audio(corpus, video_id, "snippet")
    if max_rows and len(audio) > max_rows:
        selected = np.linspace(
            0, len(audio) - 1, max_rows, dtype=np.uint16
        ).astype(np.int64)
        audio = audio[selected]
    else:
        selected = np.arange(len(audio), dtype=np.int64)
    visual = np.asarray(
        np.load(align.visual_path(corpus, video_id), mmap_mode="r")[selected],
        dtype=np.float32,
    ).mean(axis=1)
    text = aligned_text(corpus, video_id, "snippet", n_seconds, snippets)[selected]
    return {
        "audio": normalize(audio),
        "visual": normalize(visual),
        "text": normalize(text),
    }, n_seconds, snippets


def channel_rows(parts, channel):
    if channel != "concat":
        return parts[channel]
    return np.concatenate(
        [parts[name] for name in ("audio", "visual", "text")], axis=1
    ) / np.sqrt(3.0)


def fit_models(corpus, video_ids, labels, seed=234, epochs=5,
               max_rows=200, alpha=1e-4):
    models = {
        channel: SGDClassifier(
            loss="log_loss", penalty="l2", alpha=alpha,
            random_state=seed, average=True,
        )
        for channel in CHANNELS
    }
    counts = {value: sum(labels[v] == value for v in video_ids)
              for value in (0, 1)}
    if not all(counts.values()):
        raise ValueError("both bag classes are required")
    cache = {video_id: features(corpus, video_id, max_rows)[0]
             for video_id in video_ids}
    rng = np.random.default_rng(seed)
    for _ in range(epochs):
        order = np.asarray(video_ids, dtype=object)
        rng.shuffle(order)
        for item in order:
            video_id = str(item)
            target_value = int(labels[video_id])
            for channel in CHANNELS:
                rows = channel_rows(cache[video_id], channel)
                target = np.full(len(rows), target_value, dtype=np.int64)
                weight = np.full(
                    len(rows), len(video_ids) / (2.0 * counts[target_value]),
                    dtype=np.float64,
                )
                models[channel].partial_fit(
                    rows, target, classes=np.asarray([0, 1]),
                    sample_weight=weight,
                )
    return models


def tie_neutral_transport(anchor, order):
    anchor = np.asarray(anchor, dtype=np.float64)
    order = np.asarray(order, dtype=np.float64)
    if anchor.shape != order.shape:
        raise ValueError("anchor/order shape mismatch")
    output = np.empty_like(anchor)
    output[np.lexsort((anchor, order))] = np.sort(anchor, kind="stable")
    return output
