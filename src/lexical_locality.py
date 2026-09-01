"""Shared train-label lexical locality utilities.

This module parses timestamped ASR fail-closed and maps chunks to the frozen
one-second score grid.  It never reads temporal ground truth.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def load_asr(path: Path) -> tuple[dict[str, dict], dict[str, int]]:
    rows: dict[str, dict] = {}
    stats = {"dropped_missing_endpoint": 0,
             "dropped_nonfinite_endpoint": 0,
             "dropped_nonpositive_span": 0}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            row = json.loads(line)
            vid = str(row["video_id"])
            if vid in rows:
                raise ValueError(f"duplicate ASR record: {vid}")
            chunks = []
            for chunk in row.get("chunks", []):
                if chunk.get("start") is None or chunk.get("end") is None:
                    stats["dropped_missing_endpoint"] += 1
                    continue
                start = float(chunk["start"])
                end = float(chunk["end"])
                if not np.isfinite(start) or not np.isfinite(end):
                    stats["dropped_nonfinite_endpoint"] += 1
                    continue
                text = str(chunk.get("text", "")).strip()
                if end <= start:
                    stats["dropped_nonpositive_span"] += 1
                    continue
                if text:
                    chunks.append((start, end, text))
            rows[vid] = {"text": str(row.get("text", "")).strip(),
                         "chunks": chunks}
    return rows, stats


def local_texts(chunks: list[tuple[float, float, str]], length: int,
                left_context: float = 2.0,
                right_context: float = 3.0) -> tuple[list[str], np.ndarray]:
    texts: list[str] = []
    speech = np.zeros(length, dtype=np.float64)
    for t in range(length):
        selected = [text for start, end, text in chunks
                    if end > t - left_context and start < t + right_context]
        texts.append(" ".join(selected))
        speech[t] = float(bool(selected))
    return texts, speech


def new_vectorizer() -> TfidfVectorizer:
    return TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5), min_df=2,
        max_features=50000, lowercase=True, sublinear_tf=True)


def new_classifier() -> LogisticRegression:
    return LogisticRegression(
        C=1.0, class_weight="balanced", solver="liblinear",
        max_iter=1000, random_state=234)
