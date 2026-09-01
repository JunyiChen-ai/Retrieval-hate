"""Aligned, row-normalized local audio/visual/text features for diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parent.parent
BASE = REPO / "scripts/reproduction_baselines"
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))

from macilsd import align  # noqa: E402
from powa_macil.dataset import aligned_text  # noqa: E402


def row_normalize(rows):
    rows = np.asarray(rows, dtype=np.float32)
    norm = np.linalg.norm(rows, axis=1, keepdims=True)
    return rows / np.maximum(norm, 1e-6)


def aligned_local_features(corpus, video_id):
    audio, n_seconds, snippets = align.aligned_audio(
        corpus, video_id, "snippet"
    )
    visual_file = np.load(align.visual_path(corpus, video_id), mmap_mode="r")
    visual = np.asarray(visual_file, dtype=np.float32).mean(axis=1)
    text = aligned_text(corpus, video_id, "snippet", n_seconds, snippets)
    parts = {
        "audio": row_normalize(audio),
        "visual": row_normalize(visual),
        "text": row_normalize(text),
    }
    parts["concat"] = np.concatenate(
        [parts[name] for name in ("audio", "visual", "text")], axis=1
    ) / np.sqrt(3.0)
    index = align.snippet_index_for_seconds(snippets, n_seconds)
    return {name: rows[index] for name, rows in parts.items()}
