"""Frozen-VLM window verdicts as per-row input features.

Source cache: ``data/MLLM_scores/<Corpus>/*_segscoreK<k>_<tag>.jsonl`` written
by ``scripts/analysis/score_segments_mllm.py``. Each record holds ``K`` integer
scores in 0..3, one per contiguous window of ``K`` equal frame windows over
uniformly sampled frames of the whole video, i.e. window ``j`` covers the time
slice ``[j*D/K, (j+1)*D/K)`` of a video of duration ``D``. Every window was
rated in isolation from its <=4 frames plus that window's ASR text.

This module only expands those coarse verdicts onto the row grid a model
trains on (1 s rows, or I3D snippet rows) and appends two normalised position
channels. Nothing here reads labels.
"""

from __future__ import annotations

import json
import os

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CORPUS_DIR = {"hatemm": "HateMM", "hateclipseg": "HateClipSeg",
              "mhclip_en": "MHC", "mhclip_zh": "MHC_zh"}
N_LEVELS = 4          # verdict levels 0..3
GRANULARITIES = (30, 4)           # windows per video, coarse-to-fine order fixed
VERDICT_DIMS = len(GRANULARITIES) * (N_LEVELS + 1)   # per K: one-hot + level/3
SCAFFOLD_DIM = VERDICT_DIMS + 2   # + position, edge distance (revision 4)


def verdict_files(corpus, k=30, tag="qwen"):
    d = os.path.join(REPO_ROOT, "data", "MLLM_scores", CORPUS_DIR[corpus])
    suffix = "_segscoreK%d_%s.jsonl" % (k, tag)
    return sorted(os.path.join(d, f) for f in os.listdir(d)
                  if f.endswith(suffix))


def load_verdicts(corpus, k=30, tag="qwen", video_ids=None, strict=False):
    """video_id -> float array of length k (verdict levels), across all splits.

    A video that appears in several files keeps the first record read; the
    files are read in sorted name order so the choice is deterministic.
    """
    out = {}
    requested = None if video_ids is None else set(video_ids)
    for path in verdict_files(corpus, k, tag):
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                vid = str(rec["id"])
                if requested is not None and vid not in requested:
                    continue
                if strict:
                    values = np.asarray(rec['scores'], dtype=np.float64)
                    if values.shape != (k,) or not np.isfinite(values).all() or not np.isin(values, [0, 1, 2, 3]).all():
                        raise ValueError(f'invalid requested verdict: {path} {vid}')
                    if vid in out and not np.array_equal(values, out[vid]):
                        raise ValueError(f'conflicting requested verdict: {path} {vid}')
                if vid in out:
                    continue
                scores = [0.0 if s is None else float(s) for s in rec["scores"]]
                if len(scores) != k:
                    raise ValueError("%s: %s has %d scores, expected %d"
                                     % (path, vid, len(scores), k))
                out[vid] = np.asarray(scores, dtype=np.float32)
    return out


def verdict_rows(scores, row_bounds, duration):
    """Verdict level of the window containing each row's midpoint."""
    scores = np.asarray(scores, dtype=np.float32)
    k = scores.shape[0]
    rb = np.asarray(row_bounds, dtype=np.float64)
    mid = (rb[:, 0] + rb[:, 1]) / 2.0
    idx = np.floor(mid * k / max(float(duration), 1e-6)).astype(np.int64)
    idx = np.clip(idx, 0, k - 1)
    return scores[idx]


def scaffold_features(scores, row_bounds, duration):
    """(n_rows, SCAFFOLD_DIM) float32.

    Columns: for each K in GRANULARITIES, one-hot(4) + level/3 (5 columns);
    then t/D and min(t, D-t)/D. ``scores`` is a sequence with one entry per
    granularity (array of length K, or None = no verdict: those five columns
    stay zero). A single array is accepted for the first granularity only.
    """
    if scores is None or isinstance(scores, np.ndarray):
        scores = [scores] + [None] * (len(GRANULARITIES) - 1)
    assert len(scores) == len(GRANULARITIES)
    rb = np.asarray(row_bounds, dtype=np.float64)
    n = rb.shape[0]
    d = max(float(duration), 1e-6)
    mid = (rb[:, 0] + rb[:, 1]) / 2.0
    pos = np.clip(mid / d, 0.0, 1.0)
    edge = np.clip(np.minimum(mid, d - mid) / d, 0.0, 1.0)
    out = np.zeros((n, SCAFFOLD_DIM), dtype=np.float32)
    for g, sc in enumerate(scores):
        if sc is None:
            continue
        off = g * (N_LEVELS + 1)
        lev = verdict_rows(sc, rb, d)
        li = np.clip(np.rint(lev).astype(np.int64), 0, N_LEVELS - 1)
        out[np.arange(n), off + li] = 1.0
        out[:, off + N_LEVELS] = lev / float(N_LEVELS - 1)
    out[:, VERDICT_DIMS] = pos
    out[:, VERDICT_DIMS + 1] = edge
    return out
