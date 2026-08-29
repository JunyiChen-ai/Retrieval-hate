"""POWA data adapter: MACIL inputs plus aligned sentence evidence.

Feature extraction is preprocessing and is intentionally outside the method.
This adapter only aligns the already archived 1-fps BERT arrays to the exact
temporal grid used by MACIL-SD.
"""

from __future__ import annotations

import os
import json

import numpy as np
import torch

from macilsd import align
from macilsd.dataset import MacilTestDataset, MacilTrainDataset
from macilsd.utils import process_feat

TEXT_ROOT = os.path.join(align.REPO_ROOT, "results", "reproduction",
                         "features", "bert_sentence_1fps")
TEXT_DIM = 768


def text_path(corpus, video_id):
    return os.path.join(TEXT_ROOT, corpus, "%s.npy" % video_id)


def has_text(corpus, video_id):
    return os.path.exists(text_path(corpus, video_id))


def aligned_text(corpus, video_id, grid, n_seconds, snippet_bounds):
    feat = np.load(text_path(corpus, video_id)).astype(np.float32)
    if feat.ndim != 2 or feat.shape[1] != TEXT_DIM:
        raise ValueError("%s/%s text: expected (T,%d), got %s" %
                         (corpus, video_id, TEXT_DIM, feat.shape))
    # Extraction archives one row per second. Some source ASR tracks end a
    # second early/late; interval resampling also gives an explicit hold-tail
    # rule rather than silently slicing multimodal streams to different spans.
    src_bounds = align.second_bounds(feat.shape[0])
    dst = (snippet_bounds if grid == "snippet"
           else align.second_bounds(n_seconds))
    return align.resample_intervals(feat, src_bounds, dst)


def usable_text_ids(corpus, ids):
    return [v for v in ids if align.has_features(corpus, v) and has_text(corpus, v)]


class PowaTrainDataset(MacilTrainDataset):
    def __init__(self, *args, **kwargs):
        self.teacher_records = kwargs.pop("teacher_records", None)
        self.permute_teacher_channels = kwargs.pop("permute_teacher_channels", False)
        super().__init__(*args, **kwargs)
        missing = [v for v in self.video_ids if not has_text(self.corpus, v)]
        if missing:
            raise FileNotFoundError("%d ids lack BERT evidence, e.g. %s" %
                                    (len(missing), missing[:5]))
        self._text = {}
        self._teacher = {}
        for vid in self.video_ids:
            n_seconds, snip = self._meta[vid]
            self._text[vid] = aligned_text(self.corpus, vid, self.grid,
                                           n_seconds, snip)
            if self.teacher_records is not None:
                self._teacher[vid] = teacher_on_grid(
                    self.teacher_records.get((self.corpus, vid)), self.grid,
                    n_seconds, snip)

    def __getitem__(self, index):
        f_v, f_a, label = super().__getitem__(index)
        vid = self.video_ids[index // self.crop_repeat]
        f_t = process_feat(self._text[vid], self.max_seqlen, is_random=False)
        out = (f_v, f_a, torch.from_numpy(np.ascontiguousarray(f_t)), label)
        if self.teacher_records is None:
            return out
        target, mask = self._teacher[vid]
        if self.permute_teacher_channels:
            target = np.roll(target, 1, axis=-1).copy()
        target = process_feat(target, self.max_seqlen, is_random=False)
        mask = process_feat(mask[:, None], self.max_seqlen,
                            is_random=False)[:, 0]
        return out + (torch.from_numpy(np.ascontiguousarray(target)),
                      torch.from_numpy(np.ascontiguousarray(mask)))


class PowaTestDataset(MacilTestDataset):
    def __getitem__(self, index):
        f_v, f_a, index_map, n_seconds, vid = super().__getitem__(index)
        snip = align.snippet_bounds(self.corpus, vid)
        f_t = aligned_text(self.corpus, vid, self.grid, n_seconds, snip)
        # Repeat for the five visual crops, matching f_v/f_a batch semantics.
        f_t = np.repeat(f_t[None], self.n_crops, axis=0)
        return (f_v, f_a, torch.from_numpy(np.ascontiguousarray(f_t)),
                index_map, n_seconds, vid)


def load_teacher_jsonl(path):
    records = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = (rec["corpus"], rec["video_id"])
            if key in records:
                raise ValueError("duplicate teacher record %s/%s" % key)
            records[key] = rec
    return records


def teacher_on_grid(record, grid, n_seconds, snippet_bounds):
    """Sparse teacher targets/mask on the student's temporal grid."""
    bounds = (snippet_bounds if grid == "snippet" else
              align.second_bounds(n_seconds))
    target = np.zeros((len(bounds), 6), dtype=np.float32)
    mask = np.zeros(len(bounds), dtype=np.float32)
    if not record:
        return target, mask
    chunks = record.get("chunks") or []
    values = record.get("primitive_prob") or []
    if len(chunks) != len(values):
        raise ValueError("teacher chunk/value mismatch")
    midpoint = bounds.mean(1)
    names = ("hostile", "target", "violence", "sexual", "self_harm", "context")
    for chunk, value in zip(chunks, values):
        start = float(chunk.get("start") or 0.0)
        raw_end = chunk.get("end")
        end = float(raw_end) if raw_end is not None else start
        hit = (midpoint >= start) & (midpoint < end)
        if not hit.any():
            hit[np.argmin(np.abs(midpoint - (start + end) / 2.0))] = True
        target[hit] = np.asarray([value[k] for k in names], dtype=np.float32)
        mask[hit] = 1.0
    return target, mask
