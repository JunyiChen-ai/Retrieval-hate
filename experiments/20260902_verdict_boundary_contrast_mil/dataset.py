"""Datasets for the verdict-scaffolded boundary-contrast MIL candidate.

Rows live on MACIL-SD's I3D snippet grid (0.667 s). Per row:
    f_v   I3D RGB, one of five crops (1024)
    f_a   VGGish (128) ⊕ BERT sentence (768) ⊕ scaffold (12)
where the scaffold is the frozen-VLM window verdict at K = 30 and K = 4
(one-hot 4 + level/3 each) plus two normalised position channels
(src/vlm_verdict.py; revision 4 added the K = 4 granularity).

Training items are (video, crop) pairs exactly as in macilsd/dataset.py; the
validation/test items stack the five crops. Videos without a BERT array or a
verdict record are kept with zeros in those channels so the scored cohort is
the same one MACIL-SD scores (I3D + VGGish present).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.utils.data as data

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(REPO_ROOT, "src"))

from macilsd import align                      # noqa: E402
from macilsd.utils import process_feat         # noqa: E402
import vlm_verdict                             # noqa: E402

TEXT_ROOT = os.path.join(REPO_ROOT, "results", "reproduction", "features",
                         "bert_sentence_1fps")
TEXT_DIM = 768
A_EXT_DIM = align.A_DIM + TEXT_DIM + vlm_verdict.SCAFFOLD_DIM


def text_path(corpus, vid):
    return os.path.join(TEXT_ROOT, corpus, "%s.npy" % vid)


def load_text_rows(corpus, vid, snip):
    """BERT rows resampled from the 1 s grid onto the snippet grid, or zeros."""
    p = text_path(corpus, vid)
    if not os.path.exists(p):
        return None
    arr = np.load(p).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != TEXT_DIM or arr.shape[0] == 0:
        return None
    return align.resample_intervals(arr, align.second_bounds(arr.shape[0]),
                                    snip)


class ScaffoldCache:
    """Per-video (f_a_ext, n_seconds, snip_bounds), computed once."""

    def __init__(self, corpus, video_ids, verdicts):
        """``verdicts``: one dict (video_id -> scores) per granularity in
        vlm_verdict.GRANULARITIES; a single dict means the first only."""
        if isinstance(verdicts, dict):
            verdicts = [verdicts] + [{}] * (len(vlm_verdict.GRANULARITIES) - 1)
        self.corpus = corpus
        self.items = {}
        self.n_missing_text = 0
        self.n_missing_verdict = 0
        self.n_missing_by_gran = [0] * len(verdicts)
        for vid in video_ids:
            audio, n_seconds, snip = align.aligned_audio(corpus, vid, "snippet")
            text = load_text_rows(corpus, vid, snip)
            if text is None:
                self.n_missing_text += 1
                text = np.zeros((audio.shape[0], TEXT_DIM), dtype=np.float32)
            sc = [v.get(vid) for v in verdicts]
            for g, x in enumerate(sc):
                if x is None:
                    self.n_missing_by_gran[g] += 1
            if any(x is None for x in sc):
                self.n_missing_verdict += 1
            scaf = vlm_verdict.scaffold_features(sc, snip, n_seconds)
            f_a = np.concatenate([audio, text, scaf], axis=1).astype(np.float32)
            self.items[vid] = (np.ascontiguousarray(f_a), n_seconds, snip)

    def __getitem__(self, vid):
        return self.items[vid]


class TrainDataset(data.Dataset):
    def __init__(self, corpus, video_ids, labels, cache, max_seqlen,
                 crop_repeat=align.N_CROPS):
        self.corpus = corpus
        self.video_ids = list(video_ids)
        self.labels = labels
        self.cache = cache
        self.max_seqlen = int(max_seqlen)
        self.crop_repeat = int(crop_repeat)

    def __len__(self):
        return len(self.video_ids) * self.crop_repeat

    def __getitem__(self, index):
        vid = self.video_ids[index // self.crop_repeat]
        crop = index % self.crop_repeat
        f_a, n_seconds, snip = self.cache[vid]
        f_v = align.aligned_visual_crop(self.corpus, vid, crop, "snippet",
                                        n_seconds, snip)
        f_v = process_feat(f_v, self.max_seqlen, is_random=False)
        f_a = process_feat(f_a, self.max_seqlen, is_random=False)
        return (torch.from_numpy(np.ascontiguousarray(f_v, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(f_a, dtype=np.float32)),
                float(self.labels[vid]))


class EvalDataset(data.Dataset):
    """One item per video: five crops stacked, full untruncated sequence."""

    def __init__(self, corpus, video_ids, cache):
        self.corpus = corpus
        self.video_ids = list(video_ids)
        self.cache = cache

    def __len__(self):
        return len(self.video_ids)

    def __getitem__(self, index):
        vid = self.video_ids[index]
        f_a, n_seconds, snip = self.cache[vid]
        crops = [align.aligned_visual_crop(self.corpus, vid, c, "snippet",
                                           n_seconds, snip)
                 for c in range(align.N_CROPS)]
        f_v = np.stack(crops, axis=0)
        f_a = np.repeat(f_a[None], align.N_CROPS, axis=0)
        index_map = align.snippet_index_for_seconds(snip, n_seconds)
        return (torch.from_numpy(np.ascontiguousarray(f_v, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(f_a, dtype=np.float32)),
                torch.from_numpy(index_map), int(n_seconds), vid)
