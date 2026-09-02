"""Datasets for the hierarchical-evidence MIL candidate.

Rows live on MACIL-SD's I3D snippet grid (0.667 s). Per row:
    f_v   I3D RGB, one of five crops (1024)
    f_a   VGGish (128) ⊕ BERT sentence (768) ⊕ scaffold (SCAF_DIM)
The scaffold columns come from the hierarchical evidence HMM over the frozen
VLM verdicts (src/verdict_hmm.py):
    0  ell     posterior log-odds log P(s_t=1|b) - log P(s_t=0|b)  (the prior)
    1  p_s     posterior P(s_t=1)
    2  b_fine  binary K=30 verdict of the row's window
    3  b_coarse binary K=4 verdict of the row's block
    4  p_h     posterior P(h_j=1) of the row's coarse block (block-bag label)
    5  block   coarse block index j of the row (0..J-1)
Columns 0-3 are the backbone's input channels; columns 4-5 are training
bookkeeping and are always hidden from the backbone (train.py zeroes them on
the input path).

Training items are (video, crop) pairs exactly as in macilsd/dataset.py; the
validation/test items stack the five crops.
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
SCAF_DIM = 6
COL_ELL, COL_PS, COL_BF, COL_BC, COL_PH, COL_BLOCK = range(SCAF_DIM)
N_INPUT_SCAF = 4                      # columns fed to the backbone
A_EXT_DIM = align.A_DIM + TEXT_DIM + SCAF_DIM
SCAF_OFFSET = align.A_DIM + TEXT_DIM


def text_path(corpus, vid):
    return os.path.join(TEXT_ROOT, corpus, "%s.npy" % vid)


def load_text_rows(corpus, vid, snip):
    """BERT rows resampled from the 1 s grid onto the snippet grid, or None."""
    p = text_path(corpus, vid)
    if not os.path.exists(p):
        return None
    arr = np.load(p).astype(np.float32)
    if arr.ndim != 2 or arr.shape[1] != TEXT_DIM or arr.shape[0] == 0:
        return None
    return align.resample_intervals(arr, align.second_bounds(arr.shape[0]),
                                    snip)


def scaffold_rows(ell, p_s, b_fine, b_coarse, p_h, block_of_window,
                  snip, n_seconds):
    """Per-row scaffold from per-fine-window arrays (K,) and per-block (J,)."""
    k = len(ell)
    rows = lambda arr: vlm_verdict.verdict_rows(np.asarray(arr, np.float32),  # noqa: E731
                                                snip, n_seconds)
    blk = rows(block_of_window).astype(int)
    out = np.stack([rows(ell), rows(p_s), rows(b_fine),
                    np.asarray(b_coarse, np.float32)[blk],
                    np.asarray(p_h, np.float32)[blk],
                    blk.astype(np.float32)], axis=1)
    assert out.shape[1] == SCAF_DIM and k > 0
    return out.astype(np.float32)


class ScaffoldCache:
    """Per-video (f_a_ext, n_seconds, snip_bounds), computed once.

    ``scaffold_fn(vid, snip, n_seconds)`` returns the (rows, SCAF_DIM) scaffold
    or None (then zeros; counted in n_missing_verdict)."""

    def __init__(self, corpus, video_ids, scaffold_fn):
        self.corpus = corpus
        self.items = {}
        self.n_missing_text = 0
        self.n_missing_verdict = 0
        for vid in video_ids:
            audio, n_seconds, snip = align.aligned_audio(corpus, vid, "snippet")
            text = load_text_rows(corpus, vid, snip)
            if text is None:
                self.n_missing_text += 1
                text = np.zeros((audio.shape[0], TEXT_DIM), dtype=np.float32)
            scaf = scaffold_fn(vid, snip, n_seconds)
            if scaf is None:
                self.n_missing_verdict += 1
                scaf = np.zeros((audio.shape[0], SCAF_DIM), dtype=np.float32)
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
