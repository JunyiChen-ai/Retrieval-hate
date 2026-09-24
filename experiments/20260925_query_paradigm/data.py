"""Inputs of the query-tree method on the 1-second grid (README section 2.3): I3D visual crops resampled from the
snippet grid to seconds, VGGish audio (native seconds), BERT text rows (1 s grid, resampled to the video's
seconds), and the cached VLM answers on the query tree (data/vlm_tree/<Corpus>/answers_qwen7b_mod5*.jsonl)."""
from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np
import scipy.sparse as sp
import torch
from torch.utils import data

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "scripts", "reproduction_baselines"))
sys.path.insert(0, os.path.join(ROOT, "src"))
from macilsd import align                  # noqa: E402
import hier_evidence_common as hc          # noqa: E402
import qtree                               # noqa: E402

CORPUS_DIR = {"hatemm": "HateMM", "hateclipseg": "HateClipSeg"}
A_IN = align.A_DIM + hc.TEXT_DIM


def resample_matrix(src_bounds, dst_bounds):
    """Sparse (n_dst, n_src) matrix M with M @ src == align.resample_intervals(src, src_bounds, dst_bounds)."""
    src_bounds = np.asarray(src_bounds, dtype=np.float64)
    dst_bounds = np.asarray(dst_bounds, dtype=np.float64)
    starts, ends = src_bounds[:, 0], src_bounds[:, 1]
    n_src = len(starts)
    lo_all = np.searchsorted(ends, dst_bounds[:, 0], side="right")
    hi_all = np.searchsorted(starts, dst_bounds[:, 1], side="left")
    rows, cols, vals = [], [], []
    for j, (a, b) in enumerate(dst_bounds):
        lo, hi = int(lo_all[j]), int(hi_all[j])
        near = min(max(lo, 0), n_src - 1)
        if hi <= lo:
            rows.append(j); cols.append(near); vals.append(1.0)
            continue
        w = np.clip(np.minimum(ends[lo:hi], b) - np.maximum(starts[lo:hi], a), 0.0, None)
        if w.sum() <= 0.0:
            rows.append(j); cols.append(near); vals.append(1.0)
            continue
        w = (w / w.sum()).astype(np.float32)
        rows += [j] * len(w); cols += list(range(lo, hi)); vals += list(w)
    return sp.csr_matrix((np.asarray(vals, dtype=np.float32), (rows, cols)), shape=(len(dst_bounds), n_src))


class Store:
    """Per video: audio+text rows (T, A_IN) on seconds, T, the snippet->second resampling matrix."""

    def __init__(self, corpus, video_ids, text_source="bert"):
        self.corpus = corpus
        self.at, self.T, self.W = {}, {}, {}
        self.n_missing_text = 0
        for v in video_ids:
            audio, T, snip = align.aligned_audio(corpus, v, "second")
            p = hc.text_path(corpus, v, text_source)
            text = None
            if os.path.exists(p):
                arr = np.load(p).astype(np.float32)
                if arr.ndim == 2 and arr.shape[1] == hc.TEXT_DIM and arr.shape[0] > 0:
                    text = align.resample_intervals(arr, align.second_bounds(arr.shape[0]), align.second_bounds(T))
            if text is None:
                self.n_missing_text += 1
                text = np.zeros((T, hc.TEXT_DIM), dtype=np.float32)
            self.at[v] = np.ascontiguousarray(np.concatenate([audio, text], axis=1), dtype=np.float32)
            self.T[v] = int(T)
            self.W[v] = resample_matrix(snip, align.second_bounds(T))

    def visual(self, v, crop):
        x = align.load_visual_crop(self.corpus, v, crop)
        return np.ascontiguousarray(self.W[v] @ x, dtype=np.float32)


def load_answers(corpus):
    """video id -> {(a, b): answer vector (5,) int or None}; from answers_qwen7b_mod5.jsonl (and shards)."""
    base = os.path.join(ROOT, "data", "vlm_tree", CORPUS_DIR[corpus])
    out, T = {}, {}
    for path in sorted(glob.glob(os.path.join(base, "answers_qwen7b_mod5*.jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if r["id"] in out:
                continue
            out[r["id"]] = {(int(a), int(b)): (None if o is None else np.asarray(o, dtype=np.int64))
                            for a, b, o, _raw in r["nodes"]}
            T[r["id"]] = int(r["T"])
    return out, T


def observed(answers_v, T):
    """(node ids, answers (n, 5), lengths (n,)) of the answered nodes of one video in qtree.tree(T) numbering."""
    tr = qtree.tree(T)
    ids, ans, lens = [], [], []
    for (a, b), o in answers_v.items():
        if o is None:
            continue
        ids.append(tr["index"][(a, b)])
        ans.append(o)
        lens.append(b - a)
    if not ids:
        return (np.zeros(0, dtype=np.int64), np.zeros((0, qtree.N_CAT), dtype=np.int64), np.zeros(0))
    order = np.argsort(ids)
    return (np.asarray(ids)[order], np.stack(ans)[order], np.asarray(lens, dtype=np.float64)[order])


class TrainSet(data.Dataset):
    def __init__(self, store, video_ids, labels, crop_repeat=align.N_CROPS):
        self.store, self.ids, self.labels = store, list(video_ids), labels
        self.crop_repeat = int(crop_repeat)

    def __len__(self):
        return len(self.ids) * self.crop_repeat

    def __getitem__(self, i):
        v = self.ids[i // self.crop_repeat]
        crop = i % self.crop_repeat
        return self.store.visual(v, crop), self.store.at[v], float(self.labels[v]), i // self.crop_repeat


def collate(items):
    B = len(items)
    Tm = max(x[0].shape[0] for x in items)
    f_v = torch.zeros(B, Tm, items[0][0].shape[1])
    f_a = torch.zeros(B, Tm, items[0][1].shape[1])
    seq = torch.zeros(B, dtype=torch.long)
    for i, (v, a, _, _) in enumerate(items):
        T = v.shape[0]
        f_v[i, :T] = torch.from_numpy(v)
        f_a[i, :T] = torch.from_numpy(a)
        seq[i] = T
    label = torch.tensor([x[2] for x in items], dtype=torch.float32)
    idx = torch.tensor([x[3] for x in items], dtype=torch.long)
    return f_v, f_a, seq, label, idx


class LengthBatches(data.Sampler):
    """Batches of similar length (sorted by T with a random tie-break each epoch, batches shuffled); a batch
    whose longest video exceeds `long_T` seconds is shrunk so batch_size * T_max**2 stays within
    batch_size * long_T**2 (attention memory)."""

    def __init__(self, lengths, batch_size, seed, long_T=512):
        self.lengths = np.asarray(lengths)
        self.bs = int(batch_size)
        self.long_T = int(long_T)
        self.rng = np.random.RandomState(seed)

    def _batches(self):
        n = len(self.lengths)
        order = np.lexsort((self.rng.rand(n), self.lengths))
        out, i = [], 0
        while i < n:
            bs = self.bs
            while True:
                chunk = order[i:i + bs]
                tmax = self.lengths[chunk].max()
                if bs == 1 or bs * tmax ** 2 <= self.bs * self.long_T ** 2:
                    break
                bs = max(1, bs // 2)
            out.append(chunk.tolist())
            i += len(chunk)
        self.rng.shuffle(out)
        return out

    def __iter__(self):
        return iter(self._batches())

    def __len__(self):
        return len(self._batches())
