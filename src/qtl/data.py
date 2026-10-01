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
from . import qtree                               # noqa: E402

CORPUS_DIR = {"hatemm": "HateMM", "hateclipseg": "HateClipSeg", "dehate": "DeHate"}
A_IN = align.A_DIM + hc.TEXT_DIM             # one text row; model.PriorNet adds TEXT_DIM per extra text source


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

    def __init__(self, corpus, video_ids, text_sources=("bert",)):
        self.corpus = corpus
        self.at, self.T, self.W = {}, {}, {}
        self.n_missing_text = 0
        for v in video_ids:
            audio, T, snip = align.aligned_audio(corpus, v, "second")
            rows = [audio]
            for src in text_sources:                 # one 768-d row per text encoder (hc.TEXT_ROOTS)
                p = hc.text_path(corpus, v, src)
                text = None
                if os.path.exists(p):
                    arr = np.load(p).astype(np.float32)
                    if arr.ndim == 2 and arr.shape[1] == hc.TEXT_DIM and arr.shape[0] > 0:
                        text = align.resample_intervals(arr, align.second_bounds(arr.shape[0]),
                                                        align.second_bounds(T))
                if text is None:
                    self.n_missing_text += 1
                    text = np.zeros((T, hc.TEXT_DIM), dtype=np.float32)
                rows.append(text)
            self.at[v] = np.ascontiguousarray(np.concatenate(rows, axis=1), dtype=np.float32)
            self.T[v] = int(T)
            self.W[v] = resample_matrix(snip, align.second_bounds(T))

    def visual(self, v, crop):
        x = align.load_visual_crop(self.corpus, v, crop)
        return np.ascontiguousarray(self.W[v] @ x, dtype=np.float32)


ANSWER_SOURCES = {"k30": "answers_qwen7b_mod5", "words": "answers_words_qwen7b_mod5",
                  # revision 5 step 1 (README section 17.1): soft first-token P(Yes) under the per-dataset
                  # definition, one category, SOFT_LEVELS quantile levels (extract_tree_soft.py)
                  "soft_both": "soft_both_p1", "soft_frames": "soft_frames_p1", "soft_text": "soft_text_p1",
    "soft5_both": "soft5_both_p5", "soft5_frames": "soft5_frames_p5", "soft5_text": "soft5_text_p5"}
SOFT_LEVELS = 8
SOFT5_EDGES = "fixed"        # five-category variant: "fixed" = level edges [0.1, 0.5, 1.5] on the expected-level
                             # scale 0-3 (0 = confident none, 1 = some mass on a cue, 2 = clear, 3 = explicit);
                             # "quantile" = train quantiles per category (the first run; near-zero categories
                             # became noise levels and the answers carried no information, README 17.1)
SOFT5_FIXED = [0.1, 0.5, 1.5]
SOFT_EDGES = {}                                # (corpus, source) -> the level edges used (for logging)


def is_soft(source):
    return source.startswith("soft_") or source.startswith("soft5_")


def is_soft5(source):
    return source.startswith("soft5_")


def configure_source(source, soft_levels=SOFT_LEVELS, soft5_edges=None):
    """Sets the answer shape the rest of the code reads from qtree (N_CAT categories x N_LEV levels): the decoded
    answers are 5 categories x 4 levels; a soft source is 1 category x soft_levels levels. Call before anything
    that builds an answer model or a TreeBatch."""
    global SOFT_LEVELS, SOFT5_EDGES
    if is_soft5(source):                       # README 17.1 variant: five categories, expected level each, binned
        SOFT_LEVELS = int(soft_levels)
        if soft5_edges is not None:
            SOFT5_EDGES = soft5_edges
        assert SOFT5_EDGES in ("fixed", "quantile")
        if SOFT5_EDGES == "fixed":
            assert SOFT_LEVELS == len(SOFT5_FIXED) + 1, "fixed five-category edges give 4 levels"
        qtree.N_CAT, qtree.N_LEV = 5, int(soft_levels)
    elif is_soft(source):
        SOFT_LEVELS = int(soft_levels)
        qtree.N_CAT, qtree.N_LEV = 1, int(soft_levels)
    else:
        qtree.N_CAT, qtree.N_LEV = 5, 4
    return qtree.N_CAT, qtree.N_LEV


def load_soft_p(corpus, source):
    """Soft source: video id -> {(a, b): p_yes float or None}, and T; raw values before the level binning."""
    base = os.path.join(ROOT, "data", "vlm_tree", CORPUS_DIR[corpus])
    out, T, split = {}, {}, {}
    for path in sorted(glob.glob(os.path.join(base, ANSWER_SOURCES[source] + "[.]*jsonl"))):
        for line in open(path):
            r = json.loads(line)
            if r["id"] in out:
                continue
            if is_soft5(source):               # o = [[E_1..E_5], [[p0..p3] x 5]] -> the five expected levels
                out[r["id"]] = {(int(a), int(b)): (None if o is None else [float(x) for x in o[0]])
                                for a, b, o, _raw in r["nodes"]}
            else:
                out[r["id"]] = {(int(a), int(b)): (None if o is None else float(o[0])) for a, b, o, _raw in r["nodes"]}
            T[r["id"]] = int(r["T"])
            split[r["id"]] = r["split"]
    return out, T, split


def soft_edges(p_by_video, split, levels):
    """Quantile edges of the training-split p values (label-free): level = number of edges <= p, in 0..levels-1."""
    vals = np.array([p for v, d in p_by_video.items() if split.get(v) == "train" for p in d.values()
                     if p is not None])
    assert len(vals) > 0, "no training answers to set the soft levels"
    return np.quantile(vals, [k / levels for k in range(1, levels)])


def load_answers(corpus, source="k30"):
    """video id -> {(a, b): answer vector (N_CAT,) int or None}; from answers_qwen7b_mod5.jsonl (and shards).
    source "words" (revision 4, README section 14): the same questions with word-timestamp transcripts.
    source "soft_<view>" (revision 5, README section 17.1): p_yes binned into SOFT_LEVELS quantile levels of the
    training answers, as a single category; configure_source must have been called with the same levels."""
    if is_soft5(source):
        assert qtree.N_CAT == 5 and qtree.N_LEV == SOFT_LEVELS, "call data.configure_source(source, levels) first"
        P, T, split = load_soft_p(corpus, source)
        if SOFT5_EDGES == "fixed":
            edges = [np.asarray(SOFT5_FIXED, float) for _ in range(5)]
        else:
            edges = [soft_edges({v: {k: (None if e is None else e[c]) for k, e in d.items()} for v, d in P.items()},
                                split, SOFT_LEVELS) for c in range(5)]    # train quantiles per category
        SOFT_EDGES[(corpus, source)] = edges
        out = {v: {k: (None if e is None else np.array([int(np.searchsorted(edges[c], e[c], side="right"))
                                                         for c in range(5)], dtype=np.int64))
                   for k, e in d.items()} for v, d in P.items()}
        return out, T
    if is_soft(source):
        assert qtree.N_CAT == 1 and qtree.N_LEV == SOFT_LEVELS, "call data.configure_source(source, levels) first"
        P, T, split = load_soft_p(corpus, source)
        edges = soft_edges(P, split, SOFT_LEVELS)
        SOFT_EDGES[(corpus, source)] = edges
        out = {v: {k: (None if p is None else np.array([int(np.searchsorted(edges, p, side="right"))], dtype=np.int64))
                   for k, p in d.items()} for v, d in P.items()}
        return out, T
    base = os.path.join(ROOT, "data", "vlm_tree", CORPUS_DIR[corpus])
    out, T = {}, {}
    for path in sorted(glob.glob(os.path.join(base, ANSWER_SOURCES[source] + "[.]*jsonl"))):
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
