#!/usr/bin/env python
"""TERA Gate-0 — deterministic synthetic corpora for the sec 9 fixture battery.

Everything here is synthetic.  No file under `data/` or `/home/jehc223/data/` is
read, and no real label or span is ever touched.  The generated objects use the
EXACT registered schemas of appendix sec 2.2 and sec 2.9 (including the nested
`"ids": [[...]]` whole-video contract), so the production code path runs end to
end against them.

Registered fixture dimensions (appendix sec 9.1): Dv_fix = 32, Dt_fix = 16,
d_fix = 48, K = 30, V = 240, positives 40%, seed base 424242.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

DV_FIX = 32
DT_FIX = 16
K = 30
V_FIX = 240
POS_RATE = 0.4
DURATION = 60.0
SEALED_N = 17            # planted sealed ids in the corpus-spanning gold file


def unit_noise(rng, n, dim):
    z = rng.normal(size=(n, dim))
    return z / np.maximum(np.linalg.norm(z, axis=1, keepdims=True), 1e-12)


class SynthCorpus(object):
    """ids / labels / per-window visual features / text / durations / spans."""

    def __init__(self, ids, labels, seg, text, durations, spans):
        self.ids = list(ids)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.seg = np.asarray(seg, dtype=np.float32)          # [V, K, Dv]
        self.text = np.asarray(text, dtype=np.float32)        # [V, Dt]
        self.durations = dict(durations)
        self.spans = dict(spans)

    @property
    def whole(self):
        return self.seg.mean(axis=1)


def _base(rng, n, prefix, pos_rate=POS_RATE, dv=DV_FIX, dt=DT_FIX):
    """Base corpus: unit-norm visual noise per window, a signal-free text half.

    The text half is one fixed random vector shared by every video ("text half
    noise for all", appendix sec 9.2 F1).  It is deliberately NOT per-video
    noise: the registered representation L2-normalizes the text half to unit
    norm while A1's mean over K=30 normalized visual vectors has norm ~1/sqrt(K),
    so per-video text noise would swamp the diluted visual signal and no arm that
    consumes the mean-pooled representation could learn anything -- which would
    make F1's own O1 assertion (O1 reuses the A1 fold-trained head) unreachable
    for reasons that have nothing to do with temporal evidence.
    """
    ids = ["%s%04d" % (prefix, i) for i in range(n)]
    labels = (rng.random(n) < pos_rate).astype(np.int64)
    seg = unit_noise(rng, n * K, dv).reshape(n, K, dv).astype(np.float32)
    text = np.repeat(unit_noise(rng, 1, dt), n, axis=0).astype(np.float32)
    durations = {v: DURATION for v in ids}
    spans = {v: [] for v in ids}
    return ids, labels, seg, text, durations, spans


def _window_span(idx_lo, idx_hi, duration=DURATION):
    """Span covering exactly windows idx_lo..idx_hi-1 (positive-duration overlap)."""
    return [idx_lo * duration / K, idx_hi * duration / K]


def make_f1(rng, n=V_FIX, prefix="vid", amp=1.2):
    """Span-localized signal: windows {c, c+1} of every positive get +amp on e1."""
    ids, labels, seg, text, durations, spans = _base(rng, n, prefix)
    for i, vid in enumerate(ids):
        if labels[i] == 1:
            c = int(rng.integers(2, 27))
            seg[i, c, 0] += amp
            seg[i, c + 1, 0] += amp
            spans[vid] = [_window_span(c, c + 2)]
    return SynthCorpus(ids, labels, seg, text, durations, spans)


def make_f2(rng, n=V_FIX, prefix="vid", amp=0.35, base_mix=0.0):
    """Global signal only: every window of a positive gets +amp on e1.

    Unlike F1 this corpus gives every video a per-video base direction shared by
    its 30 windows (`base_mix` controls the within-video jitter; 0.0 makes the 30
    windows identical, so A0's l2n(mean) and A1's mean-of-l2n are the SAME
    vector and the two arms differ only by their independent training seeds).  Reason:
    with independent window noise the K-window mean A1 consumes has norm
    ~1/sqrt(K) against a unit-norm text block, so A1 sits ~0.08 macro-F1 below A0
    for a pure scale reason and F2's own `A1 >= A0 - 0.02` assertion cannot hold.
    Correlated windows are also the realistic case (adjacent CLIP windows of one
    video are highly correlated), and they make A0's and A1's inputs the same
    object up to normalization -- which is exactly the situation the assertion
    describes.  O1 is identical to A1 here by construction (the full-video span
    pools all K windows), so the no-headroom property is untouched.
    """
    ids = ["%s%04d" % (prefix, i) for i in range(n)]
    labels = (rng.random(n) < POS_RATE).astype(np.int64)
    base = unit_noise(rng, n, DV_FIX)[:, None, :]
    jitter = unit_noise(rng, n * K, DV_FIX).reshape(n, K, DV_FIX)
    seg = (base + base_mix * jitter).astype(np.float32)
    text = np.repeat(unit_noise(rng, 1, DT_FIX), n, axis=0).astype(np.float32)
    durations = {v: DURATION for v in ids}
    spans = {v: [] for v in ids}
    for i, vid in enumerate(ids):
        if labels[i] == 1:
            seg[i, :, 0] += amp
            spans[vid] = [[0.0, DURATION]]
    return SynthCorpus(ids, labels, seg, text, durations, spans)


def make_f3(rng, n=V_FIX, prefix="vid", amp=1.2, flip_rate=0.20):
    """F1 plus sign-flipped noise spikes on a random 20% of windows."""
    corpus = make_f1(rng, n, prefix, amp)
    mask = rng.random((len(corpus.ids), K)) < flip_rate
    corpus.seg[mask, 0] -= amp
    return corpus


def make_f4(rng, n=V_FIX, prefix="vid", amp=1.5, pos_rate=POS_RATE,
            reversed_share=0.20):
    """Ordered pair interaction: label 1 iff pA at i and pB at j with j - i >= 2.

    `reversed_share` splits the label-0 population between the two negative kinds
    the appendix allows ("j < i **or** one pattern only"); the share itself is not
    registered.  It is set to 0.20 because a 50/50 split makes the *number* of
    pattern windows uninformative about the video label, D's segment scores stay
    at chance, and the Gate-B arms then all consume noise -- i.e. the fixture
    would test nothing.  At 0.20 the weak selector can find the pattern windows
    from video labels alone, which is the precondition F4 assumes.
    """
    ids, labels, seg, text, durations, spans = _base(rng, n, prefix, pos_rate)
    pattern_windows = {}
    for i, vid in enumerate(ids):
        if labels[i] == 1:
            a = int(rng.integers(0, K - 2))
            b = int(rng.integers(a + 2, K))
            seg[i, a, 0] += amp
            seg[i, b, 1] += amp
            spans[vid] = [_window_span(a, a + 1), _window_span(b, b + 1)]
            pattern_windows[vid] = [a, b]
        else:
            if rng.random() < reversed_share:          # reversed order (j < i)
                a = int(rng.integers(0, K - 2))
                b = int(rng.integers(a + 2, K))
                seg[i, a, 1] += amp                    # pB first
                seg[i, b, 0] += amp                    # pA second
                pattern_windows[vid] = [a, b]
            else:                                      # one pattern only
                a = int(rng.integers(0, K))
                seg[i, a, int(rng.integers(0, 2))] += amp
                pattern_windows[vid] = [a]
    corpus = SynthCorpus(ids, labels, seg, text, durations, spans)
    corpus.pattern_windows = pattern_windows
    return corpus


def pattern_score_override(corpus, seed=0):
    """Per-window scores that rank the planted pattern windows first.

    Used only by the Gate-B fixtures through the registered fixture hook: on
    synthetic data the weakly supervised selector cannot learn the pattern-count
    contrast (see the fixture report), so without this the Gate-B arms would be
    fed random pairs and F4/F12 would test nothing.
    """
    rng = np.random.default_rng(seed)
    out = {}
    for vid in corpus.ids:
        scores = (rng.random(K) * 0.1).tolist()
        for w in corpus.pattern_windows.get(vid, []):
            scores[w] = 1.0 - 0.001 * w
        out[vid] = [float(x) for x in scores]
    return out


def make_f5(rng, n=V_FIX, prefix="vid", amp=1.5):
    """Single segment sufficient: label 1 iff pattern pA occurs anywhere."""
    ids, labels, seg, text, durations, spans = _base(rng, n, prefix)
    pattern_windows = {}
    for i, vid in enumerate(ids):
        a = int(rng.integers(0, K))
        if labels[i] == 1:
            seg[i, a, 0] += amp
            spans[vid] = [_window_span(a, a + 1)]
        else:
            seg[i, a, 1] += amp
        pattern_windows[vid] = [a]
    corpus = SynthCorpus(ids, labels, seg, text, durations, spans)
    corpus.pattern_windows = pattern_windows
    return corpus


def make_f7(rng, n=800, prefix="vid", zero_seg=3, zero_whole=2, missing_dur=2,
            spanless_pos=5):
    """Degenerate assets, kept at or below the registered >1% HALT rate."""
    corpus = make_f1(rng, n, prefix)
    info = {"zero_seg": [], "zero_whole": [], "missing_duration": [],
            "spanless_positive": []}
    cursor = 0
    for _ in range(zero_seg):
        vid = corpus.ids[cursor]
        corpus.seg[cursor] = 0.0
        info["zero_seg"].append(vid)
        cursor += 1
    zero_whole_ids = []
    for _ in range(zero_whole):
        vid = corpus.ids[cursor]
        zero_whole_ids.append(vid)
        info["zero_whole"].append(vid)
        cursor += 1
    for _ in range(missing_dur):
        vid = corpus.ids[cursor]
        corpus.durations[vid] = None
        info["missing_duration"].append(vid)
        cursor += 1
    taken = 0
    for i, vid in enumerate(corpus.ids):
        if taken >= spanless_pos:
            break
        if corpus.labels[i] == 1 and corpus.spans[vid] and vid not in info["zero_seg"]:
            corpus.spans[vid] = []
            info["spanless_positive"].append(vid)
            taken += 1
    corpus.zero_whole_ids = zero_whole_ids
    corpus.degenerate = info
    return corpus


def make_f7b(rng, n=V_FIX, prefix="vid", rate=0.05):
    """5% zero videos (union) -- must trip the >1% HALT rule."""
    corpus = make_f1(rng, n, prefix)
    n_zero = int(round(rate * n))
    for i in range(n_zero):
        corpus.seg[i] = 0.0
    corpus.degenerate = {"zero_seg": corpus.ids[:n_zero]}
    return corpus


# --------------------------------------------------------------- writers ----
def write_hatemm(root, train: SynthCorpus, val: SynthCorpus = None,
                 sealed_n=SEALED_N, zero_whole_ids=()):
    """Write the HateMM-shaped caches, split files and corpus-spanning spans."""
    root = Path(root)
    (root / "CLIP_Embedding/HateMM").mkdir(parents=True, exist_ok=True)
    (root / "gt/HateMM").mkdir(parents=True, exist_ok=True)

    def _write(corpus, seg_name, whole_name, jsonl_name):
        n = len(corpus.ids)
        seg = torch.tensor(corpus.seg.reshape(n * K, -1), dtype=torch.float32)
        parent = torch.tensor(np.repeat(np.arange(n), K), dtype=torch.long)
        seg_labels = torch.tensor(np.repeat(corpus.labels, K), dtype=torch.long)
        torch.save({"video_ids": list(corpus.ids), "subclip_img_feats": seg,
                    "subclip_parent": parent, "labels": seg_labels,
                    "num_subclips": K, "num_frames": 120},
                   root / "CLIP_Embedding/HateMM" / seg_name)
        whole = corpus.whole.copy()
        for vid in getattr(corpus, "zero_whole_ids", []):
            whole[corpus.ids.index(vid)] = 0.0
        torch.save({"ids": [list(corpus.ids)],
                    "img_feats": torch.tensor(whole, dtype=torch.float32),
                    "text_feats": torch.tensor(corpus.text, dtype=torch.float32),
                    "labels": torch.tensor(corpus.labels, dtype=torch.long)},
                   root / "CLIP_Embedding/HateMM" / whole_name)
        with open(root / "gt/HateMM" / jsonl_name, "w", encoding="utf-8") as handle:
            for vid, lab in zip(corpus.ids, corpus.labels):
                handle.write(json.dumps({"id": vid, "label": int(lab),
                                         "text": "(none)"}) + "\n")

    _write(train, "train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt",
           "train_openai_clip-vit-large-patch14-336_HF.pt", "train.jsonl")
    if val is not None:
        _write(val, "dev_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt",
               "dev_seen_openai_clip-vit-large-patch14-336_HF.pt", "val.jsonl")

    spans = {}
    for corpus in [c for c in (train, val) if c is not None]:
        for i, vid in enumerate(corpus.ids):
            spans[vid] = {"duration": corpus.durations[vid],
                          "spans": corpus.spans[vid], "label": int(corpus.labels[i])}
    # planted SEALED ids: present in the corpus-spanning gold file, in no split
    for i in range(sealed_n):
        spans["sealed%04d" % i] = {"duration": DURATION,
                                   "spans": [[1.0, 2.0]], "label": i % 2}
    with open(root / "gt/HateMM/hate_spans.json", "w", encoding="utf-8") as handle:
        json.dump(spans, handle)
    # the sealed test split: written once, never opened by the harness
    with open(root / "gt/HateMM/test.jsonl", "w", encoding="utf-8") as handle:
        for i in range(sealed_n):
            handle.write(json.dumps({"id": "sealed%04d" % i, "label": i % 2}) + "\n")
    return root


def write_hateclipseg(root, corpus: SynthCorpus, n_test=40):
    """Write a synthetic HateClipSeg corpus in the registered corpus-spanning form.

    The whole corpus (train + val + the sealed p11 test ids) lives in one set of
    files, exactly like the real 395-video subset, so the sec 2.8 restricted
    reader is exercised end to end.
    """
    root = Path(root)
    (root / "CLIP_Embedding/HateClipSeg").mkdir(parents=True, exist_ok=True)
    (root / "gt/HateClipSeg").mkdir(parents=True, exist_ok=True)
    n = len(corpus.ids)
    seg = torch.tensor(corpus.seg.reshape(n * K, -1), dtype=torch.float32)
    parent = torch.tensor(np.repeat(np.arange(n), K), dtype=torch.long)
    torch.save({"video_ids": list(corpus.ids), "subclip_img_feats": seg,
                "subclip_parent": parent,
                "labels": torch.tensor(np.repeat(corpus.labels, K), dtype=torch.long),
                "num_subclips": K, "num_frames": 120},
               root / "CLIP_Embedding/HateClipSeg/test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt")
    torch.save({"ids": [list(corpus.ids)],
                "img_feats": torch.tensor(corpus.whole, dtype=torch.float32),
                "text_feats": torch.tensor(corpus.text, dtype=torch.float32),
                "labels": torch.tensor(corpus.labels, dtype=torch.long)},
               root / "CLIP_Embedding/HateClipSeg/test_seen_openai_clip-vit-large-patch14-336_HF.pt")
    gold = {}
    for i, vid in enumerate(corpus.ids):
        multihot = [0, int(corpus.labels[i]), 0, 0, 0, 0]
        gold[vid] = {"duration": DURATION, "platform": "synthetic", "n_segments": 1,
                     "segments": [[0.0, DURATION, multihot]]}
    with open(root / "gt/HateClipSeg/gold_segments.json", "w", encoding="utf-8") as handle:
        json.dump(gold, handle)
    with open(root / "gt/HateClipSeg/video_durations.jsonl", "w", encoding="utf-8") as handle:
        for vid in corpus.ids:
            handle.write(json.dumps({"id": vid, "duration": DURATION}) + "\n")
    ids = list(corpus.ids)
    test = ids[-n_test:]
    rest = ids[:-n_test]
    cut = int(round(0.75 * len(rest)))
    with open(root / "gt/HateClipSeg/p11_split.json", "w", encoding="utf-8") as handle:
        json.dump({"train": sorted(rest[:cut]), "val": sorted(rest[cut:]),
                   "test": sorted(test)}, handle)
    return root


def build_dataset(kind, seed, out_dir, with_val=True, n=None, hateclipseg=False):
    """Materialize one fixture dataset; returns (path, train corpus, val corpus)."""
    rng = np.random.default_rng(seed)
    maker = {"F1": make_f1, "F2": make_f2, "F3": make_f3, "F4": make_f4,
             "F5": make_f5, "F7": make_f7, "F7b": make_f7b}[kind]
    train = maker(rng) if n is None else maker(rng, n=n)
    val = None
    if with_val:
        val_maker = {"F7": make_f1, "F7b": make_f1}.get(kind, maker)
        val = val_maker(np.random.default_rng(seed + 500), n=60, prefix="val")
        # the confirmation refit is a TRAIN-fitted model scored on val, so the
        # signal-free text half must be the same constant vector in both splits.
        val.text = np.repeat(train.text[:1], len(val.ids), axis=0).astype(np.float32)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    write_hatemm(out_dir, train, val)
    if hateclipseg:
        hcs = maker(np.random.default_rng(seed + 900), n=200, prefix="hcs")
        hcs.text = np.repeat(train.text[:1], len(hcs.ids), axis=0).astype(np.float32)
        write_hateclipseg(out_dir, hcs)
    return out_dir, train, val
