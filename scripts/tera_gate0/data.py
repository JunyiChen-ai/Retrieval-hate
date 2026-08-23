#!/usr/bin/env python
"""TERA Gate-0 — corpus loading, K=30 segment representations, failure accounting.

Implements appendix sec 2.2 (cache schemas and the nested `ids` read convention),
sec 2.3 (registered representations), sec 2.6 (durations/window boundaries),
sec 2.7 (failure accounting and the >1% HALT), sec 2.9 (canonical split source)
and the HateClipSeg binding-endpoint derivation of sec 7.10.2.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from .common import K_WINDOWS, TeraHalt, sha256_file, sha256_ids
from .guards import active_guard, load_corpus_spanning


def hash_under_guard(path):
    """SHA256 of a (possibly corpus-spanning) artifact.

    Provenance hashing reads bytes only and never deserializes an id or a label,
    but it does open the file, so it runs inside the sec 2.8 reader scope.
    """
    guard = active_guard()
    if guard is None:
        return sha256_file(path)
    with guard.reader_scope():
        return sha256_file(path)

SEG_CACHE = {
    ("HateMM", "train"): "CLIP_Embedding/HateMM/train_subclipK30_openai_clip-vit-large-patch14-336_HF.pt",
    ("HateMM", "val"): "CLIP_Embedding/HateMM/dev_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt",
    ("HateClipSeg", "all"): "CLIP_Embedding/HateClipSeg/test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt",
}
WHOLE_CACHE = {
    ("HateMM", "train"): "CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt",
    ("HateMM", "val"): "CLIP_Embedding/HateMM/dev_seen_openai_clip-vit-large-patch14-336_HF.pt",
    ("HateClipSeg", "all"): "CLIP_Embedding/HateClipSeg/test_seen_openai_clip-vit-large-patch14-336_HF.pt",
}
CORPUS_SPANNING_CACHE = {("HateClipSeg", "all")}


def l2n(z: torch.Tensor) -> torch.Tensor:
    """l2n(z) = z / max(||z||_2, 1e-12); l2n(0) = 0 (appendix sec 1)."""
    norm = z.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-12)
    return z / norm


@dataclass
class Corpus:
    dataset: str
    split: str
    ids: list
    labels: np.ndarray
    X_seg: torch.Tensor          # [V, K, d]
    X_whole: torch.Tensor        # [V, d]
    durations: dict
    spans: dict
    zero_vector_ids: list
    missing_duration_ids: list
    d: int
    dims: dict
    cache_info: dict = field(default_factory=dict)
    split_source: str = ""
    split_source_sha256: str = ""

    @property
    def index(self):
        return {v: i for i, v in enumerate(self.ids)}

    @property
    def n(self):
        return len(self.ids)

    def label_of(self, vid):
        return int(self.labels[self.index[vid]])


# ------------------------------------------------------------- raw readers --
def _read_segment_cache(path, dataset, auth, corpus_spanning):
    if corpus_spanning:
        obj = load_corpus_spanning(path, dataset, auth)
    else:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    vids = list(obj["video_ids"])                      # FLAT list (sec 2.2)
    k = int(obj["num_subclips"])
    if k != K_WINDOWS:
        raise TeraHalt("HALT_MISSING_ASSET",
                       "%s has num_subclips=%d, registered K=%d" % (path, k, K_WINDOWS))
    feats = obj["subclip_img_feats"].float()
    parent = obj["subclip_parent"].numpy().astype(np.int64)
    seg_labels = obj["labels"].numpy().astype(np.int64)
    v = len(vids)
    if feats.shape[0] != v * k:
        raise TeraHalt("HALT_MISSING_ASSET", "%s: %d rows for %d videos x %d windows"
                       % (path, feats.shape[0], v, k))
    img = torch.zeros(v, k, feats.shape[1], dtype=torch.float32)
    seen = np.zeros(v, dtype=np.int64)
    for row in range(feats.shape[0]):
        p = int(parent[row])
        img[p, seen[p]] = feats[row]
        seen[p] += 1
    if not (seen == k).all():
        raise TeraHalt("HALT_MISSING_ASSET", "%s: parent index does not tile K windows" % path)
    labels = np.zeros(v, dtype=np.int64)
    for row in range(feats.shape[0]):
        labels[int(parent[row])] = int(seg_labels[row])
    info = {"num_subclips": k, "num_frames": int(obj.get("num_frames", -1))}
    return vids, img, labels, info


def _read_whole_cache(path, dataset, auth, corpus_spanning):
    if corpus_spanning:
        obj = load_corpus_spanning(path, dataset, auth)
    else:
        obj = torch.load(path, map_location="cpu", weights_only=False)
    vids = list(obj["ids"][0])                          # NESTED ids (sec 2.2, review F-1)
    img = obj["img_feats"].float()
    txt = obj["text_feats"].float()
    labels = obj["labels"].numpy().astype(np.int64)
    return vids, img, txt, labels


# ------------------------------------------------------------ split source --
def resolve_hatemm_split(data_root, split, whole_ids, whole_labels):
    """Appendix sec 2.9 resolution order; any mismatch is HALT_SPLIT_MISMATCH."""
    jsonl = Path(data_root) / "gt/HateMM" / ("%s.jsonl" % split)
    if jsonl.exists():
        ids, labels = [], {}
        with open(jsonl, encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                ids.append(rec["id"])
                labels[rec["id"]] = int(rec["label"])
        if set(ids) != set(whole_ids):
            raise TeraHalt("HALT_SPLIT_MISMATCH",
                           "%s ids differ from cache['ids'][0]" % jsonl)
        cache_label = {v: int(l) for v, l in zip(whole_ids, whole_labels)}
        for vid in ids:
            if labels[vid] != cache_label[vid]:
                raise TeraHalt("HALT_SPLIT_MISMATCH", "label mismatch for %s" % vid)
        return "gt_jsonl", str(jsonl), sha256_file(jsonl), labels
    labels = {v: int(l) for v, l in zip(whole_ids, whole_labels)}
    return "feature_cache_embedded", "", "", labels


# ------------------------------------------------------------------ loader --
def load_corpus(data_root, dataset, split, auth, spans_required=True):
    """Load one partition and build the registered representations (sec 2.3)."""
    data_root = Path(data_root)
    key = (dataset, split if dataset == "HateMM" else "all")
    seg_path = data_root / SEG_CACHE[key]
    whole_path = data_root / WHOLE_CACHE[key]
    for path in (seg_path, whole_path):
        if not path.exists():
            raise TeraHalt("HALT_MISSING_ASSET", str(path))
    corpus_spanning = key in CORPUS_SPANNING_CACHE

    seg_ids, seg_img, seg_labels, seg_info = _read_segment_cache(
        seg_path, dataset, auth, corpus_spanning)
    whole_ids, whole_img, whole_txt, whole_labels = _read_whole_cache(
        whole_path, dataset, auth, corpus_spanning)
    if set(seg_ids) != set(whole_ids):
        raise TeraHalt("HALT_MISSING_ASSET",
                       "segment/whole-video id sets differ for %s/%s" % (dataset, split))

    if dataset == "HateMM":
        split_source, split_path, split_sha, label_map = resolve_hatemm_split(
            data_root, "train" if split == "train" else "val", whole_ids, whole_labels)
    else:
        split_source, split_path, split_sha = "hateclipseg_binding_endpoint", "", ""
        label_map = hateclipseg_endpoint(data_root, auth)
        missing = set(whole_ids) - set(label_map)
        if missing:
            raise TeraHalt("HALT_MISSING_ASSET",
                           "%d HateClipSeg ids lack a gold endpoint" % len(missing))

    ids = sorted(set(whole_ids))
    seg_pos = {v: i for i, v in enumerate(seg_ids)}
    whole_pos = {v: i for i, v in enumerate(whole_ids)}
    seg_sel = torch.as_tensor([seg_pos[v] for v in ids], dtype=torch.long)
    whole_sel = torch.as_tensor([whole_pos[v] for v in ids], dtype=torch.long)

    seg_img = seg_img[seg_sel]                    # [V, K, Dv]
    whole_img = whole_img[whole_sel]              # [V, Dv]
    whole_txt = whole_txt[whole_sel]              # [V, Dt]
    labels = np.array([label_map[v] for v in ids], dtype=np.int64)

    dv, dt = seg_img.shape[-1], whole_txt.shape[-1]
    if whole_img.shape[-1] != dv:
        raise TeraHalt("HALT_MISSING_ASSET", "segment/whole visual dims differ")
    text_block = l2n(whole_txt).unsqueeze(1).expand(-1, seg_img.shape[1], -1)
    x_seg = torch.cat([l2n(seg_img), text_block], dim=-1).contiguous()
    x_whole = torch.cat([l2n(whole_img), l2n(whole_txt)], dim=-1).contiguous()

    # failure accounting (sec 2.7) -- union across BOTH caches, computed on raw feats
    seg_zero = (seg_img.abs().sum(dim=(1, 2)) == 0).numpy()
    whole_zero = (whole_img.abs().sum(dim=1) == 0).numpy()
    zero_ids = [ids[i] for i in range(len(ids)) if bool(seg_zero[i] or whole_zero[i])]

    durations, spans = load_gold_temporal(data_root, dataset, auth, ids,
                                          required=spans_required)
    missing_dur = [v for v in ids if durations.get(v) is None or
                   (durations.get(v) is not None and durations[v] <= 0)]

    cache_info = {
        "segment_cache": {"path": str(seg_path), "sha256": hash_under_guard(seg_path),
                          "bytes": seg_path.stat().st_size, **seg_info},
        "wholevideo_cache": {"path": str(whole_path),
                             "sha256": hash_under_guard(whole_path),
                             "bytes": whole_path.stat().st_size,
                             "num_frames_status": "provenance_only"},
        "split_source_path": split_path,
    }
    return Corpus(dataset=dataset, split=split, ids=ids, labels=labels,
                  X_seg=x_seg, X_whole=x_whole, durations=durations, spans=spans,
                  zero_vector_ids=zero_ids, missing_duration_ids=missing_dur,
                  d=int(x_whole.shape[-1]),
                  dims={"Dv_observed": int(dv), "Dt_observed": int(dt),
                        "d_observed": int(dv + dt)},
                  cache_info=cache_info, split_source=split_source,
                  split_source_sha256=split_sha)


# ------------------------------------------------------------ gold temporal --
def load_gold_temporal(data_root, dataset, auth, ids, required=True):
    """Durations and gold spans, always through the sec 2.8 restricted reader."""
    data_root = Path(data_root)
    if dataset == "HateMM":
        path = data_root / "gt/HateMM/hate_spans.json"
        if not path.exists():
            if required:
                raise TeraHalt("HALT_MISSING_ASSET", str(path))
            return {v: None for v in ids}, {v: [] for v in ids}
        gold = load_corpus_spanning(path, "HateMM", auth)
        durations, spans = {}, {}
        for vid in ids:
            rec = gold.get(vid, {})
            dur = rec.get("duration", None)
            durations[vid] = float(dur) if dur is not None else None
            spans[vid] = [[float(a), float(b)] for a, b in rec.get("spans", [])]
        return durations, spans
    path = data_root / "gt/HateClipSeg/video_durations.jsonl"
    if not path.exists():
        if required:
            raise TeraHalt("HALT_MISSING_ASSET", str(path))
        return {v: None for v in ids}, {v: [] for v in ids}
    dur_recs = load_corpus_spanning(path, "HateClipSeg", auth)
    durations = {}
    for vid in ids:
        rec = dur_recs.get(vid, {})
        dur = rec.get("duration", None)
        durations[vid] = float(dur) if dur is not None else None
    # prereg sec 8.2's temporal criterion is HateMM-only: no HateClipSeg spans are read.
    return durations, {v: [] for v in ids}


def hateclipseg_endpoint(data_root, auth):
    """Binding endpoint: has >=1 segment labelled hateful (multi-hot index 1).

    Returns a single int per video; the segment list is discarded before
    returning (appendix sec 7.10.2 weak-supervision statement).
    """
    path = Path(data_root) / "gt/HateClipSeg/gold_segments.json"
    if not path.exists():
        raise TeraHalt("HALT_MISSING_ASSET", str(path))
    gold = load_corpus_spanning(path, "HateClipSeg", auth)
    out = {}
    for vid, rec in gold.items():
        out[vid] = int(any(int(seg[2][1]) == 1 for seg in rec.get("segments", [])))
    del gold
    return out


def read_p11_split(data_root):
    path = Path(data_root) / "gt/HateClipSeg/p11_split.json"
    if not path.exists():
        raise TeraHalt("HALT_MISSING_ASSET", str(path))
    with open(path, encoding="utf-8") as handle:
        obj = json.load(handle)
    return ({k: list(obj[k]) for k in ("train", "val", "test")},
            sha256_file(path))


def read_hatemm_ids(data_root, split):
    path = Path(data_root) / "gt/HateMM" / ("%s.jsonl" % split)
    if not path.exists():
        raise TeraHalt("HALT_MISSING_ASSET", str(path))
    ids = []
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                ids.append(json.loads(line)["id"])
    return ids


# -------------------------------------------------------- failure accounting --
def failure_report(corpus, halt_on_rate=True):
    """Appendix sec 2.7: HALT if |zero union missing-duration| / V > 0.01."""
    union = sorted(set(corpus.zero_vector_ids) | set(corpus.missing_duration_ids))
    rate = len(union) / max(1, corpus.n)
    report = {
        "zero_vector_videos": len(corpus.zero_vector_ids),
        "zero_vector_ids": sorted(corpus.zero_vector_ids),
        "missing_duration_videos": len(corpus.missing_duration_ids),
        "missing_duration_ids": sorted(corpus.missing_duration_ids),
        "union": len(union),
        "failure_rate": rate,
        "halt_threshold": 0.01,
    }
    if halt_on_rate and rate > 0.01:
        raise TeraHalt("HALT_DECODE_FAILURE_RATE",
                       "%d/%d = %.4f > 0.01" % (len(union), corpus.n, rate))
    return report


def split_manifest(corpus):
    return {
        "dataset": corpus.dataset,
        "split": corpus.split,
        "split_source": corpus.split_source,
        "split_source_sha256": corpus.split_source_sha256,
        "split_id_hash": sha256_ids(corpus.ids),
        "n_videos": corpus.n,
        "n_positive": int((corpus.labels == 1).sum()),
        "n_negative": int((corpus.labels == 0).sum()),
        "dims": corpus.dims,
        "caches": corpus.cache_info,
    }
