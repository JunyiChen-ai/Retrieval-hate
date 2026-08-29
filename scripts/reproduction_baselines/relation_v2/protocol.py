"""Fail-closed corpus/split/provenance checks for Relation-V2."""

from __future__ import annotations

import hashlib
import json
import os

from hate_common import data as hdata


SCOPED_LABEL_ROOT = os.path.join(hdata.SPLIT_ROOT, "scoped_labels")


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_ids(ids):
    return hashlib.sha256(("\n".join(ids) + "\n").encode()).hexdigest()


def assert_single_corpus(corpus):
    if not isinstance(corpus, str) or corpus not in hdata.CORPORA:
        raise ValueError("one corpus string is required, got %r" % (corpus,))
    return corpus


def frozen_splits(corpus):
    corpus = assert_single_corpus(corpus)
    splits = {s: hdata.load_split(corpus, s)
              for s in ("train", "val", "test")}
    for name, ids in splits.items():
        if len(ids) != len(set(ids)):
            raise RuntimeError("duplicate IDs in %s/%s" % (corpus, name))
    for a, b in (("train", "val"), ("train", "test"), ("val", "test")):
        overlap = set(splits[a]) & set(splits[b])
        if overlap:
            raise RuntimeError("split overlap %s %s/%s: %s" %
                               (corpus, a, b, sorted(overlap)[:5]))
    return splits


def scoped_labels(corpus, split):
    corpus = assert_single_corpus(corpus)
    if split not in ("train", "val", "test"):
        raise ValueError("invalid split %r" % split)
    path = os.path.join(SCOPED_LABEL_ROOT, "%s_%s.json" % (corpus, split))
    with open(path) as fh:
        payload = json.load(fh)
    ids = hdata.load_split(corpus, split)
    if payload.get("corpus") != corpus or payload.get("split") != split:
        raise RuntimeError("scoped-label identity mismatch: %s" % path)
    if payload.get("manifest_sha256") != sha256_ids(ids):
        raise RuntimeError("stale scoped labels for %s/%s" % (corpus, split))
    labels = payload.get("labels") or {}
    if set(labels) != set(ids):
        raise RuntimeError("scoped-label coverage mismatch for %s/%s" %
                           (corpus, split))
    if any(v not in (0, 1) for v in labels.values()):
        raise RuntimeError("non-binary scoped label")
    return {k: int(v) for k, v in labels.items()}, path


def verify_macil_init(corpus, checkpoint):
    """Require a same-corpus MACIL checkpoint and archived split identity."""
    corpus = assert_single_corpus(corpus)
    checkpoint = os.path.abspath(checkpoint)
    meta_path = os.path.join(os.path.dirname(checkpoint), "train_meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError("MACIL init lacks train_meta.json: %s" % checkpoint)
    with open(meta_path) as fh:
        meta = json.load(fh)
    args = meta.get("args") or {}
    trained_corpus = args.get("corpus", meta.get("corpus"))
    if trained_corpus != corpus:
        raise RuntimeError("cross-corpus init: %s checkpoint says %r" %
                           (corpus, trained_corpus))
    splits = frozen_splits(corpus)
    archived = meta.get("splits") or {}
    train_ids = archived.get("train_ids", meta.get("train_ids"))
    val_ids = archived.get("val_ids", meta.get("val_ids"))
    if train_ids is None or val_ids is None:
        raise RuntimeError("MACIL meta lacks exact train/val IDs: %s" % meta_path)
    if meta.get("method") != "macilsd" or args.get("modality") != "av":
        raise RuntimeError("init must be the reproduced MACIL-SD AV model")
    if set(train_ids) != set(splits["train"]):
        raise RuntimeError("MACIL init train IDs are not exact corpus train")
    if set(val_ids) != set(splits["val"]):
        raise RuntimeError("MACIL init val IDs are not exact corpus validation")
    if (set(train_ids) | set(val_ids)) & set(splits["test"]):
        raise RuntimeError("MACIL init contains test IDs")
    return {"checkpoint": checkpoint, "checkpoint_sha256": sha256_file(checkpoint),
            "meta": meta_path, "meta_sha256": sha256_file(meta_path)}


def verify_teacher(corpus, path):
    """Load only same-corpus train teacher rows; reject everything else."""
    corpus = assert_single_corpus(corpus)
    train_ids = set(frozen_splits(corpus)["train"])
    records = {}
    with open(path, encoding="utf-8") as fh:
        for line_number, line in enumerate(fh, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("corpus") != corpus:
                raise RuntimeError("teacher cross-corpus row at line %d" %
                                   line_number)
            vid = row.get("video_id")
            if vid not in train_ids:
                raise RuntimeError("teacher non-train ID at line %d: %r" %
                                   (line_number, vid))
            if vid in records:
                raise RuntimeError("duplicate teacher ID %r" % vid)
            records[vid] = row
    return records, {"path": os.path.abspath(path),
                     "sha256": sha256_file(path),
                     "n_records": len(records)}


def checkpoint_corpus(meta, requested):
    trained = meta.get("corpus")
    if trained != assert_single_corpus(requested):
        raise RuntimeError("checkpoint corpus %r != requested %r" %
                           (trained, requested))
