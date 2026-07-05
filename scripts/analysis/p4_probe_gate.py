#!/usr/bin/env python
"""P4 probe gate (CPU, run BEFORE any training).

For each dataset, on the TRAIN split only:
  (a) DECODABILITY -- can each archive field be linearly decoded from the frozen
      whole-video CLIP representation (concat[img,text])? 5-fold stratified CV
      logistic probe vs the field's majority-class baseline. If a field cannot be
      decoded at all, an aux head cannot learn it.
  (b) LABEL-INFORMATIVENESS -- do the archive fields carry video-label signal?
      5-fold CV logistic regression from the field encodings (one/multi-hot) to the
      hateful label vs majority. If the fields say nothing about the label, distilling
      them cannot help the label task.

Gate: if BOTH (a) and (b) fail on a dataset, the aux loss is noise there -> gate CLOSED
(do not train, report). Freezes the field vocabulary to p4_out/field_vocab_<ds>.json.

Reads only the CLIP caches + TRAIN archives; writes only p4_out/. Deterministic.
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src"))

import argparse
import json

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from utils.p4_archive_fields import (
    load_archive_records, derive_vocab, freeze_vocab, encode_record)

MODEL = "openai_clip-vit-large-patch14-336_HF"
ROOT = "/data/jehc223/RGCL"


def load_train_clip(ds):
    d = torch.load("{}/data/CLIP_Embedding/{}/train_{}.pt".format(ROOT, ds, MODEL),
                   map_location="cpu")
    ids = [i for sub in d["ids"] for i in sub]
    img = d["img_feats"].float().numpy()
    txt = d["text_feats"].float().numpy()
    lab = d["labels"].long().numpy()
    return ids, np.concatenate([img, txt], axis=1), lab


def probe_single(X, y, valid, n_splits=5, C=0.5):
    """Multinomial CV probe for a single-label field. Returns metrics vs majority."""
    Xv, yv = X[valid], y[valid]
    classes, counts = np.unique(yv, return_counts=True)
    maj_acc = counts.max() / len(yv)
    if len(classes) < 2:
        return {"n": int(len(yv)), "majority_acc": round(float(maj_acc), 4),
                "probe_acc": None, "macro_f1": None, "note": "single class"}
    skf = StratifiedKFold(n_splits=min(n_splits, counts.min()), shuffle=True,
                          random_state=0)
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(C=C, max_iter=3000, multi_class="auto"))
    pred = cross_val_predict(clf, Xv, yv, cv=skf)
    return {"n": int(len(yv)), "majority_acc": round(float(maj_acc), 4),
            "probe_acc": round(float(accuracy_score(yv, pred)), 4),
            "macro_f1": round(float(f1_score(yv, pred, average="macro")), 4),
            "beats_majority": bool(accuracy_score(yv, pred) > maj_acc)}


def probe_binary(X, y, valid, n_splits=5, C=0.5, min_pos=10):
    """CV AUC + accuracy vs majority for one binary label column."""
    Xv, yv = X[valid], y[valid].astype(int)
    npos = int(yv.sum())
    maj_acc = max(npos, len(yv) - npos) / len(yv)
    if npos < min_pos or npos > len(yv) - min_pos:
        return {"n": int(len(yv)), "pos": npos, "majority_acc": round(float(maj_acc), 4),
                "auc": None, "probe_acc": None, "note": "too rare/common for CV"}
    skf = StratifiedKFold(n_splits=min(n_splits, npos, len(yv) - npos), shuffle=True,
                          random_state=0)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
    proba = cross_val_predict(clf, Xv, yv, cv=skf, method="predict_proba")[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {"n": int(len(yv)), "pos": npos, "majority_acc": round(float(maj_acc), 4),
            "auc": round(float(roc_auc_score(yv, proba)), 4),
            "probe_acc": round(float(accuracy_score(yv, pred)), 4),
            "beats_majority": bool(accuracy_score(yv, pred) > maj_acc),
            "auc_beats_chance": bool(roc_auc_score(yv, proba) > 0.55)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="MHC,MHC_zh")
    ap.add_argument("--version", default="v2")
    ap.add_argument("--out", default="scripts/analysis/p4_out/probe_gate.json")
    args = ap.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    result = {}
    for ds in datasets:
        ids, X, ylab = load_train_clip(ds)
        recs = load_archive_records(ds, "train", args.version)
        vocab = derive_vocab(recs)
        vpath = "scripts/analysis/p4_out/field_vocab_{}.json".format(ds)
        freeze_vocab(vocab, vpath)

        # encode fields aligned to CLIP cache id order
        enc = [encode_record(recs.get(i), vocab) for i in ids]
        N = len(ids)

        # ---- (a) decodability of each field from CLIP concat ----
        decode = {}
        # explicitness (single-label)
        ex_t = np.array([e["explicitness"][0] for e in enc])
        ex_v = np.array([e["explicitness"][1] for e in enc])
        decode["explicitness"] = {"single": probe_single(X, ex_t, ex_v)}
        # multi-label fields
        for fld in ("modality", "mechanism", "target_group"):
            classes = vocab[fld]["classes"]
            T = np.array([e[fld][0] for e in enc], dtype=float)  # [N, C]
            V = np.array([e[fld][1] for e in enc])
            per_label = {}
            for j, cname in enumerate(classes):
                per_label[cname] = probe_binary(X, T[:, j], V)
            decode[fld] = {"per_label": per_label,
                           "coverage": vocab[fld].get("coverage")}

        # ---- (b) label-informativeness of the field encodings ----
        feats = []
        for e in enc:
            ex_oh = [0.0, 0.0, 0.0]
            if e["explicitness"][1]:
                ex_oh[e["explicitness"][0]] = 1.0
            row = ex_oh + list(e["modality"][0]) + list(e["mechanism"][0]) + \
                list(e["target_group"][0])
            feats.append(row)
        F = np.asarray(feats, dtype=float)
        npos = int(ylab.sum())
        maj_acc = max(npos, N - npos) / N
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1.0, max_iter=3000))
        proba = cross_val_predict(clf, F, ylab, cv=skf, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        informativeness = {
            "n": N, "pos": npos, "majority_acc": round(float(maj_acc), 4),
            "auc": round(float(roc_auc_score(ylab, proba)), 4),
            "probe_acc": round(float(accuracy_score(ylab, pred)), 4),
            "macro_f1": round(float(f1_score(ylab, pred, average="macro")), 4),
            "beats_majority": bool(accuracy_score(ylab, pred) > maj_acc),
            "auc_beats_chance": bool(roc_auc_score(ylab, proba) > 0.55),
        }

        # ---- gate decision ----
        # (a) passes if at least one field is clearly decodable (single beats majority,
        #     or any multi-label class AUC beats chance).
        a_signals = []
        s = decode["explicitness"]["single"]
        if s.get("beats_majority"):
            a_signals.append("explicitness")
        for fld in ("modality", "mechanism", "target_group"):
            for cname, m in decode[fld]["per_label"].items():
                if m.get("auc_beats_chance"):
                    a_signals.append("{}:{}".format(fld, cname))
        a_pass = len(a_signals) > 0
        b_pass = bool(informativeness["auc_beats_chance"] and
                      informativeness["beats_majority"])
        gate_open = not (not a_pass and not b_pass)  # closed only if BOTH fail

        result[ds] = {
            "vocab": vocab, "vocab_path": vpath,
            "decodability_a": decode,
            "informativeness_b": informativeness,
            "gate": {"a_pass": a_pass, "a_signals": a_signals, "b_pass": b_pass,
                     "open": gate_open},
        }

        print("\n===== P4 probe gate :: {} (train n={}, pos={}) =====".format(ds, N, npos))
        print("(a) decodability from frozen CLIP concat[img,text]:")
        print("    explicitness: probe_acc={} vs maj={} {}".format(
            s.get("probe_acc"), s.get("majority_acc"),
            "BEATS" if s.get("beats_majority") else "no"))
        for fld in ("modality", "mechanism", "target_group"):
            print("    {} (cover={}):".format(fld, decode[fld]["coverage"]))
            for cname, m in decode[fld]["per_label"].items():
                print("        {:16s} pos={:<4} auc={} acc={} vs maj={} {}".format(
                    cname, m.get("pos"), m.get("auc"), m.get("probe_acc"),
                    m.get("majority_acc"),
                    "AUC>0.55" if m.get("auc_beats_chance") else
                    (m.get("note") or "flat")))
        print("(b) field-encoding -> label: auc={} acc={} macroF1={} vs maj_acc={} -> {}".format(
            informativeness["auc"], informativeness["probe_acc"],
            informativeness["macro_f1"], informativeness["majority_acc"],
            "INFORMATIVE" if b_pass else "not"))
        print("GATE {}: a_pass={} b_pass={}".format(
            "OPEN" if gate_open else "CLOSED", a_pass, b_pass))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print("\n[out] wrote", args.out)


if __name__ == "__main__":
    main()
