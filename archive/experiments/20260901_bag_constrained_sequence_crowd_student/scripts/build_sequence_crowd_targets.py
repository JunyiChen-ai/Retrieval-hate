#!/usr/bin/env python
"""Build train-only sequence-crowd posterior targets without frame labels."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
EXP = ROOT / "experiments/20260901_bag_constrained_sequence_crowd_student"
BASE = ROOT / "scripts/reproduction_baselines"
sys.path[:0] = [str(EXP), str(BASE), str(ROOT)]
from sequence_crowd import SequenceCrowdEM  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from src.hate_local_features import aligned_local_features  # noqa: E402


SOURCES = ("lexical", "powa", "vera", "multihateloc")


def scores_jsonl(path, branch):
    out = {}
    if not Path(path).is_file():
        return out
    with open(path) as handle:
        for line in handle:
            row = json.loads(line)
            if branch in row:
                out[row["video_id"]] = np.asarray(row[branch], np.float64)
    return out


def npz_scores(path):
    with np.load(path, allow_pickle=False) as record:
        return {key: np.asarray(record[key], np.float64) for key in record.files}


def ordinalize(raw, ids, lengths, n_bins):
    edges = {}; observations = {}
    for source in SOURCES:
        pool = [raw[source][v] for v in ids if v in raw[source]]
        if not pool:
            raise RuntimeError(f"source has no train observations: {source}")
        values = np.concatenate(pool)
        if not np.isfinite(values).all():
            raise RuntimeError(f"nonfinite source: {source}")
        edges[source] = np.quantile(values, np.arange(1, n_bins) / n_bins)
    for video_id in ids:
        obs = np.full((lengths[video_id], len(SOURCES)), -1, dtype=np.int64)
        for column, source in enumerate(SOURCES):
            if video_id not in raw[source]:
                continue
            value = raw[source][video_id]
            if value.shape != (lengths[video_id],):
                raise RuntimeError(f"{source}/{video_id} length {value.shape} != {lengths[video_id]}")
            obs[:, column] = np.digitize(value, edges[source], right=False)
        observations[video_id] = obs
    return observations, {k: v.tolist() for k, v in edges.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--lexical-npz", required=True)
    ap.add_argument("--powa", required=True)
    ap.add_argument("--vera", required=True)
    ap.add_argument("--multihateloc", required=True)
    ap.add_argument("--out-root", default=str(ROOT / "data/sequence_crowd_targets"))
    ap.add_argument("--n-bins", type=int, default=5)
    ap.add_argument("--iterations", type=int, default=20)
    args = ap.parse_args()
    labels = hdata.load_labels(args.corpus)
    train_ids, _ = hdata.load_train_val(args.corpus, labels, val_frac=.1, seed=234)
    lengths = {v: len(aligned_local_features(args.corpus, v)["audio"])
               for v in train_ids}
    raw = {
        "lexical": npz_scores(args.lexical_npz),
        "powa": scores_jsonl(args.powa, "score_powa"),
        "vera": scores_jsonl(args.vera, "score_official_postprocessed"),
        "multihateloc": scores_jsonl(args.multihateloc, "score_multihateloc"),
    }
    observations, edges = ordinalize(raw, train_ids, lengths, args.n_bins)
    specs = {
        "core": (True, True),
        "token_ds": (False, True),
        "unconstrained_bsc": (True, False),
    }
    root = Path(args.out_root) / args.corpus
    root.mkdir(parents=True, exist_ok=True)
    models = {}; targets = {}
    for arm, (sequential, bag_conditioned) in specs.items():
        model = SequenceCrowdEM(len(SOURCES), args.n_bins, sequential,
                                bag_conditioned, args.iterations)
        target = model.fit(observations, labels)
        np.savez_compressed(root / f"{arm}.npz", **target)
        targets[arm] = target
        models[arm] = model.serializable()
    rank_rows = []
    for video_id in train_ids:
        if labels[video_id] != 1:
            continue
        a, b = targets["core"][video_id], targets["token_ds"][video_id]
        if np.ptp(a) <= 1e-12 or np.ptp(b) <= 1e-12:
            continue
        rho = spearmanr(a, b).statistic
        if np.isfinite(rho):
            rank_rows.append(float(rho))
    if not rank_rows:
        raise RuntimeError("no eligible positive train video for posterior ordering diagnostic")
    coverage = {
        source: {"n_videos": sum(v in raw[source] for v in train_ids),
                 "n_seconds": sum(len(raw[source][v]) for v in train_ids if v in raw[source])}
        for source in SOURCES
    }
    payload = {"corpus": args.corpus, "split": "train-fit only",
               "train_ids": train_ids, "sources": list(SOURCES),
               "source_paths": {"lexical": args.lexical_npz, "powa": args.powa,
                                "vera": args.vera, "multihateloc": args.multihateloc},
               "coverage": coverage, "ordinal_edges": edges,
               "n_bins": args.n_bins, "iterations": args.iterations,
               "frame_labels_read": False, "validation_or_test_read": False,
               "core_vs_token_ds_positive_train_mean_spearman": float(np.mean(rank_rows)),
               "core_vs_token_ds_positive_train_n": len(rank_rows),
               "models": models}
    (root / "models.json").write_text(json.dumps(payload, indent=2) + "\n")
    (root / "PROVENANCE.md").write_text(
        "# Provenance\n\n"
        "Generated 2026-09-01 by `scripts/build_sequence_crowd_targets.py`. "
        "Inputs are the readable source paths in `models.json`; each source is "
        "parsed and checked for train-fit ID coverage, finite values and 1fps shape. "
        "No frame labels, validation predictions, test predictions or digests are read.\n")
    print(json.dumps({"corpus": args.corpus, "coverage": coverage,
                      "core_vs_token_ds_positive_train_mean_spearman": float(np.mean(rank_rows)),
                      "core_vs_token_ds_positive_train_n": len(rank_rows)},
                     indent=2))


if __name__ == "__main__":
    main()
