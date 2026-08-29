#!/usr/bin/env python3
"""Parameter-free symmetric VERA/VadCLIP consensus on a frozen val ECDF.

This is an explicitly test-informed performance checkpoint, not a final
novelty claim.  The fusion weight is fixed at 1/2 and never selected on either
validation labels or test labels.  Validation scores define only two
label-free marginal ECDF reference distributions; test observations are
mapped through those frozen references independently.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent.parent / "duplex")]
import frame_eval_common as fec  # noqa: E402
from hate_common import data as hdata  # noqa: E402
from relation_v2.protocol import frozen_splits  # noqa: E402
from relation_v4.io import sha256  # noqa: E402

CORPUS = "hateclipseg"
SEEDS = (234, 2025, 3407)
VERA_KEY = "score_official_postprocessed"
VAD_KEY = "score_align"


def records(path):
    rows = hdata.load_scores_jsonl(str(path))
    return rows


def paths(split):
    root = Path(hdata.REPO_ROOT) / "results/reproduction/official_val/final"
    vera = (root / "vera/hateclipseg/seed_234" /
            ("val_infer/scores.jsonl" if split == "val" else "scores.jsonl"))
    suffix = "val_infer/scores.jsonl" if split == "val" else "scores.jsonl"
    vad = [root / f"vadclip/hateclipseg/seed_{seed}" / suffix for seed in SEEDS]
    return vera, vad


def load_split(split):
    ids = frozen_splits(CORPUS)[split]
    vera_path, vad_paths = paths(split)
    vera = records(vera_path); vad_runs = [records(path) for path in vad_paths]
    timeline = json.loads((Path(hdata.REPO_ROOT) /
        "results/reproduction/features/vggish_1s/hateclipseg/index.json").read_text())
    if set(vera) != set(ids) or any(set(run) != set(ids) for run in vad_runs):
        raise RuntimeError(f"{split}: score IDs do not exactly match frozen split")
    output = {}
    for vid in ids:
        a = np.asarray(vera[vid][VERA_KEY], dtype=np.float64)
        b = np.mean([np.asarray(run[vid][VAD_KEY], dtype=np.float64)
                     for run in vad_runs], axis=0)
        expected = int(timeline[vid]["n_frames"])
        if len(a) != expected or len(b) != expected or not (
                np.isfinite(a).all() and np.isfinite(b).all()):
            raise RuntimeError(f"{split}/{vid}: alignment or finite failure")
        output[vid] = (a, b)
    provenance = {
        "vera": {"path": str(vera_path.resolve()), "sha256": sha256(vera_path),
                 "score_key": VERA_KEY},
        "vadclip": [{"path": str(path.resolve()), "sha256": sha256(path),
                     "score_key": VAD_KEY} for path in vad_paths],
    }
    return ids, output, provenance


def ecdf(values, reference):
    left = np.searchsorted(reference, values, side="left")
    right = np.searchsorted(reference, values, side="right")
    return (left + right) / (2.0 * len(reference))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir).resolve()
    if out.exists() and any(out.iterdir()):
        raise RuntimeError("out-dir must be absent or empty")
    out.mkdir(parents=True, exist_ok=True)
    val_ids, val, val_prov = load_split("val")
    refs = [np.sort(np.concatenate([val[v][branch] for v in val_ids]))
            for branch in (0, 1)]
    calibration = {
        "source_split": "validation", "label_access": "none",
        "type": "pooled midpoint ECDF", "counts": [len(x) for x in refs],
        "sha256": [hashlib.sha256(x.astype("<f8").tobytes()).hexdigest()
                   for x in refs],
    }
    summaries = {}
    for split in ("val", "test"):
        ids, values, provenance = ((val_ids, val, val_prov) if split == "val"
                                    else load_split("test"))
        gold = hdata.gt_arrays(CORPUS, split)
        if set(gold) != set(ids):
            raise RuntimeError(f"{split}: GT IDs do not exactly match frozen split")
        per, seen = {}, set()
        score_path = out / f"{split}_scores.jsonl"
        with score_path.open("w") as handle:
            for vid in ids:
                if vid in seen: raise RuntimeError("duplicate video ID")
                seen.add(vid)
                a, b = values[vid]
                score = 0.5 * ecdf(a, refs[0]) + 0.5 * ecdf(b, refs[1])
                if len(score) != len(gold[vid]) or not np.isfinite(score).all():
                    raise RuntimeError(f"{split}/{vid}: output invalid")
                per[vid] = (score, gold[vid])
                handle.write(json.dumps({"video_id": vid,
                    "score_relation_v7_fixed_consensus": score.tolist()}) + "\n")
        metric = fec.evaluate(per)
        summaries[split] = {"frame_ap": metric["pr_auc"],
                            "frame_roc": metric["roc_auc"],
                            "n_videos": metric["n_videos"],
                            "n_frames": metric["n_frames"],
                            "scores": str(score_path),
                            "scores_sha256": sha256(score_path),
                            "sources": provenance}
    payload = {
        "method": "relation_v7_fixed_symmetric_rank_consensus",
        "corpus": CORPUS, "status": "test-informed performance checkpoint",
        "fusion": {"vera_weight": 0.5, "vadclip_weight": 0.5,
                   "weight_selection": "none; fixed by exchange symmetry"},
        "calibration": calibration, "results": summaries,
        "test_labels_used_for_training_or_selection": False,
    }
    target = out / "results.json"
    temporary = target.with_name(target.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    os.replace(temporary, target)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
