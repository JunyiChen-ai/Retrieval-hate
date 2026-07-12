#!/usr/bin/env python
"""Logical upper bound for the preregistered B1 dual-metric oracle gate.

Every selected pre-MLLM candidate arc is optimistically treated as accepted.
The true accepted-relation touched set is necessarily a subset, so failure of
this ceiling proves that relation extraction cannot rescue the B1 oracle gate.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from ssr_common import (  # noqa: E402
    atomic_write_json, canonical_json, load_config, read_jsonl, resolve,
    sha256_file, sha256_obj,
)


def load_predictions(cfg, dataset):
    rows, sources = [], {}
    for fold in range(int(cfg["folds"]["n_splits"])):
        d = resolve(cfg, "artifacts") / "oof" / dataset / "fold{}".format(fold)
        manifest = json.load(open(d / "manifest.json", encoding="utf-8"))
        path = d / "predictions.json"
        if manifest["outputs"]["predictions.json"] != sha256_file(path):
            raise RuntimeError("prediction hash mismatch")
        rows.extend(json.load(open(path, encoding="utf-8")))
        sources[str(fold)] = sha256_file(path)
    by_id = {x["query_id"]: x for x in rows}
    folds = json.load(open(resolve(cfg, "artifacts") / "folds" /
                           "{}.json".format(dataset), encoding="utf-8"))
    if len(by_id) != len(rows) or set(by_id) != {x["id"] for x in folds["records"]}:
        raise RuntimeError("OOF prediction coverage mismatch")
    return by_id, sources


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SSR computation must run under SLURM")
    cfg = load_config(args.config)
    threshold = float(cfg["b1"]["oracle_gain_min"])
    cells, sources = [], {}
    for dataset in ("MHC", "MHC_zh"):
        pair_root = resolve(cfg, "artifacts") / "pairs" / dataset
        manifest = json.load(open(pair_root / "manifest.json", encoding="utf-8"))
        arcs = read_jsonl(pair_root / "arcs.jsonl")
        if manifest["outputs"]["arcs.jsonl"] != sha256_file(pair_root / "arcs.jsonl"):
            raise RuntimeError("arc hash mismatch")
        preds, pred_sources = load_predictions(cfg, dataset)
        sources[dataset] = {"arcs": sha256_file(pair_root / "arcs.jsonl"),
                            "predictions": pred_sources}
        ordered = sorted(preds)
        y = np.asarray([int(preds[x]["query_label"]) for x in ordered])
        base = np.asarray([int(preds[x]["prediction"]) for x in ordered])
        base_acc = accuracy_score(y, base)
        base_f1 = f1_score(y, base, average="macro", zero_division=0)
        for family in ("MI", "SC"):
            positive_arcs = [x for x in arcs if x["candidate_family"] == family
                             and int(x["event"]) == 1]
            touched = {x["query_id"] for x in positive_arcs}
            oracle = base.copy()
            for i, qid in enumerate(ordered):
                if qid in touched:
                    oracle[i] = y[i]
            oracle_acc = accuracy_score(y, oracle)
            oracle_f1 = f1_score(y, oracle, average="macro", zero_division=0)
            gain_acc, gain_f1 = oracle_acc - base_acc, oracle_f1 - base_f1
            cells.append({
                "dataset": dataset, "family": family,
                "optimistic_assumption": "every_selected_candidate_is_an_accepted_reliable_relation",
                "n_candidate_arcs": sum(x["candidate_family"] == family for x in arcs),
                "n_positive_event_arcs": len(positive_arcs),
                "n_unique_event_touched_errors": len(touched),
                "n_oof_videos": len(ordered),
                "minimum_touched_for_accuracy_gate": int(math.ceil(threshold * len(ordered) - 1e-12)),
                "baseline_accuracy": base_acc, "upper_oracle_accuracy": oracle_acc,
                "upper_accuracy_gain": gain_acc,
                "baseline_macro_f1": base_f1, "upper_oracle_macro_f1": oracle_f1,
                "upper_macro_f1_gain": gain_f1,
                "dual_metric_upper_bound_pass": gain_acc >= threshold and gain_f1 >= threshold,
            })
    common_upper = [family for family in ("MI", "SC") if all(
        next(x for x in cells if x["dataset"] == dataset and x["family"] == family)
        ["dual_metric_upper_bound_pass"] for dataset in ("MHC", "MHC_zh"))]
    decision = "CONTINUE_EXTRACTION" if common_upper else "STOP_SSR"
    result = {
        "run_id": args.run_id, "status": decision,
        "logic": "true accepted relation touched queries are a subset of the all-candidates event-positive touched set",
        "oracle_gain_min_each_metric": threshold,
        "cells": cells, "common_families_surviving_upper_bound": common_upper,
        "relation_extraction_can_change_this_bound": False,
        "only_gold": "video_level_binary_label", "segment_gold_used": False,
        "B2_B3_locked": True, "source_hashes": sources,
        "config_sha256": cfg["computed_config_sha256"],
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    result["payload_sha256"] = sha256_obj(result)
    out = resolve(cfg, "artifacts") / "b1" / "preflight_oracle_upper_bound.json"
    atomic_write_json(out, result)
    print(canonical_json(result))


if __name__ == "__main__":
    main()
