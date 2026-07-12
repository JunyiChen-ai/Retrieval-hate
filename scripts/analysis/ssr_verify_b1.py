#!/usr/bin/env python
"""Machine-check every frozen B1 gate and emit the only B2 unlock artifact."""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from ssr_common import (  # noqa: E402
    atomic_write_json, canonical_json, load_config, resolve, sha256_file,
    sha256_obj,
)


def holm_adjust(raw):
    m = len(raw)
    order = sorted(range(m), key=lambda i: raw[i])
    adjusted = [1.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * float(raw[idx]))
        adjusted[idx] = min(1.0, running)
    return adjusted


def maybe_json(path):
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SSR computation must run under SLURM")
    cfg = load_config(args.config)
    preflight_path = resolve(cfg, "artifacts") / "b1" / "preflight_oracle_upper_bound.json"
    preflight = maybe_json(preflight_path)
    if preflight and preflight.get("status") == "STOP_SSR":
        raw = dict(preflight)
        stored_hash = raw.pop("payload_sha256", None)
        if stored_hash != sha256_obj(raw):
            raise RuntimeError("preflight payload hash mismatch")
        expected_common = [family for family in ("MI", "SC") if all(
            next(x for x in preflight["cells"] if x["dataset"] == dataset and
                 x["family"] == family)["dual_metric_upper_bound_pass"]
            for dataset in ("MHC", "MHC_zh"))]
        if expected_common != preflight["common_families_surviving_upper_bound"] or expected_common:
            raise RuntimeError("preflight common-family logic mismatch")
        if preflight["relation_extraction_can_change_this_bound"] is not False:
            raise RuntimeError("preflight subset logic not frozen")
        for dataset, source in preflight["source_hashes"].items():
            arc_path = resolve(cfg, "artifacts") / "pairs" / dataset / "arcs.jsonl"
            if sha256_file(arc_path) != source["arcs"]:
                raise RuntimeError("preflight arc source changed")
            for fold, digest in source["predictions"].items():
                pred_path = resolve(cfg, "artifacts") / "oof" / dataset / \
                    "fold{}".format(fold) / "predictions.json"
                if sha256_file(pred_path) != digest:
                    raise RuntimeError("preflight prediction source changed")
        payload = {
            "schema_version": 1, "run_id": args.run_id,
            "B1_DECISION": "STOP",
            "reason": "all_candidates_dual_metric_oracle_upper_bound_failed_before_relation_extraction",
            "config_sha256": cfg["computed_config_sha256"],
            "common_families": [],
            "preflight_verified": True,
            "preflight_path": str(preflight_path.relative_to(ROOT)),
            "preflight_sha256": sha256_file(preflight_path),
            "preflight_cells": preflight["cells"],
            "logical_implication": "accepted MLLM arcs are a subset of selected candidates; their event-touched oracle cannot exceed this bound",
            "relation_extraction_skipped": True,
            "relation_extraction_skip_reason": "cannot alter failed preregistered oracle gate",
            "B2_B3_locked": True,
            "only_gold": "video_level_binary_label", "segment_gold_used": False,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        payload["decision_payload_sha256"] = sha256_obj(payload)
        atomic_write_json(resolve(cfg, "artifacts") / "B1_DECISION.json", payload)
        print(canonical_json(payload))
        return
    cells = []
    source_hashes = {}
    base = resolve(cfg, "artifacts") / "b1" / "cells"
    for dataset in ("MHC", "MHC_zh"):
        coverage_path = resolve(cfg, "artifacts") / "aggregate" / dataset / "coverage.json"
        coverage = maybe_json(coverage_path)
        if coverage:
            source_hashes[str(coverage_path.relative_to(ROOT))] = sha256_file(coverage_path)
        for family in ("MI", "SC"):
            gates = {}
            for gate in ("audit", "conditional", "oracle"):
                path = base / "{}_{}_{}.json".format(dataset, family, gate)
                gates[gate] = maybe_json(path)
                if path.exists():
                    source_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
            count = coverage["cells"][family] if coverage else None
            cells.append({"dataset": dataset, "family": family,
                          "count": count, **gates})
    p_raw = [float(x["conditional"]["p_raw"])
             if x["conditional"] and x["conditional"].get("p_raw") is not None
             else 1.0 for x in cells]
    p_holm = holm_adjust(p_raw)
    for index, (cell, adjusted) in enumerate(zip(cells, p_holm)):
        cond = cell["conditional"] or {}
        cell.update({
            "p_raw_for_holm": p_raw[index],
            "p_holm": adjusted,
            "count_pass": bool(cell["count"] and
                               cell["count"]["accepted_count_gate"] == "GO"),
            "audit_pass": bool(cell["audit"] and cell["audit"]["status"] == "GO"),
            "conditional_sign_pass": bool(cond.get("delta_nll") is not None and
                                          cond["delta_nll"] > 0 and
                                          cond.get("delta_auc") is not None and
                                          cond["delta_auc"] > 0),
            "conditional_holm_pass": adjusted < float(cfg["b1"]["holm_alpha"]),
            "oracle_pass": bool(cell["oracle"] and cell["oracle"]["status"] == "GO"),
        })
        cell["all_pre_shuffle_gates_pass"] = all([
            cell["count_pass"], cell["audit_pass"],
            cell["conditional_sign_pass"], cell["conditional_holm_pass"],
            cell["oracle_pass"],
        ])
    common = [family for family in ("MI", "SC") if all(
        next(c for c in cells if c["dataset"] == dataset and
             c["family"] == family)["all_pre_shuffle_gates_pass"]
        for dataset in ("MHC", "MHC_zh"))]

    shuffles = {}
    for dataset in ("MHC", "MHC_zh"):
        path = resolve(cfg, "artifacts") / "shuffles" / dataset / "manifest.json"
        shuffles[dataset] = maybe_json(path)
        if path.exists():
            source_hashes[str(path.relative_to(ROOT))] = sha256_file(path)
    shuffle_pass = bool(common) and all(
        shuffles[d] and shuffles[d].get("status") == "OPTIMAL" and
        shuffles[d].get("verified") is True and
        shuffles[d].get("zero_fixed_points") is True and
        shuffles[d].get("query_outdegree_exact") is True and
        shuffles[d].get("neighbor_indegree_exact") is True
        for d in ("MHC", "MHC_zh"))
    decision = "GO" if common and shuffle_pass else "STOP"
    if not common:
        reason = "empty_common_family_after_count_wilson_conditional_holm_oracle"
    elif not shuffle_pass:
        reason = "exact_canonical_shuffle_missing_infeasible_or_unverified"
    else:
        reason = "all_B1_gates_pass"
    payload = {
        "schema_version": 1, "run_id": args.run_id,
        "B1_DECISION": decision, "reason": reason,
        "config_sha256": cfg["computed_config_sha256"],
        "common_families": common, "cells": cells, "shuffles": shuffles,
        "holm_family_size": 4, "holm_alpha": float(cfg["b1"]["holm_alpha"]),
        "B2_B3_locked": decision != "GO",
        "only_gold": "video_level_binary_label", "segment_gold_used": False,
        "source_hashes": source_hashes,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    payload["decision_payload_sha256"] = sha256_obj(payload)
    out = resolve(cfg, "artifacts") / "B1_DECISION.json"
    atomic_write_json(out, payload)
    print(canonical_json(payload))


if __name__ == "__main__":
    main()
