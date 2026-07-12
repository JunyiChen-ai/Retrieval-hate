#!/usr/bin/env python
"""Aggregate four-call records into authoritative B1 graph and audit packs."""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from ssr_common import (  # noqa: E402
    atomic_write_json, atomic_write_jsonl, canonical_json, id_hash, load_config,
    read_jsonl, relation_family, resolve, sha256_file,
)


def immutable_blank_csv(path, pair_ids):
    path = Path(path)
    if path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "x", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["canonical_pair_id", "validity"])
        writer.writeheader()
        for pair_id in pair_ids:
            writer.writerow({"canonical_pair_id": pair_id, "validity": ""})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, choices=["MHC", "MHC_zh"])
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SSR computation must run under SLURM")
    cfg = load_config(args.config)
    pair_root = resolve(cfg, "artifacts") / "pairs" / args.dataset
    rel_root = resolve(cfg, "artifacts") / "relations" / args.dataset
    pair_manifest = json.load(open(pair_root / "manifest.json", encoding="utf-8"))
    rel_manifest = json.load(open(rel_root / "manifest.json", encoding="utf-8"))
    if pair_manifest["status"] != "COMPLETED" or rel_manifest["status"] != "GO":
        raise RuntimeError("pair/relation upstream is not complete")
    if pair_manifest["config_sha256"] != cfg["computed_config_sha256"] or \
            rel_manifest["config_sha256"] != cfg["computed_config_sha256"]:
        raise RuntimeError("upstream config mismatch")
    for name, expected in pair_manifest["outputs"].items():
        if sha256_file(pair_root / name) != expected:
            raise RuntimeError("pair artifact changed: {}".format(name))

    pairs = read_jsonl(pair_root / "pairs.jsonl")
    arcs = read_jsonl(pair_root / "arcs.jsonl")
    records = read_jsonl(rel_root / "records.jsonl")
    calls = read_jsonl(rel_root / "calls.jsonl")
    pair_by_id = {p["canonical_pair_id"]: p for p in pairs}
    record_by_id = {r["canonical_pair_id"]: r for r in records}
    if set(pair_by_id) != set(record_by_id):
        raise RuntimeError("pair/record key mismatch")
    calls_by_pair = defaultdict(list)
    for c in calls:
        calls_by_pair[c["canonical_pair_id"]].append(c)
    if any(len(calls_by_pair[p]) != 4 for p in pair_by_id):
        raise RuntimeError("not exactly four calls per pair")

    min_rho = float(cfg["mllm"]["min_field_agreement"])
    graph = []
    for pair_id in sorted(pair_by_id):
        pair, rec = pair_by_id[pair_id], record_by_id[pair_id]
        family, predicate, rho = relation_family(
            rec, pair["video_a_label"], pair["video_b_label"], min_rho=min_rho)
        if family is None:
            if rec.get("status") != "relation":
                reason = "invalid_missing_or_failed_call"
            elif any(x["value"] == "unclear" or float(x["rho"]) < min_rho
                     for x in rec["fields"].values()):
                reason = "unclear_or_low_agreement"
            else:
                reason = "valid_relation_not_in_frozen_family"
            status = "missing/no_edge"
        else:
            reason, status = None, "accepted"
        graph.append({
            "canonical_pair_id": pair_id, "dataset": args.dataset,
            "video_a_id": pair["video_a_id"], "video_b_id": pair["video_b_id"],
            "video_a_label": int(pair["video_a_label"]),
            "video_b_label": int(pair["video_b_label"]),
            "direction_mask": pair["direction_mask"],
            "status": status, "family": family, "predicate": predicate,
            "rho": rho, "fallback_reason": reason,
            "label_application_stage": "after_label_blind_relation_frozen",
            "only_gold": "video_level_binary_label", "segment_gold_used": False,
        })
    graph_by_id = {x["canonical_pair_id"]: x for x in graph}
    directed = []
    for arc in arcs:
        g = graph_by_id[arc["canonical_pair_id"]]
        accepted = int(g["status"] == "accepted" and
                       g["family"] == arc["candidate_family"])
        row = dict(arc)
        row.update({"accepted": accepted,
                    "accepted_family": g["family"] if accepted else None,
                    "predicate": g["predicate"] if accepted else None,
                    "rho": g["rho"] if accepted else None,
                    "canonical_status": g["status"]})
        directed.append(row)

    cells = {}
    min_accepted = int(cfg["b1"]["min_accepted"])
    audit_n = int(cfg["b1"]["audit_n"])
    audit_root = resolve(cfg, "artifacts") / "audit" / args.dataset
    for family in ("MI", "SC"):
        accepted_records = [x for x in graph if x["status"] == "accepted" and
                            x["family"] == family]
        accepted_arcs = [x for x in directed if x["accepted"] and
                         x["candidate_family"] == family]
        queries = {x["query_id"] for x in accepted_arcs}
        eligible = len(accepted_records) >= min_accepted
        cells[family] = {
            "dataset": args.dataset, "family": family,
            "n_accepted_unique_canonical_records": len(accepted_records),
            "n_accepted_directed_arcs": len(accepted_arcs),
            "n_unique_queries": len(queries),
            "accepted_count_gate": "GO" if eligible else "FAIL",
            "required_min": min_accepted,
            "rho_histogram": dict(Counter(str(x["rho"]) for x in accepted_records)),
            "human_audit_status": "PENDING" if eligible else "NOT_ELIGIBLE",
        }
        if not eligible:
            continue
        selected = sorted(
            accepted_records,
            key=lambda x: id_hash("audit-v1", args.dataset, family,
                                  x["canonical_pair_id"]))[:audit_n]
        if len(selected) != audit_n:
            raise RuntimeError("audit sample cardinality error")
        pack = []
        for item in selected:
            p = pair_by_id[item["canonical_pair_id"]]
            ab = next(c for c in calls_by_pair[item["canonical_pair_id"]]
                      if c["prompt_version"] == "P0" and c["order"] == "AB")
            payload = ab.get("serialized_input") or {}
            pack.append({
                "canonical_pair_id": item["canonical_pair_id"],
                "language": cfg["datasets"][args.dataset]["language"],
                "claimed_relation_predicate": item["predicate"],
                "video_a_id": p["video_a_id"], "video_b_id": p["video_b_id"],
                "video_a_path": str((resolve(cfg, "video") / args.dataset / "All" /
                                     (p["video_a_id"] + ".mp4")).relative_to(ROOT)),
                "video_b_path": str((resolve(cfg, "video") / args.dataset / "All" /
                                     (p["video_b_id"] + ".mp4")).relative_to(ROOT)),
                "automatic_evidence_a": payload.get("evidence_a", ""),
                "automatic_evidence_b": payload.get("evidence_b", ""),
                "instructions": "valid|invalid|unclear; unclear counts invalid; pair-level relation only; no segment annotation",
            })
        fam_dir = audit_root / family
        atomic_write_jsonl(fam_dir / "audit_pack.jsonl", pack)
        pair_ids = [x["canonical_pair_id"] for x in pack]
        for name in ("A1.csv", "A2.csv", "ADJ.csv"):
            immutable_blank_csv(fam_dir / name, pair_ids)

    out = resolve(cfg, "artifacts") / "aggregate" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    atomic_write_jsonl(out / "graph.jsonl", graph)
    atomic_write_jsonl(out / "directed_arcs.jsonl", directed)
    coverage = {
        "run_id": args.run_id, "status": "COMPLETED", "dataset": args.dataset,
        "config_sha256": cfg["computed_config_sha256"], "cells": cells,
        "n_pairs": len(graph),
        "n_missing_no_edge": sum(x["status"] == "missing/no_edge" for x in graph),
        "fallback_rate": sum(x["status"] == "missing/no_edge" for x in graph) /
                         max(len(graph), 1),
        "only_gold": "video_level_binary_label", "segment_gold_used": False,
        "upstream": {"pair_manifest": sha256_file(pair_root / "manifest.json"),
                     "relation_manifest": sha256_file(rel_root / "manifest.json")},
    }
    atomic_write_json(out / "coverage.json", coverage)
    manifest = dict(coverage)
    manifest["outputs"] = {name: sha256_file(out / name) for name in
                           ("graph.jsonl", "directed_arcs.jsonl", "coverage.json")}
    manifest["slurm_job_id"] = os.environ.get("SLURM_JOB_ID")
    atomic_write_json(out / "manifest.json", manifest)
    print(canonical_json(manifest))


if __name__ == "__main__":
    main()
