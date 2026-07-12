#!/usr/bin/env python
"""Exact canonical-record derangement for the frozen SSR OOF null."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from ssr_common import (  # noqa: E402
    atomic_write_json, atomic_write_jsonl, canonical_json, load_config,
    rank_bin, read_jsonl, resolve, sha256_file,
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


def gate_summary(cfg):
    cells = []
    for dataset in ("MHC", "MHC_zh"):
        coverage = json.load(open(resolve(cfg, "artifacts") / "aggregate" / dataset /
                                  "coverage.json", encoding="utf-8"))
        for family in ("MI", "SC"):
            base = resolve(cfg, "artifacts") / "b1" / "cells"
            cell = {"dataset": dataset, "family": family,
                    "count": coverage["cells"][family]}
            for gate in ("audit", "conditional", "oracle"):
                path = base / "{}_{}_{}.json".format(dataset, family, gate)
                cell[gate] = json.load(open(path, encoding="utf-8")) if path.exists() else None
            cells.append(cell)
    p_raw = [float(x["conditional"]["p_raw"])
             if x["conditional"] and x["conditional"].get("p_raw") is not None else 1.0
             for x in cells]
    adjusted = holm_adjust(p_raw)
    for x, p in zip(cells, adjusted):
        cond = x["conditional"] or {}
        x["p_holm"] = p
        x["pass"] = bool(
            x["count"]["accepted_count_gate"] == "GO" and
            x["audit"] and x["audit"]["status"] == "GO" and
            cond.get("delta_nll") is not None and cond["delta_nll"] > 0 and
            cond.get("delta_auc") is not None and cond["delta_auc"] > 0 and
            p < float(cfg["b1"]["holm_alpha"]) and
            x["oracle"] and x["oracle"]["status"] == "GO")
    common = [family for family in ("MI", "SC")
              if all(next(x for x in cells if x["dataset"] == ds and
                          x["family"] == family)["pass"]
                     for ds in ("MHC", "MHC_zh"))]
    return cells, common


def quantile_bins(values, q):
    edges = np.quantile(values, np.linspace(0, 1, q + 1)[1:-1])
    return np.searchsorted(edges, values, side="right"), [float(x) for x in edges]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--universe", required=True, choices=["oof"])
    ap.add_argument("--dataset", required=True, choices=["MHC", "MHC_zh"])
    ap.add_argument("--run-id", required=True)
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SSR computation must run under SLURM")
    cfg = load_config(args.config)
    cells, common = gate_summary(cfg)
    if not common:
        raise RuntimeError("empty common family: shuffle must not run")
    agg = resolve(cfg, "artifacts") / "aggregate" / args.dataset
    graph = read_jsonl(agg / "graph.jsonl")
    arcs = read_jsonl(agg / "directed_arcs.jsonl")
    graph_by = {x["canonical_pair_id"]: x for x in graph}
    by_pair = defaultdict(list)
    for x in arcs:
        by_pair[x["canonical_pair_id"]].append(x)
    if set(by_pair) != set(graph_by):
        raise RuntimeError("every canonical record must have selected directions")
    cos_bins, cos_edges = quantile_bins(
        np.asarray([float(x["cosine"]) for x in arcs]), 10)
    margin_bins, margin_edges = quantile_bins(
        np.asarray([float(x["normalized_abs_margin"]) for x in arcs]), 4)
    strata = {}
    for x, cb, mb in zip(arcs, cos_bins, margin_bins):
        strata[x["arc_id"]] = (int(x["outer_fold"]), int(cb), rank_bin(x["rank"]),
                               int(mb), int(x["baseline_error"]))
    signatures = {}
    groups = defaultdict(list)
    for pair_id, dirs in by_pair.items():
        g = graph_by[pair_id]
        signature = (
            "equal" if g["video_a_label"] == g["video_b_label"] else "different",
            tuple(sorted(g["direction_mask"])),
            tuple(sorted(strata[x["arc_id"]] for x in dirs)),
        )
        signatures[pair_id] = signature
        groups[repr(signature)].append(pair_id)

    pair_ids = sorted(graph_by)
    allowed = []
    group_sizes = {k: len(v) for k, v in groups.items()}
    for members in groups.values():
        for source in members:
            for target in members:
                if source != target:
                    allowed.append((source, target))
    source_edges, target_edges = defaultdict(list), defaultdict(list)
    for idx, (source, target) in enumerate(allowed):
        source_edges[source].append(idx)
        target_edges[target].append(idx)
    early_reason = None
    if any(not source_edges[x] or not target_edges[x] for x in pair_ids):
        early_reason = "singleton_or_empty_exact_signature_group"

    out = resolve(cfg, "artifacts") / "shuffles" / args.dataset
    out.mkdir(parents=True, exist_ok=True)
    if early_reason:
        result = {"run_id": args.run_id, "status": "INFEASIBLE",
                  "verified": False, "reason": early_reason,
                  "common_families": common, "group_sizes": group_sizes,
                  "zero_fixed_points": None, "slurm_job_id": os.environ.get("SLURM_JOB_ID")}
        atomic_write_json(out / "equality_audit.json", result)
        atomic_write_json(out / "manifest.json", result)
        print(canonical_json(result))
        raise SystemExit(3)

    rows = []
    lower = []
    upper = []

    def add_eq(coeff, value):
        rows.append(coeff)
        lower.append(float(value))
        upper.append(float(value))

    for source in pair_ids:
        add_eq({i: 1.0 for i in source_edges[source]}, 1)
    for target in pair_ids:
        add_eq({i: 1.0 for i in target_edges[target]}, 1)

    def source_active(source):
        g = graph_by[source]
        return int(g["status"] == "accepted" and g["family"] in common)

    original_q = Counter()
    original_n = Counter()
    for pair_id, dirs in by_pair.items():
        if source_active(pair_id):
            for x in dirs:
                original_q[x["query_id"]] += 1
                original_n[x["neighbor_id"]] += 1
    for query, value in sorted(original_q.items()):
        coeff = {}
        for idx, (source, target) in enumerate(allowed):
            if source_active(source):
                coeff[idx] = sum(x["query_id"] == query for x in by_pair[target])
        add_eq({i: v for i, v in coeff.items() if v}, value)
    for neighbor, value in sorted(original_n.items()):
        coeff = {}
        for idx, (source, target) in enumerate(allowed):
            if source_active(source):
                coeff[idx] = sum(x["neighbor_id"] == neighbor for x in by_pair[target])
        add_eq({i: v for i, v in coeff.items() if v}, value)

    # Explicit global equalities (redundant with source assignment, retained as
    # machine-auditable preregistered invariants).
    for family in common:
        coeff = {idx: float(graph_by[source]["status"] == "accepted" and
                            graph_by[source]["family"] == family)
                 for idx, (source, _target) in enumerate(allowed)}
        add_eq({i: v for i, v in coeff.items() if v},
               sum(g["status"] == "accepted" and g["family"] == family for g in graph))
    rho_values = sorted({str(g["rho"]) for g in graph if g["rho"] is not None})
    for rho in rho_values:
        coeff = {idx: float(str(graph_by[source]["rho"]) == rho)
                 for idx, (source, _target) in enumerate(allowed)}
        add_eq({i: v for i, v in coeff.items() if v},
               sum(str(g["rho"]) == rho for g in graph))
    coeff = {idx: float(graph_by[source]["status"] == "missing/no_edge")
             for idx, (source, _target) in enumerate(allowed)}
    add_eq({i: v for i, v in coeff.items() if v},
           sum(g["status"] == "missing/no_edge" for g in graph))

    A = lil_matrix((len(rows), len(allowed)), dtype=float)
    for r, coeff in enumerate(rows):
        for c, value in coeff.items():
            A[r, c] = value
    result = milp(
        c=np.zeros(len(allowed), dtype=float), integrality=np.ones(len(allowed)),
        bounds=Bounds(np.zeros(len(allowed)), np.ones(len(allowed))),
        constraints=LinearConstraint(A.tocsr(), np.asarray(lower), np.asarray(upper)),
        options={"presolve": True},
    )
    assignment = []
    if result.x is not None:
        assignment = [{"source_pair_id": allowed[i][0],
                       "target_pair_id": allowed[i][1]}
                      for i, value in enumerate(result.x) if value > 0.5]
    mapping = {x["target_pair_id"]: x["source_pair_id"] for x in assignment}
    verified = bool(result.status == 0 and len(assignment) == len(pair_ids) and
                    len(mapping) == len(pair_ids) and
                    all(s != t for t, s in mapping.items()) and
                    all(signatures[s] == signatures[t] for t, s in mapping.items()))
    projected_q, projected_n = Counter(), Counter()
    if verified:
        for target, source in mapping.items():
            if source_active(source):
                for x in by_pair[target]:
                    projected_q[x["query_id"]] += 1
                    projected_n[x["neighbor_id"]] += 1
        verified = projected_q == original_q and projected_n == original_n
    status = "OPTIMAL" if result.status == 0 else "INFEASIBLE" if result.status == 2 else "UNVERIFIED"
    audit = {
        "run_id": args.run_id, "status": status, "verified": verified,
        "solver_status_code": int(result.status), "solver_message": str(result.message),
        "n_records": len(pair_ids), "n_binary_variables": len(allowed),
        "n_equalities": len(rows), "zero_fixed_points": bool(
            assignment and all(x["source_pair_id"] != x["target_pair_id"] for x in assignment)),
        "source_bijection": len({x["source_pair_id"] for x in assignment}) == len(pair_ids),
        "target_bijection": len({x["target_pair_id"] for x in assignment}) == len(pair_ids),
        "signature_match": bool(assignment and all(
            signatures[x["source_pair_id"]] == signatures[x["target_pair_id"]]
            for x in assignment)),
        "query_outdegree_exact": projected_q == original_q if assignment else False,
        "neighbor_indegree_exact": projected_n == original_n if assignment else False,
        "common_families": common, "family_counts": dict(Counter(
            g["family"] for g in graph if g["status"] == "accepted" and g["family"] in common)),
        "reliability_histogram": dict(Counter(str(g["rho"]) for g in graph if g["rho"] is not None)),
        "missing_count": sum(g["status"] == "missing/no_edge" for g in graph),
        "canonical_direction_coupling": True,
        "cosine_decile_edges": cos_edges, "margin_quartile_edges": margin_edges,
        "no_constraint_relaxation": True, "segment_gold_used": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_write_jsonl(out / "assignment.jsonl", assignment)
    atomic_write_json(out / "solver_model.json", {
        "variables": len(allowed), "equalities": len(rows),
        "signature_group_sizes": group_sizes, "fixed_assignment_forbidden": True,
    })
    atomic_write_json(out / "equality_audit.json", audit)
    manifest = dict(audit)
    manifest["outputs"] = {name: sha256_file(out / name) for name in
                           ("assignment.jsonl", "solver_model.json", "equality_audit.json")}
    atomic_write_json(out / "manifest.json", manifest)
    print(canonical_json(manifest))
    if not verified:
        raise SystemExit(3)


if __name__ == "__main__":
    main()
