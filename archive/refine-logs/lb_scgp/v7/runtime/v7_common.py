#!/usr/bin/env python3
"""Common deterministic machinery for the LB-SCGP G0 v7 repair.

This module intentionally contains no scipy import and no solver entry point.
It is shared by the producer and the independent replay for canonical ranking,
hash binding, frozen residual recomputation, and certificate arithmetic.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/jehc223/RGCL")
V7 = ROOT / "refine-logs" / "lb_scgp" / "v7"
RUNTIME = V7 / "runtime"
OUT_DIR = V7 / "results"
DESIGN = V7 / "G0_V7_PREREGISTERED_DESIGN.json"
V5_CONFIG = ROOT / "configs" / "lb_scgp" / "lb_scgp_v5.json"
V5_DYKSTRA = ROOT / "artifacts" / "lb_scgp" / "v5" / "g0" / "synthetic" / "dykstra.jsonl"
V5_FREEZE = ROOT / "artifacts" / "lb_scgp" / "v5" / "CONFIG_FREEZE.json"
V6_ORACLE_12883 = ROOT / "refine-logs" / "lb_scgp" / "v6" / "results" / "actual_fixture_oracle_12883.json"
V6_REPLAY_12883 = ROOT / "refine-logs" / "lb_scgp" / "v6" / "results" / "actual_fixture_replay_12883.json"
WITNESS_12866 = ROOT / "refine-logs" / "lb_scgp" / "v6" / "results" / "analytic_feasibility_witness_12866.json"
REPLAY_12866 = ROOT / "refine-logs" / "lb_scgp" / "v6" / "results" / "analytic_witness_replay_12866.json"

TOPK = 20
ETA = 1e-12
TAU = 1e-7
SIGNED_RHS = TAU + ETA
PHASE1_RESIDUAL_TOL = 1e-8
PHASE2_VI_TOL = 1e-8
COMPLEMENTARITY_TOL = 1e-6
FULL_RANK_TOL = 1e-8
ACTIVE_TOL = 1e-7


def cjson(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hobj(obj: Any) -> str:
    return hashlib.sha256(cjson(obj).encode("utf-8")).hexdigest()


def payload_hash(obj: dict[str, Any]) -> str:
    return hobj({k: v for k, v in obj.items() if k != "payload_sha256"})


def hfile(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[Any]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json_exclusive(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write((cjson(obj) + "\n").encode("utf-8"))


def append_jsonl(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("ab") as handle:
        handle.write((cjson(obj) + "\n").encode("utf-8"))


def load_design() -> dict[str, Any]:
    design = read_json(DESIGN)
    if float(design.get("eta")) != ETA:
        raise RuntimeError("v7 design eta changed or is not the preregistered value")
    contract = design.get("immutable_contract", {})
    required = {
        "topk": TOPK,
        "tau": TAU,
        "violation": 1e-6,
        "relative": 1e-7,
        "max_independent_orientations": 8,
        "max_pivots": 32,
    }
    for key, expected in required.items():
        if contract.get(key) != expected:
            raise RuntimeError("v7 design immutable contract mismatch for {}".format(key))
    return design


def load_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    cfg = read_json(V5_CONFIG)
    oriented = None
    for row in read_jsonl(V5_DYKSTRA):
        if row.get("case") == "feasible_oriented_boundary":
            oriented = row
            break
    if oriented is None:
        raise RuntimeError("missing feasible_oriented_boundary row")
    witness = read_json(WITNESS_12866)
    replay = read_json(REPLAY_12866)
    return cfg, oriented, oriented["fixture"], witness, replay


def supervision_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    supervision = cfg.get("supervision", {})
    counters = cfg.get("counters", {})
    keys = [
        "mllm_call_count",
        "ocr_call_count",
        "teacher_cache_read_count",
        "teacher_cache_write_count",
        "outer_held_label_read_count",
        "outer_held_content_read_count",
        "val_content_read_count",
        "test_content_read_count",
        "val_test_teacher_artifact_count",
    ]
    out = {
        "only_gold_supervision": supervision.get("only_gold_supervision"),
        "segment_gold_exists": bool(supervision.get("segment_gold_exists")),
        "segment_gold_used": bool(supervision.get("segment_gold_used")),
    }
    for key in keys:
        out[key] = int(counters.get(key, -1))
    out["ok"] = (
        out["only_gold_supervision"] == "parent_video_binary_label"
        and out["segment_gold_exists"] is False
        and out["segment_gold_used"] is False
        and all(int(out[key]) == 0 for key in keys)
    )
    return out


def current_source_hashes() -> dict[str, Any]:
    return {
        "design": hfile(DESIGN),
        "v5_config": hfile(V5_CONFIG),
        "v5_dykstra_jsonl": hfile(V5_DYKSTRA),
        "v5_freeze": hfile(V5_FREEZE),
        "accepted_witness_12866": hfile(WITNESS_12866),
        "accepted_replay_12866": hfile(REPLAY_12866),
        "v6_actual_fixture_oracle_12883": hfile(V6_ORACLE_12883),
        "v6_actual_fixture_replay_12883": hfile(V6_REPLAY_12883),
        "v7_common.py": hfile(RUNTIME / "v7_common.py"),
        "v7_actual_certificate.py": hfile(RUNTIME / "v7_actual_certificate.py"),
        "v7_independent_replay.py": hfile(RUNTIME / "v7_independent_replay.py"),
        "validate_v7_static.py": hfile(RUNTIME / "validate_v7_static.py"),
        "validate_v7_static.sbatch": hfile(RUNTIME / "validate_v7_static.sbatch"),
        "v7_actual_certificate.sbatch": hfile(RUNTIME / "v7_actual_certificate.sbatch"),
        "v7_independent_replay.sbatch": hfile(RUNTIME / "v7_independent_replay.sbatch"),
    }


def existing_hashes_unchanged() -> dict[str, Any]:
    out = {"ok": False, "checks": []}
    if not V6_ORACLE_12883.exists():
        out["reason"] = "missing_v6_actual_oracle_12883"
        return out
    v6 = read_json(V6_ORACLE_12883)
    refs = v6.get("source_hashes", {})
    mapping = {
        "v5_config": (V5_CONFIG, refs.get("v5_config")),
        "v5_dykstra_jsonl": (V5_DYKSTRA, refs.get("v5_dykstra_jsonl")),
        "v5_freeze": (V5_FREEZE, refs.get("v5_freeze")),
        "accepted_witness_12866": (WITNESS_12866, refs.get("accepted_witness_12866")),
        "accepted_replay_12866": (REPLAY_12866, refs.get("accepted_replay_12866")),
    }
    ok = True
    checks = []
    for name, (path, expected) in mapping.items():
        actual = hfile(path)
        row = {
            "name": name,
            "path": str(path.relative_to(ROOT)),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "ok": actual == expected and actual is not None,
        }
        ok = ok and row["ok"]
        checks.append(row)
    out["checks"] = checks
    out["ok"] = ok
    return out


def tolerance_order(values: list[float], ids: list[str], tolerance: float) -> list[int]:
    remaining = sorted(range(len(values)), key=lambda k: -float(values[k]))
    ordered: list[int] = []
    while remaining:
        anchor = float(values[remaining[0]])
        group = [k for k in remaining if anchor - float(values[k]) <= tolerance]
        group.sort(key=lambda k: str(ids[k]))
        ordered.extend(group)
        selected = set(group)
        remaining = [k for k in remaining if k not in selected]
    return ordered


def stable_rankings(gram: np.ndarray, ids: list[str], topk: int, tolerance: float) -> list[list[int]]:
    out: list[list[int]] = []
    for i in range(len(ids)):
        candidates = [j for j in range(len(ids)) if j != i]
        local = tolerance_order([float(gram[i, j]) for j in candidates], [ids[j] for j in candidates], tolerance)
        out.append([candidates[k] for k in local[:topk]])
    return out


def canonical_top20(gram: np.ndarray, ids: list[str], tolerance: float) -> list[list[int]]:
    return stable_rankings(gram, ids, TOPK, tolerance)


def canonical_rhs(ids: list[str], a: int, b: int, tolerance: float) -> float:
    return -float(tolerance) if str(ids[a]) < str(ids[b]) else float(np.nextafter(float(tolerance), math.inf))


def rank_halfspaces(ids: list[str], full_rankings: list[list[int]], tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    index = 0
    for i, full in enumerate(full_rankings):
        top = full[:TOPK]
        for r in range(TOPK - 1):
            a, b = top[r], top[r + 1]
            rows.append({
                "name": "rank_internal_{:04d}".format(index),
                "index": index,
                "query": i,
                "a": a,
                "b": b,
                "kind": "internal",
                "rhs": canonical_rhs(ids, a, b, tolerance),
            })
            index += 1
        for outsider in full[TOPK:]:
            a, b = top[TOPK - 1], outsider
            rows.append({
                "name": "rank_boundary_{:04d}".format(index),
                "index": index,
                "query": i,
                "a": a,
                "b": b,
                "kind": "boundary",
                "rhs": canonical_rhs(ids, a, b, tolerance),
            })
            index += 1
    return rows


def signed_gap_edges(ids: list[str], full_rankings: list[list[int]], tau: float, eta: float) -> list[dict[str, Any]]:
    edges: list[dict[str, Any]] = []
    index = 0
    rhs = float(tau + eta)
    n = len(ids)
    for q, full in enumerate(full_rankings):
        top = full[:TOPK]
        if len(set(top)) != TOPK:
            raise RuntimeError("duplicate top20 entry in query {}".format(q))
        if q in top:
            raise RuntimeError("self entry in top20 for query {}".format(q))
        outsiders = [j for j in full if j not in set(top)]
        expected_outsiders = [j for j in range(n) if j != q and j not in set(top)]
        if set(outsiders) != set(expected_outsiders) or len(outsiders) != n - 1 - TOPK:
            raise RuntimeError("incomplete outsider enumeration for query {}".format(q))
        for r in range(TOPK - 1):
            a, b = int(top[r]), int(top[r + 1])
            edges.append({
                "edge_index": index,
                "name": "signed_internal_{:04d}".format(index),
                "kind": "internal",
                "query": q,
                "query_id": ids[q],
                "a": a,
                "b": b,
                "a_id": ids[a],
                "b_id": ids[b],
                "rank_position": r,
                "rhs": rhs,
                "sense": "G[query,a]-G[query,b]>=tau+eta",
            })
            index += 1
        for outsider in outsiders:
            a, b = int(top[TOPK - 1]), int(outsider)
            edges.append({
                "edge_index": index,
                "name": "signed_boundary_{:04d}".format(index),
                "kind": "boundary",
                "query": q,
                "query_id": ids[q],
                "a": a,
                "b": b,
                "a_id": ids[a],
                "b_id": ids[b],
                "rank_position": TOPK - 1,
                "rhs": rhs,
                "sense": "G[query,top20th]-G[query,outsider]>=tau+eta",
            })
            index += 1
    return edges


def pair_normal(n: int, q: int, a: int, b: int, pairs: list[tuple[int, int]]) -> list[float]:
    pair_index = {pair: k for k, pair in enumerate(pairs)}
    row = np.zeros(len(pairs), dtype=np.float64)
    for j, sign in ((a, 1.0), (b, -1.0)):
        key = tuple(sorted((q, j)))
        if key[0] == key[1]:
            continue
        row[pair_index[key]] += sign
    return row.tolist()


def orientation_descriptors(gram: np.ndarray, ids: list[str], tolerance: float) -> list[dict[str, Any]]:
    n = len(ids)
    full = stable_rankings(gram, ids, n - 1, tolerance)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    descriptors: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, str, int]] = set()
    for q, ranking in enumerate(full):
        top = ranking[:TOPK]
        for r in range(TOPK - 1):
            a, b = int(top[r]), int(top[r + 1])
            if abs(float(gram[q, a]) - float(gram[q, b])) <= tolerance:
                key = (q, a, b, "internal", r)
                if key not in seen:
                    seen.add(key)
                    descriptors.append({
                        "descriptor_index": len(descriptors),
                        "kind": "internal",
                        "position": r,
                        "query": q,
                        "query_id": ids[q],
                        "a": a,
                        "b": b,
                        "a_id": ids[a],
                        "b_id": ids[b],
                        "score_gap_at_g0": float(gram[q, a] - gram[q, b]),
                        "normal": pair_normal(n, q, a, b, pairs),
                    })
        boundary = int(top[TOPK - 1])
        boundary_value = float(gram[q, boundary])
        for outsider in ranking[TOPK:]:
            outsider = int(outsider)
            if abs(boundary_value - float(gram[q, outsider])) <= tolerance:
                key = (q, boundary, outsider, "boundary", TOPK - 1)
                if key not in seen:
                    seen.add(key)
                    descriptors.append({
                        "descriptor_index": len(descriptors),
                        "kind": "boundary",
                        "position": TOPK - 1,
                        "query": q,
                        "query_id": ids[q],
                        "a": boundary,
                        "b": outsider,
                        "a_id": ids[boundary],
                        "b_id": ids[outsider],
                        "score_gap_at_g0": float(gram[q, boundary] - gram[q, outsider]),
                        "normal": pair_normal(n, q, boundary, outsider, pairs),
                    })
    return descriptors


def orientation_cell_from_assignment(
    base_rankings: list[list[int]],
    descriptors: list[dict[str, Any]],
    assignment: list[int],
    ids: list[str],
) -> list[list[int]]:
    by_query: dict[int, list[tuple[int, int, int]]] = {}
    for sign, desc in zip(assignment, descriptors):
        by_query.setdefault(int(desc["query"]), []).append((int(sign), int(desc["a"]), int(desc["b"])))
    cells = [list(row) for row in base_rankings]
    for qi, constraints in by_query.items():
        baseline = list(base_rankings[qi])
        parent = {node: node for node in baseline}

        def find(node: int) -> int:
            while parent[node] != node:
                parent[node] = parent[parent[node]]
                node = parent[node]
            return node

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for _, a, b in constraints:
            union(a, b)
        edges = {node: set() for node in baseline}
        indegree = {node: 0 for node in baseline}
        pos = {node: i for i, node in enumerate(baseline)}
        for a, b in zip(baseline[:-1], baseline[1:]):
            if find(a) != find(b):
                edges[a].add(b)
        for sign, a, b in constraints:
            u, v = (a, b) if sign > 0 else (b, a)
            edges[u].add(v)
        for u in edges:
            for v in edges[u]:
                indegree[v] += 1
        available = [node for node in baseline if indegree[node] == 0]
        order: list[int] = []
        while available:
            available.sort(key=lambda node: (pos[node], str(ids[node])))
            u = available.pop(0)
            order.append(u)
            for v in sorted(edges[u], key=lambda node: (pos[node], str(ids[node]))):
                indegree[v] -= 1
                if indegree[v] == 0:
                    available.append(v)
        if len(order) != len(baseline):
            raise RuntimeError("compatible orientation produced cyclic rank DAG")
        cells[qi] = order
    return cells


def build_cells(fixture: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    solver = cfg["solver"]
    n = int(fixture["n"])
    ids = [str(x) for x in fixture["ids"]]
    if len(set(ids)) != len(ids):
        raise RuntimeError("duplicate IDs rejected")
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    tau = float(solver["tie_tolerance"])
    if tau != TAU:
        raise RuntimeError("frozen tau mismatch")
    base_full = stable_rankings(g0, ids, topk=n - 1, tolerance=tau)
    descriptors = orientation_descriptors(g0, ids, tau)
    normals = np.asarray([d["normal"] for d in descriptors], dtype=np.float64) if descriptors else np.zeros((0, n * (n - 1) // 2))
    rank = int(np.linalg.matrix_rank(normals, tol=1e-12)) if descriptors else 0
    max_orientations = int(solver["max_independent_orientations"])
    max_pivots = int(solver["max_pivots"])
    reject_reasons = []
    if rank > max_orientations:
        reject_reasons.append("orientation_rank_over_budget")
    if len(descriptors) > max_pivots:
        reject_reasons.append("pivot_count_over_budget")
    if len(descriptors) > max_orientations:
        reject_reasons.append("descriptor_count_over_enumeration_budget")
    assignments: list[list[int]] = []
    cells: list[dict[str, Any]] = []
    if not reject_reasons:
        for assignment_tuple in itertools.product([-1, 1], repeat=len(descriptors)):
            assignment = [int(x) for x in assignment_tuple]
            try:
                full = orientation_cell_from_assignment(base_full, descriptors, assignment, ids)
                edges = signed_gap_edges(ids, full, tau, ETA)
            except Exception as exc:
                reject_reasons.append("assignment_{}_rejected_{}:{}".format(len(assignments), type(exc).__name__, str(exc)))
                continue
            final_top20 = [row[:TOPK] for row in full]
            if len(edges) != n * ((TOPK - 1) + (n - 1 - TOPK)):
                reject_reasons.append("incomplete_signed_gap_edges")
                continue
            cell_obj = {
                "cell_index": len(cells),
                "assignment": assignment,
                "orientation_descriptor_count": len(descriptors),
                "final_top20_rankings": final_top20,
                "full_outsider_order_for_enumeration": full,
                "signed_gap_edges": edges,
                "final_top20_rankings_sha256": hobj(final_top20),
                "full_outsider_order_for_enumeration_sha256": hobj(full),
                "signed_gap_edges_sha256": hobj(edges),
                "top20_sha256": hobj(final_top20),
            }
            cell_obj["cell_sha256"] = hobj({
                "cell_index": cell_obj["cell_index"],
                "assignment": assignment,
                "final_top20_rankings": final_top20,
                "full_outsider_order_for_enumeration": full,
                "signed_gap_edges": edges,
            })
            cells.append(cell_obj)
            assignments.append(assignment)
    system = {
        "descriptor_source": "canonical_top20_g0_tau_membership_or_position_near_ties",
        "rank": rank,
        "descriptors": descriptors,
        "basis_indices": list(range(rank)),
        "compatible_assignments": assignments,
        "compatible_overflow": False,
        "reject_reasons": reject_reasons,
        "max_independent_orientations": max_orientations,
        "max_pivots": max_pivots,
        "complete_adjacent_enumeration": bool(not reject_reasons and all(len(c["signed_gap_edges"]) == 528 for c in cells)),
    }
    return system, cells


def semantic_matrix(raw: Any, n: int) -> np.ndarray:
    sem = np.asarray(raw, dtype=np.float64)
    if sem.size == 0:
        return np.zeros((0, n * n), dtype=np.float64)
    if sem.ndim == 1:
        return sem.reshape(1, n * n)
    return sem.reshape((-1, n * n))


def centroid_direction(labels: np.ndarray) -> np.ndarray:
    n = len(labels)
    out = np.zeros((n, n), dtype=np.float64)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    for rows in groups:
        out[np.ix_(rows, rows)] += 1.0 / (len(rows) ** 2)
    out[np.ix_(groups[0], groups[1])] -= 1.0 / (len(groups[0]) * len(groups[1]))
    out[np.ix_(groups[1], groups[0])] -= 1.0 / (len(groups[0]) * len(groups[1]))
    return out


def rank_coefficients(labels: np.ndarray, rankings: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(labels)
    signs = 2 * labels.astype(np.int64) - 1
    top = np.asarray([row[:TOPK] for row in rankings], dtype=np.int64)
    coeff = np.zeros((n, TOPK), dtype=np.float64)
    for i in range(n):
        for r, j in enumerate(top[i], 1):
            coeff[i, r - 1] = signs[i] * (TOPK + 1 - r) * signs[j] / 210.0
    return top, coeff


def margin_data(g0: np.ndarray, labels: np.ndarray, ids: list[str], full_rankings: list[list[int]], tol: float) -> dict[str, Any]:
    baseline_full = stable_rankings(g0, ids, topk=len(ids) - 1, tolerance=tol)
    cell_top, cell_coeff = rank_coefficients(labels, full_rankings)
    base_top, base_coeff = rank_coefficients(labels, baseline_full)
    baseline_margins = np.asarray(
        [float(base_coeff[i] @ g0[i, base_top[i]]) for i in range(len(labels))],
        dtype=np.float64,
    )
    return {
        "top": cell_top,
        "coeff": cell_coeff,
        "baseline_rankings": baseline_full,
        "baseline_margins": baseline_margins,
    }


@dataclass
class LinearRow:
    name: str
    coeff: np.ndarray
    const: float
    group: str
    block: str


@dataclass
class DirectProblem:
    cell_index: int
    assignment: list[int]
    n: int
    ids: list[str]
    labels: np.ndarray
    groups: list[np.ndarray]
    g0: np.ndarray
    ell: np.ndarray
    semantic: np.ndarray
    solver: dict[str, Any]
    full_rankings: list[list[int]]
    final_top20: list[list[int]]
    signed_edges: list[dict[str, Any]]
    md: dict[str, Any]
    original_rank_rows: list[dict[str, Any]]
    centroid: np.ndarray
    caps: list[float]
    row_radius: float
    class_radius: float
    pairs: list[tuple[int, int]]
    pair_index: dict[tuple[int, int], int]
    original_linear_rows: list[LinearRow]
    signed_gap_linear_rows: list[LinearRow]
    eq_a: np.ndarray
    eq_b: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.pairs) + self.n

    @property
    def all_linear_rows(self) -> list[LinearRow]:
        return self.original_linear_rows + self.signed_gap_linear_rows


def constraint_map() -> list[dict[str, Any]]:
    return [
        {"range": [0, 0], "count": 1, "formula": "symmetry"},
        {"range": [1, 1], "count": 1, "formula": "correlation_diagonal"},
        {"range": [2, 2], "count": 1, "formula": "psd_symmetrized_input"},
        {"range": [3, 3], "count": 1, "formula": "offdiagonal_box"},
        {"range": [4, 27], "count": 24, "formula": "row_trust_{00..23}"},
        {"range": [28, 29], "count": 2, "formula": "class_mean_trust_{0,1}"},
        {"range": [30, 30], "count": 1, "formula": "semantic_radius_zero"},
        {"range": [31, 32], "count": 2, "formula": "slack_capped_simplex_{0,1}"},
        {"range": [33, 56], "count": 24, "formula": "vote_slack_{00..23}"},
        {"range": [57, 58], "count": 2, "formula": "class_mean_margin_{0,1}"},
        {"range": [59, 59], "count": 1, "formula": "global_mean_margin"},
        {"range": [60, 60], "count": 1, "formula": "centroid_distance"},
        {
            "range": [61, 588],
            "count": 528,
            "formula": "frozen original rank halfspaces for the selected compatible full outsider order",
        },
    ]


def expected_set_order(n: int) -> list[str]:
    names = ["symmetry", "correlation_diagonal", "psd_symmetrized_input", "offdiagonal_box"]
    names.extend("row_trust_{:02d}".format(i) for i in range(n))
    names.extend(["class_mean_trust_0", "class_mean_trust_1", "semantic_radius_zero"])
    names.extend(["slack_capped_simplex_0", "slack_capped_simplex_1"])
    names.extend("vote_slack_{:02d}".format(i) for i in range(n))
    names.extend(["class_mean_margin_0", "class_mean_margin_1", "global_mean_margin", "centroid_distance"])
    index = 0
    for _ in range(n):
        for _ in range(19):
            names.append("rank_internal_{:04d}".format(index))
            index += 1
        for _ in range(n - 1 - TOPK):
            names.append("rank_boundary_{:04d}".format(index))
            index += 1
    return names


def reduced_affine(problem_pairs: list[tuple[int, int]], n: int, a: np.ndarray, xi: np.ndarray | None, const: float) -> tuple[np.ndarray, float]:
    coeff = np.zeros(len(problem_pairs) + n, dtype=np.float64)
    for k, (i, j) in enumerate(problem_pairs):
        coeff[k] = float(a[i, j] + a[j, i])
    if xi is not None:
        coeff[len(problem_pairs):len(problem_pairs) + n] = np.asarray(xi, dtype=np.float64)
    diag_const = float(sum(float(a[i, i]) for i in range(n)))
    return coeff, diag_const + float(const)


def build_problem(fixture: dict[str, Any], cfg: dict[str, Any], cell: dict[str, Any]) -> DirectProblem:
    solver = cfg["solver"]
    n = int(fixture["n"])
    ids = [str(x) for x in fixture["ids"]]
    labels = np.asarray(fixture["labels"], dtype=np.int64)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    ell = np.asarray(fixture["ell"], dtype=np.float64)
    semantic = semantic_matrix(fixture["semantic"], n)
    full_rankings = [[int(v) for v in row] for row in cell["full_outsider_order_for_enumeration"]]
    final_top20 = [[int(v) for v in row] for row in cell["final_top20_rankings"]]
    signed_edges = list(cell["signed_gap_edges"])
    md = margin_data(g0, labels, ids, full_rankings, float(solver["tie_tolerance"]))
    deficits = np.maximum(ell - md["baseline_margins"], 0.0)
    caps = [float(solver["slack_budget_ratio"]) * float(deficits[rows].sum()) for rows in groups]
    centroid = centroid_direction(labels)
    original_rank_rows = rank_halfspaces(ids, full_rankings, float(solver["tie_tolerance"]))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = {pair: k for k, pair in enumerate(pairs)}
    original_linear_rows: list[LinearRow] = []
    signed_gap_linear_rows: list[LinearRow] = []

    def add_row(target: list[LinearRow], name: str, group: str, block: str, a: np.ndarray | None, x: np.ndarray | None, const: float) -> None:
        aa = np.zeros((n, n), dtype=np.float64) if a is None else np.asarray(a, dtype=np.float64)
        coeff, cc = reduced_affine(pairs, n, aa, x, const)
        target.append(LinearRow(name=name, coeff=coeff, const=cc, group=group, block=block))

    for k, rows in enumerate(groups):
        x = np.zeros(n, dtype=np.float64)
        x[rows] = -1.0
        add_row(original_linear_rows, "slack_capped_simplex_{}".format(k), "slack", "original_589", None, x, caps[k])
    for i in range(n):
        a = np.zeros((n, n), dtype=np.float64)
        for r in range(TOPK):
            a[i, int(md["top"][i, r])] += float(md["coeff"][i, r])
        x = np.zeros(n, dtype=np.float64)
        x[i] = 1.0
        add_row(original_linear_rows, "vote_slack_{:02d}".format(i), "vote", "original_589", a, x, -float(ell[i]))
    for k, rows in enumerate(groups):
        a = np.zeros((n, n), dtype=np.float64)
        for i in rows:
            for r in range(TOPK):
                a[int(i), int(md["top"][int(i), r])] += float(md["coeff"][int(i), r]) / len(rows)
        add_row(original_linear_rows, "class_mean_margin_{}".format(k), "class_margin", "original_589", a, None, -float(md["baseline_margins"][rows].mean()))
    a_global = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for r in range(TOPK):
            a_global[i, int(md["top"][i, r])] += float(md["coeff"][i, r]) / n
    add_row(original_linear_rows, "global_mean_margin", "global_margin", "original_589", a_global, None, -float(md["baseline_margins"].mean()))
    add_row(original_linear_rows, "centroid_distance", "centroid", "original_589", centroid, None, -float(np.sum(centroid * g0)))
    for row in original_rank_rows:
        a = np.zeros((n, n), dtype=np.float64)
        a[int(row["query"]), int(row["a"])] += 1.0
        a[int(row["query"]), int(row["b"])] -= 1.0
        add_row(original_linear_rows, str(row["name"]), "rank_halfspaces", "original_589", a, None, -float(row["rhs"]))
    for edge in signed_edges:
        a = np.zeros((n, n), dtype=np.float64)
        a[int(edge["query"]), int(edge["a"])] += 1.0
        a[int(edge["query"]), int(edge["b"])] -= 1.0
        add_row(signed_gap_linear_rows, str(edge["name"]), "signed_gap", "additive_signed_gap", a, None, -float(edge["rhs"]))

    eq_rows = []
    eq_rhs = []
    for row in semantic.reshape((-1, n, n)):
        coeff, cc = reduced_affine(pairs, n, row, None, 0.0)
        eq_rows.append(coeff)
        eq_rhs.append(-cc)
    eq_a = np.stack(eq_rows).astype(np.float64) if eq_rows else np.zeros((0, len(pairs) + n), dtype=np.float64)
    eq_b = np.asarray(eq_rhs, dtype=np.float64)
    return DirectProblem(
        cell_index=int(cell["cell_index"]),
        assignment=[int(v) for v in cell["assignment"]],
        n=n,
        ids=ids,
        labels=labels,
        groups=groups,
        g0=g0,
        ell=ell,
        semantic=semantic,
        solver=solver,
        full_rankings=full_rankings,
        final_top20=final_top20,
        signed_edges=signed_edges,
        md=md,
        original_rank_rows=original_rank_rows,
        centroid=centroid,
        caps=caps,
        row_radius=float(solver["row_trust_scale"]) * math.sqrt(n - 1),
        class_radius=float(solver["class_mean_trust_scale"]) * math.sqrt(n),
        pairs=pairs,
        pair_index=pair_index,
        original_linear_rows=original_linear_rows,
        signed_gap_linear_rows=signed_gap_linear_rows,
        eq_a=eq_a,
        eq_b=eq_b,
    )


def pack_g_xi(problem: DirectProblem, g: np.ndarray, xi: np.ndarray) -> np.ndarray:
    values = [float(g[i, j]) for i, j in problem.pairs]
    values.extend([float(v) for v in xi])
    return np.asarray(values, dtype=np.float64)


def unpack_g_xi(problem: DirectProblem, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.eye(problem.n, dtype=np.float64)
    for value, (i, j) in zip(x[:len(problem.pairs)], problem.pairs):
        g[i, j] = float(value)
        g[j, i] = float(value)
    xi = np.asarray(x[len(problem.pairs):len(problem.pairs) + problem.n], dtype=np.float64)
    return g, xi


def bounds_arrays(problem: DirectProblem) -> tuple[np.ndarray, np.ndarray]:
    lower = np.concatenate([
        np.full(len(problem.pairs), -1.0, dtype=np.float64),
        np.zeros(problem.n, dtype=np.float64),
    ])
    upper = np.concatenate([
        np.full(len(problem.pairs), float(problem.solver["offdiag_upper"]), dtype=np.float64),
        np.full(problem.n, max(1.0, max(problem.caps, default=0.0) + 1.0), dtype=np.float64),
    ])
    return lower, upper


def linear_values(rows: list[LinearRow], x: np.ndarray) -> np.ndarray:
    if not rows:
        return np.zeros(0, dtype=np.float64)
    a = np.stack([row.coeff for row in rows])
    c = np.asarray([row.const for row in rows], dtype=np.float64)
    return a @ x + c


def objective_value_grad(problem: DirectProblem, x: np.ndarray) -> tuple[float, np.ndarray]:
    g, xi = unpack_g_xi(problem, x)
    value = 0.5 * float(np.sum((g - problem.g0) ** 2)) + 0.5 * float(np.dot(xi, xi))
    grad = np.zeros(problem.dim, dtype=np.float64)
    for k, (i, j) in enumerate(problem.pairs):
        grad[k] = 2.0 * float(g[i, j] - problem.g0[i, j])
    grad[len(problem.pairs):] = xi
    return value, grad


def nonlinear_values_jac(problem: DirectProblem, x: np.ndarray, phase1: bool = False) -> tuple[np.ndarray, np.ndarray]:
    base = x[:-1] if phase1 else x
    t = float(x[-1]) if phase1 else 0.0
    g, _ = unpack_g_xi(problem, base)
    nvar = problem.dim + (1 if phase1 else 0)
    values: list[float] = []
    jac_rows: list[np.ndarray] = []
    sym = 0.5 * (g + g.T)
    vals, vecs = np.linalg.eigh(sym)
    idx = int(np.argmin(vals))
    v = vecs[:, idx]
    row = np.zeros(nvar, dtype=np.float64)
    for k, (i, j) in enumerate(problem.pairs):
        row[k] = 2.0 * float(v[i] * v[j])
    if phase1:
        row[-1] = -1.0
        values.append(float(vals[idx]) - t)
    else:
        values.append(float(vals[idx]))
    jac_rows.append(row)
    eps = 1e-15
    for i in range(problem.n):
        diff = g[i].copy() - problem.g0[i]
        diff[i] = 0.0
        norm = float(np.linalg.norm(diff))
        row = np.zeros(nvar, dtype=np.float64)
        if norm > eps:
            for j in range(problem.n):
                if i == j:
                    continue
                k = problem.pair_index[tuple(sorted((i, j)))]
                row[k] = -float(diff[j]) / norm
        values.append(problem.row_radius - norm)
        jac_rows.append(row)
    for rows in problem.groups:
        mean = (g[rows] - problem.g0[rows]).mean(axis=0)
        norm = float(np.linalg.norm(mean))
        row = np.zeros(nvar, dtype=np.float64)
        if norm > eps:
            row_set = set(int(v) for v in rows)
            for a, b in problem.pairs:
                deriv = 0.0
                if a in row_set:
                    deriv += float(mean[b]) / len(rows)
                if b in row_set:
                    deriv += float(mean[a]) / len(rows)
                row[problem.pair_index[(a, b)]] = -deriv / norm
        values.append(problem.class_radius - norm)
        jac_rows.append(row)
    return np.asarray(values, dtype=np.float64), np.stack(jac_rows).astype(np.float64)


def residual_sets_original(problem: DirectProblem, g: np.ndarray, xi: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, float]]:
    off = g[~np.eye(problem.n, dtype=bool)]
    eig_min = float(np.linalg.eigvalsh(0.5 * (g + g.T)).min())
    rows: list[dict[str, Any]] = []

    def add(name: str, group: str, residual: float) -> None:
        rows.append({"name": name, "group": group, "residual": float(max(0.0, residual))})

    add("symmetry", "symmetry", float(np.max(np.abs(g - g.T))))
    add("correlation_diagonal", "unit_diagonal", float(np.max(np.abs(np.diag(g) - 1.0))))
    add("psd_symmetrized_input", "psd", -eig_min)
    add("offdiagonal_box", "box", max(float(off.max() - problem.solver["offdiag_upper"]), float(-1.0 - off.min())))
    for i in range(problem.n):
        add("row_trust_{:02d}".format(i), "row_trust", float(np.linalg.norm(np.delete(g[i] - problem.g0[i], i)) - problem.row_radius))
    for k, label in enumerate((0, 1)):
        rows_k = np.flatnonzero(problem.labels == label)
        add("class_mean_trust_{}".format(k), "class_mean_trust", float(np.linalg.norm((g[rows_k] - problem.g0[rows_k]).mean(axis=0)) - problem.class_radius))
    add("semantic_radius_zero", "semantic", float(np.linalg.norm(problem.semantic @ g.reshape(-1))))
    margins = np.einsum("ir,ir->i", problem.md["coeff"], g[np.arange(problem.n)[:, None], problem.md["top"]], optimize=True)
    for k, label in enumerate((0, 1)):
        rows_k = np.flatnonzero(problem.labels == label)
        add("slack_capped_simplex_{}".format(k), "slack", max(float(-xi[rows_k].min()), float(xi[rows_k].sum() - problem.caps[k])))
    for i in range(problem.n):
        add("vote_slack_{:02d}".format(i), "vote", float(problem.ell[i] - margins[i] - xi[i]))
    for k, label in enumerate((0, 1)):
        rows_k = np.flatnonzero(problem.labels == label)
        add("class_mean_margin_{}".format(k), "class_margin", float(problem.md["baseline_margins"][rows_k].mean() - margins[rows_k].mean()))
    add("global_mean_margin", "global_margin", float(problem.md["baseline_margins"].mean() - margins.mean()))
    add("centroid_distance", "centroid", float(np.sum(problem.centroid * problem.g0) - np.sum(problem.centroid * g)))
    for row in problem.original_rank_rows:
        add(str(row["name"]), "rank_halfspaces", float(row["rhs"]) - float(g[int(row["query"]), int(row["a"])] - g[int(row["query"]), int(row["b"])]))
    if len(rows) != 589:
        raise RuntimeError("frozen original residual set count mismatch: {}".format(len(rows)))
    groups: dict[str, float] = {}
    for row in rows:
        groups[row["group"]] = max(groups.get(row["group"], 0.0), float(row["residual"]))
    return rows, groups


def signed_gap_residuals(problem: DirectProblem, g: np.ndarray) -> dict[str, Any]:
    rows = []
    min_margin = float("inf")
    for edge in problem.signed_edges:
        diff = float(g[int(edge["query"]), int(edge["a"])] - g[int(edge["query"]), int(edge["b"])])
        rhs = float(edge["rhs"])
        margin = diff - rhs
        min_margin = min(min_margin, margin)
        rows.append({
            "name": str(edge["name"]),
            "kind": str(edge["kind"]),
            "query": int(edge["query"]),
            "a": int(edge["a"]),
            "b": int(edge["b"]),
            "rhs": rhs,
            "lhs": diff,
            "margin": margin,
            "residual": float(max(0.0, rhs - diff)),
        })
    max_residual = float(max((row["residual"] for row in rows), default=0.0))
    return {
        "row_count": len(rows),
        "max_residual": max_residual,
        "min_margin": min_margin,
        "pass": bool(max_residual <= 0.0 and min_margin >= 0.0 and len(rows) == 528),
        "rows_sha256": hobj(rows),
        "rows": rows,
    }


def max_residual(rows: list[dict[str, Any]]) -> float:
    return float(max((float(row["residual"]) for row in rows), default=float("inf")))


def phase_starts(problem: DirectProblem, witness: dict[str, Any]) -> list[dict[str, Any]]:
    selected = witness.get("determination", {}).get("selected_witness_sha256")
    candidates = []
    for row in witness.get("raw_results", []):
        if row.get("witness") and (selected is None or row["witness"].get("witness_sha256") == selected):
            candidates.append(row["witness"])
    if not candidates:
        for row in witness.get("raw_results", []):
            if row.get("witness"):
                candidates.append(row["witness"])
    starts = []
    zero = np.zeros(problem.n, dtype=np.float64)
    for idx, cand in enumerate(candidates[:2]):
        g = np.asarray(cand["g"], dtype=np.float64)
        xi = np.asarray(cand["xi"], dtype=np.float64)
        starts.append({"name": "accepted_12866_{}".format(idx), "x": pack_g_xi(problem, g, xi), "t": 0.0})
    starts.append({"name": "frozen_g0_zero_slack", "x": pack_g_xi(problem, problem.g0, zero), "t": 0.0})
    return starts


def self_check_core() -> dict[str, Any]:
    design = load_design()
    cfg, oriented, fixture, witness, replay = load_inputs()
    system, cells = build_cells(fixture, cfg)
    p = build_problem(fixture, cfg, cells[0])
    set_order = expected_set_order(p.n)
    source_order = oriented.get("set_order", [])
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    signed = signed_gap_residuals(p, g0)
    thresholds_ok = (
        int(cfg["solver"]["topk"]) == TOPK
        and float(cfg["solver"]["tie_tolerance"]) == TAU
        and float(cfg["solver"]["dykstra_set_violation_tolerance"]) == 1e-6
        and float(cfg["solver"]["dykstra_relative_change_tolerance"]) == 1e-7
        and int(cfg["solver"]["max_independent_orientations"]) == 8
        and int(cfg["solver"]["max_pivots"]) == 32
    )
    return {
        "ok": bool(
            design.get("eta") == ETA
            and thresholds_ok
            and p.n == 24
            and len(cells) == len(system["compatible_assignments"]) == 2
            and int(system["rank"]) == 1
            and system["complete_adjacent_enumeration"] is True
            and len(system["descriptors"]) == 1
            and len(set_order) == 589
            and source_order == set_order
            and len(p.original_rank_rows) == 528
            and len(p.signed_edges) == 528
            and signed["row_count"] == 528
            and replay.get("accepted_feasible_replayed") is True
            and supervision_contract(cfg).get("ok") is True
            and existing_hashes_unchanged().get("ok") is True
        ),
        "thread_session": design.get("thread_session"),
        "eta": design.get("eta"),
        "design_sha256": hfile(DESIGN),
        "thresholds_ok": thresholds_ok,
        "n": p.n,
        "cell_count": len(cells),
        "orientation_rank": system["rank"],
        "orientation_descriptors": system["descriptors"],
        "compatible_assignments": system["compatible_assignments"],
        "reject_reasons": system["reject_reasons"],
        "constraint_set_count": len(set_order),
        "source_set_order_matches": source_order == set_order,
        "original_rank_halfspace_count": len(p.original_rank_rows),
        "signed_gap_count": len(p.signed_edges),
        "signed_gap_g0_cell0": {k: v for k, v in signed.items() if k != "rows"},
        "accepted_12866_replay": replay.get("accepted_feasible_replayed"),
        "supervision_contract": supervision_contract(cfg),
        "existing_hashes_unchanged": existing_hashes_unchanged(),
        "source_hashes": current_source_hashes(),
    }
