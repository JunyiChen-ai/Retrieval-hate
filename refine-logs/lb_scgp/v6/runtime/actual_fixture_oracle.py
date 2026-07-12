#!/usr/bin/env python3
"""v6 actual frozen fixture oracle.

Prospective, nonformal execution only.  This reads the frozen v5
feasible_oriented_boundary fixture and solves direct Gram-space numerical
problems for the original 589 product constraints.  It does not write outside
refine-logs/lb_scgp/v6.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, linprog, lsq_linear, minimize


ROOT = Path("/data/jehc223/RGCL")
V6 = ROOT / "refine-logs" / "lb_scgp" / "v6"
OUT_DIR = V6 / "results"
V5_CONFIG = ROOT / "configs" / "lb_scgp" / "lb_scgp_v5.json"
V5_DYKSTRA = ROOT / "artifacts" / "lb_scgp" / "v5" / "g0" / "synthetic" / "dykstra.jsonl"
V5_FREEZE = ROOT / "artifacts" / "lb_scgp" / "v5" / "CONFIG_FREEZE.json"
WITNESS_12866 = V6 / "results" / "analytic_feasibility_witness_12866.json"
REPLAY_12866 = V6 / "results" / "analytic_witness_replay_12866.json"
MAIN_RECORD = V6 / "G0_V6_ACTUAL_FIXTURE_ORACLE.md"

TOPK = 20
FULL_RANK_TOL = 1e-8
PHASE1_RESIDUAL_TOL = 1e-8
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
    with path.open("xb") as handle:
        handle.write((cjson(obj) + "\n").encode("utf-8"))


def append_jsonl(path: Path, obj: Any) -> None:
    with path.open("ab") as handle:
        handle.write((cjson(obj) + "\n").encode("utf-8"))


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
                "query": i, "a": a, "b": b, "kind": "internal",
                "index": index, "name": "rank_internal_{:04d}".format(index),
                "rhs": canonical_rhs(ids, a, b, tolerance),
            })
            index += 1
        for outsider in full[TOPK:]:
            a, b = top[TOPK - 1], outsider
            rows.append({
                "query": i, "a": a, "b": b, "kind": "boundary",
                "index": index, "name": "rank_boundary_{:04d}".format(index),
                "rhs": canonical_rhs(ids, a, b, tolerance),
            })
            index += 1
    return rows


def boundary_orientation_system(gram: np.ndarray, ids: list[str], tolerance: float, limit: int) -> dict[str, Any]:
    n = len(ids)
    full = stable_rankings(gram, ids, topk=n - 1, tolerance=tolerance)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = {pair: k for k, pair in enumerate(pairs)}
    descriptors: list[tuple[str, str, str]] = []
    normals: list[np.ndarray] = []
    for i, ranking in enumerate(full):
        boundary = ranking[TOPK - 1]
        value = float(gram[i, boundary])
        for outsider in ranking[TOPK:]:
            if abs(value - float(gram[i, outsider])) <= tolerance:
                desc = (str(ids[i]), str(ids[boundary]), str(ids[outsider]))
                if desc in descriptors:
                    continue
                normal = np.zeros(len(pairs), dtype=np.float64)
                for j, sign in ((boundary, 1.0), (outsider, -1.0)):
                    normal[pair_index[tuple(sorted((i, j)))]] += sign
                descriptors.append(desc)
                normals.append(normal)
    if not normals:
        return {
            "rank": 0,
            "descriptors": [],
            "basis_indices": [],
            "dependency_coefficients": [],
            "compatible_assignments": [[]],
            "compatible_overflow": False,
        }
    matrix = np.stack(normals)
    basis: list[int] = []
    current = np.zeros((0, matrix.shape[1]), dtype=np.float64)
    for idx, row in enumerate(matrix):
        candidate = np.vstack([current, row])
        if np.linalg.matrix_rank(candidate, tol=1e-12) > len(basis):
            basis.append(idx)
            current = candidate
    coeff = np.linalg.lstsq(matrix[basis].T, matrix.T, rcond=1e-12)[0].T
    assignments: list[list[int]] = []
    overflow = False

    def feasible(signs: list[int]) -> bool:
        if not signs:
            return True
        result = linprog(
            np.zeros(matrix.shape[1]),
            A_ub=-np.asarray(signs, dtype=np.float64)[:, None] * matrix[:len(signs)],
            b_ub=-np.ones(len(signs)),
            bounds=[(-10.0, 10.0)] * matrix.shape[1],
            method="highs",
        )
        return bool(result.success)

    def dfs(signs: list[int]) -> None:
        nonlocal overflow
        if overflow:
            return
        if len(signs) == len(normals):
            assignments.append(list(signs))
            overflow = len(assignments) >= limit
            return
        for sign in (-1, 1):
            cand = signs + [sign]
            if feasible(cand):
                dfs(cand)

    dfs([])
    return {
        "rank": len(basis),
        "descriptors": [list(x) for x in descriptors],
        "basis_indices": basis,
        "dependency_coefficients": coeff.tolist(),
        "compatible_assignments": assignments,
        "compatible_overflow": overflow,
    }


def orientation_cell_from_assignment(
    base_rankings: list[list[int]], descriptors: list[list[str]], assignment: list[int], ids: list[str]
) -> list[list[int]]:
    id_to_row = {str(vid): i for i, vid in enumerate(ids)}
    by_query: dict[str, list[tuple[int, int, int]]] = {}
    for sign, (query, a_id, b_id) in zip(assignment, descriptors):
        by_query.setdefault(query, []).append((int(sign), id_to_row[a_id], id_to_row[b_id]))
    cells = [list(row) for row in base_rankings]
    for query, constraints in by_query.items():
        qi = id_to_row[query]
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
    md: dict[str, Any]
    rank_rows: list[dict[str, Any]]
    centroid: np.ndarray
    caps: list[float]
    row_radius: float
    class_radius: float
    pairs: list[tuple[int, int]]
    pair_index: dict[tuple[int, int], int]
    linear_rows: list[LinearRow]
    eq_a: np.ndarray
    eq_b: np.ndarray

    @property
    def dim(self) -> int:
        return len(self.pairs) + self.n


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
            "formula": "for query q=0..23: ordinals 61+22q..61+22q+18 are rank_internal_{22q..22q+18}; ordinals 61+22q+19..61+22q+21 are rank_boundary_{22q+19..22q+21}",
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


def build_problem(fixture: dict[str, Any], cfg: dict[str, Any], full_rankings: list[list[int]],
                  cell_index: int, assignment: list[int]) -> DirectProblem:
    solver = cfg["solver"]
    n = int(fixture["n"])
    ids = [str(x) for x in fixture["ids"]]
    labels = np.asarray(fixture["labels"], dtype=np.int64)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    ell = np.asarray(fixture["ell"], dtype=np.float64)
    semantic = semantic_matrix(fixture["semantic"], n)
    md = margin_data(g0, labels, ids, full_rankings, float(solver["tie_tolerance"]))
    deficits = np.maximum(ell - md["baseline_margins"], 0.0)
    caps = [float(solver["slack_budget_ratio"]) * float(deficits[rows].sum()) for rows in groups]
    centroid = centroid_direction(labels)
    rank_rows = rank_halfspaces(ids, full_rankings, float(solver["tie_tolerance"]))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = {pair: k for k, pair in enumerate(pairs)}
    linear_rows: list[LinearRow] = []

    def add_row(name: str, group: str, a: np.ndarray | None, x: np.ndarray | None, const: float) -> None:
        aa = np.zeros((n, n), dtype=np.float64) if a is None else np.asarray(a, dtype=np.float64)
        coeff, cc = reduced_affine(pairs, n, aa, x, const)
        linear_rows.append(LinearRow(name=name, coeff=coeff, const=cc, group=group))

    for k, rows in enumerate(groups):
        x = np.zeros(n, dtype=np.float64)
        x[rows] = -1.0
        add_row("slack_capped_simplex_{}".format(k), "slack", None, x, caps[k])
    for i in range(n):
        a = np.zeros((n, n), dtype=np.float64)
        for r in range(TOPK):
            a[i, int(md["top"][i, r])] += float(md["coeff"][i, r])
        x = np.zeros(n, dtype=np.float64)
        x[i] = 1.0
        add_row("vote_slack_{:02d}".format(i), "vote", a, x, -float(ell[i]))
    for k, rows in enumerate(groups):
        a = np.zeros((n, n), dtype=np.float64)
        for i in rows:
            for r in range(TOPK):
                a[int(i), int(md["top"][int(i), r])] += float(md["coeff"][int(i), r]) / len(rows)
        add_row("class_mean_margin_{}".format(k), "class_margin", a, None, -float(md["baseline_margins"][rows].mean()))
    a_global = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for r in range(TOPK):
            a_global[i, int(md["top"][i, r])] += float(md["coeff"][i, r]) / n
    add_row("global_mean_margin", "global_margin", a_global, None, -float(md["baseline_margins"].mean()))
    add_row("centroid_distance", "centroid", centroid, None, -float(np.sum(centroid * g0)))
    for row in rank_rows:
        a = np.zeros((n, n), dtype=np.float64)
        a[int(row["query"]), int(row["a"])] += 1.0
        a[int(row["query"]), int(row["b"])] -= 1.0
        add_row(str(row["name"]), "rank_halfspaces", a, None, -float(row["rhs"]))

    eq_rows = []
    eq_rhs = []
    for row in semantic.reshape((-1, n, n)):
        coeff, cc = reduced_affine(pairs, n, row, None, 0.0)
        eq_rows.append(coeff)
        eq_rhs.append(-cc)
    eq_a = np.stack(eq_rows).astype(np.float64) if eq_rows else np.zeros((0, len(pairs) + n), dtype=np.float64)
    eq_b = np.asarray(eq_rhs, dtype=np.float64)
    return DirectProblem(
        cell_index=cell_index,
        assignment=assignment,
        n=n,
        ids=ids,
        labels=labels,
        groups=groups,
        g0=g0,
        ell=ell,
        semantic=semantic,
        solver=solver,
        full_rankings=full_rankings,
        md=md,
        rank_rows=rank_rows,
        centroid=centroid,
        caps=caps,
        row_radius=float(solver["row_trust_scale"]) * math.sqrt(n - 1),
        class_radius=float(solver["class_mean_trust_scale"]) * math.sqrt(n),
        pairs=pairs,
        pair_index=pair_index,
        linear_rows=linear_rows,
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


def linear_values(problem: DirectProblem, x: np.ndarray) -> np.ndarray:
    if not problem.linear_rows:
        return np.zeros(0, dtype=np.float64)
    a = np.stack([row.coeff for row in problem.linear_rows])
    c = np.asarray([row.const for row in problem.linear_rows], dtype=np.float64)
    return a @ x + c


def nonlinear_values_jac(problem: DirectProblem, x: np.ndarray, phase1: bool) -> tuple[np.ndarray, np.ndarray]:
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
            for a, b in problem.pairs:
                deriv = 0.0
                if a in rows:
                    deriv += float(mean[b]) / len(rows)
                if b in rows:
                    deriv += float(mean[a]) / len(rows)
                row[problem.pair_index[(a, b)]] = -deriv / norm
        values.append(problem.class_radius - norm)
        jac_rows.append(row)
    return np.asarray(values, dtype=np.float64), np.stack(jac_rows).astype(np.float64)


def residual_sets(problem: DirectProblem, g: np.ndarray, xi: np.ndarray) -> tuple[list[dict[str, Any]], dict[str, float]]:
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
        add("row_trust_{:02d}".format(i), "row_trust",
            float(np.linalg.norm(np.delete(g[i] - problem.g0[i], i)) - problem.row_radius))
    for k, label in enumerate((0, 1)):
        rows_k = np.flatnonzero(problem.labels == label)
        add("class_mean_trust_{}".format(k), "class_mean_trust",
            float(np.linalg.norm((g[rows_k] - problem.g0[rows_k]).mean(axis=0)) - problem.class_radius))
    add("semantic_radius_zero", "semantic", float(np.linalg.norm(problem.semantic @ g.reshape(-1))))
    margins = np.einsum(
        "ir,ir->i",
        problem.md["coeff"],
        g[np.arange(problem.n)[:, None], problem.md["top"]],
        optimize=True,
    )
    for k, label in enumerate((0, 1)):
        rows_k = np.flatnonzero(problem.labels == label)
        add("slack_capped_simplex_{}".format(k), "slack",
            max(float(-xi[rows_k].min()), float(xi[rows_k].sum() - problem.caps[k])))
    for i in range(problem.n):
        add("vote_slack_{:02d}".format(i), "vote", float(problem.ell[i] - margins[i] - xi[i]))
    for k, label in enumerate((0, 1)):
        rows_k = np.flatnonzero(problem.labels == label)
        add("class_mean_margin_{}".format(k), "class_margin",
            float(problem.md["baseline_margins"][rows_k].mean() - margins[rows_k].mean()))
    add("global_mean_margin", "global_margin", float(problem.md["baseline_margins"].mean() - margins.mean()))
    add("centroid_distance", "centroid", float(np.sum(problem.centroid * problem.g0) - np.sum(problem.centroid * g)))
    for row in problem.rank_rows:
        add(str(row["name"]), "rank_halfspaces",
            float(row["rhs"]) - float(g[int(row["query"]), int(row["a"])] - g[int(row["query"]), int(row["b"])]))
    if len(rows) != 589:
        raise RuntimeError("internal residual set count mismatch: {}".format(len(rows)))
    groups: dict[str, float] = {}
    for row in rows:
        groups[row["group"]] = max(groups.get(row["group"], 0.0), float(row["residual"]))
    return rows, groups


def max_residual(rows: list[dict[str, Any]]) -> float:
    return float(max((float(row["residual"]) for row in rows), default=float("inf")))


def objective_value_grad(problem: DirectProblem, x: np.ndarray) -> tuple[float, np.ndarray]:
    g, xi = unpack_g_xi(problem, x)
    value = 0.5 * float(np.sum((g - problem.g0) ** 2)) + 0.5 * float(np.dot(xi, xi))
    grad = np.zeros(problem.dim, dtype=np.float64)
    for k, (i, j) in enumerate(problem.pairs):
        grad[k] = 2.0 * float(g[i, j] - problem.g0[i, j])
    grad[len(problem.pairs):] = xi
    return value, grad


def phase1_solve(problem: DirectProblem, starts: list[dict[str, Any]], checkpoint: Path) -> dict[str, Any]:
    p = len(problem.pairs)
    n = problem.n
    lower = np.concatenate([
        np.full(p, -1.0, dtype=np.float64),
        np.zeros(n, dtype=np.float64),
        np.asarray([-1e-6], dtype=np.float64),
    ])
    upper = np.concatenate([
        np.full(p, float(problem.solver["offdiag_upper"]), dtype=np.float64),
        np.full(n, max(1.0, max(problem.caps, default=0.0) + 1.0), dtype=np.float64),
        np.asarray([1.0], dtype=np.float64),
    ])
    bounds = Bounds(lower, upper)
    a_lin = np.stack([row.coeff for row in problem.linear_rows])
    b_lin = -np.asarray([row.const for row in problem.linear_rows], dtype=np.float64)
    lin = LinearConstraint(np.hstack([a_lin, np.zeros((a_lin.shape[0], 1), dtype=np.float64)]), b_lin, np.inf)
    eq = LinearConstraint(np.hstack([problem.eq_a, np.zeros((problem.eq_a.shape[0], 1), dtype=np.float64)]), problem.eq_b, problem.eq_b)
    nl = NonlinearConstraint(
        lambda z: nonlinear_values_jac(problem, z, phase1=True)[0],
        np.zeros(problem.n + 3, dtype=np.float64),
        np.full(problem.n + 3, np.inf, dtype=np.float64),
        jac=lambda z: nonlinear_values_jac(problem, z, phase1=True)[1],
    )

    def fun(z: np.ndarray) -> float:
        return -float(z[-1])

    def jac(z: np.ndarray) -> np.ndarray:
        out = np.zeros_like(z)
        out[-1] = -1.0
        return out

    attempts = []
    best: dict[str, Any] | None = None
    for start in starts:
        z0 = np.concatenate([start["x"], np.asarray([start.get("t", 0.0)], dtype=np.float64)])
        z0 = np.minimum(np.maximum(z0, lower), upper)
        started = time.perf_counter()
        error = None
        result = None
        try:
            result = minimize(
                fun,
                z0,
                method="SLSQP",
                jac=jac,
                bounds=bounds,
                constraints=[lin, eq, nl],
                options={"maxiter": 500, "ftol": 1e-12, "disp": False},
            )
        except Exception as exc:
            error = "{}: {}".format(type(exc).__name__, str(exc))
        elapsed = time.perf_counter() - started
        if result is None:
            attempt = {"start_name": start["name"], "status": "error", "error": error, "elapsed_seconds": elapsed}
        else:
            z = np.asarray(result.x, dtype=np.float64)
            g, xi = unpack_g_xi(problem, z[:-1])
            res_rows, res_groups = residual_sets(problem, g, xi)
            eig_min = float(np.linalg.eigvalsh(0.5 * (g + g.T)).min())
            realized = [row[:TOPK] for row in stable_rankings(g, problem.ids, problem.n - 1, float(problem.solver["tie_tolerance"]))]
            target = [row[:TOPK] for row in problem.full_rankings]
            phase1_ok = bool(max_residual(res_rows) <= PHASE1_RESIDUAL_TOL and eig_min > FULL_RANK_TOL and realized == target)
            attempt = {
                "start_name": start["name"],
                "status": "FULL_RANK_SLATER_CANDIDATE" if phase1_ok else "NO_WITNESS_CANDIDATE",
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "elapsed_seconds": elapsed,
                "nit": None if getattr(result, "nit", None) is None else int(result.nit),
                "objective_maximized_t": float(z[-1]),
                "psd_min_eigenvalue": eig_min,
                "full_rank_margin": eig_min,
                "max_589_residual": max_residual(res_rows),
                "residual_groups": res_groups,
                "realized_top20_equal_cell": realized == target,
                "realized_top20_sha256": hobj(realized),
                "target_top20_sha256": hobj(target),
                "witness": {
                    "g": g.tolist(),
                    "xi": xi.tolist(),
                    "g_sha256": hobj(g.tolist()),
                    "xi_sha256": hobj(xi.tolist()),
                    "witness_sha256": hobj({"g": g.tolist(), "xi": xi.tolist()}),
                },
            }
        attempts.append(attempt)
        append_jsonl(checkpoint, {
            "phase": "phase_i",
            "cell_index": problem.cell_index,
            "start_name": start["name"],
            "status": attempt.get("status"),
            "max_589_residual": attempt.get("max_589_residual"),
            "full_rank_margin": attempt.get("full_rank_margin"),
            "elapsed_seconds": attempt.get("elapsed_seconds"),
        })
        if best is None or float(attempt.get("full_rank_margin", -float("inf"))) > float(best.get("full_rank_margin", -float("inf"))):
            best = attempt
    accepted = [row for row in attempts if row.get("status") == "FULL_RANK_SLATER_CANDIDATE"]
    if not accepted:
        anchor_attempts = phase1_anchor_polish(problem, starts, checkpoint)
        attempts.extend(anchor_attempts)
        for attempt in anchor_attempts:
            if best is None or float(attempt.get("full_rank_margin", -float("inf"))) > float(best.get("full_rank_margin", -float("inf"))):
                best = attempt
        accepted = [row for row in attempts if row.get("status") == "FULL_RANK_SLATER_CANDIDATE"]
    selected = max(accepted, key=lambda row: float(row["full_rank_margin"])) if accepted else best
    return {
        "cell_index": problem.cell_index,
        "assignment": problem.assignment,
        "status": "FULL_RANK_SLATER_REPLAY_PENDING" if accepted else "NO_WITNESS",
        "selected_start_name": None if selected is None else selected.get("start_name"),
        "selected_full_rank_margin": None if selected is None else selected.get("full_rank_margin"),
        "selected_max_589_residual": None if selected is None else selected.get("max_589_residual"),
        "attempts": attempts,
        "selected": selected,
    }


def phase1_anchor_polish(problem: DirectProblem, starts: list[dict[str, Any]], checkpoint: Path) -> list[dict[str, Any]]:
    p = len(problem.pairs)
    n = problem.n
    lower = np.concatenate([np.full(p, -1.0, dtype=np.float64), np.zeros(n, dtype=np.float64)])
    upper = np.concatenate([
        np.full(p, float(problem.solver["offdiag_upper"]), dtype=np.float64),
        np.full(n, max(1.0, max(problem.caps, default=0.0) + 1.0), dtype=np.float64),
    ])
    bounds = Bounds(lower, upper)
    a_lin = np.stack([row.coeff for row in problem.linear_rows])
    b_lin = -np.asarray([row.const for row in problem.linear_rows], dtype=np.float64)
    lin = LinearConstraint(a_lin, b_lin, np.inf)
    eq = LinearConstraint(problem.eq_a, problem.eq_b, problem.eq_b)
    out: list[dict[str, Any]] = []
    references = starts[:1] + [row for row in starts if row["name"] == "frozen_g0_zero_slack"]
    seen: set[str] = set()
    unique_refs = []
    for row in references:
        digest = hobj(np.asarray(row["x"], dtype=np.float64).round(15).tolist())
        if digest not in seen:
            seen.add(digest)
            unique_refs.append(row)
    for tau in (1e-6, 1e-5, 1e-4):
        nl = NonlinearConstraint(
            lambda z, tau=tau: nonlinear_values_jac(problem, z, phase1=False)[0],
            np.concatenate([np.asarray([tau], dtype=np.float64), np.zeros(problem.n + 2, dtype=np.float64)]),
            np.full(problem.n + 3, np.inf, dtype=np.float64),
            jac=lambda z, tau=tau: nonlinear_values_jac(problem, z, phase1=False)[1],
        )
        for ref in unique_refs:
            x_ref = np.asarray(ref["x"], dtype=np.float64)
            x0 = np.minimum(np.maximum(x_ref, lower), upper)

            def fun(z: np.ndarray, x_ref: np.ndarray = x_ref) -> float:
                d = z - x_ref
                return 0.5 * float(np.dot(d, d))

            def jac(z: np.ndarray, x_ref: np.ndarray = x_ref) -> np.ndarray:
                return np.asarray(z - x_ref, dtype=np.float64)

            started = time.perf_counter()
            error = None
            result = None
            try:
                result = minimize(
                    fun,
                    x0,
                    method="SLSQP",
                    jac=jac,
                    bounds=bounds,
                    constraints=[lin, eq, nl],
                    options={"maxiter": 500, "ftol": 1e-12, "disp": False},
                )
            except Exception as exc:
                error = "{}: {}".format(type(exc).__name__, str(exc))
            elapsed = time.perf_counter() - started
            if result is None:
                attempt = {
                    "start_name": "anchor_{}_tau_{:.0e}".format(ref["name"], tau),
                    "status": "error",
                    "error": error,
                    "elapsed_seconds": elapsed,
                    "anchor_tau": tau,
                }
            else:
                x = np.asarray(result.x, dtype=np.float64)
                g, xi = unpack_g_xi(problem, x)
                res_rows, res_groups = residual_sets(problem, g, xi)
                eig_min = float(np.linalg.eigvalsh(0.5 * (g + g.T)).min())
                realized = [row[:TOPK] for row in stable_rankings(g, problem.ids, problem.n - 1, float(problem.solver["tie_tolerance"]))]
                target = [row[:TOPK] for row in problem.full_rankings]
                phase1_ok = bool(max_residual(res_rows) <= PHASE1_RESIDUAL_TOL and eig_min > FULL_RANK_TOL and realized == target)
                attempt = {
                    "start_name": "anchor_{}_tau_{:.0e}".format(ref["name"], tau),
                    "status": "FULL_RANK_SLATER_CANDIDATE" if phase1_ok else "NO_WITNESS_CANDIDATE",
                    "optimizer_success": bool(result.success),
                    "optimizer_message": str(result.message),
                    "elapsed_seconds": elapsed,
                    "nit": None if getattr(result, "nit", None) is None else int(result.nit),
                    "anchor_tau": tau,
                    "anchor_objective": float(fun(x)),
                    "psd_min_eigenvalue": eig_min,
                    "full_rank_margin": eig_min,
                    "max_589_residual": max_residual(res_rows),
                    "residual_groups": res_groups,
                    "realized_top20_equal_cell": realized == target,
                    "realized_top20": realized,
                    "target_top20": target,
                    "realized_top20_sha256": hobj(realized),
                    "target_top20_sha256": hobj(target),
                    "witness": {
                        "g": g.tolist(),
                        "xi": xi.tolist(),
                        "g_sha256": hobj(g.tolist()),
                        "xi_sha256": hobj(xi.tolist()),
                        "witness_sha256": hobj({"g": g.tolist(), "xi": xi.tolist()}),
                    },
                }
            out.append(attempt)
            append_jsonl(checkpoint, {
                "phase": "phase_i_anchor",
                "cell_index": problem.cell_index,
                "start_name": attempt.get("start_name"),
                "status": attempt.get("status"),
                "max_589_residual": attempt.get("max_589_residual"),
                "full_rank_margin": attempt.get("full_rank_margin"),
                "realized_top20_equal_cell": attempt.get("realized_top20_equal_cell"),
                "elapsed_seconds": attempt.get("elapsed_seconds"),
            })
            if attempt.get("status") == "FULL_RANK_SLATER_CANDIDATE":
                return out
    return out


def phase2_solve(problem: DirectProblem, start: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    p = len(problem.pairs)
    n = problem.n
    lower = np.concatenate([np.full(p, -1.0, dtype=np.float64), np.zeros(n, dtype=np.float64)])
    upper = np.concatenate([
        np.full(p, float(problem.solver["offdiag_upper"]), dtype=np.float64),
        np.full(n, max(1.0, max(problem.caps, default=0.0) + 1.0), dtype=np.float64),
    ])
    bounds = Bounds(lower, upper)
    a_lin = np.stack([row.coeff for row in problem.linear_rows])
    b_lin = -np.asarray([row.const for row in problem.linear_rows], dtype=np.float64)
    lin = LinearConstraint(a_lin, b_lin, np.inf)
    eq = LinearConstraint(problem.eq_a, problem.eq_b, problem.eq_b)
    nl = NonlinearConstraint(
        lambda z: nonlinear_values_jac(problem, z, phase1=False)[0],
        np.zeros(problem.n + 3, dtype=np.float64),
        np.full(problem.n + 3, np.inf, dtype=np.float64),
        jac=lambda z: nonlinear_values_jac(problem, z, phase1=False)[1],
    )
    x0 = pack_g_xi(problem, np.asarray(start["g"], dtype=np.float64), np.asarray(start["xi"], dtype=np.float64))
    x0 = np.minimum(np.maximum(x0, lower), upper)

    def fun(x: np.ndarray) -> float:
        return objective_value_grad(problem, x)[0]

    def jac(x: np.ndarray) -> np.ndarray:
        return objective_value_grad(problem, x)[1]

    started = time.perf_counter()
    error = None
    result = None
    try:
        result = minimize(
            fun,
            x0,
            method="SLSQP",
            jac=jac,
            bounds=bounds,
            constraints=[lin, eq, nl],
            options={"maxiter": 500, "ftol": 1e-12, "disp": False},
        )
    except Exception as exc:
        error = "{}: {}".format(type(exc).__name__, str(exc))
    elapsed = time.perf_counter() - started
    if result is None:
        return {"cell_index": problem.cell_index, "status": "error", "error": error, "elapsed_seconds": elapsed}
    x = np.asarray(result.x, dtype=np.float64)
    g, xi = unpack_g_xi(problem, x)
    res_rows, res_groups = residual_sets(problem, g, xi)
    objective, grad = objective_value_grad(problem, x)
    kkt = kkt_certificate(problem, x, grad)
    eig_min = float(np.linalg.eigvalsh(0.5 * (g + g.T)).min())
    realized_full = stable_rankings(g, problem.ids, topk=problem.n - 1, tolerance=float(problem.solver["tie_tolerance"]))
    realized_top20 = [row[:TOPK] for row in realized_full]
    target_top20 = [row[:TOPK] for row in problem.full_rankings]
    local_supportable = bool(
        max_residual(res_rows) <= float(problem.solver["dykstra_set_violation_tolerance"])
        and realized_top20 == target_top20
        and kkt["stationarity_inf"] <= 1e-6
        and kkt["dual_min"] >= -1e-8
        and kkt["complementarity_inf"] <= 1e-6
        and kkt["psd_dual_ok"]
        and kkt["active_set_complete"]
    )
    out = {
        "cell_index": problem.cell_index,
        "assignment": problem.assignment,
        "status": "LOCAL_STATIONARY_CERTIFIED_CANDIDATE" if local_supportable else "BOUNDED_REMOVE",
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "elapsed_seconds": elapsed,
        "nit": None if getattr(result, "nit", None) is None else int(result.nit),
        "objective": objective,
        "gram_displacement_fro": float(np.linalg.norm(g - problem.g0)),
        "slack_l2": float(np.linalg.norm(xi)),
        "psd_min_eigenvalue": eig_min,
        "max_589_residual": max_residual(res_rows),
        "residual_groups": res_groups,
        "realized_top20_equal_cell": realized_top20 == target_top20,
        "realized_full_equal_cell": realized_full == problem.full_rankings,
        "realized_top20_sha256": hobj(realized_top20),
        "target_top20_sha256": hobj(target_top20),
        "full_rankings_sha256": hobj(problem.full_rankings),
        "kkt": kkt,
        "vi": {
            "source": "kkt_stationarity_dual_complementarity_bound",
            "vi_residual_bound": float(kkt["stationarity_inf"] + kkt["complementarity_inf"] + max(0.0, -kkt["dual_min"])),
        },
        "witness": {
            "g": g.tolist(),
            "xi": xi.tolist(),
            "g_sha256": hobj(g.tolist()),
            "xi_sha256": hobj(xi.tolist()),
            "witness_sha256": hobj({"g": g.tolist(), "xi": xi.tolist()}),
        },
    }
    append_jsonl(checkpoint, {
        "phase": "phase_ii",
        "cell_index": problem.cell_index,
        "status": out["status"],
        "objective": objective,
        "max_589_residual": out["max_589_residual"],
        "stationarity_inf": kkt["stationarity_inf"],
        "psd_min_eigenvalue": eig_min,
        "elapsed_seconds": elapsed,
    })
    return out


def kkt_certificate(problem: DirectProblem, x: np.ndarray, grad: np.ndarray) -> dict[str, Any]:
    lin_values = linear_values(problem, x)
    nl_values, nl_jac = nonlinear_values_jac(problem, x, phase1=False)
    eq_res = problem.eq_a @ x - problem.eq_b
    active_names: list[str] = []
    active_values: list[float] = []
    active_jac: list[np.ndarray] = []
    for row, value in zip(problem.linear_rows, lin_values):
        if float(value) <= ACTIVE_TOL:
            active_names.append(row.name)
            active_values.append(float(value))
            active_jac.append(row.coeff)
    nl_names = ["psd_min_eigenvalue"]
    nl_names.extend("row_trust_{:02d}".format(i) for i in range(problem.n))
    nl_names.extend("class_mean_trust_{}".format(i) for i in range(len(problem.groups)))
    psd_active = bool(float(nl_values[0]) <= ACTIVE_TOL)
    for name, value, jacrow in zip(nl_names, nl_values, nl_jac):
        if float(value) <= ACTIVE_TOL:
            active_names.append(name)
            active_values.append(float(value))
            active_jac.append(np.asarray(jacrow, dtype=np.float64))
    lower = np.concatenate([
        np.full(len(problem.pairs), -1.0, dtype=np.float64),
        np.zeros(problem.n, dtype=np.float64),
    ])
    upper = np.concatenate([
        np.full(len(problem.pairs), float(problem.solver["offdiag_upper"]), dtype=np.float64),
        np.full(problem.n, max(1.0, max(problem.caps, default=0.0) + 1.0), dtype=np.float64),
    ])
    for k, value in enumerate(x):
        if float(value - lower[k]) <= ACTIVE_TOL:
            row = np.zeros(problem.dim, dtype=np.float64)
            row[k] = 1.0
            active_names.append("bound_lower_{}".format(k))
            active_values.append(float(value - lower[k]))
            active_jac.append(row)
        if float(upper[k] - value) <= ACTIVE_TOL:
            row = np.zeros(problem.dim, dtype=np.float64)
            row[k] = -1.0
            active_names.append("bound_upper_{}".format(k))
            active_values.append(float(upper[k] - value))
            active_jac.append(row)
    j_eq = problem.eq_a
    j_act = np.stack(active_jac).astype(np.float64) if active_jac else np.zeros((0, problem.dim), dtype=np.float64)
    mat = np.hstack([j_eq.T, -j_act.T])
    lb = np.concatenate([np.full(j_eq.shape[0], -np.inf), np.zeros(j_act.shape[0])])
    ub = np.full(j_eq.shape[0] + j_act.shape[0], np.inf)
    if mat.shape[1] == 0:
        multipliers = np.zeros(0, dtype=np.float64)
        stationarity = grad.copy()
        lsq_status = "no_active_columns"
    else:
        result = lsq_linear(mat, -grad, bounds=(lb, ub), tol=1e-12, max_iter=1000)
        multipliers = np.asarray(result.x, dtype=np.float64)
        stationarity = grad + mat @ multipliers
        lsq_status = str(result.status)
    eq_m = multipliers[:j_eq.shape[0]]
    ineq_m = multipliers[j_eq.shape[0]:]
    comp = np.asarray(active_values, dtype=np.float64) * ineq_m if len(active_values) else np.zeros(0, dtype=np.float64)
    g, _ = unpack_g_xi(problem, x)
    psd_index = active_names.index("psd_min_eigenvalue") if "psd_min_eigenvalue" in active_names else None
    psd_mu = 0.0 if psd_index is None else float(ineq_m[psd_index])
    eigvals, eigvecs = np.linalg.eigh(0.5 * (g + g.T))
    v = eigvecs[:, int(np.argmin(eigvals))]
    s_psd = psd_mu * np.outer(v, v) if psd_active else np.zeros_like(g)
    s_eig_min = float(np.linalg.eigvalsh(0.5 * (s_psd + s_psd.T)).min())
    sg_norm = float(np.linalg.norm(s_psd @ g))
    psd_comp = float(abs(np.sum(s_psd * g)))
    if psd_active:
        psd_dual_ok = bool(psd_mu >= -1e-8 and s_eig_min >= -1e-8 and sg_norm <= 1e-5 and psd_comp <= 1e-5)
        psd_status = "active_scalar_min_eigen_dual"
    else:
        psd_dual_ok = bool(float(eigvals.min()) > ACTIVE_TOL and psd_mu == 0.0)
        psd_status = "inactive_S_psd_zero"
    active_set_complete = bool(
        all(float(vv) > ACTIVE_TOL or name in active_names for name, vv in zip([row.name for row in problem.linear_rows], lin_values))
        and all(float(vv) > ACTIVE_TOL or name in active_names for name, vv in zip(nl_names, nl_values))
    )
    return {
        "active_tolerance": ACTIVE_TOL,
        "eq_count": int(j_eq.shape[0]),
        "active_ineq_count": int(j_act.shape[0]),
        "active_names": active_names,
        "active_values": [float(vv) for vv in active_values],
        "eq_residual_inf": float(np.max(np.abs(eq_res))) if eq_res.size else 0.0,
        "stationarity_inf": float(np.max(np.abs(stationarity))) if stationarity.size else 0.0,
        "dual_min": float(np.min(ineq_m)) if ineq_m.size else 0.0,
        "complementarity_inf": float(np.max(np.abs(comp))) if comp.size else 0.0,
        "lsq_status": lsq_status,
        "eq_multipliers": eq_m.tolist(),
        "active_ineq_multipliers": ineq_m.tolist(),
        "active_set_complete": active_set_complete,
        "psd": {
            "status": psd_status,
            "active": psd_active,
            "mu": psd_mu,
            "s_psd_sha256": hobj(s_psd.tolist()),
            "s_psd_eig_min": s_eig_min,
            "sg_norm": sg_norm,
            "trace_sg_abs": psd_comp,
            "eig_margin": float(eigvals.min()),
        },
        "psd_dual_ok": psd_dual_ok,
    }


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


def build_cells(fixture: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = int(fixture["n"])
    ids = [str(x) for x in fixture["ids"]]
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    tol = float(cfg["solver"]["tie_tolerance"])
    base_full = stable_rankings(g0, ids, topk=n - 1, tolerance=tol)
    system = boundary_orientation_system(g0, ids, tolerance=tol, limit=34)
    cells = []
    for idx, assignment in enumerate(system["compatible_assignments"]):
        full = orientation_cell_from_assignment(base_full, system["descriptors"], assignment, ids)
        rows = rank_halfspaces(ids, full, tol)
        cells.append({
            "cell_index": int(idx),
            "assignment": assignment,
            "full_rankings": full,
            "final_top20_rankings": [row[:TOPK] for row in full],
            "rank_halfspace_count": len(rows),
            "internal_halfspace_count": sum(1 for row in rows if row["kind"] == "internal"),
            "boundary_halfspace_count": sum(1 for row in rows if row["kind"] == "boundary"),
            "cell_sha256": hobj(full),
            "top20_sha256": hobj([row[:TOPK] for row in full]),
        })
    return system, cells


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
        for eps in (1e-5, 1e-4, 1e-3):
            mixed = (1.0 - eps) * g + eps * np.eye(problem.n)
            starts.append({"name": "accepted_12866_identity_mix_{:.0e}".format(eps), "x": pack_g_xi(problem, mixed, xi), "t": eps * 0.25})
    starts.append({"name": "frozen_g0_zero_slack", "x": pack_g_xi(problem, problem.g0, zero), "t": 0.0})
    return starts


def self_check() -> dict[str, Any]:
    cfg, oriented, fixture, witness, replay = load_inputs()
    system, cells = build_cells(fixture, cfg)
    p = build_problem(fixture, cfg, cells[0]["full_rankings"], int(cells[0]["cell_index"]), cells[0]["assignment"])
    set_order = expected_set_order(p.n)
    source_order = oriented.get("set_order", [])
    starts = phase_starts(p, witness)
    x0 = starts[0]["x"]
    _, jac0 = nonlinear_values_jac(p, x0, phase1=False)
    return {
        "ok": bool(
            p.n == 24
            and len(cells) == len(system["compatible_assignments"])
            and len(set_order) == 589
            and source_order == set_order
            and len(p.rank_rows) == 528
            and jac0.shape == (27, p.dim)
            and replay.get("accepted_feasible_replayed") is True
            and supervision_contract(cfg).get("ok") is True
        ),
        "n": p.n,
        "cell_count": len(cells),
        "orientation_rank": system["rank"],
        "constraint_set_count": len(set_order),
        "source_set_order_matches": source_order == set_order,
        "rank_halfspace_count": len(p.rank_rows),
        "direct_variable_dim": p.dim,
        "nonlinear_jac_shape": list(jac0.shape),
        "accepted_12866_replay": replay.get("accepted_feasible_replayed"),
        "supervision_contract": supervision_contract(cfg),
    }


def main() -> int:
    if "--self-check" in sys.argv:
        print(cjson(self_check()))
        return 0 if self_check().get("ok") else 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    started = time.perf_counter()
    checkpoint = OUT_DIR / "actual_fixture_oracle_checkpoint_{}.jsonl".format(job_id)
    cfg, oriented, fixture, witness, replay = load_inputs()
    contract = supervision_contract(cfg)
    if not contract.get("ok"):
        raise RuntimeError("supervision contract failed")
    set_order = expected_set_order(int(fixture["n"]))
    if oriented.get("set_order") != set_order:
        raise RuntimeError("frozen set_order mismatch")
    system, cells = build_cells(fixture, cfg)
    if int(system["rank"]) > int(cfg["solver"]["max_independent_orientations"]):
        raise RuntimeError("orientation rank exceeds frozen budget")

    phase_i_results = []
    phase_ii_results = []
    problems: dict[int, DirectProblem] = {}
    for cell in cells:
        p = build_problem(fixture, cfg, cell["full_rankings"], int(cell["cell_index"]), cell["assignment"])
        problems[int(cell["cell_index"])] = p
        print(cjson({"event": "phase_i_start", "cell_index": p.cell_index, "job_id": job_id}), flush=True)
        p1 = phase1_solve(p, phase_starts(p, witness), checkpoint)
        phase_i_results.append(p1)
        print(cjson({
            "event": "phase_i_finish",
            "cell_index": p.cell_index,
            "status": p1["status"],
            "margin": p1.get("selected_full_rank_margin"),
            "job_id": job_id,
        }), flush=True)

    full_rank_cells = [
        row for row in phase_i_results
        if row.get("status") == "FULL_RANK_SLATER_REPLAY_PENDING" and isinstance(row.get("selected"), dict)
    ]
    if full_rank_cells:
        for row in full_rank_cells:
            p = problems[int(row["cell_index"])]
            selected_w = row["selected"]["witness"]
            print(cjson({"event": "phase_ii_start", "cell_index": p.cell_index, "job_id": job_id}), flush=True)
            p2 = phase2_solve(p, selected_w, checkpoint)
            phase_ii_results.append(p2)
            print(cjson({
                "event": "phase_ii_finish",
                "cell_index": p.cell_index,
                "status": p2["status"],
                "objective": p2.get("objective"),
                "job_id": job_id,
            }), flush=True)

    certified = [row for row in phase_ii_results if row.get("status") == "LOCAL_STATIONARY_CERTIFIED_CANDIDATE"]
    status = "LOCAL_STATIONARY_CERTIFIED_CANDIDATE_REPLAY_PENDING" if certified and len(certified) == len(cells) else (
        "BOUNDED_REMOVE" if full_rank_cells else "NO_WITNESS"
    )
    source_hashes = {
        "v5_config": hfile(V5_CONFIG),
        "v5_dykstra_jsonl": hfile(V5_DYKSTRA),
        "v5_freeze": hfile(V5_FREEZE),
        "accepted_witness_12866": hfile(WITNESS_12866),
        "accepted_replay_12866": hfile(REPLAY_12866),
        "actual_fixture_oracle.py": hfile(V6 / "runtime" / "actual_fixture_oracle.py"),
        "actual_fixture_replay.py": hfile(V6 / "runtime" / "actual_fixture_replay.py"),
        "validate_actual_fixture_v6.py": hfile(V6 / "runtime" / "validate_actual_fixture_v6.py"),
        "actual_fixture_oracle.sbatch": hfile(V6 / "runtime" / "actual_fixture_oracle.sbatch"),
        "validate_actual_fixture_v6.sbatch": hfile(V6 / "runtime" / "validate_actual_fixture_v6.sbatch"),
        "main_record": hfile(MAIN_RECORD),
    }
    out = {
        "schema_version": 1,
        "task": "lb_scgp_v6_actual_fixture_oracle",
        "slurm_job_id": job_id,
        "python": sys.version,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "elapsed_seconds": time.perf_counter() - started,
        "status": status,
        "nonclaims": [
            "No G0 PASS, freeze, formal gate, realfold, replay decision, or performance claim.",
            "NO_WITNESS/nonconvergence is not infeasibility.",
            "Factor feasibility is not presented as Gram optimality.",
        ],
        "immutable_thresholds": {
            "topk": cfg["solver"]["topk"],
            "maxiter": 500,
            "max_cycles": cfg["solver"]["max_dykstra_cycles"],
            "violation": cfg["solver"]["dykstra_set_violation_tolerance"],
            "relative": cfg["solver"]["dykstra_relative_change_tolerance"],
            "tie": cfg["solver"]["tie_tolerance"],
        },
        "supervision_boundary": contract,
        "constraint_map": constraint_map(),
        "constraint_set_count": len(set_order),
        "set_order_matches_frozen": oriented.get("set_order") == set_order,
        "orientation_system": system,
        "compatible_cells": cells,
        "accepted_12866": {
            "witness_path": str(WITNESS_12866.relative_to(ROOT)),
            "replay_path": str(REPLAY_12866.relative_to(ROOT)),
            "accepted_feasible_replayed": replay.get("accepted_feasible_replayed"),
            "selected_rank": witness.get("determination", {}).get("selected_rank"),
            "selected_witness_sha256": witness.get("determination", {}).get("selected_witness_sha256"),
            "replay_payload_sha256": replay.get("payload_sha256"),
        },
        "phase_i": phase_i_results,
        "phase_ii": phase_ii_results,
        "source_hashes": source_hashes,
        "checkpoint_path": str(checkpoint.relative_to(ROOT)),
    }
    out["payload_sha256"] = payload_hash(out)
    out_path = OUT_DIR / "actual_fixture_oracle_{}.json".format(job_id)
    write_json_exclusive(out_path, out)
    print(cjson({"status": status, "path": str(out_path.relative_to(ROOT)), "payload_sha256": out["payload_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
