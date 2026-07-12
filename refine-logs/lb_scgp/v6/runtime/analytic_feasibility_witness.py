#!/usr/bin/env python3
"""v6-only analytic-Jacobian feasibility witness supplement.

This prospective supplement does not modify frozen v5/v6 artifacts.  It reads
the frozen v5 synthetic oriented-boundary fixture, rebuilds the same v6
orientation cells, and searches for a primal feasibility witness with a
factor-space PSD parameterization.  Solver nonconvergence is never reported as
infeasibility.
"""

from __future__ import annotations

import argparse
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
from scipy.optimize import Bounds, linprog, minimize


ROOT = Path("/data/jehc223/RGCL")
V6 = ROOT / "refine-logs" / "lb_scgp" / "v6"
OUT_DIR = V6 / "results"
V5_DYKSTRA = ROOT / "artifacts" / "lb_scgp" / "v5" / "g0" / "synthetic" / "dykstra.jsonl"
V5_CONFIG = ROOT / "configs" / "lb_scgp" / "lb_scgp_v5.json"
V5_FREEZE = ROOT / "artifacts" / "lb_scgp" / "v5" / "CONFIG_FREEZE.json"
V5_LOG = ROOT / "slurm" / "logs" / "lbscgp_g0_cpu_12833.out"


def cjson(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hobj(obj: Any) -> str:
    return hashlib.sha256(cjson(obj).encode()).hexdigest()


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
    rows: list[Any] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json_exclusive(path: Path, obj: Any) -> None:
    with path.open("xb") as handle:
        handle.write((cjson(obj) + "\n").encode())


def append_jsonl(path: Path, obj: Any) -> None:
    with path.open("ab") as handle:
        handle.write((cjson(obj) + "\n").encode())


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
    ids = [str(x) for x in ids]
    out: list[list[int]] = []
    for i in range(len(ids)):
        candidates = [j for j in range(len(ids)) if j != i]
        local = tolerance_order(
            [float(gram[i, j]) for j in candidates],
            [ids[j] for j in candidates],
            tolerance,
        )
        out.append([candidates[k] for k in local[:topk]])
    return out


def canonical_rhs(ids: list[str], a: int, b: int, tolerance: float) -> float:
    return -float(tolerance) if str(ids[a]) < str(ids[b]) else float(np.nextafter(float(tolerance), math.inf))


def boundary_orientation_system(
    gram: np.ndarray,
    ids: list[str],
    topk: int,
    tolerance: float,
    compatible_limit: int,
) -> dict[str, Any]:
    n = len(ids)
    full = stable_rankings(gram, ids, topk=n - 1, tolerance=tolerance)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = {pair: k for k, pair in enumerate(pairs)}
    descriptors: list[tuple[str, str, str]] = []
    normals: list[np.ndarray] = []
    for i, ranking in enumerate(full):
        boundary = ranking[topk - 1]
        value = float(gram[i, boundary])
        for outsider in ranking[topk:]:
            if abs(value - float(gram[i, outsider])) <= tolerance:
                desc = (str(ids[i]), str(ids[boundary]), str(ids[outsider]))
                if desc in descriptors:
                    continue
                normal = np.zeros(len(pairs), dtype=np.float64)
                for j, sign in ((boundary, 1.0), (outsider, -1.0)):
                    if i == j:
                        continue
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
    current = np.zeros((0, matrix.shape[1]))
    for index, row in enumerate(matrix):
        candidate = np.vstack([current, row])
        if np.linalg.matrix_rank(candidate, tol=1e-12) > len(basis):
            basis.append(index)
            current = candidate
    coeff = np.linalg.lstsq(matrix[basis].T, matrix.T, rcond=1e-12)[0].T
    assignments: list[list[int]] = []
    overflow = False

    def feasible(signs: list[int]) -> bool:
        if not signs:
            return True
        a_ub = -np.asarray(signs, dtype=np.float64)[:, None] * matrix[:len(signs)]
        result = linprog(
            np.zeros(matrix.shape[1]),
            A_ub=a_ub,
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
            if len(assignments) >= compatible_limit:
                overflow = True
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
    base_rankings: list[list[int]],
    descriptors: list[list[str]],
    assignment: list[int],
    ids: list[str],
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
        for a, b in zip(baseline[:-1], baseline[1:]):
            if find(a) != find(b):
                edges[a].add(b)
        for sign, a, b in constraints:
            u, v = (a, b) if sign > 0 else (b, a)
            edges[u].add(v)
        for u in edges:
            for v in edges[u]:
                indegree[v] += 1
        order: list[int] = []
        baseline_pos = {node: i for i, node in enumerate(baseline)}
        available = [node for node in baseline if indegree[node] == 0]
        while available:
            available.sort(key=lambda node: (baseline_pos[node], str(ids[node])))
            u = available.pop(0)
            order.append(u)
            for v in sorted(edges[u], key=lambda node: (baseline_pos[node], str(ids[node]))):
                indegree[v] -= 1
                if indegree[v] == 0:
                    available.append(v)
        if len(order) != len(baseline):
            raise RuntimeError("compatible orientation produced cyclic rank DAG")
        cells[qi] = order
    return cells


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
    coeff = np.zeros((n, 20), dtype=np.float64)
    top = np.asarray([row[:20] for row in rankings], dtype=np.int64)
    for i in range(n):
        for r, j in enumerate(top[i], 1):
            coeff[i, r - 1] = signs[i] * (21 - r) * signs[j] / 210.0
    return top, coeff


def margin_data(
    gram0: np.ndarray,
    labels: np.ndarray,
    ids: list[str],
    full_rankings: list[list[int]],
    tolerance: float,
) -> dict[str, Any]:
    baseline_full = stable_rankings(gram0, ids, topk=len(ids) - 1, tolerance=tolerance)
    cell_top, cell_coeff = rank_coefficients(labels, full_rankings)
    base_top, base_coeff = rank_coefficients(labels, baseline_full)
    baseline_margins = np.asarray(
        [float(base_coeff[i] @ gram0[i, base_top[i]]) for i in range(len(labels))],
        dtype=np.float64,
    )
    return {
        "top": cell_top,
        "coeff": cell_coeff,
        "baseline_rankings": baseline_full,
        "baseline_margins": baseline_margins,
    }


def rank_halfspaces(ids: list[str], full_rankings: list[list[int]], tolerance: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, full in enumerate(full_rankings):
        top = full[:20]
        for r in range(19):
            a, b = top[r], top[r + 1]
            rows.append({"query": i, "a": a, "b": b, "kind": "internal", "rhs": canonical_rhs(ids, a, b, tolerance)})
        for outsider in full[20:]:
            a, b = top[19], outsider
            rows.append({"query": i, "a": a, "b": b, "kind": "boundary", "rhs": canonical_rhs(ids, a, b, tolerance)})
    return rows


def semantic_matrix(raw: Any, n: int) -> np.ndarray:
    sem = np.asarray(raw, dtype=np.float64)
    if sem.size == 0:
        return np.zeros((0, n * n), dtype=np.float64)
    if sem.ndim == 1:
        return sem.reshape(1, n * n)
    return sem.reshape((-1, n * n))


@dataclass
class LinearBlock:
    values_g: np.ndarray
    values_xi: np.ndarray
    offsets: np.ndarray
    names: list[str]
    scales: np.ndarray


@dataclass
class FeasibilityProblem:
    n: int
    rank: int
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
    eq_linear: LinearBlock
    ineq_linear: LinearBlock
    nonlinear_ineq_names: list[str]
    nonlinear_ineq_scales: np.ndarray


def _linear_block(values_g: list[np.ndarray], values_xi: list[np.ndarray], offsets: list[float],
                  names: list[str], scales: list[float], n: int) -> LinearBlock:
    if values_g:
        vg = np.stack(values_g).astype(np.float64)
    else:
        vg = np.zeros((0, n, n), dtype=np.float64)
    if values_xi:
        vx = np.stack(values_xi).astype(np.float64)
    else:
        vx = np.zeros((0, n), dtype=np.float64)
    return LinearBlock(
        values_g=vg,
        values_xi=vx,
        offsets=np.asarray(offsets, dtype=np.float64),
        names=list(names),
        scales=np.maximum(np.asarray(scales, dtype=np.float64), 1e-12),
    )


def build_problem(fixture: dict[str, Any], cfg: dict[str, Any],
                  full_rankings: list[list[int]], rank: int) -> FeasibilityProblem:
    solver = cfg["solver"]
    n = int(fixture["n"])
    ids = [str(x) for x in fixture["ids"]]
    labels = np.asarray(fixture["labels"], dtype=np.int64)
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    ell = np.asarray(fixture["ell"], dtype=np.float64)
    semantic = semantic_matrix(fixture["semantic"], n)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    md = margin_data(g0, labels, ids, full_rankings, solver["tie_tolerance"])
    deficits = np.maximum(ell - md["baseline_margins"], 0.0)
    caps = [solver["slack_budget_ratio"] * float(deficits[rows].sum()) for rows in groups]
    row_radius = float(solver["row_trust_scale"] * math.sqrt(n - 1))
    class_radius = float(solver["class_mean_trust_scale"] * math.sqrt(n))
    centroid = centroid_direction(labels)
    rank_rows = rank_halfspaces(ids, full_rankings, solver["tie_tolerance"])

    eq_g: list[np.ndarray] = []
    eq_x: list[np.ndarray] = []
    eq_b: list[float] = []
    eq_names: list[str] = []
    eq_scales: list[float] = []
    for i in range(n):
        a = np.zeros((n, n), dtype=np.float64)
        a[i, i] = 1.0
        eq_g.append(a)
        eq_x.append(np.zeros(n, dtype=np.float64))
        eq_b.append(-1.0)
        eq_names.append("unit_diagonal_{}".format(i))
        eq_scales.append(1.0)
    semantic_rows = semantic.reshape((-1, n, n))
    for k, a in enumerate(semantic_rows):
        eq_g.append(np.asarray(a, dtype=np.float64))
        eq_x.append(np.zeros(n, dtype=np.float64))
        eq_b.append(0.0)
        eq_names.append("semantic_zero_{}".format(k))
        eq_scales.append(max(1.0, float(np.linalg.norm(a))))

    ineq_g: list[np.ndarray] = []
    ineq_x: list[np.ndarray] = []
    ineq_b: list[float] = []
    ineq_names: list[str] = []
    ineq_scales: list[float] = []

    def add_ineq(name: str, a: np.ndarray | None, x: np.ndarray | None,
                 b: float, scale: float = 1.0) -> None:
        ineq_g.append(np.zeros((n, n), dtype=np.float64) if a is None else a)
        ineq_x.append(np.zeros(n, dtype=np.float64) if x is None else x)
        ineq_b.append(float(b))
        ineq_names.append(name)
        ineq_scales.append(max(float(scale), 1e-9))

    for i in range(n):
        for j in range(i + 1, n):
            a_upper = np.zeros((n, n), dtype=np.float64)
            a_upper[i, j] = -1.0
            add_ineq("offdiag_upper_{}_{}".format(i, j), a_upper, None, float(solver["offdiag_upper"]))
            a_lower = np.zeros((n, n), dtype=np.float64)
            a_lower[i, j] = 1.0
            add_ineq("offdiag_lower_{}_{}".format(i, j), a_lower, None, 1.0)
    for i in range(n):
        x = np.zeros(n, dtype=np.float64)
        x[i] = 1.0
        add_ineq("slack_nonnegative_{}".format(i), None, x, 0.0)
    for k, rows in enumerate(groups):
        x = np.zeros(n, dtype=np.float64)
        x[rows] = -1.0
        add_ineq("slack_budget_{}".format(k), None, x, caps[k], max(1.0, caps[k]))
    for i in range(n):
        a = np.zeros((n, n), dtype=np.float64)
        for r in range(20):
            a[i, int(md["top"][i, r])] += float(md["coeff"][i, r])
        x = np.zeros(n, dtype=np.float64)
        x[i] = 1.0
        add_ineq("vote_slack_{}".format(i), a, x, -float(ell[i]))
    for k, rows in enumerate(groups):
        a = np.zeros((n, n), dtype=np.float64)
        for i in rows:
            for r in range(20):
                a[int(i), int(md["top"][int(i), r])] += float(md["coeff"][int(i), r]) / len(rows)
        add_ineq("class_margin_{}".format(k), a, None, -float(md["baseline_margins"][rows].mean()))
    a_global = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for r in range(20):
            a_global[i, int(md["top"][i, r])] += float(md["coeff"][i, r]) / n
    add_ineq("global_margin", a_global, None, -float(md["baseline_margins"].mean()))
    add_ineq("centroid", centroid, None, -float(np.sum(centroid * g0)))
    for idx, row in enumerate(rank_rows):
        a = np.zeros((n, n), dtype=np.float64)
        a[int(row["query"]), int(row["a"])] += 1.0
        a[int(row["query"]), int(row["b"])] -= 1.0
        add_ineq("rank_{}_{}".format(row["kind"], idx), a, None, -float(row["rhs"]), 1.0)

    nonlinear_names = ["row_trust_{}".format(i) for i in range(n)]
    nonlinear_names.extend("class_mean_trust_{}".format(k) for k in range(len(groups)))
    nonlinear_scales = [max(1.0, row_radius)] * n + [max(1.0, class_radius)] * len(groups)

    return FeasibilityProblem(
        n=n,
        rank=int(rank),
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
        row_radius=row_radius,
        class_radius=class_radius,
        eq_linear=_linear_block(eq_g, eq_x, eq_b, eq_names, eq_scales, n),
        ineq_linear=_linear_block(ineq_g, ineq_x, ineq_b, ineq_names, ineq_scales, n),
        nonlinear_ineq_names=nonlinear_names,
        nonlinear_ineq_scales=np.asarray(nonlinear_scales, dtype=np.float64),
    )


def pack(F: np.ndarray, xi: np.ndarray) -> np.ndarray:
    return np.concatenate([F.reshape(-1), xi.reshape(-1)]).astype(np.float64)


def unpack(x: np.ndarray, n: int, rank: int) -> tuple[np.ndarray, np.ndarray]:
    F = np.asarray(x[:n * rank], dtype=np.float64).reshape(n, rank)
    xi = np.asarray(x[n * rank:n * rank + n], dtype=np.float64)
    return F, xi


def gram_from_factor(F: np.ndarray) -> np.ndarray:
    return np.asarray(F @ F.T, dtype=np.float64)


def linear_values(block: LinearBlock, G: np.ndarray, xi: np.ndarray) -> np.ndarray:
    if block.values_g.shape[0] == 0:
        return np.zeros(0, dtype=np.float64)
    return (
        np.einsum("mij,ij->m", block.values_g, G, optimize=True)
        + block.values_xi @ xi
        + block.offsets
    )


def linear_jacobian(block: LinearBlock, F: np.ndarray) -> np.ndarray:
    m = block.values_g.shape[0]
    n, rank = F.shape
    if m == 0:
        return np.zeros((0, n * rank + n), dtype=np.float64)
    sym = block.values_g + np.transpose(block.values_g, (0, 2, 1))
    jf = np.einsum("mij,jr->mir", sym, F, optimize=True).reshape(m, n * rank)
    return np.hstack([jf, block.values_xi])


def nonlinear_ineq_values_jac(problem: FeasibilityProblem, F: np.ndarray,
                              G: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    n, rank = F.shape
    values: list[float] = []
    grad_g_rows: list[np.ndarray] = []
    eps = 1e-15
    for i in range(n):
        diff = G[i].copy() - problem.g0[i]
        diff[i] = 0.0
        norm = float(np.linalg.norm(diff))
        values.append(problem.row_radius - norm)
        a = np.zeros((n, n), dtype=np.float64)
        if norm > eps:
            a[i, :] = -diff / norm
            a[i, i] = 0.0
        grad_g_rows.append(a)
    for rows in problem.groups:
        mean = (G[rows] - problem.g0[rows]).mean(axis=0)
        norm = float(np.linalg.norm(mean))
        values.append(problem.class_radius - norm)
        a = np.zeros((n, n), dtype=np.float64)
        if norm > eps:
            a[rows, :] = -(mean / norm) / len(rows)
        grad_g_rows.append(a)
    grad_g = np.stack(grad_g_rows).astype(np.float64)
    sym = grad_g + np.transpose(grad_g, (0, 2, 1))
    jf = np.einsum("mij,jr->mir", sym, F, optimize=True).reshape(len(values), n * rank)
    jx = np.zeros((len(values), n), dtype=np.float64)
    return np.asarray(values, dtype=np.float64), np.hstack([jf, jx])


def constraint_values_jac(problem: FeasibilityProblem, x: np.ndarray,
                          need_jac: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray | None]:
    F, xi = unpack(x, problem.n, problem.rank)
    G = gram_from_factor(F)
    eq = linear_values(problem.eq_linear, G, xi)
    ineq_linear = linear_values(problem.ineq_linear, G, xi)
    ineq_nl, j_nl = nonlinear_ineq_values_jac(problem, F, G)
    ineq = np.concatenate([ineq_linear, ineq_nl])
    if not need_jac:
        return eq, ineq, None, None
    j_eq = linear_jacobian(problem.eq_linear, F)
    j_ineq = np.vstack([linear_jacobian(problem.ineq_linear, F), j_nl])
    return eq, ineq, j_eq, j_ineq


def feasibility_objective_grad(problem: FeasibilityProblem, x: np.ndarray) -> tuple[float, np.ndarray]:
    eq, ineq, j_eq, j_ineq = constraint_values_jac(problem, x, need_jac=True)
    assert j_eq is not None and j_ineq is not None
    eq_scales = problem.eq_linear.scales
    ineq_scales = np.concatenate([problem.ineq_linear.scales, problem.nonlinear_ineq_scales])
    eq_weights = eq / (eq_scales ** 2)
    mask = ineq < 0.0
    obj = 0.5 * float(np.sum((eq / eq_scales) ** 2))
    grad = j_eq.T @ eq_weights
    if np.any(mask):
        obj += 0.5 * float(np.sum((ineq[mask] / ineq_scales[mask]) ** 2))
        grad += j_ineq[mask].T @ (ineq[mask] / (ineq_scales[mask] ** 2))
    return obj, np.asarray(grad, dtype=np.float64)


def eq_fun(problem: FeasibilityProblem, x: np.ndarray) -> np.ndarray:
    return constraint_values_jac(problem, x, need_jac=False)[0]


def ineq_fun(problem: FeasibilityProblem, x: np.ndarray) -> np.ndarray:
    return constraint_values_jac(problem, x, need_jac=False)[1]


def eq_jac(problem: FeasibilityProblem, x: np.ndarray) -> np.ndarray:
    return constraint_values_jac(problem, x, need_jac=True)[2]  # type: ignore[return-value]


def ineq_jac(problem: FeasibilityProblem, x: np.ndarray) -> np.ndarray:
    return constraint_values_jac(problem, x, need_jac=True)[3]  # type: ignore[return-value]


def factor_from_gram(G: np.ndarray, rank: int) -> np.ndarray:
    sym = 0.5 * (np.asarray(G, dtype=np.float64) + np.asarray(G, dtype=np.float64).T)
    vals, vecs = np.linalg.eigh(sym)
    order = np.argsort(vals)[::-1]
    vals = np.maximum(vals[order[:rank]], 0.0)
    vecs = vecs[:, order[:rank]]
    F = vecs * np.sqrt(vals)[None, :]
    if F.shape[1] < rank:
        F = np.pad(F, ((0, 0), (0, rank - F.shape[1])))
    return F.astype(np.float64)


def normalize_rows(F: np.ndarray) -> np.ndarray:
    out = np.asarray(F, dtype=np.float64).copy()
    norms = np.linalg.norm(out, axis=1)
    for i, norm in enumerate(norms):
        if norm <= 1e-15:
            out[i, 0] = 1.0
        else:
            out[i] /= norm
    return out


def spectral_affine_warmstart(g0: np.ndarray, semantic: np.ndarray,
                              offdiag_upper: float) -> np.ndarray:
    g = np.asarray(g0, dtype=np.float64).copy()
    n = g.shape[0]
    sem = semantic_matrix(semantic, n)
    if sem.shape[0] == 0:
        return g
    sem_gram = sem @ sem.T
    sem_pinv = np.linalg.pinv(sem_gram, rcond=1e-14)
    mask = ~np.eye(n, dtype=bool)
    for _ in range(30):
        g = 0.5 * (g + g.T)
        np.fill_diagonal(g, 1.0)
        flat = g.reshape(-1)
        flat = flat - sem.T @ (sem_pinv @ (sem @ flat))
        g = flat.reshape(n, n)
        g = 0.5 * (g + g.T)
        ev, vec = np.linalg.eigh(g)
        g = (vec * np.maximum(ev, 1e-10)) @ vec.T
        g = 0.5 * (g + g.T)
        np.fill_diagonal(g, 1.0)
        g[mask] = np.clip(g[mask], -1.0, offdiag_upper)
    return g


def dykstra_result_start(oriented: dict[str, Any], n: int) -> tuple[np.ndarray, np.ndarray] | None:
    values = oriented.get("result")
    if not isinstance(values, list) or len(values) < n * n + n:
        return None
    arr = np.asarray(values, dtype=np.float64)
    G = arr[:n * n].reshape(n, n)
    xi = arr[n * n:n * n + n]
    return 0.5 * (G + G.T), np.maximum(xi, 0.0)


def deterministic_starts(problem: FeasibilityProblem, oriented: dict[str, Any]) -> list[dict[str, Any]]:
    n, rank = problem.n, problem.rank
    starts: list[dict[str, Any]] = []
    zero_xi = np.zeros(n, dtype=np.float64)

    def add(name: str, G: np.ndarray, xi: np.ndarray) -> None:
        F = normalize_rows(factor_from_gram(G, rank))
        starts.append({"name": name, "x0": pack(F, np.asarray(xi, dtype=np.float64))})

    add("spectral_frozen_g0", problem.g0, zero_xi)
    dy = dykstra_result_start(oriented, n)
    if dy is not None:
        add("frozen_dykstra_result", dy[0], dy[1])
    affine = spectral_affine_warmstart(problem.g0, problem.semantic, problem.solver["offdiag_upper"])
    add("spectral_affine_semantic", affine, zero_xi)

    base = normalize_rows(factor_from_gram(problem.g0, rank))
    rng = np.random.default_rng(20260711 + 997 * rank)
    for k, scale in enumerate((1e-3, 3e-3, 1e-2)):
        F = normalize_rows(base + rng.normal(0.0, scale, size=base.shape))
        starts.append({"name": "deterministic_factor_jitter_{}".format(k), "x0": pack(F, zero_xi)})
    return starts


def residuals_np(G: np.ndarray, xi: np.ndarray, problem: FeasibilityProblem) -> dict[str, float]:
    n = problem.n
    solver = problem.solver
    margins = np.einsum(
        "ir,ir->i",
        problem.md["coeff"],
        G[np.arange(n)[:, None], problem.md["top"]],
        optimize=True,
    )
    off = G[~np.eye(n, dtype=bool)]
    rank_v = 0.0
    for row in problem.rank_rows:
        rank_v = max(
            rank_v,
            float(row["rhs"]) - float(G[int(row["query"]), int(row["a"])] - G[int(row["query"]), int(row["b"])]),
        )
    eig_min = float(np.linalg.eigvalsh(0.5 * (G + G.T)).min())
    return {
        "symmetry": float(np.max(np.abs(G - G.T))),
        "unit_diagonal": float(np.max(np.abs(np.diag(G) - 1.0))),
        "psd": max(0.0, -eig_min),
        "offdiag_box": max(0.0, float(off.max() - solver["offdiag_upper"]), float(-1.0 - off.min())),
        "row_trust": max(
            max(0.0, float(np.linalg.norm(np.delete(G[i] - problem.g0[i], i)) - problem.row_radius))
            for i in range(n)
        ),
        "class_mean_trust": max(
            max(0.0, float(np.linalg.norm((G[rows] - problem.g0[rows]).mean(axis=0)) - problem.class_radius))
            for rows in problem.groups
        ),
        "semantic": float(np.linalg.norm(problem.semantic @ G.reshape(-1))),
        "slack_nonnegative_budget": max(
            max(0.0, float(-xi[rows].min()), float(xi[rows].sum() - problem.caps[k]))
            for k, rows in enumerate(problem.groups)
        ),
        "vote_slack": max(0.0, float(np.max(problem.ell - margins - xi))),
        "class_margin": max(
            max(0.0, float(problem.md["baseline_margins"][rows].mean() - margins[rows].mean()))
            for rows in problem.groups
        ),
        "global_margin": max(0.0, float(problem.md["baseline_margins"].mean() - margins.mean())),
        "centroid": max(0.0, float(np.sum(problem.centroid * problem.g0) - np.sum(problem.centroid * G))),
        "rank_halfspaces": max(0.0, rank_v),
    }


def max_residual(residuals: dict[str, float]) -> float:
    return float(max(residuals.values(), default=float("inf")))


def realized_top20(G: np.ndarray, problem: FeasibilityProblem) -> list[list[int]]:
    full = stable_rankings(G, problem.ids, topk=problem.n - 1, tolerance=problem.solver["tie_tolerance"])
    return [row[:20] for row in full]


def projected_gradient_inf(x: np.ndarray, grad: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    pg = grad.copy()
    at_lower = x <= lower + 1e-10
    at_upper = x >= upper - 1e-10
    pg[at_lower & (grad > 0.0)] = 0.0
    pg[at_upper & (grad < 0.0)] = 0.0
    return float(np.max(np.abs(pg))) if pg.size else 0.0


def summarize_constraints(problem: FeasibilityProblem, x: np.ndarray) -> dict[str, Any]:
    eq, ineq, _, _ = constraint_values_jac(problem, x, need_jac=False)
    ineq_names = problem.ineq_linear.names + problem.nonlinear_ineq_names
    active = [
        {"name": name, "value": float(value)}
        for name, value in zip(ineq_names, ineq)
        if float(value) <= 1e-6
    ][:100]
    return {
        "eq_count": int(eq.size),
        "ineq_count": int(ineq.size),
        "max_eq_abs": float(np.max(np.abs(eq))) if eq.size else 0.0,
        "min_ineq": float(np.min(ineq)) if ineq.size else 0.0,
        "active_or_violated_first100": active,
    }


def solve_one_start(problem: FeasibilityProblem, start: dict[str, Any],
                    bounds: Bounds) -> dict[str, Any]:
    started = time.perf_counter()

    def fun(x: np.ndarray) -> float:
        return feasibility_objective_grad(problem, x)[0]

    def jac(x: np.ndarray) -> np.ndarray:
        return feasibility_objective_grad(problem, x)[1]

    error = None
    result = None
    try:
        result = minimize(
            fun,
            np.asarray(start["x0"], dtype=np.float64),
            method="L-BFGS-B",
            jac=jac,
            bounds=bounds,
            options={
                "maxiter": 700,
                "ftol": 1e-18,
                "gtol": 1e-11,
                "maxls": 50,
            },
        )
        _, ineq, _, _ = constraint_values_jac(problem, np.asarray(result.x, dtype=np.float64), need_jac=False)
        if float(np.min(ineq)) > -1e-3:
            cons = [
                {"type": "eq", "fun": lambda x: eq_fun(problem, x), "jac": lambda x: eq_jac(problem, x)},
                {"type": "ineq", "fun": lambda x: ineq_fun(problem, x), "jac": lambda x: ineq_jac(problem, x)},
            ]
            result = minimize(
                fun,
                np.asarray(result.x, dtype=np.float64),
                method="SLSQP",
                jac=jac,
                bounds=list(zip(bounds.lb, bounds.ub)),
                constraints=cons,
                options={"maxiter": 180, "ftol": 1e-12, "disp": False},
            )
    except Exception as exc:
        error = "{}: {}".format(type(exc).__name__, str(exc))
    elapsed = time.perf_counter() - started
    if result is None:
        return {
            "start_name": start["name"],
            "status": "error",
            "error": error,
            "elapsed_seconds": elapsed,
        }
    x = np.asarray(result.x, dtype=np.float64)
    F, xi = unpack(x, problem.n, problem.rank)
    G = gram_from_factor(F)
    residuals = residuals_np(G, xi, problem)
    obj, grad = feasibility_objective_grad(problem, x)
    target_top20 = [row[:20] for row in problem.full_rankings]
    got_top20 = realized_top20(G, problem)
    tol = float(problem.solver["dykstra_set_violation_tolerance"])
    np_candidate = bool(max_residual(residuals) <= tol and got_top20 == target_top20)
    stationarity = {
        "diagnostic_scope": "feasibility_penalty_bound_stationarity_only",
        "not_used_for_feasibility_acceptance": True,
        "feasibility_objective": float(obj),
        "gradient_inf_norm": float(np.max(np.abs(grad))) if grad.size else 0.0,
        "projected_gradient_inf_norm": projected_gradient_inf(x, grad, bounds.lb, bounds.ub),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }
    return {
        "start_name": start["name"],
        "status": "np_primal_feasible_candidate_pending_independent_replay" if np_candidate else "nonconverged_no_feasible_witness",
        "optimizer_status": "success" if bool(result.success) else "nonconverged",
        "optimizer_message": str(result.message),
        "elapsed_seconds": elapsed,
        "nit": None if getattr(result, "nit", None) is None else int(result.nit),
        "nfev": None if getattr(result, "nfev", None) is None else int(result.nfev),
        "njev": None if getattr(result, "njev", None) is None else int(result.njev),
        "feasibility_objective": float(obj),
        "residuals": residuals,
        "max_residual": max_residual(residuals),
        "constraint_summary": summarize_constraints(problem, x),
        "realized_top20_equal_cell": got_top20 == target_top20,
        "realized_top20_sha256": hobj(got_top20),
        "target_top20_sha256": hobj(target_top20),
        "stationarity_diagnostics": stationarity,
        "witness": {
            "parameterization": "factor_psd_G_equals_F_FT",
            "rank": problem.rank,
            "F": F.tolist(),
            "g": G.tolist(),
            "xi": xi.tolist(),
            "g_sha256": hobj(G.tolist()),
            "xi_sha256": hobj(xi.tolist()),
            "factor_sha256": hobj(F.tolist()),
            "witness_sha256": hobj({"g": G.tolist(), "xi": xi.tolist()}),
        },
    }


def solve_cell(problem: FeasibilityProblem, oriented: dict[str, Any],
               cell_index: int, checkpoint_path: Path) -> list[dict[str, Any]]:
    xi_upper = max(1.0, max(problem.caps, default=0.0) + 1.0)
    lower = np.concatenate([
        np.full(problem.n * problem.rank, -2.0, dtype=np.float64),
        np.zeros(problem.n, dtype=np.float64),
    ])
    upper = np.concatenate([
        np.full(problem.n * problem.rank, 2.0, dtype=np.float64),
        np.full(problem.n, xi_upper, dtype=np.float64),
    ])
    bounds = Bounds(lower, upper)
    out: list[dict[str, Any]] = []
    for start in deterministic_starts(problem, oriented):
        print(cjson({
            "event": "start_analytic_multistart",
            "cell_index": cell_index,
            "rank": problem.rank,
            "start_name": start["name"],
            "job_id": os.environ.get("SLURM_JOB_ID", "no_slurm_job"),
        }), flush=True)
        result = solve_one_start(problem, start, bounds)
        result["cell_index"] = int(cell_index)
        result["rank"] = int(problem.rank)
        out.append(result)
        append_jsonl(checkpoint_path, {
            "job_id": os.environ.get("SLURM_JOB_ID", "no_slurm_job"),
            "cell_index": int(cell_index),
            "rank": int(problem.rank),
            "start_name": start["name"],
            "status": result.get("status"),
            "optimizer_status": result.get("optimizer_status"),
            "max_residual": result.get("max_residual"),
            "realized_top20_equal_cell": result.get("realized_top20_equal_cell"),
            "witness_sha256": (
                result.get("witness", {}).get("witness_sha256")
                if isinstance(result.get("witness"), dict) else None
            ),
            "elapsed_seconds": result.get("elapsed_seconds"),
        })
        print(cjson({
            "event": "finish_analytic_multistart",
            "cell_index": cell_index,
            "rank": problem.rank,
            "start_name": start["name"],
            "status": result.get("status"),
            "max_residual": result.get("max_residual"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "job_id": os.environ.get("SLURM_JOB_ID", "no_slurm_job"),
        }), flush=True)
        if result.get("status") == "np_primal_feasible_candidate_pending_independent_replay":
            break
    return out


def supervision_contract(cfg: dict[str, Any]) -> dict[str, Any]:
    supervision = cfg.get("supervision", {})
    counters = cfg.get("counters", {})
    return {
        "only_gold_supervision": supervision.get("only_gold_supervision"),
        "segment_gold_exists": bool(supervision.get("segment_gold_exists")),
        "segment_gold_used": bool(supervision.get("segment_gold_used")),
        "mllm_call_count": int(counters.get("mllm_call_count", 0)),
        "ocr_call_count": int(counters.get("ocr_call_count", 0)),
        "teacher_cache_read_count": int(counters.get("teacher_cache_read_count", 0)),
        "teacher_cache_write_count": int(counters.get("teacher_cache_write_count", 0)),
        "outer_held_label_read_count": int(counters.get("outer_held_label_read_count", 0)),
        "outer_held_content_read_count": int(counters.get("outer_held_content_read_count", 0)),
        "val_content_read_count": int(counters.get("val_content_read_count", 0)),
        "test_content_read_count": int(counters.get("test_content_read_count", 0)),
        "val_test_teacher_artifact_count": int(counters.get("val_test_teacher_artifact_count", 0)),
    }


def contract_ok(contract: dict[str, Any]) -> bool:
    return (
        contract.get("only_gold_supervision") == "parent_video_binary_label"
        and contract.get("segment_gold_exists") is False
        and contract.get("segment_gold_used") is False
        and all(int(contract.get(k, -1)) == 0 for k in (
            "mllm_call_count",
            "ocr_call_count",
            "teacher_cache_read_count",
            "teacher_cache_write_count",
            "outer_held_label_read_count",
            "outer_held_content_read_count",
            "val_content_read_count",
            "test_content_read_count",
            "val_test_teacher_artifact_count",
        ))
    )


def load_fixture() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    cfg = read_json(V5_CONFIG)
    oriented = None
    for row in read_jsonl(V5_DYKSTRA):
        if row.get("case") == "feasible_oriented_boundary":
            oriented = row
            break
    if oriented is None:
        raise RuntimeError("missing feasible_oriented_boundary row")
    return cfg, oriented, oriented["fixture"]


def build_cells(fixture: dict[str, Any], cfg: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    n = int(fixture["n"])
    ids = [str(x) for x in fixture["ids"]]
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    tol = float(cfg["solver"]["tie_tolerance"])
    base_full = stable_rankings(g0, ids, topk=n - 1, tolerance=tol)
    system = boundary_orientation_system(g0, ids, topk=20, tolerance=tol, compatible_limit=34)
    cells: list[dict[str, Any]] = []
    for index, assignment in enumerate(system["compatible_assignments"]):
        full = orientation_cell_from_assignment(base_full, system["descriptors"], assignment, ids)
        cells.append({
            "cell_index": int(index),
            "assignment": assignment,
            "full_rankings": full,
            "final_top20_rankings": [row[:20] for row in full],
            "cell_sha256": hobj(full),
            "top20_sha256": hobj([row[:20] for row in full]),
        })
    return system, cells


def rank_schedule(g0: np.ndarray) -> list[int]:
    vals = np.linalg.eigvalsh(0.5 * (g0 + g0.T))
    rank0 = int(np.sum(vals > 1e-8))
    n = g0.shape[0]
    return sorted(set([max(2, min(n, rank0)), n]))


def self_check() -> dict[str, Any]:
    cfg, oriented, fixture = load_fixture()
    system, cells = build_cells(fixture, cfg)
    if not cells:
        raise RuntimeError("no compatible cells to validate")
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    rank = rank_schedule(g0)[0]
    problem = build_problem(fixture, cfg, cells[0]["full_rankings"], rank)
    start = deterministic_starts(problem, oriented)[-1]["x0"]
    eq, ineq, j_eq, j_ineq = constraint_values_jac(problem, start, need_jac=True)
    obj, grad = feasibility_objective_grad(problem, start)
    rng = np.random.default_rng(20260712)
    direction = rng.normal(size=start.shape)
    direction /= np.linalg.norm(direction)
    eps = 1e-6
    eq_p, ineq_p, _, _ = constraint_values_jac(problem, start + eps * direction, need_jac=False)
    eq_m, ineq_m, _, _ = constraint_values_jac(problem, start - eps * direction, need_jac=False)
    fd_eq = (eq_p - eq_m) / (2.0 * eps)
    fd_ineq = (ineq_p - ineq_m) / (2.0 * eps)
    an_eq = j_eq @ direction  # type: ignore[operator]
    an_ineq = j_ineq @ direction  # type: ignore[operator]
    obj_p = feasibility_objective_grad(problem, start + eps * direction)[0]
    obj_m = feasibility_objective_grad(problem, start - eps * direction)[0]
    fd_obj = (obj_p - obj_m) / (2.0 * eps)
    an_obj = float(grad @ direction)
    contract = supervision_contract(cfg)
    checks = {
        "contract_ok": contract_ok(contract),
        "cell_count": len(cells),
        "orientation_rank": int(system["rank"]),
        "rank": int(rank),
        "variable_shape": [int(start.size)],
        "factor_shape": [problem.n, problem.rank],
        "eq_shape": [int(eq.shape[0]), int(start.size)],
        "ineq_shape": [int(ineq.shape[0]), int(start.size)],
        "eq_jac_shape": list(j_eq.shape),  # type: ignore[union-attr]
        "ineq_jac_shape": list(j_ineq.shape),  # type: ignore[union-attr]
        "max_directional_eq_jac_error": float(np.max(np.abs(fd_eq - an_eq))) if eq.size else 0.0,
        "max_directional_ineq_jac_error": float(np.max(np.abs(fd_ineq - an_ineq))) if ineq.size else 0.0,
        "directional_objective_grad_error": abs(float(fd_obj) - an_obj),
        "initial_feasibility_objective": float(obj),
        "supervision_contract": contract,
    }
    checks["ok"] = bool(
        checks["contract_ok"]
        and checks["max_directional_eq_jac_error"] <= 1e-5
        and checks["max_directional_ineq_jac_error"] <= 1e-5
        and checks["directional_objective_grad_error"] <= 1e-5
    )
    return checks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        out = self_check()
        print(cjson(out))
        return 0 if out.get("ok") else 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    started = time.perf_counter()
    cfg, oriented, fixture = load_fixture()
    contract = supervision_contract(cfg)
    if not contract_ok(contract):
        raise RuntimeError("supervision/counter contract violation")
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    system, cells = build_cells(fixture, cfg)
    checkpoint_path = OUT_DIR / "analytic_feasibility_checkpoint_{}.jsonl".format(job_id)
    raw_results: list[dict[str, Any]] = []
    ranks = rank_schedule(g0)
    stop_after_first_candidate = True
    selected = None
    for cell in cells:
        if selected is not None and stop_after_first_candidate:
            break
        for rank in ranks:
            problem = build_problem(fixture, cfg, cell["full_rankings"], rank)
            cell_results = solve_cell(problem, oriented, int(cell["cell_index"]), checkpoint_path)
            for result in cell_results:
                result["assignment"] = cell["assignment"]
                result["cell_sha256"] = cell["cell_sha256"]
                result["top20_sha256"] = cell["top20_sha256"]
                raw_results.append(result)
                if result.get("status") == "np_primal_feasible_candidate_pending_independent_replay":
                    selected = result
                    break
            if selected is not None and stop_after_first_candidate:
                break

    if selected is None and raw_results:
        selected = min(raw_results, key=lambda row: float(row.get("max_residual", float("inf"))))
    elapsed = time.perf_counter() - started
    determination = {
        "status_label": (
            "np_primal_feasible_candidate_pending_independent_replay"
            if selected and selected.get("status") == "np_primal_feasible_candidate_pending_independent_replay"
            else "nonconverged_no_feasible_witness"
        ),
        "selected_cell_index": None if selected is None else int(selected["cell_index"]),
        "selected_rank": None if selected is None else int(selected["rank"]),
        "selected_start_name": None if selected is None else selected.get("start_name"),
        "selected_max_residual": None if selected is None else selected.get("max_residual"),
        "selected_witness_sha256": (
            selected.get("witness", {}).get("witness_sha256")
            if isinstance(selected, dict) and isinstance(selected.get("witness"), dict) else None
        ),
        "acceptance_requires_independent_replay": True,
        "nonconvergence_is_not_infeasibility": True,
        "evidence_strength": "prospective_numerical_candidate_not_certificate",
    }
    source_hashes = {
        "analytic_feasibility_witness.py": hfile(V6 / "runtime" / "analytic_feasibility_witness.py"),
        "analytic_witness_replay.py": hfile(V6 / "runtime" / "analytic_witness_replay.py"),
        "validate_analytic_v6.py": hfile(V6 / "runtime" / "validate_analytic_v6.py"),
        "analytic_feasibility_witness.sbatch": hfile(V6 / "runtime" / "analytic_feasibility_witness.sbatch"),
        "validate_analytic_v6.sbatch": hfile(V6 / "runtime" / "validate_analytic_v6.sbatch"),
        "v5_config": hfile(V5_CONFIG),
        "v5_freeze": hfile(V5_FREEZE),
        "v5_dykstra_jsonl": hfile(V5_DYKSTRA),
        "v5_log": hfile(V5_LOG),
    }
    out = {
        "schema_version": 1,
        "task": "lb_scgp_v6_analytic_feasibility_witness",
        "slurm_job_id": job_id,
        "python": sys.version,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "elapsed_seconds": elapsed,
        "parameterization": {
            "variables": "factor F in R^{n x r} plus slack xi",
            "gram": "G = F F^T",
            "psd_handling": "exact_by_construction",
            "unit_diagonal": "analytic equality residual diag(G)-1",
            "rank_schedule": ranks,
        },
        "objective_contract": {
            "feasibility_first_objective": "0.5*||scaled_equalities||^2 + 0.5*||negative_scaled_inequalities||^2",
            "distinct_from_frozen_scientific_projection_objective": True,
            "scientific_projection_objective_not_optimized": True,
        },
        "supervision_boundary": contract,
        "frozen_fixture": fixture,
        "orientation_system": system,
        "compatible_cells": cells,
        "raw_results": raw_results,
        "determination": determination,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
        "source_hashes": source_hashes,
    }
    out["payload_sha256"] = hobj({k: v for k, v in out.items() if k != "payload_sha256"})
    out_path = OUT_DIR / "analytic_feasibility_witness_{}.json".format(job_id)
    write_json_exclusive(out_path, out)
    print(cjson({
        "status": "OK",
        "path": str(out_path),
        "determination": determination,
        "payload_sha256": out["payload_sha256"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
