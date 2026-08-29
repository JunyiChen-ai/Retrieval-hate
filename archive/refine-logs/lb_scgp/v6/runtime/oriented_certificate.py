#!/usr/bin/env python3
"""Independent LB-SCGP v6 oriented-boundary numerical certificate.

The script intentionally does not import the v5 producer or verifier.  It
reads the frozen v5 synthetic JSONL evidence, reconstructs the
feasible_oriented_boundary fixture, enumerates final-top20 boundary cells, and
solves the resulting constrained projection with the numerical packages
available in the SLURM HateVideo environment.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linprog


ROOT = Path("/data/jehc223/RGCL")
OUT_DIR = ROOT / "refine-logs" / "lb_scgp" / "v6" / "results"
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


def write_json_exclusive(path: Path, obj: Any) -> None:
    payload = (cjson(obj) + "\n").encode()
    with path.open("xb") as handle:
        handle.write(payload)


def append_jsonl(path: Path, obj: Any) -> None:
    with path.open("ab") as handle:
        handle.write((cjson(obj) + "\n").encode())


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
        local = tolerance_order([float(gram[i, j]) for j in candidates],
                                [ids[j] for j in candidates], tolerance)
        out.append([candidates[k] for k in local[:topk]])
    return out


def canonical_rhs(ids: list[str], a: int, b: int, tolerance: float) -> float:
    return -float(tolerance) if str(ids[a]) < str(ids[b]) else float(np.nextafter(float(tolerance), math.inf))


def boundary_orientation_system(gram: np.ndarray, ids: list[str], topk: int,
                                tolerance: float, compatible_limit: int) -> dict[str, Any]:
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


def orientation_cell_from_assignment(base_rankings: list[list[int]], descriptors: list[list[str]],
                                     assignment: list[int], ids: list[str]) -> list[list[int]]:
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


def _rank_coefficients(labels: np.ndarray, rankings: list[list[int]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(labels)
    signs = 2 * labels.astype(np.int64) - 1
    coeff = np.zeros((n, 20), dtype=np.float64)
    top = np.asarray([row[:20] for row in rankings], dtype=np.int64)
    for i in range(n):
        for r, j in enumerate(top[i], 1):
            coeff[i, r - 1] = signs[i] * (21 - r) * signs[j] / 210.0
    return top, coeff


def margin_data(gram0: np.ndarray, labels: np.ndarray, ids: list[str],
                full_rankings: list[list[int]], tolerance: float) -> dict[str, Any]:
    """Candidate cell directions with frozen canonical baseline margins.

    v5 constructs candidate-cell margin directions from the oriented cell, but
    keeps class/global baselines, slack budgets, and ell from the canonical
    fixture top-20 margins.  The v6 numerical certificate must test that same
    object rather than silently moving the baseline with each candidate cell.
    """
    baseline_full = stable_rankings(gram0, ids, topk=len(ids) - 1, tolerance=tolerance)
    cell_top, cell_coeff = _rank_coefficients(labels, full_rankings)
    baseline_top, baseline_coeff = _rank_coefficients(labels, baseline_full)
    cell_margins_at_g0 = np.asarray(
        [float(cell_coeff[i] @ gram0[i, cell_top[i]]) for i in range(len(labels))],
        dtype=np.float64,
    )
    baseline_margins = np.asarray(
        [float(baseline_coeff[i] @ gram0[i, baseline_top[i]]) for i in range(len(labels))],
        dtype=np.float64,
    )
    return {
        "top": cell_top,
        "coeff": cell_coeff,
        "cell_margins_at_g0": cell_margins_at_g0,
        "baseline_rankings": baseline_full,
        "baseline_top": baseline_top,
        "baseline_coeff": baseline_coeff,
        "baseline_margins": baseline_margins,
    }


def rank_halfspaces(ids: list[str], full_rankings: list[list[int]], tolerance: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, full in enumerate(full_rankings):
        top = full[:20]
        for r in range(19):
            a, b = top[r], top[r + 1]
            out.append({"query": i, "a": a, "b": b, "kind": "internal",
                        "rhs": canonical_rhs(ids, a, b, tolerance)})
        for outsider in full[20:]:
            a, b = top[19], outsider
            out.append({"query": i, "a": a, "b": b, "kind": "boundary",
                        "rhs": canonical_rhs(ids, a, b, tolerance)})
    return out


def residuals_np(g: np.ndarray, xi: np.ndarray, fixture: dict[str, Any],
                 full_rankings: list[list[int]], cfg: dict[str, Any]) -> dict[str, float]:
    solver = cfg["solver"]
    n = int(fixture["n"])
    labels = np.asarray(fixture["labels"], dtype=np.int64)
    ids = [str(x) for x in fixture["ids"]]
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    ell = np.asarray(fixture["ell"], dtype=np.float64)
    semantic = np.asarray(fixture["semantic"], dtype=np.float64)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    md = margin_data(g0, labels, ids, full_rankings, solver["tie_tolerance"])
    margins = np.asarray([float(md["coeff"][i] @ g[i, md["top"][i]]) for i in range(n)])
    deficits = np.maximum(ell - md["baseline_margins"], 0.0)
    caps = [solver["slack_budget_ratio"] * float(deficits[rows].sum()) for rows in groups]
    off = g[~np.eye(n, dtype=bool)]
    cd = centroid_direction(labels)
    rhs_rank = rank_halfspaces(ids, full_rankings, solver["tie_tolerance"])
    rank_v = 0.0
    for row in rhs_rank:
        rank_v = max(rank_v, float(row["rhs"]) - float(g[row["query"], row["a"]] - g[row["query"], row["b"]]))
    return {
        "symmetry": float(np.max(np.abs(g - g.T))),
        "unit_diagonal": float(np.max(np.abs(np.diag(g) - 1.0))),
        "psd": max(0.0, -float(np.linalg.eigvalsh(0.5 * (g + g.T)).min())),
        "offdiag_box": max(0.0, float(off.max() - solver["offdiag_upper"]), float(-1.0 - off.min())),
        "row_trust": max(
            max(0.0, float(np.linalg.norm(np.delete(g[i] - g0[i], i)) -
                           solver["row_trust_scale"] * math.sqrt(n - 1)))
            for i in range(n)
        ),
        "class_mean_trust": max(
            max(0.0, float(np.linalg.norm((g[rows] - g0[rows]).mean(axis=0)) -
                           solver["class_mean_trust_scale"] * math.sqrt(n)))
            for rows in groups
        ),
        "semantic": float(np.linalg.norm(semantic @ g.reshape(-1))),
        "slack_nonnegative_budget": max(
            max(0.0, float(-xi[rows].min()), float(xi[rows].sum() - caps[k]))
            for k, rows in enumerate(groups)
        ),
        "vote_slack": max(0.0, float(np.max(ell - margins - xi))),
        "class_margin": max(
            max(0.0, float(md["baseline_margins"][rows].mean() - margins[rows].mean()))
            for rows in groups
        ),
        "global_margin": max(0.0, float(md["baseline_margins"].mean() - margins.mean())),
        "centroid": max(0.0, float(np.sum(cd * g0) - np.sum(cd * g))),
        "rank_halfspaces": max(0.0, rank_v),
    }


def residuals_mpmath(g: np.ndarray, xi: np.ndarray, fixture: dict[str, Any],
                     full_rankings: list[list[int]], cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        import mpmath as mp
    except Exception as exc:
        return {"available": False, "error": "{}: {}".format(type(exc).__name__, str(exc))}
    mp.mp.dps = 80
    np_res = residuals_np(g, xi, fixture, full_rankings, cfg)
    labels = np.asarray(fixture["labels"], dtype=np.int64)
    ids = [str(x) for x in fixture["ids"]]
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    ell = np.asarray(fixture["ell"], dtype=np.float64)
    semantic = np.asarray(fixture["semantic"], dtype=np.float64)
    solver = cfg["solver"]
    n = int(fixture["n"])
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    md = margin_data(g0, labels, ids, full_rankings, solver["tie_tolerance"])

    def mvec(values: np.ndarray) -> list[mp.mpf]:
        return [mp.mpf(repr(float(v))) for v in values.reshape(-1)]

    g_mp = [[mp.mpf(repr(float(g[i, j]))) for j in range(n)] for i in range(n)]
    xi_mp = [mp.mpf(repr(float(v))) for v in xi]
    sem = mvec(semantic)
    flat_g = mvec(g)
    semantic_value = abs(mp.fsum([sem[k] * flat_g[k] for k in range(len(flat_g))]))
    rank_v = mp.mpf("0")
    for row in rank_halfspaces(ids, full_rankings, solver["tie_tolerance"]):
        lhs = g_mp[row["query"]][row["a"]] - g_mp[row["query"]][row["b"]]
        rank_v = max(rank_v, mp.mpf(repr(float(row["rhs"]))) - lhs)
    margins = []
    for i in range(n):
        margins.append(mp.fsum([
            mp.mpf(repr(float(md["coeff"][i, r]))) * g_mp[i][int(md["top"][i, r])]
            for r in range(20)
        ]))
    vote = max([mp.mpf(repr(float(ell[i]))) - margins[i] - xi_mp[i] for i in range(n)])
    class_margin = mp.mpf("0")
    for rows in groups:
        base = mp.fsum([mp.mpf(repr(float(md["baseline_margins"][i]))) for i in rows]) / len(rows)
        got = mp.fsum([margins[i] for i in rows]) / len(rows)
        class_margin = max(class_margin, base - got)
    global_margin = (
        mp.fsum([mp.mpf(repr(float(v))) for v in md["baseline_margins"]]) / n -
        mp.fsum(margins) / n
    )
    try:
        sym = mp.matrix([[mp.mpf("0.5") * (g_mp[i][j] + g_mp[j][i]) for j in range(n)] for i in range(n)])
        eigvals = mp.eigsy(sym, eigvals_only=True)
        psd = max(mp.mpf("0"), -min(eigvals))
        psd_note = "mpmath_eigsy_80dps"
    except Exception as exc:
        psd = mp.mpf(repr(float(np_res["psd"])))
        psd_note = "numpy_fallback_after_{}:{}".format(type(exc).__name__, str(exc))
    selected = {
        "semantic": semantic_value,
        "vote_slack": max(mp.mpf("0"), vote),
        "class_margin": max(mp.mpf("0"), class_margin),
        "global_margin": max(mp.mpf("0"), global_margin),
        "rank_halfspaces": max(mp.mpf("0"), rank_v),
        "psd": psd,
    }
    return {
        "available": True,
        "dps": int(mp.mp.dps),
        "psd_method": psd_note,
        "selected_residuals": {k: float(v) for k, v in selected.items()},
        "max_selected_residual": float(max(selected.values())),
    }


def variable_layout(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def pack_variables(g: np.ndarray, xi: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    values = [float(g[i, j]) for i, j in pairs]
    values.extend([float(x) for x in xi])
    return np.asarray(values, dtype=np.float64)


def unpack_variables(x: np.ndarray, n: int, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, np.ndarray]:
    g = np.eye(n, dtype=np.float64)
    for value, (i, j) in zip(x[:len(pairs)], pairs):
        g[i, j] = float(value)
        g[j, i] = float(value)
    xi = np.asarray(x[len(pairs):len(pairs) + n], dtype=np.float64)
    return g, xi


def spectral_affine_warmstart(g0: np.ndarray, semantic: np.ndarray, offdiag_upper: float) -> np.ndarray:
    g = np.asarray(g0, dtype=np.float64).copy()
    n = g.shape[0]
    sem = np.asarray(semantic, dtype=np.float64)
    sem_gram = sem @ sem.T
    sem_pinv = np.linalg.pinv(sem_gram, rcond=1e-14)
    mask = ~np.eye(n, dtype=bool)
    for _ in range(40):
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


def scipy_constraint_values(g: np.ndarray, xi: np.ndarray, fixture: dict[str, Any],
                            full_rankings: list[list[int]], cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    solver = cfg["solver"]
    n = int(fixture["n"])
    labels = np.asarray(fixture["labels"], dtype=np.int64)
    ids = [str(x) for x in fixture["ids"]]
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    ell = np.asarray(fixture["ell"], dtype=np.float64)
    semantic = np.asarray(fixture["semantic"], dtype=np.float64)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    md = margin_data(g0, labels, ids, full_rankings, solver["tie_tolerance"])
    margins = np.asarray([float(md["coeff"][i] @ g[i, md["top"][i]]) for i in range(n)])
    deficits = np.maximum(ell - md["baseline_margins"], 0.0)
    caps = [solver["slack_budget_ratio"] * float(deficits[rows].sum()) for rows in groups]
    vals: list[float] = []
    names: list[str] = []

    def add(name: str, value: float) -> None:
        names.append(name)
        vals.append(float(value))

    add("psd_min_eigenvalue", float(np.linalg.eigvalsh(0.5 * (g + g.T)).min()))
    row_radius = solver["row_trust_scale"] * math.sqrt(n - 1)
    for i in range(n):
        add("row_trust_{}".format(i), row_radius - float(np.linalg.norm(np.delete(g[i] - g0[i], i))))
    class_radius = solver["class_mean_trust_scale"] * math.sqrt(n)
    for k, rows in enumerate(groups):
        add("class_mean_trust_{}".format(k),
            class_radius - float(np.linalg.norm((g[rows] - g0[rows]).mean(axis=0))))
    for k, rows in enumerate(groups):
        add("slack_budget_{}".format(k), caps[k] - float(xi[rows].sum()))
    for i in range(n):
        add("vote_slack_{}".format(i), float(margins[i] + xi[i] - ell[i]))
    for k, rows in enumerate(groups):
        add("class_margin_{}".format(k), float(margins[rows].mean() - md["baseline_margins"][rows].mean()))
    add("global_margin", float(margins.mean() - md["baseline_margins"].mean()))
    centroid = centroid_direction(labels)
    add("centroid", float(np.sum(centroid * g) - np.sum(centroid * g0)))
    for idx, row in enumerate(rank_halfspaces(ids, full_rankings, solver["tie_tolerance"])):
        add("rank_{}_{}".format(row["kind"], idx),
            float(g[row["query"], row["a"]] - g[row["query"], row["b"]] - row["rhs"]))
    eq = np.asarray([float(v) for v in (semantic @ g.reshape(-1)).reshape(-1)], dtype=np.float64)
    return np.asarray(vals, dtype=np.float64), eq, names


def solve_cell_scipy(fixture: dict[str, Any], cfg: dict[str, Any], full_rankings: list[list[int]],
                     solver_path: dict[str, Any]) -> dict[str, Any]:
    from scipy.optimize import BFGS, Bounds, NonlinearConstraint, minimize

    solver = cfg["solver"]
    n = int(fixture["n"])
    ids = [str(x) for x in fixture["ids"]]
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    semantic = np.asarray(fixture["semantic"], dtype=np.float64)
    pairs = variable_layout(n)
    zero_xi = np.zeros(n, dtype=np.float64)
    if solver_path.get("start") == "spectral_affine":
        start_g = spectral_affine_warmstart(g0, semantic, solver["offdiag_upper"])
    else:
        start_g = g0.copy()
    x0 = pack_variables(start_g, zero_xi, pairs)
    lower = [-1.0] * len(pairs) + [0.0] * n
    upper = [solver["offdiag_upper"]] * len(pairs) + [1.0] * n
    bounds = Bounds(np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64))

    def objective(x: np.ndarray) -> float:
        g, xi = unpack_variables(x, n, pairs)
        return float(0.5 * np.sum((g - g0) ** 2) + 0.5 * np.sum(xi ** 2))

    def jacobian(x: np.ndarray) -> np.ndarray:
        g, xi = unpack_variables(x, n, pairs)
        grad = [2.0 * float(g[i, j] - g0[i, j]) for i, j in pairs]
        grad.extend([float(v) for v in xi])
        return np.asarray(grad, dtype=np.float64)

    def ineq_fun(x: np.ndarray) -> np.ndarray:
        g, xi = unpack_variables(x, n, pairs)
        return scipy_constraint_values(g, xi, fixture, full_rankings, cfg)[0]

    def eq_fun(x: np.ndarray) -> np.ndarray:
        g, xi = unpack_variables(x, n, pairs)
        return scipy_constraint_values(g, xi, fixture, full_rankings, cfg)[1]

    method = solver_path["method"]
    started = time.perf_counter()
    error = None
    result = None
    try:
        if method == "SLSQP":
            cons = [
                {"type": "ineq", "fun": ineq_fun},
                {"type": "eq", "fun": eq_fun},
            ]
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                jac=jacobian,
                bounds=list(zip(lower, upper)),
                constraints=cons,
                options={
                    "ftol": float(solver_path.get("ftol", 1e-12)),
                    "maxiter": int(solver_path.get("maxiter", 3000)),
                    "disp": False,
                },
            )
        elif method == "trust-constr":
            constraints = [
                NonlinearConstraint(ineq_fun, 0.0, np.inf, finite_diff_rel_step=1e-6),
                NonlinearConstraint(eq_fun, 0.0, 0.0, finite_diff_rel_step=1e-6),
            ]
            result = minimize(
                objective,
                x0,
                method="trust-constr",
                jac=jacobian,
                hess=BFGS(),
                bounds=bounds,
                constraints=constraints,
                options={
                    "maxiter": int(solver_path.get("maxiter", 300)),
                    "gtol": float(solver_path.get("gtol", 1e-8)),
                    "xtol": float(solver_path.get("xtol", 1e-10)),
                    "verbose": 0,
                },
            )
        else:
            raise RuntimeError("unsupported scipy method {}".format(method))
    except Exception as exc:
        error = "{}: {}".format(type(exc).__name__, str(exc))
    elapsed = time.perf_counter() - started
    if result is None:
        return {
            "solver_path": solver_path["name"],
            "solver": "scipy",
            "method": method,
            "status": "error",
            "error": error,
            "elapsed_seconds": elapsed,
        }
    g_val, xi_val = unpack_variables(np.asarray(result.x, dtype=np.float64), n, pairs)
    g_val = 0.5 * (g_val + g_val.T)
    res = residuals_np(g_val, xi_val, fixture, full_rankings, cfg)
    mp_res = residuals_mpmath(g_val, xi_val, fixture, full_rankings, cfg)
    realized_full = stable_rankings(g_val, ids, topk=n - 1, tolerance=solver["tie_tolerance"])
    realized_top20 = [row[:20] for row in realized_full]
    target_top20 = [row[:20] for row in full_rankings]
    ineq_values, eq_values, ineq_names = scipy_constraint_values(g_val, xi_val, fixture, full_rankings, cfg)
    active = [
        {"name": name, "value": float(value)}
        for name, value in zip(ineq_names, ineq_values)
        if float(value) <= 1e-6
    ][:80]
    multipliers = getattr(result, "multipliers", None)
    trust_multipliers = getattr(result, "v", None)
    dual_payload = serialize_dual(multipliers if multipliers is not None else trust_multipliers)
    dual_numbers = flatten_numeric(multipliers if multipliers is not None else trust_multipliers)
    witness = {
        "g": g_val.tolist(),
        "xi": xi_val.tolist(),
        "objective": float(objective(pack_variables(g_val, xi_val, pairs))),
        "g_sha256": hobj(g_val.tolist()),
        "xi_sha256": hobj(xi_val.tolist()),
        "witness_sha256": hobj({"g": g_val.tolist(), "xi": xi_val.tolist()}),
    }
    return {
        "solver_path": solver_path["name"],
        "solver": "scipy",
        "method": method,
        "start": solver_path.get("start"),
        "status": "success" if bool(result.success) else "nonconverged",
        "message": str(result.message),
        "success": bool(result.success),
        "fun": None if result.fun is None else float(result.fun),
        "elapsed_seconds": elapsed,
        "solver_stats": {
            "nit": None if getattr(result, "nit", None) is None else int(result.nit),
            "nfev": None if getattr(result, "nfev", None) is None else int(result.nfev),
            "njev": None if getattr(result, "njev", None) is None else int(result.njev),
            "optimality": None if getattr(result, "optimality", None) is None else float(result.optimality),
            "constr_violation": None if getattr(result, "constr_violation", None) is None else float(result.constr_violation),
        },
        "constraint_value_summary": {
            "min_ineq": float(np.min(ineq_values)),
            "max_eq_abs": float(np.max(np.abs(eq_values))) if eq_values.size else 0.0,
            "active_or_near_active_first80": active,
        },
        "residuals": res,
        "max_residual": float(max(res.values())),
        "mpmath_residuals": mp_res,
        "realized_top20_equal_cell": realized_top20 == target_top20,
        "realized_full_equal_cell": realized_full == full_rankings,
        "realized_top20_sha256": hobj(realized_top20),
        "target_top20_sha256": hobj(target_top20),
        "full_rankings_sha256": hobj(full_rankings),
        "witness": witness,
        "dual_summary": {
            "source": "scipy_optimize_result_multipliers" if multipliers is not None else (
                "scipy_trust_constr_v" if trust_multipliers is not None else "not_available"
            ),
            "max_abs_dual_or_multiplier": float(max([abs(x) for x in dual_numbers], default=0.0)),
            "dual_or_multiplier_l2_norm": float(np.linalg.norm(np.asarray(dual_numbers, dtype=np.float64))) if dual_numbers else 0.0,
            "dual_payload_sha256": hobj(dual_payload),
            "kkt_fields_available": [name for name in ("optimality", "constr_violation") if getattr(result, name, None) is not None],
        },
        "dual_rows": [{"name": "scipy_multipliers_or_trust_constr_v", "dual": dual_payload}],
    }


def serialize_dual(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, np.ndarray):
        if value.size > 2048:
            return {
                "shape": list(value.shape),
                "sha256": hobj(np.asarray(value, dtype=np.float64).tolist()),
                "max_abs": float(np.max(np.abs(value))),
                "norm": float(np.linalg.norm(value.reshape(-1))),
            }
        return np.asarray(value, dtype=np.float64).tolist()
    if isinstance(value, (list, tuple)):
        return [serialize_dual(v) for v in value]
    try:
        arr = np.asarray(value, dtype=np.float64)
        return serialize_dual(arr)
    except Exception:
        return str(value)


def flatten_numeric(value: Any) -> list[float]:
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        return [float(x) for x in value.reshape(-1)]
    if isinstance(value, (int, float, np.floating)):
        return [float(value)]
    if isinstance(value, (list, tuple)):
        out: list[float] = []
        for item in value:
            out.extend(flatten_numeric(item))
        return out
    return []


def solve_cell_cvxpy(fixture: dict[str, Any], cfg: dict[str, Any], full_rankings: list[list[int]],
                     solver_path: dict[str, Any]) -> dict[str, Any]:
    import cvxpy as cp

    solver = cfg["solver"]
    n = int(fixture["n"])
    ids = [str(x) for x in fixture["ids"]]
    labels = np.asarray(fixture["labels"], dtype=np.int64)
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    initial = g0.copy()
    ell = np.asarray(fixture["ell"], dtype=np.float64)
    semantic = np.asarray(fixture["semantic"], dtype=np.float64)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    md = margin_data(g0, labels, ids, full_rankings, solver["tie_tolerance"])
    deficits = np.maximum(ell - md["baseline_margins"], 0.0)
    caps = [solver["slack_budget_ratio"] * float(deficits[rows].sum()) for rows in groups]
    rank_rows = rank_halfspaces(ids, full_rankings, solver["tie_tolerance"])
    centroid = centroid_direction(labels)
    G = cp.Variable((n, n), symmetric=True, name="G")
    xi = cp.Variable(n, name="xi")
    constraints = []
    names: list[str] = []

    def add(name: str, constraint: Any) -> None:
        constraints.append(constraint)
        names.append(name)

    add("psd", G >> 0)
    add("unit_diagonal", cp.diag(G) == 1.0)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            add("offdiag_upper_{}_{}".format(i, j), G[i, j] <= solver["offdiag_upper"])
            add("offdiag_lower_{}_{}".format(i, j), G[i, j] >= -1.0)
    row_radius = solver["row_trust_scale"] * math.sqrt(n - 1)
    for i in range(n):
        cols = [j for j in range(n) if j != i]
        add("row_trust_{}".format(i), cp.norm(G[i, cols] - g0[i, cols], 2) <= row_radius)
    class_radius = solver["class_mean_trust_scale"] * math.sqrt(n)
    for k, rows in enumerate(groups):
        add("class_mean_trust_{}".format(k),
            cp.norm(cp.sum(G[rows, :] - g0[rows, :], axis=0) / len(rows), 2) <= class_radius)
    add("semantic_zero", semantic @ cp.reshape(G, (n * n,), order="C") == 0.0)
    add("slack_nonnegative", xi >= 0.0)
    for k, rows in enumerate(groups):
        add("slack_budget_{}".format(k), cp.sum(xi[rows]) <= caps[k])
    margin_exprs = []
    for i in range(n):
        expr = 0
        for r in range(20):
            expr += float(md["coeff"][i, r]) * G[i, int(md["top"][i, r])]
        margin_exprs.append(expr)
        add("vote_slack_{}".format(i), expr + xi[i] >= ell[i])
    for k, rows in enumerate(groups):
        add("class_margin_{}".format(k),
            cp.sum([margin_exprs[int(i)] for i in rows]) / len(rows) >= float(md["baseline_margins"][rows].mean()))
    add("global_margin", cp.sum(margin_exprs) / n >= float(md["baseline_margins"].mean()))
    add("centroid", cp.sum(cp.multiply(centroid, G)) >= float(np.sum(centroid * g0)))
    for idx, row in enumerate(rank_rows):
        add("rank_{}_{}".format(row["kind"], idx),
            G[row["query"], row["a"]] - G[row["query"], row["b"]] >= float(row["rhs"]))
    objective = cp.Minimize(0.5 * cp.sum_squares(G - initial) + 0.5 * cp.sum_squares(xi))
    problem = cp.Problem(objective, constraints)
    started = time.perf_counter()
    error = None
    try:
        problem.solve(**solver_path["kwargs"])
    except Exception as exc:
        error = "{}: {}".format(type(exc).__name__, str(exc))
    elapsed = time.perf_counter() - started
    if error is not None or G.value is None or xi.value is None:
        return {
            "solver_path": solver_path["name"],
            "solver": solver_path["kwargs"].get("solver"),
            "status": getattr(problem, "status", None),
            "error": error,
            "elapsed_seconds": elapsed,
        }
    g_val = np.asarray(G.value, dtype=np.float64)
    xi_val = np.asarray(xi.value, dtype=np.float64)
    g_val = 0.5 * (g_val + g_val.T)
    res = residuals_np(g_val, xi_val, fixture, full_rankings, cfg)
    mp_res = residuals_mpmath(g_val, xi_val, fixture, full_rankings, cfg)
    realized_full = stable_rankings(g_val, ids, topk=n - 1, tolerance=solver["tie_tolerance"])
    realized_top20 = [row[:20] for row in realized_full]
    target_top20 = [row[:20] for row in full_rankings]
    extra = getattr(problem.solver_stats, "extra_stats", None)
    dual_rows = []
    dual_numbers: list[float] = []
    for name, constraint in zip(names, constraints):
        payload = serialize_dual(constraint.dual_value)
        nums = flatten_numeric(constraint.dual_value)
        dual_numbers.extend(nums)
        dual_rows.append({"name": name, "dual": payload})
    dual_summary = {
        "constraint_count": len(constraints),
        "nonnull_dual_count": sum(1 for _, c in zip(names, constraints) if c.dual_value is not None),
        "max_abs_dual": float(max([abs(x) for x in dual_numbers], default=0.0)),
        "dual_l2_norm": float(np.linalg.norm(np.asarray(dual_numbers, dtype=np.float64))) if dual_numbers else 0.0,
        "dual_payload_sha256": hobj(dual_rows),
    }
    witness = {
        "g": g_val.tolist(),
        "xi": xi_val.tolist(),
        "objective": float(0.5 * np.sum((g_val - initial) ** 2) + 0.5 * np.sum(xi_val ** 2)),
        "g_sha256": hobj(g_val.tolist()),
        "xi_sha256": hobj(xi_val.tolist()),
        "witness_sha256": hobj({"g": g_val.tolist(), "xi": xi_val.tolist()}),
    }
    return {
        "solver_path": solver_path["name"],
        "solver": solver_path["kwargs"].get("solver"),
        "status": problem.status,
        "cvxpy_value": None if problem.value is None else float(problem.value),
        "elapsed_seconds": elapsed,
        "solver_stats": {
            "solver_name": getattr(problem.solver_stats, "solver_name", None),
            "solve_time": getattr(problem.solver_stats, "solve_time", None),
            "setup_time": getattr(problem.solver_stats, "setup_time", None),
            "num_iters": getattr(problem.solver_stats, "num_iters", None),
            "extra_stats": extra,
        },
        "residuals": res,
        "max_residual": float(max(res.values())),
        "mpmath_residuals": mp_res,
        "realized_top20_equal_cell": realized_top20 == target_top20,
        "realized_full_equal_cell": realized_full == full_rankings,
        "realized_top20_sha256": hobj(realized_top20),
        "target_top20_sha256": hobj(target_top20),
        "full_rankings_sha256": hobj(full_rankings),
        "witness": witness,
        "dual_summary": dual_summary,
        "dual_rows": dual_rows,
    }


def solve_cell(fixture: dict[str, Any], cfg: dict[str, Any], full_rankings: list[list[int]],
               solver_path: dict[str, Any]) -> dict[str, Any]:
    if solver_path.get("kind") == "cvxpy":
        return solve_cell_cvxpy(fixture, cfg, full_rankings, solver_path)
    if solver_path.get("kind") == "scipy":
        return solve_cell_scipy(fixture, cfg, full_rankings, solver_path)
    return {
        "solver_path": solver_path.get("name"),
        "status": "error",
        "error": "unsupported solver path kind {}".format(solver_path.get("kind")),
    }


def solver_paths() -> dict[str, Any]:
    installed: list[str] = []
    cvxpy_error = None
    paths: list[dict[str, Any]] = [
        {
            "kind": "scipy",
            "name": "scipy_slsqp_frozen_initial",
            "method": "SLSQP",
            "start": "frozen_initial",
            "ftol": 1e-12,
            "maxiter": 3000,
        },
        {
            "kind": "scipy",
            "name": "scipy_slsqp_spectral_affine",
            "method": "SLSQP",
            "start": "spectral_affine",
            "ftol": 1e-12,
            "maxiter": 3000,
        },
    ]
    if os.environ.get("LBSCGP_V6_SLSQP_ONLY") != "1":
        paths.append({
            "kind": "scipy",
            "name": "scipy_trust_constr_spectral_affine",
            "method": "trust-constr",
            "start": "spectral_affine",
            "gtol": 1e-8,
            "xtol": 1e-10,
            "maxiter": 250,
        })
    try:
        import cvxpy as cp

        installed = list(cp.installed_solvers())
    except Exception as exc:
        cvxpy_error = "{}: {}".format(type(exc).__name__, str(exc))
        return {"installed": installed, "cvxpy_error": cvxpy_error, "paths": paths}
    if "CLARABEL" in installed:
        paths.append({
            "kind": "cvxpy",
            "name": "cvxpy_clarabel_tight",
            "kwargs": {
                "solver": "CLARABEL",
                "verbose": False,
                "max_iter": 5000,
                "tol_gap_abs": 1e-9,
                "tol_gap_rel": 1e-9,
                "tol_feas": 1e-9,
            },
        })
    if "SCS" in installed:
        paths.append({
            "kind": "cvxpy",
            "name": "cvxpy_scs_direct_tight",
            "kwargs": {
                "solver": "SCS",
                "verbose": False,
                "eps_abs": 1e-8,
                "eps_rel": 1e-8,
                "max_iters": 50000,
                "normalize": True,
                "acceleration_lookback": 20,
                "use_indirect": False,
            },
        })
        paths.append({
            "kind": "cvxpy",
            "name": "cvxpy_scs_indirect_tight",
            "kwargs": {
                "solver": "SCS",
                "verbose": False,
                "eps_abs": 1e-8,
                "eps_rel": 1e-8,
                "max_iters": 50000,
                "normalize": True,
                "acceleration_lookback": 20,
                "use_indirect": True,
            },
        })
    if "CVXOPT" in installed:
        paths.append({"kind": "cvxpy", "name": "cvxpy_cvxopt_default", "kwargs": {"solver": "CVXOPT", "verbose": False}})
    return {"installed": installed, "cvxpy_error": cvxpy_error, "paths": paths}


def cross_backend_checks(results: list[dict[str, Any]], fixture: dict[str, Any]) -> dict[str, Any]:
    feasible = [
        row for row in results
        if row.get("witness") and row.get("max_residual", float("inf")) <= 1e-6
        and row.get("realized_top20_equal_cell")
    ]
    checks = []
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    initial_x = np.concatenate([g0.reshape(-1), np.zeros(g0.shape[0])])
    for i, a in enumerate(feasible):
        wa = np.concatenate([
            np.asarray(a["witness"]["g"], dtype=np.float64).reshape(-1),
            np.asarray(a["witness"]["xi"], dtype=np.float64),
        ])
        for b in feasible[i + 1:]:
            wb = np.concatenate([
                np.asarray(b["witness"]["g"], dtype=np.float64).reshape(-1),
                np.asarray(b["witness"]["xi"], dtype=np.float64),
            ])
            normal = wa - initial_x
            vi_ab = float(normal @ (wb - wa))
            checks.append({
                "a": a["solver_path"],
                "b": b["solver_path"],
                "same_cell": a.get("cell_sha256") == b.get("cell_sha256"),
                "witness_l2_distance": float(np.linalg.norm(wa - wb)),
                "objective_abs_diff": abs(float(a["witness"]["objective"]) - float(b["witness"]["objective"])),
                "projection_vi_using_b_as_feasible_probe": vi_ab,
            })
    return {"pair_count": len(checks), "checks": checks}


def build_case_matrix(job_id: str) -> dict[str, Any]:
    def row(case: str, expected: str, execution: str, acceptance: str) -> dict[str, str]:
        return {
            "case": case,
            "expected_outcome": expected,
            "v6_execution_status": execution,
            "acceptance_condition": acceptance,
            "supervision_boundary": "parent_video_binary_label_only; segment_gold_exists=false; segment_gold_used=false; no MLLM/OCR/teacher/held/val/test",
        }

    rows = [
        row("top20_stable_ranks_21_to_N_shuffled", "LOCAL unchanged", "registered_only",
            "Top-20 hashes and exact votes unchanged while full_outsider_order hash changes."),
        row("zero_orientation_stable_true", "LOCAL", "registered_only",
            "No boundary orientations; feasibility <=1e-6, relative change <=1e-7, final_top20 stable."),
        row("zero_orientation_stable_false", "BOUNDED/nonlocal", "registered_only",
            "Scalar convergence may pass, but final_top20 hash differs and verifier rejects LOCAL."),
        row("one_boundary_known_LOCAL", "LOCAL", "executed_oriented_certificate",
            "All compatible top-20 adjacent cells enumerated; at least two paths agree on feasible residuals and selected objective."),
        row("one_boundary_known_BOUNDED_nonlocal", "BOUNDED/nonlocal", "registered_only",
            "Complete enumeration shows no cell with feasible top-20-realizing residual <=1e-6, or proof-grade infeasibility basis."),
        row("just_below_1e-6", "LOCAL threshold pass", "registered_only",
            "Independent max residual <=1e-6 and relative condition <=1e-7."),
        row("just_above_1e-6", "BOUNDED threshold fail", "registered_only",
            "Independent max residual >1e-6 even if relative condition passes."),
        row("relative_change_without_feasibility", "BOUNDED threshold fail", "registered_only",
            "Relative change <=1e-7 with feasibility >1e-6 is rejected."),
        row("canonical_ID_tie_below_1e-7", "canonical tie", "registered_only",
            "ID ordering controls tied group and rank RHS accepts equality within tolerance."),
        row("canonical_ID_tie_at_1e-7", "canonical tie boundary", "registered_only",
            "Offset exactly 1e-7 is treated as tied under stable ordering."),
        row("canonical_ID_tie_above_1e-7", "strict similarity order", "registered_only",
            "Offset greater than 1e-7 breaks tie and top-20/rank hashes follow score order."),
        row("duplicate_ID_unresolved_tie_map", "REMOVE", "registered_only",
            "Duplicate canonical IDs force unresolved_tie_map rejection."),
        row("orientation_over_budget", "REMOVE", "registered_only",
            "Independent orientation rank exceeds max_independent_orientations=8."),
        row("pivot_over_budget", "REMOVE", "registered_only",
            "Cell sequence causes pivots >32 and rollback hash parity is required."),
        row("PSD_unit_diagonal_box_trust_stress", "registered stress", "registered_only",
            "PSD, unit diagonal, offdiag box, row trust, and class-mean trust residuals all audited separately."),
        row("no_segment_manifest_negative", "REMOVE/FAIL", "registered_only",
            "Any segment_gold_exists=true or segment_gold_used=true rejects manifest."),
        row("zero_counter_manifest_negative", "REMOVE/FAIL", "registered_only",
            "Any nonzero MLLM/OCR/teacher/held/val/test counter rejects manifest."),
    ]
    obj = {
        "schema_version": 1,
        "job_id": job_id,
        "matrix_status": "machine_registered_v6",
        "rows": rows,
    }
    obj["matrix_sha256"] = hobj(rows)
    return obj


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    cfg = read_json(V5_CONFIG)
    oriented = None
    for row in read_jsonl(V5_DYKSTRA):
        if row.get("case") == "feasible_oriented_boundary":
            oriented = row
            break
    if oriented is None:
        raise RuntimeError("missing feasible_oriented_boundary row")
    fixture = oriented["fixture"]
    n = int(fixture["n"])
    ids = [str(x) for x in fixture["ids"]]
    labels = np.asarray(fixture["labels"], dtype=np.int64)
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    tol = float(cfg["solver"]["tie_tolerance"])
    base_full = stable_rankings(g0, ids, topk=n - 1, tolerance=tol)
    baseline_md = margin_data(g0, labels, ids, base_full, tol)
    system = boundary_orientation_system(g0, ids, topk=20, tolerance=tol, compatible_limit=34)
    cells = []
    for index, assignment in enumerate(system["compatible_assignments"]):
        full = orientation_cell_from_assignment(base_full, system["descriptors"], assignment, ids)
        halfspaces = rank_halfspaces(ids, full, tol)
        cells.append({
            "cell_index": index,
            "assignment": assignment,
            "full_rankings": full,
            "final_top20_rankings": [row[:20] for row in full],
            "full_outsider_order_for_enumeration": full,
            "rank_halfspace_count": len(halfspaces),
            "internal_halfspace_count": sum(1 for row in halfspaces if row["kind"] == "internal"),
            "boundary_halfspace_count": sum(1 for row in halfspaces if row["kind"] == "boundary"),
            "cell_sha256": hobj(full),
            "top20_sha256": hobj([row[:20] for row in full]),
        })
    controller = {
        "schema_version": 1,
        "object": "canonical_top20_controller_certificate",
        "invariant_object": "final top-20 exact-vote-safe full-bank Gram projection",
        "topk": 20,
        "tie_tolerance": tol,
        "canonical_id_tie_semantics": "Within tolerance, tied candidates are sorted by canonical string ID; rank RHS is -tol when id_a < id_b and nextafter(tol,+inf) otherwise.",
        "reference_final_top20_rankings": [row[:20] for row in base_full],
        "reference_full_outsider_order_for_enumeration": base_full,
        "reference_baseline_margins": baseline_md["baseline_margins"].tolist(),
        "self_exclusion_verified": all(i not in row for i, row in enumerate(base_full)),
        "orientation_system": system,
        "compatible_cells": cells,
        "controller_hashes": {
            "reference_top20_sha256": hobj([row[:20] for row in base_full]),
            "reference_full_order_sha256": hobj(base_full),
            "reference_baseline_margins_sha256": hobj(baseline_md["baseline_margins"].tolist()),
            "compatible_cells_sha256": hobj(cells),
        },
        "independent_verifier_obligations": [
            "recompute stable top-20 with canonical-ID ties",
            "verify all 19 internal inequalities per query",
            "verify every 20th-vs-self-excluded-outsider inequality",
            "rebuild compatible adjacent orientations from the full outsider order",
            "verify selected objective and witness residuals without importing producer logic",
            "verify no segment gold, MLLM, OCR, teacher, held, validation, or test access",
        ],
    }
    paths_info = solver_paths()
    if not paths_info["paths"]:
        raise RuntimeError("no numerical solver path available")
    raw_results = []
    checkpoint_path = OUT_DIR / "oriented_certificate_checkpoint_{}.jsonl".format(job_id)
    for cell in cells:
        for path in paths_info["paths"]:
            print(cjson({
                "event": "start_cell_solver_path",
                "cell_index": cell["cell_index"],
                "solver_path": path["name"],
                "job_id": job_id,
            }), flush=True)
            result = solve_cell(fixture, cfg, cell["full_rankings"], path)
            result["cell_index"] = cell["cell_index"]
            result["assignment"] = cell["assignment"]
            result["cell_sha256"] = cell["cell_sha256"]
            result["top20_sha256"] = cell["top20_sha256"]
            raw_results.append(result)
            append_jsonl(checkpoint_path, {
                "job_id": job_id,
                "cell_index": cell["cell_index"],
                "solver_path": path["name"],
                "status": result.get("status"),
                "success": result.get("success"),
                "max_residual": result.get("max_residual"),
                "mpmath_max_selected_residual": (
                    result.get("mpmath_residuals", {}).get("max_selected_residual")
                    if isinstance(result.get("mpmath_residuals"), dict) else None
                ),
                "witness_sha256": (
                    result.get("witness", {}).get("witness_sha256")
                    if isinstance(result.get("witness"), dict) else None
                ),
                "elapsed_seconds": result.get("elapsed_seconds"),
            })
            print(cjson({
                "event": "finish_cell_solver_path",
                "cell_index": cell["cell_index"],
                "solver_path": path["name"],
                "status": result.get("status"),
                "max_residual": result.get("max_residual"),
                "elapsed_seconds": result.get("elapsed_seconds"),
                "job_id": job_id,
            }), flush=True)
    feasible = [
        row for row in raw_results
        if row.get("witness") and row.get("max_residual", float("inf")) <= cfg["solver"]["dykstra_set_violation_tolerance"]
        and row.get("realized_top20_equal_cell")
    ]
    feasible_by_path: dict[str, set[int]] = {}
    for row in feasible:
        feasible_by_path.setdefault(row["solver_path"], set()).add(int(row["cell_index"]))
    complete_paths = sorted([name for name, seen in feasible_by_path.items() if len(seen) == len(cells)])
    selected = None
    if feasible:
        selected = min(feasible, key=lambda row: (float(row["witness"]["objective"]), row["solver_path"], int(row["cell_index"])))
    determination = {
        "local_feasible_cell_exists_numerically": bool(feasible),
        "complete_local_controller_certificate_numerically": bool(complete_paths),
        "complete_paths": complete_paths,
        "cell_count": len(cells),
        "feasible_witness_count": len(feasible),
        "selected_solver_path": None if selected is None else selected["solver_path"],
        "selected_cell_index": None if selected is None else int(selected["cell_index"]),
        "selected_objective": None if selected is None else float(selected["witness"]["objective"]),
        "selected_max_residual": None if selected is None else float(selected["max_residual"]),
        "evidence_strength": "numerical_not_proof_grade",
        "nonproof_reason": "Floating-point conic solves and high-precision residual replay do not constitute an exact mathematical infeasibility/existence proof.",
    }
    case_matrix = build_case_matrix(job_id)
    source_hashes = {
        "config": hfile(V5_CONFIG),
        "freeze": hfile(V5_FREEZE),
        "dykstra_jsonl": hfile(V5_DYKSTRA),
        "v5_log": hfile(V5_LOG),
        "oriented_certificate.py": hfile(ROOT / "refine-logs/lb_scgp/v6/runtime/oriented_certificate.py"),
        "oriented_certificate.sbatch": hfile(ROOT / "refine-logs/lb_scgp/v6/runtime/oriented_certificate.sbatch"),
    }
    certificate = {
        "schema_version": 1,
        "task": "lb_scgp_v6_oriented_boundary_certificate",
        "slurm_job_id": job_id,
        "python": sys.version,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "supervision_boundary": {
            "only_gold_supervision": "parent_video_binary_label",
            "segment_gold_exists": False,
            "segment_gold_used": False,
            "mllm_call_count": 0,
            "ocr_call_count": 0,
            "teacher_cache_read_count": 0,
            "teacher_cache_write_count": 0,
            "outer_held_label_read_count": 0,
            "outer_held_content_read_count": 0,
            "val_content_read_count": 0,
            "test_content_read_count": 0,
            "val_test_teacher_artifact_count": 0,
        },
        "frozen_fixture_case": "feasible_oriented_boundary",
        "frozen_fixture": fixture,
        "v5_row_status": {
            "status": oriented.get("status"),
            "cycles": oriented.get("cycles"),
            "max_set_violation": oriented.get("max_set_violation"),
            "relative_iterate_change": oriented.get("relative_iterate_change"),
            "search_reason": oriented.get("search_reason"),
            "independent_orientations": oriented.get("independent_orientations"),
        },
        "solver_paths": paths_info,
        "source_hashes": source_hashes,
        "controller": controller,
        "case_matrix": case_matrix,
        "raw_results": raw_results,
        "cross_backend_checks": cross_backend_checks(raw_results, fixture),
        "determination": determination,
        "checkpoint_path": str(checkpoint_path.relative_to(ROOT)),
    }
    controller["payload_sha256"] = hobj(controller)
    case_matrix["payload_sha256"] = hobj(case_matrix)
    certificate["payload_sha256"] = hobj({k: v for k, v in certificate.items() if k != "payload_sha256"})
    cert_path = OUT_DIR / "oriented_certificate_{}.json".format(job_id)
    controller_path = OUT_DIR / "canonical_top20_controller_{}.json".format(job_id)
    matrix_path = OUT_DIR / "case_matrix_{}.json".format(job_id)
    write_json_exclusive(cert_path, certificate)
    write_json_exclusive(controller_path, controller)
    write_json_exclusive(matrix_path, case_matrix)
    print(cjson({
        "status": "OK",
        "certificate": str(cert_path),
        "controller": str(controller_path),
        "case_matrix": str(matrix_path),
        "determination": determination,
        "payload_sha256": certificate["payload_sha256"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
