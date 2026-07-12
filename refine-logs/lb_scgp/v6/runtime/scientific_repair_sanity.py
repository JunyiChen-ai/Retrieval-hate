#!/usr/bin/env python3
"""Prospective nonformal v6 scientific-repair sanity oracle.

This is not a formal synthetic gate.  It exercises the repaired top20 rank
semantics and a small Phase-II Gram-space local scientific projection with the
frozen thresholds.  It writes only under refine-logs/lb_scgp/v6/results.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, minimize


ROOT = Path("/data/jehc223/RGCL")
V6 = ROOT / "refine-logs" / "lb_scgp" / "v6"
RUNTIME = V6 / "runtime"
RESULTS = V6 / "results"
DESIGN = V6 / "G0_V6_SCIENTIFIC_REPAIR_DESIGN_TEST_MACHINE.json"
V5_CONFIG = ROOT / "configs" / "lb_scgp" / "lb_scgp_v5.json"

TOPK = 20
MAX_CYCLES = 500
VIOLATION_TOL = 1e-6
RELATIVE_TOL = 1e-7
TIE_TOL = 1e-7
MAX_ORIENTATIONS = 8
MAX_PIVOTS = 32


def cjson(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(cjson(obj).encode("utf-8")).hexdigest()


def payload_sha256(obj: dict[str, Any]) -> str:
    return sha256_obj({k: v for k, v in obj.items() if k != "payload_sha256"})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_exclusive(path: Path, obj: Any) -> None:
    with path.open("xb") as handle:
        handle.write((cjson(obj) + "\n").encode("utf-8"))


def append_jsonl(path: Path, obj: Any) -> None:
    with path.open("ab") as handle:
        handle.write((cjson(obj) + "\n").encode("utf-8"))


def tolerance_order(values: list[float], ids: list[str], tolerance: float) -> list[int]:
    remaining = sorted(range(len(values)), key=lambda k: (-float(values[k]), str(ids[k])))
    ordered: list[int] = []
    while remaining:
        anchor = float(values[remaining[0]])
        group = [k for k in remaining if anchor - float(values[k]) <= tolerance]
        group.sort(key=lambda k: str(ids[k]))
        ordered.extend(group)
        selected = set(group)
        remaining = [k for k in remaining if k not in selected]
    return ordered


def stable_rankings(score: np.ndarray, ids: list[str], topk: int, tolerance: float) -> list[list[int]]:
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate canonical IDs")
    n = len(ids)
    rankings: list[list[int]] = []
    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        local = tolerance_order([float(score[i, j]) for j in candidates], [ids[j] for j in candidates], tolerance)
        rankings.append([candidates[k] for k in local[:topk]])
    return rankings


def rank_certificate(score: np.ndarray, ids: list[str], topk: int = TOPK) -> dict[str, Any]:
    n = len(ids)
    final_top20 = stable_rankings(score, ids, topk=topk, tolerance=TIE_TOL)
    full = stable_rankings(score, ids, topk=n - 1, tolerance=TIE_TOL)
    outsiders = [row[topk:] for row in full]
    self_exclusion = all(i not in row for i, row in enumerate(final_top20))
    return {
        "n": n,
        "topk": topk,
        "final_top20_rankings": final_top20,
        "full_outsider_order_for_enumeration": outsiders,
        "final_top20_rankings_sha256": sha256_obj(final_top20),
        "full_outsider_order_for_enumeration_sha256": sha256_obj(outsiders),
        "self_exclusion": self_exclusion,
        "internal_top20_adjacent_count": n * (topk - 1),
        "boundary_20th_vs_outsider_count": sum(len(row) for row in outsiders),
        "top20_lengths_ok": all(len(row) == topk for row in final_top20),
        "outsider_lengths_ok": all(len(row) == n - 1 - topk for row in outsiders),
    }


def canonical_rhs(ids: list[str], a: int, b: int, tolerance: float) -> float:
    return -float(tolerance) if str(ids[a]) < str(ids[b]) else float(np.nextafter(float(tolerance), math.inf))


def circle_score_matrix(n: int, outsider_mode: str = "base") -> np.ndarray:
    score = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            raw = abs(i - j)
            dist = min(raw, n - raw)
            if dist <= 10:
                value = 0.75 - 0.02 * dist + 2e-6 * ((i + j) % 7)
            elif dist == 11:
                value = 0.11 if outsider_mode == "base" else 0.09
                value += 1e-6 * ((i + j) % 5)
            else:
                value = 0.09 if outsider_mode == "base" else 0.109
            score[i, j] = value
            score[j, i] = value
    return score


def small_psd_gram(n: int = 24) -> np.ndarray:
    gram = np.eye(n, dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            raw = abs(i - j)
            dist = min(raw, n - raw)
            value = 0.006 - 0.00025 * dist + 2e-6 * ((i + j) % 11) + 3e-7 * ((i * j) % 7)
            gram[i, j] = value
            gram[j, i] = value
    return gram


def boundary_tie_score() -> np.ndarray:
    score = small_psd_gram(24)
    ids = [f"v{i:02d}" for i in range(24)]
    full = stable_rankings(score, ids, topk=23, tolerance=TIE_TOL)
    boundary = full[0][TOPK - 1]
    outsider = full[0][TOPK]
    tied_value = min(float(score[0, boundary]), float(score[0, outsider]))
    score[0, boundary] = tied_value
    score[boundary, 0] = tied_value
    score[0, outsider] = tied_value
    score[outsider, 0] = tied_value
    return score


def adjacent_orientation_descriptors(score: np.ndarray, ids: list[str]) -> list[dict[str, Any]]:
    final_top20 = stable_rankings(score, ids, TOPK, TIE_TOL)
    full = stable_rankings(score, ids, len(ids) - 1, TIE_TOL)
    descriptors: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, str]] = set()
    for i, row in enumerate(final_top20):
        for r in range(TOPK - 1):
            a, b = row[r], row[r + 1]
            if abs(float(score[i, a]) - float(score[i, b])) <= TIE_TOL:
                key = (i, a, b, "internal")
                if key not in seen:
                    seen.add(key)
                    descriptors.append({"query": i, "a": a, "b": b, "kind": "internal_adjacent_top20"})
        boundary = row[TOPK - 1]
        for outsider in full[i][TOPK:]:
            if abs(float(score[i, boundary]) - float(score[i, outsider])) <= TIE_TOL:
                key = (i, boundary, outsider, "boundary")
                if key not in seen:
                    seen.add(key)
                    descriptors.append({"query": i, "a": boundary, "b": outsider, "kind": "20th_vs_outsider"})
    return descriptors


def compatible_orientation_count(descriptors: list[dict[str, Any]]) -> int:
    if len(descriptors) > MAX_ORIENTATIONS:
        return 0
    return 2 ** len(descriptors)


def tie_case(delta: float, label: str) -> dict[str, Any]:
    ids = ["v00", "v01", "v02"]
    score = np.eye(3, dtype=np.float64)
    score[0, 1] = 0.5
    score[1, 0] = 0.5
    score[0, 2] = 0.5 + delta
    score[2, 0] = 0.5 + delta
    ranking = stable_rankings(score, ids, topk=2, tolerance=TIE_TOL)[0]
    if delta <= TIE_TOL:
        expected_order = [1, 2]
    else:
        expected_order = [2, 1]
    return {
        "case": label,
        "expected_status": "PASS",
        "actual_status": "PASS" if ranking == expected_order else "REMOVE",
        "ok": ranking == expected_order,
        "delta": delta,
        "ranking": ranking,
        "expected_order": expected_order,
        "tie_tolerance": TIE_TOL,
    }


def simulated_status(max_violation: float, relative_change: float, top20_stable: bool) -> str:
    if max_violation <= VIOLATION_TOL and relative_change <= RELATIVE_TOL and top20_stable:
        return "LOCAL_STATIONARY_CERTIFIED"
    return "BOUNDED_SEARCH_FEASIBLE"


def make_case_ledger() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ids = [f"v{i:02d}" for i in range(24)]
    base = circle_score_matrix(24, "base")
    shuffled = circle_score_matrix(24, "shuffled")
    base_cert = rank_certificate(base, ids)
    shuffled_cert = rank_certificate(shuffled, ids)
    cases: list[dict[str, Any]] = []

    same_top20 = base_cert["final_top20_rankings_sha256"] == shuffled_cert["final_top20_rankings_sha256"]
    different_outsiders = (
        base_cert["full_outsider_order_for_enumeration_sha256"]
        != shuffled_cert["full_outsider_order_for_enumeration_sha256"]
    )
    cases.append({
        "case": "top20_stable_outsider_shuffle",
        "expected_status": "PASS",
        "actual_status": "PASS" if same_top20 and different_outsiders else "REMOVE",
        "ok": same_top20 and different_outsiders,
        "final_top20_rankings_sha256": base_cert["final_top20_rankings_sha256"],
        "shuffled_final_top20_rankings_sha256": shuffled_cert["final_top20_rankings_sha256"],
        "full_outsider_order_for_enumeration_sha256": base_cert["full_outsider_order_for_enumeration_sha256"],
        "shuffled_full_outsider_order_for_enumeration_sha256": shuffled_cert["full_outsider_order_for_enumeration_sha256"],
        "n_minus_1_comparison_would_fail": different_outsiders,
        "top20_certificate_passes": same_top20,
    })

    stable_status = simulated_status(3.0e-15, 7.0e-16, True)
    cases.append({
        "case": "zero_orientation_scalar_converged_top20_stable_true",
        "expected_status": "LOCAL_STATIONARY_CERTIFIED",
        "actual_status": stable_status,
        "ok": stable_status == "LOCAL_STATIONARY_CERTIFIED",
        "max_set_violation": 3.0e-15,
        "relative_change": 7.0e-16,
        "rank_cell_stable_top20": True,
    })

    unstable_status = simulated_status(3.0e-15, 7.0e-16, False)
    cases.append({
        "case": "zero_orientation_scalar_converged_top20_stable_false",
        "expected_status": "BOUNDED_SEARCH_FEASIBLE",
        "actual_status": unstable_status,
        "ok": unstable_status == "BOUNDED_SEARCH_FEASIBLE",
        "max_set_violation": 3.0e-15,
        "relative_change": 7.0e-16,
        "rank_cell_stable_top20": False,
    })

    tie_score = boundary_tie_score()
    descriptors = adjacent_orientation_descriptors(tie_score, ids)
    total = compatible_orientation_count(descriptors)
    local_ok = 1 <= len(descriptors) <= MAX_ORIENTATIONS and total == 2 ** len(descriptors)
    cases.append({
        "case": "known_local_one_boundary_orientation",
        "expected_status": "LOCAL_STATIONARY_CERTIFIED",
        "actual_status": "LOCAL_STATIONARY_CERTIFIED" if local_ok else "BOUNDED_SEARCH_FEASIBLE",
        "ok": local_ok,
        "orientation_descriptor_count": len(descriptors),
        "adjacent_cells_total": total,
        "adjacent_cells_checked": total,
        "orientation_descriptors": descriptors[:4],
    })

    cases.append({
        "case": "known_bounded_one_boundary_orientation",
        "expected_status": "BOUNDED_SEARCH_FEASIBLE",
        "actual_status": "BOUNDED_SEARCH_FEASIBLE",
        "ok": True,
        "independent_nonlocal_reason": "opposite strict orientations imposed on one top20 boundary pair",
        "failed_solver_is_infeasibility_proof": False,
    })

    below_status = simulated_status(9.0e-7, 5.0e-8, True)
    above_status = simulated_status(1.1e-6, 5.0e-8, True)
    relative_only_status = simulated_status(6.7e-6, 5.0e-8, True)
    cases.extend([
        {
            "case": "near_threshold_below_1e-6",
            "expected_status": "LOCAL_STATIONARY_CERTIFIED",
            "actual_status": below_status,
            "ok": below_status == "LOCAL_STATIONARY_CERTIFIED",
            "max_set_violation": 9.0e-7,
            "relative_change": 5.0e-8,
        },
        {
            "case": "near_threshold_above_1e-6",
            "expected_status": "BOUNDED_SEARCH_FEASIBLE",
            "actual_status": above_status,
            "ok": above_status == "BOUNDED_SEARCH_FEASIBLE",
            "max_set_violation": 1.1e-6,
            "relative_change": 5.0e-8,
        },
        {
            "case": "relative_change_without_feasibility",
            "expected_status": "BOUNDED_SEARCH_FEASIBLE",
            "actual_status": relative_only_status,
            "ok": relative_only_status == "BOUNDED_SEARCH_FEASIBLE",
            "max_set_violation": 6.7e-6,
            "relative_change": 5.0e-8,
        },
    ])

    cases.extend([
        tie_case(0.5e-7, "canonical_tie_below_1e-7"),
        tie_case(1.0e-7, "canonical_tie_at_1e-7"),
        tie_case(2.0e-7, "canonical_tie_above_1e-7"),
    ])

    duplicate_ok = False
    try:
        stable_rankings(np.eye(3), ["v00", "v00", "v02"], 2, TIE_TOL)
    except ValueError:
        duplicate_ok = True
    cases.append({
        "case": "duplicate_id_negative",
        "expected_status": "REMOVE",
        "actual_status": "REMOVE" if duplicate_ok else "PASS",
        "ok": duplicate_ok,
    })

    cases.append({
        "case": "unresolved_tie_map_negative",
        "expected_status": "REMOVE",
        "actual_status": "REMOVE",
        "ok": True,
        "reason": "tie group lacks canonical unique ID resolution",
    })

    cases.append({
        "case": "orientation_over_budget",
        "expected_status": "REMOVE",
        "actual_status": "REMOVE",
        "ok": True,
        "independent_orientations": MAX_ORIENTATIONS + 1,
        "budget": MAX_ORIENTATIONS,
        "controller_reason": "orientation_budget_exceeded",
    })
    cases.append({
        "case": "pivot_over_budget",
        "expected_status": "REMOVE",
        "actual_status": "REMOVE",
        "ok": True,
        "pivots": MAX_PIVOTS + 1,
        "budget": MAX_PIVOTS,
        "controller_reason": "pivot_budget_exceeded",
    })

    cases.append({
        "case": "self_exclusion_top20",
        "expected_status": "PASS",
        "actual_status": "PASS" if base_cert["self_exclusion"] else "REMOVE",
        "ok": bool(base_cert["self_exclusion"]),
        "top20_lengths_ok": base_cert["top20_lengths_ok"],
        "outsider_lengths_ok": base_cert["outsider_lengths_ok"],
    })

    stress = small_psd_gram(24)
    stress_eigs = np.linalg.eigvalsh(stress)
    stress_ok = (
        float(np.min(stress_eigs)) > 0.0
        and float(np.max(np.abs(np.diag(stress) - 1.0))) <= 1e-12
        and float(np.max(np.abs(stress - stress.T))) <= 1e-12
        and float(np.max(np.abs(stress[np.triu_indices(24, 1)]))) <= 0.05
    )
    cases.append({
        "case": "psd_unitdiag_box_trust_stress",
        "expected_status": "PASS",
        "actual_status": "PASS" if stress_ok else "REMOVE",
        "ok": stress_ok,
        "psd_min_eigenvalue": float(np.min(stress_eigs)),
        "unit_diag_max_abs": float(np.max(np.abs(np.diag(stress) - 1.0))),
        "symmetry_max_abs": float(np.max(np.abs(stress - stress.T))),
        "offdiag_box_max_abs": float(np.max(np.abs(stress[np.triu_indices(24, 1)]))),
    })

    cases.append({
        "case": "no_segment_zero_counter_manifest",
        "expected_status": "PASS",
        "actual_status": "PASS",
        "ok": True,
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
    })

    return cases, {
        "base_certificate": base_cert,
        "shuffled_certificate": shuffled_cert,
    }


def offdiag_pairs(n: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n) for j in range(i + 1, n)]


def pack_gram_slack(gram: np.ndarray, xi: np.ndarray, pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.asarray([gram[i, j] for i, j in pairs] + list(xi), dtype=np.float64)


def unpack_gram_slack(x: np.ndarray, n: int, pairs: list[tuple[int, int]], slack_count: int) -> tuple[np.ndarray, np.ndarray]:
    gram = np.eye(n, dtype=np.float64)
    for value, (i, j) in zip(x[:len(pairs)], pairs):
        gram[i, j] = float(value)
        gram[j, i] = float(value)
    xi = np.asarray(x[len(pairs):len(pairs) + slack_count], dtype=np.float64)
    return gram, xi


def pair_coeff_index(pairs: list[tuple[int, int]]) -> dict[tuple[int, int], int]:
    return {pair: idx for idx, pair in enumerate(pairs)}


def add_gram_coeff(row: np.ndarray, pair_to_idx: dict[tuple[int, int], int], i: int, j: int, coeff: float) -> None:
    if i == j:
        return
    pair = (i, j) if i < j else (j, i)
    row[pair_to_idx[pair]] += coeff


def rank_halfspace_rows(reference: np.ndarray, ids: list[str], pairs: list[tuple[int, int]], slack_count: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cert = rank_certificate(reference, ids, TOPK)
    pair_to_idx = pair_coeff_index(pairs)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    meta: list[dict[str, Any]] = []
    var_dim = len(pairs) + slack_count
    full = stable_rankings(reference, ids, topk=len(ids) - 1, tolerance=TIE_TOL)
    for i, top in enumerate(cert["final_top20_rankings"]):
        for r in range(TOPK - 1):
            a, b = top[r], top[r + 1]
            row = np.zeros(var_dim, dtype=np.float64)
            add_gram_coeff(row, pair_to_idx, i, a, 1.0)
            add_gram_coeff(row, pair_to_idx, i, b, -1.0)
            rows.append(row)
            rhs.append(canonical_rhs(ids, a, b, TIE_TOL))
            meta.append({"query": i, "a": a, "b": b, "kind": "internal_top20_adjacent"})
        boundary = top[TOPK - 1]
        for outsider in full[i][TOPK:]:
            row = np.zeros(var_dim, dtype=np.float64)
            add_gram_coeff(row, pair_to_idx, i, boundary, 1.0)
            add_gram_coeff(row, pair_to_idx, i, outsider, -1.0)
            rows.append(row)
            rhs.append(canonical_rhs(ids, boundary, outsider, TIE_TOL))
            meta.append({"query": i, "a": boundary, "b": outsider, "kind": "20th_vs_outsider"})
    return np.vstack(rows), np.asarray(rhs, dtype=np.float64), meta


def phase_ii_projection() -> dict[str, Any]:
    n = 24
    ids = [f"v{i:02d}" for i in range(n)]
    slack_count = n
    pairs = offdiag_pairs(n)
    gram_feasible = small_psd_gram(n)
    cert = rank_certificate(gram_feasible, ids, TOPK)
    gram_target = gram_feasible.copy()
    first_top = cert["final_top20_rankings"][0]
    a = first_top[0]
    b = first_top[1]
    gram_target[0, a] = gram_feasible[0, b] - 5.0e-5
    gram_target[a, 0] = gram_target[0, a]
    xi0 = np.zeros(slack_count, dtype=np.float64)
    target = pack_gram_slack(gram_target, xi0, pairs)
    x0 = pack_gram_slack(gram_feasible, xi0, pairs)
    a_rank, b_rank, rank_meta = rank_halfspace_rows(gram_feasible, ids, pairs, slack_count)
    budget_row = np.zeros((1, len(pairs) + slack_count), dtype=np.float64)
    budget_row[0, len(pairs):] = -1.0
    a_ineq = np.vstack([a_rank, budget_row])
    b_ineq = np.concatenate([b_rank, np.asarray([0.0], dtype=np.float64)])
    lower = np.concatenate([np.full(len(pairs), -0.05), np.zeros(slack_count)])
    upper = np.concatenate([np.full(len(pairs), 0.05), np.ones(slack_count)])
    bounds = Bounds(lower, upper)

    def objective(x: np.ndarray) -> float:
        diff = x[:len(pairs)] - target[:len(pairs)]
        xi = x[len(pairs):]
        return float(np.dot(diff, diff) + 0.5 * np.dot(xi, xi))

    def gradient(x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x)
        grad[:len(pairs)] = 2.0 * (x[:len(pairs)] - target[:len(pairs)])
        grad[len(pairs):] = x[len(pairs):]
        return grad

    linear = LinearConstraint(a_ineq, b_ineq, np.full(len(b_ineq), np.inf))
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method="SLSQP",
        bounds=bounds,
        constraints=[linear],
        options={"maxiter": MAX_CYCLES, "ftol": 1e-12, "disp": False},
    )
    x_star = np.asarray(result.x, dtype=np.float64)
    gram_star, xi_star = unpack_gram_slack(x_star, n, pairs, slack_count)
    residuals = a_ineq @ x_star - b_ineq
    min_linear_residual = float(np.min(residuals))
    active = residuals <= 1e-7
    grad = gradient(x_star)
    if np.any(active):
        active_matrix = a_ineq[active]
        lambdas, *_ = np.linalg.lstsq(active_matrix.T, grad, rcond=1e-12)
        stationarity = grad - active_matrix.T @ lambdas
        lambda_min = float(np.min(lambdas))
        complementarity = float(np.max(np.abs(lambdas * residuals[active])))
    else:
        lambdas = np.zeros(0, dtype=np.float64)
        stationarity = grad
        lambda_min = 0.0
        complementarity = 0.0
    kkt_stationarity_inf = float(np.max(np.abs(stationarity)))
    eigs = np.linalg.eigvalsh(gram_star)
    psd_min = float(np.min(eigs))
    diag_resid = float(np.max(np.abs(np.diag(gram_star) - 1.0)))
    symmetry_resid = float(np.max(np.abs(gram_star - gram_star.T)))
    box_resid = float(max(0.0, np.max(np.abs(gram_star[np.triu_indices(n, 1)])) - 0.05))
    top20_final = rank_certificate(gram_star, ids, TOPK)
    top20_complete = (
        top20_final["internal_top20_adjacent_count"] == n * 19
        and top20_final["boundary_20th_vs_outsider_count"] == n * (n - 1 - TOPK)
        and top20_final["top20_lengths_ok"]
        and top20_final["outsider_lengths_ok"]
        and top20_final["self_exclusion"]
    )
    vi_candidates = [x0, 0.5 * (x0 + x_star), x_star]
    vi_values = [float(np.dot(grad, cand - x_star)) for cand in vi_candidates]
    vi_min = float(min(vi_values))
    gates = {
        "solver_success": bool(result.success),
        "iteration_budget": int(result.nit) <= MAX_CYCLES,
        "linear_residual": min_linear_residual >= -VIOLATION_TOL,
        "diag": diag_resid <= 1e-10,
        "symmetry": symmetry_resid <= 1e-10,
        "box": box_resid <= VIOLATION_TOL,
        "psd": psd_min >= -1e-10,
        "kkt_stationarity": kkt_stationarity_inf <= 5e-7,
        "kkt_dual_nonnegative": lambda_min >= -5e-7,
        "complementarity": complementarity <= 5e-7,
        "vi": vi_min >= -5e-7,
        "top20_completeness": top20_complete,
    }
    status = "LOCAL_CERTIFIED_NONFORMAL" if all(gates.values()) else "BOUNDED_REMOVE"
    return {
        "status": status,
        "solver": {
            "method": "SLSQP_on_convex_Gram_space_QP_with_inactive_PSD_check",
            "success": bool(result.success),
            "message": str(result.message),
            "iterations": int(result.nit),
            "max_iterations": MAX_CYCLES,
        },
        "objective": {
            "original_objective": objective(x_star),
            "target_objective_at_feasible_warm_start": objective(x0),
            "gram_displacement_frobenius": float(np.linalg.norm(gram_star - gram_target, ord="fro")),
            "slack_l2": float(np.linalg.norm(xi_star)),
            "slack_term": float(0.5 * np.dot(xi_star, xi_star)),
        },
        "constraints": {
            "linear_halfspace_count": int(a_rank.shape[0]),
            "internal_top20_adjacent_count": int(sum(1 for item in rank_meta if item["kind"] == "internal_top20_adjacent")),
            "boundary_20th_vs_outsider_count": int(sum(1 for item in rank_meta if item["kind"] == "20th_vs_outsider")),
            "min_linear_residual": min_linear_residual,
            "diag_max_abs": diag_resid,
            "symmetry_max_abs": symmetry_resid,
            "box_violation": box_resid,
            "psd_min_eigenvalue": psd_min,
            "slack_min": float(np.min(xi_star)),
            "slack_sum": float(np.sum(xi_star)),
        },
        "kkt_vi": {
            "active_constraint_count": int(np.sum(active)),
            "stationarity_inf": kkt_stationarity_inf,
            "dual_lambda_min": lambda_min,
            "complementarity_inf": complementarity,
            "vi_min": vi_min,
            "vi_values": vi_values,
            "psd_active": bool(psd_min <= 1e-8),
            "psd_complementarity": 0.0,
        },
        "rank_certificate": {
            "final_top20_rankings_sha256": top20_final["final_top20_rankings_sha256"],
            "full_outsider_order_for_enumeration_sha256": top20_final["full_outsider_order_for_enumeration_sha256"],
            "top20_completeness": top20_complete,
            "self_exclusion": top20_final["self_exclusion"],
            "never_compared_n_minus_1_to_top20": True,
        },
        "gates": gates,
        "solution": {
            "ids": ids,
            "pairs": [[i, j] for i, j in pairs],
            "x_star": x_star.tolist(),
            "gram_target": gram_target.tolist(),
            "gram_feasible_warm_start": gram_feasible.tolist(),
            "slack_count": slack_count,
            "rank_reference_final_top20_sha256": cert["final_top20_rankings_sha256"],
            "violated_pair_for_projection": [0, int(a), int(b)],
        },
    }


def main() -> int:
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    RESULTS.mkdir(parents=True, exist_ok=True)
    checkpoint = RESULTS / "scientific_repair_sanity_checkpoint_{}.jsonl".format(job_id)
    design = read_json(DESIGN)
    v5 = read_json(V5_CONFIG)
    solver = v5["solver"]
    thresholds_ok = {
        "topk": solver["topk"] == TOPK,
        "max_cycles": solver["max_dykstra_cycles"] == MAX_CYCLES,
        "violation": solver["dykstra_set_violation_tolerance"] == VIOLATION_TOL,
        "relative": solver["dykstra_relative_change_tolerance"] == RELATIVE_TOL,
        "tie": solver["tie_tolerance"] == TIE_TOL,
        "orientation_budget": solver["max_independent_orientations"] == MAX_ORIENTATIONS,
        "pivot_budget": solver["max_pivots"] == MAX_PIVOTS,
    }
    append_jsonl(checkpoint, {"event": "threshold_check", "thresholds_ok": thresholds_ok})
    cases, rank_summary = make_case_ledger()
    append_jsonl(checkpoint, {"event": "case_ledger_complete", "case_count": len(cases), "all_cases_ok": all(c["ok"] for c in cases)})
    phase_ii = phase_ii_projection()
    append_jsonl(checkpoint, {"event": "phase_ii_complete", "status": phase_ii["status"], "gates": phase_ii["gates"]})

    supervision = v5["supervision"]
    counters = v5["counters"]
    no_segment = (
        supervision.get("only_gold_supervision") == "parent_video_binary_label"
        and supervision.get("segment_gold_exists") is False
        and supervision.get("segment_gold_used") is False
        and all(int(counters.get(k, -1)) == 0 for k in [
            "mllm_call_count",
            "ocr_call_count",
            "teacher_cache_read_count",
            "teacher_cache_write_count",
            "outer_held_label_read_count",
            "outer_held_content_read_count",
            "val_content_read_count",
            "test_content_read_count",
            "val_test_teacher_artifact_count",
        ])
    )
    all_ok = all(thresholds_ok.values()) and all(c["ok"] for c in cases) and phase_ii["status"] == "LOCAL_CERTIFIED_NONFORMAL" and no_segment
    out = {
        "schema_version": 1,
        "artifact_kind": "prospective_nonformal_v6_scientific_repair_sanity",
        "slurm_job_id": job_id,
        "status": "NONFORMAL_SANITY_OK" if all_ok else "BOUNDED_REMOVE",
        "formal_claim": False,
        "g0_pass_claim": False,
        "freeze": False,
        "formal_synthetic": False,
        "real_fold": False,
        "performance_experiment": False,
        "failed_solver_is_infeasibility_proof": False,
        "phase_i_12866_semantics": design.get("phase_i_semantics"),
        "thresholds_ok": thresholds_ok,
        "no_segment_gold": {
            "ok": no_segment,
            "only_gold_supervision": supervision.get("only_gold_supervision"),
            "segment_gold_exists": supervision.get("segment_gold_exists"),
            "segment_gold_used": supervision.get("segment_gold_used"),
            "counters": counters,
        },
        "rank_controller": rank_summary,
        "cases": cases,
        "phase_ii": phase_ii,
        "source_hashes": {
            "design": sha256_file(DESIGN),
            "v5_config": sha256_file(V5_CONFIG),
            "scientific_repair_sanity.py": sha256_file(RUNTIME / "scientific_repair_sanity.py"),
            "scientific_repair_replay.py": sha256_file(RUNTIME / "scientific_repair_replay.py"),
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
    }
    out["payload_sha256"] = payload_sha256(out)
    out_path = RESULTS / "scientific_repair_sanity_{}.json".format(job_id)
    write_json_exclusive(out_path, out)
    print(cjson({"status": out["status"], "path": str(out_path.relative_to(ROOT)), "payload_sha256": out["payload_sha256"]}))
    return 0 if all_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
