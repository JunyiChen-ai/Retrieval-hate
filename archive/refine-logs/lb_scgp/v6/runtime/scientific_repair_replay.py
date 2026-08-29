#!/usr/bin/env python3
"""Independent replay for the prospective v6 scientific-repair sanity output."""

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


ROOT = Path("/data/jehc223/RGCL")
V6 = ROOT / "refine-logs" / "lb_scgp" / "v6"
RESULTS = V6 / "results"
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
    out: list[list[int]] = []
    for i in range(n):
        candidates = [j for j in range(n) if j != i]
        local = tolerance_order([float(score[i, j]) for j in candidates], [ids[j] for j in candidates], tolerance)
        out.append([candidates[k] for k in local[:topk]])
    return out


def rank_certificate(score: np.ndarray, ids: list[str], topk: int = TOPK) -> dict[str, Any]:
    n = len(ids)
    final_top20 = stable_rankings(score, ids, topk=topk, tolerance=TIE_TOL)
    full = stable_rankings(score, ids, topk=n - 1, tolerance=TIE_TOL)
    outsiders = [row[topk:] for row in full]
    return {
        "final_top20_rankings": final_top20,
        "full_outsider_order_for_enumeration": outsiders,
        "final_top20_rankings_sha256": sha256_obj(final_top20),
        "full_outsider_order_for_enumeration_sha256": sha256_obj(outsiders),
        "self_exclusion": all(i not in row for i, row in enumerate(final_top20)),
        "top20_lengths_ok": all(len(row) == topk for row in final_top20),
        "outsider_lengths_ok": all(len(row) == n - 1 - topk for row in outsiders),
        "internal_top20_adjacent_count": n * (topk - 1),
        "boundary_20th_vs_outsider_count": sum(len(row) for row in outsiders),
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


def add_gram_coeff(row: np.ndarray, pair_to_idx: dict[tuple[int, int], int], i: int, j: int, coeff: float) -> None:
    if i == j:
        return
    pair = (i, j) if i < j else (j, i)
    row[pair_to_idx[pair]] += coeff


def rank_halfspace_rows(reference: np.ndarray, ids: list[str], pairs: list[tuple[int, int]], slack_count: int) -> tuple[np.ndarray, np.ndarray]:
    pair_to_idx = {pair: idx for idx, pair in enumerate(pairs)}
    var_dim = len(pairs) + slack_count
    final_top20 = stable_rankings(reference, ids, topk=TOPK, tolerance=TIE_TOL)
    full = stable_rankings(reference, ids, topk=len(ids) - 1, tolerance=TIE_TOL)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for i, top in enumerate(final_top20):
        for r in range(TOPK - 1):
            a, b = top[r], top[r + 1]
            row = np.zeros(var_dim, dtype=np.float64)
            add_gram_coeff(row, pair_to_idx, i, a, 1.0)
            add_gram_coeff(row, pair_to_idx, i, b, -1.0)
            rows.append(row)
            rhs.append(canonical_rhs(ids, a, b, TIE_TOL))
        boundary = top[TOPK - 1]
        for outsider in full[i][TOPK:]:
            row = np.zeros(var_dim, dtype=np.float64)
            add_gram_coeff(row, pair_to_idx, i, boundary, 1.0)
            add_gram_coeff(row, pair_to_idx, i, outsider, -1.0)
            rows.append(row)
            rhs.append(canonical_rhs(ids, boundary, outsider, TIE_TOL))
    return np.vstack(rows), np.asarray(rhs, dtype=np.float64)


def recompute_phase_ii(solution: dict[str, Any]) -> dict[str, Any]:
    ids = [f"v{i:02d}" for i in range(24)]
    n = len(ids)
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
    x_star = np.asarray(solution["x_star"], dtype=np.float64)
    xi0 = np.zeros(slack_count, dtype=np.float64)
    target = pack_gram_slack(gram_target, xi0, pairs)
    x0 = pack_gram_slack(gram_feasible, xi0, pairs)
    a_rank, b_rank = rank_halfspace_rows(gram_feasible, ids, pairs, slack_count)
    budget_row = np.zeros((1, len(pairs) + slack_count), dtype=np.float64)
    budget_row[0, len(pairs):] = -1.0
    a_ineq = np.vstack([a_rank, budget_row])
    b_ineq = np.concatenate([b_rank, np.asarray([0.0])])

    def objective(x: np.ndarray) -> float:
        diff = x[:len(pairs)] - target[:len(pairs)]
        xi = x[len(pairs):]
        return float(np.dot(diff, diff) + 0.5 * np.dot(xi, xi))

    def gradient(x: np.ndarray) -> np.ndarray:
        grad = np.zeros_like(x)
        grad[:len(pairs)] = 2.0 * (x[:len(pairs)] - target[:len(pairs)])
        grad[len(pairs):] = x[len(pairs):]
        return grad

    gram_star, xi_star = unpack_gram_slack(x_star, n, pairs, slack_count)
    residuals = a_ineq @ x_star - b_ineq
    active = residuals <= 1e-7
    grad = gradient(x_star)
    if np.any(active):
        active_matrix = a_ineq[active]
        lambdas, *_ = np.linalg.lstsq(active_matrix.T, grad, rcond=1e-12)
        stationarity = grad - active_matrix.T @ lambdas
        lambda_min = float(np.min(lambdas))
        complementarity = float(np.max(np.abs(lambdas * residuals[active])))
    else:
        stationarity = grad
        lambda_min = 0.0
        complementarity = 0.0
    eigs = np.linalg.eigvalsh(gram_star)
    top20_final = rank_certificate(gram_star, ids, TOPK)
    top20_complete = (
        top20_final["internal_top20_adjacent_count"] == n * 19
        and top20_final["boundary_20th_vs_outsider_count"] == n * (n - 1 - TOPK)
        and top20_final["top20_lengths_ok"]
        and top20_final["outsider_lengths_ok"]
        and top20_final["self_exclusion"]
    )
    vi_values = [float(np.dot(grad, cand - x_star)) for cand in [x0, 0.5 * (x0 + x_star), x_star]]
    metrics = {
        "original_objective": objective(x_star),
        "min_linear_residual": float(np.min(residuals)),
        "diag_max_abs": float(np.max(np.abs(np.diag(gram_star) - 1.0))),
        "symmetry_max_abs": float(np.max(np.abs(gram_star - gram_star.T))),
        "box_violation": float(max(0.0, np.max(np.abs(gram_star[np.triu_indices(n, 1)])) - 0.05)),
        "psd_min_eigenvalue": float(np.min(eigs)),
        "slack_min": float(np.min(xi_star)),
        "slack_sum": float(np.sum(xi_star)),
        "stationarity_inf": float(np.max(np.abs(stationarity))),
        "dual_lambda_min": lambda_min,
        "complementarity_inf": complementarity,
        "vi_min": float(min(vi_values)),
        "top20_completeness": top20_complete,
        "final_top20_rankings_sha256": top20_final["final_top20_rankings_sha256"],
        "full_outsider_order_for_enumeration_sha256": top20_final["full_outsider_order_for_enumeration_sha256"],
    }
    gates = {
        "linear_residual": metrics["min_linear_residual"] >= -VIOLATION_TOL,
        "diag": metrics["diag_max_abs"] <= 1e-10,
        "symmetry": metrics["symmetry_max_abs"] <= 1e-10,
        "box": metrics["box_violation"] <= VIOLATION_TOL,
        "psd": metrics["psd_min_eigenvalue"] >= -1e-10,
        "kkt_stationarity": metrics["stationarity_inf"] <= 5e-7,
        "kkt_dual_nonnegative": metrics["dual_lambda_min"] >= -5e-7,
        "complementarity": metrics["complementarity_inf"] <= 5e-7,
        "vi": metrics["vi_min"] >= -5e-7,
        "top20_completeness": top20_complete,
    }
    return {"metrics": metrics, "gates": gates, "ok": all(gates.values())}


def replay_cases(producer: dict[str, Any]) -> dict[str, Any]:
    cases = {row["case"]: row for row in producer.get("cases", [])}
    required = {
        "top20_stable_outsider_shuffle",
        "zero_orientation_scalar_converged_top20_stable_true",
        "zero_orientation_scalar_converged_top20_stable_false",
        "known_local_one_boundary_orientation",
        "known_bounded_one_boundary_orientation",
        "near_threshold_below_1e-6",
        "near_threshold_above_1e-6",
        "relative_change_without_feasibility",
        "canonical_tie_below_1e-7",
        "canonical_tie_at_1e-7",
        "canonical_tie_above_1e-7",
        "duplicate_id_negative",
        "unresolved_tie_map_negative",
        "orientation_over_budget",
        "pivot_over_budget",
        "self_exclusion_top20",
        "psd_unitdiag_box_trust_stress",
        "no_segment_zero_counter_manifest",
    }
    ids = [f"v{i:02d}" for i in range(24)]
    base = rank_certificate(circle_score_matrix(24, "base"), ids, TOPK)
    shuffled = rank_certificate(circle_score_matrix(24, "shuffled"), ids, TOPK)
    outsider_case = cases.get("top20_stable_outsider_shuffle", {})
    outsider_replay_ok = (
        base["final_top20_rankings_sha256"] == shuffled["final_top20_rankings_sha256"]
        and base["full_outsider_order_for_enumeration_sha256"] != shuffled["full_outsider_order_for_enumeration_sha256"]
        and outsider_case.get("final_top20_rankings_sha256") == base["final_top20_rankings_sha256"]
        and outsider_case.get("full_outsider_order_for_enumeration_sha256") == base["full_outsider_order_for_enumeration_sha256"]
    )
    stress = small_psd_gram(24)
    stress_ok = (
        float(np.min(np.linalg.eigvalsh(stress))) > 0.0
        and float(np.max(np.abs(np.diag(stress) - 1.0))) <= 1e-12
        and float(np.max(np.abs(stress - stress.T))) <= 1e-12
    )
    all_cases_present = set(cases) == required
    all_case_flags = all(row.get("ok") is True for row in cases.values())
    return {
        "required_cases_present": all_cases_present,
        "all_case_flags_true": all_case_flags,
        "outsider_shuffle_recomputed": outsider_replay_ok,
        "stress_recomputed": stress_ok,
        "ok": all_cases_present and all_case_flags and outsider_replay_ok and stress_ok,
    }


def main() -> int:
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    source = os.environ.get("SCIENTIFIC_REPAIR_SANITY_PATH")
    if not source:
        raise SystemExit("SCIENTIFIC_REPAIR_SANITY_PATH is required")
    source_path = (ROOT / source).resolve() if not source.startswith("/") else Path(source)
    producer = read_json(source_path)
    payload_ok = producer.get("payload_sha256") == payload_sha256(producer)
    phase_ii_replay = recompute_phase_ii(producer["phase_ii"]["solution"])
    case_replay = replay_cases(producer)
    v5 = read_json(V5_CONFIG)
    solver = v5["solver"]
    thresholds_ok = (
        solver.get("topk") == TOPK
        and solver.get("max_dykstra_cycles") == MAX_CYCLES
        and solver.get("dykstra_set_violation_tolerance") == VIOLATION_TOL
        and solver.get("dykstra_relative_change_tolerance") == RELATIVE_TOL
        and solver.get("tie_tolerance") == TIE_TOL
        and solver.get("max_independent_orientations") == MAX_ORIENTATIONS
        and solver.get("max_pivots") == MAX_PIVOTS
    )
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
    ok = (
        payload_ok
        and producer.get("status") == "NONFORMAL_SANITY_OK"
        and thresholds_ok
        and no_segment
        and phase_ii_replay["ok"]
        and case_replay["ok"]
    )
    out = {
        "schema_version": 1,
        "artifact_kind": "prospective_nonformal_v6_scientific_repair_replay",
        "slurm_job_id": job_id,
        "source_path": str(source_path.relative_to(ROOT)),
        "source_sha256": sha256_file(source_path),
        "status": "REPLAY_OK" if ok else "BOUNDED_REMOVE",
        "formal_claim": False,
        "g0_pass_claim": False,
        "payload_ok": payload_ok,
        "thresholds_ok": thresholds_ok,
        "no_segment_gold_ok": no_segment,
        "case_replay": case_replay,
        "phase_ii_replay": phase_ii_replay,
        "failed_solver_is_infeasibility_proof": False,
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        },
    }
    out["payload_sha256"] = payload_sha256(out)
    out_path = RESULTS / "scientific_repair_replay_{}.json".format(job_id)
    write_json_exclusive(out_path, out)
    print(cjson({"status": out["status"], "path": str(out_path.relative_to(ROOT)), "payload_sha256": out["payload_sha256"]}))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
