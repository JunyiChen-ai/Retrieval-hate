#!/usr/bin/env python3
"""v7 actual signed-cell certificate producer.

All numerical execution for this script is intended to run under SLURM in the
HateVideo conda environment.  The producer may use scipy to search primals and
fit active-set multipliers, but it never accepts solver success by itself.
"""

from __future__ import annotations

import os
import platform
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, lsq_linear, minimize

from v7_common import (
    ACTIVE_TOL,
    COMPLEMENTARITY_TOL,
    DESIGN,
    ETA,
    FULL_RANK_TOL,
    OUT_DIR,
    PHASE1_RESIDUAL_TOL,
    PHASE2_VI_TOL,
    ROOT,
    RUNTIME,
    SIGNED_RHS,
    TAU,
    TOPK,
    V7,
    DirectProblem,
    bounds_arrays,
    build_cells,
    build_problem,
    canonical_top20,
    cjson,
    constraint_map,
    current_source_hashes,
    existing_hashes_unchanged,
    hfile,
    hobj,
    linear_values,
    load_design,
    load_inputs,
    max_residual,
    nonlinear_values_jac,
    objective_value_grad,
    pack_g_xi,
    payload_hash,
    phase_starts,
    residual_sets_original,
    self_check_core,
    signed_gap_residuals,
    supervision_contract,
    unpack_g_xi,
    write_json_exclusive,
    append_jsonl,
)


def matrix_for_rows(rows: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    if not rows:
        return np.zeros((0, 0), dtype=np.float64), np.zeros(0, dtype=np.float64)
    a = np.stack([row.coeff for row in rows]).astype(np.float64)
    b = -np.asarray([row.const for row in rows], dtype=np.float64)
    return a, b


def phase1_solve(problem: DirectProblem, starts: list[dict[str, Any]], checkpoint: Path) -> dict[str, Any]:
    lower, upper = bounds_arrays(problem)
    lower_z = np.concatenate([lower, np.asarray([-1e-6], dtype=np.float64)])
    upper_z = np.concatenate([upper, np.asarray([1.0], dtype=np.float64)])
    bounds = Bounds(lower_z, upper_z)
    a_lin, b_lin = matrix_for_rows(problem.all_linear_rows)
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
    selected: dict[str, Any] | None = None
    for start in starts:
        z0 = np.concatenate([np.asarray(start["x"], dtype=np.float64), np.asarray([float(start.get("t", 0.0))], dtype=np.float64)])
        z0 = np.minimum(np.maximum(z0, lower_z), upper_z)
        started = time.perf_counter()
        result = None
        error = None
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
            attempt = {
                "start_name": start["name"],
                "status": "ERROR",
                "error": error,
                "elapsed_seconds": elapsed,
            }
        else:
            z = np.asarray(result.x, dtype=np.float64)
            g, xi = unpack_g_xi(problem, z[:-1])
            rows, groups = residual_sets_original(problem, g, xi)
            signed = signed_gap_residuals(problem, g)
            eig_min = float(np.linalg.eigvalsh(0.5 * (g + g.T)).min())
            realized = canonical_top20(g, problem.ids, float(problem.solver["tie_tolerance"]))
            target = problem.final_top20
            phase1_ok = bool(
                max_residual(rows) <= PHASE1_RESIDUAL_TOL
                and eig_min > FULL_RANK_TOL
                and signed["pass"] is True
                and realized == target
            )
            attempt = {
                "start_name": start["name"],
                "status": "FULL_RANK_SIGNED_CELL_CANDIDATE" if phase1_ok else "NO_COMPATIBILITY_WITNESS_CANDIDATE",
                "optimizer_success": bool(result.success),
                "optimizer_message": str(result.message),
                "elapsed_seconds": elapsed,
                "nit": None if getattr(result, "nit", None) is None else int(result.nit),
                "objective_maximized_t": float(z[-1]),
                "psd_min_eigenvalue": eig_min,
                "full_rank_margin": eig_min,
                "max_589_original_residual": max_residual(rows),
                "original_residual_groups": groups,
                "signed_gap_summary": {k: v for k, v in signed.items() if k != "rows"},
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
            "max_589_original_residual": attempt.get("max_589_original_residual"),
            "signed_gap_min_margin": attempt.get("signed_gap_summary", {}).get("min_margin"),
            "full_rank_margin": attempt.get("full_rank_margin"),
            "elapsed_seconds": attempt.get("elapsed_seconds"),
        })
        if attempt.get("status") == "FULL_RANK_SIGNED_CELL_CANDIDATE":
            if selected is None or float(attempt["full_rank_margin"]) > float(selected["full_rank_margin"]):
                selected = attempt
    if selected is None:
        selected = min(
            attempts,
            key=lambda row: (
                float(row.get("signed_gap_summary", {}).get("max_residual", float("inf"))),
                float(row.get("max_589_original_residual", float("inf"))),
                -float(row.get("full_rank_margin", -float("inf"))),
            ),
        ) if attempts else None
    return {
        "cell_index": problem.cell_index,
        "assignment": problem.assignment,
        "status": "FULL_RANK_SIGNED_CELL_REPLAY_PENDING" if selected and selected.get("status") == "FULL_RANK_SIGNED_CELL_CANDIDATE" else "NO_COMPATIBILITY_WITNESS_NO_FARKAS",
        "selected_start_name": None if selected is None else selected.get("start_name"),
        "selected_full_rank_margin": None if selected is None else selected.get("full_rank_margin"),
        "selected_max_589_original_residual": None if selected is None else selected.get("max_589_original_residual"),
        "attempts": attempts,
        "selected": selected,
        "farkas_certificate": None,
        "incompatibility_claim": False,
        "nonconvergence_is_not_infeasibility": True,
    }


def active_set_certificate(problem: DirectProblem, x: np.ndarray, grad: np.ndarray) -> dict[str, Any]:
    g, _ = unpack_g_xi(problem, x)
    lin_rows = problem.all_linear_rows
    lin_values = linear_values(lin_rows, x)
    nl_values, nl_jac = nonlinear_values_jac(problem, x, phase1=False)
    eq_res = problem.eq_a @ x - problem.eq_b
    active_names: list[str] = []
    active_values: list[float] = []
    active_jac: list[np.ndarray] = []
    active_kinds: list[str] = []

    for row, value in zip(lin_rows, lin_values):
        if float(value) <= ACTIVE_TOL:
            active_names.append(row.name)
            active_values.append(float(value))
            active_jac.append(row.coeff)
            active_kinds.append(row.block)

    nl_names = ["psd_min_eigenvalue"]
    nl_names.extend("row_trust_{:02d}".format(i) for i in range(problem.n))
    nl_names.extend("class_mean_trust_{}".format(i) for i in range(len(problem.groups)))
    for name, value, jacrow in zip(nl_names, nl_values, nl_jac):
        if float(value) <= ACTIVE_TOL:
            active_names.append(name)
            active_values.append(float(value))
            active_jac.append(np.asarray(jacrow, dtype=np.float64))
            active_kinds.append("nonlinear_cone")

    lower, upper = bounds_arrays(problem)
    bound_rows = []
    for k, value in enumerate(x):
        if float(value - lower[k]) <= ACTIVE_TOL:
            row = np.zeros(problem.dim, dtype=np.float64)
            row[k] = 1.0
            active_names.append("bound_lower_{}".format(k))
            active_values.append(float(value - lower[k]))
            active_jac.append(row)
            active_kinds.append("bound")
            bound_rows.append(("lower", k, row, float(value - lower[k])))
        if float(upper[k] - value) <= ACTIVE_TOL:
            row = np.zeros(problem.dim, dtype=np.float64)
            row[k] = -1.0
            active_names.append("bound_upper_{}".format(k))
            active_values.append(float(upper[k] - value))
            active_jac.append(row)
            active_kinds.append("bound")
            bound_rows.append(("upper", k, row, float(upper[k] - value)))

    j_eq = problem.eq_a
    j_act = np.stack(active_jac).astype(np.float64) if active_jac else np.zeros((0, problem.dim), dtype=np.float64)
    mat = np.hstack([j_eq.T, -j_act.T])
    lb = np.concatenate([np.full(j_eq.shape[0], -np.inf), np.zeros(j_act.shape[0])])
    ub = np.full(j_eq.shape[0] + j_act.shape[0], np.inf)
    if mat.shape[1] == 0:
        multipliers = np.zeros(0, dtype=np.float64)
        stationarity = grad.copy()
        lsq_status = "no_active_columns"
        lsq_cost = None
    else:
        result = lsq_linear(mat, -grad, bounds=(lb, ub), tol=1e-12, max_iter=1000)
        multipliers = np.asarray(result.x, dtype=np.float64)
        stationarity = grad + mat @ multipliers
        lsq_status = str(result.status)
        lsq_cost = float(result.cost)
    eq_m = multipliers[:j_eq.shape[0]]
    ineq_m = multipliers[j_eq.shape[0]:]
    active_multiplier_map = {name: float(mu) for name, mu in zip(active_names, ineq_m)}

    linear_multiplier_rows = []
    for row, value in zip(lin_rows, lin_values):
        mu = float(active_multiplier_map.get(row.name, 0.0))
        linear_multiplier_rows.append({
            "name": row.name,
            "block": row.block,
            "group": row.group,
            "value": float(value),
            "multiplier": mu,
            "active": row.name in active_multiplier_map,
            "complementarity": float(abs(mu * float(value))),
        })

    bound_multiplier_rows = []
    for kind, k, _, value in bound_rows:
        name = "bound_{}_{}".format(kind, k)
        mu = float(active_multiplier_map.get(name, 0.0))
        bound_multiplier_rows.append({
            "name": name,
            "kind": kind,
            "variable_index": int(k),
            "value": float(value),
            "multiplier": mu,
            "active": True,
            "complementarity": float(abs(mu * value)),
        })

    soc_duals = []
    for i in range(problem.n):
        name = "row_trust_{:02d}".format(i)
        diff = g[i].copy() - problem.g0[i]
        diff[i] = 0.0
        norm = float(np.linalg.norm(diff))
        mu = float(active_multiplier_map.get(name, 0.0))
        beta = -mu * diff / norm if norm > 1e-15 else np.zeros(problem.n, dtype=np.float64)
        alpha = mu
        value = float(problem.row_radius - norm)
        soc_duals.append({
            "name": name,
            "kind": "row_trust",
            "t": float(problem.row_radius),
            "y": diff.tolist(),
            "alpha": float(alpha),
            "beta": beta.tolist(),
            "cone_margin": float(alpha - np.linalg.norm(beta)),
            "value": value,
            "active": name in active_multiplier_map,
            "complementarity": float(abs(alpha * problem.row_radius + float(beta @ diff))),
        })
    for k, rows in enumerate(problem.groups):
        name = "class_mean_trust_{}".format(k)
        mean = (g[rows] - problem.g0[rows]).mean(axis=0)
        norm = float(np.linalg.norm(mean))
        mu = float(active_multiplier_map.get(name, 0.0))
        beta = -mu * mean / norm if norm > 1e-15 else np.zeros(problem.n, dtype=np.float64)
        alpha = mu
        value = float(problem.class_radius - norm)
        soc_duals.append({
            "name": name,
            "kind": "class_mean_trust",
            "class_index": int(k),
            "rows": [int(v) for v in rows],
            "t": float(problem.class_radius),
            "y": mean.tolist(),
            "alpha": float(alpha),
            "beta": beta.tolist(),
            "cone_margin": float(alpha - np.linalg.norm(beta)),
            "value": value,
            "active": name in active_multiplier_map,
            "complementarity": float(abs(alpha * problem.class_radius + float(beta @ mean))),
        })

    eigvals, eigvecs = np.linalg.eigh(0.5 * (g + g.T))
    min_idx = int(np.argmin(eigvals))
    psd_mu = float(active_multiplier_map.get("psd_min_eigenvalue", 0.0))
    psd_active = bool(float(nl_values[0]) <= ACTIVE_TOL)
    s_psd = psd_mu * np.outer(eigvecs[:, min_idx], eigvecs[:, min_idx]) if psd_active else np.zeros_like(g)
    sg = s_psd @ g
    psd_eig_min = float(np.linalg.eigvalsh(0.5 * (s_psd + s_psd.T)).min())
    psd = {
        "active": psd_active,
        "status": "active_min_eigen_psd_dual" if psd_active else "inactive_S_zero",
        "mu": psd_mu,
        "eig_margin": float(eigvals[min_idx]),
        "S": s_psd.tolist(),
        "S_sha256": hobj(s_psd.tolist()),
        "S_eig_min": psd_eig_min,
        "SG_fro": float(np.linalg.norm(sg)),
        "trace_SG": float(np.sum(s_psd * g)),
        "trace_SG_abs": float(abs(np.sum(s_psd * g))),
        "dual_feasible": bool(psd_mu >= -1e-8 and psd_eig_min >= -1e-8 and ((psd_active and True) or (not psd_active and float(eigvals[min_idx]) > ACTIVE_TOL and abs(psd_mu) <= 1e-12))),
    }

    linear_comp = max((row["complementarity"] for row in linear_multiplier_rows), default=0.0)
    bound_comp = max((row["complementarity"] for row in bound_multiplier_rows), default=0.0)
    soc_comp = max((row["complementarity"] for row in soc_duals), default=0.0)
    comp_inf = float(max(linear_comp, bound_comp, soc_comp, psd["trace_SG_abs"]))
    linear_dual_min = min([row["multiplier"] for row in linear_multiplier_rows] + [0.0])
    bound_dual_min = min([row["multiplier"] for row in bound_multiplier_rows] + [0.0])
    soc_cone_min = min([row["cone_margin"] for row in soc_duals] + [0.0])
    dual_violation = float(max(0.0, -linear_dual_min, -bound_dual_min, -soc_cone_min, -psd_eig_min))
    stationarity_inf = float(np.max(np.abs(stationarity))) if stationarity.size else 0.0
    vi_bound = float(stationarity_inf + comp_inf + dual_violation)
    active_set_complete = bool(
        all(float(vv) > ACTIVE_TOL or row.name in active_multiplier_map for row, vv in zip(lin_rows, lin_values))
        and all(float(vv) > ACTIVE_TOL or name in active_multiplier_map for name, vv in zip(nl_names, nl_values))
    )
    return {
        "active_tolerance": ACTIVE_TOL,
        "eq_count": int(j_eq.shape[0]),
        "eq_residual_inf": float(np.max(np.abs(eq_res))) if eq_res.size else 0.0,
        "eq_multipliers": eq_m.tolist(),
        "active_ineq_count": int(j_act.shape[0]),
        "active_names": active_names,
        "active_kinds": active_kinds,
        "active_values": [float(vv) for vv in active_values],
        "active_ineq_multipliers": ineq_m.tolist(),
        "linear_multipliers": linear_multiplier_rows,
        "bound_multipliers": bound_multiplier_rows,
        "soc_duals": soc_duals,
        "psd": psd,
        "stationarity_inf": stationarity_inf,
        "stationarity_sha256": hobj(stationarity.tolist()),
        "linear_dual_min": float(linear_dual_min),
        "bound_dual_min": float(bound_dual_min),
        "soc_cone_min": float(soc_cone_min),
        "dual_violation": dual_violation,
        "complementarity_inf": comp_inf,
        "vi_residual_bound": vi_bound,
        "lsq_status": lsq_status,
        "lsq_cost": lsq_cost,
        "active_set_complete": active_set_complete,
        "dual_feasible": bool(dual_violation <= 1e-8 and psd["dual_feasible"]),
        "certificate_acceptance": bool(
            stationarity_inf <= PHASE2_VI_TOL
            and vi_bound <= PHASE2_VI_TOL
            and comp_inf <= COMPLEMENTARITY_TOL
            and active_set_complete
            and dual_violation <= 1e-8
            and psd["dual_feasible"]
        ),
    }


def phase2_solve(problem: DirectProblem, start: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    lower, upper = bounds_arrays(problem)
    bounds = Bounds(lower, upper)
    a_lin, b_lin = matrix_for_rows(problem.all_linear_rows)
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
    result = None
    error = None
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
        return {"cell_index": problem.cell_index, "status": "ERROR", "error": error, "elapsed_seconds": elapsed}
    x = np.asarray(result.x, dtype=np.float64)
    g, xi = unpack_g_xi(problem, x)
    rows, groups = residual_sets_original(problem, g, xi)
    signed = signed_gap_residuals(problem, g)
    objective, grad = objective_value_grad(problem, x)
    kkt = active_set_certificate(problem, x, grad)
    eig_min = float(np.linalg.eigvalsh(0.5 * (g + g.T)).min())
    realized_top20 = canonical_top20(g, problem.ids, float(problem.solver["tie_tolerance"]))
    local_supportable = bool(
        max_residual(rows) <= float(problem.solver["dykstra_set_violation_tolerance"])
        and signed["pass"] is True
        and realized_top20 == problem.final_top20
        and kkt["certificate_acceptance"] is True
    )
    status = "LOCAL_STATIONARY_CERTIFIED_CANDIDATE_REPLAY_PENDING" if local_supportable else "PRIMAL_DUAL_CERTIFICATE_FAILURE_PIVOT"
    out = {
        "cell_index": problem.cell_index,
        "assignment": problem.assignment,
        "status": status,
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
        "elapsed_seconds": elapsed,
        "nit": None if getattr(result, "nit", None) is None else int(result.nit),
        "objective": objective,
        "gram_displacement_fro": float(np.linalg.norm(g - problem.g0)),
        "slack_l2": float(np.linalg.norm(xi)),
        "psd_min_eigenvalue": eig_min,
        "max_589_original_residual": max_residual(rows),
        "original_residual_groups": groups,
        "signed_gap_summary": {k: v for k, v in signed.items() if k != "rows"},
        "realized_top20_equal_cell": realized_top20 == problem.final_top20,
        "realized_top20_sha256": hobj(realized_top20),
        "target_top20_sha256": hobj(problem.final_top20),
        "kkt": kkt,
        "vi": {
            "source": "reconstructed_linear_soc_psd_dual_certificate",
            "vi_residual_bound": kkt["vi_residual_bound"],
            "required": PHASE2_VI_TOL,
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
        "status": status,
        "objective": objective,
        "max_589_original_residual": out["max_589_original_residual"],
        "signed_gap_min_margin": signed["min_margin"],
        "stationarity_inf": kkt["stationarity_inf"],
        "vi_residual_bound": kkt["vi_residual_bound"],
        "complementarity_inf": kkt["complementarity_inf"],
        "psd_min_eigenvalue": eig_min,
        "elapsed_seconds": elapsed,
    })
    return out


def main() -> int:
    if "--self-check" in sys.argv:
        out = self_check_core()
        print(cjson(out))
        return 0 if out.get("ok") else 2

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    started = time.perf_counter()
    checkpoint = OUT_DIR / "v7_actual_certificate_checkpoint_{}.jsonl".format(job_id)
    design = load_design()
    cfg, oriented, fixture, witness, replay = load_inputs()
    contract = supervision_contract(cfg)
    if not contract.get("ok"):
        raise RuntimeError("supervision contract failed")
    unchanged = existing_hashes_unchanged()
    if not unchanged.get("ok"):
        raise RuntimeError("existing frozen hash check failed")
    system, cells = build_cells(fixture, cfg)
    if system["reject_reasons"]:
        raise RuntimeError("canonical signed cell build rejected: {}".format(system["reject_reasons"]))

    phase_i = []
    phase_ii = []
    problems: dict[int, DirectProblem] = {}
    for cell in cells:
        problem = build_problem(fixture, cfg, cell)
        problems[problem.cell_index] = problem
        print(cjson({"event": "phase_i_start", "cell_index": problem.cell_index, "job_id": job_id}), flush=True)
        p1 = phase1_solve(problem, phase_starts(problem, witness), checkpoint)
        phase_i.append(p1)
        print(cjson({
            "event": "phase_i_finish",
            "cell_index": problem.cell_index,
            "status": p1["status"],
            "selected_margin": p1.get("selected_full_rank_margin"),
            "job_id": job_id,
        }), flush=True)

    for row in phase_i:
        if row.get("status") != "FULL_RANK_SIGNED_CELL_REPLAY_PENDING":
            append_jsonl(checkpoint, {
                "phase": "phase_ii_skipped",
                "cell_index": row.get("cell_index"),
                "reason": row.get("status"),
            })
            continue
        problem = problems[int(row["cell_index"])]
        selected = row["selected"]["witness"]
        print(cjson({"event": "phase_ii_start", "cell_index": problem.cell_index, "job_id": job_id}), flush=True)
        p2 = phase2_solve(problem, selected, checkpoint)
        phase_ii.append(p2)
        print(cjson({
            "event": "phase_ii_finish",
            "cell_index": problem.cell_index,
            "status": p2["status"],
            "objective": p2.get("objective"),
            "job_id": job_id,
        }), flush=True)

    all_phase_i = bool(phase_i) and all(row.get("status") == "FULL_RANK_SIGNED_CELL_REPLAY_PENDING" for row in phase_i)
    all_phase_ii = len(phase_ii) == len(cells) and all(row.get("status") == "LOCAL_STATIONARY_CERTIFIED_CANDIDATE_REPLAY_PENDING" for row in phase_ii)
    status = "LOCAL_STATIONARY_CERTIFIED_CANDIDATE_REPLAY_PENDING" if all_phase_i and all_phase_ii else "PIVOT_TRIGGERED_CERTIFICATE_FAILURE_REPLAY_PENDING"
    out = {
        "schema_version": 1,
        "task": "lb_scgp_g0_v7_actual_signed_cell_certificate",
        "thread_session": design.get("thread_session"),
        "slurm_job_id": job_id,
        "python": sys.version,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "elapsed_seconds": time.perf_counter() - started,
        "status": status,
        "eta": ETA,
        "tau": TAU,
        "signed_gap_rhs": SIGNED_RHS,
        "immutable_thresholds": {
            "topk": int(cfg["solver"]["topk"]),
            "tie": float(cfg["solver"]["tie_tolerance"]),
            "violation": float(cfg["solver"]["dykstra_set_violation_tolerance"]),
            "relative": float(cfg["solver"]["dykstra_relative_change_tolerance"]),
            "max_independent_orientations": int(cfg["solver"]["max_independent_orientations"]),
            "max_pivots": int(cfg["solver"]["max_pivots"]),
        },
        "supervision_boundary": contract,
        "existing_hashes_unchanged": unchanged,
        "constraint_map": constraint_map(),
        "constraint_set_count": 589,
        "orientation_system": system,
        "compatible_cells": cells,
        "phase_i": phase_i,
        "phase_ii": phase_ii,
        "farkas_certificates": [],
        "defects_or_repairs": [],
        "checkpoint_path": str(checkpoint.relative_to(ROOT)),
        "source_hashes": current_source_hashes(),
        "design_path": str(DESIGN.relative_to(ROOT)),
        "design_sha256": hfile(DESIGN),
        "nonclaims": [
            "No G0 PASS, freeze, formal gate, realfold, G1, performance, validation, test, teacher, MLLM, OCR, or segment-level claim.",
            "NO_COMPATIBILITY_WITNESS_NO_FARKAS is not an infeasibility certificate.",
            "A pivot-triggering certificate failure requires fresh result-to-claim review.",
        ],
    }
    out["payload_sha256"] = payload_hash(out)
    out_path = OUT_DIR / "v7_actual_certificate_{}.json".format(job_id)
    write_json_exclusive(out_path, out)
    print(cjson({"status": status, "path": str(out_path.relative_to(ROOT)), "payload_sha256": out["payload_sha256"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
