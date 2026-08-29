#!/usr/bin/env python3
"""Solver-free independent replay for the LB-SCGP G0 v7 certificate."""

from __future__ import annotations

import os
import platform
import sys
from pathlib import Path
from typing import Any

import numpy as np

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
    SIGNED_RHS,
    TAU,
    DirectProblem,
    bounds_arrays,
    build_cells,
    build_problem,
    canonical_top20,
    cjson,
    current_source_hashes,
    existing_hashes_unchanged,
    hfile,
    hobj,
    linear_values,
    load_design,
    load_inputs,
    max_residual,
    objective_value_grad,
    payload_hash,
    read_json,
    residual_sets_original,
    signed_gap_residuals,
    supervision_contract,
    unpack_g_xi,
    write_json_exclusive,
)


def replay_stationarity(problem: DirectProblem, x: np.ndarray, kkt: dict[str, Any]) -> dict[str, Any]:
    g, _ = unpack_g_xi(problem, x)
    objective, grad = objective_value_grad(problem, x)
    stationarity = grad.copy()
    eq_m = np.asarray(kkt.get("eq_multipliers", []), dtype=np.float64)
    if eq_m.size != problem.eq_a.shape[0]:
        return {"ok": False, "reason": "eq_multiplier_count_mismatch"}
    stationarity += problem.eq_a.T @ eq_m
    lin_values = linear_values(problem.all_linear_rows, x)
    stored_linear = {row.get("name"): row for row in kkt.get("linear_multipliers", [])}
    linear_checks = []
    dual_violation = 0.0
    comp_inf = 0.0
    for row, value in zip(problem.all_linear_rows, lin_values):
        stored = stored_linear.get(row.name)
        if stored is None:
            return {"ok": False, "reason": "missing_linear_multiplier_{}".format(row.name)}
        mu = float(stored.get("multiplier", 0.0))
        stationarity -= mu * row.coeff
        dual_violation = max(dual_violation, max(0.0, -mu))
        comp = abs(mu * float(value))
        comp_inf = max(comp_inf, comp)
        linear_checks.append({
            "name": row.name,
            "block": row.block,
            "recomputed_value": float(value),
            "stored_value": float(stored.get("value", float("nan"))),
            "value_match": abs(float(value) - float(stored.get("value", float("nan")))) <= 1e-8,
            "multiplier": mu,
            "complementarity": float(comp),
        })

    lower, upper = bounds_arrays(problem)
    bound_checks = []
    for stored in kkt.get("bound_multipliers", []):
        kind = str(stored.get("kind"))
        idx = int(stored.get("variable_index"))
        mu = float(stored.get("multiplier", 0.0))
        if kind == "lower":
            value = float(x[idx] - lower[idx])
            stationarity[idx] -= mu
        elif kind == "upper":
            value = float(upper[idx] - x[idx])
            stationarity[idx] += mu
        else:
            return {"ok": False, "reason": "unknown_bound_kind_{}".format(kind)}
        dual_violation = max(dual_violation, max(0.0, -mu))
        comp = abs(mu * value)
        comp_inf = max(comp_inf, comp)
        bound_checks.append({
            "name": stored.get("name"),
            "kind": kind,
            "variable_index": idx,
            "recomputed_value": value,
            "stored_value": float(stored.get("value", float("nan"))),
            "value_match": abs(value - float(stored.get("value", float("nan")))) <= 1e-8,
            "multiplier": mu,
            "complementarity": float(comp),
        })

    soc_checks = []
    for stored in kkt.get("soc_duals", []):
        name = str(stored.get("name"))
        kind = str(stored.get("kind"))
        alpha = float(stored.get("alpha", 0.0))
        beta = np.asarray(stored.get("beta", []), dtype=np.float64)
        y_stored = np.asarray(stored.get("y", []), dtype=np.float64)
        if beta.size != problem.n or y_stored.size != problem.n:
            return {"ok": False, "reason": "soc_vector_size_mismatch_{}".format(name)}
        if kind == "row_trust":
            row_idx = int(name.split("_")[-1])
            y = g[row_idx].copy() - problem.g0[row_idx]
            y[row_idx] = 0.0
            t = float(problem.row_radius)
            for a, b in problem.pairs:
                k = problem.pair_index[(a, b)]
                if a == row_idx:
                    stationarity[k] -= beta[b]
                if b == row_idx:
                    stationarity[k] -= beta[a]
        elif kind == "class_mean_trust":
            rows = [int(v) for v in stored.get("rows", [])]
            if not rows:
                return {"ok": False, "reason": "missing_class_rows_{}".format(name)}
            rows_np = np.asarray(rows, dtype=np.int64)
            row_set = set(rows)
            y = (g[rows_np] - problem.g0[rows_np]).mean(axis=0)
            t = float(problem.class_radius)
            m = float(len(rows))
            for a, b in problem.pairs:
                k = problem.pair_index[(a, b)]
                if a in row_set:
                    stationarity[k] -= beta[b] / m
                if b in row_set:
                    stationarity[k] -= beta[a] / m
        else:
            return {"ok": False, "reason": "unknown_soc_kind_{}".format(kind)}
        cone_margin = float(alpha - np.linalg.norm(beta))
        value = float(t - np.linalg.norm(y))
        comp = float(abs(alpha * t + float(beta @ y)))
        dual_violation = max(dual_violation, max(0.0, -cone_margin, -alpha))
        comp_inf = max(comp_inf, comp)
        soc_checks.append({
            "name": name,
            "kind": kind,
            "value": value,
            "stored_value": float(stored.get("value", float("nan"))),
            "value_match": abs(value - float(stored.get("value", float("nan")))) <= 1e-8,
            "y_match": bool(np.max(np.abs(y - y_stored)) <= 1e-8),
            "alpha": alpha,
            "cone_margin": cone_margin,
            "complementarity": comp,
        })

    psd_stored = kkt.get("psd", {})
    s_psd = np.asarray(psd_stored.get("S", []), dtype=np.float64)
    if s_psd.shape != (problem.n, problem.n):
        return {"ok": False, "reason": "psd_S_shape_mismatch"}
    for idx, (i, j) in enumerate(problem.pairs):
        stationarity[idx] -= 2.0 * float(s_psd[i, j])
    s_eigs = np.linalg.eigvalsh(0.5 * (s_psd + s_psd.T))
    dual_violation = max(dual_violation, max(0.0, -float(s_eigs.min())))
    sg = s_psd @ g
    trace_sg = float(np.sum(s_psd * g))
    comp_inf = max(comp_inf, abs(trace_sg))
    g_eigs = np.linalg.eigvalsh(0.5 * (g + g.T))
    psd_check = {
        "active": bool(psd_stored.get("active")),
        "S_sha256": hobj(s_psd.tolist()),
        "stored_S_sha256": psd_stored.get("S_sha256"),
        "S_hash_match": hobj(s_psd.tolist()) == psd_stored.get("S_sha256"),
        "S_eig_min": float(s_eigs.min()),
        "G_eig_min": float(g_eigs.min()),
        "SG_fro": float(np.linalg.norm(sg)),
        "trace_SG": trace_sg,
        "trace_SG_abs": abs(trace_sg),
        "inactive_S_zero_ok": bool((not psd_stored.get("active")) and np.linalg.norm(s_psd) <= 1e-12 and float(g_eigs.min()) > ACTIVE_TOL),
        "dual_feasible": bool(float(s_eigs.min()) >= -1e-8),
    }
    stationarity_inf = float(np.max(np.abs(stationarity))) if stationarity.size else 0.0
    vi_bound = float(stationarity_inf + comp_inf + dual_violation)
    eq_res = problem.eq_a @ x - problem.eq_b
    all_value_matches = (
        all(row["value_match"] for row in linear_checks)
        and all(row["value_match"] for row in bound_checks)
        and all(row["value_match"] and row["y_match"] for row in soc_checks)
        and psd_check["S_hash_match"]
    )
    ok = bool(
        all_value_matches
        and (float(np.max(np.abs(eq_res))) if eq_res.size else 0.0) <= 1e-8
        and dual_violation <= 1e-8
        and comp_inf <= COMPLEMENTARITY_TOL
        and stationarity_inf <= PHASE2_VI_TOL
        and vi_bound <= PHASE2_VI_TOL
        and psd_check["dual_feasible"]
    )
    return {
        "ok": ok,
        "objective": objective,
        "eq_residual_inf": float(np.max(np.abs(eq_res))) if eq_res.size else 0.0,
        "stationarity_inf": stationarity_inf,
        "stationarity_sha256": hobj(stationarity.tolist()),
        "stored_stationarity_sha256": kkt.get("stationarity_sha256"),
        "stationarity_hash_match": hobj(stationarity.tolist()) == kkt.get("stationarity_sha256"),
        "dual_violation": dual_violation,
        "complementarity_inf": comp_inf,
        "vi_residual_bound": vi_bound,
        "linear_checks": linear_checks,
        "bound_checks": bound_checks,
        "soc_checks": soc_checks,
        "psd_check": psd_check,
        "all_value_matches": all_value_matches,
    }


def replay_phase_i(certificate: dict[str, Any], cells: list[dict[str, Any]], fixture: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    cell_by_index = {int(cell["cell_index"]): cell for cell in cells}
    for row in certificate.get("phase_i", []):
        cell_index = int(row.get("cell_index"))
        problem = build_problem(fixture, cfg, cell_by_index[cell_index])
        selected = row.get("selected")
        if not isinstance(selected, dict) or not isinstance(selected.get("witness"), dict):
            out.append({
                "cell_index": cell_index,
                "source_status": row.get("status"),
                "replay_status": "NO_SELECTED_WITNESS",
                "compatibility_replay_ok": False,
                "incompatibility_claim": False,
            })
            continue
        g = np.asarray(selected["witness"]["g"], dtype=np.float64)
        xi = np.asarray(selected["witness"]["xi"], dtype=np.float64)
        rows, groups = residual_sets_original(problem, g, xi)
        signed = signed_gap_residuals(problem, g)
        top20 = canonical_top20(g, problem.ids, float(problem.solver["tie_tolerance"]))
        eig_min = float(np.linalg.eigvalsh(0.5 * (g + g.T)).min())
        ok = bool(
            row.get("status") == "FULL_RANK_SIGNED_CELL_REPLAY_PENDING"
            and max_residual(rows) <= PHASE1_RESIDUAL_TOL
            and eig_min > FULL_RANK_TOL
            and signed["pass"] is True
            and top20 == problem.final_top20
        )
        out.append({
            "cell_index": cell_index,
            "source_status": row.get("status"),
            "max_589_original_residual": max_residual(rows),
            "original_residual_groups": groups,
            "signed_gap_summary": {k: v for k, v in signed.items() if k != "rows"},
            "psd_min_eigenvalue": eig_min,
            "top20_equal_cell": top20 == problem.final_top20,
            "top20_sha256": hobj(top20),
            "target_top20_sha256": hobj(problem.final_top20),
            "compatibility_replay_ok": ok,
            "incompatibility_claim": False,
            "nonconvergence_is_not_infeasibility": row.get("status") == "NO_COMPATIBILITY_WITNESS_NO_FARKAS",
        })
    return out


def replay_phase_ii(certificate: dict[str, Any], cells: list[dict[str, Any]], fixture: dict[str, Any], cfg: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    cell_by_index = {int(cell["cell_index"]): cell for cell in cells}
    for row in certificate.get("phase_ii", []):
        cell_index = int(row.get("cell_index"))
        problem = build_problem(fixture, cfg, cell_by_index[cell_index])
        g = np.asarray(row["witness"]["g"], dtype=np.float64)
        xi = np.asarray(row["witness"]["xi"], dtype=np.float64)
        x = np.concatenate([[float(g[i, j]) for i, j in problem.pairs], xi.astype(np.float64)])
        rows, groups = residual_sets_original(problem, g, xi)
        signed = signed_gap_residuals(problem, g)
        top20 = canonical_top20(g, problem.ids, float(problem.solver["tie_tolerance"]))
        objective, _ = objective_value_grad(problem, x)
        kkt = replay_stationarity(problem, x, row.get("kkt", {}))
        local_ok = bool(
            row.get("status") == "LOCAL_STATIONARY_CERTIFIED_CANDIDATE_REPLAY_PENDING"
            and abs(objective - float(row.get("objective", float("inf")))) <= 1e-8
            and max_residual(rows) <= float(problem.solver["dykstra_set_violation_tolerance"])
            and signed["pass"] is True
            and top20 == problem.final_top20
            and kkt.get("ok") is True
        )
        failure_replayed = bool(row.get("status") == "PRIMAL_DUAL_CERTIFICATE_FAILURE_PIVOT" and not local_ok)
        out.append({
            "cell_index": cell_index,
            "source_status": row.get("status"),
            "objective": objective,
            "stored_objective": row.get("objective"),
            "objective_matches": abs(objective - float(row.get("objective", float("inf")))) <= 1e-8,
            "max_589_original_residual": max_residual(rows),
            "original_residual_groups": groups,
            "signed_gap_summary": {k: v for k, v in signed.items() if k != "rows"},
            "top20_equal_cell": top20 == problem.final_top20,
            "kkt_replay": kkt,
            "local_stationary_replay_ok": local_ok,
            "certificate_failure_replayed": failure_replayed,
        })
    return out


def main() -> int:
    cert_env = os.environ.get("V7_CERTIFICATE_PATH")
    if not cert_env:
        raise RuntimeError("V7_CERTIFICATE_PATH is required")
    cert_path = ROOT / cert_env if not cert_env.startswith("/") else Path(cert_env)
    certificate = read_json(cert_path)
    design = load_design()
    cfg, oriented, fixture, witness, replay = load_inputs()
    system, cells = build_cells(fixture, cfg)
    payload_ok = payload_hash(certificate) == certificate.get("payload_sha256")
    cell_hashes_ok = (
        hobj(cells) == hobj(certificate.get("compatible_cells", []))
        and hobj(system) == hobj(certificate.get("orientation_system", {}))
    )
    hashes_unchanged = existing_hashes_unchanged()
    source_current = current_source_hashes()
    source_match_rows = []
    for key, stored in certificate.get("source_hashes", {}).items():
        if key in source_current:
            source_match_rows.append({"name": key, "stored": stored, "current": source_current[key], "ok": stored == source_current[key]})
    source_hashes_match = all(row["ok"] for row in source_match_rows)
    phase_i = replay_phase_i(certificate, cells, fixture, cfg)
    phase_ii = replay_phase_ii(certificate, cells, fixture, cfg)
    phase_i_structural_ok = all(
        row.get("compatibility_replay_ok") or row.get("nonconvergence_is_not_infeasibility")
        for row in phase_i
    )
    phase_ii_structural_ok = all(
        row.get("local_stationary_replay_ok") or row.get("certificate_failure_replayed")
        for row in phase_ii
    )
    local_certified = (
        certificate.get("status") == "LOCAL_STATIONARY_CERTIFIED_CANDIDATE_REPLAY_PENDING"
        and phase_i
        and phase_ii
        and all(row.get("compatibility_replay_ok") for row in phase_i)
        and all(row.get("local_stationary_replay_ok") for row in phase_ii)
        and len(phase_ii) == len(cells)
    )
    pivot_triggered = (
        certificate.get("status") == "PIVOT_TRIGGERED_CERTIFICATE_FAILURE_REPLAY_PENDING"
        and phase_i_structural_ok
        and phase_ii_structural_ok
    )
    ok = bool(
        payload_ok
        and cell_hashes_ok
        and source_hashes_match
        and hashes_unchanged.get("ok")
        and supervision_contract(cfg).get("ok")
        and (local_certified or pivot_triggered)
    )
    status = "REPLAY_OK_LOCAL_STATIONARY_CERTIFIED" if ok and local_certified else (
        "REPLAY_OK_PIVOT_TRIGGERED" if ok and pivot_triggered else "REPLAY_FAIL"
    )
    out = {
        "schema_version": 1,
        "task": "lb_scgp_g0_v7_independent_replay",
        "thread_session": design.get("thread_session"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "no_slurm_job"),
        "python": sys.version,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "certificate_path": str(cert_path.relative_to(ROOT)),
        "certificate_payload_ok": payload_ok,
        "design_sha256": hfile(DESIGN),
        "eta": ETA,
        "tau": TAU,
        "signed_gap_rhs": SIGNED_RHS,
        "cell_hashes_ok": cell_hashes_ok,
        "source_hashes_match": source_hashes_match,
        "source_match_rows": source_match_rows,
        "existing_hashes_unchanged": hashes_unchanged,
        "supervision_boundary_ok": supervision_contract(cfg).get("ok"),
        "orientation_system": system,
        "phase_i_replay": phase_i,
        "phase_ii_replay": phase_ii,
        "local_certified": local_certified,
        "pivot_triggered": pivot_triggered,
        "status": status,
        "ok": ok,
        "nonclaims": [
            "No G0 PASS, freeze, formal gate, realfold, G1, performance, validation, test, teacher, MLLM, OCR, or segment-level claim.",
            "Pivot-triggered replay is not an incompatibility proof unless a replayed Farkas/conic certificate is present.",
        ],
    }
    out["payload_sha256"] = payload_hash(out)
    out_path = OUT_DIR / "v7_independent_replay_{}.json".format(os.environ.get("SLURM_JOB_ID", "no_slurm_job"))
    write_json_exclusive(out_path, out)
    print(cjson({"status": status, "path": str(out_path.relative_to(ROOT)), "payload_sha256": out["payload_sha256"]}))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
