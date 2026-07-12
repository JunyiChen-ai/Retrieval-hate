#!/usr/bin/env python3
"""Solver-free independent replay for the v6 actual fixture oracle."""

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
OUT_DIR = V6 / "results"
V5_CONFIG = ROOT / "configs" / "lb_scgp" / "lb_scgp_v5.json"
V5_DYKSTRA = ROOT / "artifacts" / "lb_scgp" / "v5" / "g0" / "synthetic" / "dykstra.jsonl"
TOPK = 20
ACTIVE_TOL = 1e-7


def cjson(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hobj(obj: Any) -> str:
    return hashlib.sha256(cjson(obj).encode("utf-8")).hexdigest()


def hfile(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def payload_hash(obj: dict[str, Any]) -> str:
    return hobj({k: v for k, v in obj.items() if k != "payload_sha256"})


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
    out = []
    for i in range(len(ids)):
        candidates = [j for j in range(len(ids)) if j != i]
        local = tolerance_order([float(gram[i, j]) for j in candidates], [ids[j] for j in candidates], tolerance)
        out.append([candidates[k] for k in local[:topk]])
    return out


def canonical_rhs(ids: list[str], a: int, b: int, tolerance: float) -> float:
    return -float(tolerance) if str(ids[a]) < str(ids[b]) else float(np.nextafter(float(tolerance), math.inf))


def rank_halfspaces(ids: list[str], full_rankings: list[list[int]], tolerance: float) -> list[dict[str, Any]]:
    rows = []
    index = 0
    for i, full in enumerate(full_rankings):
        top = full[:TOPK]
        for r in range(TOPK - 1):
            a, b = top[r], top[r + 1]
            rows.append({"name": "rank_internal_{:04d}".format(index), "query": i, "a": a, "b": b, "kind": "internal", "rhs": canonical_rhs(ids, a, b, tolerance)})
            index += 1
        for outsider in full[TOPK:]:
            a, b = top[TOPK - 1], outsider
            rows.append({"name": "rank_boundary_{:04d}".format(index), "query": i, "a": a, "b": b, "kind": "boundary", "rhs": canonical_rhs(ids, a, b, tolerance)})
            index += 1
    return rows


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
    return {"top": cell_top, "coeff": cell_coeff, "baseline_margins": baseline_margins}


def residual_sets(g: np.ndarray, xi: np.ndarray, fixture: dict[str, Any], cfg: dict[str, Any],
                  full_rankings: list[list[int]]) -> tuple[list[dict[str, Any]], dict[str, float]]:
    solver = cfg["solver"]
    n = int(fixture["n"])
    labels = np.asarray(fixture["labels"], dtype=np.int64)
    ids = [str(x) for x in fixture["ids"]]
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    ell = np.asarray(fixture["ell"], dtype=np.float64)
    semantic = semantic_matrix(fixture["semantic"], n)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    md = margin_data(g0, labels, ids, full_rankings, float(solver["tie_tolerance"]))
    row_radius = float(solver["row_trust_scale"]) * math.sqrt(n - 1)
    class_radius = float(solver["class_mean_trust_scale"]) * math.sqrt(n)
    deficits = np.maximum(ell - md["baseline_margins"], 0.0)
    caps = [float(solver["slack_budget_ratio"]) * float(deficits[rows].sum()) for rows in groups]
    centroid = centroid_direction(labels)
    off = g[~np.eye(n, dtype=bool)]
    eig_min = float(np.linalg.eigvalsh(0.5 * (g + g.T)).min())
    margins = np.einsum("ir,ir->i", md["coeff"], g[np.arange(n)[:, None], md["top"]], optimize=True)
    rows = []

    def add(name: str, group: str, residual: float) -> None:
        rows.append({"name": name, "group": group, "residual": float(max(0.0, residual))})

    add("symmetry", "symmetry", float(np.max(np.abs(g - g.T))))
    add("correlation_diagonal", "unit_diagonal", float(np.max(np.abs(np.diag(g) - 1.0))))
    add("psd_symmetrized_input", "psd", -eig_min)
    add("offdiagonal_box", "box", max(float(off.max() - solver["offdiag_upper"]), float(-1.0 - off.min())))
    for i in range(n):
        add("row_trust_{:02d}".format(i), "row_trust", float(np.linalg.norm(np.delete(g[i] - g0[i], i)) - row_radius))
    for k, label in enumerate((0, 1)):
        rr = np.flatnonzero(labels == label)
        add("class_mean_trust_{}".format(k), "class_mean_trust", float(np.linalg.norm((g[rr] - g0[rr]).mean(axis=0)) - class_radius))
    add("semantic_radius_zero", "semantic", float(np.linalg.norm(semantic @ g.reshape(-1))))
    for k, rr in enumerate(groups):
        add("slack_capped_simplex_{}".format(k), "slack", max(float(-xi[rr].min()), float(xi[rr].sum() - caps[k])))
    for i in range(n):
        add("vote_slack_{:02d}".format(i), "vote", float(ell[i] - margins[i] - xi[i]))
    for k, rr in enumerate(groups):
        add("class_mean_margin_{}".format(k), "class_margin", float(md["baseline_margins"][rr].mean() - margins[rr].mean()))
    add("global_mean_margin", "global_margin", float(md["baseline_margins"].mean() - margins.mean()))
    add("centroid_distance", "centroid", float(np.sum(centroid * g0) - np.sum(centroid * g)))
    for row in rank_halfspaces(ids, full_rankings, float(solver["tie_tolerance"])):
        add(row["name"], "rank_halfspaces", float(row["rhs"]) - float(g[row["query"], row["a"]] - g[row["query"], row["b"]]))
    groups_out: dict[str, float] = {}
    for row in rows:
        groups_out[row["group"]] = max(groups_out.get(row["group"], 0.0), float(row["residual"]))
    return rows, groups_out


def max_residual(rows: list[dict[str, Any]]) -> float:
    return float(max((float(row["residual"]) for row in rows), default=float("inf")))


def objective(g: np.ndarray, xi: np.ndarray, fixture: dict[str, Any]) -> float:
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    return 0.5 * float(np.sum((g - g0) ** 2)) + 0.5 * float(np.dot(xi, xi))


def replay_kkt(row: dict[str, Any], fixture: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    # The replay intentionally verifies only stored KKT certificate fields and
    # recomputed primal residual/topology.  It does not solve for multipliers.
    kkt = row.get("kkt", {})
    psd = kkt.get("psd", {})
    ok = bool(
        kkt.get("stationarity_inf", float("inf")) <= 1e-6
        and kkt.get("dual_min", -float("inf")) >= -1e-8
        and kkt.get("complementarity_inf", float("inf")) <= 1e-6
        and kkt.get("active_set_complete") is True
        and kkt.get("psd_dual_ok") is True
    )
    return {
        "ok": ok,
        "stationarity_inf": kkt.get("stationarity_inf"),
        "dual_min": kkt.get("dual_min"),
        "complementarity_inf": kkt.get("complementarity_inf"),
        "active_set_complete": kkt.get("active_set_complete"),
        "psd": psd,
    }


def supervision_ok(cfg: dict[str, Any]) -> bool:
    s = cfg.get("supervision", {})
    c = cfg.get("counters", {})
    return (
        s.get("only_gold_supervision") == "parent_video_binary_label"
        and s.get("segment_gold_exists") is False
        and s.get("segment_gold_used") is False
        and all(int(c.get(k, -1)) == 0 for k in (
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


def mpmath_selected(g: np.ndarray, xi: np.ndarray, fixture: dict[str, Any], cfg: dict[str, Any],
                    full_rankings: list[list[int]]) -> dict[str, Any]:
    try:
        import mpmath as mp
    except Exception as exc:
        return {"available": False, "error": "{}: {}".format(type(exc).__name__, str(exc))}
    mp.mp.dps = 100
    rows, _ = residual_sets(g, xi, fixture, cfg, full_rankings)
    selected = {row["name"]: row["residual"] for row in rows if row["group"] in {"semantic", "rank_halfspaces", "vote", "class_margin", "global_margin", "psd"}}
    try:
        sym = mp.matrix([[mp.mpf(repr(float(0.5 * (g[i, j] + g[j, i])))) for j in range(g.shape[0])] for i in range(g.shape[0])])
        eigvals = mp.eigsy(sym, eigvals_only=True)
        psd = max(mp.mpf("0"), -min(eigvals))
        psd_method = "mpmath_eigsy_100dps"
    except Exception as exc:
        psd = mp.mpf(str(max(0.0, -float(np.linalg.eigvalsh(0.5 * (g + g.T)).min()))))
        psd_method = "numpy_fallback_after_{}:{}".format(type(exc).__name__, str(exc))
    max_sel = max([float(v) for v in selected.values()] + [float(psd)], default=0.0)
    return {"available": True, "dps": 100, "psd": float(psd), "psd_method": psd_method, "max_selected_residual": max_sel}


def main() -> int:
    path = os.environ.get("ACTUAL_ORACLE_PATH")
    if not path:
        raise RuntimeError("ACTUAL_ORACLE_PATH is required")
    oracle_path = ROOT / path if not path.startswith("/") else Path(path)
    oracle = read_json(oracle_path)
    cfg = read_json(V5_CONFIG)
    oriented = next(row for row in read_jsonl(V5_DYKSTRA) if row.get("case") == "feasible_oriented_boundary")
    fixture = oriented["fixture"]
    payload_ok = payload_hash(oracle) == oracle.get("payload_sha256")
    hash_ok = (
        oracle.get("source_hashes", {}).get("v5_config") == hfile(V5_CONFIG)
        and oracle.get("source_hashes", {}).get("v5_dykstra_jsonl") == hfile(V5_DYKSTRA)
    )
    phase_i_replay = []
    for cell in oracle.get("phase_i", []):
        selected = cell.get("selected")
        if not isinstance(selected, dict) or not isinstance(selected.get("witness"), dict):
            phase_i_replay.append({"cell_index": cell.get("cell_index"), "status": "no_selected_witness"})
            continue
        g = np.asarray(selected["witness"]["g"], dtype=np.float64)
        xi = np.asarray(selected["witness"]["xi"], dtype=np.float64)
        full = oracle["compatible_cells"][int(cell["cell_index"])]["full_rankings"]
        rows, groups = residual_sets(g, xi, fixture, cfg, full)
        top20 = [row[:TOPK] for row in stable_rankings(g, [str(x) for x in fixture["ids"]], int(fixture["n"]) - 1, float(cfg["solver"]["tie_tolerance"]))]
        phase_i_replay.append({
            "cell_index": cell.get("cell_index"),
            "source_status": cell.get("status"),
            "max_589_residual": max_residual(rows),
            "residual_groups": groups,
            "psd_min_eigenvalue": float(np.linalg.eigvalsh(0.5 * (g + g.T)).min()),
            "full_rank_replay_ok": bool(
                cell.get("status") == "FULL_RANK_SLATER_REPLAY_PENDING"
                and max_residual(rows) <= 1e-8
                and float(np.linalg.eigvalsh(0.5 * (g + g.T)).min()) > 1e-8
                and top20 == oracle["compatible_cells"][int(cell["cell_index"])]["final_top20_rankings"]
            ),
            "mpmath": mpmath_selected(g, xi, fixture, cfg, full),
        })
    phase_ii_replay = []
    for row in oracle.get("phase_ii", []):
        g = np.asarray(row["witness"]["g"], dtype=np.float64)
        xi = np.asarray(row["witness"]["xi"], dtype=np.float64)
        full = oracle["compatible_cells"][int(row["cell_index"])]["full_rankings"]
        rows, groups = residual_sets(g, xi, fixture, cfg, full)
        top20 = [r[:TOPK] for r in stable_rankings(g, [str(x) for x in fixture["ids"]], int(fixture["n"]) - 1, float(cfg["solver"]["tie_tolerance"]))]
        kkt = replay_kkt(row, fixture, cfg)
        obj = objective(g, xi, fixture)
        phase_ii_replay.append({
            "cell_index": row.get("cell_index"),
            "source_status": row.get("status"),
            "objective": obj,
            "objective_matches": abs(obj - float(row.get("objective", float("inf")))) <= 1e-8,
            "max_589_residual": max_residual(rows),
            "residual_groups": groups,
            "top20_equal_cell": top20 == oracle["compatible_cells"][int(row["cell_index"])]["final_top20_rankings"],
            "kkt": kkt,
            "mpmath": mpmath_selected(g, xi, fixture, cfg, full),
            "local_stationary_replay_ok": bool(
                row.get("status") == "LOCAL_STATIONARY_CERTIFIED_CANDIDATE"
                and abs(obj - float(row.get("objective", float("inf")))) <= 1e-8
                and max_residual(rows) <= 1e-6
                and top20 == oracle["compatible_cells"][int(row["cell_index"])]["final_top20_rankings"]
                and kkt["ok"]
            ),
        })
    all_phase_i_ok = all((r.get("full_rank_replay_ok") or r.get("source_status") == "NO_WITNESS") for r in phase_i_replay)
    all_phase_ii_ok = all((r.get("local_stationary_replay_ok") or r.get("source_status") == "BOUNDED_REMOVE") for r in phase_ii_replay)
    certified = bool(phase_ii_replay) and all(r.get("local_stationary_replay_ok") for r in phase_ii_replay) and len(phase_ii_replay) == len(oracle.get("compatible_cells", []))
    replay_status = "REPLAY_OK_LOCAL_STATIONARY_CERTIFIED" if certified else "REPLAY_OK_BOUNDED_REMOVE"
    ok = bool(payload_ok and hash_ok and supervision_ok(cfg) and int(oracle.get("constraint_set_count", -1)) == 589 and all_phase_i_ok and all_phase_ii_ok)
    out = {
        "schema_version": 1,
        "task": "lb_scgp_v6_actual_fixture_replay",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "no_slurm_job"),
        "python": sys.version,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "oracle_path": str(oracle_path.relative_to(ROOT)),
        "oracle_payload_ok": payload_ok,
        "hash_binding_ok": hash_ok,
        "supervision_boundary_ok": supervision_ok(cfg),
        "constraint_set_count": oracle.get("constraint_set_count"),
        "phase_i_replay": phase_i_replay,
        "phase_ii_replay": phase_ii_replay,
        "status": replay_status if ok else "REPLAY_FAIL_BOUNDED_REMOVE",
        "ok": ok,
        "nonclaims": ["No G0 PASS/freeze/formal/realfold/performance claim.", "NO_WITNESS is nonconvergence only, not infeasibility."],
    }
    out["payload_sha256"] = payload_hash(out)
    out_path = OUT_DIR / "actual_fixture_replay_{}.json".format(os.environ.get("SLURM_JOB_ID", "no_slurm_job"))
    write_json_exclusive(out_path, out)
    print(cjson({"status": out["status"], "path": str(out_path.relative_to(ROOT)), "payload_sha256": out["payload_sha256"]}))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
