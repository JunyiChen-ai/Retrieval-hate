#!/usr/bin/env python3
"""Independent replay of LB-SCGP v6 serialized primal witnesses."""

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


def write_json_exclusive(path: Path, obj: Any) -> None:
    with path.open("xb") as handle:
        handle.write((cjson(obj) + "\n").encode())


def validate_payload(obj: dict[str, Any]) -> bool:
    expected = obj.get("payload_sha256")
    if not expected:
        return False
    return hobj({k: v for k, v in obj.items() if k != "payload_sha256"}) == expected


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


def margin_data(gram0: np.ndarray, labels: np.ndarray, ids: list[str],
                full_rankings: list[list[int]], tolerance: float) -> dict[str, Any]:
    baseline_full = stable_rankings(gram0, ids, topk=len(ids) - 1, tolerance=tolerance)
    cell_top, cell_coeff = rank_coefficients(labels, full_rankings)
    base_top, base_coeff = rank_coefficients(labels, baseline_full)
    return {
        "top": cell_top,
        "coeff": cell_coeff,
        "baseline_margins": np.asarray(
            [float(base_coeff[i] @ gram0[i, base_top[i]]) for i in range(len(labels))],
            dtype=np.float64,
        ),
    }


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
    rank_v = 0.0
    for row in rank_halfspaces(ids, full_rankings, solver["tie_tolerance"]):
        rank_v = max(rank_v, float(row["rhs"]) - float(g[row["query"], row["a"]] - g[row["query"], row["b"]]))
    centroid = centroid_direction(labels)
    eig_min = float(np.linalg.eigvalsh(0.5 * (g + g.T)).min())
    return {
        "symmetry": float(np.max(np.abs(g - g.T))),
        "unit_diagonal": float(np.max(np.abs(np.diag(g) - 1.0))),
        "psd": max(0.0, -eig_min),
        "psd_min_eigenvalue": eig_min,
        "offdiag_box": max(0.0, float(off.max() - solver["offdiag_upper"]), float(-1.0 - off.min())),
        "row_trust": max(
            max(0.0, float(np.linalg.norm(np.delete(g[i] - g0[i], i)) - solver["row_trust_scale"] * math.sqrt(n - 1)))
            for i in range(n)
        ),
        "class_mean_trust": max(
            max(0.0, float(np.linalg.norm((g[rows] - g0[rows]).mean(axis=0)) - solver["class_mean_trust_scale"] * math.sqrt(n)))
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
        "centroid": max(0.0, float(np.sum(centroid * g0) - np.sum(centroid * g))),
        "rank_halfspaces": max(0.0, rank_v),
    }


def residuals_mpmath(g: np.ndarray, xi: np.ndarray, fixture: dict[str, Any],
                     full_rankings: list[list[int]], cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        import mpmath as mp
    except Exception as exc:
        return {"available": False, "error": "{}: {}".format(type(exc).__name__, str(exc))}
    mp.mp.dps = 80
    solver = cfg["solver"]
    n = int(fixture["n"])
    labels = np.asarray(fixture["labels"], dtype=np.int64)
    ids = [str(x) for x in fixture["ids"]]
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    ell = np.asarray(fixture["ell"], dtype=np.float64)
    semantic = np.asarray(fixture["semantic"], dtype=np.float64)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    md = margin_data(g0, labels, ids, full_rankings, solver["tie_tolerance"])

    g_mp = [[mp.mpf(repr(float(g[i, j]))) for j in range(n)] for i in range(n)]
    xi_mp = [mp.mpf(repr(float(v))) for v in xi]
    flat_g = [mp.mpf(repr(float(v))) for v in g.reshape(-1)]
    sem = [mp.mpf(repr(float(v))) for v in semantic.reshape(-1)]
    semantic_value = abs(mp.fsum([sem[k] * flat_g[k] for k in range(len(flat_g))]))
    margins = []
    for i in range(n):
        margins.append(mp.fsum([
            mp.mpf(repr(float(md["coeff"][i, r]))) * g_mp[i][int(md["top"][i, r])]
            for r in range(20)
        ]))
    rank_v = mp.mpf("0")
    for row in rank_halfspaces(ids, full_rankings, solver["tie_tolerance"]):
        lhs = g_mp[row["query"]][row["a"]] - g_mp[row["query"]][row["b"]]
        rank_v = max(rank_v, mp.mpf(repr(float(row["rhs"]))) - lhs)
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
        psd = mp.mpf(repr(float(residuals_np(g, xi, fixture, full_rankings, cfg)["psd"])))
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


def replay_one(result: dict[str, Any], fixture: dict[str, Any],
               full_rankings: list[list[int]], cfg: dict[str, Any]) -> dict[str, Any]:
    if not result.get("witness"):
        return {
            "solver_path": result.get("solver_path"),
            "cell_index": result.get("cell_index"),
            "status": "no_witness",
            "source_status": result.get("status"),
            "source_error": result.get("error"),
        }
    witness = result["witness"]
    g = np.asarray(witness["g"], dtype=np.float64)
    xi = np.asarray(witness["xi"], dtype=np.float64)
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    ids = [str(x) for x in fixture["ids"]]
    n = int(fixture["n"])
    pairs_objective = float(0.5 * np.sum((g - g0) ** 2) + 0.5 * np.sum(xi ** 2))
    res = residuals_np(g, xi, fixture, full_rankings, cfg)
    mp_res = residuals_mpmath(g, xi, fixture, full_rankings, cfg)
    realized_full = stable_rankings(g, ids, topk=n - 1, tolerance=cfg["solver"]["tie_tolerance"])
    target_top20 = [row[:20] for row in full_rankings]
    realized_top20 = [row[:20] for row in realized_full]
    max_residual = max(v for k, v in res.items() if k != "psd_min_eigenvalue")
    replay_hash = hobj({"g": witness["g"], "xi": witness["xi"]})
    return {
        "solver_path": result.get("solver_path"),
        "cell_index": result.get("cell_index"),
        "source_status": result.get("status"),
        "source_success": result.get("success"),
        "witness_sha256": replay_hash,
        "witness_sha256_matches": replay_hash == witness.get("witness_sha256"),
        "objective_replay": pairs_objective,
        "objective_source": witness.get("objective"),
        "objective_abs_delta": abs(pairs_objective - float(witness.get("objective", float("nan")))),
        "residuals": res,
        "max_residual": float(max_residual),
        "source_max_residual": result.get("max_residual"),
        "max_residual_abs_delta": None if result.get("max_residual") is None else abs(float(result["max_residual"]) - float(max_residual)),
        "mpmath_residuals": mp_res,
        "realized_top20_equal_cell": realized_top20 == target_top20,
        "realized_full_equal_cell": realized_full == full_rankings,
        "realized_top20_sha256": hobj(realized_top20),
        "target_top20_sha256": hobj(target_top20),
        "full_rankings_sha256": hobj(full_rankings),
        "kkt_or_stationarity_proxy": {
            "source_dual_summary": result.get("dual_summary"),
            "source_solver_stats": result.get("solver_stats"),
            "replay_note": "Replay verifies primal residuals and serialized objective only; it does not create proof-grade KKT evidence.",
        },
        "feasible_under_replay": bool(
            max_residual <= cfg["solver"]["dykstra_set_violation_tolerance"]
            and realized_top20 == target_top20
            and isinstance(mp_res, dict)
            and mp_res.get("available") is True
            and float(mp_res.get("max_selected_residual", float("inf"))) <= cfg["solver"]["dykstra_set_violation_tolerance"]
        ),
    }


def resolve_certificate_path() -> Path:
    env_path = os.environ.get("CERTIFICATE_PATH")
    if env_path:
        path = Path(env_path)
        return path if path.is_absolute() else ROOT / path
    candidates = sorted(OUT_DIR.glob("oriented_certificate_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError("CERTIFICATE_PATH not set and no oriented_certificate_*.json exists")
    return candidates[-1]


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    cert_path = resolve_certificate_path()
    cert = read_json(cert_path)
    cfg = read_json(V5_CONFIG)
    fixture = cert["frozen_fixture"]
    cells = {int(row["cell_index"]): row for row in cert["controller"]["compatible_cells"]}
    replay_rows = []
    for result in cert.get("raw_results", []):
        cell = cells[int(result["cell_index"])]
        replay_rows.append(replay_one(result, fixture, cell["full_rankings"], cfg))
    feasible_rows = [row for row in replay_rows if row.get("feasible_under_replay") is True]
    selected_source = cert.get("determination", {})
    selected_replay = None
    if selected_source.get("selected_solver_path") is not None:
        for row in replay_rows:
            if row.get("solver_path") == selected_source.get("selected_solver_path") and \
                    row.get("cell_index") == selected_source.get("selected_cell_index"):
                selected_replay = row
                break
    out = {
        "schema_version": 1,
        "task": "lb_scgp_v6_witness_replay",
        "slurm_job_id": job_id,
        "python": sys.version,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "certificate_path": str(cert_path.relative_to(ROOT)),
        "certificate_sha256": hfile(cert_path),
        "certificate_payload_ok": validate_payload(cert),
        "certificate_determination": selected_source,
        "replay_rows": replay_rows,
        "feasible_replay_count": len(feasible_rows),
        "local_feasible_cell_exists_replayed": bool(feasible_rows),
        "selected_replay": selected_replay,
        "source_hashes": {
            "witness_replay.py": hfile(V6 / "runtime" / "witness_replay.py"),
            "witness_replay.sbatch": hfile(V6 / "runtime" / "witness_replay.sbatch"),
            "v5_config": hfile(V5_CONFIG),
        },
        "supervision_boundary": cert.get("supervision_boundary"),
        "evidence_strength": "numerical_not_proof_grade",
    }
    out["payload_sha256"] = hobj({k: v for k, v in out.items() if k != "payload_sha256"})
    out_path = OUT_DIR / "witness_replay_{}.json".format(job_id)
    write_json_exclusive(out_path, out)
    print(cjson({
        "status": "OK",
        "path": str(out_path),
        "payload_sha256": out["payload_sha256"],
        "local_feasible_cell_exists_replayed": out["local_feasible_cell_exists_replayed"],
        "feasible_replay_count": out["feasible_replay_count"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
