#!/usr/bin/env python3
"""Independent mpmath replay for the v6 analytic feasibility witness."""

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


def margin_data(g0: np.ndarray, labels: np.ndarray, ids: list[str],
                full_rankings: list[list[int]], tolerance: float) -> dict[str, Any]:
    baseline_full = stable_rankings(g0, ids, topk=len(ids) - 1, tolerance=tolerance)
    cell_top, cell_coeff = rank_coefficients(labels, full_rankings)
    base_top, base_coeff = rank_coefficients(labels, baseline_full)
    baseline_margins = np.asarray(
        [float(base_coeff[i] @ g0[i, base_top[i]]) for i in range(len(labels))],
        dtype=np.float64,
    )
    return {"top": cell_top, "coeff": cell_coeff, "baseline_margins": baseline_margins}


def residuals_np(g: np.ndarray, xi: np.ndarray, witness_doc: dict[str, Any],
                 full_rankings: list[list[int]], cfg: dict[str, Any]) -> dict[str, float]:
    solver = cfg["solver"]
    fixture = witness_doc["frozen_fixture"]
    n = int(fixture["n"])
    labels = np.asarray(fixture["labels"], dtype=np.int64)
    ids = [str(x) for x in fixture["ids"]]
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    ell = np.asarray(fixture["ell"], dtype=np.float64)
    semantic = semantic_matrix(fixture["semantic"], n)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    md = margin_data(g0, labels, ids, full_rankings, solver["tie_tolerance"])
    margins = np.einsum("ir,ir->i", md["coeff"], g[np.arange(n)[:, None], md["top"]], optimize=True)
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


def residuals_mpmath(g: np.ndarray, xi: np.ndarray, witness_doc: dict[str, Any],
                     full_rankings: list[list[int]], cfg: dict[str, Any]) -> dict[str, Any]:
    try:
        import mpmath as mp
    except Exception as exc:
        return {"available": False, "error": "{}: {}".format(type(exc).__name__, str(exc))}
    mp.mp.dps = 100
    fixture = witness_doc["frozen_fixture"]
    solver = cfg["solver"]
    n = int(fixture["n"])
    labels = np.asarray(fixture["labels"], dtype=np.int64)
    ids = [str(x) for x in fixture["ids"]]
    g0 = np.asarray(fixture["gram0"], dtype=np.float64)
    ell = np.asarray(fixture["ell"], dtype=np.float64)
    semantic = semantic_matrix(fixture["semantic"], n)
    groups = [np.flatnonzero(labels == value) for value in (0, 1)]
    md = margin_data(g0, labels, ids, full_rankings, solver["tie_tolerance"])
    deficits = np.maximum(ell - md["baseline_margins"], 0.0)
    caps = [solver["slack_budget_ratio"] * float(deficits[rows].sum()) for rows in groups]
    g_mp = [[mp.mpf(repr(float(g[i, j]))) for j in range(n)] for i in range(n)]
    xi_mp = [mp.mpf(repr(float(v))) for v in xi]
    sem = [mp.mpf(repr(float(v))) for v in semantic.reshape(-1)]
    flat_g = [mp.mpf(repr(float(v))) for v in g.reshape(-1)]
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
    slack = mp.mpf("0")
    for k, rows in enumerate(groups):
        slack = max(slack, -min([xi_mp[int(i)] for i in rows]))
        slack = max(slack, mp.fsum([xi_mp[int(i)] for i in rows]) - mp.mpf(repr(float(caps[k]))))
    class_margin = mp.mpf("0")
    for rows in groups:
        base = mp.fsum([mp.mpf(repr(float(md["baseline_margins"][i]))) for i in rows]) / len(rows)
        got = mp.fsum([margins[i] for i in rows]) / len(rows)
        class_margin = max(class_margin, base - got)
    global_margin = (
        mp.fsum([mp.mpf(repr(float(v))) for v in md["baseline_margins"]]) / n
        - mp.fsum(margins) / n
    )
    try:
        sym = mp.matrix([[mp.mpf("0.5") * (g_mp[i][j] + g_mp[j][i]) for j in range(n)] for i in range(n)])
        eigvals = mp.eigsy(sym, eigvals_only=True)
        psd = max(mp.mpf("0"), -min(eigvals))
        psd_note = "mpmath_eigsy_100dps"
    except Exception as exc:
        psd = mp.mpf(repr(float(residuals_np(g, xi, witness_doc, full_rankings, cfg)["psd"])))
        psd_note = "numpy_fallback_after_{}:{}".format(type(exc).__name__, str(exc))
    selected = {
        "semantic": semantic_value,
        "slack_nonnegative_budget": max(mp.mpf("0"), slack),
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


def resolve_witness_path() -> Path:
    env_path = os.environ.get("ANALYTIC_WITNESS_PATH")
    if env_path:
        path = Path(env_path)
        return path if path.is_absolute() else ROOT / path
    candidates = sorted(OUT_DIR.glob("analytic_feasibility_witness_*.json"), key=lambda p: p.stat().st_mtime)
    if not candidates:
        raise RuntimeError("ANALYTIC_WITNESS_PATH not set and no analytic_feasibility_witness_*.json exists")
    return candidates[-1]


def supervision_ok(boundary: dict[str, Any]) -> bool:
    required_zero = (
        "mllm_call_count",
        "ocr_call_count",
        "teacher_cache_read_count",
        "teacher_cache_write_count",
        "outer_held_label_read_count",
        "outer_held_content_read_count",
        "val_content_read_count",
        "test_content_read_count",
        "val_test_teacher_artifact_count",
    )
    return (
        boundary.get("only_gold_supervision") == "parent_video_binary_label"
        and boundary.get("segment_gold_exists") is False
        and boundary.get("segment_gold_used") is False
        and all(int(boundary.get(k, -1)) == 0 for k in required_zero)
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    witness_path = resolve_witness_path()
    doc = read_json(witness_path)
    cfg = read_json(V5_CONFIG)
    selected = None
    det = doc.get("determination", {})
    for result in doc.get("raw_results", []):
        if (
            det.get("selected_cell_index") == result.get("cell_index")
            and det.get("selected_rank") == result.get("rank")
            and det.get("selected_start_name") == result.get("start_name")
        ):
            selected = result
            break
    replay = None
    accepted = False
    if selected and isinstance(selected.get("witness"), dict):
        witness = selected["witness"]
        g = np.asarray(witness["g"], dtype=np.float64)
        xi = np.asarray(witness["xi"], dtype=np.float64)
        cells = {int(row["cell_index"]): row for row in doc.get("compatible_cells", [])}
        cell = cells[int(selected["cell_index"])]
        full_rankings = cell["full_rankings"]
        np_res = residuals_np(g, xi, doc, full_rankings, cfg)
        mp_res = residuals_mpmath(g, xi, doc, full_rankings, cfg)
        ids = [str(x) for x in doc["frozen_fixture"]["ids"]]
        realized = [row[:20] for row in stable_rankings(g, ids, topk=len(ids) - 1, tolerance=cfg["solver"]["tie_tolerance"])]
        target = [row[:20] for row in full_rankings]
        replay_hash = hobj({"g": witness["g"], "xi": witness["xi"]})
        max_np = max(v for k, v in np_res.items() if k != "psd_min_eigenvalue")
        accepted = bool(
            validate_payload(doc)
            and supervision_ok(doc.get("supervision_boundary", {}))
            and replay_hash == witness.get("witness_sha256")
            and max_np <= cfg["solver"]["dykstra_set_violation_tolerance"]
            and realized == target
            and isinstance(mp_res, dict)
            and mp_res.get("available") is True
            and float(mp_res.get("max_selected_residual", float("inf"))) <= cfg["solver"]["dykstra_set_violation_tolerance"]
        )
        replay = {
            "source_status": selected.get("status"),
            "cell_index": selected.get("cell_index"),
            "rank": selected.get("rank"),
            "start_name": selected.get("start_name"),
            "witness_sha256": replay_hash,
            "witness_sha256_matches": replay_hash == witness.get("witness_sha256"),
            "residuals": np_res,
            "max_residual": float(max_np),
            "mpmath_residuals": mp_res,
            "realized_top20_equal_cell": realized == target,
            "realized_top20_sha256": hobj(realized),
            "target_top20_sha256": hobj(target),
            "stationarity_diagnostics": {
                "source": selected.get("stationarity_diagnostics"),
                "replay_note": "Replay acceptance uses primal feasibility and top20 checks only; stationarity is reported separately.",
            },
        }
    acceptance_label = "accepted_feasible_replayed" if accepted else "not_accepted_no_replayed_feasible_witness"
    out = {
        "schema_version": 1,
        "task": "lb_scgp_v6_analytic_witness_replay",
        "slurm_job_id": job_id,
        "python": sys.version,
        "platform": platform.platform(),
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "analytic_witness_path": str(witness_path.relative_to(ROOT)),
        "analytic_witness_sha256": hfile(witness_path),
        "analytic_witness_payload_ok": validate_payload(doc),
        "acceptance_label": acceptance_label,
        "accepted_feasible_replayed": accepted,
        "selected_replay": replay,
        "supervision_boundary_ok": supervision_ok(doc.get("supervision_boundary", {})),
        "source_hashes": {
            "analytic_witness_replay.py": hfile(V6 / "runtime" / "analytic_witness_replay.py"),
            "analytic_feasibility_witness.py": hfile(V6 / "runtime" / "analytic_feasibility_witness.py"),
            "v5_config": hfile(V5_CONFIG),
        },
        "limitations": [
            "This is a numerical witness replay, not a mathematical infeasibility proof.",
            "A missing accepted witness is nonconvergence/no-witness, not infeasibility.",
        ],
    }
    out["payload_sha256"] = hobj({k: v for k, v in out.items() if k != "payload_sha256"})
    out_path = OUT_DIR / "analytic_witness_replay_{}.json".format(job_id)
    write_json_exclusive(out_path, out)
    print(cjson({
        "status": "OK",
        "path": str(out_path),
        "acceptance_label": acceptance_label,
        "payload_sha256": out["payload_sha256"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
