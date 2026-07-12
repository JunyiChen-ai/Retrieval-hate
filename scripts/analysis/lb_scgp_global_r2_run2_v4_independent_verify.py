#!/usr/bin/env python
"""Fresh independent semantic verifier for Run2 synthetic KKT.

This verifier intentionally does not import the Run2 producer or common module.
It rebuilds certificate consensus, Q, M_Q, A^T, KKT residuals, source bindings,
rank/factor replay, and fail-closed injections from serialized payloads.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/jehc223/RGCL")
RUN2 = "LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v4"
SCHEMA_ID = "scgp_global_synth_kkt_payload_v4"
TRI_OBSERVABLES = (
    "visual_reference_observable",
    "text_audio_reference_observable",
    "harmful_surface_observable",
    "dehumanizing_or_threat_surface_observable",
    "cross_modal_binding_observable",
    "source_alignment_observable",
    "counter_context_observable",
    "context_shift_observable",
)
TRI_STATES = ("contradicted", "supported", "unresolved")
MODALITY_OBSERVABLE = "modality_binding_observable"
MODALITY_STATES = ("multi_modal", "single_modal", "text_audio", "unresolved", "visual_audio", "visual_text")
REQUIRED_CERT_KEYS = (("schema_version",) + TRI_OBSERVABLES + (MODALITY_OBSERVABLE, "parse_flags"))
TOP_KEYS = {
    "schema_version",
    "artifact_schema_id",
    "run_id",
    "terminal_state",
    "authorized_boundary",
    "no_success_claim",
    "slurm_policy",
    "config_path",
    "source_manifest_path",
    "access_ledger_path",
    "primal",
    "metric",
    "movement_metrics",
    "affine_normals",
    "box_coordinate_normals",
    "soc_normals",
    "psd_normal",
    "halfspace_normals",
    "stationarity",
    "dual_feasibility",
    "complementarity",
    "duality_gap",
    "case_matrix",
    "orth_cap_matrix",
    "rank_failure_probe",
    "schema_fixture_results",
    "injection_results_expected",
    "gold_isolation",
    "dirty_binding",
    "acceptance",
    "hashes",
    "payload_sha256",
}
CASE_KEYS = {
    "case_id",
    "case_role",
    "system",
    "ids",
    "labels",
    "d",
    "replicas",
    "consensus_records",
    "operator",
    "movement_metrics",
    "primal_residuals",
    "rank_audit",
    "factor_replay",
    "robust_coverage",
    "finite_vi_diagnostics",
    "acceptance_path",
    "kkt_status",
    "hashes",
}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_obj(obj: Any) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def sha256_file(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def payload_hash(obj: dict[str, Any]) -> str:
    clone = dict(obj)
    clone.pop("payload_sha256", None)
    return sha256_obj(clone)


def floatify(value: float) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise RuntimeError("non-finite value cannot be serialized")
    if abs(value) < 5e-16:
        return 0.0
    return float(f"{value:.15g}")


def vector_to_list(vec: np.ndarray) -> list[float]:
    return [floatify(x) for x in np.asarray(vec, dtype=np.float64).reshape(-1)]


def matrix_to_list(mat: np.ndarray) -> list[list[float]]:
    arr = np.asarray(mat, dtype=np.float64)
    if arr.ndim != 2:
        raise RuntimeError("expected two dimensional matrix")
    return [[floatify(x) for x in row] for row in arr]


def root_path(path: str | Path) -> Path:
    raw = Path(path)
    candidate = raw if raw.is_absolute() else ROOT / raw
    resolved = candidate.resolve()
    resolved.relative_to(ROOT.resolve())
    return resolved


def read_json(path: str | Path) -> Any:
    with open(root_path(path), encoding="utf-8") as handle:
        return json.load(handle)


def format_schema_errors(errors: list[Any]) -> str:
    rendered = []
    for error in errors[:20]:
        location = "$"
        if getattr(error, "absolute_path", None):
            location = "$." + ".".join(str(item) for item in error.absolute_path)
        rendered.append(f"{location}: {error.message}")
    return "; ".join(rendered)


def validate_manifest_schema(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    try:
        from jsonschema import Draft7Validator, RefResolver
        from jsonschema.exceptions import SchemaError
    except Exception as exc:  # noqa: BLE001 - fail closed on missing validator dependency
        raise RuntimeError("jsonschema dependency unavailable; independent verifier refuses PASS") from exc

    payload_schema_path = root_path(config["paths"]["payload_schema"])
    case_schema_path = root_path(config["paths"]["case_schema"])
    payload_schema = read_json(config["paths"]["payload_schema"])
    case_schema = read_json(config["paths"]["case_schema"])
    try:
        Draft7Validator.check_schema(payload_schema)
        Draft7Validator.check_schema(case_schema)
    except SchemaError as exc:
        raise RuntimeError(f"Run2-v4 JSON Schema is invalid: {exc.message}") from exc

    resolver = RefResolver(
        base_uri=payload_schema_path.parent.as_uri() + "/",
        referrer=payload_schema,
        store={
            case_schema_path.as_uri(): case_schema,
            case_schema_path.name: case_schema,
        },
    )
    payload_errors = sorted(
        Draft7Validator(payload_schema, resolver=resolver).iter_errors(manifest),
        key=lambda error: list(error.absolute_path),
    )
    if payload_errors:
        raise RuntimeError(f"payload schema validation failed: {format_schema_errors(payload_errors)}")

    case_validator = Draft7Validator(case_schema)
    for idx, case in enumerate(manifest.get("case_matrix", {}).get("cases", [])):
        case_errors = sorted(case_validator.iter_errors(case), key=lambda error: list(error.absolute_path))
        if case_errors:
            raise RuntimeError(f"case[{idx}] schema validation failed: {format_schema_errors(case_errors)}")


def finite_array(arr: np.ndarray, name: str) -> None:
    if not np.all(np.isfinite(arr)):
        raise RuntimeError(f"non-finite numeric payload in {name}")


def validate_cert(record: dict[str, Any]) -> None:
    if set(record) != set(REQUIRED_CERT_KEYS):
        raise RuntimeError("certificate key set mismatch")
    if record["schema_version"] != "scgp_global_cert_v2":
        raise RuntimeError("certificate schema version mismatch")
    for field in TRI_OBSERVABLES:
        item = record[field]
        if set(item) != {"state", "confidence"}:
            raise RuntimeError(f"{field} extra/missing keys")
        if item["state"] not in TRI_STATES:
            raise RuntimeError(f"{field} invalid state")
        if not isinstance(item["confidence"], int) or not 0 <= item["confidence"] <= 4:
            raise RuntimeError(f"{field} invalid confidence")
    modality = record[MODALITY_OBSERVABLE]
    if set(modality) != {"state", "confidence"} or modality["state"] not in MODALITY_STATES:
        raise RuntimeError("modality observable invalid")
    if not isinstance(modality["confidence"], int) or not 0 <= modality["confidence"] <= 4:
        raise RuntimeError("modality confidence invalid")
    if not isinstance(record["parse_flags"], list) or any(not isinstance(flag, str) for flag in record["parse_flags"]):
        raise RuntimeError("parse_flags invalid")


def consensus(group: list[dict[str, Any]]) -> dict[str, Any]:
    for record in group:
        validate_cert(record)
    out: dict[str, Any] = {"schema_version": "scgp_global_cert_v2"}
    for field in TRI_OBSERVABLES:
        counts = {state: 0 for state in TRI_STATES}
        for record in group:
            counts[record[field]["state"]] += 1
        best = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        out[field] = {"state": best[0] if best[1] > len(group) / 2 else "unresolved", "confidence": 0}
    counts = {state: 0 for state in MODALITY_STATES}
    for record in group:
        counts[record[MODALITY_OBSERVABLE]["state"]] += 1
    best = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    out[MODALITY_OBSERVABLE] = {"state": best[0] if best[1] > len(group) / 2 else "unresolved", "confidence": 0}
    out["parse_flags"] = []
    validate_cert(out)
    return out


def encode(record: dict[str, Any]) -> np.ndarray:
    validate_cert(record)
    values = []
    for field in TRI_OBSERVABLES:
        values.extend(1.0 if record[field]["state"] == state else 0.0 for state in TRI_STATES)
    values.extend(1.0 if record[MODALITY_OBSERVABLE]["state"] == state else 0.0 for state in MODALITY_STATES)
    return np.asarray(values, dtype=np.float64)


def row_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise RuntimeError("zero feature row")
    return x / norms


def orth_cap(phi: np.ndarray, ids: list[str], cap: int = 8) -> tuple[np.ndarray, int]:
    centered = phi - phi.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s.size == 0:
        return np.zeros((phi.shape[0], 0), dtype=np.float64), 0
    threshold = max(1e-8, 1e-7 * float(s[0]))
    raw_rank = int(np.sum(s > threshold))
    q = u[:, : min(cap, raw_rank)].copy()
    for col in range(q.shape[1]):
        pivot = max(range(q.shape[0]), key=lambda row: (abs(float(q[row, col])), str(ids[row])))
        if q[pivot, col] < 0:
            q[:, col] *= -1.0
    return q, raw_rank


def vech(mat: np.ndarray) -> np.ndarray:
    rows, cols = np.tril_indices(mat.shape[0])
    return np.asarray(mat)[rows, cols]


def unvech_dual(vec: np.ndarray, rank: int) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    if vec.size != rank * (rank + 1) // 2:
        raise RuntimeError("nu length does not match Q rank")
    out = np.zeros((rank, rank), dtype=np.float64)
    rows, cols = np.tril_indices(rank)
    for value, row, col in zip(vec, rows, cols):
        if row == col:
            out[row, col] = value
        else:
            out[row, col] = 0.5 * value
            out[col, row] = 0.5 * value
    return out


def structural_moment(q: np.ndarray, gram: np.ndarray) -> np.ndarray:
    if q.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    n = gram.shape[0]
    out = q.T @ (gram - np.eye(n)) @ q / float(n)
    return 0.5 * (out + out.T)


def structural_adjoint(q: np.ndarray, nu: np.ndarray) -> np.ndarray:
    if q.shape[1] == 0:
        return np.zeros((q.shape[0], q.shape[0]), dtype=np.float64)
    s = unvech_dual(nu, q.shape[1])
    out = q @ s @ q.T / float(q.shape[0])
    return 0.5 * (out + out.T)


def psd_gram_from_features(features: np.ndarray) -> np.ndarray:
    v = row_normalize(features)
    gram = v @ v.T
    gram = 0.5 * (gram + gram.T)
    np.fill_diagonal(gram, 1.0)
    off = gram - np.eye(gram.shape[0], dtype=np.float64)
    max_abs = np.max(np.abs(off)) if off.size else 0.0
    if max_abs >= 0.999:
        gram = 0.94 * gram + 0.06 * np.eye(gram.shape[0], dtype=np.float64)
        np.fill_diagonal(gram, 1.0)
    return gram


def deterministic_baseline_gram(n: int, d: int = 4) -> np.ndarray:
    rows = []
    for idx in range(n):
        rows.append([math.cos((idx + 1) * (j + 1)) + 0.2 * (idx == j) for j in range(d)])
    return psd_gram_from_features(np.asarray(rows, dtype=np.float64))


def deterministic_rank_d_features(n: int, d: int, case_id: str, system: str) -> np.ndarray:
    if d < 1:
        raise RuntimeError("rank dimension must be positive")
    phase = (sum(ord(ch) for ch in f"{case_id}:{system}") % 997) / 997.0
    if d == 1:
        rows = [[1.0 if idx % 2 == 0 else -1.0] for idx in range(n)]
    elif d == 2:
        rows = []
        for idx in range(n):
            theta = 2.0 * math.pi * ((idx + 0.5) / max(n, 1) + phase)
            rows.append([math.cos(theta), math.sin(theta)])
    else:
        golden = math.pi * (3.0 - math.sqrt(5.0))
        rows = []
        for idx in range(n):
            z = 1.0 - 2.0 * (idx + 0.5) / float(n)
            radius = math.sqrt(max(0.0, 1.0 - z * z))
            theta = (idx + 1) * golden + 2.0 * math.pi * phase
            row = [radius * math.cos(theta), radius * math.sin(theta), z]
            for _extra in range(3, d):
                row.append(0.0)
            rows.append(row)
    return row_normalize(np.asarray(rows, dtype=np.float64))


def rank_d_correlation_target(n: int, d: int, case_id: str, system: str) -> np.ndarray:
    features = deterministic_rank_d_features(n, d, case_id, system)
    gram = features @ features.T
    gram = 0.5 * (gram + gram.T)
    np.fill_diagonal(gram, 1.0)
    off = gram - np.eye(n, dtype=np.float64)
    max_abs = float(np.max(np.abs(off))) if off.size else 0.0
    if max_abs >= 1.0 - 1e-4:
        raise RuntimeError("rank-d correlation target violates off-diagonal box")
    return gram


def psd_null_projector(gram: np.ndarray) -> np.ndarray:
    eigval, eigvec = np.linalg.eigh(0.5 * (np.asarray(gram, dtype=np.float64) + np.asarray(gram, dtype=np.float64).T))
    eps = max(1e-8, 1e-7 * max(float(np.max(eigval)) if eigval.size else 0.0, 1.0))
    null = eigvec[:, eigval <= eps]
    if null.size == 0:
        return np.zeros_like(gram)
    projector = null @ null.T
    return 0.5 * (projector + projector.T)


def rank_deficient_structural_solution(
    q: np.ndarray,
    k_consensus: np.ndarray,
    lambda_struct: float,
    d: int,
    case_id: str,
    system: str,
    movement_target: float = 0.012,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_rank = int(q.shape[1])
    g_star = rank_d_correlation_target(q.shape[0], d, case_id, system)
    if q_rank == 0:
        r0 = np.zeros(0, dtype=np.float64)
        return g_star.copy(), g_star, r0, vech(structural_moment(q, g_star)), r0, np.zeros_like(g_star), np.zeros_like(g_star)
    target_delta = vech(structural_moment(q, k_consensus)) - vech(structural_moment(q, g_star))
    if np.linalg.norm(target_delta) <= 1e-12:
        target_delta = np.linspace(1.0, 2.0, q_rank * (q_rank + 1) // 2, dtype=np.float64)
    unit_nu = target_delta / max(float(np.linalg.norm(target_delta)), 1e-12)
    adj_unit = structural_adjoint(q, unit_nu)
    off = adj_unit - np.diag(np.diag(adj_unit))
    max_off = float(np.max(np.abs(off))) if off.size else 0.0
    if max_off <= 1e-12:
        raise RuntimeError("structural forcing produced zero off-diagonal movement")
    projector = psd_null_projector(g_star)
    projector_off = projector - np.diag(np.diag(projector))
    projector_off_max = float(np.max(np.abs(projector_off))) if projector_off.size else 0.0
    if np.linalg.norm(projector) <= 1e-12:
        raise RuntimeError("rank-d target has no PSD boundary normal")
    psd_scale = min(0.25 * movement_target / max(projector_off_max, 1e-12), 0.01)
    s_psd = psd_scale * projector
    scale = movement_target / max_off
    off_mask = ~np.eye(g_star.shape[0], dtype=bool)
    for _ in range(30):
        nu = scale * unit_nu
        adj = structural_adjoint(q, nu)
        movement = (adj - np.diag(np.diag(adj))) + (s_psd - np.diag(np.diag(s_psd)))
        movement = 0.5 * (movement + movement.T)
        g0 = g_star - movement
        g0 = 0.5 * (g0 + g0.T)
        np.fill_diagonal(g0, 1.0)
        movement_off_max = float(np.max(np.abs(movement[off_mask]))) if off_mask.size else 0.0
        movement_fro = float(np.linalg.norm(movement))
        r_abs_max = float(np.max(np.abs(nu / float(lambda_struct)))) if nu.size else 0.0
        if 0.005 < movement_off_max <= 0.018 and movement_fro > 0.005 and r_abs_max <= 0.20:
            r_struct = -nu / float(lambda_struct)
            b_struct = vech(structural_moment(q, g_star)) - r_struct
            return g0, g_star, r_struct, b_struct, nu, adj, s_psd
        scale *= 0.7
    raise RuntimeError("could not construct rank-deficient feasible moved structural solution")


def rank_tail_audit(eigenvalues: np.ndarray, d: int, reconstruction_residual: float) -> dict[str, Any]:
    eigval = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    n = eigval.size
    eps = max(1e-8, 1e-7 * max(float(eigval[0]) if n else 0.0, 1.0))
    rank = int(np.sum(eigval > eps))
    positive_mass = float(np.maximum(eigval, 0.0).sum())
    omitted = float(np.maximum(eigval[d:], 0.0).sum()) if d < n else 0.0
    negative = float(np.maximum(-eigval, 0.0).sum())
    tail_ratio = omitted / max(positive_mass, 1e-12)
    lambda_d = float(eigval[d - 1]) if 0 < d <= n else (float(eigval[-1]) if n else 0.0)
    lambda_dplus1 = float(eigval[d]) if d < n else 0.0
    lambda_min = float(eigval[-1]) if n else 0.0
    passed = (
        rank <= d
        and omitted <= max(1e-6, 1e-8 * n)
        and tail_ratio <= 1e-8
        and negative <= max(1e-6, 1e-8 * n)
        and lambda_min >= -1e-7
        and reconstruction_residual <= 1e-6
    )
    return {
        "lambda_d": floatify(lambda_d),
        "lambda_dplus1": floatify(lambda_dplus1),
        "rank_eps": rank,
        "eps_rank": floatify(eps),
        "positive_eigenmass": floatify(positive_mass),
        "omitted_positive_eigenmass_beyond_d": floatify(omitted),
        "tail_ratio": floatify(tail_ratio),
        "negative_eigenmass": floatify(negative),
        "lambda_min": floatify(lambda_min),
        "reconstruction_residual": floatify(reconstruction_residual),
        "status": "PASS" if passed else "ENCODER_RANK_GATE_FAIL",
        "failure_policy": "return_null_no_truncation_schema_tolerance_rescue",
    }


def rank_factor_audit(gram: np.ndarray, d: int) -> tuple[bool, dict[str, Any], np.ndarray | None]:
    eigval, eigvec = np.linalg.eigh(0.5 * (gram + gram.T))
    order = np.argsort(-eigval, kind="mergesort")
    eigval = eigval[order]
    eigvec = eigvec[:, order]
    clipped = np.maximum(eigval, 0.0)
    eps = max(1e-8, 1e-7 * max(float(clipped[0]) if clipped.size else 0.0, 1.0))
    rank = int(np.sum(clipped > eps))
    if rank > d:
        return False, rank_tail_audit(eigval, d, 1e99), None
    y = np.zeros((gram.shape[0], d), dtype=np.float64)
    if rank:
        y[:, :rank] = eigvec[:, :rank] * np.sqrt(clipped[:rank])[None, :]
    residual = float(np.linalg.norm(y @ y.T - gram) / max(1.0, np.linalg.norm(gram)))
    audit = rank_tail_audit(eigval, d, residual)
    return audit["status"] == "PASS", audit, y


def procrustes_align(y: np.ndarray, z0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    u, _, vt = np.linalg.svd(np.asarray(y).T @ np.asarray(z0), full_matrices=False)
    rotation = u @ vt
    z_star = y @ rotation
    orth_resid = float(np.linalg.norm(rotation.T @ rotation - np.eye(rotation.shape[0])))
    return z_star, rotation, orth_resid


def implementation_hashes(paths: list[str]) -> tuple[str, list[dict[str, str]]]:
    rows = []
    for path in paths:
        fs_path = root_path(path)
        rows.append({"path": fs_path.relative_to(ROOT).as_posix(), "sha256": sha256_file(fs_path)})
    return sha256_obj(rows), rows


def replay_case_primal(case: dict[str, Any]) -> dict[str, Any]:
    if set(case) != CASE_KEYS:
        raise RuntimeError(f"case schema keys mismatch for {case.get('case_id')}")
    consensus_records = [consensus(group) for group in case["replicas"]]
    if consensus_records != case["consensus_records"]:
        raise RuntimeError(f"consensus replay mismatch for {case['case_id']}")
    phi = np.stack([encode(record) for record in consensus_records], axis=0)
    system = case["system"]
    d = int(case["d"])
    if system == "REMOVE":
        q = np.zeros((len(case["ids"]), 0), dtype=np.float64)
        gram = deterministic_baseline_gram(len(case["ids"]), min(4, d))
        g0 = gram.copy()
        b_struct = vech(structural_moment(q, gram))
        r_struct = np.zeros_like(b_struct)
        structural_nu = np.zeros_like(b_struct)
        structural_adj = np.zeros_like(gram)
        psd_dual_s = np.zeros_like(gram)
    else:
        q, raw_rank = orth_cap(phi, case["ids"], cap=8)
        if raw_rank != case["operator"]["raw_rank_before_cap"] or q.shape[1] != case["operator"]["q_rank"]:
            raise RuntimeError(f"orth_cap replay mismatch for {case['case_id']}")
        if system == "NOISE":
            noise = np.asarray(
                [[math.sin((i + 1) * (j + 2)) for j in range(phi.shape[1])] for i in range(phi.shape[0])],
                dtype=np.float64,
            )
            phi_for_gram = phi + 0.15 * noise
        elif system == "SHUFFLE":
            phi_for_gram = phi[list(reversed(range(phi.shape[0]))), :]
        else:
            phi_for_gram = phi
        gram_seed = psd_gram_from_features(phi_for_gram)
        g0, gram, r_struct, b_struct, structural_nu, structural_adj, psd_dual_s = rank_deficient_structural_solution(
            q=q,
            k_consensus=gram_seed,
            lambda_struct=1.0,
            d=d,
            case_id=case["case_id"],
            system=system,
        )
    operator = case["operator"]
    expected_m_shape = [int(q.shape[1]), int(q.shape[1])]
    if operator["actual_orth_cap_executed"] is not True or operator["rank_cap"] != 8:
        raise RuntimeError(f"operator orth-cap flags mismatch for {case['case_id']}")
    if system == "REMOVE" and operator["raw_rank_before_cap"] != 0:
        raise RuntimeError(f"REMOVE raw rank mismatch for {case['case_id']}")
    if operator["Q_shape"] != [len(case["ids"]), int(q.shape[1])] or operator["M_Q_shape"] != expected_m_shape:
        raise RuntimeError(f"operator shape mismatch for {case['case_id']}")
    if operator["q_rank"] != int(q.shape[1]) or operator["r_struct_length"] != int(r_struct.size):
        raise RuntimeError(f"operator rank/residual length mismatch for {case['case_id']}")
    if operator["b_struct"] != vector_to_list(b_struct):
        raise RuntimeError(f"operator b_struct mismatch for {case['case_id']}")
    if operator["m_q_formula"] != "Q^T(G-I)Q/N" or operator["vech_valid"] is not True:
        raise RuntimeError(f"operator formula/vech mismatch for {case['case_id']}")
    return {
        "G0": g0,
        "G_star": gram,
        "r_struct": r_struct,
        "Q": q,
        "b_struct": b_struct,
        "structural_nu": structural_nu,
        "structural_adjoint": structural_adj,
        "psd_dual_S": psd_dual_s,
    }


def verify_case_hashes(case: dict[str, Any], replay: dict[str, Any]) -> None:
    expected_operator_hash = sha256_obj(
        {
            "case_id": case["case_id"],
            "Q_shape": case["operator"]["Q_shape"],
            "b_struct": case["operator"]["b_struct"],
            "m_q_formula": case["operator"]["m_q_formula"],
        }
    )
    if case["hashes"]["operator_hash"] != expected_operator_hash:
        raise RuntimeError(f"case operator hash mismatch: {case['case_id']}")
    expected_primal_hash = sha256_obj(
        {
            "G0": matrix_to_list(replay["G0"]),
            "G_star": matrix_to_list(replay["G_star"]),
            "r_struct": vector_to_list(replay["r_struct"]),
        }
    )
    if case["hashes"]["primal_hash"] != expected_primal_hash:
        raise RuntimeError(f"case primal hash mismatch: {case['case_id']}")
    expected_case_payload = sha256_obj({key: value for key, value in case.items() if key != "hashes"})
    if case["hashes"]["case_payload_sha256"] != expected_case_payload:
        raise RuntimeError(f"case payload hash mismatch: {case['case_id']}")


def verify_finite_vi(case: dict[str, Any]) -> None:
    finite_vi = case["finite_vi_diagnostics"]
    if finite_vi != {
        "computed": True,
        "max_probe_violation": 0.0,
        "acceptance_role": "non_accepting_diagnostic_only",
        "attempted_acceptance": False,
    }:
        raise RuntimeError(f"finite-VI diagnostic attempted acceptance or drifted: {case['case_id']}")


def primal_residuals(
    gram: np.ndarray,
    g0: np.ndarray,
    r_struct: np.ndarray,
    q: np.ndarray,
    b_struct: np.ndarray,
    labels: list[int],
) -> list[dict[str, Any]]:
    n = gram.shape[0]
    eig = np.linalg.eigvalsh(0.5 * (gram + gram.T))
    off_mask = ~np.eye(n, dtype=bool)
    off = gram[off_mask]
    row_diff = gram - g0
    class_vals = []
    for cls in sorted(set(labels)):
        idx = [i for i, label in enumerate(labels) if label == cls]
        if idx:
            class_vals.append(float(np.linalg.norm(row_diff[idx, :].mean(axis=0))))
    struct = vech(structural_moment(q, gram)) - b_struct if q.shape[1] else np.zeros(0)
    values = {
        "symmetry_fro": float(np.linalg.norm(gram - gram.T)),
        "unit_diag_inf": float(np.max(np.abs(np.diag(gram) - 1.0))),
        "psd_min_violation": max(0.0, float(-np.min(eig))),
        "offdiag_box_violation": max(0.0, float(np.max(off - (1.0 - 1e-4))) if off.size else 0.0, float(np.max((-1.0 + 1e-4) - off)) if off.size else 0.0),
        "coordinate_trust_violation": max(0.0, float(np.max(np.abs(row_diff[off_mask]) - 0.02)) if off.size else 0.0),
        "row_trust_violation": max(0.0, float(np.max(np.linalg.norm(row_diff - np.diag(np.diag(row_diff)), axis=1)) - 0.05 * math.sqrt(max(n - 1, 1)))),
        "class_trust_violation": max(0.0, float(max(class_vals) if class_vals else 0.0) - 0.02 * math.sqrt(max(n, 1))),
        "structural_equality_l2": float(np.linalg.norm(r_struct - struct)),
        "structural_band_violation": max(0.0, float(np.max(np.abs(r_struct)) - 0.25) if r_struct.size else 0.0),
    }
    return [{"name": key, "value": floatify(val)} for key, val in values.items()]


def movement_metrics(replay: dict[str, Any]) -> dict[str, Any]:
    gram = replay["G_star"]
    g0 = replay["G0"]
    movement = gram - g0
    off_mask = ~np.eye(gram.shape[0], dtype=bool)
    structural_nu = replay["structural_nu"]
    structural_adj = replay["structural_adjoint"]
    return {
        "fro_norm_G_star_minus_G0": floatify(np.linalg.norm(movement)),
        "max_abs_offdiag_change": floatify(np.max(np.abs(movement[off_mask])) if off_mask.size else 0.0),
        "positive_threshold": 0.005,
        "structural_dual_l2": floatify(np.linalg.norm(structural_nu)),
        "structural_residual_l2": floatify(np.linalg.norm(replay["r_struct"])),
        "structural_adjoint_offdiag_l2": floatify(np.linalg.norm((structural_adj - np.diag(np.diag(structural_adj)))[off_mask]) if off_mask.size else 0.0),
    }


def factor_replay_metrics(gram: np.ndarray, d: int, rank_audit: dict[str, Any], factor: np.ndarray | None) -> dict[str, Any]:
    if factor is None:
        return {
            "factor_returned_null": True,
            "gram_reconstruction_residual": 1e99,
            "zstar_gram_residual": 1e99,
            "procrustes_orthogonality_residual": 1e99,
            "nondegenerate": False,
        }
    z0 = factor.copy()
    zstar, _rotation, orth_resid = procrustes_align(factor, z0)
    return {
        "factor_returned_null": False,
        "gram_reconstruction_residual": rank_audit["reconstruction_residual"],
        "zstar_gram_residual": floatify(np.linalg.norm(zstar @ zstar.T - gram) / max(1.0, np.linalg.norm(gram))),
        "procrustes_orthogonality_residual": floatify(orth_resid),
        "nondegenerate": bool(np.min(np.linalg.norm(zstar, axis=1)) > 1e-8),
    }


def robust_coverage_metrics(case: dict[str, Any]) -> dict[str, Any]:
    labels = case["labels"]
    system = case["system"]
    counts = {
        "0": int(sum(1 for label in labels if label == 0 and system == "ROBUST_COVERAGE")),
        "1": int(sum(1 for label in labels if label == 1 and system == "ROBUST_COVERAGE")),
    }
    coverage_pass = counts["0"] >= 2 and counts["1"] >= 2 and system == "ROBUST_COVERAGE"
    return {
        "coverage_gate_pass": bool(coverage_pass),
        "robust_constraints_enabled": False,
        "robust_query_count_by_class": counts,
        "safety_claim": "disabled" if not coverage_pass else "not_claimed",
    }


def verify_case_serialized_metrics(case: dict[str, Any], replay: dict[str, Any]) -> None:
    expected_movement = movement_metrics(replay)
    if case["movement_metrics"] != expected_movement:
        raise RuntimeError(f"case movement_metrics mismatch: {case['case_id']}")
    expected_residuals = primal_residuals(
        replay["G_star"],
        replay["G0"],
        replay["r_struct"],
        replay["Q"],
        replay["b_struct"],
        case["labels"],
    )
    if case["primal_residuals"] != expected_residuals:
        raise RuntimeError(f"case primal_residuals mismatch: {case['case_id']}")
    rank_pass, expected_rank_audit, factor = rank_factor_audit(replay["G_star"], int(case["d"]))
    if case["rank_audit"] != expected_rank_audit:
        raise RuntimeError(f"case rank_audit mismatch: {case['case_id']}")
    expected_factor = factor_replay_metrics(replay["G_star"], int(case["d"]), expected_rank_audit, factor)
    if case["factor_replay"] != expected_factor:
        raise RuntimeError(f"case factor_replay mismatch: {case['case_id']}")
    if case["robust_coverage"] != robust_coverage_metrics(case):
        raise RuntimeError(f"case robust_coverage mismatch: {case['case_id']}")
    movement_gate = bool(
        case["system"] != "FULL"
        or (
            expected_movement["fro_norm_G_star_minus_G0"] > expected_movement["positive_threshold"]
            and expected_movement["max_abs_offdiag_change"] > expected_movement["positive_threshold"]
            and expected_movement["structural_dual_l2"] > expected_movement["positive_threshold"]
            and expected_movement["structural_residual_l2"] > expected_movement["positive_threshold"]
        )
    )
    residual_gate = all(item["value"] <= 1e-6 for item in expected_residuals)
    expected_status = "PASS" if rank_pass and movement_gate and residual_gate else "FAIL"
    if case["kkt_status"] != expected_status:
        raise RuntimeError(f"case kkt_status mismatch: {case['case_id']}")


def require_close(actual: float, expected: float, label: str, tol: float = 1e-9) -> None:
    if abs(float(actual) - float(expected)) > tol:
        raise RuntimeError(f"{label} mismatch: expected {expected}, got {actual}")


def require_array_close(actual: Any, expected: np.ndarray, label: str, tol: float = 1e-9) -> None:
    arr = np.asarray(actual, dtype=np.float64)
    if arr.shape != expected.shape or not np.allclose(arr, expected, atol=tol, rtol=0.0):
        raise RuntimeError(f"{label} mismatch")


def require_zero_normal_block(block: dict[str, Any], sign: str, label: str) -> None:
    if block["present"] is not True:
        raise RuntimeError(f"{label} normal missing")
    if block["sign_convention"] != sign:
        raise RuntimeError(f"{label} sign convention mismatch")
    require_close(block["component_norm"], 0.0, f"{label} component_norm", tol=1e-12)
    require_close(block["dual_feasibility_residual"], 0.0, f"{label} dual residual", tol=1e-12)
    require_close(block["complementarity_max"], 0.0, f"{label} complementarity", tol=1e-12)


def verify_serialized_kkt_blocks(
    manifest: dict[str, Any],
    config: dict[str, Any],
    full_case: dict[str, Any],
    replay: dict[str, Any],
) -> dict[str, float]:
    metric = manifest["metric"]
    if metric != {
        "G_block": "identity_on_symmetric_Gram_entries",
        "r_struct_block": "lambda_struct_identity",
        "lambda_struct": config["projection_contract"]["lambda_struct"],
        "H_positive_definite": True,
    }:
        raise RuntimeError("metric block mismatch")

    g0 = replay["G0"]
    gstar = replay["G_star"]
    r_struct = replay["r_struct"]
    q = replay["Q"]
    b_struct = replay["b_struct"]
    structural_nu = replay["structural_nu"]
    structural_adj = replay["structural_adjoint"]
    psd_dual_s = replay["psd_dual_S"]
    movement = gstar - g0
    off_mask = ~np.eye(gstar.shape[0], dtype=bool)
    diag_dual = np.diag(structural_adj) + np.diag(psd_dual_s)
    equality_normal_g = -structural_adj
    diagonal_normal_g = np.diag(diag_dual)
    affine_normal_g = equality_normal_g + diagonal_normal_g
    psd_normal_g = -psd_dual_s
    normal_g = affine_normal_g + psd_normal_g
    lambda_struct = float(metric["lambda_struct"])
    grad_norm = math.sqrt(float(np.linalg.norm(movement)) ** 2 + float(np.linalg.norm(lambda_struct * r_struct)) ** 2)
    normal_sum_norm = math.sqrt(float(np.linalg.norm(normal_g)) ** 2 + float(np.linalg.norm(structural_nu)) ** 2)
    stationarity_g = movement + normal_g
    stationarity_r = lambda_struct * r_struct + structural_nu
    stationarity_norm = math.sqrt(float(np.linalg.norm(stationarity_g)) ** 2 + float(np.linalg.norm(stationarity_r)) ** 2)
    normalized = stationarity_norm / (1.0 + float(np.linalg.norm(movement)) + float(np.linalg.norm(r_struct)))
    psd_lambda_min = float(np.min(np.linalg.eigvalsh(0.5 * (psd_dual_s + psd_dual_s.T))))
    psd_complementarity = abs(float(np.trace(psd_dual_s @ gstar)))

    require_array_close(manifest["primal"]["G0"], g0, "top-level G0")
    require_array_close(manifest["primal"]["G_star"], gstar, "top-level G_star")
    require_array_close(manifest["primal"]["r_struct"], r_struct, "top-level r_struct")
    if manifest["primal"]["residual_summaries"] != full_case["primal_residuals"]:
        raise RuntimeError("top-level primal residual summaries mismatch FULL case")
    expected_objective = 0.5 * float(np.linalg.norm(movement)) ** 2 + 0.5 * lambda_struct * float(np.linalg.norm(r_struct)) ** 2
    require_close(manifest["primal"]["objective_value"], expected_objective, "objective", tol=1e-10)
    if manifest["movement_metrics"] != full_case["movement_metrics"]:
        raise RuntimeError("top-level movement metrics mismatch FULL case")

    affine = manifest["affine_normals"]
    if affine["structural_sign_convention"] != "normal_G=-A_struct^T nu, normal_r=nu":
        raise RuntimeError("affine structural sign convention mismatch")
    require_close(affine["symmetry_affine_dual_fro_norm"], 0.0, "symmetry affine dual", tol=1e-12)
    require_array_close(affine["diagonal_affine_dual"], diag_dual, "diagonal affine dual")
    require_array_close(affine["structural_nu"], structural_nu, "structural_nu")
    require_close(affine["normal_G_fro_norm"], float(np.linalg.norm(affine_normal_g)), "normal_G_fro_norm")
    require_close(affine["normal_r_l2_norm"], float(np.linalg.norm(structural_nu)), "normal_r_l2_norm")

    require_zero_normal_block(
        manifest["box_coordinate_normals"]["offdiag_box"],
        "lower normal is -E_ij, upper normal is +E_ij with nonnegative multipliers",
        "offdiag_box",
    )
    require_zero_normal_block(
        manifest["box_coordinate_normals"]["coordinate_trust"],
        "trust band normals use +/- coordinate directions with nonnegative multipliers",
        "coordinate_trust",
    )
    require_zero_normal_block(
        manifest["box_coordinate_normals"]["structural_band"],
        "structural residual band normals use +/- residual directions with nonnegative multipliers",
        "structural_band",
    )
    require_zero_normal_block(
        manifest["soc_normals"]["row_trust"],
        "Lorentz dual for ||row_delta||_2 <= rho_row",
        "row_trust",
    )
    require_zero_normal_block(
        manifest["soc_normals"]["class_trust"],
        "Lorentz dual for class mean trust balls",
        "class_trust",
    )

    psd = manifest["psd_normal"]
    if psd["present"] is not True or psd["normal_contribution_sign"] != "v_psd=-S_psd":
        raise RuntimeError("PSD normal presence/sign mismatch")
    require_array_close(psd["S_psd"], psd_dual_s, "PSD S_psd", tol=1e-9)
    if np.linalg.norm(psd_dual_s) <= 1e-12:
        raise RuntimeError("PSD normal is unexpectedly zero for rank-deficient FULL case")
    if psd_lambda_min < -1e-7:
        raise RuntimeError("PSD dual is not positive semidefinite")
    require_close(psd["dual_lambda_min"], psd_lambda_min, "PSD dual lambda min", tol=1e-9)
    require_close(psd["complementarity_trace"], psd_complementarity, "PSD complementarity trace", tol=1e-9)
    if psd_complementarity > 1e-6:
        raise RuntimeError("PSD complementarity failed")

    halfspace = manifest["halfspace_normals"]
    if halfspace != {
        "present": True,
        "robust_constraints_enabled": False,
        "sign_convention": "nonnegative multipliers for <= halfspaces",
        "total_multiplier_l1": 0.0,
    }:
        raise RuntimeError("halfspace normal block mismatch")

    stationarity = manifest["stationarity"]
    if stationarity["status"] != "PASS":
        raise RuntimeError("stationarity status is not PASS")
    require_close(stationarity["residual_norm"], stationarity_norm, "serialized stationarity residual", tol=1e-9)
    require_close(stationarity["normalized_residual"], normalized, "serialized normalized stationarity", tol=1e-9)
    require_close(stationarity["gradient_norm"], grad_norm, "stationarity gradient norm")
    require_close(stationarity["normal_sum_norm"], normal_sum_norm, "stationarity normal sum norm")
    require_close(stationarity["acceptance_tolerance"], 1e-6, "stationarity tolerance", tol=0.0)
    if normalized > 1e-6:
        raise RuntimeError("recomputed stationarity residual failed")

    dual = manifest["dual_feasibility"]
    expected_dual = {
        "status": "PASS",
        "linear_multiplier_min": 0.0,
        "soc_cone_residual_max": 0.0,
        "psd_dual_lambda_min": floatify(psd_lambda_min),
        "affine_unrestricted": True,
    }
    if dual != expected_dual:
        raise RuntimeError("dual feasibility block mismatch")

    complementarity = manifest["complementarity"]
    expected_per_family = [
        {"name": "box_coordinate", "value": 0.0},
        {"name": "soc", "value": 0.0},
        {"name": "psd", "value": floatify(psd_complementarity)},
        {"name": "halfspace", "value": 0.0},
        {"name": "structural_band", "value": 0.0},
    ]
    if complementarity["status"] != "PASS" or complementarity["per_family"] != expected_per_family:
        raise RuntimeError("complementarity block mismatch")
    require_close(complementarity["max_abs"], psd_complementarity, "complementarity max_abs", tol=1e-9)

    if manifest["duality_gap"] != {
        "dual_objective_materialized": False,
        "gap_pass_claimed": False,
        "note": "No dual objective pass is claimed; stationarity plus valid normal decomposition is the acceptance path.",
    }:
        raise RuntimeError("duality gap block mismatch")

    structural_residual = np.asarray(manifest["primal"]["r_struct"], dtype=np.float64) - (
        vech(structural_moment(q, gstar)) - b_struct
    )
    if np.linalg.norm(structural_residual) > 1e-6:
        raise RuntimeError("structural equality residual failed")
    if np.linalg.norm(structural_nu) <= manifest["movement_metrics"]["positive_threshold"]:
        raise RuntimeError("structural dual is not binding")
    return {
        "stationarity_normalized_residual": normalized,
        "movement_fro": float(np.linalg.norm(movement)),
        "movement_offdiag_max": float(np.max(np.abs(movement[off_mask])) if off_mask.size else 0.0),
        "structural_dual_l2": float(np.linalg.norm(structural_nu)),
        "objective_value": expected_objective,
    }


def verify_source_and_access(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    source_path = manifest["source_manifest_path"]
    access_path = manifest["access_ledger_path"]
    if sha256_file(root_path(source_path)) != manifest["hashes"]["source_manifest_sha256"]:
        raise RuntimeError("source manifest file hash mismatch")
    if sha256_file(root_path(access_path)) != manifest["hashes"]["access_ledger_sha256"]:
        raise RuntimeError("access ledger file hash mismatch")
    source = read_json(source_path)
    access = read_json(access_path)
    if source["config_sha256"] != manifest["hashes"]["config_sha256"]:
        raise RuntimeError("source manifest config hash mismatch")
    if source["payload_schema_sha256"] != manifest["hashes"]["payload_schema_sha256"]:
        raise RuntimeError("source manifest payload schema hash mismatch")
    if source["case_schema_sha256"] != manifest["hashes"]["case_schema_sha256"]:
        raise RuntimeError("source manifest case schema hash mismatch")
    if source["implementation_sha256"] != manifest["hashes"]["implementation_sha256"]:
        raise RuntimeError("source manifest implementation hash mismatch")
    if source["relevant_tree_sha256"] != manifest["dirty_binding"]["relevant_tree_sha256"]:
        raise RuntimeError("relevant tree binding mismatch")
    recomputed_tree = sha256_obj(
        {
            "source_rows": source["source_rows"],
            "implementation_sha256": source["implementation_sha256"],
            "relevant_git_status": source["relevant_git_status"],
            "artifact_outputs_excluded": True,
        }
    )
    if recomputed_tree != source["relevant_tree_sha256"]:
        raise RuntimeError("source manifest relevant-tree recomputation mismatch")
    if sha256_obj(access["access_ledger"]) != access["access_ledger_sha256"]:
        raise RuntimeError("access ledger internal hash mismatch")
    for rel, expected in config["hash_bindings"]["run1_frozen"].items():
        if sha256_file(root_path(rel)) != expected:
            raise RuntimeError(f"Run1 frozen hash changed: {rel}")
    for rel, expected in config["hash_bindings"]["authoritative_inputs"].items():
        if sha256_file(root_path(rel)) != expected:
            raise RuntimeError(f"authoritative input hash changed: {rel}")
    for row in source["source_rows"]["run2_implementation_files"]:
        if sha256_file(root_path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"Run2 source hash mismatch: {row['path']}")
    impl_hash, impl_rows = implementation_hashes(config["implementation_files"])
    if impl_hash != manifest["hashes"]["implementation_sha256"] or impl_hash != source["implementation_sha256"]:
        raise RuntimeError("implementation aggregate hash mismatch")
    if impl_rows != source["source_rows"]["run2_implementation_files"]:
        raise RuntimeError("implementation source rows mismatch")
    if sha256_file(root_path(config["paths"]["payload_schema"])) != manifest["hashes"]["payload_schema_sha256"]:
        raise RuntimeError("payload schema hash mismatch")
    if sha256_file(root_path(config["paths"]["case_schema"])) != manifest["hashes"]["case_schema_sha256"]:
        raise RuntimeError("case schema hash mismatch")
    run1_artifact = "artifacts/lb_scgp_global/v1/m0/contract_freeze.json"
    run1_lock = "artifacts/lb_scgp_global/v1/m0/contract_freeze.json.publish.lock"
    if sha256_file(root_path(run1_artifact)) != manifest["hashes"]["run1_artifact_sha256"]:
        raise RuntimeError("Run1 artifact top-level hash mismatch")
    if sha256_file(root_path(run1_lock)) != manifest["hashes"]["run1_lock_sha256"]:
        raise RuntimeError("Run1 lock top-level hash mismatch")
    forbidden = []
    for record in access["access_ledger"]:
        path = record.get("path", "")
        lowered = path.lower()
        if path.startswith("data/") and record.get("kind") != "declared_not_opened":
            forbidden.append(path)
        if any(token in lowered for token in ["query_z", "query_labels", "teacher"]):
            forbidden.append(path)
    if forbidden:
        raise RuntimeError(f"forbidden access ledger paths: {forbidden}")
    nonzero = {key: value for key, value in access["zero_counters"].items() if value != 0}
    if nonzero:
        raise RuntimeError(f"nonzero access counters: {nonzero}")
    return {
        "source_manifest_sha256": manifest["hashes"]["source_manifest_sha256"],
        "access_ledger_sha256": manifest["hashes"]["access_ledger_sha256"],
        "relevant_tree_sha256": source["relevant_tree_sha256"],
    }


def verify_declared_paths(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    expected_paths = {
        "config_path": "configs/lb_scgp_global_r2/m0_synth_kkt_v4.json",
        "source_manifest_path": config["paths"]["source_manifest_path"],
        "access_ledger_path": config["paths"]["access_ledger_path"],
    }
    for key, expected in expected_paths.items():
        if manifest[key] != expected:
            raise RuntimeError(f"{key} binding mismatch: expected {expected}, got {manifest[key]}")
        root_path(manifest[key])


def verify_top_level_hashes(
    manifest: dict[str, Any],
    config: dict[str, Any],
    full_case: dict[str, Any],
    verify_files: bool,
) -> None:
    if manifest["hashes"]["case_matrix_sha256"] != sha256_obj(manifest["case_matrix"]):
        raise RuntimeError("case matrix hash mismatch")
    if manifest["hashes"]["operator_hash"] != full_case["hashes"]["operator_hash"]:
        raise RuntimeError("top-level operator hash mismatch")
    if verify_files:
        if manifest["hashes"]["config_sha256"] != sha256_file(root_path("configs/lb_scgp_global_r2/m0_synth_kkt_v4.json")):
            raise RuntimeError("config top-level hash mismatch")
        if manifest["hashes"]["payload_schema_sha256"] != sha256_file(root_path(config["paths"]["payload_schema"])):
            raise RuntimeError("payload schema top-level hash mismatch")
        if manifest["hashes"]["case_schema_sha256"] != sha256_file(root_path(config["paths"]["case_schema"])):
            raise RuntimeError("case schema top-level hash mismatch")
        run1_artifact = "artifacts/lb_scgp_global/v1/m0/contract_freeze.json"
        run1_lock = "artifacts/lb_scgp_global/v1/m0/contract_freeze.json.publish.lock"
        if manifest["hashes"]["run1_artifact_sha256"] != sha256_file(root_path(run1_artifact)):
            raise RuntimeError("Run1 artifact top-level hash mismatch")
        if manifest["hashes"]["run1_lock_sha256"] != sha256_file(root_path(run1_lock)):
            raise RuntimeError("Run1 lock top-level hash mismatch")
        impl_hash, _ = implementation_hashes(config["implementation_files"])
        if manifest["hashes"]["implementation_sha256"] != impl_hash:
            raise RuntimeError("implementation top-level hash mismatch")


def verify_core(manifest: dict[str, Any], config: dict[str, Any], verify_files: bool) -> dict[str, Any]:
    validate_manifest_schema(manifest, config)
    if set(manifest) != TOP_KEYS:
        raise RuntimeError(f"top-level schema keys mismatch: {sorted(set(manifest) ^ TOP_KEYS)}")
    if manifest["payload_sha256"] != payload_hash(manifest):
        raise RuntimeError("payload_sha256 mismatch")
    if manifest["artifact_schema_id"] != SCHEMA_ID or manifest["run_id"] != RUN2:
        raise RuntimeError("run/schema mismatch")
    if manifest["terminal_state"] != "PRODUCED_PENDING_INDEPENDENT_VERIFY":
        raise RuntimeError("terminal state mismatch")
    if manifest["authorized_boundary"] != {"run_id": RUN2, "synthetic_only": True, "run3_or_later_locked": True}:
        raise RuntimeError("authorized boundary mismatch")
    if manifest["no_success_claim"] is not True:
        raise RuntimeError("success claim attempted")
    if manifest["slurm_policy"]["required"] is not True or manifest["slurm_policy"]["gpu"] != 0:
        raise RuntimeError("SLURM policy mismatch")
    verify_declared_paths(manifest, config)
    if manifest["acceptance"]["acceptance_path"] != "serialized_h_metric_normal_cone_kkt":
        raise RuntimeError("finite VI or non-KKT acceptance attempted")
    if manifest["acceptance"]["finite_vi_can_accept"] is not False:
        raise RuntimeError("finite VI marked accepting")
    if manifest["acceptance"]["producer_status"] != "PASS_CANDIDATE" or manifest["acceptance"]["semantic_verifier_required"] is not True:
        raise RuntimeError("acceptance status mismatch")

    case_matrix = manifest["case_matrix"]
    if case_matrix["status"] != "PASS":
        raise RuntimeError("case matrix not PASS")
    cases = case_matrix["cases"]
    systems = {case["system"] for case in cases}
    required_systems = {"FULL", "REMOVE", "SHUFFLE", "NOISE", "AMBIGUOUS", "ROBUST_COVERAGE"}
    if systems != required_systems or len(cases) != 6:
        raise RuntimeError("missing required system cases")
    case_replays = {}
    for case in cases:
        verify_finite_vi(case)
        if case["acceptance_path"] != "serialized_h_metric_normal_cone_kkt" or case["kkt_status"] != "PASS":
            raise RuntimeError(f"case acceptance/status mismatch: {case['case_id']}")
        replay = replay_case_primal(case)
        verify_case_hashes(case, replay)
        verify_case_serialized_metrics(case, replay)
        case_replays[case["case_id"]] = replay
    full_case = next(case for case in cases if case["system"] == "FULL")
    full_replay = case_replays[full_case["case_id"]]
    verify_top_level_hashes(manifest, config, full_case, verify_files=verify_files)
    if verify_files:
        file_binding = verify_source_and_access(manifest, config)
    else:
        file_binding = {}

    g0 = np.asarray(manifest["primal"]["G0"], dtype=np.float64)
    gstar = np.asarray(manifest["primal"]["G_star"], dtype=np.float64)
    r = np.asarray(manifest["primal"]["r_struct"], dtype=np.float64)
    finite_array(g0, "G0")
    finite_array(gstar, "G_star")
    finite_array(r, "r_struct")
    if np.linalg.norm(gstar - g0) <= manifest["movement_metrics"]["positive_threshold"]:
        raise RuntimeError("FULL attempted no-movement acceptance")
    if g0.shape != gstar.shape or not np.allclose(g0, g0.T, atol=1e-12):
        raise RuntimeError("FULL baseline G0 shape/symmetry mismatch")
    if not np.allclose(np.diag(g0), 1.0, atol=1e-8):
        raise RuntimeError("FULL baseline G0 unit diagonal violated")
    if not np.allclose(np.diag(gstar), 1.0, atol=1e-8):
        raise RuntimeError("unit diagonal violated")
    if np.min(np.linalg.eigvalsh(0.5 * (gstar + gstar.T))) < -1e-7:
        raise RuntimeError("PSD primal violated")
    off_mask = ~np.eye(gstar.shape[0], dtype=bool)
    if np.max(np.abs(gstar[off_mask])) > 1.0 - 1e-4 + 1e-9:
        raise RuntimeError("off-diagonal box violated")
    movement = gstar - g0
    move_fro = float(np.linalg.norm(movement))
    move_off = float(np.max(np.abs(movement[off_mask])))
    if move_fro <= manifest["movement_metrics"]["positive_threshold"] or move_off <= manifest["movement_metrics"]["positive_threshold"]:
        raise RuntimeError("movement nondegeneration gate failed")

    kkt_metrics = verify_serialized_kkt_blocks(manifest, config, full_case, full_replay)

    rank_pass, rank_audit, factor = rank_factor_audit(gstar, full_case["d"])
    if not rank_pass or factor is None:
        raise RuntimeError("rank/factor replay failed")
    zstar_resid = float(np.linalg.norm(factor @ factor.T - gstar) / max(1.0, np.linalg.norm(gstar)))
    if zstar_resid > 1e-6 or np.min(np.linalg.norm(factor, axis=1)) <= 1e-8:
        raise RuntimeError("factor nondegeneration failed")
    remove_case = next(case for case in cases if case["system"] == "REMOVE")
    if remove_case["movement_metrics"]["fro_norm_G_star_minus_G0"] != 0.0:
        raise RuntimeError("REMOVE/null did not replay G0")
    rank_probe = manifest["rank_failure_probe"]
    probe_pass, probe_audit, probe_factor = rank_factor_audit(np.eye(4, dtype=np.float64), int(rank_probe["d"]))
    if rank_probe["case_id"] != "RANK_FAILURE_RETURNS_NULL" or rank_probe["expected_status"] != "ENCODER_RANK_GATE_FAIL":
        raise RuntimeError("rank failure probe identity/status mismatch")
    if rank_probe["factor_returned_null"] is not True or probe_factor is not None or probe_pass:
        raise RuntimeError("rank failure probe did not return null")
    if rank_probe["rank_audit"] != probe_audit:
        raise RuntimeError("rank failure probe audit mismatch")
    if any(item["status"] != "REJECT" for item in manifest["schema_fixture_results"]["invalid_schema"]):
        raise RuntimeError("invalid schema fixture accepted")
    if manifest["schema_fixture_results"]["unresolved_values"]["schema_status"] != "PASS":
        raise RuntimeError("unresolved value fixture failed")
    if any(value != "PASS" for value in manifest["orth_cap_matrix"].values()):
        raise RuntimeError("orth_cap matrix failed")
    return {
        **kkt_metrics,
        "rank_eps": rank_audit["rank_eps"],
        "zstar_gram_residual": zstar_resid,
        **file_binding,
    }


def run_injections(manifest: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    def refresh_payload(mutated: dict[str, Any]) -> dict[str, Any]:
        mutated["payload_sha256"] = payload_hash(mutated)
        return mutated

    def refresh_case_and_matrix(mutated: dict[str, Any], case_index: int) -> dict[str, Any]:
        case = mutated["case_matrix"]["cases"][case_index]
        case["hashes"]["case_payload_sha256"] = sha256_obj({key: value for key, value in case.items() if key != "hashes"})
        mutated["hashes"]["case_matrix_sha256"] = sha256_obj(mutated["case_matrix"])
        return refresh_payload(mutated)

    mutations = {}
    m = copy.deepcopy(manifest)
    m["primal"]["G_star"][0][0] = 1e100
    mutations["nan_overflow"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["hashes"]["operator_hash"] = "0" * 64
    mutations["perturbed_artifact_source_operator_hash"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["unexpected"] = True
    mutations["invalid_extra_missing_schema_fields"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["psd_normal"]["normal_contribution_sign"] = "v_psd=+S_psd"
    mutations["wrong_dual_sign"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m.pop("soc_normals")
    mutations["incomplete_cone_family"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["source_manifest_path"] = "data/gt/MHC/test.jsonl"
    mutations["forbidden_path"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["rank_failure_probe"]["factor_returned_null"] = False
    mutations["rank_failure"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["case_matrix"]["cases"][0]["movement_metrics"]["fro_norm_G_star_minus_G0"] += 0.125
    mutations["bogus_case_movement_metrics"] = refresh_case_and_matrix(m, 0)
    m = copy.deepcopy(manifest)
    m["case_matrix"]["cases"][1]["primal_residuals"][0]["value"] = 0.25
    mutations["bogus_case_primal_residual"] = refresh_case_and_matrix(m, 1)
    m = copy.deepcopy(manifest)
    m["case_matrix"]["cases"][2]["rank_audit"]["rank_eps"] = 0
    mutations["bogus_case_rank_audit"] = refresh_case_and_matrix(m, 2)
    m = copy.deepcopy(manifest)
    m["case_matrix"]["cases"][3]["factor_replay"]["zstar_gram_residual"] = 0.25
    mutations["bogus_case_factor_replay"] = refresh_case_and_matrix(m, 3)
    m = copy.deepcopy(manifest)
    m["case_matrix"]["cases"][5]["robust_coverage"]["coverage_gate_pass"] = False
    mutations["bogus_case_robust_coverage"] = refresh_case_and_matrix(m, 5)
    m = copy.deepcopy(manifest)
    m["rank_failure_probe"]["rank_audit"]["rank_eps"] = 0
    mutations["bogus_rank_failure_audit"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["acceptance"]["acceptance_path"] = "finite_vi"
    mutations["finite_vi_only_attempted_acceptance"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["primal"]["G_star"] = copy.deepcopy(m["primal"]["G0"])
    m["movement_metrics"]["fro_norm_G_star_minus_G0"] = 0.0
    m["movement_metrics"]["max_abs_offdiag_change"] = 0.0
    mutations["identity_no_movement_claims_full"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["box_coordinate_normals"]["offdiag_box"]["component_norm"] = 5e-7
    mutations["malformed_normal_residual"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["halfspace_normals"]["present"] = False
    mutations["malformed_normal_presence"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["complementarity"]["max_abs"] = 5e-7
    m["complementarity"]["per_family"][0]["value"] = 5e-7
    mutations["malformed_complementarity"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["dual_feasibility"]["status"] = "FAIL"
    mutations["malformed_dual_status"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["stationarity"]["status"] = "FAIL"
    mutations["malformed_stationarity_status"] = refresh_payload(m)
    m = copy.deepcopy(manifest)
    m["case_matrix"]["cases"][0]["finite_vi_diagnostics"]["computed"] = False
    mutations["malformed_finite_vi_diagnostic"] = refresh_case_and_matrix(m, 0)
    results = {}
    for name, mutated in mutations.items():
        try:
            verify_core(mutated, config, verify_files=False)
        except Exception as exc:  # noqa: BLE001 - serialized negative test evidence
            results[name] = {"status": "REJECT", "reason": str(exc)[:500]}
        else:
            results[name] = {"status": "UNEXPECTED_ACCEPT", "reason": ""}
    expected = manifest["injection_results_expected"]
    if set(expected) != set(results):
        raise RuntimeError(f"injection key mismatch: expected={sorted(expected)} actual={sorted(results)}")
    if any(value != "REJECT" for value in expected.values()):
        raise RuntimeError("injection expected result is not uniformly REJECT")
    bad = {key: value for key, value in results.items() if value["status"] != "REJECT"}
    if bad:
        raise RuntimeError(f"injection unexpectedly accepted: {bad}")
    return results


def publish_json(path: str | Path, obj: Any) -> None:
    fs_path = root_path(path)
    fs_path.parent.mkdir(parents=True, exist_ok=True)
    lock = fs_path.with_name(fs_path.name + ".publish.lock")
    fd_lock = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    tmp = None
    try:
        os.write(fd_lock, str(os.getpid()).encode("ascii"))
        os.fsync(fd_lock)
        os.close(fd_lock)
        fd_lock = -1
        if fs_path.exists():
            raise FileExistsError(f"refusing to overwrite {fs_path}")
        fd, tmp = tempfile.mkstemp(prefix=fs_path.name + ".tmp.", dir=str(fs_path.parent))
        with os.fdopen(fd, "wb") as handle:
            handle.write((canonical_json(obj) + "\n").encode("utf-8"))
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, fs_path)
        os.unlink(tmp)
        tmp = None
    finally:
        if fd_lock >= 0:
            os.close(fd_lock)
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("independent verification must run under SLURM")
    config = read_json(args.config)
    manifest = read_json(args.manifest)
    try:
        metrics = verify_core(manifest, config, verify_files=True)
        injections = run_injections(manifest, config)
        decision = {
            "schema_version": "lb_scgp_global_r2_run2_v4_semantic_verification_v1",
            "run_id": RUN2,
            "decision": "PASS",
            "manifest_path": args.manifest,
            "manifest_file_sha256": sha256_file(root_path(args.manifest)),
            "manifest_payload_sha256": manifest["payload_sha256"],
            "metrics": metrics,
            "injection_results": injections,
            "acceptance_path": "serialized_h_metric_normal_cone_kkt",
            "finite_vi_acceptance": false_value(),
            "medium_findings_closed": {
                "M1_strict_schema_semantic_verifier": True,
                "M2_dirty_binding_run1_run2_relevant_tree": True,
                "M3_orth_cap_and_M_Q_executed_with_rank_cap_cases": True
            },
        }
        publish_json(args.out, decision)
        return 0
    except Exception as exc:  # noqa: BLE001 - publish fail-closed decision
        decision = {
            "schema_version": "lb_scgp_global_r2_run2_v4_semantic_verification_v1",
            "run_id": RUN2,
            "decision": "FAIL",
            "manifest_path": args.manifest,
            "manifest_file_sha256": sha256_file(root_path(args.manifest)) if root_path(args.manifest).exists() else "",
            "reason": str(exc),
        }
        publish_json(args.out, decision)
        print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
        return 1


def false_value() -> bool:
    return False


if __name__ == "__main__":
    raise SystemExit(main())
