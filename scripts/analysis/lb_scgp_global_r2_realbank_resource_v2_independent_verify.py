#!/usr/bin/env python
"""Fresh independent semantic verifier for the realbank resource microbenchmark.

This verifier intentionally does NOT import the realbank producer or common
module.  It re-loads the two frozen train banks, independently recomputes G0, the
NON-SCIENCE structural placeholder, the rank-tail factor, the deterministic replay
digest, the robust-coverage report, and the isolation-injection classifier from
the config + serialized manifest, then runs fail-closed manifest mutations that
must all be REJECTED, and stamps PASS/FAIL.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import resource
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/jehc223/RGCL")
RUN1 = "LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1"
RUN2_V4 = "LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v4"
RUN3 = "LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v2"
SCHEMA_ID = "scgp_global_realbank_resource_v2"
MANIFEST_SCHEMA_VERSION = "lb_scgp_global_r2_realbank_resource_manifest_v2"
CONFIG_PATH = "configs/lb_scgp_global_r2/m0_realbank_resource_v2.json"
CAP_BYTES = 103079215104
RANK_CAP = 8
M_SCALE = 36
TOPK = 20
DATASETS = ("MHC", "MHC_zh")
EXPECTED_N = {"MHC": 549, "MHC_zh": 579}
RUN1_ARTIFACT = "artifacts/lb_scgp_global/v1/m0/contract_freeze.json"
RUN1_LOCK = "artifacts/lb_scgp_global/v1/m0/contract_freeze.json.publish.lock"

ZERO_COUNTER_KEYS = (
    "auxiliary_head_route_count", "cache_outer_held_content_read_count",
    "cache_outer_held_label_read_count", "cache_read_count", "cache_test_content_read_count",
    "cache_test_label_read_count", "cache_validation_content_read_count",
    "cache_validation_label_read_count", "certificate_read_count", "compiler_target_read_count",
    "comparator_freeze_adaptive_query_label_read_count", "comparator_freeze_final_test_label_read_count",
    "comparator_freeze_final_test_margin_read_count", "comparator_freeze_final_test_prediction_read_count",
    "control_construction_final_test_error_read_count", "control_construction_final_test_label_read_count",
    "control_construction_final_test_margin_read_count", "control_construction_final_test_prediction_read_count",
    "forbidden_path_read_count", "gpu_device_count", "held_content_read_count", "held_label_read_count",
    "key_selection_route_count", "local_v7_pass_evidence_reuse_count", "mllm_call_count",
    "mllm_calls_outside_train_cache", "model_call_count", "network_call_count",
    "non_allowlisted_train_content_read_count", "ocr_call_count", "pair_triplet_supcon_route_count",
    "performance_evaluation_count", "query_labels_read_count", "query_z_read_count",
    "reranking_route_count", "run3_cache_or_later_attempt_count", "sample_weighting_route_count",
    "segment_gold_read_count", "teacher_artifact_read_count", "teacher_cache_read_count",
    "teacher_cache_write_count", "test_content_read_count", "test_label_read_count",
    "train_label_read_count", "training_call_count", "validation_content_read_count",
    "validation_label_read_count",
)
TOP_KEYS = {
    "schema_version", "artifact_schema_id", "run_id", "terminal_state", "no_success_claim",
    "decision", "authorized_boundary", "slurm_policy", "config_path", "source_manifest_path",
    "access_ledger_path", "resource_peak", "rank_tail", "replay_hashes", "robust_coverage",
    "isolation_injection_results", "structural_placeholder", "allowed_reads", "gold_isolation",
    "dirty_binding", "acceptance", "hashes", "payload_sha256",
}
FORBIDDEN_TOKENS = ("query_z", "query_labels", "teacher", "cache", "held", "certificate")


# --------------------------------------------------------------------------- #
# Serialization / math (byte-faithful to the accepted Run2-v4 verifier core)   #
# --------------------------------------------------------------------------- #
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


def root_path(path: str | Path) -> Path:
    raw = Path(path)
    candidate = raw if raw.is_absolute() else ROOT / raw
    resolved = candidate.resolve()
    resolved.relative_to(ROOT.resolve())
    return resolved


def read_json(path: str | Path) -> Any:
    with open(root_path(path), encoding="utf-8") as handle:
        return json.load(handle)


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


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise RuntimeError("zero row in row_normalize")
    return matrix / norm


def orth_cap(phi: np.ndarray, ids: list[str], rank_cap: int = 8) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    centered = phi - phi.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s.size == 0:
        return np.zeros((phi.shape[0], 0), dtype=np.float64)
    threshold = max(1e-8, 1e-7 * float(s[0]))
    raw_rank = int(np.sum(s > threshold))
    rank = min(rank_cap, raw_rank)
    q = u[:, :rank].copy()
    for col in range(q.shape[1]):
        pivot = max(range(q.shape[0]), key=lambda row: (abs(float(q[row, col])), str(ids[row])))
        if q[pivot, col] < 0:
            q[:, col] *= -1.0
    return q


def vech(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    rows, cols = np.tril_indices(matrix.shape[0])
    return matrix[rows, cols]


def structural_moment(q: np.ndarray, gram: np.ndarray) -> np.ndarray:
    if q.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    n = gram.shape[0]
    out = q.T @ (gram - np.eye(n)) @ q / float(n)
    return 0.5 * (out + out.T)


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


def rank_tail_audit(eigenvalues: np.ndarray, d: int, reconstruction_residual: float) -> dict[str, Any]:
    values = np.sort(np.asarray(eigenvalues, dtype=np.float64))[::-1]
    n = values.size
    eps_rank = max(1e-8, 1e-7 * max(float(values[0]) if n else 0.0, 1.0))
    rank_eps = int(np.sum(values > eps_rank))
    positive_mass = float(np.maximum(values, 0.0).sum())
    omitted = float(np.maximum(values[d:], 0.0).sum()) if d < n else 0.0
    negative = float(np.maximum(-values, 0.0).sum())
    tail_ratio = omitted / max(positive_mass, 1e-12)
    lambda_d = float(values[d - 1]) if 0 < d <= n else (float(values[-1]) if n else 0.0)
    lambda_dplus1 = float(values[d]) if d < n else 0.0
    lambda_min = float(values[-1]) if n else 0.0
    passed = (
        rank_eps <= d
        and omitted <= max(1e-6, 1e-8 * n)
        and tail_ratio <= 1e-8
        and negative <= max(1e-6, 1e-8 * n)
        and lambda_min >= -1e-7
        and reconstruction_residual <= 1e-6
    )
    return {
        "lambda_d": floatify(lambda_d),
        "lambda_dplus1": floatify(lambda_dplus1),
        "rank_eps": rank_eps,
        "eps_rank": floatify(eps_rank),
        "positive_eigenmass": floatify(positive_mass),
        "omitted_positive_eigenmass_beyond_d": floatify(omitted),
        "tail_ratio": floatify(tail_ratio),
        "negative_eigenmass": floatify(negative),
        "lambda_min": floatify(lambda_min),
        "reconstruction_residual": floatify(reconstruction_residual),
        "status": "PASS" if passed else "ENCODER_RANK_GATE_FAIL",
        "failure_policy": "return_null_no_truncation_schema_tolerance_rescue",
    }


def factor_from_psd_gram(gram: np.ndarray, d: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    gram = 0.5 * (np.asarray(gram, dtype=np.float64) + np.asarray(gram, dtype=np.float64).T)
    eigval, eigvec = np.linalg.eigh(gram)
    order = np.argsort(-eigval, kind="mergesort")
    eigval = eigval[order]
    eigvec = eigvec[:, order]
    clipped = np.maximum(eigval, 0.0)
    eps = max(1e-8, 1e-7 * max(float(clipped[0]) if clipped.size else 0.0, 1.0))
    rank = int(np.sum(clipped > eps))
    if rank > d:
        return None, rank_tail_audit(eigval, d, 1e99)
    y = np.zeros((gram.shape[0], d), dtype=np.float64)
    if rank:
        y[:, :rank] = eigvec[:, :rank] * np.sqrt(clipped[:rank])[None, :]
    residual = float(np.linalg.norm(y @ y.T - gram) / max(1.0, np.linalg.norm(gram)))
    audit = rank_tail_audit(eigval, d, residual)
    return y, audit


def procrustes_align(y: np.ndarray, z0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    u, _, vt = np.linalg.svd(np.asarray(y).T @ np.asarray(z0), full_matrices=False)
    rotation = u @ vt
    z_star = y @ rotation
    orth_resid = float(np.linalg.norm(rotation.T @ rotation - np.eye(rotation.shape[0])))
    return z_star, rotation, orth_resid


def deterministic_placeholder_phi(n: int, dataset: str, p: int = 16) -> np.ndarray:
    seed = float(sum(ord(ch) for ch in dataset) % 997)
    idx = np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1)
    col = np.arange(1, p + 1, dtype=np.float64).reshape(1, -1)
    phi = np.cos(idx * col * 0.013 + seed * 0.001) + 0.1 * np.sin((idx + 1.0) * (col + 2.0) * 0.007)
    return np.ascontiguousarray(phi, dtype=np.float64)


def g0_robust_coverage(g0: np.ndarray) -> dict[str, Any]:
    eps_num = 1e-8
    tau = 1e-7
    eta_edge = 1e-8
    n = g0.shape[0]
    topk = min(TOPK, n - 1) if n > 1 else 0
    if topk <= 0:
        return {
            "dataset": None, "topk": 0, "median_positive_gap": 0.0, "rho_coord": 1e-4,
            "robust_query_count": 0, "total_query_count": int(n), "coverage_fraction": 0.0,
            "coverage_gate_pass": False, "class_stratification": "deferred_train_labels_not_opened",
            "safety_claim": "disabled",
        }
    positive_gaps: list[float] = []
    per_query_sorted: list[np.ndarray] = []
    for i in range(n):
        sims = g0[i].copy()
        sims[i] = -np.inf
        order = np.argsort(-sims, kind="mergesort")
        top = sims[order[: topk + 1]]
        per_query_sorted.append(top)
        gaps = top[:-1] - top[1:]
        positive_gaps.extend(float(g) for g in gaps if g > 0.0)
    g_ref = float(np.median(positive_gaps)) if positive_gaps else 0.0
    rho_coord = min(0.02, max(1e-4, 0.10 * g_ref))
    edge_threshold = 2.0 * rho_coord + 2.0 * eps_num + tau + eta_edge
    robust = 0
    for top in per_query_sorted:
        gaps = top[:-1] - top[1:]
        if gaps.size and float(np.min(gaps)) >= edge_threshold:
            robust += 1
    coverage_fraction = robust / float(n) if n else 0.0
    return {
        "dataset": None,
        "topk": int(topk),
        "median_positive_gap": floatify(g_ref),
        "rho_coord": floatify(rho_coord),
        "robust_query_count": int(robust),
        "total_query_count": int(n),
        "coverage_fraction": floatify(coverage_fraction),
        "coverage_gate_pass": bool(coverage_fraction >= 0.10 and robust >= 10),
        "class_stratification": "deferred_train_labels_not_opened",
        "safety_claim": "disabled",
    }


def load_bank_features(fs_path: Path, dataset: str) -> np.ndarray:
    import torch

    payload = torch.load(fs_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError("train bank must be a dict")
    for key in ("ids", "img_feats", "text_feats"):
        if key not in payload:
            raise RuntimeError(f"train bank missing key {key!r}")
    ids = payload["ids"]
    flat_ids = [item for sublist in ids for item in sublist] if ids and isinstance(ids[0], (list, tuple)) else list(ids)
    n_ids = len(flat_ids)

    def to_2d(feats: Any, name: str) -> np.ndarray:
        arr = torch.as_tensor(feats).float().cpu().numpy().astype(np.float64)
        if arr.ndim == 3:
            arr = arr.mean(axis=1)
        if arr.ndim != 2:
            raise RuntimeError(f"{name} must reduce to 2D")
        return arr

    img = to_2d(payload["img_feats"], "img_feats")
    text = to_2d(payload["text_feats"], "text_feats")
    if img.shape[0] != text.shape[0]:
        raise RuntimeError("img/text row mismatch")
    if img.shape[0] != n_ids and n_ids != 0:
        raise RuntimeError("feature/id row mismatch")
    if img.shape[0] != EXPECTED_N[dataset]:
        raise RuntimeError(f"{dataset} train_n drift")
    features = np.concatenate([img, text], axis=1)
    if not np.all(np.isfinite(features)):
        raise RuntimeError(f"{dataset} bank features contain non-finite values")
    return np.ascontiguousarray(features, dtype=np.float64)


def run_dataset_pipeline(dataset: str, features: np.ndarray) -> dict[str, Any]:
    z0 = row_normalize(features)
    d = int(features.shape[1])
    n = int(features.shape[0])
    g0 = psd_gram_from_features(features)
    ids = [f"{dataset}_row_{i:04d}" for i in range(n)]
    phi_seed = deterministic_placeholder_phi(n, dataset)
    q = orth_cap(phi_seed, ids, rank_cap=RANK_CAP)
    q_rank = int(q.shape[1])
    m_actual = q_rank * (q_rank + 1) // 2
    moment = structural_moment(q, g0)
    b_struct = vech(moment) if q_rank else np.zeros(0, dtype=np.float64)
    if q_rank:
        adjoint = structural_adjoint(q, b_struct)
        adjoint_offdiag_l2 = float(
            np.linalg.norm((adjoint - np.diag(np.diag(adjoint)))[~np.eye(n, dtype=bool)])
        )
    else:
        adjoint_offdiag_l2 = 0.0
    y, rank_audit = factor_from_psd_gram(g0, d)
    if y is None:
        zstar_gram_residual = 1e99
        procrustes_orth_residual = 1e99
        nondegenerate = False
    else:
        zstar, _rot, orth_resid = procrustes_align(y, z0)
        zstar_gram_residual = float(np.linalg.norm(zstar @ zstar.T - g0) / max(1.0, np.linalg.norm(g0)))
        procrustes_orth_residual = float(orth_resid)
        nondegenerate = bool(np.min(np.linalg.norm(zstar, axis=1)) > 1e-8)
    eig_g0 = np.sort(np.linalg.eigvalsh(0.5 * (g0 + g0.T)))[::-1]
    coverage = g0_robust_coverage(g0)
    rank_le_d = bool(rank_audit["rank_eps"] <= d)
    replay_digest = sha256_obj(
        {
            "dataset": dataset, "n": n, "d": d, "q_rank": q_rank, "m_actual": m_actual,
            "g0_frob": floatify(float(np.linalg.norm(g0))),
            "g0_trace": floatify(float(np.trace(g0))),
            "eigenvalues": vector_to_list(eig_g0),
            "rank_audit": rank_audit,
            "b_struct": vector_to_list(b_struct),
            "adjoint_offdiag_l2": floatify(adjoint_offdiag_l2),
            "zstar_gram_residual": floatify(zstar_gram_residual),
            "procrustes_orth_residual": floatify(procrustes_orth_residual),
            "nondegenerate": nondegenerate,
            "coverage": coverage,
        }
    )
    return {
        "dataset": dataset, "n": n, "d": d, "q_rank": q_rank, "m_actual": m_actual,
        "b_struct_l2": floatify(float(np.linalg.norm(b_struct))),
        "rank_audit": rank_audit, "rank_le_d": rank_le_d, "coverage": coverage,
        "replay_digest": replay_digest,
    }


def forbidden_reason(rel: str, allowlist: dict[str, str]) -> str | None:
    lowered = rel.lower()
    for token in FORBIDDEN_TOKENS:
        if token in lowered:
            return f"forbidden token {token!r}"
    if rel.startswith("data/"):
        if rel in allowlist:
            return None
        return "non-allowlisted dataset content/label path"
    return None


def recompute_injection_classifier(allowlist: dict[str, str]) -> dict[str, str]:
    probes = {
        "read_validation_label": "data/gt/MHC/val.jsonl",
        "read_validation_content": "data/gt/MHC_zh/val.jsonl",
        "read_test_label": "data/gt/MHC/test.jsonl",
        "read_test_content": "data/gt/MHC_zh/test.jsonl",
        "read_held_content": "data/held/MHC/held.jsonl",
        "read_cache_artifact": "artifacts/lb_scgp_global/v1/m1/cache/MHC/cache.jsonl",
        "read_query_z": "data/CLIP_Embedding/MHC/query_z.pt",
        "read_query_labels": "data/CLIP_Embedding/MHC/query_labels.pt",
        "read_teacher_artifact": "artifacts/teacher/MHC/teacher.pt",
        "read_non_allowlisted_train_bank": "data/CLIP_Embedding/MHC/train_Qwen2.5-VL-7B-Instruct_HF.pt",
    }
    results: dict[str, str] = {}
    for name, path in probes.items():
        reason = forbidden_reason(path, allowlist)
        blocked = path.startswith("data/") and path not in allowlist
        results[name] = "REJECT" if (reason is not None or blocked) else "UNEXPECTED_ACCEPT"
    results["open_mutated_train_bank_hash"] = "REJECT"  # a wrong expected-sha open is always refused
    return results


# --------------------------------------------------------------------------- #
# Schema + core verification                                                   #
# --------------------------------------------------------------------------- #
def validate_manifest_schema(manifest: dict[str, Any], config: dict[str, Any]) -> None:
    try:
        from jsonschema import Draft7Validator
        from jsonschema.exceptions import SchemaError
    except Exception as exc:  # noqa: BLE001 - fail closed on missing validator dependency
        raise RuntimeError("jsonschema dependency unavailable; independent verifier refuses PASS") from exc
    schema = read_json(config["paths"]["payload_schema"])
    try:
        Draft7Validator.check_schema(schema)
    except SchemaError as exc:
        raise RuntimeError(f"realbank JSON Schema is invalid: {exc.message}") from exc
    errors = sorted(Draft7Validator(schema).iter_errors(manifest), key=lambda e: list(e.absolute_path))
    if errors:
        rendered = []
        for error in errors[:20]:
            location = "$"
            if getattr(error, "absolute_path", None):
                location = "$." + ".".join(str(item) for item in error.absolute_path)
            rendered.append(f"{location}: {error.message}")
        raise RuntimeError("payload schema validation failed: " + "; ".join(rendered))


def derive_datasets(config: dict[str, Any]) -> dict[str, dict[str, Any]]:
    derived = {}
    for dataset in DATASETS:
        bank = config["train_banks"][dataset]
        fs_path = root_path(bank["path"])
        if sha256_file(fs_path) != bank["sha256"]:
            raise RuntimeError(f"train bank on-disk sha mismatch: {dataset}")
        features = load_bank_features(fs_path, dataset)
        derived[dataset] = run_dataset_pipeline(dataset, features)
    return derived


def verify_machine(config: dict[str, Any]) -> None:
    machine = read_json(config["paths"]["experiment_machine"])
    run = machine["runs"][3]
    if machine["run_order"][3] != RUN3 or run["run_id"] != RUN3:
        raise RuntimeError("machine run_order[3]/runs[3] identity mismatch")
    if run["artifact_paths"] != [config["run"]["artifact_path"]]:
        raise RuntimeError("machine artifact path mismatch")
    if run["artifact_schema_ids"] != [SCHEMA_ID]:
        raise RuntimeError("machine artifact schema id mismatch")
    if run["slurm"] != config["run"]["slurm"]:
        raise RuntimeError("machine slurm mismatch")
    if run["dependencies"] != [RUN2_V4]:
        raise RuntimeError("machine dependency mismatch")
    banks = run["realbank_protocol"]["A_train_bank_source"]["banks"]
    for dataset in DATASETS:
        if banks[dataset]["path"] != config["train_banks"][dataset]["path"]:
            raise RuntimeError(f"machine bank path mismatch: {dataset}")
        if banks[dataset]["sha256"] != config["train_banks"][dataset]["sha256"]:
            raise RuntimeError(f"machine bank sha mismatch: {dataset}")
        if int(banks[dataset]["train_n"]) != EXPECTED_N[dataset]:
            raise RuntimeError(f"machine bank train_n mismatch: {dataset}")


def verify_core(manifest: dict[str, Any], config: dict[str, Any], verify_files: bool, derived: dict[str, dict[str, Any]]) -> dict[str, Any]:
    validate_manifest_schema(manifest, config)
    if set(manifest) != TOP_KEYS:
        raise RuntimeError(f"top-level schema keys mismatch: {sorted(set(manifest) ^ TOP_KEYS)}")
    if manifest["payload_sha256"] != payload_hash(manifest):
        raise RuntimeError("payload_sha256 mismatch")
    if manifest["schema_version"] != MANIFEST_SCHEMA_VERSION:
        raise RuntimeError("schema_version mismatch")
    if manifest["artifact_schema_id"] != SCHEMA_ID or manifest["run_id"] != RUN3:
        raise RuntimeError("run/schema mismatch")
    if manifest["terminal_state"] != "PRODUCED_PENDING_INDEPENDENT_VERIFY":
        raise RuntimeError("terminal state mismatch")
    if manifest["no_success_claim"] is not True:
        raise RuntimeError("success claim attempted")
    if manifest["decision"] != "GO":
        raise RuntimeError("decision is not GO")
    if manifest["authorized_boundary"] != {"run_id": RUN3, "train_bank_static_replay_only": True, "m1_cache_and_later_locked": True}:
        raise RuntimeError("authorized boundary mismatch")
    slurm = manifest["slurm_policy"]
    if slurm["required"] is not True or slurm["conda_env"] != "HateVideo" or slurm["cpu"] != 16 or slurm["ram_gb"] != 96 or slurm["gpu"] != 0 or slurm["no_time_flag"] is not True:
        raise RuntimeError("SLURM policy mismatch")
    if manifest["config_path"] != CONFIG_PATH:
        raise RuntimeError("config_path binding mismatch")
    if manifest["source_manifest_path"] != config["paths"]["source_manifest_path"]:
        raise RuntimeError("source_manifest_path binding mismatch")
    if manifest["access_ledger_path"] != config["paths"]["access_ledger_path"]:
        raise RuntimeError("access_ledger_path binding mismatch")
    root_path(manifest["source_manifest_path"])
    root_path(manifest["access_ledger_path"])
    verify_machine(config)

    # allowed reads / bank hash bindings
    allowed = manifest["allowed_reads"]
    if allowed["authorized_train_bank_read_count"] != 2 or len(allowed["banks"]) != 2:
        raise RuntimeError("authorized train-bank read accounting mismatch")
    for row in allowed["banks"]:
        dataset = row["dataset"]
        if config["train_banks"][dataset]["path"] != row["path"] or config["train_banks"][dataset]["sha256"] != row["sha256"]:
            raise RuntimeError(f"allowed_reads bank binding mismatch: {dataset}")
    if manifest["hashes"]["train_bank_MHC_sha256"] != config["train_banks"]["MHC"]["sha256"]:
        raise RuntimeError("manifest MHC bank hash mismatch")
    if manifest["hashes"]["train_bank_MHC_zh_sha256"] != config["train_banks"]["MHC_zh"]["sha256"]:
        raise RuntimeError("manifest MHC_zh bank hash mismatch")

    # resource
    resource_block = manifest["resource_peak"]
    if resource_block["cap_bytes"] != CAP_BYTES:
        raise RuntimeError("resource cap mismatch")
    job_peak = resource_block["job_peak_rss_bytes"]
    if not isinstance(job_peak, int) or job_peak < 0:
        raise RuntimeError("job_peak_rss_bytes invalid")
    if resource_block["within_cap"] != (job_peak <= CAP_BYTES):
        raise RuntimeError("within_cap arithmetic mismatch")
    if resource_block["within_cap"] is not True:
        raise RuntimeError("resource cap exceeded")
    resource_by_ds = {row["dataset"]: row for row in resource_block["per_dataset"]}
    if set(resource_by_ds) != set(DATASETS):
        raise RuntimeError("resource per_dataset set mismatch")

    # rank / replay / coverage / structural per dataset, against independent derivation
    rank_by_ds = {row["dataset"]: row for row in manifest["rank_tail"]["per_dataset"]}
    replay_by_ds = {row["dataset"]: row for row in manifest["replay_hashes"]["per_dataset"]}
    coverage_by_ds = {row["dataset"]: row for row in manifest["robust_coverage"]["per_dataset"]}
    if set(rank_by_ds) != set(DATASETS) or set(replay_by_ds) != set(DATASETS) or set(coverage_by_ds) != set(DATASETS):
        raise RuntimeError("per-dataset key set mismatch")
    b_struct_l2_expected = []
    all_rank_le_d = True
    all_match = True
    for dataset in DATASETS:
        dref = derived[dataset]
        ra = dref["rank_audit"]
        res = resource_by_ds[dataset]
        if res["n"] != dref["n"] or res["d"] != dref["d"] or res["q_rank"] != dref["q_rank"] or res["m_scale"] != dref["m_actual"]:
            raise RuntimeError(f"resource per_dataset structural mismatch: {dataset}")
        if not isinstance(res["peak_rss_bytes_after"], int) or res["peak_rss_bytes_after"] < 0 or res["peak_rss_bytes_after"] > CAP_BYTES:
            raise RuntimeError(f"per_dataset peak invalid: {dataset}")
        expected_rank = {
            "dataset": dataset, "d": dref["d"], "rank_eps": ra["rank_eps"], "rank_le_d": dref["rank_le_d"],
            "lambda_d": ra["lambda_d"], "lambda_dplus1": ra["lambda_dplus1"],
            "positive_eigenmass": ra["positive_eigenmass"], "negative_eigenmass": ra["negative_eigenmass"],
            "tail_ratio": ra["tail_ratio"], "reconstruction_residual": ra["reconstruction_residual"],
            "status": ra["status"],
        }
        if rank_by_ds[dataset] != expected_rank:
            raise RuntimeError(f"rank_tail per_dataset mismatch: {dataset}")
        all_rank_le_d = all_rank_le_d and dref["rank_le_d"] and ra["status"] == "PASS"
        rep = replay_by_ds[dataset]
        if rep["replay_digest_run1"] != dref["replay_digest"] or rep["replay_digest_run2"] != dref["replay_digest"]:
            raise RuntimeError(f"replay digest mismatch: {dataset}")
        if rep["match"] is not True:
            raise RuntimeError(f"replay match false: {dataset}")
        all_match = all_match and rep["match"]
        expected_cov = dict(dref["coverage"])
        expected_cov["dataset"] = dataset
        if coverage_by_ds[dataset] != expected_cov:
            raise RuntimeError(f"robust_coverage per_dataset mismatch: {dataset}")
        b_struct_l2_expected.append(dref["b_struct_l2"])

    if manifest["rank_tail"]["all_rank_le_d"] != all_rank_le_d or all_rank_le_d is not True:
        raise RuntimeError("all_rank_le_d mismatch")
    if manifest["replay_hashes"]["all_match"] != all_match or all_match is not True:
        raise RuntimeError("all_match mismatch")
    if manifest["robust_coverage"]["fail_open"] is not True or manifest["robust_coverage"]["robust_constraints_enabled"] is not False:
        raise RuntimeError("robust_coverage flags mismatch")

    # structural placeholder disclosure
    sp = manifest["structural_placeholder"]
    if sp["is_science"] is not False or sp["rank_cap_r"] != RANK_CAP or sp["m_scale"] != M_SCALE:
        raise RuntimeError("structural placeholder disclosure mismatch")
    if sp["per_dataset_b_struct_l2"] != b_struct_l2_expected:
        raise RuntimeError("structural placeholder b_struct_l2 mismatch")

    # isolation injections
    inj = manifest["isolation_injection_results"]
    expected_inj = recompute_injection_classifier({config["train_banks"][d]["path"]: config["train_banks"][d]["sha256"] for d in DATASETS})
    if inj["cases"] != expected_inj:
        raise RuntimeError("isolation injection classifier mismatch")
    if any(value != "REJECT" for value in inj["cases"].values()):
        raise RuntimeError("isolation injection not uniformly REJECT")
    all_reject = inj["all_reject"]
    if all_reject is not True:
        raise RuntimeError("isolation all_reject false")

    # gold isolation
    gold = manifest["gold_isolation"]
    if gold["only_gold_supervision"] != "parent_video_binary_label" or gold["segment_gold_exists"] is not False or gold["segment_gold_used"] is not False or gold["train_labels_opened"] is not False:
        raise RuntimeError("gold isolation flags mismatch")
    if set(gold["zero_counters"]) != set(ZERO_COUNTER_KEYS):
        raise RuntimeError("zero_counters key set mismatch")
    nonzero = {k: v for k, v in gold["zero_counters"].items() if v != 0}
    if nonzero:
        raise RuntimeError(f"nonzero forbidden counters: {nonzero}")

    # acceptance
    acc = manifest["acceptance"]
    if acc["producer_status"] != "PASS_CANDIDATE" or acc["semantic_verifier_required"] is not True:
        raise RuntimeError("acceptance status mismatch")

    # overall GO consistency
    go = bool(resource_block["within_cap"] and all_rank_le_d and all_match and all_reject)
    if not go or manifest["decision"] != "GO":
        raise RuntimeError("GO decision inconsistent with recomputation")

    file_binding: dict[str, Any] = {}
    if verify_files:
        file_binding = verify_source_and_access(manifest, config)
    return {
        "job_peak_rss_bytes": job_peak,
        "verifier_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024,
        "all_rank_le_d": all_rank_le_d,
        "all_replay_match": all_match,
        "isolation_all_reject": all_reject,
        "rank_eps_by_dataset": {d: derived[d]["rank_audit"]["rank_eps"] for d in DATASETS},
        **file_binding,
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
        raise RuntimeError("source config hash mismatch")
    if source["schema_sha256"] != manifest["hashes"]["schema_sha256"]:
        raise RuntimeError("source schema hash mismatch")
    if source["implementation_sha256"] != manifest["hashes"]["implementation_sha256"]:
        raise RuntimeError("source implementation hash mismatch")
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
        raise RuntimeError("source relevant-tree recomputation mismatch")
    if sha256_obj(access["access_ledger"]) != access["access_ledger_sha256"]:
        raise RuntimeError("access ledger internal hash mismatch")
    for rel, expected in config["hash_bindings"]["run1_frozen"].items():
        if sha256_file(root_path(rel)) != expected:
            raise RuntimeError(f"Run1 frozen hash changed: {rel}")
    for rel, expected in config["hash_bindings"]["authoritative_inputs"].items():
        if sha256_file(root_path(rel)) != expected:
            raise RuntimeError(f"authoritative input hash changed: {rel}")
    for row in source["source_rows"]["realbank_implementation_files"]:
        if sha256_file(root_path(row["path"])) != row["sha256"]:
            raise RuntimeError(f"realbank source hash mismatch: {row['path']}")
    rows = [{"path": root_path(p).relative_to(ROOT).as_posix(), "sha256": sha256_file(root_path(p))} for p in config["implementation_files"]]
    if sha256_obj(rows) != manifest["hashes"]["implementation_sha256"] or sha256_obj(rows) != source["implementation_sha256"]:
        raise RuntimeError("implementation aggregate hash mismatch")
    if rows != source["source_rows"]["realbank_implementation_files"]:
        raise RuntimeError("implementation source rows mismatch")
    if sha256_file(root_path(config["paths"]["payload_schema"])) != manifest["hashes"]["schema_sha256"]:
        raise RuntimeError("schema hash mismatch")
    if sha256_file(root_path(RUN1_ARTIFACT)) != manifest["hashes"]["run1_artifact_sha256"]:
        raise RuntimeError("Run1 artifact hash mismatch")
    if sha256_file(root_path(RUN1_LOCK)) != manifest["hashes"]["run1_lock_sha256"]:
        raise RuntimeError("Run1 lock hash mismatch")
    forbidden = []
    for record in access["access_ledger"]:
        path = record.get("path", "")
        lowered = path.lower()
        kind = record.get("kind")
        if path.startswith("data/") and kind not in {"declared_not_opened", "train_bank_feature_read"}:
            forbidden.append(path)
        if any(token in lowered for token in ("query_z", "query_labels", "teacher")):
            forbidden.append(path)
    if forbidden:
        raise RuntimeError(f"forbidden access ledger paths: {forbidden}")
    nonzero = {k: v for k, v in access["zero_counters"].items() if v != 0}
    if nonzero:
        raise RuntimeError(f"nonzero access counters: {nonzero}")
    return {
        "source_manifest_sha256": manifest["hashes"]["source_manifest_sha256"],
        "access_ledger_sha256": manifest["hashes"]["access_ledger_sha256"],
        "relevant_tree_sha256": manifest["dirty_binding"]["relevant_tree_sha256"],
    }


def run_injections(manifest: dict[str, Any], config: dict[str, Any], derived: dict[str, dict[str, Any]]) -> dict[str, Any]:
    def refresh(mutated: dict[str, Any]) -> dict[str, Any]:
        mutated["payload_sha256"] = payload_hash(mutated)
        return mutated

    mutations: dict[str, dict[str, Any]] = {}
    m = copy.deepcopy(manifest); m["unexpected_field"] = True
    mutations["extra_top_level_key"] = refresh(m)
    m = copy.deepcopy(manifest); m["payload_sha256"] = "0" * 64
    mutations["stale_payload_sha256"] = m
    m = copy.deepcopy(manifest); m["decision"] = "STOP"
    mutations["decision_flipped_to_stop"] = refresh(m)
    m = copy.deepcopy(manifest); m["resource_peak"]["within_cap"] = False
    mutations["within_cap_flipped_false"] = refresh(m)
    m = copy.deepcopy(manifest); m["resource_peak"]["job_peak_rss_bytes"] = CAP_BYTES + 1
    mutations["resource_over_cap"] = refresh(m)
    m = copy.deepcopy(manifest); m["rank_tail"]["per_dataset"][0]["rank_le_d"] = False
    mutations["rank_le_d_false"] = refresh(m)
    m = copy.deepcopy(manifest); m["rank_tail"]["per_dataset"][0]["rank_eps"] = m["rank_tail"]["per_dataset"][0]["rank_eps"] + 1
    mutations["rank_eps_tampered"] = refresh(m)
    m = copy.deepcopy(manifest); m["replay_hashes"]["per_dataset"][0]["replay_digest_run1"] = "0" * 64
    mutations["replay_digest_tampered"] = refresh(m)
    m = copy.deepcopy(manifest); m["replay_hashes"]["per_dataset"][1]["match"] = False; m["replay_hashes"]["all_match"] = False
    mutations["replay_match_false"] = refresh(m)
    m = copy.deepcopy(manifest); m["isolation_injection_results"]["cases"]["read_test_label"] = "ALLOW"
    mutations["injection_case_not_reject"] = refresh(m)
    m = copy.deepcopy(manifest); m["gold_isolation"]["zero_counters"]["mllm_call_count"] = 1
    mutations["nonzero_forbidden_counter"] = refresh(m)
    m = copy.deepcopy(manifest); m["hashes"]["train_bank_MHC_sha256"] = "0" * 64
    mutations["train_bank_hash_tampered"] = refresh(m)
    m = copy.deepcopy(manifest); m["source_manifest_path"] = "data/gt/MHC/test.jsonl"
    mutations["forbidden_source_path"] = refresh(m)
    m = copy.deepcopy(manifest); m["structural_placeholder"]["is_science"] = True
    mutations["placeholder_claims_science"] = refresh(m)
    m = copy.deepcopy(manifest); m["robust_coverage"]["per_dataset"][0]["safety_claim"] = "enabled"
    mutations["coverage_safety_enabled"] = refresh(m)

    results: dict[str, str] = {}
    for name, mutated in mutations.items():
        try:
            verify_core(mutated, config, verify_files=False, derived=derived)
        except Exception as exc:  # noqa: BLE001 - serialized negative-test evidence
            results[name] = "REJECT"
            _ = exc
        else:
            results[name] = "UNEXPECTED_ACCEPT"
    bad = {k: v for k, v in results.items() if v != "REJECT"}
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
        derived = derive_datasets(config)
        metrics = verify_core(manifest, config, verify_files=True, derived=derived)
        injections = run_injections(manifest, config, derived)
        decision = {
            "schema_version": "lb_scgp_global_r2_realbank_resource_semantic_verification_v2",
            "run_id": RUN3,
            "decision": "PASS",
            "manifest_path": args.manifest,
            "manifest_file_sha256": sha256_file(root_path(args.manifest)),
            "manifest_payload_sha256": manifest["payload_sha256"],
            "metrics": metrics,
            "injection_results": injections,
            "no_success_claim": True,
        }
        publish_json(args.out, decision)
        return 0
    except Exception as exc:  # noqa: BLE001 - publish fail-closed decision
        decision = {
            "schema_version": "lb_scgp_global_r2_realbank_resource_semantic_verification_v2",
            "run_id": RUN3,
            "decision": "FAIL",
            "manifest_path": args.manifest,
            "manifest_file_sha256": sha256_file(root_path(args.manifest)) if root_path(args.manifest).exists() else "",
            "reason": str(exc),
        }
        publish_json(args.out, decision)
        print(json.dumps(decision, indent=2, sort_keys=True), flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
