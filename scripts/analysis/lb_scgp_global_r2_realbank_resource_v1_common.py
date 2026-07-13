#!/usr/bin/env python
"""Realbank static/resource microbenchmark utilities for LB-SCGP Global-R2.

Run LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1.  This module is deliberately
self-contained: the pure numerical/serialization helpers are byte-faithful
copies of the frozen-and-ACCEPTED Run2-v4 synthetic-KKT code
(scripts/analysis/lb_scgp_global_r2_run2_v4_common.py + ..._independent_verify.py),
so the realbank pipeline reuses exactly the verified linear algebra without a
cross-lineage import.  The realbank-specific orchestration (16 CPU / 96 GB SLURM
guard, runs[3] machine verifier, train-bank-aware access ledger, real-bank load,
NON-SCIENCE structural placeholder, resource/rank/replay/coverage/isolation
pipeline, source manifest) is new.

Discipline: this run reads only the two preregistered frozen CLIP-L/336 train
banks (train content, hash-checked) plus repo source/schema/config/machine/run1
files.  It never opens validation/test labels or content, held content, caches,
teacher artifacts, query_z, or query_labels; it opens no train labels; it runs no
MLLM/OCR/model/network/GPU/training; it makes no performance/accuracy claim.  The
only project gold is parent_video_binary_label and it is not read here.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/jehc223/RGCL")
RUN1 = "LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1"
RUN2_V4 = "LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v4"
RUN3 = "LBSCGP-GLOBAL-G0-M0-REALBANK-RESOURCE-v1"
SCHEMA_ID = "scgp_global_realbank_resource_v1"
MANIFEST_SCHEMA_VERSION = "lb_scgp_global_r2_realbank_resource_manifest_v1"
SOURCE_MANIFEST_SCHEMA_VERSION = "lb_scgp_global_r2_realbank_resource_source_manifest_v1"
SEMANTIC_VERIFICATION_SCHEMA_VERSION = "lb_scgp_global_r2_realbank_resource_semantic_verification_v1"

CAP_BYTES = 103079215104  # 96 GiB hard resource STOP
RANK_CAP = 8              # r_max from FINAL_PROPOSAL.md orth_cap
M_SCALE = 36              # r(r+1)/2 with r=8; placeholder structural scale
TOPK = 20                 # ordinary kNN endpoint used for the G0 coverage report
DATASETS = ("MHC", "MHC_zh")
EXPECTED_N = {"MHC": 549, "MHC_zh": 579}
CONFIG_PATH = "configs/lb_scgp_global_r2/m0_realbank_resource_v1.json"
RUN1_ARTIFACT = "artifacts/lb_scgp_global/v1/m0/contract_freeze.json"
RUN1_LOCK = "artifacts/lb_scgp_global/v1/m0/contract_freeze.json.publish.lock"

# Forbidden-read counters that must all remain exactly 0.  Kept identical to the
# strict schema definitions.zero_counters required set (three-way alignment).
ZERO_COUNTER_KEYS = (
    "auxiliary_head_route_count",
    "cache_outer_held_content_read_count",
    "cache_outer_held_label_read_count",
    "cache_read_count",
    "cache_test_content_read_count",
    "cache_test_label_read_count",
    "cache_validation_content_read_count",
    "cache_validation_label_read_count",
    "certificate_read_count",
    "compiler_target_read_count",
    "comparator_freeze_adaptive_query_label_read_count",
    "comparator_freeze_final_test_label_read_count",
    "comparator_freeze_final_test_margin_read_count",
    "comparator_freeze_final_test_prediction_read_count",
    "control_construction_final_test_error_read_count",
    "control_construction_final_test_label_read_count",
    "control_construction_final_test_margin_read_count",
    "control_construction_final_test_prediction_read_count",
    "forbidden_path_read_count",
    "gpu_device_count",
    "held_content_read_count",
    "held_label_read_count",
    "key_selection_route_count",
    "local_v7_pass_evidence_reuse_count",
    "mllm_call_count",
    "mllm_calls_outside_train_cache",
    "model_call_count",
    "network_call_count",
    "non_allowlisted_train_content_read_count",
    "ocr_call_count",
    "pair_triplet_supcon_route_count",
    "performance_evaluation_count",
    "query_labels_read_count",
    "query_z_read_count",
    "reranking_route_count",
    "run3_cache_or_later_attempt_count",
    "sample_weighting_route_count",
    "segment_gold_read_count",
    "teacher_artifact_read_count",
    "teacher_cache_read_count",
    "teacher_cache_write_count",
    "test_content_read_count",
    "test_label_read_count",
    "train_label_read_count",
    "training_call_count",
    "validation_content_read_count",
    "validation_label_read_count",
)


# --------------------------------------------------------------------------- #
# Serialization / hashing helpers (byte-faithful to the frozen Run2-v4 module) #
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


def canonical_root_path(path: Path | str) -> tuple[Path, Path]:
    root = ROOT.resolve()
    raw = Path(path)
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        raise RuntimeError(f"path escapes repository root: {path}") from exc
    return resolved, rel


def read_json(path: Path | str) -> Any:
    fs_path, _ = canonical_root_path(path)
    with open(fs_path, encoding="utf-8") as handle:
        return json.load(handle)


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


def payload_hash(obj: dict[str, Any], field: str = "payload_sha256") -> str:
    clone = dict(obj)
    clone.pop(field, None)
    return sha256_obj(clone)


def _fsync_dir(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_publish_json(path: Path | str, obj: Any) -> None:
    fs_path, _ = canonical_root_path(path)
    fs_path.parent.mkdir(parents=True, exist_ok=True)
    lock = fs_path.with_name(fs_path.name + ".publish.lock")
    lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    tmp = None
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        os.fsync(lock_fd)
        os.close(lock_fd)
        lock_fd = -1
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
        _fsync_dir(fs_path.parent)
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} drift: expected {expected!r}, got {actual!r}")


# --------------------------------------------------------------------------- #
# Numerical core (byte-faithful to the frozen Run2-v4 module)                  #
# --------------------------------------------------------------------------- #
def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise RuntimeError("zero row in row_normalize")
    return matrix / norm


def orth_cap(phi: np.ndarray, ids: list[str], rank_cap: int = 8) -> tuple[np.ndarray, dict[str, Any]]:
    phi = np.asarray(phi, dtype=np.float64)
    centered = phi - phi.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s.size == 0:
        q = np.zeros((phi.shape[0], 0), dtype=np.float64)
        raw_rank = 0
        threshold = 1e-8
    else:
        threshold = max(1e-8, 1e-7 * float(s[0]))
        raw_rank = int(np.sum(s > threshold))
        rank = min(rank_cap, raw_rank)
        q = u[:, :rank].copy()
        for col in range(q.shape[1]):
            pivot = max(range(q.shape[0]), key=lambda row: (abs(float(q[row, col])), str(ids[row])))
            if q[pivot, col] < 0:
                q[:, col] *= -1.0
    return q, {
        "actual_orth_cap_executed": True,
        "rank_cap": int(rank_cap),
        "raw_rank_before_cap": int(raw_rank),
        "q_rank": int(q.shape[1]),
        "threshold": floatify(threshold),
    }


def vech(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    rows, cols = np.tril_indices(matrix.shape[0])
    return matrix[rows, cols]


def structural_moment(q: np.ndarray, gram: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    gram = np.asarray(gram, dtype=np.float64)
    n = gram.shape[0]
    if q.shape[0] != n:
        raise RuntimeError("Q/G shape mismatch")
    if q.shape[1] == 0:
        return np.zeros((0, 0), dtype=np.float64)
    moment = q.T @ (gram - np.eye(n, dtype=np.float64)) @ q / float(n)
    return 0.5 * (moment + moment.T)


def unvech_dual(vec: np.ndarray, rank: int) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64).reshape(-1)
    expected = rank * (rank + 1) // 2
    if vec.size != expected:
        raise RuntimeError(f"vech dual length mismatch: {vec.size} != {expected}")
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
    q = np.asarray(q, dtype=np.float64)
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


def schema_requires_no_additional_properties(schema: Any, path: str = "$") -> list[str]:
    errors: list[str] = []
    if isinstance(schema, dict):
        if (schema.get("type") == "object" or "properties" in schema) and schema.get("additionalProperties") is not False:
            errors.append(path)
        for key, value in schema.items():
            if key in {"properties", "definitions"} and isinstance(value, dict):
                for name, child in value.items():
                    errors.extend(schema_requires_no_additional_properties(child, f"{path}.{key}.{name}"))
            elif key in {"items"}:
                errors.extend(schema_requires_no_additional_properties(value, f"{path}.{key}"))
            elif isinstance(value, dict) and "$ref" not in value:
                errors.extend(schema_requires_no_additional_properties(value, f"{path}.{key}"))
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    errors.extend(schema_requires_no_additional_properties(item, f"{path}.{key}[{idx}]"))
    return errors


def validate_manifest_against_schema(manifest: dict[str, Any], schema_path: Path | str) -> None:
    try:
        from jsonschema import Draft7Validator
        from jsonschema.exceptions import SchemaError
    except Exception as exc:  # noqa: BLE001 - fail closed on missing validator dependency
        raise RuntimeError("jsonschema dependency unavailable; refusing to validate realbank payload") from exc
    schema = read_json(schema_path)
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
        raise RuntimeError("realbank payload schema validation failed: " + "; ".join(rendered))


# --------------------------------------------------------------------------- #
# SLURM / machine-plan guards                                                  #
# --------------------------------------------------------------------------- #
def require_slurm_realbank() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("realbank run must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus and int(cpus) != 16:
        raise RuntimeError(f"realbank requires exactly 16 CPU, got {cpus}")
    mem = os.environ.get("SLURM_MEM_PER_NODE") or os.environ.get("SLURM_MEM_PER_CPU")
    if mem and int(mem) not in {98304, 96000, 96}:
        raise RuntimeError(f"realbank requires 96GB memory allocation, got {mem}")
    for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        value = os.environ.get(key)
        if value and value not in {"0", "(null)", "NoDevFiles"}:
            raise RuntimeError(f"realbank is CPU-only but {key}={value}")


def expected_slurm_block() -> dict[str, Any]:
    return {"cpu": 16, "ram_gb": 96, "gpu": 0, "env": "HateVideo", "no_time_flag": True}


def verify_machine_realbank(cfg: dict[str, Any]) -> dict[str, Any]:
    machine_path = cfg["paths"]["experiment_machine"]
    machine_hash = sha256_file(canonical_root_path(machine_path)[0])
    machine = read_json(machine_path)
    run = machine["runs"][3]
    assert_equal(machine["run_order"][3], RUN3, "machine run_order[3]")
    assert_equal(run["run_id"], RUN3, "machine realbank run_id")
    assert_equal(run["artifact_paths"], [cfg["run"]["artifact_path"]], "machine realbank artifact path")
    assert_equal(run["artifact_schema_ids"], [SCHEMA_ID], "machine realbank schema id")
    assert_equal(run["slurm"], cfg["run"]["slurm"], "machine realbank slurm")
    assert_equal(run["dependencies"], [RUN2_V4], "machine realbank dependency")
    protocol = run["realbank_protocol"]["A_train_bank_source"]["banks"]
    for dataset in DATASETS:
        assert_equal(protocol[dataset]["path"], cfg["train_banks"][dataset]["path"], f"machine bank path {dataset}")
        assert_equal(protocol[dataset]["sha256"], cfg["train_banks"][dataset]["sha256"], f"machine bank sha {dataset}")
        assert_equal(int(protocol[dataset]["train_n"]), EXPECTED_N[dataset], f"machine bank train_n {dataset}")
    return {"machine_sha256": machine_hash, "machine_run_record_run_id": run["run_id"]}


# --------------------------------------------------------------------------- #
# Access ledger (train-bank aware; every val/test/held/cache/... read forbidden) #
# --------------------------------------------------------------------------- #
FORBIDDEN_TOKENS = (
    ("query_z", "query_z_read_count"),
    ("query_labels", "query_labels_read_count"),
    ("teacher", "teacher_artifact_read_count"),
    ("cache", "cache_read_count"),
    ("held", "held_content_read_count"),
    ("certificate", "certificate_read_count"),
)


def forbidden_reason(rel: str, allowlist: dict[str, str]) -> str | None:
    """Return a rejection reason if `rel` is a forbidden read, else None.

    Only the two allowlisted train-bank paths under data/ may be opened; every
    other data/ path (val/test/held content or labels) and any cache / teacher /
    query_z / query_labels / certificate path is forbidden.  Pure classifier: it
    performs no read and mutates no counter.
    """
    lowered = rel.lower()
    for token, _counter in FORBIDDEN_TOKENS:
        if token in lowered:
            return f"forbidden token {token!r} in path"
    if rel.startswith("data/"):
        if rel in allowlist:
            return None
        return "non-allowlisted dataset content/label path"
    return None


class RealbankAccessLedger:
    def __init__(self, allowlist: dict[str, str]) -> None:
        self.records: list[dict[str, Any]] = []
        self.counters = {key: 0 for key in ZERO_COUNTER_KEYS}
        self.allowlist = dict(allowlist)  # rel_path -> expected sha256
        self.authorized_train_bank_read_count = 0
        self.banks: list[dict[str, str]] = []

    def hash_source_file(self, path: Path | str, purpose: str, scope: str) -> str:
        fs_path, rel_path = canonical_root_path(path)
        rel = rel_path.as_posix()
        reason = forbidden_reason(rel, self.allowlist)
        if reason is not None or rel.startswith("data/"):
            raise RuntimeError(f"hash_source_file refuses dataset/forbidden path {rel}: {reason}")
        digest = sha256_file(fs_path)
        self.records.append(
            {"kind": "file_hash", "path": rel, "purpose": purpose, "scope": scope, "sha256": digest}
        )
        return digest

    def open_train_bank(self, dataset: str, path: Path | str, expected_sha256: str) -> Path:
        fs_path, rel_path = canonical_root_path(path)
        rel = rel_path.as_posix()
        if rel not in self.allowlist:
            raise RuntimeError(f"train-bank path not on allowlist: {rel}")
        if self.allowlist[rel] != expected_sha256:
            raise RuntimeError(f"train-bank allowlist sha mismatch for {rel}")
        digest = sha256_file(fs_path)
        if digest != expected_sha256:
            raise RuntimeError(f"train-bank on-disk sha mismatch for {rel}")
        self.authorized_train_bank_read_count += 1
        self.banks.append({"dataset": dataset, "path": rel, "sha256": digest})
        self.records.append(
            {"kind": "train_bank_feature_read", "dataset": dataset, "path": rel,
             "scope": "authorized_train_bank", "sha256": digest}
        )
        return fs_path

    def record_declared_not_opened(self, path: str, purpose: str, sha256: str) -> None:
        self.records.append(
            {"kind": "declared_not_opened", "path": path, "purpose": purpose,
             "scope": "declared_provenance_not_opened", "sha256": sha256}
        )

    def fields(self) -> dict[str, Any]:
        return {
            "access_ledger": self.records,
            "access_ledger_sha256": sha256_obj(self.records),
            "zero_counters": dict(self.counters),
            "authorized_train_bank_read_count": self.authorized_train_bank_read_count,
        }


# --------------------------------------------------------------------------- #
# Real-bank load + NON-SCIENCE structural placeholder + pipeline              #
# --------------------------------------------------------------------------- #
def load_bank_features(fs_path: Path, dataset: str) -> np.ndarray:
    """Load the frozen CLIP-L/336 pooler train bank into Z0 = concat(img,text).

    The bank dict is {ids, img_feats, text_feats, labels}.  Labels are NOT read
    (train labels stay closed).  img/text pooler feats are per-video 2D tensors
    (mean-pooled over frames defensively if a 3D tensor is encountered).
    """
    import torch  # HateVideo-provided; function-level import kept off module load

    payload = torch.load(fs_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"train bank must be a dict, got {type(payload)!r}")
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
            raise RuntimeError(f"{name} must reduce to 2D, got shape {arr.shape}")
        return arr

    img = to_2d(payload["img_feats"], "img_feats")
    text = to_2d(payload["text_feats"], "text_feats")
    if img.shape[0] != text.shape[0]:
        raise RuntimeError(f"img/text row mismatch: {img.shape[0]} vs {text.shape[0]}")
    if img.shape[0] != n_ids and n_ids != 0:
        raise RuntimeError(f"feature/id row mismatch: {img.shape[0]} vs {n_ids}")
    if img.shape[0] != EXPECTED_N[dataset]:
        raise RuntimeError(f"{dataset} train_n drift: {img.shape[0]} != {EXPECTED_N[dataset]}")
    features = np.concatenate([img, text], axis=1)
    if not np.all(np.isfinite(features)):
        raise RuntimeError(f"{dataset} bank features contain non-finite values")
    return np.ascontiguousarray(features, dtype=np.float64)


def deterministic_placeholder_phi(n: int, dataset: str, p: int = 16) -> np.ndarray:
    """Deterministic, label-blind, well-conditioned N x p seed for orth_cap.

    Uses only the row index, column index, and dataset name.  It reads no label,
    no cache, and no MLLM output; it is a NON-SCIENCE placeholder that opens the
    orth_cap / structural-moment code path at the real N so the measured peak RSS
    is faithful to (an upper bound on) the eventual FULL structural target.
    """
    seed = float(sum(ord(ch) for ch in dataset) % 997)
    idx = np.arange(1, n + 1, dtype=np.float64).reshape(-1, 1)
    col = np.arange(1, p + 1, dtype=np.float64).reshape(1, -1)
    phi = np.cos(idx * col * 0.013 + seed * 0.001) + 0.1 * np.sin((idx + 1.0) * (col + 2.0) * 0.007)
    return np.ascontiguousarray(phi, dtype=np.float64)


def run_dataset_pipeline(dataset: str, features: np.ndarray) -> dict[str, Any]:
    """Deterministic resource/rank pipeline over one real train bank.

    Builds G0 = psd_gram_from_features(Z0), exercises the NON-SCIENCE structural
    placeholder (Q via orth_cap, b_struct = vech(M_Q(G0)), dense adjoint), runs
    the rank-tail factor + Procrustes, and returns a deterministic replay digest
    plus the reported per-dataset fields.  No labels, no performance.
    """
    z0 = row_normalize(features)
    d = int(features.shape[1])
    n = int(features.shape[0])
    g0 = psd_gram_from_features(features)

    ids = [f"{dataset}_row_{i:04d}" for i in range(n)]
    phi_seed = deterministic_placeholder_phi(n, dataset)
    q, q_info = orth_cap(phi_seed, ids, rank_cap=RANK_CAP)
    q_rank = int(q.shape[1])
    m_actual = q_rank * (q_rank + 1) // 2
    moment = structural_moment(q, g0)
    b_struct = vech(moment) if q_rank else np.zeros(0, dtype=np.float64)
    if q_rank:
        adjoint = structural_adjoint(q, b_struct)  # N x N dense structural-path allocation
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
            "dataset": dataset,
            "n": n,
            "d": d,
            "q_rank": q_rank,
            "m_actual": m_actual,
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
        "dataset": dataset,
        "n": n,
        "d": d,
        "q_rank": q_rank,
        "m_actual": m_actual,
        "b_struct_l2": floatify(float(np.linalg.norm(b_struct))),
        "rank_audit": rank_audit,
        "rank_le_d": rank_le_d,
        "coverage": coverage,
        "replay_digest": replay_digest,
    }


def g0_robust_coverage(g0: np.ndarray) -> dict[str, Any]:
    """Label-free G0 robust-coverage report (fail-open, safety disabled).

    Follows FINAL_PROPOSAL.md coverage geometry without class stratification:
    class-conditioned counts require train labels, which are intentionally not
    opened, so class stratification is deferred.  Low coverage never fails the
    run; it only disables the robust safety claim.
    """
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
        top = sims[order[: topk + 1]]  # top20 plus the outsider (21st)
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


def isolation_injection_cases(allowlist: dict[str, str]) -> dict[str, str]:
    """Every attempt must be REJECTED by the guard.  Pure probes: no read, no
    counter mutation of the reported ledger; each runs against the classifier /
    a throwaway ledger."""
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
        blocked_as_data = path.startswith("data/") and path not in allowlist
        results[name] = "REJECT" if (reason is not None or blocked_as_data) else "UNEXPECTED_ACCEPT"
    # A mutated-hash open must also be rejected by open_train_bank's sha check.
    mhc_path = "data/CLIP_Embedding/MHC/train_openai_clip-vit-large-patch14-336_HF.pt"
    throwaway = RealbankAccessLedger(allowlist)
    try:
        throwaway.open_train_bank("MHC", mhc_path, "0" * 64)
    except Exception:  # noqa: BLE001 - negative-test evidence
        results["open_mutated_train_bank_hash"] = "REJECT"
    else:
        results["open_mutated_train_bank_hash"] = "UNEXPECTED_ACCEPT"
    return results


# --------------------------------------------------------------------------- #
# Source binding (old-protected snapshot + relevant tree)                      #
# --------------------------------------------------------------------------- #
def old_protected_hash_manifest() -> tuple[str, int]:
    roots = [
        ROOT / "configs/lb_scgp",
        ROOT / "artifacts/lb_scgp",
        ROOT / "refine-logs/lb_scgp",
        ROOT / "scripts/analysis",
        ROOT / "scripts/slurm",
    ]
    paths: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            rel = path.relative_to(ROOT).as_posix()
            if rel.startswith(("configs/lb_scgp/", "artifacts/lb_scgp/", "refine-logs/lb_scgp/")):
                paths.append(path)
            elif (
                path.name.startswith("lb_scgp_")
                and not path.name.startswith("lb_scgp_global_r2_")
                and path.suffix in {".py", ".sbatch"}
            ):
                paths.append(path)
    rows = []
    for path in sorted(set(paths), key=lambda item: item.relative_to(ROOT).as_posix()):
        rel = path.relative_to(ROOT).as_posix()
        rows.append(f"{sha256_file(path)}  {rel}\n")
    return sha256_bytes("".join(rows).encode("utf-8")), len(rows)


def implementation_hashes(paths: list[str]) -> tuple[str, list[dict[str, str]]]:
    rows = []
    for path in paths:
        fs_path, rel = canonical_root_path(path)
        rows.append({"path": rel.as_posix(), "sha256": sha256_file(fs_path)})
    return sha256_obj(rows), rows


def relevant_git_status(paths: list[str]) -> list[str]:
    proc = subprocess.run(
        ["git", "status", "--porcelain", "--"] + paths,
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return [line for line in proc.stdout.splitlines() if line.strip()]


def verify_expected_hashes(paths: dict[str, str]) -> dict[str, str]:
    actual = {}
    for rel, expected in paths.items():
        digest = sha256_file(canonical_root_path(rel)[0])
        assert_equal(digest, expected, f"hash {rel}")
        actual[rel] = digest
    return actual


def build_source_manifest(cfg: dict[str, Any], ledger: RealbankAccessLedger) -> dict[str, Any]:
    authoritative = verify_expected_hashes(cfg["hash_bindings"]["authoritative_inputs"])
    for rel, purpose in [
        (cfg["paths"]["experiment_machine"], "authoritative_input_hash"),
        (cfg["paths"]["experiment_tracker"], "authoritative_input_hash"),
    ]:
        ledger.hash_source_file(rel, purpose, "authoritative_input")
    run1 = verify_expected_hashes(cfg["hash_bindings"]["run1_frozen"])
    for rel, digest in cfg["hash_bindings"]["declared_validation_test_provenance_not_opened"].items():
        ledger.record_declared_not_opened(rel, "validation_test_declared_provenance", digest)
    cfg_hash = ledger.hash_source_file(CONFIG_PATH, "realbank_config_hash", "schema_or_source")
    schema_hash = ledger.hash_source_file(cfg["paths"]["payload_schema"], "realbank_schema_hash", "schema_or_source")
    impl_hash, impl_rows = implementation_hashes(cfg["implementation_files"])
    old_hash, old_count = old_protected_hash_manifest()
    expected_old = cfg["hash_bindings"]["old_protected_pre_snapshot"]
    assert_equal(old_hash, expected_old["manifest_sha256"], "old protected manifest")
    assert_equal(old_count, expected_old["path_count"], "old protected path count")
    source_rows = {
        "authoritative_inputs": authoritative,
        "run1_frozen": run1,
        "realbank_implementation_files": impl_rows,
        "old_protected": {
            "manifest_sha256": old_hash,
            "path_count": old_count,
            "snapshot_scope": expected_old["snapshot_scope"],
        },
        "schemas": {cfg["paths"]["payload_schema"]: schema_hash},
        "train_banks_opened": [dict(row) for row in ledger.banks],
        "declared_validation_test_provenance_not_opened": cfg["hash_bindings"]["declared_validation_test_provenance_not_opened"],
    }
    relevant_paths = sorted(
        set(
            cfg["implementation_files"]
            + list(cfg["hash_bindings"]["authoritative_inputs"])
            + list(cfg["hash_bindings"]["run1_frozen"])
        )
    )
    manifest = {
        "schema_version": SOURCE_MANIFEST_SCHEMA_VERSION,
        "run_id": RUN3,
        "source_rows": source_rows,
        "implementation_sha256": impl_hash,
        "config_sha256": cfg_hash,
        "schema_sha256": schema_hash,
        "relevant_git_status": relevant_git_status(relevant_paths),
        "artifact_outputs_excluded_from_source_binding": True,
        "docs_tracker_post_run_changes_separately_measurable": True,
    }
    manifest["relevant_tree_sha256"] = sha256_obj(
        {
            "source_rows": source_rows,
            "implementation_sha256": impl_hash,
            "relevant_git_status": manifest["relevant_git_status"],
            "artifact_outputs_excluded": True,
        }
    )
    return manifest
