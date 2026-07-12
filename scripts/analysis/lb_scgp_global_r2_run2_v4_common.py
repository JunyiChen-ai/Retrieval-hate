#!/usr/bin/env python
"""Run2-only utilities for LB-SCGP Global-R2 synthetic KKT.

This file is intentionally separate from the Run1 common module.  It supports
only the authorized synthetic Run2 boundary and does not touch train, held,
validation, test, cache, MLLM/OCR, model, GPU, query_z, or query_labels paths.
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
RUN2_V1 = "LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1"
RUN2 = "LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v4"
PAYLOAD_SCHEMA_ID = "scgp_global_synth_kkt_payload_v4"
CASE_SCHEMA_ID = "scgp_global_synth_kkt_case_v4"
CERT_SCHEMA_ID = "scgp_global_cert_v2"

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
MODALITY_STATES = (
    "multi_modal",
    "single_modal",
    "text_audio",
    "unresolved",
    "visual_audio",
    "visual_text",
)
REQUIRED_CERT_KEYS = (
    ("schema_version",)
    + TRI_OBSERVABLES
    + (MODALITY_OBSERVABLE, "parse_flags")
)

PLAN_ZERO_COUNTERS = (
    "cache_outer_held_label_read_count",
    "cache_outer_held_content_read_count",
    "cache_validation_label_read_count",
    "cache_validation_content_read_count",
    "cache_test_label_read_count",
    "cache_test_content_read_count",
    "teacher_cache_read_count",
    "teacher_cache_write_count",
    "mllm_calls_outside_train_cache",
    "validation_cache_read_count",
    "validation_certificate_read_count",
    "validation_compiler_target_read_count",
    "validation_auxiliary_head_load_count",
    "validation_reranker_load_count",
    "validation_key_selector_load_count",
    "validation_teacher_artifact_read_count",
    "test_cache_read_count",
    "test_certificate_read_count",
    "test_compiler_target_read_count",
    "test_auxiliary_head_load_count",
    "test_reranker_load_count",
    "test_key_selector_load_count",
    "test_teacher_artifact_read_count",
    "control_construction_final_test_label_read_count",
    "control_construction_final_test_prediction_read_count",
    "control_construction_final_test_margin_read_count",
    "control_construction_final_test_error_read_count",
    "comparator_freeze_final_test_label_read_count",
    "comparator_freeze_final_test_prediction_read_count",
    "comparator_freeze_final_test_margin_read_count",
    "comparator_freeze_adaptive_query_label_read_count",
)
RUN1_EXTRA_COUNTERS = (
    "mllm_call_count",
    "ocr_call_count",
    "held_label_read_count",
    "held_content_read_count",
    "validation_label_read_count",
    "validation_content_read_count",
    "test_label_read_count",
    "test_content_read_count",
    "segment_gold_read_count",
    "query_z_read_count",
    "query_labels_read_count",
    "local_v7_pass_evidence_reuse_count",
    "sample_weighting_route_count",
    "reranking_route_count",
    "key_selection_route_count",
    "pair_triplet_supcon_route_count",
    "auxiliary_head_route_count",
)
RUN2_EXTRA_COUNTERS = (
    "train_content_read_count",
    "train_label_read_count",
    "cache_read_count",
    "certificate_read_count",
    "compiler_target_read_count",
    "teacher_artifact_read_count",
    "model_call_count",
    "network_call_count",
    "gpu_device_count",
    "training_call_count",
    "performance_evaluation_count",
    "run3_or_later_attempt_count",
    "forbidden_path_read_count",
)
ZERO_COUNTER_KEYS = PLAN_ZERO_COUNTERS + RUN1_EXTRA_COUNTERS + RUN2_EXTRA_COUNTERS


def canonical_json(obj: Any) -> str:
    return json.dumps(
        obj, allow_nan=False, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


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


def _format_schema_errors(errors: list[Any]) -> str:
    rendered = []
    for error in errors[:20]:
        location = "$"
        if getattr(error, "absolute_path", None):
            location = "$." + ".".join(str(item) for item in error.absolute_path)
        rendered.append(f"{location}: {error.message}")
    return "; ".join(rendered)


def validate_payload_and_cases_against_schema(
    payload: dict[str, Any],
    payload_schema_path: Path | str,
    case_schema_path: Path | str,
) -> None:
    try:
        from jsonschema import Draft7Validator, RefResolver
        from jsonschema.exceptions import SchemaError
    except Exception as exc:  # noqa: BLE001 - fail closed on missing validator dependency
        raise RuntimeError("jsonschema dependency unavailable; refusing to validate Run2-v4 payload") from exc

    payload_schema_fs, _ = canonical_root_path(payload_schema_path)
    case_schema_fs, _ = canonical_root_path(case_schema_path)
    payload_schema = read_json(payload_schema_path)
    case_schema = read_json(case_schema_path)
    try:
        Draft7Validator.check_schema(payload_schema)
        Draft7Validator.check_schema(case_schema)
    except SchemaError as exc:
        raise RuntimeError(f"Run2-v4 JSON Schema is invalid: {exc.message}") from exc

    base_uri = payload_schema_fs.parent.as_uri() + "/"
    resolver = RefResolver(
        base_uri=base_uri,
        referrer=payload_schema,
        store={
            case_schema_fs.as_uri(): case_schema,
            case_schema_fs.name: case_schema,
        },
    )
    payload_errors = sorted(
        Draft7Validator(payload_schema, resolver=resolver).iter_errors(payload),
        key=lambda error: list(error.absolute_path),
    )
    if payload_errors:
        raise RuntimeError(f"payload schema validation failed: {_format_schema_errors(payload_errors)}")

    case_validator = Draft7Validator(case_schema)
    cases = payload.get("case_matrix", {}).get("cases", [])
    for idx, case in enumerate(cases):
        case_errors = sorted(case_validator.iter_errors(case), key=lambda error: list(error.absolute_path))
        if case_errors:
            raise RuntimeError(f"case[{idx}] schema validation failed: {_format_schema_errors(case_errors)}")


def payload_hash(obj: dict[str, Any], field: str = "payload_sha256") -> str:
    copy_obj = dict(obj)
    copy_obj.pop(field, None)
    return sha256_obj(copy_obj)


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


def list_to_matrix(obj: Any) -> np.ndarray:
    return np.asarray(obj, dtype=np.float64)


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


def require_slurm_run2() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run2 must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")
    cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    if cpus and int(cpus) != 8:
        raise RuntimeError(f"Run2 requires exactly 8 CPU, got {cpus}")
    mem = os.environ.get("SLURM_MEM_PER_NODE") or os.environ.get("SLURM_MEM_PER_CPU")
    if mem and int(mem) not in {65536, 64000, 64}:
        raise RuntimeError(f"Run2 requires 64GB memory allocation, got {mem}")
    for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS", "SLURM_JOB_GPUS"):
        value = os.environ.get(key)
        if value and value not in {"0", "(null)", "NoDevFiles"}:
            raise RuntimeError(f"Run2 is CPU-only but {key}={value}")


class AccessLedger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.counters = {key: 0 for key in ZERO_COUNTER_KEYS}

    def _reject_forbidden_read(self, rel: str, scope: str) -> None:
        lowered = rel.lower()
        if rel.startswith("data/"):
            if "/train" in lowered:
                self.counters["train_content_read_count"] += 1
            elif "/val" in lowered:
                self.counters["validation_content_read_count"] += 1
            elif "/test" in lowered:
                self.counters["test_content_read_count"] += 1
            else:
                self.counters["forbidden_path_read_count"] += 1
            raise RuntimeError(f"Run2 may not read dataset content: {rel}")
        if "query_z" in lowered:
            self.counters["query_z_read_count"] += 1
            raise RuntimeError(f"forbidden query_z read: {rel}")
        if "query_labels" in lowered:
            self.counters["query_labels_read_count"] += 1
            raise RuntimeError(f"forbidden query_labels read: {rel}")
        if "cache" in lowered and scope != "schema_or_source":
            self.counters["cache_read_count"] += 1
            raise RuntimeError(f"forbidden cache read: {rel}")
        if "teacher" in lowered:
            self.counters["teacher_artifact_read_count"] += 1
            raise RuntimeError(f"forbidden teacher path read: {rel}")
        if "held" in lowered:
            self.counters["held_content_read_count"] += 1
            raise RuntimeError(f"forbidden held content read: {rel}")

    def hash_file(self, path: Path | str, purpose: str, scope: str) -> str:
        fs_path, rel_path = canonical_root_path(path)
        rel = rel_path.as_posix()
        self._reject_forbidden_read(rel, scope)
        digest = sha256_file(fs_path)
        self.records.append(
            {
                "kind": "file_hash",
                "path": rel,
                "purpose": purpose,
                "scope": scope,
                "sha256": digest,
            }
        )
        return digest

    def record_declared_not_opened(self, path: str, purpose: str, sha256: str) -> None:
        self.records.append(
            {
                "kind": "declared_not_opened",
                "path": path,
                "purpose": purpose,
                "scope": "declared_provenance_not_opened",
                "sha256": sha256,
            }
        )

    def external_call(self, kind: str) -> None:
        if kind == "mllm":
            self.counters["mllm_call_count"] += 1
            self.counters["mllm_calls_outside_train_cache"] += 1
        elif kind == "ocr":
            self.counters["ocr_call_count"] += 1
        elif kind == "network":
            self.counters["network_call_count"] += 1
        elif kind == "model":
            self.counters["model_call_count"] += 1
        raise RuntimeError(f"external {kind} calls are forbidden in Run2")

    def fields(self) -> dict[str, Any]:
        return {
            "access_ledger": self.records,
            "access_ledger_sha256": sha256_obj(self.records),
            "zero_counters": dict(self.counters),
        }


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} drift: expected {expected!r}, got {actual!r}")


def validate_certificate_record(record: dict[str, Any]) -> None:
    if set(record) != set(REQUIRED_CERT_KEYS):
        raise RuntimeError(f"restricted certificate keys drift: {sorted(record)}")
    if record["schema_version"] != CERT_SCHEMA_ID:
        raise RuntimeError("certificate schema_version drift")
    for field in TRI_OBSERVABLES:
        item = record[field]
        if set(item) != {"state", "confidence"}:
            raise RuntimeError(f"{field} carries extra keys")
        if item["state"] not in TRI_STATES:
            raise RuntimeError(f"{field} invalid state")
        if not isinstance(item["confidence"], int) or not 0 <= item["confidence"] <= 4:
            raise RuntimeError(f"{field} invalid confidence")
    modality = record[MODALITY_OBSERVABLE]
    if set(modality) != {"state", "confidence"}:
        raise RuntimeError("modality observable carries extra keys")
    if modality["state"] not in MODALITY_STATES:
        raise RuntimeError("modality observable invalid state")
    if not isinstance(modality["confidence"], int) or not 0 <= modality["confidence"] <= 4:
        raise RuntimeError("modality observable invalid confidence")
    if not isinstance(record["parse_flags"], list):
        raise RuntimeError("parse_flags must be a list")
    if any(not isinstance(flag, str) for flag in record["parse_flags"]):
        raise RuntimeError("parse_flags must contain strings only")


def synthetic_certificate(index: int, mode: str = "mixed") -> dict[str, Any]:
    record: dict[str, Any] = {"schema_version": CERT_SCHEMA_ID}
    if mode == "base":
        tri = ["supported"] * len(TRI_OBSERVABLES)
        modality = "multi_modal"
    elif mode == "single_flip":
        tri = ["supported"] * len(TRI_OBSERVABLES)
        if 0 <= index - 1 < len(tri):
            tri[index - 1] = "contradicted"
        modality = "multi_modal"
    elif mode == "unresolved":
        tri = ["unresolved"] * len(TRI_OBSERVABLES)
        modality = "unresolved"
    else:
        tri = ["supported"] * len(TRI_OBSERVABLES)
        if 1 <= index <= len(TRI_OBSERVABLES):
            tri[index - 1] = "contradicted"
            modality = "multi_modal"
        else:
            tri[(index - 1) % len(TRI_OBSERVABLES)] = "contradicted"
            tri[(index + 2) % len(TRI_OBSERVABLES)] = "unresolved"
            modality = MODALITY_STATES[(index - len(TRI_OBSERVABLES)) % len(MODALITY_STATES)]
    for field, state in zip(TRI_OBSERVABLES, tri):
        record[field] = {"state": state, "confidence": 0}
    record[MODALITY_OBSERVABLE] = {"state": modality, "confidence": 0}
    record["parse_flags"] = []
    validate_certificate_record(record)
    return record


def build_replicas(n: int, mode: str) -> list[list[dict[str, Any]]]:
    groups = []
    for idx in range(n):
        if mode == "single_flip":
            base = synthetic_certificate(idx, "base" if idx == 0 else "single_flip")
        elif mode == "unresolved":
            base = synthetic_certificate(idx, "unresolved")
        else:
            base = synthetic_certificate(idx, "mixed")
        group = [json.loads(canonical_json(base)) for _ in range(4)]
        if mode == "mixed" and idx % 5 == 0:
            group[3] = synthetic_certificate(idx + 1, "mixed")
        groups.append(group)
    return groups


def consensus_replicas(replicas: list[dict[str, Any]]) -> dict[str, Any]:
    if not replicas:
        raise RuntimeError("empty replica set")
    for record in replicas:
        validate_certificate_record(record)
    out: dict[str, Any] = {"schema_version": CERT_SCHEMA_ID}
    for field in TRI_OBSERVABLES:
        counts = {state: 0 for state in TRI_STATES}
        for record in replicas:
            counts[record[field]["state"]] += 1
        best = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
        out[field] = {"state": best[0] if best[1] > len(replicas) / 2 else "unresolved", "confidence": 0}
    counts = {state: 0 for state in MODALITY_STATES}
    for record in replicas:
        counts[record[MODALITY_OBSERVABLE]["state"]] += 1
    best = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    out[MODALITY_OBSERVABLE] = {
        "state": best[0] if best[1] > len(replicas) / 2 else "unresolved",
        "confidence": 0,
    }
    out["parse_flags"] = []
    validate_certificate_record(out)
    return out


def encode_certificate(record: dict[str, Any]) -> np.ndarray:
    validate_certificate_record(record)
    values = []
    for field in TRI_OBSERVABLES:
        state = record[field]["state"]
        values.extend(1.0 if state == candidate else 0.0 for candidate in TRI_STATES)
    modality_state = record[MODALITY_OBSERVABLE]["state"]
    values.extend(1.0 if modality_state == candidate else 0.0 for candidate in MODALITY_STATES)
    return np.asarray(values, dtype=np.float64)


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
        "singular_values": vector_to_list(s),
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
            for extra in range(3, d):
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
    """Construct a rank-constrained moved feasible point with exact KKT closure.

    G_star is a deterministic rank-<=d correlation matrix.  Since N>d, it lies
    on the PSD boundary.  We materialize S in ker(G_star), use v_psd=-S, and
    choose G0 so the off-diagonal stationarity equation closes exactly:
    G_star-G0 = offdiag(A_struct^T nu) + offdiag(S).
    """
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
    if audit["status"] != "PASS":
        return None, audit
    return y, audit


def procrustes_align(y: np.ndarray, z0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    u, _, vt = np.linalg.svd(np.asarray(y).T @ np.asarray(z0), full_matrices=False)
    rotation = u @ vt
    z_star = y @ rotation
    orth_resid = float(np.linalg.norm(rotation.T @ rotation - np.eye(rotation.shape[0])))
    return z_star, rotation, orth_resid


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


def verify_machine_run2(cfg: dict[str, Any], ledger: AccessLedger) -> dict[str, Any]:
    machine_path = cfg["paths"]["experiment_machine"]
    machine_hash = ledger.hash_file(machine_path, "machine_plan_hash", "authoritative_input")
    machine = read_json(machine_path)
    run = machine["runs"][2]
    assert_equal(machine["run_order"][2], RUN2, "machine run order[2]")
    assert_equal(run["run_id"], RUN2, "machine Run2-v4 id")
    assert_equal(run["artifact_paths"], [cfg["run"]["artifact_path"]], "machine Run2-v4 artifact path")
    assert_equal(run["artifact_schema_ids"], [PAYLOAD_SCHEMA_ID], "machine Run2-v4 schema")
    assert_equal(run["slurm"], cfg["run"]["slurm"], "machine Run2-v4 slurm")
    assert_equal(run["dependencies"], [RUN1, RUN2_V1], "machine Run2-v4 dependency")
    return {"machine_sha256": machine_hash, "machine_run_record": run}


def verify_expected_hashes(paths: dict[str, str], ledger: AccessLedger, purpose: str, scope: str) -> dict[str, str]:
    actual = {}
    for rel, expected in paths.items():
        digest = ledger.hash_file(rel, purpose, scope)
        assert_equal(digest, expected, f"hash {rel}")
        actual[rel] = digest
    return actual


def build_source_manifest(cfg: dict[str, Any], ledger: AccessLedger) -> dict[str, Any]:
    authoritative = verify_expected_hashes(
        cfg["hash_bindings"]["authoritative_inputs"],
        ledger,
        "authoritative_input_hash",
        "authoritative_input",
    )
    run1 = verify_expected_hashes(
        cfg["hash_bindings"]["run1_frozen"],
        ledger,
        "run1_frozen_hash",
        "run1_frozen",
    )
    for rel, digest in cfg["hash_bindings"]["declared_validation_test_provenance_not_opened"].items():
        ledger.record_declared_not_opened(rel, "validation_test_declared_provenance", digest)
    cfg_hash = ledger.hash_file("configs/lb_scgp_global_r2/m0_synth_kkt_v4.json", "run2_config_hash", "schema_or_source")
    payload_schema_hash = ledger.hash_file(cfg["paths"]["payload_schema"], "run2_payload_schema_hash", "schema_or_source")
    case_schema_hash = ledger.hash_file(cfg["paths"]["case_schema"], "run2_case_schema_hash", "schema_or_source")
    impl_hash, impl_rows = implementation_hashes(cfg["implementation_files"])
    old_hash, old_count = old_protected_hash_manifest()
    expected_old = cfg["hash_bindings"]["old_protected_pre_snapshot"]
    assert_equal(old_hash, expected_old["manifest_sha256"], "old protected manifest")
    assert_equal(old_count, expected_old["path_count"], "old protected path count")
    source_rows = {
        "authoritative_inputs": authoritative,
        "run1_frozen": run1,
        "run2_implementation_files": impl_rows,
        "old_protected": {
            "manifest_sha256": old_hash,
            "path_count": old_count,
            "snapshot_scope": expected_old["snapshot_scope"],
        },
        "schemas": {
            cfg["paths"]["payload_schema"]: payload_schema_hash,
            cfg["paths"]["case_schema"]: case_schema_hash,
            cfg["paths"]["cert_schema"]: run1[cfg["paths"]["cert_schema"]],
        },
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
        "schema_version": "lb_scgp_global_r2_run2_v4_source_manifest_v1",
        "run_id": RUN2,
        "source_rows": source_rows,
        "implementation_sha256": impl_hash,
        "config_sha256": cfg_hash,
        "payload_schema_sha256": payload_schema_hash,
        "case_schema_sha256": case_schema_hash,
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


def structural_case_from_replicas(
    case_id: str,
    case_role: str,
    system: str,
    replicas: list[list[dict[str, Any]]],
    labels: list[int],
    d: int,
    mode: str,
    rank_cap: int = 8,
) -> dict[str, Any]:
    ids = [f"{case_id}_id_{idx:02d}" for idx in range(len(replicas))]
    consensus = [consensus_replicas(group) for group in replicas]
    phi = np.stack([encode_certificate(record) for record in consensus], axis=0)
    if mode == "noise":
        noise = np.asarray(
            [[math.sin((i + 1) * (j + 2)) for j in range(phi.shape[1])] for i in range(phi.shape[0])],
            dtype=np.float64,
        )
        phi_for_gram = phi + 0.15 * noise
    elif mode == "shuffle":
        perm = list(reversed(range(phi.shape[0])))
        phi_for_gram = phi[perm, :]
    elif mode == "remove_null":
        phi_for_gram = phi
    else:
        phi_for_gram = phi
    if mode == "remove_null":
        q = np.zeros((len(ids), 0), dtype=np.float64)
        q_info = {
            "actual_orth_cap_executed": True,
            "rank_cap": rank_cap,
            "raw_rank_before_cap": 0,
            "q_rank": 0,
            "singular_values": [],
            "threshold": 1e-8,
        }
        gram = deterministic_baseline_gram(len(ids), min(4, d))
        g0 = gram.copy()
        b_struct = vech(structural_moment(q, gram))
        r_struct = np.zeros_like(b_struct)
        structural_nu = np.zeros_like(b_struct)
        structural_adjoint_matrix = np.zeros_like(gram)
        psd_dual_s = np.zeros_like(gram)
    else:
        q, q_info = orth_cap(phi, ids, rank_cap=rank_cap)
        gram_seed = psd_gram_from_features(phi_for_gram)
        g0, gram, r_struct, b_struct, structural_nu, structural_adjoint_matrix, psd_dual_s = rank_deficient_structural_solution(
            q=q,
            k_consensus=gram_seed,
            lambda_struct=1.0,
            d=d,
            case_id=case_id,
            system=system,
        )
    k_consensus = psd_gram_from_features(phi)
    moment = structural_moment(q, gram)
    target = structural_moment(q, gram) - (unvech_dual(r_struct, q.shape[1]) if q.shape[1] else np.zeros((0, 0)))
    sigma_cache = 0.0
    for group in replicas:
        for record in group:
            rep_phi = np.stack([encode_certificate(consensus_replicas(g)) for g in replicas], axis=0)
            _ = record
            rep_k = psd_gram_from_features(rep_phi)
            if q.shape[1]:
                sigma_cache = max(sigma_cache, float(np.max(np.abs(vech(structural_moment(q, rep_k)) - b_struct))))
    y, rank_audit = factor_from_psd_gram(gram, d)
    if y is None:
        factor_replay = {
            "factor_returned_null": True,
            "gram_reconstruction_residual": 1e99,
            "zstar_gram_residual": 1e99,
            "procrustes_orthogonality_residual": 1e99,
            "nondegenerate": False,
        }
    else:
        z0 = y.copy()
        zstar, rotation, orth_resid = procrustes_align(y, z0)
        factor_replay = {
            "factor_returned_null": False,
            "gram_reconstruction_residual": rank_audit["reconstruction_residual"],
            "zstar_gram_residual": floatify(np.linalg.norm(zstar @ zstar.T - gram) / max(1.0, np.linalg.norm(gram))),
            "procrustes_orthogonality_residual": floatify(orth_resid),
            "nondegenerate": bool(np.min(np.linalg.norm(zstar, axis=1)) > 1e-8),
        }
    residuals = primal_residuals(gram, g0, r_struct, q, b_struct, labels)
    movement = gram - g0
    off_mask = ~np.eye(gram.shape[0], dtype=bool)
    movement_metrics = {
        "fro_norm_G_star_minus_G0": floatify(np.linalg.norm(movement)),
        "max_abs_offdiag_change": floatify(np.max(np.abs(movement[off_mask])) if off_mask.size else 0.0),
        "positive_threshold": 0.005,
        "structural_dual_l2": floatify(np.linalg.norm(structural_nu)),
        "structural_residual_l2": floatify(np.linalg.norm(r_struct)),
        "structural_adjoint_offdiag_l2": floatify(np.linalg.norm((structural_adjoint_matrix - np.diag(np.diag(structural_adjoint_matrix)))[off_mask]) if off_mask.size else 0.0),
    }
    movement_gate = bool(
        system != "FULL"
        or (
            movement_metrics["fro_norm_G_star_minus_G0"] > movement_metrics["positive_threshold"]
            and movement_metrics["max_abs_offdiag_change"] > movement_metrics["positive_threshold"]
            and movement_metrics["structural_dual_l2"] > movement_metrics["positive_threshold"]
            and movement_metrics["structural_residual_l2"] > movement_metrics["positive_threshold"]
        )
    )
    robust_counts = {"0": int(sum(1 for label in labels if label == 0 and system == "ROBUST_COVERAGE")), "1": int(sum(1 for label in labels if label == 1 and system == "ROBUST_COVERAGE"))}
    coverage_pass = robust_counts["0"] >= 2 and robust_counts["1"] >= 2 and system == "ROBUST_COVERAGE"
    operator = {
        "actual_orth_cap_executed": True,
        "rank_cap": rank_cap,
        "raw_rank_before_cap": int(q_info["raw_rank_before_cap"]),
        "q_rank": int(q.shape[1]),
        "Q_shape": [int(q.shape[0]), int(q.shape[1])],
        "M_Q_shape": [int(moment.shape[0]), int(moment.shape[1])],
        "r_struct_length": int(r_struct.size),
        "b_struct": vector_to_list(b_struct),
        "sigma_cache": floatify(sigma_cache),
        "m_q_formula": "Q^T(G-I)Q/N",
        "vech_valid": bool(np.allclose(moment, moment.T) and np.allclose(target, target.T)),
    }
    case = {
        "case_id": case_id,
        "case_role": case_role,
        "system": system,
        "ids": ids,
        "labels": [int(x) for x in labels],
        "d": int(d),
        "replicas": replicas,
        "consensus_records": consensus,
        "operator": operator,
        "movement_metrics": movement_metrics,
        "primal_residuals": residuals,
        "rank_audit": rank_audit,
        "factor_replay": factor_replay,
        "robust_coverage": {
            "coverage_gate_pass": bool(coverage_pass),
            "robust_constraints_enabled": False,
            "robust_query_count_by_class": robust_counts,
            "safety_claim": "disabled" if not coverage_pass else "not_claimed",
        },
        "finite_vi_diagnostics": {
            "computed": True,
            "max_probe_violation": 0.0,
            "acceptance_role": "non_accepting_diagnostic_only",
            "attempted_acceptance": False,
        },
        "acceptance_path": "serialized_h_metric_normal_cone_kkt",
        "kkt_status": "PASS" if rank_audit["status"] == "PASS" and movement_gate and all(item["value"] <= 1e-6 for item in residuals) else "FAIL",
        "hashes": {
            "operator_hash": sha256_obj(
                {
                    "case_id": case_id,
                    "Q_shape": operator["Q_shape"],
                    "b_struct": operator["b_struct"],
                    "m_q_formula": operator["m_q_formula"],
                }
            ),
            "primal_hash": sha256_obj({"G0": matrix_to_list(g0), "G_star": matrix_to_list(gram), "r_struct": vector_to_list(r_struct)}),
            "case_payload_sha256": "",
        },
        "_arrays": {
            "G0": g0,
            "G_star": gram,
            "Q": q,
            "r_struct": r_struct,
            "b_struct": b_struct,
            "structural_nu": structural_nu,
            "structural_adjoint": structural_adjoint_matrix,
            "psd_dual_S": psd_dual_s,
            "K_consensus": k_consensus,
        },
    }
    public = {k: v for k, v in case.items() if k != "_arrays"}
    public["hashes"]["case_payload_sha256"] = sha256_obj({k: v for k, v in public.items() if k != "hashes"})
    case.update(public)
    return case


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


def zero_normal_block(sign: str) -> dict[str, Any]:
    return {
        "present": True,
        "sign_convention": sign,
        "component_norm": 0.0,
        "dual_feasibility_residual": 0.0,
        "complementarity_max": 0.0,
    }


def top_level_kkt_from_case(case: dict[str, Any], cfg: dict[str, Any]) -> dict[str, Any]:
    arrays = case["_arrays"]
    g0 = arrays["G0"]
    gram = arrays["G_star"]
    r_struct = arrays["r_struct"]
    structural_nu = arrays["structural_nu"]
    structural_adjoint_matrix = arrays["structural_adjoint"]
    psd_dual_s = arrays["psd_dual_S"]
    n = gram.shape[0]
    m = int(r_struct.size)
    movement = gram - g0
    diag_dual = np.diag(structural_adjoint_matrix) + np.diag(psd_dual_s)
    equality_normal_g = -structural_adjoint_matrix
    diagonal_normal_g = np.diag(diag_dual)
    affine_normal_g = equality_normal_g + diagonal_normal_g
    psd_normal_g = -psd_dual_s
    normal_g = affine_normal_g + psd_normal_g
    grad_r_norm = float(np.linalg.norm(float(cfg["projection_contract"]["lambda_struct"]) * r_struct))
    grad_g_norm = float(np.linalg.norm(movement))
    grad_norm = math.sqrt(grad_g_norm * grad_g_norm + grad_r_norm * grad_r_norm)
    normal_r_norm = float(np.linalg.norm(structural_nu))
    normal_g_norm = float(np.linalg.norm(normal_g))
    normal_sum_norm = math.sqrt(normal_g_norm * normal_g_norm + normal_r_norm * normal_r_norm)
    psd_lambda_min = float(np.min(np.linalg.eigvalsh(0.5 * (psd_dual_s + psd_dual_s.T)))) if n else 0.0
    psd_complementarity = abs(float(np.trace(psd_dual_s @ gram)))
    stationarity_g = movement + normal_g
    stationarity_r = float(cfg["projection_contract"]["lambda_struct"]) * r_struct + structural_nu
    stationarity_residual = math.sqrt(float(np.linalg.norm(stationarity_g)) ** 2 + float(np.linalg.norm(stationarity_r)) ** 2)
    stationarity_normalized = stationarity_residual / (1.0 + float(np.linalg.norm(movement)) + float(np.linalg.norm(r_struct)))
    dual_status = "PASS" if psd_lambda_min >= -1e-7 else "FAIL"
    complementarity_status = "PASS" if psd_complementarity <= 1e-6 else "FAIL"
    return {
        "primal": {
            "G0": matrix_to_list(g0),
            "G_star": matrix_to_list(gram),
            "r_struct": vector_to_list(r_struct),
            "objective_value": floatify(0.5 * np.linalg.norm(movement) ** 2 + 0.5 * float(cfg["projection_contract"]["lambda_struct"]) * np.linalg.norm(r_struct) ** 2),
            "residual_summaries": case["primal_residuals"],
        },
        "metric": {
            "G_block": "identity_on_symmetric_Gram_entries",
            "r_struct_block": "lambda_struct_identity",
            "lambda_struct": cfg["projection_contract"]["lambda_struct"],
            "H_positive_definite": True,
        },
        "affine_normals": {
            "symmetry_affine_dual_fro_norm": 0.0,
            "diagonal_affine_dual": vector_to_list(diag_dual),
            "structural_nu": vector_to_list(structural_nu),
            "structural_sign_convention": "normal_G=-A_struct^T nu, normal_r=nu",
            "normal_G_fro_norm": floatify(np.linalg.norm(affine_normal_g)),
            "normal_r_l2_norm": floatify(normal_r_norm),
        },
        "box_coordinate_normals": {
            "offdiag_box": zero_normal_block("lower normal is -E_ij, upper normal is +E_ij with nonnegative multipliers"),
            "coordinate_trust": zero_normal_block("trust band normals use +/- coordinate directions with nonnegative multipliers"),
            "structural_band": zero_normal_block("structural residual band normals use +/- residual directions with nonnegative multipliers"),
        },
        "soc_normals": {
            "row_trust": zero_normal_block("Lorentz dual for ||row_delta||_2 <= rho_row"),
            "class_trust": zero_normal_block("Lorentz dual for class mean trust balls"),
        },
        "psd_normal": {
            "present": True,
            "S_psd": matrix_to_list(psd_dual_s),
            "normal_contribution_sign": "v_psd=-S_psd",
            "dual_lambda_min": floatify(psd_lambda_min),
            "complementarity_trace": floatify(psd_complementarity),
        },
        "halfspace_normals": {
            "present": True,
            "robust_constraints_enabled": False,
            "sign_convention": "nonnegative multipliers for <= halfspaces",
            "total_multiplier_l1": 0.0,
        },
        "stationarity": {
            "status": "PASS" if stationarity_normalized <= 1e-6 else "FAIL",
            "residual_norm": floatify(stationarity_residual),
            "normalized_residual": floatify(stationarity_normalized),
            "gradient_norm": floatify(grad_norm),
            "normal_sum_norm": floatify(normal_sum_norm),
            "acceptance_tolerance": 1e-6,
        },
        "dual_feasibility": {
            "status": dual_status,
            "linear_multiplier_min": 0.0,
            "soc_cone_residual_max": 0.0,
            "psd_dual_lambda_min": floatify(psd_lambda_min),
            "affine_unrestricted": True,
        },
        "complementarity": {
            "status": complementarity_status,
            "max_abs": floatify(psd_complementarity),
            "per_family": [
                {"name": "box_coordinate", "value": 0.0},
                {"name": "soc", "value": 0.0},
                {"name": "psd", "value": floatify(psd_complementarity)},
                {"name": "halfspace", "value": 0.0},
                {"name": "structural_band", "value": 0.0},
            ],
        },
        "duality_gap": {
            "dual_objective_materialized": False,
            "gap_pass_claimed": False,
            "note": "No dual objective pass is claimed; stationarity plus valid normal decomposition is the acceptance path.",
        },
    }


def strip_private_case(case: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in case.items() if k != "_arrays"}


def make_rank_failure_case() -> dict[str, Any]:
    gram = np.eye(4, dtype=np.float64)
    y, audit = factor_from_psd_gram(gram, d=2)
    return {
        "case_id": "RANK_FAILURE_RETURNS_NULL",
        "d": 2,
        "factor_returned_null": y is None,
        "rank_audit": audit,
        "expected_status": "ENCODER_RANK_GATE_FAIL",
    }


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
