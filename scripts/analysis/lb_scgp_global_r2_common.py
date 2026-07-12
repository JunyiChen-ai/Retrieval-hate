#!/usr/bin/env python
"""Global-R2 M0 contract primitives for LB-SCGP.

This namespace is separate from the retired local LB-SCGP v1-v7 code.  M0 uses
only synthetic fixtures plus manifest/provenance hashing.  It has no MLLM,
OCR, GPU, training, validation, held, test, query_z, or query_labels route.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/data/jehc223/RGCL")
RUN1 = "LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1"
CONTRACT_SCHEMA_ID = "scgp_global_contract_freeze_v1"
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
ZERO_COUNTER_KEYS = PLAN_ZERO_COUNTERS + RUN1_EXTRA_COUNTERS
FORBIDDEN_ROUTE_NAMES = (
    "local-v7",
    "sample weighting",
    "reranking",
    "key selection",
    "pair/triplet/SupCon",
    "auxiliary head",
    "test teacher",
    "segment route",
)


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


def require_slurm_cpu() -> None:
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run1 validation and freeze must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")
    for key in ("SLURM_GPUS", "SLURM_GPUS_ON_NODE", "SLURM_STEP_GPUS"):
        value = os.environ.get(key)
        if value and value not in {"0", "(null)"}:
            raise RuntimeError(f"Run1 is CPU-only but {key}={value}")


class AccessLedger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.counters = {key: 0 for key in ZERO_COUNTER_KEYS}

    def _reject_forbidden_read(self, rel: str, scope: str) -> None:
        lowered = rel.lower()
        if "query_z" in lowered:
            self.counters["query_z_read_count"] += 1
            raise RuntimeError(f"forbidden query_z read: {rel}")
        if "query_labels" in lowered:
            self.counters["query_labels_read_count"] += 1
            raise RuntimeError(f"forbidden query_labels read: {rel}")
        if scope not in {"declared_provenance_not_opened", "old_protected_hash_only"}:
            if "/val" in lowered or "validation" in lowered:
                self.counters["validation_content_read_count"] += 1
                raise RuntimeError(f"forbidden validation content read: {rel}")
            if "/test" in lowered or "test" in lowered:
                self.counters["test_content_read_count"] += 1
                raise RuntimeError(f"forbidden test content read: {rel}")
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

    def external_call(self, kind: str) -> None:
        if kind == "mllm":
            self.counters["mllm_call_count"] += 1
            self.counters["mllm_calls_outside_train_cache"] += 1
        elif kind == "ocr":
            self.counters["ocr_call_count"] += 1
        raise RuntimeError(f"external {kind} calls are forbidden in Run1")

    def fields(self) -> dict[str, Any]:
        return {
            "access_ledger": self.records,
            "access_ledger_sha256": sha256_obj(self.records),
            "zero_counters": dict(self.counters),
        }


def payload_hash(obj: dict[str, Any], field: str = "payload_sha256") -> str:
    copy_obj = dict(obj)
    copy_obj.pop(field, None)
    return sha256_obj(copy_obj)


def implementation_hashes(paths: list[str]) -> tuple[str, list[dict[str, str]]]:
    rows = []
    for path in paths:
        fs_path, rel = canonical_root_path(path)
        rows.append({"path": rel.as_posix(), "sha256": sha256_file(fs_path)})
    return sha256_obj(rows), rows


def git_dirty_hash() -> dict[str, str]:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    diff = subprocess.check_output(["git", "diff", "--binary", "--", "."], cwd=ROOT)
    staged = subprocess.check_output(["git", "diff", "--cached", "--binary", "--", "."], cwd=ROOT)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines()
    h = hashlib.sha256(diff + b"\0STAGED\0" + staged)
    for rel in sorted(untracked):
        path = ROOT / rel
        h.update(rel.encode("utf-8") + b"\0")
        if path.is_file():
            h.update(bytes.fromhex(sha256_file(path)))
    return {"git_head": head, "dirty_tree_sha256": h.hexdigest()}


def validate_certificate_record(record: dict[str, Any]) -> None:
    if tuple(record.keys()) != tuple(sorted(record.keys())):
        # The parser accepts semantic JSON order, but canonical fixtures should
        # sort keys to make schema drift visible during Run1.
        pass
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


def certificate_feature_names() -> list[str]:
    names = []
    for field in TRI_OBSERVABLES:
        names.extend(f"{field}:{state}" for state in TRI_STATES)
    names.extend(f"{MODALITY_OBSERVABLE}:{state}" for state in MODALITY_STATES)
    return names


def encode_certificate(record: dict[str, Any]) -> np.ndarray:
    validate_certificate_record(record)
    values = []
    for field in TRI_OBSERVABLES:
        state = record[field]["state"]
        values.extend(1.0 if state == candidate else 0.0 for candidate in TRI_STATES)
    modality_state = record[MODALITY_OBSERVABLE]["state"]
    values.extend(1.0 if modality_state == candidate else 0.0 for candidate in MODALITY_STATES)
    return np.asarray(values, dtype=np.float64)


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
        state = best[0] if best[1] > len(replicas) / 2 else "unresolved"
        out[field] = {"state": state, "confidence": 0}
    counts = {state: 0 for state in MODALITY_STATES}
    for record in replicas:
        counts[record[MODALITY_OBSERVABLE]["state"]] += 1
    best = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[0]
    state = best[0] if best[1] > len(replicas) / 2 else "unresolved"
    out[MODALITY_OBSERVABLE] = {"state": state, "confidence": 0}
    out["parse_flags"] = []
    validate_certificate_record(out)
    return out


def row_normalize(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norm <= 0):
        raise RuntimeError("certificate encoding produced a zero row")
    return matrix / norm


def orth_cap(phi: np.ndarray, ids: list[str], rank_cap: int = 8) -> np.ndarray:
    phi = np.asarray(phi, dtype=np.float64)
    if phi.ndim != 2:
        raise RuntimeError("Phi must be two dimensional")
    centered = phi - phi.mean(axis=0, keepdims=True)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    if s.size == 0:
        return np.zeros((phi.shape[0], 0), dtype=np.float64)
    threshold = max(1e-8, 1e-7 * float(s[0]))
    rank = min(rank_cap, int(np.sum(s > threshold)))
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
    q = np.asarray(q, dtype=np.float64)
    gram = np.asarray(gram, dtype=np.float64)
    if gram.shape[0] != gram.shape[1] or gram.shape[0] != q.shape[0]:
        raise RuntimeError("Q/G shape mismatch")
    n = gram.shape[0]
    moment = q.T @ (gram - np.eye(n, dtype=np.float64)) @ q / float(n)
    return 0.5 * (moment + moment.T)


def structural_operator_summary(q: np.ndarray, gram: np.ndarray, k_consensus: np.ndarray) -> dict[str, Any]:
    moment = structural_moment(q, gram)
    target = structural_moment(q, k_consensus)
    residual = vech(moment - target)
    return {
        "Q_shape": [int(q.shape[0]), int(q.shape[1])],
        "M_Q_shape": [int(moment.shape[0]), int(moment.shape[1])],
        "b_struct_length": int(vech(target).size),
        "r_struct_length": int(residual.size),
        "operator_materialization": "implicit_QT_G_minus_I_Q_over_N",
        "vech_valid": bool(np.allclose(moment, moment.T)),
    }


@dataclass(frozen=True)
class ProjectionConfig:
    box_delta: float = 1e-4
    lambda_struct: float = 1.0
    row_trust_scale: float = 0.05
    class_trust_scale: float = 0.02
    robust_enabled: bool = False


def global_projection_interface(n: int, labels: list[int], q_rank: int, cfg: ProjectionConfig) -> dict[str, Any]:
    if cfg.lambda_struct <= 0:
        raise RuntimeError("lambda_struct must be positive for strong convexity")
    if cfg.robust_enabled:
        robust_status = "enabled_only_after_coverage_gate"
    else:
        robust_status = "disabled_fail_open_for_geometry"
    return {
        "variable": "X=(G,r_struct)",
        "n": int(n),
        "q_rank": int(q_rank),
        "objective": "0.5||G-G0||_F^2 + 0.5 lambda_struct ||r_struct||_2^2",
        "strongly_convex_metric": {"G": "identity", "r_struct": cfg.lambda_struct},
        "constraints": [
            "G symmetric",
            "diag(G)=1",
            "G PSD",
            "-1+delta <= G_ij <= 1-delta for i!=j",
            "|G_ij-G0_ij| <= rho_coord for i!=j",
            "row trust balls",
            "class mean trust balls for parent_video_binary_label classes",
            "r_struct=A_struct vec(G)-b_struct",
            "optional robust halfspaces only after coverage gate"
        ],
        "box_delta": cfg.box_delta,
        "row_trust_radius": cfg.row_trust_scale * math.sqrt(max(n - 1, 1)),
        "class_trust_radius_by_class": {
            str(cls): cfg.class_trust_scale * math.sqrt(max(n, 1))
            for cls in sorted(set(int(x) for x in labels))
        },
        "robust_constraints": robust_status,
        "ambiguous_case_policy": {
            "geometry": "fail_open",
            "claims": "fail_closed"
        },
    }


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
        "lambda_d": lambda_d,
        "lambda_dplus1": lambda_dplus1,
        "rank_eps": rank_eps,
        "eps_rank": eps_rank,
        "positive_eigenmass": positive_mass,
        "omitted_positive_eigenmass_beyond_d": omitted,
        "tail_ratio": tail_ratio,
        "negative_eigenmass": negative,
        "lambda_min": lambda_min,
        "reconstruction_residual": float(reconstruction_residual),
        "status": "PASS" if passed else "ENCODER_RANK_GATE_FAIL",
        "failure_policy": "null_no_truncation_schema_tolerance_rescue",
    }


def factor_from_psd_gram(gram: np.ndarray, d: int) -> tuple[np.ndarray | None, dict[str, Any]]:
    gram = 0.5 * (np.asarray(gram, dtype=np.float64) + np.asarray(gram, dtype=np.float64).T)
    eigval, eigvec = np.linalg.eigh(gram)
    order = np.argsort(-eigval, kind="mergesort")
    eigval = eigval[order]
    eigvec = eigvec[:, order]
    clipped = np.maximum(eigval, 0.0)
    rank = int(np.sum(clipped > max(1e-8, 1e-7 * max(float(clipped[0]), 1.0))))
    if rank > d:
        audit = rank_tail_audit(eigval, d, float("inf"))
        return None, audit
    y = np.zeros((gram.shape[0], d), dtype=np.float64)
    if rank:
        y[:, :rank] = eigvec[:, :rank] * np.sqrt(clipped[:rank])[None, :]
    residual = float(np.linalg.norm(y @ y.T - gram) / max(1.0, np.linalg.norm(gram)))
    audit = rank_tail_audit(eigval, d, residual)
    if audit["status"] != "PASS":
        return None, audit
    return y, audit


def procrustes_align(y: np.ndarray, z0: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    y = np.asarray(y, dtype=np.float64)
    z0 = np.asarray(z0, dtype=np.float64)
    u, _, vt = np.linalg.svd(y.T @ z0, full_matrices=False)
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
    payload = "".join(rows).encode("utf-8")
    return sha256_bytes(payload), len(rows)
