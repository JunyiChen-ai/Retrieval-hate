#!/usr/bin/env python
"""Frozen numerical primitives for LB-SCGP G0.

This module is deliberately dataset-agnostic.  The only supervised quantity
accepted by the G0 producer is a parent-video binary-label vector.  There is no
segment, timestamp, span, localization, stance, target, mechanism or rationale
gold interface.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import platform
import resource
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from scipy.optimize import linprog, minimize, nnls


ROOT = Path("/data/jehc223/RGCL")
SCHEMA_VERSION = 1
PROTECTED_PATH_PREFIXES = (
    "configs/lb_scgp/lb_scgp_sanitizer_sources.json",
    "artifacts/lb_scgp/quarantine/",
    "artifacts/lb_scgp/inputs/MHC_zh/fold4/outer_train_subclips.pt",
    "artifacts/ssr/v1/folds/",
    "data/CLIP_Embedding/",
    "data/video/",
    "data/gt/",
    "data/ASR/",
    "data/Archive/",
    "data/MLLM_scores/",
    "data/Summaries/",
    "data/Counterfactual/",
)
FORMAL_FORBIDDEN_KEY_EXACT = {
    "source_lineage", "source_path", "source_sha256", "source_config",
    "whole_video_cache", "subclip_cache", "mixed_whole_video_cache",
    "mixed_subclip_cache", "quarantine_sanitizer_manifest",
}
FORMAL_FORBIDDEN_KEY_FRAGMENTS = ("legacy_", "mixed", "quarantine")
FORMAL_FORBIDDEN_HASH_KEY_FRAGMENTS = (
    "source", "legacy", "mixed", "quarantine", "manifest", "subclip"
)
ZERO_COUNTER_KEYS = (
    "mllm_call_count", "ocr_call_count", "teacher_cache_read_count",
    "teacher_cache_write_count", "outer_held_label_read_count",
    "outer_held_content_read_count", "val_content_read_count",
    "test_content_read_count", "val_test_teacher_artifact_count",
)
FORMAL_ARTIFACT_EXCLUDE_PREFIXES = (
    "artifacts/lb_scgp/v1/",
    "artifacts/lb_scgp/v2/",
)
MUTABLE_AUDIT_TRAIL_PATHS = (
    "refine-logs/lb_scgp/EXPERIMENT_TRACKER.md",
    "TARGET_LOOP.md",
    "TARGET_STATE.json",
    "TARGET_FINDINGS.md",
    "TARGET_REVIEW_RAW.md",
    "refine-logs/lb_scgp/G0_V2_REPAIR_HANDOFF.md",
    "refine-logs/lb_scgp/G0_FREEZE_EXECUTION_V2.md",
)
MUTABLE_AUDIT_TRAIL_PREFIXES = (
    "refine-logs/lb_scgp/runtime/",
)


def _checked_rel_path(value, field):
    text = str(value).replace("\\", "/")
    if not text or text.startswith("/") or text == ".." or \
            text.startswith("../") or "/../" in text:
        raise RuntimeError("invalid {} entry: {}".format(field, value))
    return text


def _checked_rel_prefix(value, field):
    text = _checked_rel_path(value, field)
    return text if text.endswith("/") else text + "/"


def dirty_state_policy(cfg=None):
    lineage = (cfg or {}).get("lineage", {})
    version = lineage.get("version", "v1")
    if version not in {"v1", "v2"}:
        required = (
            "formal_artifact_exclude_prefixes",
            "dirty_state_excluded_paths",
            "dirty_state_excluded_prefixes",
        )
        missing = [key for key in required if key not in lineage]
        if missing:
            raise RuntimeError("lineage missing explicit dirty-state policy {}".format(missing))
    artifact_prefixes = tuple(_checked_rel_prefix(prefix, "formal_artifact_exclude_prefixes")
                              for prefix in lineage.get(
                                  "formal_artifact_exclude_prefixes",
                                  FORMAL_ARTIFACT_EXCLUDE_PREFIXES))
    artifact_namespace = lineage.get("artifact_namespace")
    if artifact_namespace:
        namespace_prefix = _checked_rel_prefix(
            artifact_namespace, "artifact_namespace")
        if namespace_prefix not in artifact_prefixes:
            raise RuntimeError(
                "artifact_namespace absent from formal artifact exclusion: {}".format(
                    namespace_prefix))
    dirty_paths = tuple(_checked_rel_path(path, "dirty_state_excluded_paths")
                        for path in lineage.get(
                            "dirty_state_excluded_paths",
                            lineage.get("mutable_records_excluded_from_freeze_inputs",
                                        MUTABLE_AUDIT_TRAIL_PATHS)))
    dirty_prefixes = tuple(_checked_rel_prefix(prefix, "dirty_state_excluded_prefixes")
                           for prefix in lineage.get(
                               "dirty_state_excluded_prefixes",
                               MUTABLE_AUDIT_TRAIL_PREFIXES))
    return artifact_prefixes, dirty_paths, dirty_prefixes


class AccessLedger:
    """Fail-closed G0 read/call ledger from which zero counters are derived."""
    def __init__(self):
        self.records = []
        self.counters = {key: 0 for key in ZERO_COUNTER_KEYS}

    def record_file(self, path, purpose, scope, digest=None):
        _, rel = canonical_root_path(path)
        self.records.append({"kind": "file_read", "path": rel.as_posix(),
                             "purpose": str(purpose), "scope": str(scope),
                             "sha256": digest})

    def hash_file(self,path,purpose,scope):
        fs_path, rel = canonical_root_path(path)
        digest=sha256_file(fs_path)
        self.records.append({"kind": "file_read", "path": rel.as_posix(),
                             "purpose": str(purpose), "scope": str(scope),
                             "sha256": digest})
        return digest

    def read_json(self,path,purpose,scope):
        fs_path, _ = canonical_root_path(path)
        digest=self.hash_file(path,purpose,scope)
        with open(fs_path,encoding="utf-8") as handle: obj=json.load(handle)
        return obj,digest

    def read_jsonl(self,path,purpose,scope):
        fs_path, _ = canonical_root_path(path)
        digest=self.hash_file(path,purpose,scope)
        with open(fs_path,encoding="utf-8") as handle:
            rows=[json.loads(line) for line in handle if line.strip()]
        return rows,digest

    def record_bank_member(self, path, member, scope):
        if scope in {"outer_held_label", "outer_held_content", "val_content",
                     "test_content", "teacher_cache", "forbidden_combined_cache"}:
            if scope == "forbidden_combined_cache":
                self.counters["outer_held_label_read_count"] += 1
                self.counters["outer_held_content_read_count"] += 1
            else:
                key = {"outer_held_label":"outer_held_label_read_count",
                       "outer_held_content":"outer_held_content_read_count",
                       "val_content":"val_content_read_count",
                       "test_content":"test_content_read_count",
                       "teacher_cache":"teacher_cache_read_count"}[scope]
                self.counters[key] += 1
            raise RuntimeError("forbidden G0 read {}:{}".format(path, member))
        _, rel = canonical_root_path(path)
        self.records.append({"kind":"bank_member_read","path":rel.as_posix(),
                             "member":str(member),"scope":str(scope)})

    def record_call(self, kind):
        key={"mllm":"mllm_call_count","ocr":"ocr_call_count"}.get(kind)
        if key is None: raise RuntimeError("unknown external call kind")
        self.counters[key]+=1
        raise RuntimeError("external {} call forbidden in G0".format(kind))

    def fields(self):
        return {"access_ledger": list(self.records),
                "access_ledger_sha256": sha256_obj(self.records),
                **dict(self.counters)}


def canonical_json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_obj(obj) -> str:
    return sha256_bytes(canonical_json(obj).encode("utf-8"))


def sha256_file(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_root_path(path, root=ROOT, must_be_under_root=True):
    """Return an absolute filesystem path plus stable ROOT-relative path.

    Relative inputs are resolved under ROOT, not under the process cwd.  Callers
    that read, hash, publish, or ledger project files keep must_be_under_root at
    the default so symlink/absolute escapes fail closed.
    """
    root = Path(root).resolve()
    raw = Path(path)
    candidate = raw if raw.is_absolute() else root / raw
    resolved = candidate.resolve()
    try:
        rel = resolved.relative_to(root)
    except ValueError as exc:
        if must_be_under_root:
            raise RuntimeError(
                "path escapes LB-SCGP ROOT: {} -> {}".format(path, resolved)
            ) from exc
        return resolved, Path(str(resolved))
    return resolved, rel


def root_relative_path(path, root=ROOT, must_be_under_root=True):
    _, rel = canonical_root_path(path, root, must_be_under_root)
    return rel.as_posix()


def relpath(path, root=ROOT):
    return root_relative_path(path, root)


def is_protected_path(path):
    rel = relpath(path)
    return any(rel == prefix.rstrip("/") or rel.startswith(prefix)
               for prefix in PROTECTED_PATH_PREFIXES)


def _looks_like_protected_locator(value):
    if not isinstance(value, str):
        return False
    normalized = value.replace("\\", "/")
    return any(normalized == prefix.rstrip("/") or prefix in normalized or
               normalized.startswith(prefix)
               for prefix in PROTECTED_PATH_PREFIXES)


def assert_no_formal_forbidden_surface(obj, context="formal_input", path="$"):
    """Reject formal records that carry quarantine/mixed/protected lineage."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_s = str(key)
            key_l = key_s.lower()
            child = "{}.{}".format(path, key_s)
            if key_l in FORMAL_FORBIDDEN_KEY_EXACT or \
                    any(fragment in key_l for fragment in FORMAL_FORBIDDEN_KEY_FRAGMENTS):
                raise RuntimeError("{} carries forbidden formal key {}".format(context, child))
            if key_l.endswith("sha256") and \
                    any(fragment in key_l for fragment in FORMAL_FORBIDDEN_HASH_KEY_FRAGMENTS):
                raise RuntimeError("{} carries prohibited formal hash key {}".format(context, child))
            if isinstance(value, str) and _looks_like_protected_locator(value):
                raise RuntimeError("{} carries protected locator at {}".format(context, child))
            assert_no_formal_forbidden_surface(value, context, child)
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            assert_no_formal_forbidden_surface(value, context, "{}[{}]".format(path, index))
    elif isinstance(obj, str) and _looks_like_protected_locator(obj):
        raise RuntimeError("{} carries protected locator at {}".format(context, path))


def load_config(path, access_ledger):
    if access_ledger is None: raise RuntimeError("config read requires AccessLedger")
    cfg,_ = access_ledger.read_json(path,"config_load","frozen_config")
    assert_no_formal_forbidden_surface(cfg, "formal_config")
    if cfg["authorization"] != {
            "authorized_stages": ["G0"],
            "locked_stages": ["G1", "G2", "G3", "G4"],
            "teacher_calls_allowed": False,
            "new_ocr_calls_allowed": False}:
        raise RuntimeError("authorization drift")
    if cfg["supervision"] != {
            "only_gold_supervision": "parent_video_binary_label",
            "segment_gold_exists": False, "segment_gold_used": False}:
        raise RuntimeError("supervision contract drift")
    if any(int(cfg["counters"].get(key, -1)) != 0 for key in ZERO_COUNTER_KEYS):
        raise RuntimeError("nonzero forbidden-access/call counter in config")
    dirty_state_policy(cfg)
    cfg["config_canonical_sha256"] = sha256_obj(cfg)
    return cfg


def resolve(cfg, key):
    cfg_root, _ = canonical_root_path(cfg["paths"]["root"])
    if cfg_root != ROOT.resolve():
        raise RuntimeError("LB-SCGP root drift: {}".format(cfg["paths"]["root"]))
    return canonical_root_path(cfg["paths"][key])[0]


def require_slurm(expected_gpu=None):
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("LB-SCGP computation must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if expected_gpu is True and not visible:
        raise RuntimeError("GPU task has no CUDA_VISIBLE_DEVICES")


def _fsync_dir(path):
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def exclusive_publish(path, payload: bytes):
    """O_EXCL reservation plus fsync and atomic publish; never clobber."""
    path = canonical_root_path(path)[0]
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".publish.lock")
    lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    tmp = None
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii"))
        os.fsync(lock_fd)
        os.close(lock_fd)
        lock_fd = -1
        if path.exists():
            raise FileExistsError("refusing to overwrite {}".format(path))
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(tmp, path)
        os.unlink(tmp)
        tmp = None
        _fsync_dir(path.parent)
    finally:
        if lock_fd >= 0:
            os.close(lock_fd)
        if tmp and os.path.exists(tmp):
            os.unlink(tmp)
        # The O_EXCL sentinel is intentionally persistent.  A completed or
        # interrupted formal path can never be silently reused.


def publish_json(path, obj):
    exclusive_publish(path, (canonical_json(obj) + "\n").encode("utf-8"))


def publish_jsonl(path, rows):
    payload = "".join(canonical_json(row) + "\n" for row in rows)
    exclusive_publish(path, payload.encode("utf-8"))


def payload_hash(obj, field="payload_sha256"):
    copy = dict(obj)
    copy.pop(field, None)
    return sha256_obj(copy)


def git_state(root=ROOT, cfg=None):
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    artifact_prefixes, dirty_paths, dirty_prefixes = dirty_state_policy(cfg)
    mutable_excludes = [
        ":(exclude){}".format(path) for path in dirty_paths
    ] + [":(exclude){}**".format(prefix) for prefix in dirty_prefixes]
    artifact_excludes = [
        ":(exclude){}**".format(prefix) for prefix in artifact_prefixes
    ]
    diff = subprocess.check_output(
        ["git", "diff", "--binary", "--", ".",
         ":(exclude)slurm/logs/**", *artifact_excludes, *mutable_excludes,
         ":(exclude)configs/lb_scgp/lb_scgp_sanitizer_sources.json",
         ":(exclude)artifacts/lb_scgp/quarantine/**",
         ":(exclude)artifacts/ssr/v1/folds/**",
         ":(exclude)data/CLIP_Embedding/**", ":(exclude)data/video/**",
         ":(exclude)data/gt/**", ":(exclude)data/ASR/**",
         ":(exclude)data/Archive/**", ":(exclude)data/MLLM_scores/**",
         ":(exclude)data/Summaries/**", ":(exclude)data/Counterfactual/**"], cwd=root)
    staged = subprocess.check_output(
        ["git", "diff", "--cached", "--binary", "--", ".",
         ":(exclude)slurm/logs/**", *artifact_excludes, *mutable_excludes,
         ":(exclude)configs/lb_scgp/lb_scgp_sanitizer_sources.json",
         ":(exclude)artifacts/lb_scgp/quarantine/**",
         ":(exclude)artifacts/ssr/v1/folds/**",
         ":(exclude)data/CLIP_Embedding/**", ":(exclude)data/video/**",
         ":(exclude)data/gt/**", ":(exclude)data/ASR/**",
         ":(exclude)data/Archive/**", ":(exclude)data/MLLM_scores/**",
         ":(exclude)data/Summaries/**", ":(exclude)data/Counterfactual/**"], cwd=root)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"],
        cwd=root, text=True).splitlines()
    h = hashlib.sha256(diff + b"\0STAGED\0" + staged)
    for rel in sorted(untracked):
        if rel.startswith(artifact_prefixes) or rel.startswith("slurm/logs/"):
            continue
        if rel in dirty_paths or rel.startswith(dirty_prefixes):
            continue
        if any(rel == prefix.rstrip("/") or rel.startswith(prefix)
               for prefix in PROTECTED_PATH_PREFIXES):
            continue
        path = root / rel
        h.update(rel.encode("utf-8") + b"\0")
        if path.is_file():
            h.update(bytes.fromhex(sha256_file(path)))
    return head, h.hexdigest()


def implementation_hash(cfg):
    rows = []
    for rel in cfg["implementation_files"]:
        path, stable_rel = canonical_root_path(rel)
        rows.append({"path": stable_rel.as_posix(), "sha256": sha256_file(path)})
    return sha256_obj(rows), rows


def runtime_metadata():
    import scipy
    import torch
    gpu_name = None
    gpu_uuid = None
    gpu_total_memory_gib = None
    cuda_version = torch.version.cuda
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_uuid = getattr(props, "uuid", None)
        gpu_total_memory_gib = float(props.total_memory) / (1024.0 ** 3)
    return {
        "python_version": platform.python_version(),
        "numpy_version": np.__version__, "scipy_version": scipy.__version__,
        "torch_version": torch.__version__, "cuda_version": cuda_version,
        "cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "slurm_job_gpus": os.environ.get("SLURM_JOB_GPUS"),
        "slurm_gpus": os.environ.get("SLURM_GPUS"),
        "slurm_gpus_on_node": os.environ.get("SLURM_GPUS_ON_NODE"),
        "slurm_step_gpus": os.environ.get("SLURM_STEP_GPUS"),
        "gpu_name": gpu_name,
        "gpu_uuid": gpu_uuid,
        "gpu_total_memory_gib": gpu_total_memory_gib,
    }


def provenance_base(cfg, run_id, stage, verifier_hash=None, access_ledger=None):
    head, dirty = git_state(cfg=cfg)
    impl_hash, impl_files = implementation_hash(cfg)
    out = {
        "schema_version": SCHEMA_VERSION, "run_id": run_id,
        "stage": stage, "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "git_head": head, "dirty_diff_sha256": dirty,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "config_canonical_sha256": cfg["config_canonical_sha256"],
        "implementation_sha256": impl_hash,
        "implementation_files": impl_files,
        "independent_verifier_sha256": verifier_hash,
        "only_gold_supervision": "parent_video_binary_label",
        "segment_gold_exists": False, "segment_gold_used": False,
    }
    out.update(runtime_metadata())
    if access_ledger is None:
        raise RuntimeError("formal provenance requires an AccessLedger")
    out.update(access_ledger.fields())
    return out


def projector_preimage_ball(y, x0, operator, radius, root_rtol=1e-12):
    """Project onto {x: ||L(x-x0)||_2 <= radius} in Euclidean space."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    op = np.asarray(operator, dtype=np.float64)
    q = op @ (y - x0)
    qnorm = float(np.linalg.norm(q))
    if qnorm <= radius:
        return y.copy(), {"mu": 0.0, "root_residual": 0.0,
                          "input_operator_norm": qnorm}
    gram = op @ op.T
    if radius == 0.0:
        p = y - op.T @ (np.linalg.pinv(gram, rcond=1e-14) @ q)
        return p, {"mu": None,
                   "root_residual": float(np.linalg.norm(op @ (p - x0))),
                   "input_operator_norm": qnorm}
    eigval, eigvec = np.linalg.eigh(0.5 * (gram + gram.T))
    qhat = eigvec.T @ q

    def eval_mu(mu):
        denom = 1.0 + mu * eigval
        u = eigvec @ (qhat / denom)
        p = y - mu * (op.T @ u)
        return p, float(np.linalg.norm(op @ (p - x0)))

    lo, hi = 0.0, 1.0
    _, value = eval_mu(hi)
    while value > radius and hi < 1e18:
        hi *= 2.0
        _, value = eval_mu(hi)
    if value > radius:
        raise RuntimeError("failed to bracket preimage-ball root")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        _, value = eval_mu(mid)
        if value > radius:
            lo = mid
        else:
            hi = mid
        if hi - lo <= root_rtol * max(1.0, hi):
            break
    mu = 0.5 * (lo + hi)
    p, value = eval_mu(mu)
    return p, {"mu": float(mu), "root_residual": abs(value - radius),
               "input_operator_norm": qnorm}


def row_operator(n, row):
    cols = [j for j in range(n) if j != row]
    op = np.zeros((n - 1, n * n), dtype=np.float64)
    op[np.arange(n - 1), row * n + np.asarray(cols)] = 1.0
    return op


def class_mean_operator(n, rows):
    rows = np.asarray(rows, dtype=np.int64)
    op = np.zeros((n, n * n), dtype=np.float64)
    for col in range(n):
        op[col, rows * n + col] = 1.0 / len(rows)
    return op


def project_psd(y):
    sym = 0.5 * (np.asarray(y, dtype=np.float64) + np.asarray(y).T)
    eigval, eigvec = np.linalg.eigh(sym)
    clipped = np.maximum(eigval, 0.0)
    p = (eigvec * clipped) @ eigvec.T
    return 0.5 * (p + p.T), eigval


def project_halfspace(y, normal, rhs):
    y = np.asarray(y, dtype=np.float64)
    normal = np.asarray(normal, dtype=np.float64)
    value = float(normal @ y)
    if value >= rhs:
        return y.copy(), 0.0
    tau = (rhs - value) / float(normal @ normal)
    return y + tau * normal, float(tau)


def capped_simplex_nonnegative(v, cap):
    v = np.asarray(v, dtype=np.float64)
    positive = np.maximum(v, 0.0)
    if float(positive.sum()) <= cap:
        return positive
    lo, hi = float(np.min(v) - cap), float(np.max(v))
    for _ in range(120):
        mid = 0.5 * (lo + hi)
        total = float(np.maximum(v - mid, 0.0).sum())
        if total > cap:
            lo = mid
        else:
            hi = mid
    return np.maximum(v - 0.5 * (lo + hi), 0.0)


def dense_reference_ball(y, x0, operator, radius):
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    x0 = np.asarray(x0, dtype=np.float64).reshape(-1)
    op = np.asarray(operator, dtype=np.float64)
    if radius == 0.0:
        gram = op @ op.T
        return y - op.T @ (np.linalg.pinv(gram, rcond=1e-14) @
                           (op @ (y - x0)))
    result = minimize(
        lambda x: 0.5 * float(np.dot(x - y, x - y)), y.copy(),
        jac=lambda x: x - y,
        constraints={"type": "ineq",
                     "fun": lambda x: radius ** 2 -
                     float(np.dot(op @ (x - x0), op @ (x - x0))),
                     "jac": lambda x: -2.0 * op.T @ (op @ (x - x0))},
        method="SLSQP", options={"ftol": 1e-13, "maxiter": 4000})
    if not result.success:
        raise RuntimeError("dense reference failed: {}".format(result.message))
    return np.asarray(result.x, dtype=np.float64)


def _tolerance_aware_order(values, ids, tolerance):
    """Descending, anchor-grouped tolerance order with canonical-ID ties.

    The best remaining similarity anchors a tie group containing every value
    within ``tolerance`` of that anchor.  This avoids non-transitive pairwise
    tolerance comparisons while making the boundary semantics explicit.
    """
    remaining = sorted(range(len(values)), key=lambda k: -float(values[k]))
    ordered = []
    while remaining:
        anchor = float(values[remaining[0]])
        group = [k for k in remaining
                 if anchor - float(values[k]) <= tolerance]
        group.sort(key=lambda k: str(ids[k]))
        ordered.extend(group)
        selected = set(group)
        remaining = [k for k in remaining if k not in selected]
    return ordered


def stable_rankings(gram, ids, topk=20, tolerance=1e-7):
    gram = np.asarray(gram, dtype=np.float64)
    ids = [str(x) for x in ids]
    rankings = []
    for i in range(len(ids)):
        candidates = [j for j in range(len(ids)) if j != i]
        local = _tolerance_aware_order(
            [float(gram[i, j]) for j in candidates],
            [ids[j] for j in candidates], tolerance)
        rankings.append([candidates[k] for k in local[:topk]])
    return rankings


def exact_vote_ledger(gram, labels, ids, topk=20):
    labels = np.asarray(labels, dtype=np.int64)
    ranks = stable_rankings(gram, ids, topk=topk)
    rows = []
    for i, neigh in enumerate(ranks):
        vote = 0.0
        ordered = []
        for rank, j in enumerate(neigh, 1):
            weight = topk + 1 - rank
            similarity = float(gram[i, j])
            vote += weight * similarity * (2 * int(labels[j]) - 1)
            ordered.append({"rank": rank, "id": str(ids[j]),
                            "row": int(j), "label": int(labels[j]),
                            "cosine": similarity, "weight": weight})
        rows.append({"query_id": str(ids[i]), "query_row": i,
                     "query_label": int(labels[i]), "neighbors": ordered,
                     "weighted_signed_vote": float(vote),
                     "prediction": int(vote >= 0.0)})
    return rows


def true_class_margins(gram, labels, ids, topk=20):
    ledger = exact_vote_ledger(gram, labels, ids, topk=topk)
    return np.asarray([
        (2 * row["query_label"] - 1) * row["weighted_signed_vote"] / 210.0
        for row in ledger], dtype=np.float64), ledger


def rank_cell_pairs(gram, ids, topk=20):
    """Complete 19 internal and twentieth-vs-all-outsider inequalities."""
    gram = np.asarray(gram, dtype=np.float64)
    ids = [str(x) for x in ids]
    pairs = []
    rankings = stable_rankings(gram, ids, topk=len(ids) - 1)
    for i, full in enumerate(rankings):
        top = full[:topk]
        for r in range(topk - 1):
            pairs.append((i, top[r], top[r + 1], "internal"))
        boundary = top[topk - 1]
        for outsider in full[topk:]:
            pairs.append((i, boundary, outsider, "boundary"))
    return pairs


def canonical_rank_rhs(id_a, id_b, tolerance=1e-7):
    """Minimum sim(a)-sim(b) that preserves a-before-b total order."""
    if str(id_a) < str(id_b):
        return -float(tolerance)
    return float(np.nextafter(float(tolerance), math.inf))


def rank_cell_violation(gram, pairs, ids=None, tolerance=1e-7):
    if not pairs:
        return 0.0
    return max(0.0, max((canonical_rank_rhs(ids[a], ids[b], tolerance)
                         if ids is not None else 0.0) -
                        float(gram[i, a] - gram[i, b])
                        for i, a, b, _ in pairs))


def boundary_tie_count(gram, ids, topk=20, tolerance=1e-7):
    full = stable_rankings(gram, ids, topk=len(ids) - 1)
    ties = []
    normals = []
    for i, ranking in enumerate(full):
        boundary = ranking[topk - 1]
        value = float(gram[i, boundary])
        for j in ranking[topk:]:
            if abs(value - float(gram[i, j])) <= tolerance:
                descriptor = (str(ids[i]), str(ids[boundary]), str(ids[j]))
                if descriptor not in ties:
                    ties.append(descriptor)
                    normal = np.zeros(gram.size, dtype=np.float64)
                    normal[i * len(ids) + boundary] = 1.0
                    normal[i * len(ids) + j] = -1.0
                    normals.append(normal)
    independent = int(np.linalg.matrix_rank(np.stack(normals), tol=1e-12)) if normals else 0
    return independent, sorted(ties)


def boundary_orientation_system(gram, ids, topk=20, tolerance=1e-7,
                                compatible_limit=34):
    """Orientation arrangement in the symmetric Gram tangent space."""
    full = stable_rankings(gram, ids, topk=len(ids)-1, tolerance=tolerance)
    pairs=[(i,j) for i in range(len(ids)) for j in range(i+1,len(ids))]
    pair_index={pair:k for k,pair in enumerate(pairs)}
    descriptors=[]; normals=[]
    for i,ranking in enumerate(full):
        boundary=ranking[topk-1]; value=float(gram[i,boundary])
        for outsider in ranking[topk:]:
            if abs(value-float(gram[i,outsider]))<=tolerance:
                desc=(str(ids[i]),str(ids[boundary]),str(ids[outsider]))
                if desc in descriptors: continue
                normal=np.zeros(len(pairs),dtype=np.float64)
                for j,sign in ((boundary,1.0),(outsider,-1.0)):
                    if i==j: continue
                    normal[pair_index[tuple(sorted((i,j)))]]+=sign
                descriptors.append(desc); normals.append(normal)
    if not normals:
        return {"rank":0,"descriptors":[],"basis_indices":[],
                "dependency_coefficients":[],"compatible_assignments":[[]],
                "compatible_overflow":False}
    matrix=np.stack(normals); basis=[]; current=np.zeros((0,matrix.shape[1]))
    for index,row in enumerate(matrix):
        candidate=np.vstack([current,row])
        if np.linalg.matrix_rank(candidate,tol=1e-12)>len(basis):
            basis.append(index); current=candidate
    coeff=(np.linalg.lstsq(matrix[basis].T,matrix.T,rcond=1e-12)[0].T)
    if float(np.max(np.linalg.norm(coeff@matrix[basis]-matrix,axis=1)))>1e-9:
        raise RuntimeError("orientation dependency reconstruction failed")
    assignments=[]; overflow=False; m=len(normals)
    def feasible(signs):
        if not signs: return True
        a=-np.asarray(signs,dtype=np.float64)[:,None]*matrix[:len(signs)]
        result=linprog(np.zeros(matrix.shape[1]),A_ub=a,b_ub=-np.ones(len(signs)),
                       bounds=[(-10.0,10.0)]*matrix.shape[1],method="highs")
        return bool(result.success)
    def dfs(signs):
        nonlocal overflow
        if overflow: return
        if len(signs)==m:
            assignments.append(list(signs))
            if len(assignments)>=compatible_limit: overflow=True
            return
        for sign in (-1,1):
            candidate=signs+[sign]
            if feasible(candidate): dfs(candidate)
    dfs([])
    return {"rank":len(basis),"descriptors":descriptors,"basis_indices":basis,
            "dependency_coefficients":coeff.tolist(),
            "compatible_assignments":assignments,"compatible_overflow":overflow}


def orientation_cell_from_assignment(base_rankings, descriptors, assignment, ids):
    """Build a total rank cell by DAG + canonical topological ordering."""
    id_to_row={str(vid):i for i,vid in enumerate(ids)}
    by_query={}
    for sign,(query,a_id,b_id) in zip(assignment,descriptors):
        by_query.setdefault(query,[]).append((int(sign),id_to_row[a_id],id_to_row[b_id]))
    cells=[list(row) for row in base_rankings]
    for query,constraints in by_query.items():
        qi=id_to_row[query]; baseline=list(base_rankings[qi])
        parent={node:node for node in baseline}
        def find(node):
            while parent[node]!=node:
                parent[node]=parent[parent[node]]; node=parent[node]
            return node
        def union(a,b):
            ra,rb=find(a),find(b)
            if ra!=rb: parent[rb]=ra
        for _,a,b in constraints: union(a,b)
        edges={node:set() for node in baseline}; indegree={node:0 for node in baseline}
        for a,b in zip(baseline[:-1],baseline[1:]):
            if find(a)!=find(b): edges[a].add(b)
        for sign,a,b in constraints:
            u,v=(a,b) if sign>0 else (b,a); edges[u].add(v)
        for u in edges:
            for v in edges[u]: indegree[v]+=1
        order=[]; baseline_pos={node:i for i,node in enumerate(baseline)}
        available=[node for node in baseline if indegree[node]==0]
        while available:
            available.sort(key=lambda node:(baseline_pos[node],str(ids[node])))
            u=available.pop(0); order.append(u)
            for v in sorted(edges[u],key=lambda node:(baseline_pos[node],str(ids[node]))):
                indegree[v]-=1
                if indegree[v]==0: available.append(v)
        if len(order)!=len(baseline):
            raise RuntimeError("compatible orientation produced cyclic rank DAG")
        for sign,a,b in constraints:
            if (order.index(a)<order.index(b)) != (sign>0):
                raise RuntimeError("orientation assignment not realized by rank DAG")
        cells[qi]=order
    return cells


def cone_audit(columns, target):
    columns = np.asarray(columns, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    if columns.ndim != 2 or columns.shape[0] != target.size:
        raise ValueError("cone shape mismatch")
    coeff, _ = nnls(columns, target, maxiter=max(1000, 20 * columns.shape[1]))
    projection = columns @ coeff
    residual = target - projection
    target_norm = max(float(np.linalg.norm(target)), 1e-15)
    separation = float(np.linalg.norm(residual) / target_norm)
    if np.linalg.norm(residual) == 0.0:
        witness = np.zeros_like(residual)
    else:
        witness = residual / np.linalg.norm(residual)
    cone_inner = columns.T @ witness
    primal = float(np.linalg.norm(residual))
    witness_value = float(witness @ target)
    duality_gap = abs(primal - witness_value)
    return {
        "coefficients": coeff, "projection": projection,
        "residual": residual, "witness": witness,
        "relative_separation": separation,
        "max_cone_witness_inner": float(np.max(cone_inner)) if cone_inner.size else 0.0,
        "min_cone_witness_inner": float(np.min(cone_inner)) if cone_inner.size else 0.0,
        "duality_gap": float(duality_gap),
    }


def _deterministic_subspace_basis(projector, dimension, tolerance=1e-10):
    basis = []
    for axis in range(dimension):
        v = projector[:, axis].copy()
        for q in basis:
            v -= q * float(q @ v)
        norm = float(np.linalg.norm(v))
        if norm > tolerance:
            v /= norm
            pivot = int(np.argmax(np.abs(v)))
            if v[pivot] < 0:
                v = -v
            basis.append(v)
        if len(basis) == int(round(np.trace(projector))):
            break
    return np.column_stack(basis) if basis else np.zeros((dimension, 0))


def deterministic_psd_factor(gram, negative_reject=-1e-7,
                             repeat_tolerance=1e-10):
    gram = 0.5 * (np.asarray(gram, dtype=np.float64) +
                  np.asarray(gram, dtype=np.float64).T)
    eigval, eigvec = np.linalg.eigh(gram)
    if float(eigval.min()) < negative_reject:
        raise ValueError("eigenvalue below rejection threshold")
    order = np.argsort(-eigval, kind="mergesort")
    eigval, eigvec = eigval[order], eigvec[:, order]
    deterministic = np.zeros_like(eigvec)
    start = 0
    while start < len(eigval):
        end = start + 1
        scale = max(1.0, abs(float(eigval[start])))
        while end < len(eigval) and abs(float(eigval[end] - eigval[start])) <= (
                repeat_tolerance * scale):
            end += 1
        projector = eigvec[:, start:end] @ eigvec[:, start:end].T
        deterministic[:, start:end] = _deterministic_subspace_basis(
            projector, gram.shape[0])[:, :end - start]
        start = end
    clipped = np.maximum(eigval, 0.0)
    factor = deterministic * np.sqrt(clipped)[None, :]
    return factor, eigval


def procrustes_align_factor(factor, reference):
    factor = np.asarray(factor, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    u, _, vt = np.linalg.svd(factor.T @ reference, full_matrices=False)
    rotation = u @ vt
    aligned = factor @ rotation
    return aligned, rotation


def load_npz_members_only(path, allowed_names, forbidden_names, access_ledger):
    """Read only explicit NPY members; forbidden held-out arrays remain unopened."""
    path = canonical_root_path(path)[0]
    allowed = set(allowed_names)
    forbidden = set(forbidden_names)
    out, opened = {}, []
    with zipfile.ZipFile(path, "r") as archive:
        by_stem = {Path(name).stem: name for name in archive.namelist()
                   if name.endswith(".npy")}
        missing = allowed - set(by_stem)
        if missing:
            raise RuntimeError("missing bank members {}".format(sorted(missing)))
        for stem in sorted(allowed):
            if stem in forbidden:
                raise RuntimeError("attempted forbidden bank member")
            scope = "outer_held_ids_sentinel" if stem == "query_ids" else "outer_train_bank"
            access_ledger.record_bank_member(path, stem, scope)
            member = by_stem[stem]
            with archive.open(member, "r") as handle:
                payload = handle.read()
            access_ledger.records[-1]["sha256"] = sha256_bytes(payload)
            out[stem] = np.load(io.BytesIO(payload), allow_pickle=False)
            opened.append(stem)
    if set(opened) & forbidden:
        raise RuntimeError("forbidden held-out bank member opened")
    return out, opened


def hash_npz_members_only(path, allowed_names, forbidden_names, access_ledger,
                          purpose, scope):
    """Hash only explicit NPY members without opening forbidden members."""
    path = canonical_root_path(path)[0]
    allowed = set(allowed_names)
    forbidden = set(forbidden_names)
    out = {}
    with zipfile.ZipFile(path, "r") as archive:
        by_stem = {Path(name).stem: name for name in archive.namelist()
                   if name.endswith(".npy")}
        missing = allowed - set(by_stem)
        if missing:
            raise RuntimeError("missing bank members {}".format(sorted(missing)))
        for stem in sorted(allowed):
            if stem in forbidden:
                raise RuntimeError("attempted forbidden bank member hash")
            member = by_stem[stem]
            with archive.open(member, "r") as handle:
                digest = sha256_bytes(handle.read())
            out[stem] = digest
            _, rel = canonical_root_path(path)
            access_ledger.records.append({
                "kind": "npz_member_hash", "path": rel.as_posix(),
                "member": stem, "purpose": purpose, "scope": scope,
                "sha256": digest})
    return out


def peak_rss_gib():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return float(value) / (1024.0 * 1024.0)


class Stopwatch:
    def __init__(self):
        self.start = time.perf_counter()

    def seconds(self):
        return float(time.perf_counter() - self.start)
