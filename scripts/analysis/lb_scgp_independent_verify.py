#!/usr/bin/env python
"""Independent LB-SCGP G0 verifier.

This file intentionally imports none of lb_scgp_common, lb_scgp_g0, SSR
solver, projector, ranking, evaluator, factor or rollback implementations.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import platform
import random
import shutil
import subprocess
import tempfile
import zipfile
from pathlib import Path

import numpy as np
import scipy
from scipy.optimize import linprog, minimize, nnls


ROOT = Path("/data/jehc223/RGCL")
VERIFIER_ACCESS_LEDGER=[]
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


def cjson(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def hobj(obj):
    return hashlib.sha256(cjson(obj).encode("utf-8")).hexdigest()


def hbytes(payload):
    return hashlib.sha256(payload).hexdigest()


def payload_digest(obj):
    copy_obj = dict(obj)
    copy_obj.pop("payload_sha256", None)
    return hobj(copy_obj)


def canonical_root_path(path, must_be_under_root=True):
    root = ROOT.resolve()
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


def _relpath(path):
    _, rel = canonical_root_path(path)
    return rel.as_posix()


def _resolve_cfg_path(cfg, key):
    return canonical_root_path(cfg["paths"][key])[0]


def hfile(path):
    path, rel_path = canonical_root_path(path)
    rel = rel_path.as_posix()
    if rel.startswith(("configs/","artifacts/lb_scgp/v1/",
                       "artifacts/lb_scgp/v2/","artifacts/lb_scgp/v3/",
                       "artifacts/lb_scgp/v4/","artifacts/lb_scgp/v5/",
                       "artifacts/ssr/v1/",
                       "data/CLIP_Embedding/","refine-logs/lb_scgp/","TARGET_")):
        VERIFIER_ACCESS_LEDGER.append({"path":rel,"purpose":"independent_hash_read",
                                       "scope":"protected_input_or_artifact"})
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _forbidden_path_string(value):
    if not isinstance(value, str):
        return False
    normalized = value.replace("\\", "/")
    candidates = {normalized}
    try:
        candidates.add(_relpath(value).replace("\\", "/"))
    except Exception:
        try:
            path = Path(value)
            if not path.is_absolute():
                path = ROOT / path
            candidates.add(str(path.resolve()).replace("\\", "/"))
        except Exception:
            pass
    for item in candidates:
        if any(item == prefix.rstrip("/") or item.startswith(prefix) or prefix in item
               for prefix in PROTECTED_PATH_PREFIXES):
            return True
    return False


def _forbidden_formal_surfaces(obj, path="$"):
    bad = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            key_s = str(key)
            key_l = key_s.lower()
            child = "{}.{}".format(path, key_s)
            if key_l in FORMAL_FORBIDDEN_KEY_EXACT or \
                    any(fragment in key_l for fragment in FORMAL_FORBIDDEN_KEY_FRAGMENTS):
                bad.append({"path": child, "reason": "forbidden_key"})
            if key_l.endswith("sha256") and \
                    any(fragment in key_l for fragment in FORMAL_FORBIDDEN_HASH_KEY_FRAGMENTS):
                bad.append({"path": child, "reason": "prohibited_hash_key"})
            if isinstance(value, str) and _forbidden_path_string(value):
                bad.append({"path": child, "reason": "forbidden_path_string", "value": value})
            bad.extend(_forbidden_formal_surfaces(value, child))
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            bad.extend(_forbidden_formal_surfaces(value, "{}[{}]".format(path, index)))
    elif isinstance(obj, str) and _forbidden_path_string(obj):
        bad.append({"path": path, "reason": "forbidden_path_string", "value": obj})
    return bad


def assert_no_forbidden_formal_surface(obj, context):
    bad = _forbidden_formal_surfaces(obj)
    if bad:
        raise RuntimeError("{} carries forbidden formal surfaces {}".format(context, bad[:5]))


def read_json(path):
    path, rel_path = canonical_root_path(path)
    rel = rel_path.as_posix()
    VERIFIER_ACCESS_LEDGER.append({"path":rel,"purpose":"independent_json_read",
                                   "scope":"protected_input_or_artifact"})
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path):
    path, rel_path = canonical_root_path(path)
    rel = rel_path.as_posix()
    VERIFIER_ACCESS_LEDGER.append({"path":rel,"purpose":"independent_jsonl_read",
                                   "scope":"protected_input_or_artifact"})
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def recursively_finite(obj):
    if isinstance(obj,dict): return all(recursively_finite(v) for v in obj.values())
    if isinstance(obj,(list,tuple)): return all(recursively_finite(v) for v in obj)
    if isinstance(obj,(float,np.floating)): return math.isfinite(float(obj))
    return True


def verify_payload(obj):
    copy = dict(obj); stored = copy.pop("payload_sha256", None)
    return stored is not None and hobj(copy) == stored


DEFAULT_LINEAGE_RUN_IDS = {
    "freeze": "LBSCGP-G0-FREEZE-v1",
    "code_audit": "LBSCGP-G0-CODE-AUDIT-v1",
    "synthetic": "LBSCGP-G0-SYNTH-v1",
    "realfold": "LBSCGP-G0-REAL-MHC_zh-F4-S0-v1",
    "replay": "LBSCGP-G0-REAL-REPLAY-MHC_zh-F4-S0-v1",
    "decision": "LBSCGP-G0-DECISION-v1",
}
DEFAULT_FREEZE_PROTOCOL_INPUT_KEYS = (
    "experiment_plan", "experiment_tracker", "problem_anchor",
    "final_proposal", "review_summary", "refinement_report",
    "target_loop", "target_state",
)


def _lineage_run_id(cfg, key):
    return cfg.get("lineage", {}).get("run_ids", {}).get(
        key, DEFAULT_LINEAGE_RUN_IDS[key])


def _config_rel(cfg):
    return cfg.get("lineage", {}).get(
        "config_path",
        "configs/lb_scgp/{}.json".format(cfg.get("config_name", "lb_scgp_v1")))


def _freeze_protocol_input_keys(cfg):
    return tuple(cfg.get("lineage", {}).get(
        "freeze_input_path_keys", DEFAULT_FREEZE_PROTOCOL_INPUT_KEYS))


def current_implementation(cfg):
    rows=[]
    for rel in cfg["implementation_files"]:
        path, stable_rel = canonical_root_path(rel)
        if not path.exists(): return None,[]
        rows.append({"path":stable_rel.as_posix(),"sha256":hfile(path)})
    return hobj(rows),rows


def current_dirty_hash(cfg=None):
    artifact_prefixes, dirty_paths, dirty_prefixes = dirty_state_policy(cfg)
    mutable_excludes = [
        ":(exclude){}".format(path) for path in dirty_paths
    ] + [":(exclude){}**".format(prefix) for prefix in dirty_prefixes]
    artifact_excludes = [
        ":(exclude){}**".format(prefix) for prefix in artifact_prefixes
    ]
    excludes=[":(exclude)slurm/logs/**",*artifact_excludes,*mutable_excludes,
              ":(exclude)configs/lb_scgp/lb_scgp_sanitizer_sources.json",
              ":(exclude)artifacts/lb_scgp/quarantine/**",
              ":(exclude)artifacts/ssr/v1/folds/**",
              ":(exclude)data/CLIP_Embedding/**",":(exclude)data/video/**",
              ":(exclude)data/gt/**",":(exclude)data/ASR/**",
              ":(exclude)data/Archive/**",":(exclude)data/MLLM_scores/**",
              ":(exclude)data/Summaries/**",":(exclude)data/Counterfactual/**"]
    diff=subprocess.check_output(["git","diff","--binary","--",".",*excludes],cwd=ROOT)
    staged=subprocess.check_output(["git","diff","--cached","--binary","--",".",*excludes],cwd=ROOT)
    untracked=subprocess.check_output(["git","ls-files","--others","--exclude-standard"],cwd=ROOT,text=True).splitlines()
    h=hashlib.sha256(diff+b"\0STAGED\0"+staged)
    for rel in sorted(untracked):
        if rel.startswith(artifact_prefixes) or rel.startswith("slurm/logs/"): continue
        if rel in dirty_paths or rel.startswith(dirty_prefixes): continue
        if any(rel == prefix.rstrip("/") or rel.startswith(prefix)
               for prefix in PROTECTED_PATH_PREFIXES): continue
        path=ROOT/rel; h.update(rel.encode("utf-8")+b"\0")
        if path.is_file(): h.update(bytes.fromhex(hfile(path)))
    return h.hexdigest()


def publish_exclusive(path, obj):
    path = canonical_root_path(path)[0]; path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".publish.lock")
    lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    tmp = None
    try:
        os.write(lock_fd, str(os.getpid()).encode("ascii")); os.fsync(lock_fd)
        os.close(lock_fd); lock_fd = -1
        if path.exists():
            raise FileExistsError("refusing overwrite {}".format(path))
        fd, tmp = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            handle.write(cjson(obj) + "\n"); handle.flush(); os.fsync(handle.fileno())
        os.link(tmp, path); os.unlink(tmp); tmp = None
        dfd = os.open(str(path.parent), os.O_RDONLY); os.fsync(dfd); os.close(dfd)
    finally:
        if lock_fd >= 0: os.close(lock_fd)
        if tmp and os.path.exists(tmp): os.unlink(tmp)
        # Persistent O_EXCL sentinel: formal paths are never reusable.


CODE_AUDIT_RECORD_TYPE = "LB_SCGP_G0_CODE_AUDIT_INDEPENDENT_REVIEW_RECORD_V4"
CODE_AUDIT_ARTIFACT_TYPE = "LB_SCGP_G0_CODE_AUDIT_PASS_ARTIFACT_V4"
CODE_AUDIT_INDEX_TYPE = "LB_SCGP_G0_CODE_AUDIT_PUBLICATION_INDEX_V4"
CODE_AUDIT_REVIEW_SCOPE = "LB-SCGP G0 v4 formal code audit for LBSCGP-G0-FREEZE-v4"
CODE_AUDIT_REVIEW_PROCESS_IDENTITY = "fresh_independent_gpt_5_5_xhigh"
CODE_AUDIT_DIR = Path("g0/code_audit")
CODE_AUDIT_REVIEW_ARTIFACT = "review.md"
CODE_AUDIT_RECORD_ARTIFACT = "review_record.json"
CODE_AUDIT_JSON_ARTIFACT = "audit.json"
CODE_AUDIT_INDEX_ARTIFACT = "publication_index.json"
CODE_AUDIT_EXACT_FORMAL_PREFIXES_V4 = (
    "artifacts/lb_scgp/v1/",
    "artifacts/lb_scgp/v2/",
    "artifacts/lb_scgp/v3/",
    "artifacts/lb_scgp/v4/",
)
CODE_AUDIT_EXACT_FORMAL_PREFIXES_V5 = (
    "artifacts/lb_scgp/v1/",
    "artifacts/lb_scgp/v2/",
    "artifacts/lb_scgp/v3/",
    "artifacts/lb_scgp/v4/",
    "artifacts/lb_scgp/v5/",
)
CODE_AUDIT_EXPECTED_PRIOR_HASH_PATHS = (
    "artifacts/lb_scgp/v1/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v1/CONFIG_FREEZE.json.publish.lock",
    "artifacts/lb_scgp/v2/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v2/CONFIG_FREEZE.json.publish.lock",
    "artifacts/lb_scgp/v3/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v3/CONFIG_FREEZE.json.publish.lock",
)
CODE_AUDIT_EXPECTED_PRIOR_HASH_PATHS_V5 = (
    "artifacts/lb_scgp/v1/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v1/CONFIG_FREEZE.json.publish.lock",
    "artifacts/lb_scgp/v2/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v2/CONFIG_FREEZE.json.publish.lock",
    "artifacts/lb_scgp/v3/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v3/CONFIG_FREEZE.json.publish.lock",
    "artifacts/lb_scgp/v4/CONFIG_FREEZE.json",
    "artifacts/lb_scgp/v4/CONFIG_FREEZE.json.publish.lock",
    "artifacts/lb_scgp/v4/g0/code_audit/audit.json",
    "artifacts/lb_scgp/v4/g0/code_audit/audit.json.publish.lock",
    "artifacts/lb_scgp/v4/g0/code_audit/publication_index.json",
    "artifacts/lb_scgp/v4/g0/code_audit/publication_index.json.publish.lock",
    "artifacts/lb_scgp/v4/g0/code_audit/review.md",
    "artifacts/lb_scgp/v4/g0/code_audit/review.md.publish.lock",
    "artifacts/lb_scgp/v4/g0/code_audit/review_record.json",
    "artifacts/lb_scgp/v4/g0/code_audit/review_record.json.publish.lock",
)
CODE_AUDIT_REVIEW_RECORD_KEYS = frozenset({
    "schema_version", "record_type", "run_id", "stage", "status",
    "lineage_version", "config_path", "artifact_namespace",
    "freeze_run_id", "freeze_path", "freeze_file_sha256",
    "freeze_payload_sha256", "config_canonical_sha256",
    "implementation_sha256", "independent_verifier_sha256",
    "review_report_path", "review_report_sha256", "reviewer_identity",
    "review_process_identity", "review_scope", "critical", "high",
    "important", "no_segment_gold_pass", "formal_pass_authorized",
    "independent_reviewer", "repair_executor_created",
    "only_gold_supervision", "segment_gold_exists", "segment_gold_used",
    "payload_sha256",
})
CODE_AUDIT_ARTIFACT_KEYS = frozenset({
    "schema_version", "artifact_type", "run_id", "stage", "status",
    "critical", "high", "important", "no_segment_gold_pass",
    "formal_pass_authorized", "slurm_job_id", "config_path",
    "config_file_sha256", "config_canonical_sha256", "artifact_namespace",
    "lineage_version", "freeze_path", "freeze_file_sha256",
    "freeze_lock_path", "freeze_lock_sha256", "freeze_payload_sha256",
    "freeze_run_id", "freeze_stage", "git_head", "dirty_diff_sha256",
    "frozen_dirty_diff_sha256", "implementation_sha256",
    "implementation_files", "independent_verifier_sha256",
    "review_report_path", "review_report_sha256", "review_record_path",
    "review_record_sha256", "review_record_payload_sha256",
    "reviewer_identity", "review_process_identity", "review_scope",
    "only_gold_supervision", "segment_gold_exists", "segment_gold_used",
    "mllm_call_count", "ocr_call_count", "teacher_cache_read_count",
    "teacher_cache_write_count", "outer_held_label_read_count",
    "outer_held_content_read_count", "val_content_read_count",
    "test_content_read_count", "val_test_teacher_artifact_count",
    "formal_model_optimizer_evaluator_outer_held_read_count",
    "access_ledger", "access_ledger_sha256", "dirty_policy",
    "frozen_input_rehashes", "allowed_bank_member_sha256",
    "forbidden_bank_members_not_opened", "prior_lineage_no_clobber_hashes",
    "audit_publish_wrapper", "authorization_gate", "downstream_contract",
    "output_files", "python_version", "numpy_version", "scipy_version",
    "torch_version", "conda_env", "payload_sha256",
})
CODE_AUDIT_INDEX_KEYS = frozenset({
    "schema_version", "artifact_type", "run_id", "stage", "status",
    "output_files", "lock_files", "payload_sha256",
})


def _require_exact_keys(name, obj, expected):
    keys = set(obj)
    if keys != set(expected):
        missing = sorted(set(expected) - keys)
        extra = sorted(keys - set(expected))
        raise RuntimeError("{} schema keys mismatch missing={} extra={}".format(
            name, missing, extra))


def _require_nonempty_string(name, value):
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("{} must be a nonempty string".format(name))


def _lineage_path(cfg, key):
    value = cfg.get("lineage", {}).get(key)
    _require_nonempty_string("lineage.{}".format(key), value)
    return value


def _code_audit_contract(cfg):
    version = cfg.get("lineage", {}).get("version")
    if version not in {"v4", "v5"}:
        return None
    suffix = version.upper()
    exact_prefixes = {
        "v4": CODE_AUDIT_EXACT_FORMAL_PREFIXES_V4,
        "v5": CODE_AUDIT_EXACT_FORMAL_PREFIXES_V5,
    }[version]
    prior_paths = {
        "v4": CODE_AUDIT_EXPECTED_PRIOR_HASH_PATHS,
        "v5": CODE_AUDIT_EXPECTED_PRIOR_HASH_PATHS_V5,
    }[version]
    return {
        "version": version,
        "record_type": "LB_SCGP_G0_CODE_AUDIT_INDEPENDENT_REVIEW_RECORD_{}".format(suffix),
        "artifact_type": "LB_SCGP_G0_CODE_AUDIT_PASS_ARTIFACT_{}".format(suffix),
        "index_type": "LB_SCGP_G0_CODE_AUDIT_PUBLICATION_INDEX_{}".format(suffix),
        "review_scope": "LB-SCGP G0 {} formal code audit for {}".format(
            version, _lineage_run_id(cfg, "freeze")),
        "exact_formal_prefixes": exact_prefixes,
        "prior_hash_paths": prior_paths,
    }


def _read_bytes_logged(path):
    path, rel_path = canonical_root_path(path)
    rel = rel_path.as_posix()
    VERIFIER_ACCESS_LEDGER.append({"path": rel, "purpose": "independent_bytes_read",
                                   "scope": "review_or_formal_artifact"})
    with open(path, "rb") as handle:
        return handle.read()


def _zero_counter_fields(obj, include_formal_outer=False):
    fields = {key: int(obj.get(key, -1)) for key in ZERO_COUNTER_KEYS}
    if include_formal_outer:
        fields["formal_model_optimizer_evaluator_outer_held_read_count"] = int(
            obj.get("formal_model_optimizer_evaluator_outer_held_read_count", -1))
    return fields


def _check_supervision_and_zero_counters(name, obj):
    if obj.get("only_gold_supervision") != "parent_video_binary_label" or \
            obj.get("segment_gold_exists") is not False or \
            obj.get("segment_gold_used") is not False:
        raise RuntimeError("{} supervision contract drift".format(name))
    counters = _zero_counter_fields(obj, include_formal_outer=True)
    if any(value != 0 for value in counters.values()):
        raise RuntimeError("{} has nonzero forbidden counter {}".format(name, counters))


def _config_sha_and_path(cfg):
    rel = _config_rel(cfg)
    path = canonical_root_path(rel)[0]
    return rel, hfile(path), hobj(cfg)


def _artifact_namespace(cfg):
    namespace = cfg.get("lineage", {}).get("artifact_namespace")
    if not isinstance(namespace, str) or namespace != cfg.get("paths", {}).get("artifacts"):
        raise RuntimeError("artifact namespace/config paths drift")
    return namespace


def _verify_dirty_policy_contract(cfg, freeze):
    contract = _code_audit_contract(cfg)
    if contract is None:
        raise RuntimeError("strict code-audit dirty contract requires v4/v5 lineage")
    artifact_prefixes, dirty_paths, dirty_prefixes = dirty_state_policy(cfg)
    if tuple(artifact_prefixes) != contract["exact_formal_prefixes"]:
        raise RuntimeError("{} formal artifact exclusions are not exact".format(
            contract["version"]))
    if list(artifact_prefixes) != freeze.get("formal_artifact_exclude_prefixes") or \
            list(dirty_paths) != freeze.get("dirty_state_excluded_paths") or \
            list(dirty_prefixes) != freeze.get("dirty_state_excluded_prefixes"):
        raise RuntimeError("freeze dirty policy does not match config")
    forbidden_exact = {"", ".", "/", "refine-logs/lb_scgp", "refine-logs/lb_scgp/"}
    if any(path in forbidden_exact for path in dirty_paths):
        raise RuntimeError("dirty path exclusion is too broad")
    if any(prefix in forbidden_exact for prefix in dirty_prefixes):
        raise RuntimeError("dirty prefix exclusion is too broad")
    review_path = _lineage_path(cfg, "review_report_path")
    record_path = _lineage_path(cfg, "review_record_path")
    if review_path not in dirty_paths or record_path not in dirty_paths:
        raise RuntimeError("future review/report sidecar paths are not exact dirty exclusions")
    namespace_prefix = _checked_rel_prefix(_artifact_namespace(cfg), "artifact_namespace")
    if namespace_prefix not in artifact_prefixes:
        raise RuntimeError("{} namespace not excluded as formal artifact prefix".format(
            contract["version"]))
    return {
        "formal_artifact_exclude_prefixes": list(artifact_prefixes),
        "dirty_state_excluded_paths": list(dirty_paths),
        "dirty_state_excluded_prefixes": list(dirty_prefixes),
    }


def _verify_wrapper_contract(cfg):
    wrapper_rel = _lineage_path(cfg, "audit_publish_wrapper_path")
    if wrapper_rel not in cfg.get("implementation_files", []):
        raise RuntimeError("audit-publish wrapper absent from implementation hash set")
    wrapper_path = canonical_root_path(wrapper_rel)[0]
    wrapper_sha = hfile(wrapper_path)
    text = wrapper_path.read_text(encoding="utf-8")
    required_fragments = (
        "#SBATCH --cpus-per-task=2",
        "#SBATCH --mem=4G",
        "TASK=${TASK:?set TASK=audit-publish}",
        "--task audit-publish",
        "--review \"$REVIEW\"",
        "--review-record \"$REVIEW_RECORD\"",
    )
    if "#SBATCH --time" in text or any(fragment not in text for fragment in required_fragments):
        raise RuntimeError("audit-publish wrapper contract invalid")
    return {
        "path": wrapper_rel,
        "sha256": wrapper_sha,
        "cpus_per_task": 2,
        "mem": "4G",
        "no_time_directive": True,
        "conda_env": "HateVideo",
        "offline": True,
    }


def _verify_prior_lineage_hashes(cfg):
    contract = _code_audit_contract(cfg)
    if contract is None:
        raise RuntimeError("strict prior-lineage hashes require v4/v5 lineage")
    expected = cfg.get("lineage", {}).get("prior_lineage_no_clobber_hashes")
    if not isinstance(expected, dict) or set(expected) != set(contract["prior_hash_paths"]):
        raise RuntimeError("prior lineage hash set is missing or too broad")
    observed = []
    for rel in contract["prior_hash_paths"]:
        actual = hfile(ROOT / rel)
        if actual != expected[rel]:
            raise RuntimeError("prior lineage no-clobber hash drift {}".format(rel))
        observed.append({"path": rel, "sha256": actual})
    return observed


def _verify_freeze_for_audit_publish(cfg):
    contract = _code_audit_contract(cfg)
    namespace = _artifact_namespace(cfg)
    if contract is None:
        raise RuntimeError("audit-publish is only authorized for v4/v5")
    artifacts = _resolve_cfg_path(cfg, "artifacts")
    if _relpath(artifacts) != namespace:
        raise RuntimeError("artifact namespace path drift")
    freeze_path = artifacts / "CONFIG_FREEZE.json"
    freeze_lock = freeze_path.with_name(freeze_path.name + ".publish.lock")
    if not freeze_path.exists() or not freeze_lock.exists():
        raise RuntimeError("missing {} freeze artifact or lock".format(
            contract["version"]))
    code_audit_dir = artifacts / CODE_AUDIT_DIR
    if code_audit_dir.exists():
        raise RuntimeError("code-audit namespace already exists")
    freeze = read_json(freeze_path)
    assert_no_forbidden_formal_surface(freeze, "config_freeze")
    freeze_file_sha = hfile(freeze_path)
    freeze_lock_sha = hfile(freeze_lock)
    if not verify_payload(freeze):
        raise RuntimeError("freeze payload hash invalid")
    config_rel, config_file_sha, config_hash = _config_sha_and_path(cfg)
    impl_hash, impl_rows = current_implementation(cfg)
    verifier_hash = hfile(Path(__file__))
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    current_dirty = current_dirty_hash(cfg)
    if freeze.get("run_id") != _lineage_run_id(cfg, "freeze") or \
            freeze.get("stage") != "G0_FREEZE" or freeze.get("status") != "FROZEN":
        raise RuntimeError("freeze identity invalid")
    if freeze.get("lineage_version") != contract["version"] or \
            freeze.get("config_canonical_sha256") != config_hash or \
            freeze.get("implementation_sha256") != impl_hash or \
            freeze.get("independent_verifier_sha256") != verifier_hash:
        raise RuntimeError("freeze code/config/verifier binding invalid")
    if freeze.get("git_head") != head or freeze.get("dirty_diff_sha256") != current_dirty:
        raise RuntimeError("current dirty/head state differs from frozen state")
    if freeze.get("access_ledger_sha256") != hobj(freeze.get("access_ledger", [])):
        raise RuntimeError("freeze access ledger hash invalid")
    _check_supervision_and_zero_counters("freeze", freeze)
    dirty_policy = _verify_dirty_policy_contract(cfg, freeze)
    required_freeze_keys = (
        "checkpoint", "bank", "outer_train_feature_cache",
        "sanitized_provenance", "sanitizer_decision", "remove_ledger",
    )
    expected_paths = {config_rel}
    expected_paths |= {_relpath(_resolve_cfg_path(cfg, key)) for key in required_freeze_keys}
    expected_paths |= {_relpath(_resolve_cfg_path(cfg, key)) for key in _freeze_protocol_input_keys(cfg)}
    input_paths = {row.get("path") for row in freeze.get("input_files", [])}
    ledger_paths = {row.get("path") for row in freeze.get("access_ledger", [])}
    if input_paths != expected_paths or ledger_paths != expected_paths:
        raise RuntimeError("freeze input/access-ledger path contract drift")
    rehashes = []
    for row in freeze.get("input_files", []):
        rel = row.get("path")
        path = canonical_root_path(rel)[0]
        if "member_sha256" in row:
            actual = _hash_npz_allowed_members(path, cfg)
            if actual != row.get("member_sha256"):
                raise RuntimeError("frozen bank member drift {}".format(rel))
            if row.get("forbidden_members_not_opened") != cfg["sealed_real_fixture"]["forbidden_bank_members"]:
                raise RuntimeError("forbidden bank member ledger drift")
            rehashes.append({"path": rel, "member_sha256": actual,
                             "forbidden_members_not_opened": row.get("forbidden_members_not_opened")})
        else:
            actual = hfile(path)
            if actual != row.get("sha256"):
                raise RuntimeError("frozen input drift {}".format(rel))
            rehashes.append({"path": rel, "sha256": actual})
    return {
        "artifacts": artifacts,
        "freeze": freeze,
        "freeze_path": _relpath(freeze_path),
        "freeze_file_sha256": freeze_file_sha,
        "freeze_lock_path": _relpath(freeze_lock),
        "freeze_lock_sha256": freeze_lock_sha,
        "config_path": config_rel,
        "config_file_sha256": config_file_sha,
        "config_canonical_sha256": config_hash,
        "implementation_sha256": impl_hash,
        "implementation_files": impl_rows,
        "independent_verifier_sha256": verifier_hash,
        "git_head": head,
        "dirty_diff_sha256": current_dirty,
        "dirty_policy": dirty_policy,
        "frozen_input_rehashes": rehashes,
    }


def _validate_review_record(cfg, args, freeze_info):
    contract = _code_audit_contract(cfg)
    expected_review_rel = _lineage_path(cfg, "review_report_path")
    expected_record_rel = _lineage_path(cfg, "review_record_path")
    review_path, review_rel = canonical_root_path(args.review)
    record_path, record_rel = canonical_root_path(args.review_record)
    if review_rel.as_posix() != expected_review_rel or record_rel.as_posix() != expected_record_rel:
        raise RuntimeError("review/report sidecar paths are not the exact lineage paths")
    review_bytes = _read_bytes_logged(review_path)
    review_sha = hbytes(review_bytes)
    record_bytes = _read_bytes_logged(record_path)
    record = read_json(record_path)
    record_sha = hbytes(record_bytes)
    _require_exact_keys("review_record", record, CODE_AUDIT_REVIEW_RECORD_KEYS)
    if record.get("payload_sha256") != payload_digest(record):
        raise RuntimeError("review record payload hash invalid")
    checks = {
        "schema_version": record.get("schema_version") == 1,
        "record_type": record.get("record_type") == contract["record_type"],
        "run_id": record.get("run_id") == _lineage_run_id(cfg, "code_audit"),
        "stage": record.get("stage") == "G0_CODE_AUDIT",
        "status": record.get("status") == "PASS",
        "lineage_version": record.get("lineage_version") == contract["version"],
        "config_path": record.get("config_path") == freeze_info["config_path"],
        "artifact_namespace": record.get("artifact_namespace") == _artifact_namespace(cfg),
        "freeze_run_id": record.get("freeze_run_id") == _lineage_run_id(cfg, "freeze"),
        "freeze_path": record.get("freeze_path") == freeze_info["freeze_path"],
        "freeze_file_sha256": record.get("freeze_file_sha256") == freeze_info["freeze_file_sha256"],
        "freeze_payload_sha256": record.get("freeze_payload_sha256") == freeze_info["freeze"]["payload_sha256"],
        "config_canonical_sha256": record.get("config_canonical_sha256") == freeze_info["config_canonical_sha256"],
        "implementation_sha256": record.get("implementation_sha256") == freeze_info["implementation_sha256"],
        "independent_verifier_sha256": record.get("independent_verifier_sha256") == freeze_info["independent_verifier_sha256"],
        "review_report_path": record.get("review_report_path") == expected_review_rel,
        "review_report_sha256": record.get("review_report_sha256") == review_sha,
        "process": record.get("review_process_identity") == CODE_AUDIT_REVIEW_PROCESS_IDENTITY,
        "scope": record.get("review_scope") == contract["review_scope"],
        "critical": int(record.get("critical", -1)) == 0,
        "high": int(record.get("high", -1)) == 0,
        "important": isinstance(record.get("important"), int) and int(record.get("important")) >= 0,
        "no_segment": record.get("no_segment_gold_pass") is True,
        "authorized": record.get("formal_pass_authorized") is True,
        "independent": record.get("independent_reviewer") is True,
        "not_repair_executor": record.get("repair_executor_created") is False,
        "supervision": record.get("only_gold_supervision") == "parent_video_binary_label" and
            record.get("segment_gold_exists") is False and record.get("segment_gold_used") is False,
    }
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError("review record failed strict checks {}".format(failed))
    _require_nonempty_string("reviewer_identity", record.get("reviewer_identity"))
    return {
        "record": record,
        "record_sha256": record_sha,
        "record_bytes": record_bytes,
        "review_bytes": review_bytes,
        "review_sha256": review_sha,
    }


def _lock_bytes(run_id, rel):
    return (cjson({"lock_type": "LB_SCGP_FORMAL_NO_CLOBBER",
                   "run_id": run_id, "path": rel}) + "\n").encode("utf-8")


def _transaction_publish_code_audit(final_dir, file_payloads, lock_payloads):
    if final_dir.exists():
        raise RuntimeError("code-audit final directory already exists")
    parent = final_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp = parent / (".code_audit.publish.tmp.{}.{}".format(
        os.environ.get("SLURM_JOB_ID", "noslurm"), os.getpid()))
    if tmp.exists():
        raise RuntimeError("temporary code-audit transaction path already exists")
    tmp.mkdir(mode=0o700)
    try:
        for name, payload in file_payloads.items():
            path = tmp / name
            with open(path, "xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            lock_path = tmp / (name + ".publish.lock")
            with open(lock_path, "xb") as handle:
                handle.write(lock_payloads[name])
                handle.flush()
                os.fsync(handle.fileno())
        dfd = os.open(str(tmp), os.O_RDONLY)
        try:
            os.fsync(dfd)
        finally:
            os.close(dfd)
        os.rename(str(tmp), str(final_dir))
        pfd = os.open(str(parent), os.O_RDONLY)
        try:
            os.fsync(pfd)
        finally:
            os.close(pfd)
    except Exception:
        if tmp.exists():
            shutil.rmtree(tmp)
        raise


def audit_publish(cfg, args):
    contract = _code_audit_contract(cfg)
    if args.run_id != _lineage_run_id(cfg, "code_audit"):
        raise RuntimeError("wrong code-audit run id")
    if contract is None:
        raise RuntimeError("audit-publish is only authorized for v4/v5")
    if not args.review or not args.review_record:
        raise RuntimeError("audit-publish requires --review and --review-record")
    assert_no_forbidden_formal_surface(cfg, "formal_config")
    freeze_info = _verify_freeze_for_audit_publish(cfg)
    review_info = _validate_review_record(cfg, args, freeze_info)
    prior_hashes = _verify_prior_lineage_hashes(cfg)
    wrapper = _verify_wrapper_contract(cfg)
    import torch
    artifacts = freeze_info["artifacts"]
    final_dir = artifacts / CODE_AUDIT_DIR
    output_names = (
        CODE_AUDIT_REVIEW_ARTIFACT,
        CODE_AUDIT_RECORD_ARTIFACT,
        CODE_AUDIT_JSON_ARTIFACT,
        CODE_AUDIT_INDEX_ARTIFACT,
    )
    for name in output_names:
        if (final_dir / name).exists() or (final_dir / (name + ".publish.lock")).exists():
            raise RuntimeError("preexisting code-audit output or lock {}".format(name))
    review_rel = _relpath(final_dir / CODE_AUDIT_REVIEW_ARTIFACT)
    record_rel = _relpath(final_dir / CODE_AUDIT_RECORD_ARTIFACT)
    audit_rel = _relpath(final_dir / CODE_AUDIT_JSON_ARTIFACT)
    index_rel = _relpath(final_dir / CODE_AUDIT_INDEX_ARTIFACT)
    review_output = {"path": review_rel, "sha256": review_info["review_sha256"]}
    record_output = {"path": record_rel, "sha256": review_info["record_sha256"]}
    auth_gate = {
        "wrapper_task": "audit-publish",
        "wrapper_run_id": args.run_id,
        "config_path": freeze_info["config_path"],
        "review_path": _lineage_path(cfg, "review_report_path"),
        "review_record_path": _lineage_path(cfg, "review_record_path"),
        "no_manual_pass_written": True,
        "publisher_recomputed_binding_checks": True,
    }
    audit = {
        "schema_version": 1,
        "artifact_type": contract["artifact_type"],
        "run_id": args.run_id,
        "stage": "G0_CODE_AUDIT",
        "status": "PASS",
        "critical": 0,
        "high": 0,
        "important": int(review_info["record"]["important"]),
        "no_segment_gold_pass": True,
        "formal_pass_authorized": True,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "config_path": freeze_info["config_path"],
        "config_file_sha256": freeze_info["config_file_sha256"],
        "config_canonical_sha256": freeze_info["config_canonical_sha256"],
        "artifact_namespace": _artifact_namespace(cfg),
        "lineage_version": contract["version"],
        "freeze_path": freeze_info["freeze_path"],
        "freeze_file_sha256": freeze_info["freeze_file_sha256"],
        "freeze_lock_path": freeze_info["freeze_lock_path"],
        "freeze_lock_sha256": freeze_info["freeze_lock_sha256"],
        "freeze_payload_sha256": freeze_info["freeze"]["payload_sha256"],
        "freeze_run_id": freeze_info["freeze"]["run_id"],
        "freeze_stage": freeze_info["freeze"]["stage"],
        "git_head": freeze_info["git_head"],
        "dirty_diff_sha256": freeze_info["dirty_diff_sha256"],
        "frozen_dirty_diff_sha256": freeze_info["freeze"]["dirty_diff_sha256"],
        "implementation_sha256": freeze_info["implementation_sha256"],
        "implementation_files": freeze_info["implementation_files"],
        "independent_verifier_sha256": freeze_info["independent_verifier_sha256"],
        "review_report_path": _lineage_path(cfg, "review_report_path"),
        "review_report_sha256": review_info["review_sha256"],
        "review_record_path": _lineage_path(cfg, "review_record_path"),
        "review_record_sha256": review_info["record_sha256"],
        "review_record_payload_sha256": review_info["record"]["payload_sha256"],
        "reviewer_identity": review_info["record"]["reviewer_identity"],
        "review_process_identity": review_info["record"]["review_process_identity"],
        "review_scope": review_info["record"]["review_scope"],
        "only_gold_supervision": "parent_video_binary_label",
        "segment_gold_exists": False,
        "segment_gold_used": False,
        "mllm_call_count": 0,
        "ocr_call_count": 0,
        "teacher_cache_read_count": 0,
        "teacher_cache_write_count": 0,
        "outer_held_label_read_count": 0,
        "outer_held_content_read_count": 0,
        "val_content_read_count": 0,
        "test_content_read_count": 0,
        "val_test_teacher_artifact_count": 0,
        "formal_model_optimizer_evaluator_outer_held_read_count": 0,
        "access_ledger": list(VERIFIER_ACCESS_LEDGER),
        "access_ledger_sha256": hobj(VERIFIER_ACCESS_LEDGER),
        "dirty_policy": freeze_info["dirty_policy"],
        "frozen_input_rehashes": freeze_info["frozen_input_rehashes"],
        "allowed_bank_member_sha256": cfg["sealed_real_fixture"]["bank_member_sha256"],
        "forbidden_bank_members_not_opened": cfg["sealed_real_fixture"]["forbidden_bank_members"],
        "prior_lineage_no_clobber_hashes": prior_hashes,
        "audit_publish_wrapper": wrapper,
        "authorization_gate": auth_gate,
        "downstream_contract": {
            "producer_consumer": "_load_freeze_and_audit",
            "decision_consumer": "lb_scgp_independent_verify.decide",
            "strict_schema": contract["artifact_type"],
            "additional_fields_rejected": True,
        },
        "output_files": [review_output, record_output],
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": torch.__version__,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    _require_exact_keys("code_audit", audit, CODE_AUDIT_ARTIFACT_KEYS - {"payload_sha256"})
    audit["payload_sha256"] = payload_digest(audit)
    _require_exact_keys("code_audit", audit, CODE_AUDIT_ARTIFACT_KEYS)
    audit_bytes = (cjson(audit) + "\n").encode("utf-8")
    lock_payloads = {
        CODE_AUDIT_REVIEW_ARTIFACT: _lock_bytes(args.run_id, review_rel),
        CODE_AUDIT_RECORD_ARTIFACT: _lock_bytes(args.run_id, record_rel),
        CODE_AUDIT_JSON_ARTIFACT: _lock_bytes(args.run_id, audit_rel),
        CODE_AUDIT_INDEX_ARTIFACT: _lock_bytes(args.run_id, index_rel),
    }
    index = {
        "schema_version": 1,
        "artifact_type": contract["index_type"],
        "run_id": args.run_id,
        "stage": "G0_CODE_AUDIT",
        "status": "PASS",
        "output_files": [
            review_output,
            record_output,
            {"path": audit_rel, "sha256": hbytes(audit_bytes)},
        ],
        "lock_files": [
            {"path": _relpath(final_dir / (name + ".publish.lock")),
             "sha256": hbytes(lock_payloads[name])}
            for name in output_names
        ],
    }
    index["payload_sha256"] = payload_digest(index)
    _require_exact_keys("publication_index", index, CODE_AUDIT_INDEX_KEYS)
    file_payloads = {
        CODE_AUDIT_REVIEW_ARTIFACT: review_info["review_bytes"],
        CODE_AUDIT_RECORD_ARTIFACT: review_info["record_bytes"],
        CODE_AUDIT_JSON_ARTIFACT: audit_bytes,
        CODE_AUDIT_INDEX_ARTIFACT: (cjson(index) + "\n").encode("utf-8"),
    }
    _transaction_publish_code_audit(final_dir, file_payloads, lock_payloads)
    print(cjson({"status": "PASS", "run_id": args.run_id,
                 "payload_sha256": audit["payload_sha256"]}))


def verify_code_audit_publication(cfg, artifacts, freeze, audit, config_hash, impl_hash):
    contract = _code_audit_contract(cfg)
    if contract is None:
        return verify_payload(audit) and audit.get("status") == "PASS" and \
            audit.get("run_id") == _lineage_run_id(cfg, "code_audit") and \
            audit.get("stage") == "G0_CODE_AUDIT" and \
            int(audit.get("critical", -1)) == 0 and int(audit.get("high", -1)) == 0 and \
            audit.get("config_canonical_sha256") == config_hash and \
            audit.get("implementation_sha256") == impl_hash and \
            audit.get("access_ledger_sha256") == hobj(audit.get("access_ledger", []))
    assert_no_forbidden_formal_surface(audit, "code_audit")
    _require_exact_keys("code_audit", audit, CODE_AUDIT_ARTIFACT_KEYS)
    if not verify_payload(audit):
        return False
    code_dir = artifacts / CODE_AUDIT_DIR
    index_path = code_dir / CODE_AUDIT_INDEX_ARTIFACT
    review_path = code_dir / CODE_AUDIT_REVIEW_ARTIFACT
    record_path = code_dir / CODE_AUDIT_RECORD_ARTIFACT
    audit_path = code_dir / CODE_AUDIT_JSON_ARTIFACT
    index = read_json(index_path)
    _require_exact_keys("publication_index", index, CODE_AUDIT_INDEX_KEYS)
    if not verify_payload(index):
        return False
    review_sha = hfile(review_path)
    record_sha = hfile(record_path)
    audit_sha = hfile(audit_path)
    record = read_json(record_path)
    _require_exact_keys("review_record", record, CODE_AUDIT_REVIEW_RECORD_KEYS)
    if record.get("payload_sha256") != payload_digest(record):
        return False
    lock_ok = True
    lock_files = index.get("lock_files", [])
    if len(lock_files) != 4:
        return False
    for row in lock_files:
        lock_path = canonical_root_path(row.get("path"))[0]
        lock_ok &= lock_path.exists() and hfile(lock_path) == row.get("sha256")
    expected_outputs = [
        {"path": _relpath(review_path), "sha256": review_sha},
        {"path": _relpath(record_path), "sha256": record_sha},
        {"path": _relpath(audit_path), "sha256": audit_sha},
    ]
    if index.get("output_files") != expected_outputs:
        return False
    if audit.get("output_files") != expected_outputs[:2]:
        return False
    freeze_path = artifacts / "CONFIG_FREEZE.json"
    freeze_lock = freeze_path.with_name(freeze_path.name + ".publish.lock")
    dirty_policy = _verify_dirty_policy_contract(cfg, freeze)
    prior_hashes = _verify_prior_lineage_hashes(cfg)
    wrapper = _verify_wrapper_contract(cfg)
    checks = {
        "artifact_type": audit.get("artifact_type") == contract["artifact_type"],
        "identity": audit.get("run_id") == _lineage_run_id(cfg, "code_audit") and
            audit.get("stage") == "G0_CODE_AUDIT" and audit.get("status") == "PASS",
        "index": index.get("artifact_type") == contract["index_type"] and
            index.get("run_id") == audit.get("run_id") and
            index.get("stage") == audit.get("stage") and index.get("status") == "PASS",
        "findings": int(audit.get("critical", -1)) == 0 and
            int(audit.get("high", -1)) == 0 and audit.get("no_segment_gold_pass") is True and
            audit.get("formal_pass_authorized") is True,
        "config": audit.get("config_canonical_sha256") == config_hash and
            audit.get("config_path") == _config_rel(cfg) and
            audit.get("config_file_sha256") == hfile(canonical_root_path(_config_rel(cfg))[0]),
        "freeze": audit.get("freeze_path") == _relpath(freeze_path) and
            audit.get("freeze_file_sha256") == hfile(freeze_path) and
            audit.get("freeze_lock_path") == _relpath(freeze_lock) and
            freeze_lock.exists() and audit.get("freeze_lock_sha256") == hfile(freeze_lock) and
            audit.get("freeze_payload_sha256") == freeze.get("payload_sha256") and
            audit.get("freeze_run_id") == freeze.get("run_id") and
            audit.get("freeze_stage") == freeze.get("stage"),
        "implementation": audit.get("implementation_sha256") == impl_hash and
            audit.get("independent_verifier_sha256") == hfile(Path(__file__)),
        "dirty": audit.get("dirty_diff_sha256") == current_dirty_hash(cfg) and
            audit.get("frozen_dirty_diff_sha256") == freeze.get("dirty_diff_sha256") and
            audit.get("dirty_diff_sha256") == audit.get("frozen_dirty_diff_sha256"),
        "lineage": audit.get("artifact_namespace") == _artifact_namespace(cfg) and
            audit.get("lineage_version") == contract["version"],
        "record": audit.get("review_report_path") == record.get("review_report_path") and
            audit.get("review_report_sha256") == review_sha and
            audit.get("review_record_path") == _lineage_path(cfg, "review_record_path") and
            audit.get("review_record_sha256") == record_sha and
            audit.get("review_record_payload_sha256") == record.get("payload_sha256") and
            audit.get("reviewer_identity") == record.get("reviewer_identity") and
            audit.get("review_process_identity") == CODE_AUDIT_REVIEW_PROCESS_IDENTITY and
            audit.get("review_scope") == contract["review_scope"],
        "record_values": record.get("record_type") == contract["record_type"] and
            record.get("run_id") == audit.get("run_id") and record.get("stage") == audit.get("stage") and
            record.get("status") == "PASS" and int(record.get("critical", -1)) == 0 and
            int(record.get("high", -1)) == 0 and record.get("no_segment_gold_pass") is True and
            record.get("formal_pass_authorized") is True and
            record.get("independent_reviewer") is True and record.get("repair_executor_created") is False and
            record.get("lineage_version") == contract["version"] and
            record.get("artifact_namespace") == _artifact_namespace(cfg) and
            record.get("review_scope") == contract["review_scope"],
        "access": audit.get("access_ledger_sha256") == hobj(audit.get("access_ledger", [])),
        "dirty_policy": audit.get("dirty_policy") == dirty_policy,
        "prior": audit.get("prior_lineage_no_clobber_hashes") == prior_hashes,
        "wrapper": audit.get("audit_publish_wrapper") == wrapper,
        "supervision": audit.get("only_gold_supervision") == "parent_video_binary_label" and
            audit.get("segment_gold_exists") is False and audit.get("segment_gold_used") is False and
            record.get("only_gold_supervision") == "parent_video_binary_label" and
            record.get("segment_gold_exists") is False and record.get("segment_gold_used") is False,
        "zero": all(value == 0 for value in _zero_counter_fields(
            audit, include_formal_outer=True).values()),
        "bank": audit.get("allowed_bank_member_sha256") == cfg["sealed_real_fixture"]["bank_member_sha256"] and
            audit.get("forbidden_bank_members_not_opened") == cfg["sealed_real_fixture"]["forbidden_bank_members"],
        "lock": lock_ok,
    }
    return all(checks.values())


def _independent_ball_reference(row):
    y = np.asarray(row["input"], dtype=np.float64)
    center = np.asarray(row["center"], dtype=np.float64)
    op = np.asarray(row["operator"], dtype=np.float64)
    radius = float(row["radius"])
    projected = np.asarray(row["projected"], dtype=np.float64)
    if radius == 0.0:
        gram=op@op.T
        reference=y-op.T@(np.linalg.pinv(gram,rcond=1e-14)@(op@(y-center)))
        reference_success=True
    else:
        result = minimize(
            lambda x: 0.5 * float(np.dot(x-y, x-y)), y.copy(), jac=lambda x: x-y,
            constraints={"type":"ineq",
                         "fun":lambda x: radius**2-float(np.dot(op@(x-center), op@(x-center))),
                         "jac":lambda x: -2.0*op.T@(op@(x-center))},
            method="SLSQP", options={"ftol":1e-13,"maxiter":4000})
        reference_success=result.success; reference=result.x if result.success else None
    if not reference_success:
        return False, {"case": row["case"], "reason": "reference_failed"}
    feasibility = max(0.0, float(np.linalg.norm(op@(projected-center))) - radius)
    error = float(np.linalg.norm(projected-reference))
    if radius == 0.0:
        idem_reference=projected-op.T@(np.linalg.pinv(op@op.T,rcond=1e-14)@(op@(projected-center)))
        idem_error=float(np.linalg.norm(idem_reference-projected))
    else:
        idem = minimize(
            lambda x: 0.5*float(np.dot(x-projected, x-projected)), projected.copy(),
            jac=lambda x:x-projected,
            constraints={"type":"ineq",
                         "fun":lambda x: radius**2-float(np.dot(op@(x-center), op@(x-center))),
                         "jac":lambda x:-2.0*op.T@(op@(x-center))},
            method="SLSQP", options={"ftol":1e-13,"maxiter":4000})
        idem_error = float(np.linalg.norm(idem.x-projected)) if idem.success else float("inf")
    rng = np.random.default_rng(int(row["probe_seed"]))
    normal = y-projected; vi = -float("inf"); directional_min=float("inf")
    for _ in range(int(row["probe_count"])):
        raw=rng.normal(size=y.size)
        if radius==0.0:
            raw=raw-op.T@(np.linalg.pinv(op@op.T,rcond=1e-14)@(op@raw)); candidate=center+raw
        else:
            candidate=center+raw; value=op@(candidate-center); value_norm=float(np.linalg.norm(value))
            if value_norm>radius and value_norm>0:
                candidate=center+(candidate-center)*(radius/value_norm)*0.999999
        vi=max(vi,float(normal@(candidate-projected)))
        direction=candidate-projected; norm=float(np.linalg.norm(direction))
        if norm>0:
            direction/=norm; eps=1e-7; plus=projected+eps*direction
            if float(np.linalg.norm(op@(plus-center)))<=radius+1e-10:
                directional_min=min(directional_min,(0.5*np.dot(plus-y,plus-y)-0.5*np.dot(projected-y,projected-y))/eps)
    op_value=op@(projected-center); root=abs(float(np.linalg.norm(op_value))-radius)
    if radius==0.0:
        stationarity=float(np.linalg.norm(normal-op.T@(np.linalg.pinv(op@op.T,rcond=1e-14)@(op@normal))))
        complementarity=0.0
    else:
        kkt_vector=op.T@op_value; mu=max(0.0,float(normal@kkt_vector)/max(float(kkt_vector@kkt_vector),1e-30))
        stationarity=float(np.linalg.norm(normal-mu*kkt_vector)); complementarity=abs(mu*(float(np.linalg.norm(op_value))-radius))
    fd=max(0.0,-directional_min) if directional_min<float("inf") else 0.0
    ok = feasibility <= 1e-7 and error <= 1e-7 and idem_error <= 1e-7 and max(0.0,vi) <= 1e-8 and stationarity<=1e-7 and complementarity<=1e-7 and root<=1e-7 and fd<=1e-7
    return ok, {"case":row["case"],"feasibility":feasibility,
                "dense_reference_error":error,"idempotence":idem_error,
                "variational_inequality_max":max(0.0,vi),
                "kkt_stationarity":stationarity,"complementarity":complementarity,
                "scalar_root_residual":root,"finite_difference_optimum_violation":fd}


def verify_projectors(rows):
    checks=[]; ok=True
    for row in rows:
        if row["case"] in {"row_ball","class_mean_ball","semantic_positive_radius","semantic_zero_radius"}:
            passed, detail=_independent_ball_reference(row)
        elif row["case"]=="psd_symmetrized":
            y=np.asarray(row["input"],dtype=np.float64); p=np.asarray(row["projected"],dtype=np.float64)
            sym=0.5*(y+y.T); eigval,eigvec=np.linalg.eigh(sym)
            reference=(eigvec*np.maximum(eigval,0.0))@eigvec.T
            detail={"case":row["case"],"reference_error":float(np.linalg.norm(reference-p)),
                    "minimum_eigenvalue":float(np.linalg.eigvalsh(p).min()),
                    "symmetry_error":float(np.max(np.abs(p-p.T))),
                    "kkt_stationarity":max(0.0,float(np.linalg.eigvalsh(0.5*(y+y.T)-p).max())),
                    "complementarity":float(np.linalg.norm((0.5*(y+y.T)-p)@p))}
            rng=np.random.default_rng(int(row["probe_seed"])); normal=y-p; vi=-float("inf"); fd=float("inf")
            for _ in range(int(row["probe_count"])):
                aa=rng.normal(size=p.shape); z=aa@aa.T; vi=max(vi,float(np.sum(normal*(z-p))))
                d=z-p; norm=float(np.linalg.norm(d))
                if norm>0:
                    d/=norm; eps=1e-7; plus=p+eps*d
                    fd=min(fd,(0.5*np.linalg.norm(plus-y)**2-0.5*np.linalg.norm(p-y)**2)/eps)
            detail["variational_inequality_max"]=max(0.0,vi); detail["finite_difference_optimum_violation"]=max(0.0,-fd)
            passed=detail["reference_error"]<=1e-7 and detail["minimum_eigenvalue"]>=-1e-7 and detail["symmetry_error"]<=1e-10 and detail["kkt_stationarity"]<=1e-7 and detail["complementarity"]<=1e-7 and detail["variational_inequality_max"]<=1e-8 and detail["finite_difference_optimum_violation"]<=1e-7
        elif row["case"]=="halfspace":
            y=np.asarray(row["input"],dtype=np.float64); a=np.asarray(row["normal"],dtype=np.float64)
            rhs=float(row["rhs"]); p=np.asarray(row["projected"],dtype=np.float64)
            tau=max(0.0,(rhs-float(a@y))/float(a@a)); reference=y+tau*a
            detail={"case":row["case"],"reference_error":float(np.linalg.norm(reference-p)),
                    "feasibility":max(0.0,rhs-float(a@p)),
                    "kkt_stationarity":float(np.linalg.norm((y-p)+tau*a)),
                    "complementarity":abs(tau*(float(a@p)-rhs))}
            rng=np.random.default_rng(int(row["probe_seed"])); normal=y-p; vi=-float("inf"); fd=float("inf")
            for _ in range(int(row["probe_count"])):
                z=rng.normal(size=len(y))
                if float(a@z)<rhs:z+=((rhs-float(a@z))/float(a@a))*a
                vi=max(vi,float(normal@(z-p))); d=z-p; norm=float(np.linalg.norm(d))
                if norm>0:
                    d/=norm; eps=1e-7; plus=p+eps*d
                    if float(a@plus)>=rhs-1e-12:fd=min(fd,(0.5*np.dot(plus-y,plus-y)-0.5*np.dot(p-y,p-y))/eps)
            detail["variational_inequality_max"]=max(0.0,vi); detail["finite_difference_optimum_violation"]=max(0.0,-fd)
            passed=detail["reference_error"]<=1e-7 and detail["feasibility"]<=1e-7 and detail["kkt_stationarity"]<=1e-7 and detail["complementarity"]<=1e-7 and detail["variational_inequality_max"]<=1e-8 and detail["finite_difference_optimum_violation"]<=1e-7
        elif row["case"]=="slack_capped_simplex":
            y=np.asarray(row["input"],dtype=np.float64); p=np.asarray(row["projected"],dtype=np.float64); cap=float(row["cap"])
            ref=minimize(lambda x:0.5*float(np.dot(x-y,x-y)),np.maximum(y,0),jac=lambda x:x-y,
                         bounds=[(0,None)]*len(y),
                         constraints={"type":"ineq","fun":lambda x:cap-float(x.sum()),
                                      "jac":lambda x:-np.ones(len(y))},method="SLSQP",
                         options={"ftol":1e-13,"maxiter":1000})
            rng=np.random.default_rng(int(row["probe_seed"])); normal=y-p; vi=-float("inf"); fd=float("inf")
            for _ in range(int(row["probe_count"])):
                z=rng.random(len(y)); z=z/max(float(z.sum()),1e-15)*cap*rng.random(); vi=max(vi,float(normal@(z-p)))
                d=z-p; norm=float(np.linalg.norm(d))
                if norm>0:
                    d/=norm; eps=1e-7; plus=p+eps*d
                    if float(plus.min())>=-1e-12 and float(plus.sum())<=cap+1e-12:fd=min(fd,(0.5*np.dot(plus-y,plus-y)-0.5*np.dot(p-y,p-y))/eps)
            active=p>1e-10; lam=float(np.mean(y[active]-p[active])) if active.any() and abs(float(p.sum()-cap))<=1e-8 else 0.0
            nu=p-y+lam; nu[active]=0.0
            def project_again(v):
                pos=np.maximum(v,0.0)
                if float(pos.sum())<=cap:return pos
                lo,hi=float(pos.min()-cap),float(pos.max())
                for _ in range(120):
                    mid=0.5*(lo+hi)
                    if float(np.maximum(pos-mid,0).sum())>cap:lo=mid
                    else:hi=mid
                return np.maximum(pos-0.5*(lo+hi),0)
            detail={"case":row["case"],"reference_error":float(np.linalg.norm(ref.x-p)),
                    "feasibility":max(0.0,float(-p.min()),float(p.sum()-cap)),
                    "variational_inequality_max":max(0.0,vi),"idempotence":float(np.linalg.norm(project_again(p)-p)),
                    "kkt_stationarity":float(np.linalg.norm((p-y)+lam-np.maximum(nu,0))),
                    "complementarity":abs(lam*(float(p.sum())-cap))+abs(float(np.maximum(nu,0)@p)),
                    "finite_difference_optimum_violation":max(0.0,-fd)}
            passed=ref.success and all(detail[k]<=1e-7 for k in ("reference_error","feasibility","idempotence","kkt_stationarity","complementarity","finite_difference_optimum_violation")) and detail["variational_inequality_max"]<=1e-8
        else:
            passed=False; detail={"case":row.get("case"),"reason":"unknown_projector"}
        metrics=row.get("metrics",{})
        metric_gate=all(math.isfinite(float(v)) for v in metrics.values())
        metric_gate &= float(metrics.get("operator_adjoint_dot_error",0.0))<=1e-10
        for key in ("feasibility","kkt_stationarity","complementarity",
                    "scalar_root_residual","idempotence","finite_difference_optimum_violation",
                    "dense_reference_error"):
            metric_gate &= float(metrics.get(key,float("inf")))<=1e-7
        metric_gate &= float(metrics.get("variational_inequality_max",float("inf")))<=1e-8
        passed=passed and metric_gate; detail["producer_metric_gate"]=bool(metric_gate)
        checks.append(detail); ok &= passed
    return bool(ok), checks


def _centroid_value(gram,labels):
    groups=[np.flatnonzero(labels==v) for v in (0,1)]
    return float(gram[np.ix_(groups[0],groups[0])].mean()+
                 gram[np.ix_(groups[1],groups[1])].mean()-
                 gram[np.ix_(groups[0],groups[1])].mean()-
                 gram[np.ix_(groups[1],groups[0])].mean())


def _margin_vector(gram,labels,ids):
    values=[]
    for i in range(len(ids)):
        vote=0.0
        for rank,j in enumerate(_rank(gram,ids,i)[:20],1):
            vote+=(21-rank)*float(gram[i,j])*(2*int(labels[j])-1)
        values.append((2*int(labels[i])-1)*vote/210.0)
    return np.asarray(values)


def _product_violations(row):
    fixture=row["fixture"]; n=int(fixture["n"]); labels=np.asarray(fixture["labels"],dtype=np.int64)
    ids=fixture["ids"]; gram0=np.asarray(fixture["gram0"],dtype=np.float64)
    semantic=np.asarray(fixture["semantic"],dtype=np.float64); ell=np.asarray(fixture["ell"],dtype=np.float64)
    x=np.asarray(row["projected"],dtype=np.float64); gram=x[:n*n].reshape(n,n); slack=x[n*n:]
    values={"symmetry":float(np.max(np.abs(gram-gram.T))),
            "diagonal":float(np.max(np.abs(np.diag(gram)-1.0))),
            "psd":max(0.0,-float(np.linalg.eigvalsh(0.5*(gram+gram.T)).min())),
            "box":max(0.0,float(gram[~np.eye(n,dtype=bool)].max()-0.9999),
                      float(-1.0-gram[~np.eye(n,dtype=bool)].min())),
            "semantic":float(np.linalg.norm(semantic@gram.reshape(-1)))}
    values["row_trust"]=max(max(0.0,float(np.linalg.norm(
        np.delete(gram[i]-gram0[i],i))-0.05*math.sqrt(n-1))) for i in range(n))
    values["class_mean_trust"]=max(max(0.0,float(np.linalg.norm(
        (gram[labels==v]-gram0[labels==v]).mean(axis=0))-0.02*math.sqrt(n))) for v in (0,1))
    selected=fixture.get("selected_cell_rankings") or [_rank(gram0,ids,i) for i in range(n)]
    def margins_for(g,rankings):
        out=[]
        for i,ranking in enumerate(rankings):
            vote=sum((21-r)*float(g[i,j])*(2*int(labels[j])-1)
                     for r,j in enumerate(ranking[:20],1))
            out.append((2*int(labels[i])-1)*vote/210.0)
        return np.asarray(out)
    margins=margins_for(gram,selected); margins0=_margin_vector(gram0,labels,ids)
    deficits=np.maximum(ell-margins0,0.0)
    values["slack"]=max(max(0.0,float(-slack[labels==v].min()),
                                float(slack[labels==v].sum()-0.2*deficits[labels==v].sum())) for v in (0,1))
    values["vote"]=max(0.0,float(np.max(ell-margins-slack)))
    values["class_margin"]=max(max(0.0,float(margins0[labels==v].mean()-margins[labels==v].mean())) for v in (0,1))
    values["global_margin"]=max(0.0,float(margins0.mean()-margins.mean()))
    values["centroid"]=max(0.0,_centroid_value(gram0,labels)-_centroid_value(gram,labels))
    rank_violation=0.0; rank_count=0
    for i in range(n):
        full=selected[i]; top=full[:20]
        for r in range(19):
            a,b=top[r],top[r+1]; rhs=-1e-7 if str(ids[a])<str(ids[b]) else float(np.nextafter(1e-7,math.inf))
            rank_violation=max(rank_violation,rhs-float(gram[i,a]-gram[i,b])); rank_count+=1
        for outsider in full[20:]:
            a,b=top[19],outsider; rhs=-1e-7 if str(ids[a])<str(ids[b]) else float(np.nextafter(1e-7,math.inf))
            rank_violation=max(rank_violation,rhs-float(gram[i,a]-gram[i,b])); rank_count+=1
    values["rank"]=max(0.0,rank_violation)
    cell_equal=all(_rank(gram,ids,i)[:20]==selected[i][:20] for i in range(n))
    return values,rank_count,cell_equal


def _rank_cell_equal(gram,gram0,ids):
    return all(_rank(gram,ids,i)[:20]==_rank(gram0,ids,i)[:20] for i in range(len(ids)))


def _independent_product_sets(row):
    f=row["fixture"]; n=int(f["n"]); labels=np.asarray(f["labels"],dtype=np.int64)
    ids=f["ids"]; g0=np.asarray(f["gram0"],dtype=np.float64); semantic=np.asarray(f["semantic"],dtype=np.float64)
    ell=np.asarray(f["ell"],dtype=np.float64); size=n*n+n; sets=[]
    def add(name,project): sets.append((name,project))
    def sym(x): y=x.copy(); g=y[:n*n].reshape(n,n); y[:n*n]=(0.5*(g+g.T)).reshape(-1); return y
    add("symmetry",sym)
    def diag(x): y=x.copy(); g=y[:n*n].reshape(n,n); np.fill_diagonal(g,1.0); return y
    add("correlation_diagonal",diag)
    def psd(x):
        y=x.copy(); g=y[:n*n].reshape(n,n); ev,vec=np.linalg.eigh(0.5*(g+g.T)); y[:n*n]=((vec*np.maximum(ev,0))@vec.T).reshape(-1); return y
    add("psd_symmetrized_input",psd)
    mask=(~np.eye(n,dtype=bool)).reshape(-1)
    def box(x): y=x.copy(); y[:n*n][mask]=np.clip(y[:n*n][mask],-1.0,0.9999); return y
    add("offdiagonal_box",box)
    for i in range(n):
        radius=0.05*math.sqrt(n-1); cols=np.asarray([j for j in range(n) if j!=i])
        def rowp(x,i=i,cols=cols,radius=radius):
            y=x.copy(); g=y[:n*n].reshape(n,n); q=g[i,cols]-g0[i,cols]; norm=float(np.linalg.norm(q))
            if norm>radius: g[i,cols]=g0[i,cols]+q*(radius/norm)
            return y
        add("row_trust_{:02d}".format(i),rowp)
    for label in (0,1):
        rows=np.flatnonzero(labels==label); radius=0.02*math.sqrt(n)
        def classp(x,rows=rows,radius=radius):
            y=x.copy(); g=y[:n*n].reshape(n,n); q=(g[rows]-g0[rows]).mean(axis=0); norm=float(np.linalg.norm(q))
            if norm>radius: g[rows]-=(1.0-radius/norm)*q
            return y
        add("class_mean_trust_{}".format(label),classp)
    def semp(x):
        y=x.copy(); q=semantic@y[:n*n]; y[:n*n]-=semantic.T@(np.linalg.pinv(semantic@semantic.T,rcond=1e-14)@q); return y
    add("semantic_radius_zero",semp)
    margins0=_margin_vector(g0,labels,ids); deficits=np.maximum(ell-margins0,0.0)
    for label in (0,1):
        rows=np.flatnonzero(labels==label); cap=0.2*float(deficits[rows].sum())
        def slackp(x,rows=rows,cap=cap):
            y=x.copy(); v=np.maximum(y[n*n+rows],0.0)
            if float(v.sum())>cap:
                lo,hi=float(v.min()-cap),float(v.max())
                for _ in range(120):
                    mid=0.5*(lo+hi)
                    if float(np.maximum(v-mid,0).sum())>cap: lo=mid
                    else: hi=mid
                v=np.maximum(v-0.5*(lo+hi),0)
            y[n*n+rows]=v; return y
        add("slack_capped_simplex_{}".format(label),slackp)
    def half(name,a,rhs):
        norm=float(a@a)
        def hp(x,a=a,rhs=rhs,norm=norm):
            value=float(a@x); return x if value>=rhs else x+((rhs-value)/norm)*a
        add(name,hp)
    directions=[]; rankings=f.get("selected_cell_rankings") or [_rank(g0,ids,i) for i in range(n)]
    for i,ranking in enumerate(rankings):
        a=np.zeros(size); sign=2*int(labels[i])-1
        for rank,j in enumerate(ranking[:20],1): a[i*n+j]=sign*(21-rank)*(2*int(labels[j])-1)/210.0
        directions.append(a); full=a.copy(); full[n*n+i]=1.0; half("vote_slack_{:02d}".format(i),full,float(ell[i]))
    for label in (0,1):
        rows=np.flatnonzero(labels==label); a=sum((directions[i] for i in rows),np.zeros(size))/len(rows)
        half("class_mean_margin_{}".format(label),a,float(margins0[rows].mean()))
    half("global_mean_margin",sum(directions,np.zeros(size))/n,float(margins0.mean()))
    a=np.zeros(size); groups=[np.flatnonzero(labels==v) for v in (0,1)]
    for rows in groups:
        for i in rows: a[i*n+rows]+=1.0/(len(rows)**2)
    for i in groups[0]: a[i*n+groups[1]]-=1.0/(len(groups[0])*len(groups[1]))
    for i in groups[1]: a[i*n+groups[0]]-=1.0/(len(groups[0])*len(groups[1]))
    half("centroid_distance",a,float(a[:n*n]@g0.reshape(-1)))
    index=0
    for i,fullrank in enumerate(rankings):
        top=fullrank[:20]; pairs=[(top[r],top[r+1],"internal") for r in range(19)]
        pairs += [(top[19],j,"boundary") for j in fullrank[20:]]
        for aa,bb,kind in pairs:
            direction=np.zeros(size); direction[i*n+aa]=1; direction[i*n+bb]=-1
            rhs=-1e-7 if str(ids[aa])<str(ids[bb]) else float(np.nextafter(1e-7,math.inf))
            half("rank_{}_{:04d}".format(kind,index),direction,rhs); index+=1
    return sets


def _replay_product_dykstra(row):
    sets=_independent_product_sets(row)
    if [name for name,_ in sets]!=row.get("set_order"): return False,{"reason":"set_order_mismatch"}
    x=np.asarray(row["input"],dtype=np.float64); corrections=[np.zeros_like(x) for _ in sets]
    trace=row["persistent_correction_trace"]
    for cycle_record in trace:
        before=x.copy(); norms=[]
        for index,(_,project) in enumerate(sets):
            y=x+corrections[index]; new=project(y); corrections[index]=y-new; x=new
            norms.append(float(np.linalg.norm(corrections[index])))
        if np.linalg.norm(before-np.asarray(cycle_record["before_vector"]))>1e-9 or \
                np.linalg.norm(x-np.asarray(cycle_record["after_vector"]))>1e-8 or \
                np.max(np.abs(np.asarray(norms)-np.asarray(cycle_record["correction_norms"])))>1e-8:
            return False,{"reason":"projector_transition_or_correction_mismatch",
                          "cycle":cycle_record["cycle"]}
    return np.linalg.norm(x-np.asarray(row["projected"]))<=1e-8,{"cycles_replayed":len(trace)}


def _independent_orientation_system(gram,ids,tolerance=1e-7,limit=34):
    n=len(ids); rankings=[_rank(gram,ids,i,tolerance) for i in range(n)]
    pairs=[(i,j) for i in range(n) for j in range(i+1,n)]; pidx={p:k for k,p in enumerate(pairs)}
    desc=[]; normals=[]
    for i,ranking in enumerate(rankings):
        boundary=ranking[19]; value=float(gram[i,boundary])
        for outsider in ranking[20:]:
            if abs(value-float(gram[i,outsider]))<=tolerance:
                d=(str(ids[i]),str(ids[boundary]),str(ids[outsider]))
                if d in desc: continue
                normal=np.zeros(len(pairs))
                for j,sign in ((boundary,1.0),(outsider,-1.0)):
                    normal[pidx[tuple(sorted((i,j)))]]+=sign
                desc.append(d); normals.append(normal)
    if not normals: return {"rank":0,"descriptors":[],"basis":[],"coeff":[],"assignments":[[]],"overflow":False}
    matrix=np.stack(normals); basis=[]; current=np.zeros((0,matrix.shape[1]))
    for i,row in enumerate(matrix):
        if np.linalg.matrix_rank(np.vstack([current,row]),tol=1e-12)>len(basis):
            basis.append(i); current=np.vstack([current,row])
    coeff=np.linalg.lstsq(matrix[basis].T,matrix.T,rcond=1e-12)[0].T
    assignments=[]; overflow=False
    def feasible(signs):
        result=linprog(np.zeros(matrix.shape[1]),A_ub=-np.asarray(signs)[:,None]*matrix[:len(signs)],
                       b_ub=-np.ones(len(signs)),bounds=[(-10,10)]*matrix.shape[1],method="highs")
        return result.success
    def dfs(signs):
        nonlocal overflow
        if overflow:return
        if len(signs)==len(normals):
            assignments.append(list(signs)); overflow=len(assignments)>=limit; return
        for sign in (-1,1):
            if feasible(signs+[sign]):dfs(signs+[sign])
    dfs([])
    return {"rank":len(basis),"descriptors":desc,"basis":basis,"coeff":coeff.tolist(),
            "assignments":assignments,"overflow":overflow}


def _independent_cell_from_assignment(base,descriptors,assignment,ids):
    idrow={str(v):i for i,v in enumerate(ids)}; cells=[list(r) for r in base]; grouped={}
    for sign,(query,a,b) in zip(assignment,descriptors): grouped.setdefault(query,[]).append((int(sign),idrow[a],idrow[b]))
    for query,constraints in grouped.items():
        qi=idrow[query]; baseline=list(base[qi]); parent={v:v for v in baseline}
        def find(v):
            while parent[v]!=v:
                parent[v]=parent[parent[v]]; v=parent[v]
            return v
        def union(a,b):
            ra,rb=find(a),find(b)
            if ra!=rb:parent[rb]=ra
        for _,a,b in constraints:union(a,b)
        edges={v:set() for v in baseline}; indegree={v:0 for v in baseline}; pos={v:i for i,v in enumerate(baseline)}
        for a,b in zip(baseline[:-1],baseline[1:]):
            if find(a)!=find(b): edges[a].add(b)
        for sign,a,b in constraints:
            u,v=(a,b) if sign>0 else (b,a); edges[u].add(v)
        for u in edges:
            for v in edges[u]: indegree[v]+=1
        avail=[v for v in baseline if indegree[v]==0]; order=[]
        while avail:
            avail.sort(key=lambda v:(pos[v],str(ids[v]))); u=avail.pop(0); order.append(u)
            for v in sorted(edges[u],key=lambda v:(pos[v],str(ids[v]))):
                indegree[v]-=1
                if indegree[v]==0:avail.append(v)
        if len(order)!=len(baseline): return None
        for sign,a,b in constraints:
            if (order.index(a)<order.index(b))!=(sign>0):return None
        cells[qi]=order
    return cells


def verify_dykstra(rows, expected):
    details=[]; ok=True
    for row in rows:
        name=row["case"]; status=row["status"]; exp=expected[name]
        if row.get("kind")=="contradictory_scalar":
            x=np.asarray(row["input"],dtype=np.float64); corr=[np.zeros(1),np.zeros(1)]; replay=True
            for record in row["persistent_correction_trace"]:
                before=x.copy(); norms=[]
                for index,(rhs,sign) in enumerate(((1.0,1.0),(0.0,-1.0))):
                    y=x+corr[index]; value=sign*float(y[0]); new=y.copy()
                    if value<sign*rhs: new[0]+=(sign*rhs-value)*sign
                    corr[index]=y-new; x=new; norms.append(float(np.linalg.norm(corr[index])))
                replay &= np.linalg.norm(before-np.asarray(record["before_vector"]))<=1e-12 and np.linalg.norm(x-np.asarray(record["after_vector"]))<=1e-12 and np.max(np.abs(np.asarray(norms)-np.asarray(record["correction_norms"])))<=1e-12
            violation=max(0.0,1.0-float(x[0]),float(x[0]))
            passed=status==exp=="BOUNDED_SEARCH_FEASIBLE" and int(row["cycles"])==int(row["max_cycles"]) and violation>1e-6 and replay
            details.append({"case":name,"status":status,"independent_set_violation":violation,"independent_replay":bool(replay)})
            ok &= passed; continue
        values,rank_count,cell_stable=_product_violations(row)
        set_violation=max(values.values())
        trace=row.get("persistent_correction_trace",[])
        trace_gate=len(trace)==int(row["cycles"]) and len(trace)>0
        for index,cycle in enumerate(trace):
            before=cycle.get("before_vector"); after=cycle.get("after_vector")
            trace_gate &= hobj(before)==cycle.get("before_sha256") and hobj(after)==cycle.get("after_sha256")
            trace_gate &= len(cycle.get("correction_norms",[]))==int(row["set_count"])
            if index>0: trace_gate &= trace[index-1].get("after_sha256")==cycle.get("before_sha256")
        trace_gate &= trace[-1].get("after_sha256")==hobj(row["projected"])
        if len(trace)>1:
            trace_gate &= any(trace[0]["correction_norms"][j]>0 and trace[1]["correction_norms"][j]>0
                              for j in range(int(row["set_count"])))
        passed=status==exp
        if status=="LOCAL_STATIONARY_CERTIFIED":
            passed &= set_violation<=1e-6 and float(row["relative_iterate_change"])<=1e-7 and cell_stable
        elif status=="BOUNDED_SEARCH_FEASIBLE":
            passed &= int(row["cycles"])==int(row["max_cycles"])
        else: passed=False
        replay_gate,replay_detail=_replay_product_dykstra(row)
        search_gate=True; search_detail={}
        if status=="LOCAL_STATIONARY_CERTIFIED":
            f=row["fixture"]; n=int(f["n"])
            reference=np.asarray(row.get("orientation_reference_gram"),dtype=np.float64)
            system=_independent_orientation_system(reference,f["ids"])
            ledger=row.get("adjacent_cell_ledger",[])
            search_gate=(row.get("search_reason")=="all_adjacent_checked" and
                         int(row.get("adjacent_cells_checked",-1))==int(row.get("adjacent_cells_total",-2))==len(system["assignments"]) and
                         int(row.get("independent_orientations",-1))==system["rank"] and
                         row.get("orientation_basis_indices",[])==system["basis"] and
                         np.linalg.norm(np.asarray(row.get("orientation_dependency_coefficients",[]))-
                                        np.asarray(system["coeff"]))<=1e-8 and
                         len(ledger)==len(system["assignments"]) and int(row.get("pivots",99))<=32)
            objectives=[]
            for expected_assignment,cell in zip(system["assignments"],ledger):
                base=[_rank(reference,f["ids"],i) for i in range(n)]
                rebuilt=_independent_cell_from_assignment(base,system["descriptors"],expected_assignment,f["ids"])
                temp=dict(row); temp["projected"]=cell["result"]; temp["persistent_correction_trace"]=cell["trace"]
                temp["fixture"]=dict(row["fixture"]); temp["fixture"]["selected_cell_rankings"]=cell["cell_rankings"]
                cell_values,_,cell_stable_ind=_product_violations(temp); cell_replay,_=_replay_product_dykstra(temp)
                objective=float(np.dot(np.asarray(cell["result"])-np.asarray(row["input"]),
                                       np.asarray(cell["result"])-np.asarray(row["input"])))
                search_gate &= (rebuilt is not None and cell["cell_rankings"]==rebuilt and
                                cell["assignment"]==expected_assignment and cell["status"]=="LOCAL_STATIONARY_CERTIFIED" and
                                max(cell_values.values())<=1e-6 and cell_stable_ind and cell_replay and
                                abs(objective-float(cell["objective"]))<=1e-8)
                objectives.append(objective)
            if objectives:
                best=ledger[int(np.argmin(objectives))]["result"]
                search_gate &= np.linalg.norm(np.asarray(best)-np.asarray(row["projected"]))<=1e-8
            search_detail={"rank":system["rank"],"compatible_cells":len(system["assignments"]),
                           "ledger_cells":len(ledger)}
        passed &= trace_gate and replay_gate and search_gate and rank_count==528 and int(row["set_count"])==589
        details.append({"case":name,"status":status,"expected":exp,
                        "independent_set_violation":set_violation,
                        "violations":values,"rank_halfspace_count":rank_count,
                        "rank_cell_stable":cell_stable,"trace_gate":bool(trace_gate),
                        "independent_replay":bool(replay_gate),"replay_detail":replay_detail})
        details[-1]["independent_rank_search"]=bool(search_gate); details[-1]["rank_search_detail"]=search_detail
        ok &= passed
    return bool(ok),details


def _rank(gram,ids,i,tolerance=1e-7):
    candidates=[j for j in range(len(ids)) if j!=i]
    remaining=sorted(candidates,key=lambda j:-float(gram[i,j])); ordered=[]
    while remaining:
        anchor=float(gram[i,remaining[0]])
        group=[j for j in remaining if anchor-float(gram[i,j])<=tolerance]
        group.sort(key=lambda j:str(ids[j])); ordered.extend(group)
        selected=set(group); remaining=[j for j in remaining if j not in selected]
    return ordered


def verify_rank(rows):
    details=[]; ok=True
    exact=next((r for r in rows if r["case"]=="exact_top20"),None)
    if exact is None: return False,[{"reason":"missing_exact_top20"}]
    gram=np.asarray(exact["gram"],dtype=np.float64); ids=exact["ids"]; labels=np.asarray(exact["labels"])
    ok &= len(exact["ledger"])==24 and len(ids)==24 and len(set(ids))==24
    for i,record in enumerate(exact["ledger"]):
        ranking=_rank(gram,ids,i)[:20]
        got=[int(x["row"]) for x in record["neighbors"]]
        vote=sum((21-r)*float(gram[i,j])*(2*int(labels[j])-1) for r,j in enumerate(ranking,1))
        passed=ranking==got and abs(vote-float(record["weighted_signed_vote"]))<=1e-12 and int(vote>=0)==int(record["prediction"])
        passed &= (record["query_id"]==str(ids[i]) and int(record["query_row"])==i and
                   int(record["query_label"])==int(labels[i]) and len(record["neighbors"])==20 and
                   all(int(neighbor["row"])!=i for neighbor in record["neighbors"]))
        for rank,(neighbor,j) in enumerate(zip(record["neighbors"],ranking),1):
            passed &= (neighbor["id"]==str(ids[j]) and int(neighbor["rank"])==rank and
                       int(neighbor["weight"])==21-rank and int(neighbor["label"])==int(labels[j]) and
                       abs(float(neighbor["cosine"])-float(gram[i,j]))<=1e-12)
        ok &= passed
    orientation=next((r for r in rows if r["case"]=="orientation_over_budget"),None)
    if orientation is None: ok=False
    else:
        og=np.asarray(orientation["gram"],dtype=np.float64); oids=orientation["ids"]
        system=_independent_orientation_system(og,oids); count=system["rank"]
        budget=int(orientation["budget"])
        ok &= count==int(orientation["independent_orientations"]) and count>budget and orientation["status"]=="REMOVE"
        ok &= orientation.get("orientation_basis_indices")==system["basis"] and \
            np.linalg.norm(np.asarray(orientation.get("orientation_dependency_coefficients"))-np.asarray(system["coeff"]))<=1e-8
        ok &= orientation["remove_replay_sha256"]==orientation["direct_remove_sha256"]
        parity,rd,dd=independent_rollback_replay(int(orientation["rollback_seed"])); ok &= parity and rd==dd==orientation["remove_replay_sha256"]
    pivot=next((r for r in rows if r["case"]=="pivot_over_budget"),None)
    if pivot is None: ok=False
    else:
        hashes=pivot["cell_hashes"]; count=sum(hashes[i]!=hashes[i-1] for i in range(1,len(hashes)))
        budget=int(pivot["budget"]); ok &= count==int(pivot["pivots"]) and count>budget and pivot["status"]=="REMOVE"
        ok &= pivot["remove_replay_sha256"]==pivot["direct_remove_sha256"]
        parity,rd,dd=independent_rollback_replay(int(pivot["rollback_seed"])); ok &= parity and rd==dd==pivot["remove_replay_sha256"]
    for name,reason_token in (("unresolved_tie_map","unresolved_tie_map"),
                              ("incomplete_adjacent_enumeration","not_converged")):
        item=next((r for r in rows if r["case"]==name),None)
        if item is None: ok=False
        else:
            ok &= item["status"]=="REMOVE" and item["remove_replay_sha256"]==item["direct_remove_sha256"]
            parity,rd,dd=independent_rollback_replay(int(item["rollback_seed"])); ok &= parity and rd==dd==item["remove_replay_sha256"]
            if name=="unresolved_tie_map": ok &= reason_token in item.get("controller_reason","")
            else: ok &= item.get("controller_status")!="LOCAL_STATIONARY_CERTIFIED"
    tie=next((r for r in rows if r["case"]=="simultaneous_ties"),None)
    if tie is None: ok=False
    else:
        tg=np.asarray(tie["gram"],dtype=np.float64); tr=_rank(tg,tie["ids"],0)[:20]
        ok &= tr==[int(x) for x in tie["ranking"]] and tie["status"]=="PASS"
        ok &= all(bool(cell["pass"]) and bool(cell["tied"])==(float(cell["offset"])<=1e-7)
                  for cell in tie.get("boundary_tolerance_checks",[]))
    details.append({"exact_rows":len(exact["ledger"]),"parity":bool(ok)})
    return bool(ok),details


def verify_farkas(rows):
    ok=True; details=[]
    for row in rows:
        a=np.asarray(row["columns"],dtype=np.float64); d=np.asarray(row["target"],dtype=np.float64)
        # Independent dual solve: maximize w^T d subject to A^T w <= 0,
        # ||w||_2 <= 1.  This does not call NNLS or derive w from its residual.
        constraints=[{"type":"ineq","fun":lambda w,j=j: -float(a[:,j]@w)}
                     for j in range(a.shape[1])]
        constraints.append({"type":"ineq","fun":lambda w:1.0-float(w@w)})
        starts=[np.zeros_like(d),d/max(np.linalg.norm(d),1e-15)]
        best=None
        for start in starts:
            result=minimize(lambda w:-float(w@d),start,method="SLSQP",
                            constraints=constraints,
                            options={"ftol":1e-13,"maxiter":4000})
            if result.success and (best is None or float(result.x@d)>float(best@d)):
                best=result.x
        if best is None:
            return False,details+[{"case":row["case"],"reason":"dual_solver_failed"}]
        witness=best/max(1.0,float(np.linalg.norm(best)))
        dual_value=max(0.0,float(witness@d))
        separation=dual_value/max(float(np.linalg.norm(d)),1e-15)
        max_inner=float(np.max(a.T@witness));
        primal_norm=float(np.linalg.norm(np.asarray(row["residual"],dtype=np.float64)))
        gap=abs(primal_norm-dual_value)
        if row["expected_membership"]=="out": passed=separation>=0.25 and max_inner<=1e-8 and gap<=1e-5
        else: passed=separation<=1e-8
        passed &= abs(separation-float(row["relative_separation"]))<=1e-8
        details.append({"case":row["case"],"separation":separation,"max_witness_inner":max_inner,"gap":gap})
        ok &= passed
    return bool(ok),details


def _independent_factor(gram):
    gram=0.5*(gram+gram.T); eigval,eigvec=np.linalg.eigh(gram)
    if float(eigval.min())<-1e-7: raise ValueError("negative")
    order=np.argsort(-eigval,kind="mergesort"); eigval=eigval[order]; eigvec=eigvec[:,order]
    basis=np.zeros_like(eigvec); start=0; n=len(eigval)
    while start<n:
        end=start+1; scale=max(1.0,abs(float(eigval[start])))
        while end<n and abs(float(eigval[end]-eigval[start]))<=1e-10*scale: end+=1
        projector=eigvec[:,start:end]@eigvec[:,start:end].T; vectors=[]
        for axis in range(n):
            v=projector[:,axis].copy()
            for q in vectors: v-=q*float(q@v)
            norm=float(np.linalg.norm(v))
            if norm>1e-10:
                v/=norm; pivot=int(np.argmax(np.abs(v))); v=-v if v[pivot]<0 else v; vectors.append(v)
            if len(vectors)==end-start: break
        basis[:,start:end]=np.column_stack(vectors); start=end
    return basis*np.sqrt(np.maximum(eigval,0))[None,:]


def verify_factor(rows):
    ok=True; details=[]
    repeated=next((r for r in rows if r["case"]=="repeated_and_null"),None)
    negative=next((r for r in rows if r["case"]=="negative_reject"),None)
    if repeated is None or negative is None: return False,[{"reason":"missing_factor_case"}]
    gram=np.asarray(repeated["gram"],dtype=np.float64); factor=np.asarray(repeated["factor"],dtype=np.float64)
    aligned=np.asarray(repeated["aligned_factor"],dtype=np.float64); rotation=np.asarray(repeated["rotation"],dtype=np.float64)
    independent=_independent_factor(gram); reference=np.asarray(repeated["reference"],dtype=np.float64)
    u,_,vt=np.linalg.svd(independent.T@reference,full_matrices=False); independent_aligned=independent@(u@vt)
    errors={"factor_reconstruction":float(np.linalg.norm(factor@factor.T-gram)),
            "aligned_reconstruction":float(np.linalg.norm(aligned@aligned.T-gram)),
            "rotation_row_orthogonality":float(np.linalg.norm(rotation@rotation.T-np.eye(rotation.shape[0]))),
            "row_reconstruction":float(np.max(np.linalg.norm(aligned@aligned.T-gram,axis=1))),
            "deterministic_repeat":float(np.linalg.norm(independent-factor)),
            "procrustes":float(np.linalg.norm(independent_aligned-reference))}
    ok &= all(v<=1e-6 for v in errors.values()) and repeated["status"]=="PASS"
    reject_matrix=np.diag([1.0,0.5,float(negative["minimum_eigenvalue"])]); rejected=False
    try: _independent_factor(reject_matrix)
    except ValueError: rejected=True
    ok &= negative["status"]=="PASS" and rejected
    ok &= int(round(np.sum(np.linalg.eigvalsh(gram)<=1e-10)))==2
    details.append(errors); return bool(ok),details


def _state_equal(a,b):
    import torch
    if torch.is_tensor(a) and torch.is_tensor(b): return torch.equal(a,b)
    if isinstance(a,np.ndarray) and isinstance(b,np.ndarray): return np.array_equal(a,b)
    if isinstance(a,dict) and isinstance(b,dict):
        return set(a)==set(b) and all(_state_equal(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)) and isinstance(b,(list,tuple)):
        return len(a)==len(b) and all(_state_equal(x,y) for x,y in zip(a,b))
    return a==b


def _json_state(obj):
    import torch
    if torch.is_tensor(obj): return {"dtype":str(obj.dtype),"shape":list(obj.shape),
                                     "value":obj.detach().cpu().numpy().tolist()}
    if isinstance(obj,np.ndarray): return {"dtype":str(obj.dtype),"shape":list(obj.shape),"value":obj.tolist()}
    if isinstance(obj,dict): return {str(k):_json_state(v) for k,v in sorted(obj.items(),key=lambda x:str(x[0]))}
    if isinstance(obj,(list,tuple)): return [_json_state(v) for v in obj]
    if isinstance(obj,(str,int,float,bool)) or obj is None: return obj
    return repr(obj)


def _full_state_digest(state):
    rows=[]
    for key,value in sorted(state["model"].items()):
        rows.append((key,hobj(value.detach().cpu().numpy().tolist())))
    rows += [("optimizer",hobj(_json_state(state["opt"]))),
             ("scheduler",hobj(_json_state(state["sched"]))),
             ("scaler",hobj(_json_state(state["scaler"]))),
             ("torch_rng",hobj(state["torch"].tolist())),
             ("numpy_rng",hobj(_json_state(state["numpy"]))),
             ("python_rng",hobj(_json_state(state["python"]))),
             ("cursor",hobj(state["cursor"]))]
    return hobj(rows)


def independent_rollback_replay(seed):
    import torch
    torch.use_deterministic_algorithms(True)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    def build():
        m=torch.nn.Linear(4,4,bias=True); o=torch.optim.AdamW(m.parameters(),lr=1e-3)
        s=torch.optim.lr_scheduler.StepLR(o,step_size=1,gamma=0.9)
        c=torch.amp.GradScaler("cpu",enabled=True,init_scale=128.0,growth_interval=1); return m,o,s,c
    model,opt,sched,scaler=build(); x=torch.arange(32,dtype=torch.float32).reshape(8,4)/31.0
    y=torch.flip(x,dims=[0]); cursor={"sampler_cursor":0,"epoch_cursor":0}
    snap={"model":copy.deepcopy(model.state_dict()),"opt":copy.deepcopy(opt.state_dict()),
          "sched":copy.deepcopy(sched.state_dict()),"scaler":copy.deepcopy(scaler.state_dict()),
          "torch":torch.get_rng_state().clone(),"numpy":copy.deepcopy(np.random.get_state()),
          "python":random.getstate(),"cursor":copy.deepcopy(cursor)}
    order=torch.randperm(len(x)); scale=1.0+float(np.random.random())+random.random()
    opt.zero_grad(); loss=torch.square(model(x[order])-y[order]).mean()*scale
    scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
    cursor["sampler_cursor"]+=len(x); cursor["epoch_cursor"]+=1
    model.load_state_dict(snap["model"]); opt.load_state_dict(snap["opt"])
    sched.load_state_dict(snap["sched"]); scaler.load_state_dict(snap["scaler"])
    torch.set_rng_state(snap["torch"]); np.random.set_state(snap["numpy"]); random.setstate(snap["python"])
    cursor=copy.deepcopy(snap["cursor"])
    def remove(m,o,s,c,cur):
        order=torch.randperm(len(x)); ns=1.0+1e-3*float(np.random.random())
        if random.random()<0.5: order=torch.flip(order,dims=[0])
        for start in range(0,len(x),4):
            batch=order[start:start+4]; o.zero_grad(); l=torch.square(m(x[batch])-x[batch]).mean()*ns
            c.scale(l).backward(); c.step(o); c.update(); cur["sampler_cursor"]+=len(batch)
        s.step(); cur["epoch_cursor"]+=1
    remove(model,opt,sched,scaler,cursor)
    rollback={"model":copy.deepcopy(model.state_dict()),"opt":copy.deepcopy(opt.state_dict()),
              "sched":copy.deepcopy(sched.state_dict()),"scaler":copy.deepcopy(scaler.state_dict()),
              "torch":torch.get_rng_state().clone(),"numpy":copy.deepcopy(np.random.get_state()),
              "python":random.getstate(),"cursor":copy.deepcopy(cursor)}
    direct,dopt,dsched,dscaler=build(); direct.load_state_dict(snap["model"]); dopt.load_state_dict(snap["opt"])
    dsched.load_state_dict(snap["sched"]); dscaler.load_state_dict(snap["scaler"])
    torch.set_rng_state(snap["torch"]); np.random.set_state(snap["numpy"]); random.setstate(snap["python"])
    dc=copy.deepcopy(snap["cursor"]); remove(direct,dopt,dsched,dscaler,dc)
    direct_state={"model":direct.state_dict(),"opt":dopt.state_dict(),"sched":dsched.state_dict(),
                  "scaler":dscaler.state_dict(),"torch":torch.get_rng_state().clone(),
                  "numpy":copy.deepcopy(np.random.get_state()),"python":random.getstate(),"cursor":dc}
    return _state_equal(rollback,direct_state),_full_state_digest(rollback),_full_state_digest(direct_state)


def verify_manifest_files(manifest, root, cfg, expected_run, expected_stage,
                          predecessor_hashes=None):
    if not verify_payload(manifest): return False
    impl,_=current_implementation(cfg)
    if manifest.get("run_id")!=expected_run or manifest.get("stage")!=expected_stage:
        return False
    if manifest.get("config_canonical_sha256")!=hobj(cfg) or \
            manifest.get("implementation_sha256")!=impl or \
            manifest.get("independent_verifier_sha256")!=hfile(Path(__file__)):
        return False
    for key,value in (predecessor_hashes or {}).items():
        if manifest.get(key)!=value: return False
    for row in manifest.get("output_files",[]):
        path=root/row["path"]
        if not path.exists() or hfile(path)!=row["sha256"] or \
                not path.with_name(path.name+".publish.lock").exists(): return False
    if manifest.get("access_ledger_sha256")!=hobj(manifest.get("access_ledger",[])):
        return False
    return all(int(manifest.get(key,-1))==0 for key in ZERO_COUNTER_KEYS) and \
        manifest.get("only_gold_supervision")=="parent_video_binary_label" and \
        manifest.get("segment_gold_exists") is False and manifest.get("segment_gold_used") is False


def verify_synthetic(cfg, artifacts):
    synth=artifacts/"g0/synthetic"; manifest=read_json(synth/"manifest.json")
    freeze=read_json(artifacts/"CONFIG_FREEZE.json"); audit=read_json(artifacts/"g0/code_audit/audit.json")
    exact_files={"cases.jsonl","projectors.jsonl","dykstra.jsonl","rank_cells.jsonl",
                 "exact_vote.jsonl","farkas.jsonl","factor.jsonl",
                 "manifest.json"}
    observed={p.name for p in synth.iterdir() if p.is_file() and not p.name.endswith(".publish.lock")}
    file_set_gate=observed==exact_files
    cases=read_jsonl(synth/"cases.jsonl")
    projectors=read_jsonl(synth/"projectors.jsonl"); dykstra=read_jsonl(synth/"dykstra.jsonl")
    rank=read_jsonl(synth/"rank_cells.jsonl"); farkas=read_jsonl(synth/"farkas.jsonl")
    factor=read_jsonl(synth/"factor.jsonl"); rollback=[cases[0]["rollback"]] if len(cases)==1 and "rollback" in cases[0] else []
    exact_vote=read_jsonl(synth/"exact_vote.jsonl")
    expected=cfg["synthetic"]["expected_cases"]
    case_sets={"projectors":{r.get("case") for r in projectors},
               "dykstra":{r.get("case") for r in dykstra},
               "rank_cells":{r.get("case") for r in rank},
               "farkas":{r.get("case") for r in farkas},
               "factor":{r.get("case") for r in factor},
               "rollback":{r.get("case") for r in rollback}}
    cardinality_gate=len(cases)==1 and cases[0].get("expected")==expected
    cardinality_gate &= all(case_sets[group]==set(expected[group]) and
                            len(rows)==len(expected[group])
                            for group,rows in [("projectors",projectors),("dykstra",dykstra),
                                               ("rank_cells",rank),("farkas",farkas),
                                               ("factor",factor),("rollback",rollback)])
    cardinality_gate &= len(exact_vote)==1 and exact_vote[0].get("case")=="exact_top20"
    cardinality_gate &= len(rank)>0 and hobj(exact_vote[0])==hobj(next(r for r in rank if r.get("case")=="exact_top20"))
    gates={"exact_file_set":file_set_gate,"exact_case_ledger":bool(cardinality_gate)}; details={}
    gates["manifest"]=verify_manifest_files(
        manifest,ROOT,cfg,_lineage_run_id(cfg, "synthetic"),"G0_SYNTHETIC",
        {"freeze_payload_sha256":freeze.get("payload_sha256"),
         "code_audit_payload_sha256":audit.get("payload_sha256")}) and \
        verify_payload(freeze) and verify_payload(audit) and manifest.get("status")=="PASS"
    artifacts_rel = _relpath(artifacts)
    expected_access={(_config_rel(cfg),"frozen_config","config_load",None),
                     (artifacts_rel + "/CONFIG_FREEZE.json","freeze","predecessor_verify",None),
                     (artifacts_rel + "/g0/code_audit/audit.json","code_audit","predecessor_verify",None)}
    for item in freeze.get("input_files",[]):
        if "member_sha256" in item:
            expected_access |= {(item["path"],"freeze_input","frozen_input_rehash",member)
                                for member in item["member_sha256"]}
        else:
            expected_access.add((item["path"],"freeze_input","frozen_input_rehash",None))
    gates["manifest"] &= {(r.get("path"),r.get("scope"),r.get("purpose"),r.get("member"))
                          for r in manifest.get("access_ledger",[])}==expected_access and \
        len(manifest.get("access_ledger",[]))==len(expected_access)
    gates["projectors"],details["projectors"]=verify_projectors(projectors)
    gates["dykstra"],details["dykstra"]=verify_dykstra(dykstra,cfg["synthetic"]["expected_cases"]["dykstra"])
    gates["rank_exact_vote"],details["rank_exact_vote"]=verify_rank(rank)
    gates["farkas"],details["farkas"]=verify_farkas(farkas)
    gates["factor"],details["factor"]=verify_factor(factor)
    gates["rollback"]=len(rollback)==1 and rollback[0]["status"]=="PASS" and \
        rollback[0]["rollback_replay_sha256"]==rollback[0]["direct_remove_sha256"]
    if len(rollback)==1:
        parity,rollback_digest,direct_digest=independent_rollback_replay(int(rollback[0]["seed"]))
        gates["rollback"] &= rollback[0].get("fixture_version")=="linear4_adamw_stepLR_two_random_batches_v1" and parity and \
            rollback_digest==direct_digest==rollback[0]["rollback_replay_sha256"]
    gates["finite"]=recursively_finite({"cases":cases,"projectors":projectors,
        "dykstra":dykstra,"rank":rank,"exact_vote":exact_vote,"farkas":farkas,
        "factor":factor,"rollback":rollback,"manifest":manifest,"details":details})
    return all(gates.values()),gates,details


def _load_real_bank(cfg):
    path=_resolve_cfg_path(cfg,"bank");allowed=set(cfg["sealed_real_fixture"]["allowed_bank_members"])
    forbidden=set(cfg["sealed_real_fixture"]["forbidden_bank_members"]);out={};opened=[]
    with zipfile.ZipFile(path,"r") as archive:
        members={Path(name).stem:name for name in archive.namelist() if name.endswith(".npy")}
        missing=allowed-set(members)
        if missing: raise RuntimeError("missing real bank members {}".format(sorted(missing)))
        for stem in sorted(allowed):
            if stem in forbidden:raise RuntimeError("forbidden bank member")
            VERIFIER_ACCESS_LEDGER.append({"path":_relpath(path),
                                           "purpose":"independent_npz_member_read",
                                           "scope":"outer_held_ids_sentinel" if stem=="query_ids" else "outer_train_bank",
                                           "member":stem})
            with archive.open(members[stem],"r") as handle:
                payload=handle.read()
            digest=hashlib.sha256(payload).hexdigest()
            if digest!=cfg["sealed_real_fixture"]["bank_member_sha256"][stem]:
                raise RuntimeError("bank member hash mismatch {}".format(stem))
            VERIFIER_ACCESS_LEDGER[-1]["sha256"]=digest
            out[stem]=np.load(io.BytesIO(payload),allow_pickle=False)
            opened.append(stem)
    if set(opened)&forbidden: raise RuntimeError("forbidden real bank member opened")
    return out,opened


def _hash_npz_allowed_members(path, cfg):
    path=canonical_root_path(path)[0]; fixture=cfg["sealed_real_fixture"]
    allowed=set(fixture["allowed_bank_members"]); forbidden=set(fixture["forbidden_bank_members"])
    hashes={}
    with zipfile.ZipFile(path,"r") as archive:
        members={Path(name).stem:name for name in archive.namelist() if name.endswith(".npy")}
        missing=allowed-set(members)
        if missing: raise RuntimeError("missing bank members {}".format(sorted(missing)))
        for stem in sorted(allowed):
            if stem in forbidden: raise RuntimeError("forbidden bank member hash")
            VERIFIER_ACCESS_LEDGER.append({"path":_relpath(path),
                                           "purpose":"independent_npz_member_hash",
                                           "scope":"bank_allowed_members",
                                           "member":stem})
            with archive.open(members[stem],"r") as handle:
                hashes[stem]=hashlib.sha256(handle.read()).hexdigest()
    return hashes


def _real_constraint_audit(g,xi,g0,labels,ids,ell,caps,cell_rankings):
    n=len(ids);labels=np.asarray(labels);top=np.asarray([row[:20] for row in cell_rankings],dtype=np.int64)
    signs=2*labels-1;weights=np.arange(20,0,-1)[None,:]
    coeff=signs[:,None]*weights*signs[top]/210.0
    margins=np.sum(coeff*g[np.arange(n)[:,None],top],axis=1)
    canonical=[_rank(g0,ids,i) for i in range(n)];ctop=np.asarray([r[:20] for r in canonical])
    ccoeff=signs[:,None]*weights*signs[ctop]/210.0
    baseline=np.sum(ccoeff*g0[np.arange(n)[:,None],ctop],axis=1)
    groups=[np.flatnonzero(labels==v) for v in (0,1)];off=g[~np.eye(n,dtype=bool)]
    values={"symmetry":float(np.max(np.abs(g-g.T))),"diagonal":float(np.max(np.abs(np.diag(g)-1))),
            "psd":max(0.0,-float(np.linalg.eigvalsh(0.5*(g+g.T)).min())),
            "box":max(0.0,float(off.max()-0.9999),float(-1-off.min())),
            "row_trust":max(max(0.0,float(np.linalg.norm(np.delete(g[i]-g0[i],i))-0.05*math.sqrt(n-1))) for i in range(n)),
            "class_mean_trust":max(max(0.0,float(np.linalg.norm((g[r]-g0[r]).mean(axis=0))-0.02*math.sqrt(n))) for r in groups),
            "slack":max(max(0.0,float(-xi[r].min()),float(xi[r].sum()-caps[k])) for k,r in enumerate(groups)),
            "vote":max(0.0,float(np.max(ell-margins-xi))),
            "class_margin":max(max(0.0,float(baseline[r].mean()-margins[r].mean())) for r in groups),
            "global_margin":max(0.0,float(baseline.mean()-margins.mean()))}
    def centroid(gram):
        return float(gram[np.ix_(groups[0],groups[0])].mean()+gram[np.ix_(groups[1],groups[1])].mean()-
                     gram[np.ix_(groups[0],groups[1])].mean()-gram[np.ix_(groups[1],groups[0])].mean())
    values["centroid"]=max(0.0,centroid(g0)-centroid(g));rank_v=0.0;count=0
    for i,full in enumerate(cell_rankings):
        pairs=[(full[r],full[r+1]) for r in range(19)]+[(full[19],j) for j in full[20:]]
        for a,b in pairs:
            rhs=-1e-7 if str(ids[a])<str(ids[b]) else float(np.nextafter(1e-7,math.inf))
            rank_v=max(rank_v,rhs-float(g[i,a]-g[i,b]));count+=1
    values["rank"]=max(0.0,rank_v)
    return values,margins,baseline,count


def _independent_exact_vote(gram, labels, ids):
    labels=np.asarray(labels,dtype=np.int64); rows=[]
    for i in range(len(ids)):
        ranking=_rank(gram,ids,i)[:20]; vote=0.0; neighbors=[]
        for rank,j in enumerate(ranking,1):
            weight=21-rank; sim=float(gram[i,j])
            vote+=weight*sim*(2*int(labels[j])-1)
            neighbors.append({"rank":rank,"id":str(ids[j]),"row":int(j),
                              "label":int(labels[j]),"cosine":sim,
                              "weight":weight})
        rows.append({"query_id":str(ids[i]),"query_row":i,
                     "query_label":int(labels[i]),"neighbors":neighbors,
                     "weighted_signed_vote":float(vote),
                     "prediction":int(vote>=0.0)})
    return rows


def _canonical_rhs(ids,a,b,tolerance):
    return -tolerance if str(ids[a])<str(ids[b]) else float(np.nextafter(tolerance,math.inf))


def _project_psd_matrix(g):
    ev,vec=np.linalg.eigh(0.5*(g+g.T))
    return (vec*np.maximum(ev,0.0))@vec.T


def _centroid_direction(labels):
    labels=np.asarray(labels,dtype=np.int64); n=len(labels); out=np.zeros((n,n),dtype=np.float64)
    groups=[np.flatnonzero(labels==v) for v in (0,1)]
    for rows in groups:
        out[np.ix_(rows,rows)] += 1.0/(len(rows)**2)
    out[np.ix_(groups[0],groups[1])] -= 1.0/(len(groups[0])*len(groups[1]))
    out[np.ix_(groups[1],groups[0])] -= 1.0/(len(groups[0])*len(groups[1]))
    return out


def _rank_data_v(g0,labels,ids,cfg,rankings_override=None,baseline_margins=None):
    n=len(ids); tol=cfg["solver"]["tie_tolerance"]
    rankings=rankings_override if rankings_override is not None else [_rank(g0,ids,i,tol) for i in range(n)]
    top=np.asarray([row[:20] for row in rankings],dtype=np.int64)
    weights=np.arange(20,0,-1,dtype=np.float64)[None,:]
    signs=2*np.asarray(labels,dtype=np.int64)-1
    coeff=signs[:,None]*weights*signs[top]/210.0
    margins=np.sum(coeff*g0[np.arange(n)[:,None],top],axis=1)
    if baseline_margins is not None: margins=np.asarray(baseline_margins,dtype=np.float64)
    ri=[];ra=[];rb=[];rr=[]
    for i,row in enumerate(rankings):
        for r in range(19):
            a,b=row[r],row[r+1]; ri.append(i);ra.append(a);rb.append(b);rr.append(_canonical_rhs(ids,a,b,tol))
        for outsider in row[20:]:
            a,b=row[19],outsider; ri.append(i);ra.append(a);rb.append(b);rr.append(_canonical_rhs(ids,a,b,tol))
    return {"rankings":rankings,"top":top,"coeff":coeff,"margins":margins,
            "rank_i":np.asarray(ri,dtype=np.int32),"rank_a":np.asarray(ra,dtype=np.int32),
            "rank_b":np.asarray(rb,dtype=np.int32),"rank_rhs":np.asarray(rr,dtype=np.float64)}


def _real_values_v(g,xi,g0,labels,rank,ell,caps,cfg):
    n=len(labels); labels=np.asarray(labels); solver=cfg["solver"]
    margins=np.sum(rank["coeff"]*g[np.arange(n)[:,None],rank["top"]],axis=1)
    groups=[np.flatnonzero(labels==v) for v in (0,1)]
    off=g[~np.eye(n,dtype=bool)]
    centroid=_centroid_direction(labels)
    return {"symmetry":float(np.max(np.abs(g-g.T))),
            "diagonal":float(np.max(np.abs(np.diag(g)-1.0))),
            "psd":max(0.0,-float(np.linalg.eigvalsh(0.5*(g+g.T)).min())),
            "box":max(0.0,float(off.max()-solver["offdiag_upper"]),float(-1.0-off.min())),
            "row_trust":max(max(0.0,float(np.linalg.norm(np.delete(g[i]-g0[i],i))-
                                                solver["row_trust_scale"]*math.sqrt(n-1))) for i in range(n)),
            "class_mean_trust":max(max(0.0,float(np.linalg.norm((g[rows]-g0[rows]).mean(axis=0))-
                                                solver["class_mean_trust_scale"]*math.sqrt(n))) for rows in groups),
            "slack":max(max(0.0,float(-xi[rows].min()),float(xi[rows].sum()-caps[k]))
                        for k,rows in enumerate(groups)),
            "vote":max(0.0,float(np.max(ell-margins-xi))),
            "class_margin":max(max(0.0,float(rank["margins"][rows].mean()-margins[rows].mean()))
                               for rows in groups),
            "global_margin":max(0.0,float(rank["margins"].mean()-margins.mean())),
            "centroid":max(0.0,float(np.sum(centroid*g0)-np.sum(centroid*g))),
            "rank":max(0.0,float(np.max(rank["rank_rhs"]-
                (g[rank["rank_i"],rank["rank_a"]]-g[rank["rank_i"],rank["rank_b"]]))))}


def _independent_real_dykstra(g0,labels,ids,cfg,rankings_override=None,baseline_margins=None):
    n=len(ids); solver=cfg["solver"]
    rank=_rank_data_v(g0,labels,ids,cfg,rankings_override,baseline_margins)
    ell=np.maximum(rank["margins"],1e-4)
    deficits=np.maximum(ell-rank["margins"],0.0)
    groups=[np.flatnonzero(np.asarray(labels)==v) for v in (0,1)]
    caps=[solver["slack_budget_ratio"]*float(deficits[rows].sum()) for rows in groups]
    g=g0.copy(); xi=np.zeros(n,dtype=np.float64)
    dense={name:np.zeros_like(g) for name in ("sym","diag","psd","box")}
    row_corr=np.zeros_like(g); class_corr=[np.zeros(n),np.zeros(n)]
    slack_corr=[np.zeros(len(rows)) for rows in groups]
    vote_corr=np.zeros(n); class_margin_corr=np.zeros(2); global_corr=0.0
    centroid_corr=0.0; rank_corr=np.zeros(len(rank["rank_i"]),dtype=np.float64)
    centroid=_centroid_direction(labels); centroid_rhs=float(np.sum(centroid*g0))
    trace_hashes=[]; relative=float("inf")
    def state_sha():
        return hobj({"g":g.tolist(),"xi":xi.tolist()})
    def correction_state():
        return {
            "dense": {k: hobj(v.tolist()) for k, v in dense.items()},
            "row": hobj(row_corr.tolist()),
            "class_mean": [hobj(v.tolist()) for v in class_corr],
            "slack": [hobj(v.tolist()) for v in slack_corr],
            "vote": hobj(vote_corr.tolist()),
            "class_margin": hobj(class_margin_corr.tolist()),
            "global_margin": hobj(float(global_corr)),
            "centroid": hobj(float(centroid_corr)),
            "rank": hobj(rank_corr.tolist()),
        }
    def correction_norms():
        return {"dense":{k:float(np.linalg.norm(v)) for k,v in dense.items()},
                "row":float(np.linalg.norm(row_corr)),
                "class_mean":[float(np.linalg.norm(v)) for v in class_corr],
                "slack":[float(np.linalg.norm(v)) for v in slack_corr],
                "vote":float(np.linalg.norm(vote_corr)),
                "class_margin":float(np.linalg.norm(class_margin_corr)),
                "global_margin":abs(float(global_corr)),
                "centroid":abs(float(centroid_corr)),
                "rank":float(np.linalg.norm(rank_corr))}
    def add_transition(transitions, name, before):
        transitions.append({"projector": name,
                            "before_sha256": before,
                            "after_sha256": state_sha(),
                            "correction_state_sha256": hobj(correction_state()),
                            "correction_norms": correction_norms()})
    for cycle in range(1,int(solver["max_dykstra_cycles"])+1):
        before_g=g.copy(); before_xi=xi.copy()
        before=hobj({"g":before_g.tolist(),"xi":before_xi.tolist()})
        projector_transitions=[]
        projector_before=state_sha(); y=g+dense["sym"]; new=0.5*(y+y.T); dense["sym"]=y-new; g=new
        add_transition(projector_transitions,"symmetry",projector_before)
        projector_before=state_sha(); y=g+dense["diag"]; new=y.copy(); np.fill_diagonal(new,1.0); dense["diag"]=y-new; g=new
        add_transition(projector_transitions,"diagonal",projector_before)
        projector_before=state_sha(); y=g+dense["psd"]; new=_project_psd_matrix(y); dense["psd"]=y-new; g=new
        add_transition(projector_transitions,"psd",projector_before)
        projector_before=state_sha(); y=g+dense["box"]; new=y.copy(); mask=~np.eye(n,dtype=bool)
        new[mask]=np.clip(new[mask],-1.0,solver["offdiag_upper"]); dense["box"]=y-new; g=new
        add_transition(projector_transitions,"box",projector_before)
        radius=solver["row_trust_scale"]*math.sqrt(n-1)
        projector_before=state_sha()
        for i in range(n):
            y=g[i]+row_corr[i]; cols=np.arange(n)!=i; q=y[cols]-g0[i,cols]
            norm=float(np.linalg.norm(q)); new=y.copy()
            if norm>radius: new[cols]=g0[i,cols]+q*(radius/norm)
            row_corr[i]=y-new; g[i]=new
        add_transition(projector_transitions,"row",projector_before)
        radius=solver["class_mean_trust_scale"]*math.sqrt(n)
        projector_before=state_sha()
        for k,rows in enumerate(groups):
            y=g[rows]+class_corr[k][None,:]; q=(y-g0[rows]).mean(axis=0); norm=float(np.linalg.norm(q))
            adjust=(1.0-radius/norm)*q if norm>radius else np.zeros(n)
            g[rows]=y-adjust; class_corr[k]=adjust
        add_transition(projector_transitions,"class_mean",projector_before)
        projector_before=state_sha()
        for k,rows in enumerate(groups):
            y=xi[rows]+slack_corr[k]; new=np.maximum(y,0.0)
            if float(new.sum())>caps[k]:
                lo,hi=float(new.min()-caps[k]),float(new.max())
                for _ in range(100):
                    mid=0.5*(lo+hi)
                    if float(np.maximum(new-mid,0).sum())>caps[k]: lo=mid
                    else: hi=mid
                new=np.maximum(new-0.5*(lo+hi),0)
            slack_corr[k]=y-new; xi[rows]=new
        add_transition(projector_transitions,"slack",projector_before)
        projector_before=state_sha()
        for i in range(n):
            cols=rank["top"][i]; coef=rank["coeff"][i]; alpha=vote_corr[i]
            yg=g[i,cols]+alpha*coef; yx=xi[i]+alpha
            value=float(coef@yg+yx); norm2=float(coef@coef+1.0)
            tau=max(0.0,(ell[i]-value)/norm2); g[i,cols]=yg+tau*coef; xi[i]=yx+tau; vote_corr[i]=-tau
        add_transition(projector_transitions,"vote",projector_before)
        projector_before=state_sha()
        for k,rows in enumerate(groups):
            alpha=class_margin_corr[k]; scale=1.0/len(rows)
            for i in rows: g[i,rank["top"][i]]+=alpha*rank["coeff"][i]*scale
            value=float(np.mean([rank["coeff"][i]@g[i,rank["top"][i]] for i in rows]))
            rhs=float(rank["margins"][rows].mean())
            norm2=float(sum(rank["coeff"][i]@rank["coeff"][i] for i in rows))*scale*scale
            tau=max(0.0,(rhs-value)/norm2)
            for i in rows: g[i,rank["top"][i]]+=tau*rank["coeff"][i]*scale
            class_margin_corr[k]=-tau
        alpha=global_corr; scale=1.0/n
        for i in range(n): g[i,rank["top"][i]]+=alpha*rank["coeff"][i]*scale
        value=float(np.mean([rank["coeff"][i]@g[i,rank["top"][i]] for i in range(n)]))
        rhs=float(rank["margins"].mean())
        norm2=float(sum(c@c for c in rank["coeff"]))*scale*scale
        tau=max(0.0,(rhs-value)/norm2)
        for i in range(n): g[i,rank["top"][i]]+=tau*rank["coeff"][i]*scale
        global_corr=-tau
        add_transition(projector_transitions,"mean",projector_before)
        projector_before=state_sha(); y=g+centroid_corr*centroid; value=float(np.sum(centroid*y)); norm2=float(np.sum(centroid*centroid))
        tau=max(0.0,(centroid_rhs-value)/norm2); g=y+tau*centroid; centroid_corr=-tau
        add_transition(projector_transitions,"centroid",projector_before)
        projector_before=state_sha()
        for k in range(len(rank_corr)):
            i=int(rank["rank_i"][k]); a=int(rank["rank_a"][k]); b=int(rank["rank_b"][k]); alpha=rank_corr[k]
            ya=g[i,a]+alpha; yb=g[i,b]-alpha; value=ya-yb
            tau=max(0.0,(rank["rank_rhs"][k]-value)/2.0)
            g[i,a]=ya+tau; g[i,b]=yb-tau; rank_corr[k]=-tau
        add_transition(projector_transitions,"rank",projector_before)
        values=_real_values_v(g,xi,g0,labels,rank,ell,caps,cfg)
        relative=float(math.sqrt(np.linalg.norm(g-before_g)**2+np.linalg.norm(xi-before_xi)**2)/
                       max(math.sqrt(np.linalg.norm(before_g)**2+np.linalg.norm(before_xi)**2),1e-15))
        after=hobj({"g":g.tolist(),"xi":xi.tolist()})
        trace_hashes.append({"cycle":cycle,"before_sha256":before,"after_sha256":after,
                             "max_independent_set_violation":max(values.values()),
                             "relative_iterate_change":relative,
                             "correction_norms":correction_norms(),
                             "correction_state_sha256":hobj(correction_state()),
                             "projector_transitions":projector_transitions,
                             "projector_transitions_sha256":hobj(projector_transitions)})
        if max(values.values())<=solver["dykstra_set_violation_tolerance"] and \
                relative<=solver["dykstra_relative_change_tolerance"]:
            realized=[_rank(g,ids,i,solver["tie_tolerance"]) for i in range(n)]
            status="CELL_CONVERGED" if realized==rank["rankings"] else "BOUNDED_SEARCH_FEASIBLE"
            return {"status":status,"g":g,"xi":xi,"rank":rank,"ell":ell,"caps":caps,
                    "trace":trace_hashes,"values":values,"relative":relative}
    values=_real_values_v(g,xi,g0,labels,rank,ell,caps,cfg)
    return {"status":"BOUNDED_SEARCH_FEASIBLE","g":g,"xi":xi,"rank":rank,"ell":ell,
            "caps":caps,"trace":trace_hashes,"values":values,"relative":relative}


def _verify_real_rank_search(numerics, ids, labels, g0, cfg):
    search=numerics.get("rank_search",{})
    if search.get("reason")!="all_adjacent_checked":
        return False,{"reason":"not_all_adjacent_checked"}
    reference=np.asarray(search.get("orientation_reference_gram"),dtype=np.float64)
    if reference.shape!=(len(ids),len(ids)):
        return False,{"reason":"missing_orientation_reference_gram"}
    if hobj(reference.tolist())!=search.get("orientation_reference_gram_sha256"):
        return False,{"reason":"orientation_reference_hash_mismatch"}
    system=_independent_orientation_system(reference,ids)
    cells=search.get("cells",[])
    ok=(int(search.get("independent_orientations",-1))==system["rank"] and
        search.get("orientation_descriptors",[])==system["descriptors"] and
        search.get("basis_indices",[])==system["basis"] and
        np.linalg.norm(np.asarray(search.get("dependency_coefficients",[]))-
                       np.asarray(system["coeff"]))<=1e-8 and
        int(search.get("adjacent_cells_total",-1))==len(system["assignments"]) and
        int(search.get("adjacent_cells_checked",-2))==len(system["assignments"]) and
        len(cells)==len(system["assignments"]) and
        int(search.get("pivots",99))<=32)
    rebuilt_hashes=[]
    base=[_rank(reference,ids,i) for i in range(len(ids))]
    canonical=_rank_data_v(g0,labels,ids,cfg)
    replay_objectives=[]
    replay_hashes=[]
    replay_traces=[]
    for expected_assignment,row in zip(system["assignments"],cells):
        rebuilt=_independent_cell_from_assignment(base,system["descriptors"],
                                                  expected_assignment,ids)
        rebuilt_hash=hobj(rebuilt) if rebuilt is not None else None
        rebuilt_hashes.append(rebuilt_hash)
        if rebuilt is not None:
            replay=_independent_real_dykstra(g0,labels,ids,cfg,
                                             rankings_override=rebuilt,
                                             baseline_margins=canonical["margins"])
            result_hash=hobj({"g":replay["g"].tolist(),"xi":replay["xi"].tolist()})
            objective=float(np.linalg.norm(replay["g"]-g0)**2+np.linalg.norm(replay["xi"])**2)
            trace_transition_hash = hobj(
                [item.get("projector_transitions_sha256") for item in replay["trace"]])
            final_correction_hash = (
                replay["trace"][-1].get("correction_state_sha256") if replay["trace"] else None)
            replay_objectives.append(objective)
            replay_hashes.append(result_hash)
            replay_traces.append(replay["trace"])
            ok &= row.get("target_sha256")==result_hash and \
                abs(float(row.get("objective",float("inf")))-objective)<=1e-7 and \
                row.get("status")=="CELL_CONVERGED" and replay["status"]=="CELL_CONVERGED" and \
                int(row.get("cycles",-1))==len(replay["trace"]) and \
                row.get("trace_sha256")==hobj(replay["trace"]) and \
                row.get("trace_projector_transitions_sha256")==trace_transition_hash and \
                row.get("final_correction_state_sha256")==final_correction_hash and \
                abs(float(row.get("max_violation",float("inf")))-max(replay["values"].values()))<=1e-8
        ok &= (rebuilt is not None and row.get("assignment")==expected_assignment and
               row.get("cell_sha256",row.get("cell_rankings_sha256"))==rebuilt_hash and
               (("cell_rankings" not in row) or row.get("cell_rankings")==rebuilt) and
               row.get("status")=="CELL_CONVERGED")
    selected=[row["cell"] for row in numerics.get("_rank_rows",[])]
    selected_hash=hobj(selected) if selected else None
    if selected_hash is not None:
        ok &= search.get("selected_cell_rankings_sha256")==selected_hash
        if search.get("best_cell_sha256") is not None:
            ok &= search.get("best_cell_sha256")==selected_hash
    target_hash=hobj({"g":np.asarray(numerics.get("target_gram"),dtype=np.float64).tolist(),
                      "xi":np.asarray(numerics.get("slack"),dtype=np.float64).tolist()})
    if replay_objectives:
        best_index=int(np.argmin(replay_objectives))
        ok &= replay_hashes[best_index]==target_hash
        selected_trace = replay_traces[best_index]
        projector_rows = numerics.get("_projector_rows", [])
        ok &= hobj(selected_trace)==numerics.get("selected_trace_sha256") and \
            hobj([item.get("projector_transitions_sha256") for item in selected_trace])== \
            numerics.get("selected_trace_projector_transitions_sha256") and \
            (selected_trace[-1].get("correction_state_sha256") if selected_trace else None)== \
            numerics.get("selected_final_correction_state_sha256") and \
            len(projector_rows)==len(selected_trace) and hobj(projector_rows)==hobj(selected_trace)
    return bool(ok),{"rank":system["rank"],"compatible_cells":len(system["assignments"]),
                     "ledger_cells":len(cells),"rebuilt_cell_hashes":rebuilt_hashes,
                     "replayed_cell_hashes":replay_hashes,
                     "replayed_objective_min":min(replay_objectives) if replay_objectives else None}


def _signed_tangent_families(z0, labels, anchor):
    same=[]; opposite=[]; signed=[]; meta=[]
    for j in range(len(z0)):
        if j==anchor: continue
        tangent=z0[j]-z0[anchor]*float(z0[anchor]@z0[j])
        tangent/=max(float(np.linalg.norm(tangent)),1e-15)
        if labels[j]==labels[anchor]:
            same.append((j,tangent)); signed.append(tangent); meta.append(("same",int(j)))
        else:
            column=-tangent; opposite.append((j,column)); signed.append(column); meta.append(("opposite",int(j)))
    return same,opposite,np.asarray(signed,dtype=np.float64),meta


def _combo_column(a,b):
    column=np.asarray(a,dtype=np.float64)+np.asarray(b,dtype=np.float64)
    return column/max(float(np.linalg.norm(column)),1e-15)


def _pair_oracle(signed,meta,witness):
    if len(signed)<2: return {"family":"pair","value":-float("inf"),"indices":[],"column":None}
    gram=signed@signed.T; inner=signed@witness; best=(-float("inf"),None,None)
    for a in range(len(signed)-1):
        denom=np.sqrt(np.maximum(2.0+2.0*gram[a,a+1:],1e-15))
        values=(inner[a]+inner[a+1:])/denom
        local=int(np.argmax(values)); value=float(values[local])
        if value>best[0]: best=(value,a,a+1+local)
    return {"family":"pair","value":best[0],
            "indices":[meta[best[1]],meta[best[2]]],
            "column":_combo_column(signed[best[1]],signed[best[2]])}


def _triplet_oracle(same,opposite,witness):
    if not same or not opposite:
        return {"family":"triplet","value":-float("inf"),"indices":[],"column":None}
    same_cols=np.asarray([row[1] for row in same],dtype=np.float64)
    opp_cols=np.asarray([row[1] for row in opposite],dtype=np.float64)
    denom=np.sqrt(np.maximum(2.0+2.0*(same_cols@opp_cols.T),1e-15))
    values=((same_cols@witness)[:,None]+(opp_cols@witness)[None,:])/denom
    a,b=np.unravel_index(int(np.argmax(values)),values.shape)
    return {"family":"triplet","value":float(values[a,b]),
            "indices":[int(same[a][0]),int(opposite[b][0])],
            "column":_combo_column(same_cols[a],opp_cols[b])}


PINNED_REGISTERED_CONE_DEFINITION = {
    "singleton": "for each anchor i and memory j!=i, the unit tangent to z_j at z_i, signed attractive for same parent label and repulsive for opposite parent label",
    "pair": "all normalized sums of two distinct singleton columns for the same anchor",
    "triplet": "all normalized sums of one same-label attractive singleton and one opposite-label repulsive singleton for the same anchor",
    "supcon": "for each anchor, the normalized sum of all same-label attractive singleton columns",
    "labels": "parent-video binary labels only",
}


def _registered_cone_definition(cfg):
    definition = dict(cfg.get("registered_cone_definition", {}))
    if definition != PINNED_REGISTERED_CONE_DEFINITION:
        raise RuntimeError("registered cone definition drift")
    return definition


def _independent_real_cone(z0,target,labels,reported,cfg):
    labels=np.asarray(labels,dtype=np.int64); disp=target-z0
    max_iter=int(cfg["solver"].get("farkas_column_generation_max_iter",64))
    sep_tol=float(cfg["solver"].get("farkas_separation_tolerance",1e-8))
    residuals=[]; flat_targets=[]; active_banks=[]; ledgers=[]
    universe={"singleton_rgcl_columns":0,"pair_columns":0,"triplet_columns":0,
              "supcon_columns":len(z0),"active_generated_columns":0}
    family_max={"singleton":-float("inf"),"pair":-float("inf"),
                "triplet":-float("inf"),"supcon":-float("inf")}
    overflow=False
    for i in range(len(z0)):
        d=disp[i]-z0[i]*float(z0[i]@disp[i])
        same,opposite,signed,meta=_signed_tangent_families(z0,labels,i)
        supcon=np.sum(np.asarray([row[1] for row in same]),axis=0) if same else np.zeros(z0.shape[1])
        supcon/=max(float(np.linalg.norm(supcon)),1e-15)
        active=[row.copy() for row in signed]+[supcon.copy()]
        active_meta=[("singleton",m) for m in meta]+[("supcon",int(i))]
        oracle_trace=[]
        local_overflow=False
        for iteration in range(max_iter+1):
            matrix=np.asarray(active,dtype=np.float64).T
            coeff,_=nnls(matrix,d,maxiter=max(1000,20*len(active)))
            residual=d-matrix@coeff; norm=float(np.linalg.norm(residual))
            witness=residual/max(norm,1e-15)
            singleton_value=float(np.max(signed@witness)) if len(signed) else -float("inf")
            supcon_value=float(supcon@witness)
            pair=_pair_oracle(signed,meta,witness); triplet=_triplet_oracle(same,opposite,witness)
            best=max([{"family":"singleton","value":singleton_value,"indices":[],"column":None},
                      {"family":"supcon","value":supcon_value,"indices":[int(i)],"column":None},
                      pair,triplet],key=lambda row:row["value"])
            oracle_trace.append({"iteration":iteration,"family":best["family"],
                                 "value":best["value"],"indices":best["indices"]})
            if best["value"]<=sep_tol or best["column"] is None: break
            active.append(best["column"]); active_meta.append((best["family"],best["indices"]))
        else:
            local_overflow=True; overflow=True
        matrix=np.asarray(active,dtype=np.float64).T
        coeff,_=nnls(matrix,d,maxiter=max(1000,20*len(active)))
        residual=d-matrix@coeff; residuals.append(residual); flat_targets.append(d)
        active_banks.append(np.asarray(active,dtype=np.float64))
        witness=residual/max(float(np.linalg.norm(residual)),1e-15)
        singleton_max=float(np.max(signed@witness)) if len(signed) else -float("inf")
        supcon_max=float(supcon@witness)
        pair_max=_pair_oracle(signed,meta,witness)["value"]
        triplet_max=_triplet_oracle(same,opposite,witness)["value"]
        family_max["singleton"]=max(family_max["singleton"],singleton_max)
        family_max["supcon"]=max(family_max["supcon"],supcon_max)
        family_max["pair"]=max(family_max["pair"],pair_max)
        family_max["triplet"]=max(family_max["triplet"],triplet_max)
        universe["singleton_rgcl_columns"]+=len(signed)
        universe["pair_columns"]+=len(signed)*(len(signed)-1)//2
        universe["triplet_columns"]+=len(same)*len(opposite)
        universe["active_generated_columns"]+=len(active)
        ledgers.append({"anchor":i,"same_columns":len(same),"opposite_columns":len(opposite),
                        "singleton_universe":len(signed),
                        "pair_universe":len(signed)*(len(signed)-1)//2,
                        "triplet_universe":len(same)*len(opposite),
                        "supcon_universe":1,"active_columns":len(active),
                        "active_column_meta_sha256":hobj(active_meta),
                        "active_columns_sha256":hobj(np.asarray(active,dtype=np.float64).tolist()),
                        "column_generation_iterations":len(oracle_trace)-1,
                        "separation_overflow":local_overflow,
                        "oracle_trace":oracle_trace,
                        "nnls_residual_norm":float(np.linalg.norm(residual))})
    residual=np.concatenate(residuals); flat=np.concatenate(flat_targets)
    norm=float(np.linalg.norm(residual)); witness=residual/max(norm,1e-15)
    active_max=-float("inf"); offset=0
    for active in active_banks:
        wi=witness[offset:offset+z0.shape[1]]; offset+=z0.shape[1]
        active_max=max(active_max,float(np.max(active@wi)))
    definition=_registered_cone_definition(cfg)
    universe_sha=hobj({"definition":definition,"universe":universe,"anchors":ledgers})
    return {"relative_separation":norm/max(float(np.linalg.norm(flat)),1e-15),
            "max_family_inner":max(family_max.values()),
            "family_max_witness_inner":family_max,
            "max_active_inner":active_max,
            "max_cone_witness_inner":max(active_max,max(family_max.values())),
            "duality_gap":abs(norm-float(witness@flat)),
            "universe":universe,"universe_sha256":universe_sha,
            "separation_overflow":overflow,
            "reported_definition_match":reported.get("registered_cone_definition")==definition,
            "reported_universe_match":universe==reported.get("universe"),
            "reported_family_error":max(abs(float(family_max[k])-float(reported.get("family_max_witness_inner",{}).get(k,float("inf")))) for k in family_max)}


def verify_real(cfg, artifacts):
    base=artifacts/"g0/real/MHC_zh/fold4"
    required=["timings.json","numerics.json","projectors.jsonl","rank_cells.jsonl",
              "exact_vote.jsonl","farkas.json","factor.json","fit_rollback.json",
              "fit_replay.json","resource.json","manifest.json"]
    if not all((base/name).exists() for name in required):
        return False,{"required_files":False,"missing":[name for name in required if not (base/name).exists()]},{}
    manifest=read_json(base/"manifest.json"); numerics=read_json(base/"numerics.json")
    timing=read_json(base/"timings.json"); farkas=read_json(base/"farkas.json")
    factor=read_json(base/"factor.json"); fit=read_json(base/"fit_rollback.json")
    resource=read_json(base/"resource.json")
    replay=read_json(base/"fit_replay.json")
    projectors=read_jsonl(base/"projectors.jsonl")
    rank_rows=read_jsonl(base/"rank_cells.jsonl")
    exact_vote=read_jsonl(base/"exact_vote.jsonl")
    bank,opened=_load_real_bank(cfg)
    fixture=cfg["sealed_real_fixture"]
    ids=[str(x) for x in bank["memory_ids"].tolist()]
    held=[str(x) for x in bank["query_ids"].tolist()]
    labels=np.asarray(bank["memory_labels"],dtype=np.int64)
    z0=np.asarray(bank["memory_z"],dtype=np.float64)
    z0/=np.linalg.norm(z0,axis=1,keepdims=True)
    g0=z0@z0.T
    target=np.asarray(numerics.get("target_gram"),dtype=np.float64)
    xi=np.asarray(numerics.get("slack"),dtype=np.float64)
    ell=np.asarray(numerics.get("ell"),dtype=np.float64)
    caps=[float(x) for x in numerics.get("slack_caps",[])]
    cell_rankings=[row["cell"] for row in rank_rows]
    target_rankings=[_rank(target,ids,i,cfg["solver"]["tie_tolerance"]) for i in range(len(ids))]
    constraint_values,_,_,rank_count=_real_constraint_audit(
        target,xi,g0,labels,ids,ell,caps,cell_rankings)
    reported_violations={k:float(v) for k,v in numerics.get("violations",{}).items()}
    rank_search_ok,rank_search_detail=_verify_real_rank_search(
        {**numerics,"_rank_rows":rank_rows,"_projector_rows":projectors},ids,labels,g0,cfg)
    independent_factor=_independent_factor(target)
    u,_,vt=np.linalg.svd(independent_factor.T@z0,full_matrices=False)
    independent_aligned=independent_factor@(u@vt)
    emitted=target if numerics.get("trainable_target") is True else g0
    independent_vote=_independent_exact_vote(emitted,labels,ids)
    abstract=_independent_real_cone(z0,independent_aligned,labels,farkas["abstract"],cfg)
    realized=np.asarray(fit.get("realized_bank"),dtype=np.float64)
    realized_cone=_independent_real_cone(z0,realized,labels,farkas["realized"],cfg)
    samples=[float(x) for x in timing.get("refresh_timing_samples_seconds",[])]
    p95=max(samples) if len(samples)==1 else (
        float(np.percentile(samples,95,method="higher")) if samples else float("nan"))
    recomputed_h10=cfg["cost"]["contingency"]*cfg["cost"]["folds"]*(
        (2*float(timing.get("remove_fullfold_seconds",float("nan"))) +
         cfg["cost"]["refresh_multiplier"]*p95 +
         float(timing.get("final_bank_seconds",float("nan"))))/3600.0)
    forbidden_paths=set(PROTECTED_PATH_PREFIXES)
    manifest_forbidden_surfaces=_forbidden_formal_surfaces({
        "input_files": manifest.get("input_files", []),
        "access_ledger": manifest.get("access_ledger", []),
    })
    manifest_input_paths={row.get("path") for row in manifest.get("input_files",[])}
    manifest_access_paths={row.get("path") for row in manifest.get("access_ledger",[])}
    contract=fit.get("selective_cache_contract",{})
    gates={
        "manifest":verify_manifest_files(manifest,ROOT,cfg,
                                         _lineage_run_id(cfg, "realfold"),
                                         "G0_REAL_FOLD") and manifest.get("status")=="PASS",
        "bank_isolation":set(opened)==set(fixture["allowed_bank_members"]) and
            set(manifest.get("opened_bank_members",[]))==set(fixture["allowed_bank_members"]) and
            manifest.get("forbidden_bank_members_opened")==[] and
            bool(forbidden_paths) and
            not manifest_forbidden_surfaces and
            not any(_forbidden_path_string(path) for path in manifest_input_paths if path) and
            not any(_forbidden_path_string(path) for path in manifest_access_paths if path) and
            contract.get("source")=="authoritative_outer_train_only_artifacts" and
            contract.get("combined_train_cache_opened") is False and
            contract.get("segment_cache_opened") is False and
            contract.get("segment_objective_allowed") is False and
            contract.get("mmap_used") is False and
            int(contract.get("outer_held_content_or_label_reads",-1))==0,
        "bank_members":len(ids)==fixture["outer_train_n"] and
            len(held)==fixture["outer_held_n"] and not (set(ids)&set(held)) and
            hobj(ids)==fixture["memory_ids_sha256"] and
            hobj(held)==fixture["query_ids_sha256"] and
            set(np.unique(labels).tolist())=={0,1},
        "local_stationary":numerics.get("target_status")=="LOCAL_STATIONARY_CERTIFIED",
        "numerics":float(numerics.get("max_independent_violation",float("inf")))<=1e-6 and
            float(numerics.get("relative_iterate_change",float("inf")))<=1e-7 and
            max(constraint_values.values())<=1e-6 and
            all(abs(constraint_values[k]-reported_violations.get(k,float("inf")))<=1e-8
                for k in constraint_values) and
            rank_count==int(numerics.get("rank_halfspace_count",-1)) and
            len(projectors)==int(numerics.get("cycles",-2)),
        "rank_search":rank_search_ok,
        "rank_tie_vote_invariance":cell_rankings==target_rankings,
        "exact_vote":len(exact_vote)==len(ids) and hobj(independent_vote)==hobj(exact_vote) and
            hobj(emitted.tolist())==numerics.get("emitted_exact_vote_gram_sha256"),
        "fit":float(fit.get("displacement_cosine",-1))>=0.8 and float(fit.get("relative_target_residual",float("inf")))<=0.5 and fit.get("rollback_hash_identical") is True,
        "fit_replay":verify_payload(replay) and replay.get("status")=="PASS" and
            replay.get("producer_fit_sha256")==hfile(base/"fit_rollback.json") and
            replay.get("producer_numerics_sha256")==hfile(base/"numerics.json") and
            replay.get("producer_factor_sha256")==hfile(base/"factor.json") and
            replay.get("realized_bank_sha256")==fit.get("realized_bank_sha256") and
            replay.get("batch_order_sha256")==fit.get("batch_order_sha256") and
            replay.get("segment_cache_used") is False and
            float(replay.get("lambda_seg", -1.0))==0.0 and
            replay.get("gates",{}).get("no_segment_objective") is True and
            replay.get("rollback",{}).get("rollback_replay_sha256")==fit.get("rollback_replay_sha256") and
            replay.get("rollback",{}).get("direct_remove_sha256")==fit.get("direct_remove_sha256"),
        "factor":float(factor.get("gram_reconstruction_error",float("inf")))<=1e-6 and
            float(factor.get("aligned_gram_reconstruction_error",float("inf")))<=1e-6 and
            hobj(independent_aligned.tolist())==factor.get("aligned_factor_sha256"),
        "farkas":float(farkas.get("abstract_relative_separation",-1))>=0.25 and
            float(farkas.get("realized_relative_separation",-1))>=0.25 and
            float(farkas.get("max_duality_gap",float("inf")))<=1e-5 and
            abs(abstract["relative_separation"]-float(farkas.get("abstract_relative_separation",float("nan"))))<=1e-8 and
            abs(realized_cone["relative_separation"]-float(farkas.get("realized_relative_separation",float("nan"))))<=1e-8 and
            max(abstract["max_cone_witness_inner"],realized_cone["max_cone_witness_inner"])<=1e-8 and
            max(abstract["duality_gap"],realized_cone["duality_gap"])<=1e-5 and
            abstract["universe_sha256"]==farkas["abstract"].get("universe_sha256") and
            realized_cone["universe_sha256"]==farkas["realized"].get("universe_sha256") and
            abstract["reported_definition_match"] and realized_cone["reported_definition_match"] and
            abstract["reported_universe_match"] and realized_cone["reported_universe_match"] and
            abstract["reported_family_error"]<=1e-8 and realized_cone["reported_family_error"]<=1e-8 and
            abstract["separation_overflow"] is False and realized_cone["separation_overflow"] is False,
        "resources":float(resource.get("peak_gpu_gib",float("inf")))<=24 and
            float(resource.get("peak_host_rss_gib",float("inf")))<=64 and
            resource.get("one_gpu") is True and
            int(resource.get("torch_cuda_device_count",-1))==1 and
            int(resource.get("cuda_visible_device_count",-1))==1 and
            bool(resource.get("gpu_name")),
        "cost":timing.get("H10_formula")==cfg["cost"]["formula"] and
            timing.get("H10_p95_sample_semantics")==cfg["cost"]["p95_sample_semantics"] and
            abs(float(timing.get("p95_refresh_seconds",float("nan")))-p95)<=1e-9 and
            math.isfinite(float(timing.get("H10_upper_gpu_hours",float("inf")))) and
            abs(recomputed_h10-float(timing.get("H10_upper_gpu_hours",float("nan"))))<=1e-9 and
            float(timing.get("H10_upper_gpu_hours",float("inf")))<160,
        "no_endpoint":int(manifest.get("outer_held_prediction_count",-1))==0,
    }
    details={"H10_upper_gpu_hours":timing.get("H10_upper_gpu_hours"),
             "rank_search":rank_search_detail,
             "rank_tie_vote_invariance":cell_rankings==target_rankings,
             "constraint_values":constraint_values,
             "opened_bank_members":opened,
             "manifest_forbidden_surfaces":manifest_forbidden_surfaces,
             "fit_replay_status":replay.get("status"),
             "abstract_cone":abstract,
             "realized_cone":realized_cone}
    return all(gates.values()),gates,details


def decide(cfg,args):
    if args.run_id != _lineage_run_id(cfg, "decision"):
        raise RuntimeError("wrong decision run id")
    assert_no_forbidden_formal_surface(cfg, "formal_config")
    artifacts=_resolve_cfg_path(cfg,"artifacts")
    freeze=read_json(artifacts/"CONFIG_FREEZE.json"); audit=read_json(artifacts/"g0/code_audit/audit.json")
    assert_no_forbidden_formal_surface(freeze, "config_freeze")
    assert_no_forbidden_formal_surface(audit, "code_audit")
    config_hash=hobj(cfg)
    impl,_=current_implementation(cfg)
    freeze_gate=verify_payload(freeze) and freeze.get("status")=="FROZEN" and \
        freeze.get("run_id") == _lineage_run_id(cfg, "freeze") and \
        freeze.get("stage")=="G0_FREEZE" and \
        freeze.get("config_canonical_sha256")==config_hash and freeze.get("implementation_sha256")==impl and \
        freeze.get("independent_verifier_sha256")==hfile(Path(__file__)) and \
        freeze.get("dirty_diff_sha256")==current_dirty_hash(cfg)
    required_freeze_keys = (
        "checkpoint", "bank", "outer_train_feature_cache",
        "sanitized_provenance", "sanitizer_decision", "remove_ledger",
    )
    expected_freeze_paths = {_config_rel(cfg)}
    expected_freeze_paths |= {
        _relpath(_resolve_cfg_path(cfg, key)) for key in required_freeze_keys
    }
    expected_freeze_paths |= {
        _relpath(_resolve_cfg_path(cfg, key))
        for key in _freeze_protocol_input_keys(cfg)
    }
    def freeze_input_ok(row):
        path=canonical_root_path(row["path"])[0]
        if not path.exists(): return False
        if "member_sha256" in row:
            return _hash_npz_allowed_members(path,cfg)==row["member_sha256"]
        return hfile(path)==row["sha256"]
    freeze_gate &= freeze.get("access_ledger_sha256")==hobj(freeze.get("access_ledger",[])) and \
        {row.get("path") for row in freeze.get("input_files",[])}==expected_freeze_paths and \
        {row.get("path") for row in freeze.get("access_ledger",[])}==expected_freeze_paths and \
        all(freeze_input_ok(row) for row in freeze.get("input_files",[]))
    audit_gate = verify_code_audit_publication(
        cfg, artifacts, freeze, audit, config_hash, impl)
    synth_ok,synth_gates,synth_details=verify_synthetic(cfg,artifacts)
    real_ok,real_gates,real_details=verify_real(cfg,artifacts)
    counters={key:max(int(freeze.get(key,-1)),int(audit.get(key,-1))) for key in ZERO_COUNTER_KEYS}
    zero_gate=all(v==0 for v in counters.values())
    supervision_gate=all(obj.get("only_gold_supervision")=="parent_video_binary_label" and obj.get("segment_gold_exists") is False and obj.get("segment_gold_used") is False for obj in (freeze,audit))
    all_gates={"freeze":freeze_gate,"code_audit":audit_gate,"synthetic":synth_ok,
               "real":real_ok,"zero_forbidden_counters":zero_gate,
               "supervision_contract":supervision_gate}
    decision="GO" if all(all_gates.values()) else "STOP"
    import torch
    result={"schema_version":1,"run_id":args.run_id,"stage":"G0_DECISION",
            "status":decision,"G1_G4_locked":True,"fresh_G1_authorization_required":True,
            "gates":all_gates,"synthetic_gates":synth_gates,"real_gates":real_gates,
            "synthetic_details":synth_details,"real_details":real_details,
            "only_gold_supervision":"parent_video_binary_label",
            "segment_gold_exists":False,"segment_gold_used":False,
            "slurm_job_id":os.environ.get("SLURM_JOB_ID"),
            "config_canonical_sha256":config_hash,
            "independent_verifier_sha256":hfile(Path(__file__)),
            "python_version":platform.python_version(),"numpy_version":np.__version__,
            "scipy_version":scipy.__version__,"torch_version":torch.__version__,
            "conda_env":os.environ.get("CONDA_DEFAULT_ENV"),**counters}
    result["verifier_access_ledger"]=list(VERIFIER_ACCESS_LEDGER)
    result["verifier_access_ledger_sha256"]=hobj(result["verifier_access_ledger"])
    result["payload_sha256"]=hobj(result)
    publish_exclusive(artifacts/"G0_DECISION.json",result)
    print(cjson({"status":decision,"gates":all_gates,"run_id":args.run_id}))
    if decision!="GO": raise SystemExit(2)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--config",required=True)
    parser.add_argument("--task",required=True,choices=["decide", "audit-publish"])
    parser.add_argument("--run-id",required=True)
    parser.add_argument("--review")
    parser.add_argument("--review-record")
    args=parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"): raise RuntimeError("independent verifier must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV")!="HateVideo": raise RuntimeError("expected HateVideo")
    cfg=read_json(args.config)
    assert_no_forbidden_formal_surface(cfg, "formal_config")
    if cfg["authorization"]["authorized_stages"]!=["G0"] or cfg["authorization"]["locked_stages"]!=["G1","G2","G3","G4"]: raise RuntimeError("authorization drift")
    dirty_state_policy(cfg)
    if args.task == "audit-publish":
        audit_publish(cfg, args)
    else:
        decide(cfg,args)


if __name__=="__main__": main()
