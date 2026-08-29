#!/usr/bin/env python
from __future__ import annotations

import copy
import json
import os
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import scipy
import torch

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from lb_scgp_common import (  # noqa: E402
    AccessLedger,
    canonical_json,
    dirty_state_policy,
    git_state,
    implementation_hash,
    load_config,
    payload_hash,
    root_relative_path,
    sha256_file,
    sha256_obj,
)
import lb_scgp_g0 as producer  # noqa: E402


OUT = ROOT / "refine-logs/lb_scgp/runtime/v5_repair_checks"
JOB_ID = os.environ.get("SLURM_JOB_ID", "noslurm")
RUN_ROOT = OUT / ("fixture_" + JOB_ID)
RESULT_PATH = OUT / ("regression_result_" + JOB_ID + ".json")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(obj) + "\n", encoding="utf-8")


def write_text(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def lock_bytes(run_id, rel):
    return (canonical_json({
        "lock_type": "LB_SCGP_FORMAL_NO_CLOBBER",
        "run_id": run_id,
        "path": rel,
    }) + "\n").encode("utf-8")


def code_audit_contract(cfg):
    version = cfg["lineage"]["version"]
    suffix = version.upper()
    return {
        "record_type": "LB_SCGP_G0_CODE_AUDIT_INDEPENDENT_REVIEW_RECORD_{}".format(suffix),
        "artifact_type": "LB_SCGP_G0_CODE_AUDIT_PASS_ARTIFACT_{}".format(suffix),
        "index_type": "LB_SCGP_G0_CODE_AUDIT_PUBLICATION_INDEX_{}".format(suffix),
        "review_scope": "LB-SCGP G0 {} formal code audit for {}".format(
            version, cfg["lineage"]["run_ids"]["freeze"]),
    }


def prior_rows(cfg):
    expected = cfg["lineage"]["prior_lineage_no_clobber_hashes"]
    return [{"path": path, "sha256": sha256_file(ROOT / path)}
            for path in sorted(expected)]


def base_fixture_config(case_dir):
    cfg = json.loads((ROOT / "configs/lb_scgp/lb_scgp_v5.json").read_text(
        encoding="utf-8"))
    cfg["paths"]["artifacts"] = root_relative_path(case_dir / "artifacts")
    cfg_path = case_dir / "config.json"
    review_path = case_dir / "source_review.md"
    record_path = case_dir / "source_review_record.json"
    cfg["lineage"]["config_path"] = root_relative_path(cfg_path)
    cfg["lineage"]["review_report_path"] = root_relative_path(review_path)
    cfg["lineage"]["review_record_path"] = root_relative_path(record_path)
    dirty_paths = list(cfg["lineage"]["dirty_state_excluded_paths"])
    for rel in (root_relative_path(review_path), root_relative_path(record_path)):
        if rel not in dirty_paths:
            dirty_paths.append(rel)
    cfg["lineage"]["dirty_state_excluded_paths"] = dirty_paths
    cfg["lineage"]["mutable_records_excluded_from_freeze_inputs"] = dirty_paths
    return cfg, cfg_path, review_path, record_path


def build_fixture(name, mutator=None):
    case_dir = RUN_ROOT / name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    cfg, cfg_path, review_source, record_source = base_fixture_config(case_dir)
    write_json(cfg_path, cfg)

    load_ledger = AccessLedger()
    loaded = load_config(str(cfg_path), load_ledger)
    contract = code_audit_contract(loaded)
    artifacts = Path(loaded["paths"]["artifacts"])
    artifacts_abs = ROOT / artifacts
    code_dir = artifacts_abs / "g0/code_audit"
    code_dir.mkdir(parents=True)

    input_path = case_dir / "frozen_input.txt"
    write_text(input_path, "temporary v5 schema regression input\n")
    impl_hash, impl_rows = implementation_hash(loaded)
    verifier_sha = sha256_file(ROOT / "scripts/analysis/lb_scgp_independent_verify.py")
    head, dirty = git_state(cfg=loaded)
    artifact_excludes, dirty_paths, dirty_prefixes = dirty_state_policy(loaded)

    freeze_path = artifacts_abs / "CONFIG_FREEZE.json"
    freeze_lock_path = freeze_path.with_name(freeze_path.name + ".publish.lock")
    freeze = {
        "schema_version": 1,
        "run_id": loaded["lineage"]["run_ids"]["freeze"],
        "stage": "G0_FREEZE",
        "slurm_job_id": JOB_ID,
        "git_head": head,
        "dirty_diff_sha256": dirty,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "config_canonical_sha256": loaded["config_canonical_sha256"],
        "implementation_sha256": impl_hash,
        "implementation_files": impl_rows,
        "independent_verifier_sha256": verifier_sha,
        "only_gold_supervision": "parent_video_binary_label",
        "segment_gold_exists": False,
        "segment_gold_used": False,
        "status": "FROZEN",
        "freeze_name": loaded["lineage"]["freeze_name"],
        "lineage_version": "v5",
        "input_files": [
            {"path": root_relative_path(cfg_path), "sha256": sha256_file(cfg_path)},
            {"path": root_relative_path(input_path), "sha256": sha256_file(input_path)},
        ],
        "formal_artifact_exclude_prefixes": list(artifact_excludes),
        "dirty_state_excluded_paths": list(dirty_paths),
        "dirty_state_excluded_prefixes": list(dirty_prefixes),
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
    }
    freeze["payload_sha256"] = payload_hash(freeze)
    write_json(freeze_path, freeze)
    freeze_lock_path.write_bytes(lock_bytes(
        loaded["lineage"]["run_ids"]["freeze"], root_relative_path(freeze_path)))

    review_text = "# Temporary v5 schema regression review\n\nNot a formal review.\n"
    write_text(review_source, review_text)
    review_sha = sha256_file(review_source)

    record = {
        "schema_version": 1,
        "record_type": contract["record_type"],
        "run_id": loaded["lineage"]["run_ids"]["code_audit"],
        "stage": "G0_CODE_AUDIT",
        "status": "PASS",
        "lineage_version": "v5",
        "config_path": loaded["lineage"]["config_path"],
        "artifact_namespace": loaded["lineage"]["artifact_namespace"],
        "freeze_run_id": loaded["lineage"]["run_ids"]["freeze"],
        "freeze_path": root_relative_path(freeze_path),
        "freeze_file_sha256": sha256_file(freeze_path),
        "freeze_payload_sha256": freeze["payload_sha256"],
        "config_canonical_sha256": loaded["config_canonical_sha256"],
        "implementation_sha256": impl_hash,
        "independent_verifier_sha256": verifier_sha,
        "review_report_path": loaded["lineage"]["review_report_path"],
        "review_report_sha256": review_sha,
        "reviewer_identity": "temporary_runtime_fixture",
        "review_process_identity": "fresh_independent_gpt_5_5_xhigh",
        "review_scope": contract["review_scope"],
        "critical": 0,
        "high": 0,
        "important": 0,
        "no_segment_gold_pass": True,
        "formal_pass_authorized": True,
        "independent_reviewer": True,
        "repair_executor_created": False,
        "only_gold_supervision": "parent_video_binary_label",
        "segment_gold_exists": False,
        "segment_gold_used": False,
    }
    record["payload_sha256"] = payload_hash(record)
    write_json(record_source, record)
    record_sha = sha256_file(record_source)

    review_artifact = code_dir / "review.md"
    record_artifact = code_dir / "review_record.json"
    audit_artifact = code_dir / "audit.json"
    index_artifact = code_dir / "publication_index.json"
    write_text(review_artifact, review_text)
    write_json(record_artifact, record)

    review_output = {"path": root_relative_path(review_artifact), "sha256": sha256_file(review_artifact)}
    record_output = {"path": root_relative_path(record_artifact), "sha256": sha256_file(record_artifact)}
    audit = {
        "schema_version": 1,
        "artifact_type": contract["artifact_type"],
        "run_id": loaded["lineage"]["run_ids"]["code_audit"],
        "stage": "G0_CODE_AUDIT",
        "status": "PASS",
        "critical": 0,
        "high": 0,
        "important": 0,
        "no_segment_gold_pass": True,
        "formal_pass_authorized": True,
        "slurm_job_id": JOB_ID,
        "config_path": loaded["lineage"]["config_path"],
        "config_file_sha256": sha256_file(cfg_path),
        "config_canonical_sha256": loaded["config_canonical_sha256"],
        "artifact_namespace": loaded["lineage"]["artifact_namespace"],
        "lineage_version": "v5",
        "freeze_path": root_relative_path(freeze_path),
        "freeze_file_sha256": sha256_file(freeze_path),
        "freeze_lock_path": root_relative_path(freeze_lock_path),
        "freeze_lock_sha256": sha256_file(freeze_lock_path),
        "freeze_payload_sha256": freeze["payload_sha256"],
        "freeze_run_id": loaded["lineage"]["run_ids"]["freeze"],
        "freeze_stage": "G0_FREEZE",
        "git_head": head,
        "dirty_diff_sha256": dirty,
        "frozen_dirty_diff_sha256": freeze["dirty_diff_sha256"],
        "implementation_sha256": impl_hash,
        "implementation_files": impl_rows,
        "independent_verifier_sha256": verifier_sha,
        "review_report_path": loaded["lineage"]["review_report_path"],
        "review_report_sha256": review_sha,
        "review_record_path": loaded["lineage"]["review_record_path"],
        "review_record_sha256": record_sha,
        "review_record_payload_sha256": record["payload_sha256"],
        "reviewer_identity": record["reviewer_identity"],
        "review_process_identity": record["review_process_identity"],
        "review_scope": record["review_scope"],
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
        "access_ledger": [],
        "access_ledger_sha256": sha256_obj([]),
        "dirty_policy": {
            "formal_artifact_exclude_prefixes": list(artifact_excludes),
            "dirty_state_excluded_paths": list(dirty_paths),
            "dirty_state_excluded_prefixes": list(dirty_prefixes),
        },
        "frozen_input_rehashes": copy.deepcopy(freeze["input_files"]),
        "allowed_bank_member_sha256": loaded["sealed_real_fixture"]["bank_member_sha256"],
        "forbidden_bank_members_not_opened": loaded["sealed_real_fixture"]["forbidden_bank_members"],
        "prior_lineage_no_clobber_hashes": prior_rows(loaded),
        "audit_publish_wrapper": {
            "path": loaded["lineage"]["audit_publish_wrapper_path"],
            "sha256": sha256_file(ROOT / loaded["lineage"]["audit_publish_wrapper_path"]),
            "cpus_per_task": 2,
            "mem": "4G",
            "no_time_directive": True,
            "conda_env": "HateVideo",
            "offline": True,
        },
        "authorization_gate": {
            "wrapper_task": "audit-publish",
            "wrapper_run_id": loaded["lineage"]["run_ids"]["code_audit"],
            "config_path": loaded["lineage"]["config_path"],
            "review_path": loaded["lineage"]["review_report_path"],
            "review_record_path": loaded["lineage"]["review_record_path"],
            "no_manual_pass_written": True,
            "publisher_recomputed_binding_checks": True,
        },
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
    if mutator:
        mutator(loaded, freeze, record, audit)
    audit["payload_sha256"] = payload_hash(audit)
    write_json(audit_artifact, audit)

    output_names = ("review.md", "review_record.json", "audit.json", "publication_index.json")
    lock_payloads = {}
    for name in output_names:
        rel = root_relative_path(code_dir / name)
        payload = lock_bytes(loaded["lineage"]["run_ids"]["code_audit"], rel)
        lock_payloads[name] = payload
        (code_dir / (name + ".publish.lock")).write_bytes(payload)
    audit_output = {"path": root_relative_path(audit_artifact), "sha256": sha256_file(audit_artifact)}
    index = {
        "schema_version": 1,
        "artifact_type": contract["index_type"],
        "run_id": loaded["lineage"]["run_ids"]["code_audit"],
        "stage": "G0_CODE_AUDIT",
        "status": "PASS",
        "output_files": [review_output, record_output, audit_output],
        "lock_files": [
            {"path": root_relative_path(code_dir / (name + ".publish.lock")),
             "sha256": sha256_file(code_dir / (name + ".publish.lock"))}
            for name in output_names
        ],
    }
    if mutator is mutate_path:
        index["output_files"][0]["path"] = "refine-logs/lb_scgp/runtime/v5_repair_checks/wrong_review.md"
    index["payload_sha256"] = payload_hash(index)
    write_json(index_artifact, index)
    return cfg_path, case_dir


def mutate_wrong_hash(_cfg, _freeze, _record, audit):
    audit["review_record_sha256"] = "0" * 64


def mutate_dirty(_cfg, _freeze, _record, audit):
    audit["dirty_diff_sha256"] = "1" * 64


def mutate_run_id(_cfg, _freeze, _record, audit):
    audit["run_id"] = "LBSCGP-G0-CODE-AUDIT-v4"


def mutate_path(_cfg, _freeze, record, audit):
    record["review_report_path"] = "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md"
    record["payload_sha256"] = payload_hash(record)
    audit["review_report_path"] = record["review_report_path"]


def consume(cfg_path):
    ledger = AccessLedger()
    cfg = load_config(str(cfg_path), ledger)
    return producer._load_freeze_and_audit(cfg, ledger)


def run_case(name, mutator, expect_ok):
    cfg_path, case_dir = build_fixture(name, mutator)
    try:
        freeze, audit = consume(cfg_path)
        ok = freeze["run_id"] == "LBSCGP-G0-FREEZE-v5" and \
            audit["run_id"] == "LBSCGP-G0-CODE-AUDIT-v5"
        error = None
    except Exception as exc:
        ok = False
        error = "{}: {}".format(type(exc).__name__, exc)
    return {
        "case": name,
        "expect_ok": expect_ok,
        "observed_ok": ok,
        "pass": ok is expect_ok,
        "error": error,
        "fixture_dir": root_relative_path(case_dir),
    }


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("regression must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected HateVideo")
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True)
    cases = [
        run_case("valid_v5_strict_schema", None, True),
        run_case("wrong_review_record_hash", mutate_wrong_hash, False),
        run_case("wrong_dirty_hash", mutate_dirty, False),
        run_case("wrong_code_audit_run_id", mutate_run_id, False),
        run_case("wrong_review_path", mutate_path, False),
    ]
    formal_v5_exists = (ROOT / "artifacts/lb_scgp/v5").exists()
    result = {
        "schema_version": 1,
        "slurm_job_id": JOB_ID,
        "status": "PASS" if all(item["pass"] for item in cases) and not formal_v5_exists else "FAIL",
        "cases": cases,
        "formal_v5_namespace_created": formal_v5_exists,
        "fixture_root": root_relative_path(RUN_ROOT),
        "fixture_root_removed": False,
        "result_path": root_relative_path(RESULT_PATH),
        "analogous_path_imports_exercised": [
            "lb_scgp_g0._load_freeze_and_audit",
            "lb_scgp_g0._verify_v4_code_audit_schema",
            "lb_scgp_common.git_state",
            "lb_scgp_common.dirty_state_policy",
            "lb_scgp_common.implementation_hash"
        ],
    }
    shutil.rmtree(RUN_ROOT)
    result["fixture_root_removed"] = not RUN_ROOT.exists()
    write_json(RESULT_PATH, result)
    print(canonical_json(result))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
