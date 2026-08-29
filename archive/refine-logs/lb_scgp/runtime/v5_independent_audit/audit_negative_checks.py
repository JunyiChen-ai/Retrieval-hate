#!/usr/bin/env python
from __future__ import annotations

import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import scipy
import torch

ROOT = Path("/data/jehc223/RGCL")
OUT = ROOT / "refine-logs/lb_scgp/runtime/v5_independent_audit"
JOB_ID = os.environ.get("SLURM_JOB_ID", "noslurm")
RUN_ROOT = OUT / ("fixture_" + JOB_ID)
RESULT_PATH = OUT / ("negative_checks_" + JOB_ID + ".json")

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
import lb_scgp_independent_verify as verifier  # noqa: E402


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical_json(obj) + "\n", encoding="utf-8")


def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def lock_bytes(run_id: str, rel: str) -> bytes:
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


def prior_rows_common(cfg):
    expected = cfg["lineage"]["prior_lineage_no_clobber_hashes"]
    return [{"path": path, "sha256": sha256_file(ROOT / path)}
            for path in sorted(expected)]


def base_fixture_config(case_dir: Path):
    cfg = json.loads((ROOT / "configs/lb_scgp/lb_scgp_v5.json").read_text(
        encoding="utf-8"))
    cfg["paths"]["artifacts"] = root_relative_path(case_dir / "artifacts")
    cfg_path = case_dir / "config.json"
    review_source = case_dir / "source_review.md"
    record_source = case_dir / "source_review_record.json"
    cfg["lineage"]["config_path"] = root_relative_path(cfg_path)
    cfg["lineage"]["review_report_path"] = root_relative_path(review_source)
    cfg["lineage"]["review_record_path"] = root_relative_path(record_source)
    dirty_paths = list(cfg["lineage"]["dirty_state_excluded_paths"])
    for rel in (root_relative_path(review_source), root_relative_path(record_source)):
        if rel not in dirty_paths:
            dirty_paths.append(rel)
    cfg["lineage"]["dirty_state_excluded_paths"] = dirty_paths
    cfg["lineage"]["mutable_records_excluded_from_freeze_inputs"] = dirty_paths
    return cfg, cfg_path, review_source, record_source


def build_fixture(name, mutator=None, late_mutator=None, index_mutator=None):
    case_dir = RUN_ROOT / name
    if case_dir.exists():
        shutil.rmtree(case_dir)
    case_dir.mkdir(parents=True)
    cfg, cfg_path, review_source, record_source = base_fixture_config(case_dir)
    write_json(cfg_path, cfg)

    load_ledger = AccessLedger()
    loaded = load_config(str(cfg_path), load_ledger)
    contract = code_audit_contract(loaded)
    artifacts = ROOT / loaded["paths"]["artifacts"]
    code_dir = artifacts / "g0/code_audit"
    code_dir.mkdir(parents=True)

    frozen_input = case_dir / "frozen_input.txt"
    write_text(frozen_input, "independent v5 strict-schema fixture\n")
    impl_hash, impl_rows = implementation_hash(loaded)
    verifier_sha = sha256_file(ROOT / "scripts/analysis/lb_scgp_independent_verify.py")
    head, dirty = git_state(cfg=loaded)
    artifact_excludes, dirty_paths, dirty_prefixes = dirty_state_policy(loaded)
    freeze_path = artifacts / "CONFIG_FREEZE.json"
    freeze_lock_path = freeze_path.with_name(freeze_path.name + ".publish.lock")
    freeze = {
        "schema_version": 1,
        "run_id": loaded["lineage"]["run_ids"]["freeze"],
        "stage": "G0_FREEZE",
        "status": "FROZEN",
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
        "freeze_name": loaded["lineage"]["freeze_name"],
        "lineage_version": "v5",
        "input_files": [
            {"path": root_relative_path(cfg_path), "sha256": sha256_file(cfg_path)},
            {"path": root_relative_path(frozen_input), "sha256": sha256_file(frozen_input)},
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

    review_text = "# Independent temporary v5 audit fixture\n\nNot a formal review.\n"
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
        "reviewer_identity": "independent_runtime_fixture",
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

    review_artifact = code_dir / "review.md"
    record_artifact = code_dir / "review_record.json"
    audit_artifact = code_dir / "audit.json"
    index_artifact = code_dir / "publication_index.json"
    write_text(review_artifact, review_text)
    review_output = {"path": root_relative_path(review_artifact),
                     "sha256": sha256_file(review_artifact)}
    wrapper = verifier._verify_wrapper_contract(loaded)
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
        "review_record_sha256": None,
        "review_record_payload_sha256": None,
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
        "prior_lineage_no_clobber_hashes": prior_rows_common(loaded),
        "audit_publish_wrapper": wrapper,
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
        "output_files": [review_output, None],
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "numpy_version": np.__version__,
        "scipy_version": scipy.__version__,
        "torch_version": torch.__version__,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    }
    if mutator:
        mutator(loaded, freeze, record, audit)
    record["payload_sha256"] = payload_hash(record)
    write_json(record_source, record)
    write_json(record_artifact, record)
    record_output = {"path": root_relative_path(record_artifact),
                     "sha256": sha256_file(record_artifact)}
    audit["review_record_sha256"] = record_output["sha256"]
    audit["review_record_payload_sha256"] = record["payload_sha256"]
    audit["output_files"] = [review_output, record_output]
    if late_mutator:
        late_mutator(loaded, freeze, record, audit)
    audit["payload_sha256"] = payload_hash(audit)
    write_json(audit_artifact, audit)

    output_names = ("review.md", "review_record.json", "audit.json", "publication_index.json")
    for output_name in output_names:
        rel = root_relative_path(code_dir / output_name)
        (code_dir / (output_name + ".publish.lock")).write_bytes(
            lock_bytes(loaded["lineage"]["run_ids"]["code_audit"], rel))
    index = {
        "schema_version": 1,
        "artifact_type": contract["index_type"],
        "run_id": loaded["lineage"]["run_ids"]["code_audit"],
        "stage": "G0_CODE_AUDIT",
        "status": "PASS",
        "output_files": [
            review_output,
            record_output,
            {"path": root_relative_path(audit_artifact),
             "sha256": sha256_file(audit_artifact)},
        ],
        "lock_files": [
            {"path": root_relative_path(code_dir / (output_name + ".publish.lock")),
             "sha256": sha256_file(code_dir / (output_name + ".publish.lock"))}
            for output_name in output_names
        ],
    }
    if index_mutator:
        index_mutator(index)
    index["payload_sha256"] = payload_hash(index)
    write_json(index_artifact, index)
    return cfg_path, case_dir


def mutate_wrong_record_hash(_cfg, _freeze, _record, audit):
    audit["review_record_sha256"] = "0" * 64


def mutate_wrong_dirty(_cfg, _freeze, _record, audit):
    audit["dirty_diff_sha256"] = "1" * 64


def mutate_wrong_run_id(_cfg, _freeze, _record, audit):
    audit["run_id"] = "LBSCGP-G0-CODE-AUDIT-v4"


def mutate_audit_extra(_cfg, _freeze, _record, audit):
    audit["unexpected_schema_drift"] = True


def mutate_record_extra(_cfg, _freeze, record, _audit):
    record["unexpected_schema_drift"] = True


def mutate_v4_fallback(_cfg, _freeze, record, audit):
    record["record_type"] = "LB_SCGP_G0_CODE_AUDIT_INDEPENDENT_REVIEW_RECORD_V4"
    audit["artifact_type"] = "LB_SCGP_G0_CODE_AUDIT_PASS_ARTIFACT_V4"
    audit["lineage_version"] = "v4"


def mutate_prior_hash(_cfg, _freeze, _record, audit):
    rows = copy.deepcopy(audit["prior_lineage_no_clobber_hashes"])
    rows[0]["sha256"] = "2" * 64
    audit["prior_lineage_no_clobber_hashes"] = rows


def mutate_segment_gold(_cfg, _freeze, record, audit):
    record["segment_gold_exists"] = True
    audit["segment_gold_exists"] = True


def mutate_path_index(index):
    index["output_files"][0]["path"] = "refine-logs/lb_scgp/runtime/v5_independent_audit/wrong_review.md"


def producer_consume(cfg_path: Path):
    ledger = AccessLedger()
    cfg = load_config(str(cfg_path), ledger)
    freeze, audit = producer._load_freeze_and_audit(cfg, ledger)
    return freeze, audit, ledger


def decision_consume(cfg_path: Path):
    cfg = verifier.read_json(cfg_path)
    artifacts = verifier._resolve_cfg_path(cfg, "artifacts")
    freeze = verifier.read_json(artifacts / "CONFIG_FREEZE.json")
    audit = verifier.read_json(artifacts / "g0/code_audit/audit.json")
    impl_hash, _ = verifier.current_implementation(cfg)
    return verifier.verify_code_audit_publication(
        cfg, artifacts, freeze, audit, verifier.hobj(cfg), impl_hash)


def run_producer_case(name, mutator=None, late_mutator=None, index_mutator=None,
                      expect_ok=False, decision_expected=None):
    cfg_path, case_dir = build_fixture(name, mutator, late_mutator, index_mutator)
    error = None
    observed_ok = False
    decision_ok = None
    decision_error = None
    try:
        freeze, audit, _ledger = producer_consume(cfg_path)
        observed_ok = (
            freeze["run_id"] == "LBSCGP-G0-FREEZE-v5"
            and audit["run_id"] == "LBSCGP-G0-CODE-AUDIT-v5"
        )
    except Exception as exc:
        error = "{}: {}".format(type(exc).__name__, exc)
    if decision_expected is not None:
        try:
            decision_ok = bool(decision_consume(cfg_path))
        except Exception as exc:
            decision_ok = False
            decision_error = "{}: {}".format(type(exc).__name__, exc)
    return {
        "case": name,
        "fixture_dir": root_relative_path(case_dir),
        "producer_expect_ok": expect_ok,
        "producer_observed_ok": observed_ok,
        "producer_pass": observed_ok is expect_ok,
        "producer_error": error,
        "decision_consumer_expect_ok": decision_expected,
        "decision_consumer_observed_ok": decision_ok,
        "decision_consumer_pass": (
            True if decision_expected is None else decision_ok is decision_expected
        ),
        "decision_consumer_error": decision_error,
    }


def static_import_and_compile_checks():
    targets = [
        "lb_scgp_common",
        "lb_scgp_g0",
        "lb_scgp_independent_verify",
        "lb_scgp_real_replay",
    ]
    imported = []
    for module_name in targets:
        importlib.import_module(module_name)
        imported.append(module_name)
    compiled = []
    for rel in [
            "scripts/analysis/lb_scgp_common.py",
            "scripts/analysis/lb_scgp_g0.py",
            "scripts/analysis/lb_scgp_independent_verify.py",
            "scripts/analysis/lb_scgp_real_replay.py",
            "scripts/slurm/lb_scgp_g0_audit_publish.sbatch",
            "scripts/slurm/lb_scgp_g0_cpu.sbatch",
            "scripts/slurm/lb_scgp_g0_gpu.sbatch"]:
        path = ROOT / rel
        if rel.endswith(".py"):
            compile(path.read_text(encoding="utf-8"), rel, "exec")
        compiled.append(rel)
    return {"imported": imported, "compiled": compiled}


def run_wrapper_negative(env_updates, expect_code, name):
    env = os.environ.copy()
    env.update(env_updates)
    proc = subprocess.run(
        ["bash", "scripts/slurm/lb_scgp_g0_audit_publish.sbatch"],
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=120,
    )
    return {
        "case": name,
        "returncode": proc.returncode,
        "expect_code": expect_code,
        "pass": proc.returncode == expect_code,
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
    }


def transaction_tests():
    tx_root = RUN_ROOT / "transaction"
    if tx_root.exists():
        shutil.rmtree(tx_root)
    tx_root.mkdir(parents=True)
    payloads = {
        "review.md": b"review\n",
        "review_record.json": b"{}\n",
        "audit.json": b"{}\n",
        "publication_index.json": b"{}\n",
    }
    locks = {name: b"lock\n" for name in payloads}
    existing = tx_root / "existing" / "code_audit"
    existing.mkdir(parents=True)
    existing_error = None
    try:
        verifier._transaction_publish_code_audit(existing, payloads, locks)
    except Exception as exc:
        existing_error = "{}: {}".format(type(exc).__name__, exc)
    existing_ok = existing_error is not None and existing.exists()

    fail_parent = tx_root / "forced_failure"
    fail_final = fail_parent / "code_audit"
    original_rename = verifier.os.rename

    def boom(_src, _dst):
        raise RuntimeError("forced transaction rename failure")

    forced_error = None
    try:
        verifier.os.rename = boom
        verifier._transaction_publish_code_audit(fail_final, payloads, locks)
    except Exception as exc:
        forced_error = "{}: {}".format(type(exc).__name__, exc)
    finally:
        verifier.os.rename = original_rename
    tmp_residue = sorted(p.name for p in fail_parent.glob(".code_audit.publish.tmp.*"))
    forced_ok = forced_error is not None and not fail_final.exists() and not tmp_residue

    positive = tx_root / "positive" / "code_audit"
    verifier._transaction_publish_code_audit(positive, payloads, locks)
    positive_files = sorted(p.name for p in positive.iterdir() if p.is_file())
    positive_ok = positive.exists() and set(positive_files) == set(
        list(payloads) + [name + ".publish.lock" for name in payloads])
    shutil.rmtree(tx_root)
    return {
        "existing_output_no_clobber": {
            "pass": existing_ok,
            "error": existing_error,
        },
        "forced_failure_cleanup": {
            "pass": forced_ok,
            "error": forced_error,
            "tmp_residue": tmp_residue,
        },
        "positive_atomic_temp_publish": {
            "pass": positive_ok,
            "files": positive_files,
        },
        "transaction_root_removed": not tx_root.exists(),
    }


def protected_hash_checks():
    cfg = verifier.read_json(ROOT / "configs/lb_scgp/lb_scgp_v5.json")
    prior_expected = cfg["lineage"]["prior_lineage_no_clobber_hashes"]
    prior_observed = {path: verifier.hfile(ROOT / path) for path in sorted(prior_expected)}
    v5 = {
        "artifacts/lb_scgp/v5/CONFIG_FREEZE.json":
            verifier.hfile(ROOT / "artifacts/lb_scgp/v5/CONFIG_FREEZE.json"),
        "artifacts/lb_scgp/v5/CONFIG_FREEZE.json.publish.lock":
            verifier.hfile(ROOT / "artifacts/lb_scgp/v5/CONFIG_FREEZE.json.publish.lock"),
    }
    return {
        "prior_ok": prior_observed == prior_expected,
        "prior_observed": prior_observed,
        "v5_freeze_observed": v5,
        "v5_freeze_expected": {
            "artifacts/lb_scgp/v5/CONFIG_FREEZE.json":
                "254e45afe9c0355892824c0c26bc73b4b0854cb20c67c3703982762fad010931",
            "artifacts/lb_scgp/v5/CONFIG_FREEZE.json.publish.lock":
                "54dc06d236d5fc1f3ac96400f1a81faeb1d2c0c8e5af075065d1964260de98a9",
        },
        "v5_ok": v5 == {
            "artifacts/lb_scgp/v5/CONFIG_FREEZE.json":
                "254e45afe9c0355892824c0c26bc73b4b0854cb20c67c3703982762fad010931",
            "artifacts/lb_scgp/v5/CONFIG_FREEZE.json.publish.lock":
                "54dc06d236d5fc1f3ac96400f1a81faeb1d2c0c8e5af075065d1964260de98a9",
        },
    }


def formal_v5_residue():
    code_dir = ROOT / "artifacts/lb_scgp/v5/g0/code_audit"
    tmp = sorted(p.as_posix() for p in (ROOT / "artifacts/lb_scgp/v5/g0").glob(
        ".code_audit.publish.tmp.*")) if (ROOT / "artifacts/lb_scgp/v5/g0").exists() else []
    files = sorted(p.relative_to(ROOT).as_posix()
                   for p in (ROOT / "artifacts/lb_scgp/v5").rglob("*")
                   if p.is_file())
    return {
        "code_audit_exists": code_dir.exists(),
        "tmp_paths": tmp,
        "files": files,
        "pre_publication_clean": (
            not code_dir.exists()
            and not tmp
            and files == [
                "artifacts/lb_scgp/v5/CONFIG_FREEZE.json",
                "artifacts/lb_scgp/v5/CONFIG_FREEZE.json.publish.lock",
            ]
        ),
    }


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("audit negative checks must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")
    if RUN_ROOT.exists():
        shutil.rmtree(RUN_ROOT)
    RUN_ROOT.mkdir(parents=True)
    try:
        static_checks = static_import_and_compile_checks()
        producer_cases = [
            run_producer_case("valid_v5_strict_schema", expect_ok=True),
            run_producer_case("wrong_review_record_hash", late_mutator=mutate_wrong_record_hash, expect_ok=False),
            run_producer_case("wrong_dirty_hash", mutate_wrong_dirty, expect_ok=False),
            run_producer_case("wrong_code_audit_run_id", mutate_wrong_run_id, expect_ok=False),
            run_producer_case("audit_schema_extra_field", mutate_audit_extra, expect_ok=False),
            run_producer_case("review_record_extra_field", mutate_record_extra, expect_ok=False),
            run_producer_case("v4_type_fallback_rejected", mutate_v4_fallback, expect_ok=False),
            run_producer_case("prior_lineage_hash_drift", mutate_prior_hash, expect_ok=False),
            run_producer_case("segment_gold_drift", mutate_segment_gold, expect_ok=False),
            run_producer_case("publication_index_path_drift", index_mutator=mutate_path_index, expect_ok=False),
        ]
        wrapper_cases = [
            run_wrapper_negative({
                "TASK": "freeze",
                "RUN_ID": "LBSCGP-G0-CODE-AUDIT-v5",
                "CONFIG": "configs/lb_scgp/lb_scgp_v5.json",
                "REVIEW": "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.md",
                "REVIEW_RECORD": "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.record.json",
            }, 2, "wrapper_wrong_task"),
            run_wrapper_negative({
                "TASK": "audit-publish",
                "RUN_ID": "LBSCGP-G0-CODE-AUDIT-v4",
                "CONFIG": "configs/lb_scgp/lb_scgp_v5.json",
                "REVIEW": "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.md",
                "REVIEW_RECORD": "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.record.json",
            }, 2, "wrapper_wrong_run_id"),
            run_wrapper_negative({
                "TASK": "audit-publish",
                "RUN_ID": "LBSCGP-G0-CODE-AUDIT-v5",
                "CONFIG": "configs/lb_scgp/lb_scgp_v5.json",
                "REVIEW": "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md",
                "REVIEW_RECORD": "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.record.json",
            }, 2, "wrapper_wrong_review_path"),
        ]
        tx = transaction_tests()
        hashes = protected_hash_checks()
        residue = formal_v5_residue()
    finally:
        if RUN_ROOT.exists():
            shutil.rmtree(RUN_ROOT)
    result = {
        "schema_version": 1,
        "slurm_job_id": JOB_ID,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "static_import_compile": static_checks,
        "producer_consumer_matrix": producer_cases,
        "wrapper_negative_matrix": wrapper_cases,
        "transaction_tests": tx,
        "protected_hash_checks": hashes,
        "formal_v5_residue": residue,
        "fixture_root": root_relative_path(RUN_ROOT),
        "fixture_root_removed": not RUN_ROOT.exists(),
    }
    result["status"] = "PASS" if (
        all(item["producer_pass"] and item["decision_consumer_pass"]
            for item in producer_cases)
        and all(item["pass"] for item in wrapper_cases)
        and all(item["pass"] for item in tx.values() if isinstance(item, dict))
        and tx["transaction_root_removed"] is True
        and hashes["prior_ok"] and hashes["v5_ok"]
        and residue["pre_publication_clean"]
        and result["fixture_root_removed"]
    ) else "FAIL"
    write_json(RESULT_PATH, result)
    print(canonical_json({"status": result["status"],
                          "result": root_relative_path(RESULT_PATH),
                          "slurm_job_id": JOB_ID}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
