#!/usr/bin/env python
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path("/data/jehc223/RGCL")
CONFIG = "configs/lb_scgp/lb_scgp_v4.json"
RUN_ID = "LBSCGP-G0-CODE-AUDIT-v4"
REVIEW = "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md"
RECORD = "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.record.json"
RESULT = ROOT / "refine-logs/lb_scgp/runtime/v4_audit_checks/audit_publish_checks_result.json"


sys.path.insert(0, str(ROOT / "scripts/analysis"))
import lb_scgp_independent_verify as verifier  # noqa: E402


def write_bytes(rel: str, data: bytes) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def remove_path(rel: str) -> None:
    path = ROOT / rel
    if path.is_dir():
        shutil.rmtree(path)
    elif path.exists():
        path.unlink()


def clean_all() -> None:
    remove_path(REVIEW)
    remove_path(RECORD)
    remove_path("refine-logs/lb_scgp/G0_V4_DIRTY_DRIFT_SENTINEL.tmp")
    remove_path("artifacts/lb_scgp/v4/g0/code_audit")
    g0 = ROOT / "artifacts/lb_scgp/v4/g0"
    if g0.exists():
        for path in g0.glob(".code_audit.publish.tmp.*"):
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        try:
            g0.rmdir()
        except OSError:
            pass


def run_cmd(args, env=None):
    merged = os.environ.copy()
    if env:
        merged.update(env)
    proc = subprocess.run(
        args,
        cwd=ROOT,
        env=merged,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "cmd": args,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-1000:],
        "stderr_tail": proc.stderr[-1000:],
    }


def publisher_cmd(review=REVIEW, record=RECORD, config=CONFIG, run_id=RUN_ID):
    return [
        sys.executable,
        "scripts/analysis/lb_scgp_independent_verify.py",
        "--config",
        config,
        "--task",
        "audit-publish",
        "--run-id",
        run_id,
        "--review",
        review,
        "--review-record",
        record,
    ]


def valid_review_text() -> bytes:
    text = "\n".join(
        [
            "# Temporary LB-SCGP G0 v4 Audit-Publish Harness Review",
            "",
            "This file is a temporary negative-test fixture created inside a SLURM audit job.",
            "It is not the final independent review and must be removed before publication.",
            "",
        ]
    )
    return text.encode("utf-8")


def build_record(review_sha: str, overrides=None, extra=None):
    cfg = verifier.read_json(ROOT / CONFIG)
    freeze_info = verifier._verify_freeze_for_audit_publish(cfg)
    record = {
        "schema_version": 1,
        "record_type": verifier.CODE_AUDIT_RECORD_TYPE,
        "run_id": RUN_ID,
        "stage": "G0_CODE_AUDIT",
        "status": "PASS",
        "lineage_version": "v4",
        "config_path": freeze_info["config_path"],
        "artifact_namespace": cfg["lineage"]["artifact_namespace"],
        "freeze_run_id": cfg["lineage"]["run_ids"]["freeze"],
        "freeze_path": freeze_info["freeze_path"],
        "freeze_file_sha256": freeze_info["freeze_file_sha256"],
        "freeze_payload_sha256": freeze_info["freeze"]["payload_sha256"],
        "config_canonical_sha256": freeze_info["config_canonical_sha256"],
        "implementation_sha256": freeze_info["implementation_sha256"],
        "independent_verifier_sha256": freeze_info["independent_verifier_sha256"],
        "review_report_path": REVIEW,
        "review_report_sha256": review_sha,
        "reviewer_identity": "temporary_v4_audit_harness",
        "review_process_identity": verifier.CODE_AUDIT_REVIEW_PROCESS_IDENTITY,
        "review_scope": verifier.CODE_AUDIT_REVIEW_SCOPE,
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
    if overrides:
        record.update(overrides)
    if extra:
        record.update(extra)
    record["payload_sha256"] = verifier.payload_digest(record)
    return record


def write_record(record) -> None:
    write_bytes(RECORD, (verifier.cjson(record) + "\n").encode("utf-8"))


def prepare_valid_pair(overrides=None, extra=None):
    review = valid_review_text()
    write_bytes(REVIEW, review)
    record = build_record(verifier.hbytes(review), overrides=overrides, extra=extra)
    write_record(record)
    return record


def expect_failure(name, setup, allow_preexisting_residue=False):
    clean_all()
    setup()
    result = run_cmd(publisher_cmd())
    residue = formal_residue()
    clean_all()
    residue_ok = residue["tmp_absent"] and (
        allow_preexisting_residue or residue["code_audit_absent"])
    return {
        "name": name,
        "expected": "failure_no_formal_residue",
        "passed": result["returncode"] != 0 and residue_ok and
        formal_residue()["code_audit_absent"] and formal_residue()["tmp_absent"],
        "result": result,
        "residue_before_cleanup": residue,
        "residue_after_cleanup": formal_residue(),
    }


def formal_residue():
    code_dir = ROOT / "artifacts/lb_scgp/v4/g0/code_audit"
    g0 = ROOT / "artifacts/lb_scgp/v4/g0"
    tmp = []
    if g0.exists():
        tmp = [p.as_posix() for p in g0.glob(".code_audit.publish.tmp.*")]
    return {
        "review_source_exists": (ROOT / REVIEW).exists(),
        "record_source_exists": (ROOT / RECORD).exists(),
        "code_audit_absent": not code_dir.exists(),
        "tmp_absent": len(tmp) == 0,
        "tmp_paths": tmp,
    }


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("audit checks must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")
    if (ROOT / REVIEW).exists() or (ROOT / RECORD).exists():
        raise RuntimeError("refusing to overwrite pre-existing review or record")
    if (ROOT / "artifacts/lb_scgp/v4/g0/code_audit").exists():
        raise RuntimeError("refusing to run with pre-existing code_audit directory")

    clean_all()
    checks = []
    review = valid_review_text()
    review_sha = verifier.hbytes(review)

    checks.append(expect_failure("missing_review_report", lambda: write_record(build_record(review_sha))))
    checks.append(expect_failure("missing_review_record", lambda: write_bytes(REVIEW, review)))
    checks.append(expect_failure("malformed_review_record_json", lambda: (write_bytes(REVIEW, review), write_bytes(RECORD, b"{not-json\n"))))
    checks.append(expect_failure("extra_review_record_field", lambda: prepare_valid_pair(extra={"unexpected": "field"})))
    checks.append(expect_failure("wrong_review_report_hash", lambda: prepare_valid_pair({"review_report_sha256": "0" * 64})))
    checks.append(expect_failure("wrong_record_run_id", lambda: prepare_valid_pair({"run_id": "WRONG"})))
    checks.append(expect_failure("wrong_config_path_in_record", lambda: prepare_valid_pair({"config_path": "configs/lb_scgp/lb_scgp_v3.json"})))
    checks.append(expect_failure("wrong_freeze_file_sha256", lambda: prepare_valid_pair({"freeze_file_sha256": "0" * 64})))
    checks.append(expect_failure("wrong_implementation_sha256", lambda: prepare_valid_pair({"implementation_sha256": "0" * 64})))
    checks.append(expect_failure("nonzero_critical", lambda: prepare_valid_pair({"critical": 1})))
    checks.append(expect_failure("nonzero_high", lambda: prepare_valid_pair({"high": 1})))
    checks.append(expect_failure("no_segment_gold_false", lambda: prepare_valid_pair({"no_segment_gold_pass": False})))
    checks.append(expect_failure("segment_gold_fields_true", lambda: prepare_valid_pair({"segment_gold_exists": True, "segment_gold_used": True})))
    checks.append(expect_failure("formal_pass_not_authorized", lambda: prepare_valid_pair({"formal_pass_authorized": False})))
    checks.append(expect_failure("repair_executor_claimed", lambda: prepare_valid_pair({"repair_executor_created": True})))
    checks.append(expect_failure("wrong_review_process_identity", lambda: prepare_valid_pair({"review_process_identity": "prior_repair_executor"})))

    def dirty_drift():
        prepare_valid_pair()
        write_bytes("refine-logs/lb_scgp/G0_V4_DIRTY_DRIFT_SENTINEL.tmp", b"dirty drift\n")

    checks.append(expect_failure("dirty_state_drift", dirty_drift))

    def preexisting_code_dir():
        prepare_valid_pair()
        code_dir = ROOT / "artifacts/lb_scgp/v4/g0/code_audit"
        code_dir.mkdir(parents=True, exist_ok=True)
        (code_dir / "foreign.txt").write_text("foreign\n", encoding="utf-8")
        (code_dir / "audit.json.publish.lock").write_text("foreign lock\n", encoding="utf-8")

    checks.append(expect_failure(
        "preexisting_code_audit_dir_and_lock",
        preexisting_code_dir,
        allow_preexisting_residue=True,
    ))

    clean_all()
    write_bytes(REVIEW, review)
    valid_record = build_record(review_sha)
    write_record(valid_record)
    cfg = verifier.read_json(ROOT / CONFIG)
    freeze_info = verifier._verify_freeze_for_audit_publish(cfg)
    valid_info = verifier._validate_review_record(
        cfg,
        type("Args", (), {"review": REVIEW, "review_record": RECORD})(),
        freeze_info,
    )
    prior = verifier._verify_prior_lineage_hashes(cfg)
    wrapper = verifier._verify_wrapper_contract(cfg)
    positive = {
        "name": "positive_prepublication_validation_without_transaction",
        "expected": "validation_pass_no_publication",
        "passed": valid_info["record"]["payload_sha256"] == valid_record["payload_sha256"]
        and len(prior) == 6
        and wrapper["path"] == "scripts/slurm/lb_scgp_g0_audit_publish.sbatch"
        and formal_residue()["code_audit_absent"],
        "record_payload_sha256": valid_record["payload_sha256"],
    }
    clean_all()

    wrapper_checks = []
    wrapper_checks.append({
        "name": "wrapper_wrong_task",
        "expected": "failure_before_python",
        "result": run_cmd(
            ["bash", "scripts/slurm/lb_scgp_g0_audit_publish.sbatch"],
            {
                "TASK": "freeze",
                "RUN_ID": RUN_ID,
                "CONFIG": CONFIG,
                "REVIEW": REVIEW,
                "REVIEW_RECORD": RECORD,
            },
        ),
    })
    wrapper_checks.append({
        "name": "wrapper_wrong_run_id",
        "expected": "failure_before_python",
        "result": run_cmd(
            ["bash", "scripts/slurm/lb_scgp_g0_audit_publish.sbatch"],
            {
                "TASK": "audit-publish",
                "RUN_ID": "WRONG",
                "CONFIG": CONFIG,
                "REVIEW": REVIEW,
                "REVIEW_RECORD": RECORD,
            },
        ),
    })
    wrapper_checks.append({
        "name": "wrapper_wrong_review_path",
        "expected": "failure_before_python",
        "result": run_cmd(
            ["bash", "scripts/slurm/lb_scgp_g0_audit_publish.sbatch"],
            {
                "TASK": "audit-publish",
                "RUN_ID": RUN_ID,
                "CONFIG": CONFIG,
                "REVIEW": "refine-logs/lb_scgp/runtime/v4_audit_checks/not_the_review.md",
                "REVIEW_RECORD": RECORD,
            },
        ),
    })
    wrapper_checks.append({
        "name": "wrapper_wrong_config_path",
        "expected": "failure_before_python",
        "result": run_cmd(
            ["bash", "scripts/slurm/lb_scgp_g0_audit_publish.sbatch"],
            {
                "TASK": "audit-publish",
                "RUN_ID": RUN_ID,
                "CONFIG": "configs/lb_scgp/lb_scgp_v3.json",
                "REVIEW": REVIEW,
                "REVIEW_RECORD": RECORD,
            },
        ),
    })
    for row in wrapper_checks:
        row["passed"] = row["result"]["returncode"] != 0

    direct_wrong_path = run_cmd(
        publisher_cmd(
            review="refine-logs/lb_scgp/runtime/v4_audit_checks/not_the_review.md",
            record=RECORD,
        )
    )
    direct_wrong_config = run_cmd(publisher_cmd(config="configs/lb_scgp/lb_scgp_v3.json"))

    summary = {
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "python": sys.version,
        "negative_checks": checks,
        "positive_check": positive,
        "wrapper_checks": wrapper_checks,
        "direct_wrong_review_path": {
            "expected": "failure",
            "passed": direct_wrong_path["returncode"] != 0,
            "result": direct_wrong_path,
        },
        "direct_wrong_config_path": {
            "expected": "failure",
            "passed": direct_wrong_config["returncode"] != 0,
            "result": direct_wrong_config,
        },
        "final_residue": formal_residue(),
    }
    summary["all_passed"] = (
        all(row["passed"] for row in checks)
        and positive["passed"]
        and all(row["passed"] for row in wrapper_checks)
        and summary["direct_wrong_review_path"]["passed"]
        and summary["direct_wrong_config_path"]["passed"]
        and summary["final_residue"]["code_audit_absent"]
        and summary["final_residue"]["tmp_absent"]
        and not summary["final_residue"]["review_source_exists"]
        and not summary["final_residue"]["record_source_exists"]
    )
    RESULT.parent.mkdir(parents=True, exist_ok=True)
    RESULT.write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS" if summary["all_passed"] else "FAIL",
                      "result": str(RESULT),
                      "slurm_job_id": os.environ.get("SLURM_JOB_ID")}, sort_keys=True))
    if not summary["all_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
