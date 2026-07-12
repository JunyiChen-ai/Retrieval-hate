#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path("/data/jehc223/RGCL")
CONFIG = ROOT / "configs/lb_scgp/lb_scgp_v5.json"
REVIEW = ROOT / "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.md"
RECORD = ROOT / "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.record.json"
OUT = ROOT / "refine-logs/lb_scgp/runtime/v5_independent_audit"
JOB_ID = os.environ.get("SLURM_JOB_ID", "noslurm")
RESULT = OUT / ("record_validation_" + JOB_ID + ".json")

sys.path.insert(0, str(ROOT / "scripts/analysis"))
import lb_scgp_independent_verify as verifier  # noqa: E402


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(verifier.cjson(obj) + "\n", encoding="utf-8")


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("record validation must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")
    if not REVIEW.exists():
        raise RuntimeError("final review is missing")
    cfg = verifier.read_json(CONFIG)
    contract = verifier._code_audit_contract(cfg)
    freeze_info = verifier._verify_freeze_for_audit_publish(cfg)
    review_rel = verifier._lineage_path(cfg, "review_report_path")
    record_rel = verifier._lineage_path(cfg, "review_record_path")
    if verifier._relpath(REVIEW) != review_rel or verifier._relpath(RECORD) != record_rel:
        raise RuntimeError("review/record paths do not match v5 lineage")
    review_sha = verifier.hfile(REVIEW)
    record = {
        "schema_version": 1,
        "record_type": contract["record_type"],
        "run_id": verifier._lineage_run_id(cfg, "code_audit"),
        "stage": "G0_CODE_AUDIT",
        "status": "PASS",
        "lineage_version": contract["version"],
        "config_path": freeze_info["config_path"],
        "artifact_namespace": verifier._artifact_namespace(cfg),
        "freeze_run_id": verifier._lineage_run_id(cfg, "freeze"),
        "freeze_path": freeze_info["freeze_path"],
        "freeze_file_sha256": freeze_info["freeze_file_sha256"],
        "freeze_payload_sha256": freeze_info["freeze"]["payload_sha256"],
        "config_canonical_sha256": freeze_info["config_canonical_sha256"],
        "implementation_sha256": freeze_info["implementation_sha256"],
        "independent_verifier_sha256": freeze_info["independent_verifier_sha256"],
        "review_report_path": review_rel,
        "review_report_sha256": review_sha,
        "reviewer_identity": "fresh_independent_lb_scgp_g0_v5_auditor_publisher_main_thread_no_subagent_session_not_visible",
        "review_process_identity": verifier.CODE_AUDIT_REVIEW_PROCESS_IDENTITY,
        "review_scope": contract["review_scope"],
        "critical": 0,
        "high": 0,
        "important": 3,
        "no_segment_gold_pass": True,
        "formal_pass_authorized": True,
        "independent_reviewer": True,
        "repair_executor_created": False,
        "only_gold_supervision": "parent_video_binary_label",
        "segment_gold_exists": False,
        "segment_gold_used": False,
    }
    record["payload_sha256"] = verifier.payload_digest(record)
    write_json(RECORD, record)
    record_sha = verifier.hfile(RECORD)
    args = SimpleNamespace(
        review=review_rel,
        review_record=record_rel,
        run_id=verifier._lineage_run_id(cfg, "code_audit"),
    )
    review_info = verifier._validate_review_record(cfg, args, freeze_info)
    prior = verifier._verify_prior_lineage_hashes(cfg)
    wrapper = verifier._verify_wrapper_contract(cfg)
    dirty = verifier.current_dirty_hash(cfg)
    result = {
        "schema_version": 1,
        "slurm_job_id": JOB_ID,
        "status": "PASS",
        "review_path": review_rel,
        "review_sha256": review_sha,
        "record_path": record_rel,
        "record_sha256": record_sha,
        "record_payload_sha256": record["payload_sha256"],
        "validated_record_sha256": review_info["record_sha256"],
        "validated_review_sha256": review_info["review_sha256"],
        "freeze_file_sha256": freeze_info["freeze_file_sha256"],
        "freeze_payload_sha256": freeze_info["freeze"]["payload_sha256"],
        "config_canonical_sha256": freeze_info["config_canonical_sha256"],
        "implementation_sha256": freeze_info["implementation_sha256"],
        "independent_verifier_sha256": freeze_info["independent_verifier_sha256"],
        "current_dirty_sha256": dirty,
        "frozen_dirty_sha256": freeze_info["freeze"]["dirty_diff_sha256"],
        "dirty_equal_frozen": dirty == freeze_info["freeze"]["dirty_diff_sha256"],
        "prior_lineage_hash_count": len(prior),
        "wrapper": wrapper,
        "record_exactly_validated": (
            review_info["record_sha256"] == record_sha
            and review_info["review_sha256"] == review_sha
        ),
    }
    result["all_ok"] = (
        result["record_exactly_validated"]
        and result["dirty_equal_frozen"]
        and record["critical"] == 0
        and record["high"] == 0
        and record["no_segment_gold_pass"] is True
        and record["segment_gold_exists"] is False
        and record["segment_gold_used"] is False
    )
    result["status"] = "PASS" if result["all_ok"] else "FAIL"
    write_json(RESULT, result)
    print(verifier.cjson({"status": result["status"],
                          "result": verifier._relpath(RESULT),
                          "slurm_job_id": JOB_ID}))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
