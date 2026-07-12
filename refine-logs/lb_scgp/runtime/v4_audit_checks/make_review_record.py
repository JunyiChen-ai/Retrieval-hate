#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path("/data/jehc223/RGCL")
CONFIG = "configs/lb_scgp/lb_scgp_v4.json"
REVIEW = "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.md"
RECORD = "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V4.record.json"
RUN_ID = "LBSCGP-G0-CODE-AUDIT-v4"


sys.path.insert(0, str(ROOT / "scripts/analysis"))
import lb_scgp_independent_verify as verifier  # noqa: E402


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("record generation must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")
    review_path = ROOT / REVIEW
    record_path = ROOT / RECORD
    if not review_path.exists():
        raise RuntimeError("missing final review")
    if record_path.exists():
        raise RuntimeError("refusing to overwrite existing review record")
    cfg = verifier.read_json(ROOT / CONFIG)
    freeze_info = verifier._verify_freeze_for_audit_publish(cfg)
    review_bytes = review_path.read_bytes()
    review_sha = verifier.hbytes(review_bytes)
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
        "reviewer_identity": "sole_fresh_independent_gpt_5_5_xhigh_auditor_publisher",
        "review_process_identity": verifier.CODE_AUDIT_REVIEW_PROCESS_IDENTITY,
        "review_scope": verifier.CODE_AUDIT_REVIEW_SCOPE,
        "critical": 0,
        "high": 0,
        "important": 2,
        "no_segment_gold_pass": True,
        "formal_pass_authorized": True,
        "independent_reviewer": True,
        "repair_executor_created": False,
        "only_gold_supervision": "parent_video_binary_label",
        "segment_gold_exists": False,
        "segment_gold_used": False,
    }
    record["payload_sha256"] = verifier.payload_digest(record)
    record_path.write_text(verifier.cjson(record) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "RECORD_WRITTEN",
        "review_sha256": review_sha,
        "record_payload_sha256": record["payload_sha256"],
        "record_path": RECORD,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }, sort_keys=True))


if __name__ == "__main__":
    main()
