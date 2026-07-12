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
RESULT = ROOT / "refine-logs/lb_scgp/runtime/v4_audit_checks/final_review_record_validation.json"


sys.path.insert(0, str(ROOT / "scripts/analysis"))
import lb_scgp_independent_verify as verifier  # noqa: E402


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("validation must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")
    cfg = verifier.read_json(ROOT / CONFIG)
    freeze_info = verifier._verify_freeze_for_audit_publish(cfg)
    args = type("Args", (), {"review": REVIEW, "review_record": RECORD})()
    review_info = verifier._validate_review_record(cfg, args, freeze_info)
    result = {
        "status": "PASS",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "review_sha256": review_info["review_sha256"],
        "record_sha256": review_info["record_sha256"],
        "record_payload_sha256": review_info["record"]["payload_sha256"],
        "critical": review_info["record"]["critical"],
        "high": review_info["record"]["high"],
        "no_segment_gold_pass": review_info["record"]["no_segment_gold_pass"],
        "formal_pass_authorized": review_info["record"]["formal_pass_authorized"],
    }
    RESULT.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
