#!/usr/bin/env python
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/RGCL")
CONFIG = "configs/lb_scgp/lb_scgp_v5.json"
REVIEW = "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.md"
RECORD = "refine-logs/lb_scgp/G0_FORMAL_CODE_AUDIT_REVIEW_V5.record.json"
OUT = ROOT / "refine-logs/lb_scgp/runtime/v5_independent_audit"
JOB_ID = os.environ.get("SLURM_JOB_ID", "noslurm")
RESULT = OUT / ("post_publication_verification_" + JOB_ID + ".json")

sys.path.insert(0, str(ROOT / "scripts/analysis"))
import lb_scgp_common as common  # noqa: E402
import lb_scgp_g0 as producer  # noqa: E402
import lb_scgp_independent_verify as verifier  # noqa: E402


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8")


def main():
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("post-publication verification must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda environment HateVideo")

    cfg = verifier.read_json(ROOT / CONFIG)
    artifacts = verifier._resolve_cfg_path(cfg, "artifacts")
    code_dir = artifacts / verifier.CODE_AUDIT_DIR
    freeze = verifier.read_json(artifacts / "CONFIG_FREEZE.json")
    audit = verifier.read_json(code_dir / "audit.json")
    index = verifier.read_json(code_dir / "publication_index.json")
    record = verifier.read_json(code_dir / "review_record.json")
    impl_hash, _ = verifier.current_implementation(cfg)
    config_hash = verifier.hobj(cfg)
    current_dirty = verifier.current_dirty_hash(cfg)

    decision_error = None
    try:
        decision_consumer_ok = verifier.verify_code_audit_publication(
            cfg, artifacts, freeze, audit, config_hash, impl_hash)
    except Exception as exc:
        decision_consumer_ok = False
        decision_error = "{}: {}".format(type(exc).__name__, exc)

    ledger = common.AccessLedger()
    producer_error = None
    try:
        producer_cfg = common.load_config(ROOT / CONFIG, ledger)
        producer_freeze, producer_audit = producer._load_freeze_and_audit(
            producer_cfg, ledger)
        producer_consumer_ok = (
            producer_freeze["payload_sha256"] == freeze["payload_sha256"]
            and producer_audit["payload_sha256"] == audit["payload_sha256"]
        )
    except Exception as exc:
        producer_consumer_ok = False
        producer_error = "{}: {}".format(type(exc).__name__, exc)

    expected_formal = {
        "review.md",
        "review.md.publish.lock",
        "review_record.json",
        "review_record.json.publish.lock",
        "audit.json",
        "audit.json.publish.lock",
        "publication_index.json",
        "publication_index.json.publish.lock",
    }
    observed_formal = {p.name for p in code_dir.iterdir() if p.is_file()}
    tmp_paths = [p.relative_to(ROOT).as_posix()
                 for p in (artifacts / "g0").glob(".code_audit.publish.tmp.*")]
    source_review_sha = verifier.hfile(ROOT / REVIEW)
    source_record_sha = verifier.hfile(ROOT / RECORD)
    published_review_sha = verifier.hfile(code_dir / "review.md")
    published_record_sha = verifier.hfile(code_dir / "review_record.json")
    audit_sha = verifier.hfile(code_dir / "audit.json")
    index_sha = verifier.hfile(code_dir / "publication_index.json")
    lock_hashes = {
        row["path"]: verifier.hfile(ROOT / row["path"])
        for row in index.get("lock_files", [])
    }
    zero_keys = [
        "mllm_call_count",
        "ocr_call_count",
        "teacher_cache_read_count",
        "teacher_cache_write_count",
        "outer_held_label_read_count",
        "outer_held_content_read_count",
        "val_content_read_count",
        "test_content_read_count",
        "val_test_teacher_artifact_count",
        "formal_model_optimizer_evaluator_outer_held_read_count",
    ]
    zero_counters = {key: int(audit.get(key, -1)) for key in zero_keys}
    prior_hashes = verifier._verify_prior_lineage_hashes(cfg)
    result = {
        "schema_version": 1,
        "slurm_job_id": JOB_ID,
        "config_path": CONFIG,
        "artifact_namespace": verifier._relpath(artifacts),
        "producer_consumer_ok": producer_consumer_ok,
        "producer_consumer_error": producer_error,
        "decision_consumer_ok": decision_consumer_ok,
        "decision_consumer_error": decision_error,
        "source_review_sha256": source_review_sha,
        "published_review_sha256": published_review_sha,
        "source_record_sha256": source_record_sha,
        "published_record_sha256": published_record_sha,
        "audit_sha256": audit_sha,
        "audit_payload_sha256": audit.get("payload_sha256"),
        "publication_index_sha256": index_sha,
        "publication_index_payload_sha256": index.get("payload_sha256"),
        "record_payload_sha256": record.get("payload_sha256"),
        "lock_hashes": lock_hashes,
        "output_files": index.get("output_files"),
        "prior_lineage_no_clobber_hashes": prior_hashes,
        "dirty_binding": {
            "audit_dirty": audit.get("dirty_diff_sha256"),
            "freeze_dirty": freeze.get("dirty_diff_sha256"),
            "current_dirty": current_dirty,
        },
        "dirty_equal_frozen": (
            audit.get("dirty_diff_sha256") == freeze.get("dirty_diff_sha256")
            and current_dirty == freeze.get("dirty_diff_sha256")
        ),
        "source_copy_ok": (
            source_review_sha == published_review_sha
            and source_record_sha == published_record_sha
        ),
        "schema_payload_ok": (
            verifier.verify_payload(audit)
            and verifier.verify_payload(index)
            and record.get("payload_sha256") == verifier.payload_digest(record)
        ),
        "zero_counters": zero_counters,
        "zero_counters_ok": all(value == 0 for value in zero_counters.values()),
        "no_segment_gold_ok": (
            audit.get("only_gold_supervision") == "parent_video_binary_label"
            and audit.get("segment_gold_exists") is False
            and audit.get("segment_gold_used") is False
            and record.get("only_gold_supervision") == "parent_video_binary_label"
            and record.get("segment_gold_exists") is False
            and record.get("segment_gold_used") is False
        ),
        "formal_file_set_ok": observed_formal == expected_formal,
        "tmp_paths": tmp_paths,
        "tmp_absent": len(tmp_paths) == 0,
        "producer_access_ledger_sha256": common.sha256_obj(ledger.records),
        "producer_access_ledger": ledger.records,
    }
    result["all_ok"] = (
        result["producer_consumer_ok"]
        and result["decision_consumer_ok"]
        and result["source_copy_ok"]
        and result["schema_payload_ok"]
        and result["dirty_equal_frozen"]
        and result["zero_counters_ok"]
        and result["no_segment_gold_ok"]
        and result["formal_file_set_ok"]
        and result["tmp_absent"]
    )
    result["status"] = "PASS" if result["all_ok"] else "FAIL"
    write_json(RESULT, result)
    print(json.dumps({"status": result["status"],
                      "result": verifier._relpath(RESULT),
                      "slurm_job_id": JOB_ID},
                     sort_keys=True, separators=(",", ":")))
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
