#!/usr/bin/env python
"""Realbank static/resource microbenchmark producer (candidate only).

Emits a PRODUCED_PENDING_INDEPENDENT_VERIFY candidate.  The final PASS is decided
by the separate independent verifier.  The producer publishes only a clean GO
candidate (peak RSS under cap, rank_eps<=d on every dataset, in-job replay hashes
match, every isolation injection REJECTED); any gate failure raises fail-closed
and the wrapper cleans up.  No training, no kNN, no performance/accuracy claim.
"""
from __future__ import annotations

import argparse
import os
import resource
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from lb_scgp_global_r2_realbank_resource_v2_common import (  # noqa: E402
    CAP_BYTES,
    CONFIG_PATH,
    DATASETS,
    M_SCALE,
    RANK_CAP,
    RUN1_ARTIFACT,
    RUN1_LOCK,
    RUN3,
    SCHEMA_ID,
    MANIFEST_SCHEMA_VERSION,
    RealbankAccessLedger,
    assert_equal,
    build_source_manifest,
    canonical_json,
    canonical_root_path,
    exclusive_publish_json,
    implementation_hashes,
    isolation_injection_cases,
    load_bank_features,
    payload_hash,
    read_json,
    require_slurm_realbank,
    run_dataset_pipeline,
    schema_requires_no_additional_properties,
    sha256_bytes,
    sha256_file,
    validate_manifest_against_schema,
    verify_machine_realbank,
)


def peak_rss_bytes() -> int:
    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


def verify_config_and_schema(cfg: dict[str, Any]) -> None:
    assert_equal(cfg["run"]["run_id"], RUN3, "config run id")
    assert_equal(cfg["run"]["schema_id"], SCHEMA_ID, "config schema id")
    assert_equal(cfg["run"]["artifact_path"], "artifacts/lb_scgp_global/v1/m0/realbank_resource/decision.json", "artifact path")
    assert_equal(cfg["authorization"]["authorized_run_ids"], [RUN3], "authorized run list")
    for key in [
        "mllm_calls_allowed",
        "ocr_calls_allowed",
        "gpu_allowed",
        "network_or_model_calls_allowed",
        "performance_evaluation_allowed",
        "query_labels_allowed",
        "query_z_allowed",
        "validation_or_test_allowed",
        "training_allowed",
        "m1_cache_or_later_allowed",
    ]:
        assert_equal(cfg["authorization"][key], False, f"authorization {key}")
    assert_equal(cfg["authorization"]["train_bank_read_allowed"], True, "train_bank_read_allowed")
    schema = read_json(cfg["paths"]["payload_schema"])
    errors = schema_requires_no_additional_properties(schema)
    if errors:
        raise RuntimeError(f"schema {cfg['paths']['payload_schema']} is not strict at {errors[:5]}")


def cleanup_created_outputs(paths: list[str]) -> None:
    for rel in paths:
        path, _ = canonical_root_path(rel)
        lock = path.with_name(path.name + ".publish.lock")
        for target in (path, lock):
            try:
                target.unlink()
            except FileNotFoundError:
                pass


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-json", required=True)
    args = parser.parse_args()

    require_slurm_realbank()
    assert_equal(args.run_id, RUN3, "authorized run id")

    cfg = read_json(args.config)
    assert_equal(args.config, CONFIG_PATH, "config path literal")
    verify_config_and_schema(cfg)
    verify_machine_realbank(cfg)

    for rel in [
        cfg["run"]["artifact_path"],
        cfg["paths"]["source_manifest_path"],
        cfg["paths"]["access_ledger_path"],
        cfg["paths"]["semantic_verification_path"],
    ]:
        path, _ = canonical_root_path(rel)
        if path.exists() or path.with_name(path.name + ".publish.lock").exists():
            raise FileExistsError(f"realbank no-clobber refusal: {rel}")

    validation = read_json(args.validation_json)
    assert_equal(validation["run_id"], RUN3, "validation run id")
    assert_equal(validation["status"], "PASS", "validation status")

    allowlist = {cfg["train_banks"][d]["path"]: cfg["train_banks"][d]["sha256"] for d in DATASETS}
    ledger = RealbankAccessLedger(allowlist)

    resource_rows: list[dict[str, Any]] = []
    rank_rows: list[dict[str, Any]] = []
    replay_rows: list[dict[str, Any]] = []
    coverage_rows: list[dict[str, Any]] = []
    b_struct_l2: list[float] = []
    for dataset in DATASETS:
        bank = cfg["train_banks"][dataset]
        fs_path = ledger.open_train_bank(dataset, bank["path"], bank["sha256"])
        features = load_bank_features(fs_path, dataset)
        run1 = run_dataset_pipeline(dataset, features)
        run2 = run_dataset_pipeline(dataset, features)
        peak_after = peak_rss_bytes()
        rank_audit = run1["rank_audit"]
        cov = dict(run1["coverage"])
        cov["dataset"] = dataset
        resource_rows.append(
            {
                "dataset": dataset,
                "n": run1["n"],
                "d": run1["d"],
                "q_rank": run1["q_rank"],
                "m_scale": run1["m_actual"],
                "peak_rss_bytes_after": peak_after,
            }
        )
        rank_rows.append(
            {
                "dataset": dataset,
                "d": run1["d"],
                "rank_eps": rank_audit["rank_eps"],
                "rank_le_d": run1["rank_le_d"],
                "lambda_d": rank_audit["lambda_d"],
                "lambda_dplus1": rank_audit["lambda_dplus1"],
                "positive_eigenmass": rank_audit["positive_eigenmass"],
                "negative_eigenmass": rank_audit["negative_eigenmass"],
                "tail_ratio": rank_audit["tail_ratio"],
                "reconstruction_residual": rank_audit["reconstruction_residual"],
                "status": rank_audit["status"],
            }
        )
        replay_rows.append(
            {
                "dataset": dataset,
                "replay_digest_run1": run1["replay_digest"],
                "replay_digest_run2": run2["replay_digest"],
                "match": bool(run1["replay_digest"] == run2["replay_digest"]),
            }
        )
        coverage_rows.append(cov)
        b_struct_l2.append(run1["b_struct_l2"])

    injections = isolation_injection_cases(allowlist)
    injections_all_reject = all(value == "REJECT" for value in injections.values())

    job_peak = peak_rss_bytes()
    within_cap = job_peak <= CAP_BYTES
    all_rank_le_d = all(row["rank_le_d"] and row["status"] == "PASS" for row in rank_rows)
    all_replay_match = all(row["match"] for row in replay_rows)
    go = bool(within_cap and all_rank_le_d and all_replay_match and injections_all_reject)
    if not go:
        raise RuntimeError(
            "realbank producer refuses to publish non-GO candidate: "
            f"within_cap={within_cap} all_rank_le_d={all_rank_le_d} "
            f"all_replay_match={all_replay_match} injections_all_reject={injections_all_reject}"
        )

    source_manifest = build_source_manifest(cfg, ledger)
    access_fields = ledger.fields()
    source_manifest_sha256 = sha256_bytes((canonical_json(source_manifest) + "\n").encode("utf-8"))
    access_ledger_sha256 = sha256_bytes((canonical_json(access_fields) + "\n").encode("utf-8"))
    impl_hash, _ = implementation_hashes(cfg["implementation_files"])

    zero_counters = dict(ledger.counters)
    manifest: dict[str, Any] = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_schema_id": SCHEMA_ID,
        "run_id": RUN3,
        "terminal_state": "PRODUCED_PENDING_INDEPENDENT_VERIFY",
        "no_success_claim": True,
        "decision": "GO",
        "authorized_boundary": {
            "run_id": RUN3,
            "train_bank_static_replay_only": True,
            "m1_cache_and_later_locked": True,
        },
        "slurm_policy": {
            "required": True,
            "conda_env": "HateVideo",
            "job_id": os.environ.get("SLURM_JOB_ID", ""),
            "cpu": cfg["run"]["slurm"]["cpu"],
            "ram_gb": cfg["run"]["slurm"]["ram_gb"],
            "gpu": cfg["run"]["slurm"]["gpu"],
            "no_time_flag": cfg["run"]["slurm"]["no_time_flag"],
        },
        "config_path": args.config,
        "source_manifest_path": cfg["paths"]["source_manifest_path"],
        "access_ledger_path": cfg["paths"]["access_ledger_path"],
        "resource_peak": {
            "job_peak_rss_bytes": job_peak,
            "cap_bytes": CAP_BYTES,
            "within_cap": within_cap,
            "measurement": "resource.getrusage(RUSAGE_SELF).ru_maxrss",
            "per_dataset": resource_rows,
        },
        "rank_tail": {"all_rank_le_d": all_rank_le_d, "per_dataset": rank_rows},
        "replay_hashes": {"all_match": all_replay_match, "per_dataset": replay_rows},
        "robust_coverage": {
            "fail_open": True,
            "robust_constraints_enabled": False,
            "per_dataset": coverage_rows,
        },
        "isolation_injection_results": {"all_reject": injections_all_reject, "cases": injections},
        "structural_placeholder": {
            "is_science": False,
            "policy": "synthetic_label_blind_placeholder_b_struct_opens_structural_code_path",
            "rank_cap_r": RANK_CAP,
            "m_scale": M_SCALE,
            "note": "b_struct = vech(M_Q(G0)) from a deterministic label-blind Phi seed; opens the orth_cap/structural-moment/adjoint path at real N; NON-SCIENCE placeholder; the real cache b_struct arrives at M1.",
            "per_dataset_b_struct_l2": b_struct_l2,
        },
        "allowed_reads": {
            "authorized_train_bank_read_count": ledger.authorized_train_bank_read_count,
            "banks": [dict(row) for row in ledger.banks],
        },
        "gold_isolation": {
            "only_gold_supervision": "parent_video_binary_label",
            "segment_gold_exists": False,
            "segment_gold_used": False,
            "train_labels_opened": False,
            "zero_counters": zero_counters,
        },
        "dirty_binding": {
            "run1_frozen_hashes_bound": True,
            "source_files_bound": True,
            "artifact_outputs_excluded_from_source_binding": True,
            "docs_tracker_post_run_changes_separately_measurable": True,
            "old_protected_manifest_sha256": source_manifest["source_rows"]["old_protected"]["manifest_sha256"],
            "relevant_tree_sha256": source_manifest["relevant_tree_sha256"],
        },
        "acceptance": {
            "producer_status": "PASS_CANDIDATE",
            "semantic_verifier_required": True,
            "decision_rule": "GO iff job_peak_rss_bytes<=96GiB AND rank_eps<=d (all datasets) AND in-job replay hash match (all datasets) AND all isolation injections REJECT",
        },
        "hashes": {
            "config_sha256": sha256_file(canonical_root_path(args.config)[0]),
            "schema_sha256": sha256_file(canonical_root_path(cfg["paths"]["payload_schema"])[0]),
            "implementation_sha256": impl_hash,
            "source_manifest_sha256": source_manifest_sha256,
            "access_ledger_sha256": access_ledger_sha256,
            "run1_artifact_sha256": sha256_file(canonical_root_path(RUN1_ARTIFACT)[0]),
            "run1_lock_sha256": sha256_file(canonical_root_path(RUN1_LOCK)[0]),
            "train_bank_MHC_sha256": cfg["train_banks"]["MHC"]["sha256"],
            "train_bank_MHC_zh_sha256": cfg["train_banks"]["MHC_zh"]["sha256"],
        },
    }
    manifest["payload_sha256"] = payload_hash(manifest)
    validate_manifest_against_schema(manifest, cfg["paths"]["payload_schema"])

    artifact_path, _ = canonical_root_path(cfg["run"]["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    created_outputs: list[str] = []
    try:
        exclusive_publish_json(cfg["paths"]["source_manifest_path"], source_manifest)
        created_outputs.append(cfg["paths"]["source_manifest_path"])
        exclusive_publish_json(cfg["paths"]["access_ledger_path"], access_fields)
        created_outputs.append(cfg["paths"]["access_ledger_path"])
        exclusive_publish_json(cfg["run"]["artifact_path"], manifest)
        created_outputs.append(cfg["run"]["artifact_path"])
    except Exception:
        cleanup_created_outputs(created_outputs)
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
