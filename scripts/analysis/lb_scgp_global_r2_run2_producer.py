#!/usr/bin/env python
"""Run2 synthetic KKT producer.

The producer emits a candidate payload only.  Final Run2 PASS is decided by
the separate independent verifier.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from lb_scgp_global_r2_run2_common import (  # noqa: E402
    PAYLOAD_SCHEMA_ID,
    RUN2,
    AccessLedger,
    assert_equal,
    build_replicas,
    build_source_manifest,
    canonical_json,
    canonical_root_path,
    encode_certificate,
    exclusive_publish_json,
    implementation_hashes,
    make_rank_failure_case,
    payload_hash,
    read_json,
    require_slurm_run2,
    schema_requires_no_additional_properties,
    sha256_bytes,
    sha256_file,
    sha256_obj,
    strip_private_case,
    structural_case_from_replicas,
    top_level_kkt_from_case,
    validate_certificate_record,
    verify_machine_run2,
)


def invalid_schema_fixtures() -> list[dict[str, Any]]:
    base = build_replicas(1, "single_flip")[0][0]
    extra = json.loads(canonical_json(base))
    extra["verdict"] = "hateful"
    missing = json.loads(canonical_json(base))
    missing.pop("parse_flags")
    invalid = json.loads(canonical_json(base))
    invalid["visual_reference_observable"]["confidence"] = 7
    fixtures = [
        ("extra_forbidden_field", extra),
        ("missing_required_field", missing),
        ("invalid_confidence", invalid),
    ]
    out = []
    for name, record in fixtures:
        try:
            validate_certificate_record(record)
            status = "UNEXPECTED_ACCEPT"
        except Exception as exc:  # noqa: BLE001 - serialized fixture result
            status = "REJECT"
            reason = str(exc)
        else:
            reason = ""
        out.append({"fixture": name, "status": status, "reason": reason})
    return out


def orth_cap_fixture_matrix(d: int) -> dict[str, Any]:
    cases = []
    specs = [
        ("below_cap", 4, "single_flip", 3),
        ("at_cap", 9, "single_flip", 8),
        ("above_cap", 12, "mixed", 8),
    ]
    for name, n, mode, expected_q in specs:
        labels = [idx % 2 for idx in range(n)]
        case = structural_case_from_replicas(
            case_id=f"ORTH_{name.upper()}",
            case_role=f"orth_cap_{name}",
            system="FULL",
            replicas=build_replicas(n, mode),
            labels=labels,
            d=d,
            mode="full",
        )
        raw = case["operator"]["raw_rank_before_cap"]
        q_rank = case["operator"]["q_rank"]
        if name == "above_cap":
            passed = raw > 8 and q_rank == expected_q
        else:
            passed = q_rank == expected_q
        cases.append(
            {
                "fixture": name,
                "raw_rank_before_cap": raw,
                "q_rank": q_rank,
                "expected_q_rank": expected_q,
                "status": "PASS" if passed else "FAIL",
            }
        )
    align_case = structural_case_from_replicas(
        case_id="ORTH_SIGN_ALIGNMENT",
        case_role="replica_sign_basis_alignment",
        system="FULL",
        replicas=build_replicas(9, "single_flip"),
        labels=[idx % 2 for idx in range(9)],
        d=d,
        mode="full",
    )
    sign_pass = align_case["operator"]["q_rank"] == 8 and align_case["operator"]["actual_orth_cap_executed"]
    cases.append(
        {
            "fixture": "sign_basis_alignment",
            "q_rank": align_case["operator"]["q_rank"],
            "canonical_orientation": "pivot_largest_abs_entry_positive_with_id_tiebreak",
            "status": "PASS" if sign_pass else "FAIL",
        }
    )
    return {
        "below_cap": next(item["status"] for item in cases if item["fixture"] == "below_cap"),
        "at_cap": next(item["status"] for item in cases if item["fixture"] == "at_cap"),
        "above_cap": next(item["status"] for item in cases if item["fixture"] == "above_cap"),
        "sign_basis_alignment": next(item["status"] for item in cases if item["fixture"] == "sign_basis_alignment"),
        "details": cases,
    }


def build_case_matrix(d: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    full = structural_case_from_replicas(
        case_id="FULL_SYNTH_KKT",
        case_role="primary_full_global_projection",
        system="FULL",
        replicas=build_replicas(10, "mixed"),
        labels=[idx % 2 for idx in range(10)],
        d=d,
        mode="full",
    )
    remove = structural_case_from_replicas(
        case_id="REMOVE_NULL_PARITY",
        case_role="remove_null_certified_parity",
        system="REMOVE",
        replicas=build_replicas(6, "unresolved"),
        labels=[idx % 2 for idx in range(6)],
        d=d,
        mode="remove_null",
    )
    shuffle = structural_case_from_replicas(
        case_id="SHUFFLE_SYNTH_CONTROL",
        case_role="identity_corruption_shuffle",
        system="SHUFFLE",
        replicas=build_replicas(10, "mixed"),
        labels=[idx % 2 for idx in range(10)],
        d=d,
        mode="shuffle",
    )
    noise = structural_case_from_replicas(
        case_id="NOISE_SYNTH_CONTROL",
        case_role="covariance_matched_noise_synthetic",
        system="NOISE",
        replicas=build_replicas(10, "mixed"),
        labels=[idx % 2 for idx in range(10)],
        d=d,
        mode="noise",
    )
    ambiguous = structural_case_from_replicas(
        case_id="AMBIGUOUS_COVERAGE_LOW",
        case_role="ambiguous_fail_open_geometry_fail_closed_claim",
        system="AMBIGUOUS",
        replicas=build_replicas(5, "single_flip"),
        labels=[0, 0, 0, 1, 1],
        d=d,
        mode="full",
    )
    robust = structural_case_from_replicas(
        case_id="ROBUST_COVERAGE_REPORTED",
        case_role="robust_coverage_report_safety_off_until_gate",
        system="ROBUST_COVERAGE",
        replicas=build_replicas(8, "single_flip"),
        labels=[0, 1, 0, 1, 0, 1, 0, 1],
        d=d,
        mode="full",
    )
    cases = [full, remove, shuffle, noise, ambiguous, robust]
    public_cases = [strip_private_case(case) for case in cases]
    status = "PASS" if all(case["kkt_status"] == "PASS" for case in public_cases) else "FAIL"
    return cases, {"status": status, "cases": public_cases}


def verify_config_and_schema(cfg: dict[str, Any]) -> None:
    assert_equal(cfg["run"]["run_id"], RUN2, "config run id")
    assert_equal(cfg["run"]["schema_id"], PAYLOAD_SCHEMA_ID, "config schema id")
    assert_equal(cfg["run"]["artifact_path"], "artifacts/lb_scgp_global/v1/m0/synth_kkt/manifest.json", "artifact path")
    assert_equal(cfg["authorization"]["authorized_run_ids"], [RUN2], "authorized run list")
    for key in [
        "mllm_calls_allowed",
        "ocr_calls_allowed",
        "gpu_allowed",
        "network_or_model_calls_allowed",
        "performance_evaluation_allowed",
        "query_labels_allowed",
        "query_z_allowed",
        "run3_or_later_allowed",
        "training_allowed",
    ]:
        assert_equal(cfg["authorization"][key], False, f"authorization {key}")
    for rel in [cfg["paths"]["payload_schema"], cfg["paths"]["case_schema"]]:
        schema = read_json(rel)
        errors = schema_requires_no_additional_properties(schema)
        if errors:
            raise RuntimeError(f"schema {rel} is not strict at {errors[:5]}")


def publish_children(cfg: dict[str, Any], source_manifest: dict[str, Any], access_fields: dict[str, Any]) -> tuple[str, str]:
    source_path = cfg["paths"]["source_manifest_path"]
    access_path = cfg["paths"]["access_ledger_path"]
    exclusive_publish_json(source_path, source_manifest)
    exclusive_publish_json(access_path, access_fields)
    return sha256_file(source_path), sha256_file(access_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-json", required=True)
    args = parser.parse_args()

    require_slurm_run2()
    assert_equal(args.run_id, RUN2, "authorized run id")

    cfg = read_json(args.config)
    verify_config_and_schema(cfg)
    artifact_path, _ = canonical_root_path(cfg["run"]["artifact_path"])
    for rel in [
        cfg["paths"]["artifact_path"],
        cfg["paths"]["source_manifest_path"],
        cfg["paths"]["access_ledger_path"],
        cfg["paths"]["semantic_verification_path"],
    ]:
        path, _ = canonical_root_path(rel)
        if path.exists() or path.with_name(path.name + ".publish.lock").exists():
            raise FileExistsError(f"Run2 no-clobber refusal: {rel}")

    with open(args.validation_json, encoding="utf-8") as handle:
        validation = json.load(handle)
    assert_equal(validation["run_id"], RUN2, "validation run id")
    assert_equal(validation["status"], "PASS", "validation status")

    ledger = AccessLedger()
    cfg_hash = ledger.hash_file(args.config, "config_hash", "schema_or_source")
    machine_summary = verify_machine_run2(cfg, ledger)
    source_manifest = build_source_manifest(cfg, ledger)
    access_fields = ledger.fields()
    source_manifest_sha256 = sha256_bytes((canonical_json(source_manifest) + "\n").encode("utf-8"))
    access_ledger_sha256 = sha256_bytes((canonical_json(access_fields) + "\n").encode("utf-8"))

    d = int(cfg["rank_factor_contract"]["d"])
    private_cases, public_case_matrix = build_case_matrix(d=d)
    orth_matrix = orth_cap_fixture_matrix(d=d)
    if any(value == "FAIL" for key, value in orth_matrix.items() if key != "details"):
        public_case_matrix["status"] = "FAIL"

    primary_case = private_cases[0]
    kkt = top_level_kkt_from_case(primary_case, cfg)
    unresolved_record = build_replicas(1, "unresolved")[0][0]
    unresolved_norm = float(np.linalg.norm(encode_certificate(unresolved_record)))
    impl_hash, _ = implementation_hashes(cfg["implementation_files"])
    run1_artifact_sha = sha256_file("artifacts/lb_scgp_global/v1/m0/contract_freeze.json")
    run1_lock_sha = sha256_file("artifacts/lb_scgp_global/v1/m0/contract_freeze.json.publish.lock")
    payload_schema_sha = sha256_file(cfg["paths"]["payload_schema"])
    case_matrix_sha = sha256_obj(public_case_matrix)
    operator_hash = primary_case["hashes"]["operator_hash"]
    zero_counters = {key: 0 for key in ledger.counters}
    manifest: dict[str, Any] = {
        "schema_version": "lb_scgp_global_r2_synth_kkt_manifest_v1",
        "artifact_schema_id": PAYLOAD_SCHEMA_ID,
        "run_id": RUN2,
        "terminal_state": "PRODUCED_PENDING_INDEPENDENT_VERIFY",
        "authorized_boundary": {
            "run_id": RUN2,
            "synthetic_only": True,
            "run3_or_later_locked": True,
        },
        "no_success_claim": True,
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
        **kkt,
        "movement_metrics": primary_case["movement_metrics"],
        "case_matrix": public_case_matrix,
        "orth_cap_matrix": {key: value for key, value in orth_matrix.items() if key != "details"},
        "rank_failure_probe": make_rank_failure_case(),
        "schema_fixture_results": {
            "invalid_schema": invalid_schema_fixtures(),
            "unresolved_values": {
                "state": "unresolved",
                "schema_status": "PASS",
                "encoded_row_norm_positive": bool(unresolved_norm > 0.0),
            },
        },
        "injection_results_expected": {
            "nan_overflow": "REJECT",
            "perturbed_artifact_source_operator_hash": "REJECT",
            "invalid_extra_missing_schema_fields": "REJECT",
            "wrong_dual_sign": "REJECT",
            "incomplete_cone_family": "REJECT",
            "forbidden_path": "REJECT",
            "rank_failure": "REJECT",
            "finite_vi_only_attempted_acceptance": "REJECT",
        },
        "gold_isolation": {
            "only_gold_supervision": "parent_video_binary_label",
            "segment_gold_exists": False,
            "segment_gold_used": False,
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
            "producer_status": "PASS_CANDIDATE" if public_case_matrix["status"] == "PASS" else "FAIL",
            "semantic_verifier_required": True,
            "acceptance_path": "serialized_h_metric_normal_cone_kkt",
            "finite_vi_can_accept": False,
        },
        "hashes": {
            "config_sha256": cfg_hash,
            "payload_schema_sha256": payload_schema_sha,
            "run1_artifact_sha256": run1_artifact_sha,
            "run1_lock_sha256": run1_lock_sha,
            "implementation_sha256": impl_hash,
            "operator_hash": operator_hash,
            "case_matrix_sha256": case_matrix_sha,
            "source_manifest_sha256": source_manifest_sha256,
            "access_ledger_sha256": access_ledger_sha256,
        },
    }
    manifest["payload_sha256"] = payload_hash(manifest)
    if manifest["acceptance"]["producer_status"] != "PASS_CANDIDATE":
        raise RuntimeError("producer refuses to publish failing candidate")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    exclusive_publish_json(cfg["paths"]["source_manifest_path"], source_manifest)
    exclusive_publish_json(cfg["paths"]["access_ledger_path"], access_fields)
    exclusive_publish_json(cfg["run"]["artifact_path"], manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
