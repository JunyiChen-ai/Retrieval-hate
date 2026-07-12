#!/usr/bin/env python
"""Run1 contract freeze producer for LB-SCGP Global-R2 M0."""
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

from lb_scgp_global_r2_common import (  # noqa: E402
    CERT_SCHEMA_ID,
    CONTRACT_SCHEMA_ID,
    FORBIDDEN_ROUTE_NAMES,
    RUN1,
    TRI_OBSERVABLES,
    AccessLedger,
    ProjectionConfig,
    canonical_root_path,
    consensus_replicas,
    encode_certificate,
    exclusive_publish_json,
    factor_from_psd_gram,
    git_dirty_hash,
    global_projection_interface,
    implementation_hashes,
    old_protected_hash_manifest,
    payload_hash,
    procrustes_align,
    read_json,
    require_slurm_cpu,
    row_normalize,
    sha256_file,
    sha256_obj,
    structural_operator_summary,
)


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label} drift: expected {expected!r}, got {actual!r}")


def synthetic_certificate(state_shift: int = 0) -> dict[str, Any]:
    tri_states = ["supported", "contradicted", "unresolved"]
    record: dict[str, Any] = {"schema_version": CERT_SCHEMA_ID}
    for idx, field in enumerate(TRI_OBSERVABLES):
        record[field] = {"state": tri_states[(idx + state_shift) % len(tri_states)], "confidence": 0}
    record["modality_binding_observable"] = {"state": "multi_modal", "confidence": 0}
    record["parse_flags"] = []
    return record


def run_interface_fixture() -> dict[str, Any]:
    ids = ["train_a", "train_b", "train_c", "train_d"]
    labels = [0, 1, 0, 1]
    replicas = []
    for row in range(len(ids)):
        group = [synthetic_certificate(row % 3) for _ in range(4)]
        if row == 2:
            group[3] = synthetic_certificate(2)
        replicas.append(group)
    consensus = [consensus_replicas(group) for group in replicas]
    phi = np.stack([encode_certificate(record) for record in consensus], axis=0)
    v = row_normalize(phi)
    k_consensus = v @ v.T
    q = row_normalize(np.eye(len(ids), dtype=np.float64))
    q = q[:, : min(3, q.shape[1])]
    g0 = np.eye(len(ids), dtype=np.float64)
    g0[0, 1] = g0[1, 0] = 0.1
    g0[0, 2] = g0[2, 0] = 0.2
    g0[1, 3] = g0[3, 1] = 0.15
    operator = structural_operator_summary(q, g0, k_consensus)
    projection = global_projection_interface(
        n=len(ids),
        labels=labels,
        q_rank=int(q.shape[1]),
        cfg=ProjectionConfig(robust_enabled=False),
    )
    gram = np.eye(4, dtype=np.float64)
    y, rank_audit = factor_from_psd_gram(gram, d=4)
    if y is None:
        raise RuntimeError("synthetic factor interface unexpectedly failed")
    z_star, rotation, orth_resid = procrustes_align(y, np.eye(4, dtype=np.float64))
    return {
        "status": "PASS",
        "replica_count_per_video": 4,
        "synthetic_only": True,
        "mllm_calls": 0,
        "Phi_shape": [int(phi.shape[0]), int(phi.shape[1])],
        "feature_names_sha256": sha256_obj({"schema_id": CERT_SCHEMA_ID, "p": int(phi.shape[1])}),
        "K_C_diag_min": float(np.diag(k_consensus).min()),
        "K_C_diag_max": float(np.diag(k_consensus).max()),
        "structural_operator": operator,
        "projection_interface": projection,
        "rank_tail_audit": rank_audit,
        "factor_shape": [int(y.shape[0]), int(y.shape[1])],
        "procrustes": {
            "Zstar_shape": [int(z_star.shape[0]), int(z_star.shape[1])],
            "rotation_shape": [int(rotation.shape[0]), int(rotation.shape[1])],
            "orthogonality_residual": orth_resid,
        },
    }


def verify_machine_run(cfg: dict[str, Any], ledger: AccessLedger) -> dict[str, Any]:
    machine_path = cfg["paths"]["experiment_machine"]
    machine_hash = ledger.hash_file(machine_path, "machine_plan_hash", "authoritative_input")
    machine = read_json(machine_path)
    run = machine["runs"][0]
    assert_equal(machine["run_order"][0], RUN1, "machine run order[0]")
    assert_equal(run["run_id"], RUN1, "machine Run1 id")
    assert_equal(run["artifact_paths"], [cfg["run"]["artifact_path"]], "machine Run1 artifact path")
    assert_equal(run["artifact_schema_ids"], [CONTRACT_SCHEMA_ID], "machine Run1 schema")
    assert_equal(run["slurm"], cfg["run"]["slurm"], "machine Run1 slurm")
    if run["dependencies"]:
        raise RuntimeError("Run1 must not have dependencies")
    return {"machine_sha256": machine_hash, "machine_run_record": run}


def verify_hash_bindings(cfg: dict[str, Any], ledger: AccessLedger) -> dict[str, Any]:
    actual = {}
    for rel, expected in cfg["hash_bindings"]["authoritative_inputs"].items():
        digest = ledger.hash_file(rel, "authoritative_input_hash", "authoritative_input")
        assert_equal(digest, expected, f"hash {rel}")
        actual[rel] = digest
    train = {}
    for rel, expected in cfg["hash_bindings"]["train_bank_provenance_members"].items():
        digest = ledger.hash_file(rel, "train_bank_provenance_hash", "train_bank_provenance")
        assert_equal(digest, expected, f"train provenance hash {rel}")
        train[rel] = digest
    return {
        "authoritative_inputs": actual,
        "train_bank_provenance_members": train,
        "declared_validation_test_provenance_not_opened": cfg["hash_bindings"]["declared_validation_test_provenance_not_opened"],
    }


def verify_old_protected(cfg: dict[str, Any]) -> dict[str, Any]:
    digest, count = old_protected_hash_manifest()
    expected = cfg["hash_bindings"]["old_protected_pre_snapshot"]
    assert_equal(count, expected["path_count"], "old protected path count")
    assert_equal(digest, expected["manifest_sha256"], "old protected manifest")
    return {
        "current_manifest_sha256": digest,
        "current_path_count": count,
        "matched_preimplementation_snapshot": True,
        "snapshot_scope": expected["snapshot_scope"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--validation-json", required=True)
    args = parser.parse_args()

    require_slurm_cpu()
    assert_equal(args.run_id, RUN1, "authorized run id")

    ledger = AccessLedger()
    cfg_hash = ledger.hash_file(args.config, "config_hash", "authoritative_input")
    cfg = read_json(args.config)
    assert_equal(cfg["run"]["run_id"], RUN1, "config run id")
    assert_equal(cfg["run"]["schema_id"], CONTRACT_SCHEMA_ID, "config schema id")
    assert_equal(cfg["run"]["artifact_path"], "artifacts/lb_scgp_global/v1/m0/contract_freeze.json", "artifact path")
    assert_equal(cfg["authorization"]["authorized_run_ids"], [RUN1], "authorized run list")
    if any(cfg["authorization"][key] for key in [
        "mllm_calls_allowed",
        "ocr_calls_allowed",
        "performance_evaluation_allowed",
        "query_labels_allowed",
        "query_z_allowed",
        "run2_or_run3_allowed",
        "training_allowed",
    ]):
        raise RuntimeError("authorization must be fail-closed for Run1")
    assert_equal(cfg["supervision"]["only_gold_supervision"], "parent_video_binary_label", "gold supervision")
    assert_equal(cfg["supervision"]["segment_gold_exists"], False, "segment_gold_exists")
    assert_equal(cfg["supervision"]["segment_gold_used"], False, "segment_gold_used")

    artifact_path, _ = canonical_root_path(cfg["run"]["artifact_path"])
    if artifact_path.exists() or artifact_path.with_name(artifact_path.name + ".publish.lock").exists():
        raise FileExistsError("Run1 artifact or lock already exists")

    with open(args.validation_json, encoding="utf-8") as handle:
        validation = json.load(handle)
    assert_equal(validation["run_id"], RUN1, "validation run id")
    assert_equal(validation["status"], "PASS", "validation status")

    contract_schema_hash = ledger.hash_file(cfg["paths"]["contract_schema"], "contract_schema_hash", "schema")
    cert_schema_hash = ledger.hash_file(cfg["paths"]["cert_schema"], "cert_schema_hash", "schema")
    source_hashes = verify_hash_bindings(cfg, ledger)
    machine_summary = verify_machine_run(cfg, ledger)
    old_protected = verify_old_protected(cfg)
    interface_fixture = run_interface_fixture()
    impl_hash, impl_files = implementation_hashes(cfg["implementation_files"])

    slurm_script = cfg["paths"]["slurm_script"]
    wrapper = cfg["paths"]["wrapper"]
    exact_command = f"sbatch {slurm_script}"
    artifact: dict[str, Any] = {
        "artifact_schema_id": CONTRACT_SCHEMA_ID,
        "run_id": RUN1,
        "terminal_state": "FROZEN",
        "no_success_claim": True,
        "next_boundary": "fresh independent code+freeze audit before Run2 or Run3",
        "slurm_policy": {
            "required": True,
            "conda_env": "HateVideo",
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "cpu": cfg["run"]["slurm"]["cpu"],
            "ram_gb": cfg["run"]["slurm"]["ram_gb"],
            "gpu": cfg["run"]["slurm"]["gpu"],
            "no_time_flag": cfg["run"]["slurm"]["no_time_flag"],
        },
        "exact_run1_command": exact_command,
        "config_sha256": cfg_hash,
        "config_path": args.config,
        "source_hashes": source_hashes,
        "machine_plan": machine_summary,
        "schema_hashes": {
            cfg["paths"]["contract_schema"]: contract_schema_hash,
            cfg["paths"]["cert_schema"]: cert_schema_hash,
        },
        "implementation_sha256": impl_hash,
        "implementation_files": impl_files,
        "wrapper_hashes": {
            wrapper: sha256_file(wrapper),
            slurm_script: sha256_file(slurm_script),
        },
        "validator": validation,
        "old_protected_hash_comparison": old_protected,
        "contract": {
            "only_gold_supervision": "parent_video_binary_label",
            "segment_gold_exists": False,
            "segment_gold_used": False,
            "fragment_gold_exists": False,
            "segment_gold_used_for_selection": False,
            "allowed_run_ids": [RUN1],
            "frozen_comparator_name": "frozen_moving_strongest_same_protocol_non_mllm_comparator",
            "control_names": [
                "REMOVE",
                "SHUFFLE",
                "NOISE",
                "DIRECT-MOMENT",
                "DIRECT-CERT-FEATURE",
                "SCALAR-PROPENSITY",
            ],
            "validation_test_hash_policy": "recorded as plan provenance only; not opened by Run1",
        },
        "restricted_certificate_schema": {
            "schema_id": CERT_SCHEMA_ID,
            "extra_keys_free_text_targets_mechanisms_timestamps_spans_localization_verdicts_rationales_rejected": True,
            "confidence_role": "parse_agreement_diagnostic_not_weight_or_selector",
        },
        "global_r2_interfaces": {
            "replica_consensus_common_basis_Q_and_M_Q": interface_fixture,
            "projection": cfg["projection_contract"],
            "kkt_acceptance": {
                "only_go_path": "serialized H-metric normal-cone/KKT certificate",
                "finite_vi": "diagnostic_only",
                "psd_sign_convention": "v_psd=-S_psd",
            },
            "rank_tail": {
                "serializes_lambda_d": True,
                "serializes_lambda_dplus1": True,
                "serializes_positive_tail_mass": True,
                "serializes_tail_ratio": True,
                "serializes_negative_mass": True,
                "serializes_lambda_min": True,
                "serializes_reconstruction": True,
                "rank_failure_policy": "null_no_truncation_schema_tolerance_rescue",
            },
        },
        "forbidden_routes": {
            name: "rejected_fail_closed" for name in FORBIDDEN_ROUTE_NAMES
        },
        **ledger.fields(),
        **git_dirty_hash(),
    }
    artifact["payload_sha256"] = payload_hash(artifact)
    exclusive_publish_json(cfg["run"]["artifact_path"], artifact)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
