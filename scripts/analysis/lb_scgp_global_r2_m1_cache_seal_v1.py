#!/usr/bin/env python
"""M1 Stage 3 - CPU cache seal decision.

Independently re-verifies BOTH dataset cache banks produced by Stage 2 and emits the
scgp_global_cache_seal_v1 decision.  For each dataset it: re-reads cache.jsonl, re-
validates every record against scgp_global_cache_replica_v2, recomputes the Merkle
root over the record leaves and checks it equals the producer manifest root, checks
call_count == 4 * unique_pack_count, checks all forbidden zero_counters are exactly 0,
and binds train_id_allowlist_sha256 / prompt_hash / input_builder_hash /
model_processor_hash / cache_merkle_root.  GO iff BOTH datasets verify.  Labels enter
only after this seal decision; this tool reads no label and does no training/kNN.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from lb_scgp_global_r2_m1_cache_v1_common import (  # noqa: E402
    DATASETS,
    REPLICAS,
    RUN_SEAL,
    SEAL_MANIFEST_SCHEMA_VERSION,
    SEAL_SCHEMA_ID,
    ZERO_COUNTER_KEYS,
    assert_equal,
    canonical_root_path,
    exclusive_publish_json,
    merkle_root,
    payload_hash,
    read_json,
    record_leaf_hash,
    require_slurm_seal,
    sha256_file,
    validate_against_schema,
    verify_machine_seal,
)


def verify_config(cfg: dict[str, Any]) -> None:
    assert_equal(cfg["run"]["run_id"], RUN_SEAL, "config run id")
    assert_equal(cfg["run"]["schema_id"], SEAL_SCHEMA_ID, "config schema id")
    for key in (
        "external_network_or_model_api_allowed",
        "ocr_calls_allowed",
        "label_read_allowed",
        "gpu_allowed",
        "mllm_inference_allowed",
        "validation_or_test_allowed",
        "training_allowed",
        "performance_evaluation_allowed",
    ):
        assert_equal(cfg["authorization"][key], False, f"authorization {key}")
    assert_equal(cfg["authorization"]["cache_artifact_read_allowed"], True, "cache_artifact_read_allowed")


def read_cache_records(rel: str, schema_rel: str, dataset: str) -> list[dict[str, Any]]:
    fs_path, _ = canonical_root_path(rel)
    records: list[dict[str, Any]] = []
    with open(fs_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            validate_against_schema(record, schema_rel, f"seal {dataset} record")
            records.append(record)
    return records


def verify_dataset(cfg: dict[str, Any], dataset: str) -> dict[str, Any]:
    entry = cfg["datasets"][dataset]
    schema_rel = cfg["paths"]["replica_schema"]
    records = read_cache_records(entry["cache_jsonl"], schema_rel, dataset)
    manifest = read_json(entry["cache_manifest"])

    assert_equal(manifest["dataset"], dataset, f"{dataset} manifest dataset")
    assert_equal(manifest["run_id"], entry["producer_run_id"], f"{dataset} manifest run id")
    assert_equal(len(records), manifest["record_count"], f"{dataset} record count")

    # recompute Merkle root independently
    leaves = [record_leaf_hash(r) for r in records]
    recomputed_root = merkle_root(leaves)
    root_ok = bool(recomputed_root == manifest["cache_merkle_root"])

    # call_count == 4 * unique_pack_count
    call_ok = bool(manifest["call_count"] == REPLICAS * manifest["unique_pack_count"])

    # every replica index present per video, exactly R replicas per video
    per_video: dict[str, set] = {}
    for record in records:
        per_video.setdefault(record["video_id"], set()).add(record["replica_index"])
    replicas_ok = all(indices == set(range(REPLICAS)) for indices in per_video.values())

    # forbidden zero counters all 0
    zc = manifest.get("zero_counters", {})
    zero_ok = all(int(zc.get(k, 1)) == 0 for k in ZERO_COUNTER_KEYS)

    # required provenance hashes present and non-empty
    hashes_ok = all(bool(manifest.get(k)) for k in ("train_id_allowlist_sha256", "prompt_hash",
                                                      "input_builder_hash", "model_processor_hash",
                                                      "cache_merkle_root"))

    # payload integrity of the manifest
    payload_ok = bool(manifest.get("payload_sha256") == payload_hash(manifest))

    verified = bool(root_ok and call_ok and replicas_ok and zero_ok and hashes_ok and payload_ok)
    # exactly the eight scgp_global_cache_seal_v1 fields, plus a diagnostics block
    return {
        "dataset": dataset,
        "train_id_allowlist_sha256": manifest["train_id_allowlist_sha256"],
        "cache_merkle_root": manifest["cache_merkle_root"],
        "prompt_hash": manifest["prompt_hash"],
        "input_builder_hash": manifest["input_builder_hash"],
        "model_processor_hash": manifest["model_processor_hash"],
        "call_count": manifest["call_count"],
        "zero_counters": {k: int(zc.get(k, 1)) for k in ZERO_COUNTER_KEYS},
        "seal_checks": {
            "merkle_root_recomputed_match": root_ok,
            "recomputed_merkle_root": recomputed_root,
            "call_count_equals_4_unique": call_ok,
            "unique_pack_count": manifest["unique_pack_count"],
            "record_count": len(records),
            "video_count": manifest["video_count"],
            "all_videos_have_R_replicas": replicas_ok,
            "zero_counters_all_zero": zero_ok,
            "provenance_hashes_present": hashes_ok,
            "manifest_payload_ok": payload_ok,
            "verified": verified,
            "cache_jsonl_sha256": sha256_file(canonical_root_path(entry["cache_jsonl"])[0]),
            "cache_manifest_sha256": sha256_file(canonical_root_path(entry["cache_manifest"])[0]),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    require_slurm_seal()
    assert_equal(args.run_id, RUN_SEAL, "authorized run id")
    cfg = read_json(args.config)
    verify_config(cfg)
    verify_machine_seal(cfg)

    for rel in (cfg["run"]["artifact_path"],):
        path, _ = canonical_root_path(rel)
        if path.exists() or path.with_name(path.name + ".publish.lock").exists():
            raise FileExistsError(f"M1 seal no-clobber refusal: {rel}")

    per_dataset = [verify_dataset(cfg, dataset) for dataset in DATASETS]
    go = all(entry["seal_checks"]["verified"] for entry in per_dataset)

    decision = {
        "schema_version": SEAL_MANIFEST_SCHEMA_VERSION,
        "artifact_schema_id": SEAL_SCHEMA_ID,
        "run_id": RUN_SEAL,
        "terminal_state": "CACHE_SEALED" if go else "CACHE_SEAL_STOP",
        "decision": "GO" if go else "STOP",
        "no_success_claim": True,
        "labels_enter_after_this_seal_only": True,
        "gate_rule": "GO iff BOTH dataset cache banks verify: independent Merkle root match, call_count==4*unique_pack_count, all forbidden zero_counters==0, exactly R replicas per video, provenance hashes present, manifest payload intact",
        "per_dataset": per_dataset,
        "gold_isolation": {
            "only_gold_supervision": "parent_video_binary_label",
            "train_labels_opened_by_seal": False,
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hashes": {
            "config_sha256": sha256_file(canonical_root_path(args.config)[0]),
            "replica_schema_sha256": sha256_file(canonical_root_path(cfg["paths"]["replica_schema"])[0]),
            "seal_schema_sha256": sha256_file(canonical_root_path(cfg["paths"]["seal_schema"])[0]),
        },
    }
    decision["payload_sha256"] = payload_hash(decision)
    validate_against_schema(decision, cfg["paths"]["seal_schema"], "seal decision")

    if not go:
        # publish the STOP decision (auditable) then fail-closed before the compiler.
        exclusive_publish_json(cfg["run"]["artifact_path"], decision)
        raise RuntimeError("M1 cache seal STOP: one or both dataset banks failed verification; STOP before compiler")

    exclusive_publish_json(cfg["run"]["artifact_path"], decision)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
