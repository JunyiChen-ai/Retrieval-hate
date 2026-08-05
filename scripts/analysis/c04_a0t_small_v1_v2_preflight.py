#!/usr/bin/env python
"""CPU preflight/freeze for the prospective C04-A0T-SMALL-v1 tranche.

This program never loads a model, decodes a frame, reads a label value, or
touches development/test paths.  It materializes the ID-only 200+200 allowlists,
source-content hashes, deterministic role/JL payloads, access ledger, empty GPU
ledger, and a one-use resource ticket.  It does not submit a job.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from c04_a0t_small_v1_v2_common import (  # noqa: E402
    ADDITIVE_INPUT_DIM,
    ADDITIVE_TAG,
    ARTIFACT_ROOT,
    CONFIG_RELATIVE,
    DATASETS,
    LE3_INPUT_DIM,
    LE3_TAG,
    ROLE_DIM,
    RUN_ID,
    SCHEMA_VERSION,
    SELECT_N,
    config_contract_sha256,
    dense_rademacher_payload,
    exclusive_publish_bytes,
    exclusive_publish_json,
    load_json,
    materialize_role_map,
    merkle_root,
    project_train_asr_line,
    prompt_hashes,
    model_hash_closure,
    require_exact_keys,
    root_path,
    selection_digest,
    self_test_fixtures,
    sha256_bytes,
    sha256_file,
    sha256_obj,
    source_hash_closure,
    train_asr_path,
    validate_schema,
    verify_bound_file_map,
    verify_closure_hash,
    video_path,
)


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


class AccessAudit:
    """Runtime evidence for every data/video open controlled by this program."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def train_asr(self, cfg: dict[str, Any], dataset: str) -> Path:
        path = train_asr_path(cfg, dataset)
        self.events.append({
            "operation": "OPEN_TRAIN_ASR_PROJECTED_FIELDS_ONLY",
            "dataset": dataset,
            "resolved_path": path.as_posix(),
            "path_sha256": sha256_bytes(path.as_posix().encode("utf-8")),
        })
        return path

    def train_video(self, cfg: dict[str, Any], dataset: str, video_id: str) -> Path:
        path = video_path(cfg, dataset, video_id)
        physical_root = Path(cfg["datasets"][dataset]["physical_train_video_root"])
        self.events.append({
            "operation": "HASH_TRAIN_VIDEO",
            "dataset": dataset,
            "video_id_sha256": sha256_bytes(video_id.encode("utf-8")),
            "resolved_train_relative": path.relative_to(physical_root).as_posix(),
            "regular_file_device": path.stat().st_dev,
            "regular_file_inode": path.stat().st_ino,
        })
        return path

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "c04_guarded_access_audit_v2",
            "event_count": len(self.events),
            "events_merkle_root": merkle_root(self.events),
            "events": self.events,
            "static_surface_assertions": {
                "ocr_entrypoint_present": False,
                "external_api_client_present": False,
                "network_entrypoint_present": False,
                "dev_or_test_locator_present": False,
                "cross_dataset_locator_present": False,
                "slurm_submit_release_resubmit_entrypoint_present": False,
            },
            "static_assertions_are_not_runtime_counters": True,
        }


def verify_static_config(cfg: dict[str, Any]) -> None:
    assert_equal(cfg["run"]["run_id"], RUN_ID, "config run id")
    assert_equal(cfg["run"]["implementation_version"], "v2_prospective", "implementation")
    assert_equal(cfg["schema_version"], "c04_a0t_small_v1_v2_config_v1", "config schema")
    assert_equal(tuple(cfg["run"]["datasets"]), DATASETS, "dataset order")
    assert_equal(cfg["selection"]["count_per_dataset"], SELECT_N, "selection count")
    assert_equal(cfg["selection"]["sort"], "ascending_sha256_utf8_concatenation", "selection sort")
    assert_equal(cfg["resources"]["gpu_count"], 1, "GPU count")
    assert_equal(cfg["resources"]["cpus"], 8, "CPU count")
    assert_equal(cfg["resources"]["ram_gb"], 64, "RAM")
    assert_equal(cfg["resources"]["small_cap_gpu_seconds"], 7200, "small cap")
    assert_equal(cfg["resources"]["watchdog_reserve_seconds"], 120, "reserve")
    assert_equal(cfg["resources"]["watchdog_term_then_kill_seconds"], 30, "kill after")
    assert_equal(cfg["review"]["design_verdict"], "GO_0C_0H_0I", "design verdict")
    for key in (
        "test_authorized",
        "dev_authorized",
        "ocr_authorized",
        "external_api_authorized",
        "network_authorized",
        "cross_dataset_authorized",
        "label_value_authorized_before_seal",
        "chain_authorized",
        "release_authorized",
        "resubmit_authorized",
    ):
        assert_equal(cfg["authorization"][key], False, f"authorization.{key}")
    for key in (
        "teacher_authorized",
        "gpu_authorized",
        "slurm_authorized",
        "small_tranche_execution_authorized",
    ):
        assert_equal(cfg["authorization"][key], False, f"preflight authorization.{key}")
    if cfg["authorization"]["preflight_materialization_authorized"] is not True:
        raise RuntimeError("HALT_INVALID_FREEZE: preflight authorization is false")
    verify_bound_file_map(cfg["frozen_design_hashes"], "frozen design")
    verify_bound_file_map(cfg["implementation_hashes"], "implementation")
    assert_equal(prompt_hashes(), cfg["prompt_hashes"], "prompt hashes")
    for dataset in DATASETS:
        source = train_asr_path(cfg, dataset)
        assert_equal(source.stat().st_size, cfg["datasets"][dataset]["train_asr_size"], "ASR size")
        assert_equal(sha256_file(source), cfg["datasets"][dataset]["train_asr_sha256"], "ASR hash")


def verify_code_resource_authorization(cfg: dict[str, Any]) -> tuple[dict[str, Any], str]:
    review = cfg["review"]
    assert_equal(review["code_resource_verdict"], "GO", "code/resource config verdict")
    relative = review["code_resource_authorization_manifest"]
    pin = review["code_resource_authorization_sha256"]
    if not isinstance(pin, str) or len(pin) != 64 or any(c not in "0123456789abcdef" for c in pin):
        raise RuntimeError("HALT_REVIEW_LINEAGE: code/resource authorization SHA is unpinned")
    path = root_path(relative)
    assert_equal(sha256_file(path), pin, "code/resource authorization file")
    manifest = load_json(relative)
    validate_schema(
        manifest,
        cfg["schemas"]["stage_authorization"],
        "code/resource authorization",
    )
    body = verify_closure_hash(manifest, "code/resource authorization")
    assert_equal(body["run_id"], RUN_ID, "code/resource run id")
    assert_equal(body["implementation_version"], "v2_prospective", "code/resource implementation")
    assert_equal(body["stage"], "CPU_PREFLIGHT", "code/resource stage")
    assert_equal(body["verdict"], "GO", "code/resource verdict")
    assert_equal(body["config_contract_sha256"], config_contract_sha256(cfg), "config contract")
    assert_equal(body["authorization_snapshot"], cfg["authorization"], "authorization snapshot")
    assert_equal(body["implementation_hashes"], cfg["implementation_hashes"], "implementation closure")
    assert_equal(body["frozen_design_hashes"], cfg["frozen_design_hashes"], "design closure")
    assert_equal(
        body["design_go_review_sha256"],
        cfg["frozen_design_hashes"]["refine-logs/C04_V4_DESIGN_REVIEW.md"],
        "design GO review",
    )
    assert_equal(body["source_hash_closure"], source_hash_closure(cfg), "source closure")
    assert_equal(body["model_hash_closure"], model_hash_closure(cfg), "model closure")
    assert_equal(body["payload_binding"], "NO_PREFLIGHT_PAYLOAD_YET", "payload binding")
    return manifest, pin


def run_self_tests(cfg: dict[str, Any]) -> dict[str, Any]:
    checks = dict(self_test_fixtures())
    for role in ("S", "P", "T", "H"):
        payload = materialize_role_map(role)
        checks[f"role_{role}_shape"] = (
            len(payload["indices"]) == ROLE_DIM
            and len(payload["signs"]) == ROLE_DIM
            and len(set(payload["indices"])) == ROLE_DIM
            and set(payload["signs"]) <= {-1, 1}
        )
    checks["no_test_paths"] = all(
        "test" not in cfg["datasets"][dataset][field].lower()
        for dataset in DATASETS
        for field in ("train_asr", "video_root", "allowlist_path", "source_manifest_path")
    )
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"HALT_INVALID_FREEZE: self-test failure {failed}")
    return {
        "schema_version": "c04_a0t_small_v1_impl_v2_self_test_v1",
        "run_id": RUN_ID,
        "checks": checks,
        "all_passed": True,
    }


def verify_model_snapshot(cfg: dict[str, Any]) -> dict[str, Any]:
    snapshot = Path(cfg["model"]["snapshot_path"])
    if not snapshot.is_absolute() or not snapshot.is_dir():
        raise RuntimeError("HALT_INVALID_FREEZE: model snapshot missing")
    groups: dict[str, Any] = {}
    for group in ("model", "processor"):
        lines = bytearray()
        rows = []
        for expected in cfg["model"]["files"][group]:
            relative = expected["path"]
            path = snapshot / relative
            if not path.is_file():
                raise RuntimeError(f"HALT_INVALID_FREEZE: model file missing {relative}")
            size = path.stat().st_size
            digest = sha256_file(path)
            assert_equal(size, expected["size"], f"{group} size {relative}")
            assert_equal(digest, expected["sha256"], f"{group} hash {relative}")
            lines.extend(f"{relative}\t{size}\t{digest}\n".encode("utf-8"))
            rows.append({"path": relative, "size": size, "sha256": digest})
        tree_hash = sha256_bytes(bytes(lines))
        assert_equal(tree_hash, cfg["model"][f"{group}_tree_sha256"], f"{group} tree hash")
        groups[group] = {"tree_sha256": tree_hash, "files": rows}
    return groups


def load_dataset_evidence(
    cfg: dict[str, Any],
    dataset: str,
    audit: AccessAudit,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    spec = cfg["datasets"][dataset]
    source = audit.train_asr(cfg, dataset)
    assert_equal(sha256_file(source), spec["train_asr_sha256"], f"{dataset} ASR hash")
    rows: list[dict[str, Any]] = []
    counters = {
        "label_field_syntactically_skipped": 0,
        "label_value_materialized": 0,
    }
    with source.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            projected, row_counts = project_train_asr_line(line)
            counters["label_field_syntactically_skipped"] += row_counts[
                "label_field_syntactically_skipped"
            ]
            counters["label_value_materialized"] += row_counts["label_value_materialized"]
            rows.append(projected)
    assert_equal(len(rows), spec["expected_train_n"], f"{dataset} train count")
    ids = [row["id"] for row in rows]
    if len(ids) != len(set(ids)):
        raise RuntimeError(f"HALT_INVALID_FREEZE: duplicate train ID in {dataset}")
    ranked = sorted(rows, key=lambda row: (selection_digest(dataset, row["id"]), row["id"]))
    return ranked[:SELECT_N], counters


def preflight(cfg: dict[str, Any]) -> None:
    namespace = root_path(ARTIFACT_ROOT)
    if namespace.exists():
        raise FileExistsError(f"no-clobber namespace refusal: {ARTIFACT_ROOT}")
    code_authorization, code_authorization_sha256 = verify_code_resource_authorization(cfg)
    self_test = run_self_tests(cfg)
    model_snapshot = verify_model_snapshot(cfg)
    audit = AccessAudit()

    staged: dict[str, bytes] = {}
    dataset_manifests: dict[str, Any] = {}
    aggregate_counters: dict[str, int] = {}
    for dataset in DATASETS:
        selected, counters = load_dataset_evidence(cfg, dataset, audit)
        for key, value in counters.items():
            aggregate_counters[key] = aggregate_counters.get(key, 0) + value
        allowlist_rows = [
            {
                "rank": rank,
                "video_id": row["id"],
                "selection_sha256": selection_digest(dataset, row["id"]),
            }
            for rank, row in enumerate(selected)
        ]
        source_rows = []
        for row in selected:
            video_relative = f"data/video/{dataset}/All/{row['id']}.mp4"
            video_fs = audit.train_video(cfg, dataset, row["id"])
            physical_root = Path(cfg["datasets"][dataset]["physical_train_video_root"])
            identity = video_fs.stat()
            source_rows.append({
                "video_id": row["id"],
                "language": row["language"],
                "transcript_scalar_count": len(list(row["transcript"])),
                "transcript_sha256": sha256_bytes(row["transcript"].encode("utf-8")),
                "video_path": video_relative,
                "resolved_train_relative": video_fs.relative_to(physical_root).as_posix(),
                "regular_file_device": identity.st_dev,
                "regular_file_inode": identity.st_ino,
                "video_size": video_fs.stat().st_size,
                "video_sha256": sha256_file(video_fs),
            })
        allow_rel = cfg["datasets"][dataset]["allowlist_path"]
        source_rel = cfg["datasets"][dataset]["source_manifest_path"]
        allow_obj = {
            "schema_version": "c04_a0t_small_allowlist_v1",
            "run_id": RUN_ID,
            "dataset": dataset,
            "selection_contract": cfg["selection"],
            "count": len(allowlist_rows),
            "rows": allowlist_rows,
            "merkle_root": merkle_root(allowlist_rows),
        }
        source_obj = {
            "schema_version": "c04_a0t_small_source_manifest_v1",
            "run_id": RUN_ID,
            "dataset": dataset,
            "train_asr_path": cfg["datasets"][dataset]["train_asr"],
            "train_asr_sha256": cfg["datasets"][dataset]["train_asr_sha256"],
            "count": len(source_rows),
            "rows": source_rows,
            "merkle_root": merkle_root(source_rows),
        }
        staged[allow_rel] = canonical_bytes(allow_obj)
        staged[source_rel] = canonical_bytes(source_obj)
        dataset_manifests[dataset] = {
            "allowlist_sha256": sha256_bytes(staged[allow_rel]),
            "allowlist_merkle_root": allow_obj["merkle_root"],
            "source_manifest_sha256": sha256_bytes(staged[source_rel]),
            "source_manifest_merkle_root": source_obj["merkle_root"],
        }

    role_hashes: dict[str, str] = {}
    for role in ("S", "P", "T", "H"):
        role_rel = cfg["maps"]["role_maps"][role]
        role_payload = materialize_role_map(role)
        staged[role_rel] = canonical_bytes(role_payload)
        role_hashes[role] = sha256_bytes(staged[role_rel])

    le3 = dense_rademacher_payload(LE3_TAG, ROLE_DIM, LE3_INPUT_DIM, 1.0 / 16.0)
    additive = dense_rademacher_payload(
        ADDITIVE_TAG, ROLE_DIM, ADDITIVE_INPUT_DIM, 1.0 / 16.0
    )
    staged[cfg["maps"]["le3_payload_path"]] = le3
    staged[cfg["maps"]["additive_payload_path"]] = additive
    map_hashes = {
        "roles": role_hashes,
        "le3_f32le_sha256": sha256_bytes(le3),
        "additive_f32le_sha256": sha256_bytes(additive),
    }

    gpu_ledger = {
        "schema_version": "c04_gpu_ledger_v2",
        "run_id": RUN_ID,
        "implementation_version": "v2_prospective",
        "cap_gpu_seconds": cfg["resources"]["small_cap_gpu_seconds"],
        "ledger_revision": 0,
        "state": "GENESIS_UNCLAIMED",
        "jobs": [],
        "aggregate_accounted_gpu_seconds": 0,
        "aggregate_reconciled_terminal_gpu_seconds": 0,
        "requires_terminal_reconciliation": False,
        "resubmit_authorized": False,
        "single_allocation_only": True,
        "code_resource_authorization_sha256": code_authorization_sha256,
        "config_contract_sha256": config_contract_sha256(cfg),
    }
    gpu_ledger["payload_sha256"] = sha256_obj(gpu_ledger)
    completed_seconds = 0
    remaining = cfg["resources"]["small_cap_gpu_seconds"]
    if remaining <= cfg["resources"]["minimum_submit_remaining_seconds"]:
        raise RuntimeError("HALT_RESOURCE_CAP: insufficient remaining GPU seconds")
    ticket = {
        "schema_version": "c04_resource_ticket_v2",
        "run_id": RUN_ID,
        "implementation_version": "v2_prospective",
        "single_use": True,
        "consumed": False,
        "authorized_slurm_allocation_count": 1,
        "completed_gpu_seconds": completed_seconds,
        "cap_gpu_seconds": cfg["resources"]["small_cap_gpu_seconds"],
        "remaining_seconds": remaining,
        "watchdog_seconds": remaining - cfg["resources"]["watchdog_reserve_seconds"],
        "issued_by_slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "no_submit_performed": True,
        "genesis_gpu_ledger_sha256": "",
        "code_resource_authorization_sha256": code_authorization_sha256,
        "config_contract_sha256": config_contract_sha256(cfg),
    }
    gpu_rel = cfg["paths"]["gpu_ledger"]
    ticket_rel = cfg["paths"]["resource_ticket"]
    staged[gpu_rel] = canonical_bytes(gpu_ledger)
    ticket["genesis_gpu_ledger_sha256"] = sha256_bytes(staged[gpu_rel])
    ticket["payload_sha256"] = sha256_obj(ticket)
    staged[ticket_rel] = canonical_bytes(ticket)
    preflight_access = {
        "schema_version": "c04_access_ledger_v2",
        "run_id": RUN_ID,
        "stage": "preflight",
        "projected_field_counters": aggregate_counters,
        "guarded_runtime_evidence": audit.snapshot(),
        "no_teacher_or_frame_code_invoked_in_this_program": True,
        "static_surface_assertions_are_not_runtime_counters": True,
    }
    staged[cfg["paths"]["access_ledger"]] = canonical_bytes(preflight_access)

    preflight_manifest = {
        "schema_version": "c04_a0t_small_preflight_manifest_v2",
        "run_id": RUN_ID,
        "implementation_version": "v2_prospective",
        "terminal_state": "PREFLIGHT_HASH_FREEZE_PENDING_PAYLOAD_REVIEW",
        "execution_authorized": False,
        "datasets": dataset_manifests,
        "model_snapshot": model_snapshot,
        "prompt_hashes": prompt_hashes(),
        "map_hashes": map_hashes,
        "projected_field_counters": aggregate_counters,
        "guarded_access_audit": audit.snapshot(),
        "code_resource_authorization_sha256": code_authorization_sha256,
        "config_contract_sha256": config_contract_sha256(cfg),
        "self_test": self_test,
        "staged_output_hashes": {
            relative: sha256_bytes(payload) for relative, payload in sorted(staged.items())
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    preflight_manifest["payload_sha256"] = sha256_obj(preflight_manifest)
    staged[cfg["paths"]["preflight_manifest"]] = canonical_bytes(preflight_manifest)

    namespace.parent.mkdir(parents=True, exist_ok=True)
    temp_namespace = Path(
        tempfile.mkdtemp(prefix=namespace.name + ".tmp.", dir=str(namespace.parent))
    )
    try:
        for relative, payload in sorted(staged.items()):
            final_path = root_path(relative)
            try:
                suffix = final_path.relative_to(namespace)
            except ValueError as error:
                raise RuntimeError(
                    f"HALT_INVALID_FREEZE: staged path outside namespace {relative}"
                ) from error
            target = temp_namespace / suffix
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        directory_fd = os.open(str(temp_namespace), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temp_namespace, namespace)
    except Exception:
        shutil.rmtree(temp_namespace, ignore_errors=True)
        raise


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("self-test", "freeze"), required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("C04 preflight requires a reviewed SLURM allocation")
    cfg = load_json(CONFIG_RELATIVE)
    verify_static_config(cfg)
    verify_code_resource_authorization(cfg)
    if args.mode == "self-test":
        result = run_self_tests(cfg)
        print(json.dumps(result, sort_keys=True))
        return 0
    preflight(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
