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
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from c04_a0t_small_v1_common import (  # noqa: E402
    ADDITIVE_INPUT_DIM,
    ADDITIVE_TAG,
    ARTIFACT_ROOT,
    DATASETS,
    LE3_INPUT_DIM,
    LE3_TAG,
    ROLE_DIM,
    RUN_ID,
    SCHEMA_VERSION,
    SELECT_N,
    dense_rademacher_payload,
    exclusive_publish_bytes,
    exclusive_publish_json,
    load_json,
    materialize_role_map,
    merkle_root,
    project_train_asr_line,
    prompt_hashes,
    root_path,
    selection_digest,
    self_test_fixtures,
    sha256_bytes,
    sha256_file,
    sha256_obj,
    video_path,
)


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def verify_static_config(cfg: dict[str, Any], run_id: str) -> None:
    assert_equal(run_id, RUN_ID, "CLI run id")
    assert_equal(cfg["run"]["run_id"], RUN_ID, "config run id")
    assert_equal(cfg["schema_version"], "c04_a0t_small_v1_config_v1", "config schema")
    assert_equal(tuple(cfg["run"]["datasets"]), DATASETS, "dataset order")
    assert_equal(cfg["selection"]["count_per_dataset"], SELECT_N, "selection count")
    assert_equal(cfg["selection"]["sort"], "ascending_sha256_utf8_concatenation", "selection sort")
    assert_equal(cfg["resources"]["gpu_count"], 1, "GPU count")
    assert_equal(cfg["resources"]["cpus"], 8, "CPU count")
    assert_equal(cfg["resources"]["ram_gb"], 64, "RAM")
    assert_equal(cfg["resources"]["small_cap_gpu_seconds"], 7200, "small cap")
    assert_equal(cfg["resources"]["watchdog_reserve_seconds"], 120, "reserve")
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
    if cfg["authorization"]["preflight_materialization_authorized"] is not True:
        raise RuntimeError("HALT_INVALID_FREEZE: preflight authorization is false")
    for relative, expected in cfg["frozen_design_hashes"].items():
        assert_equal(sha256_file(root_path(relative)), expected, f"frozen design hash {relative}")
    for relative, expected in cfg["implementation_hashes"].items():
        assert_equal(sha256_file(root_path(relative)), expected, f"implementation hash {relative}")
    assert_equal(prompt_hashes(), cfg["prompt_hashes"], "prompt hashes")


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
        "schema_version": "c04_a0t_small_v1_self_test_v1",
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
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    spec = cfg["datasets"][dataset]
    source = root_path(spec["train_asr"])
    assert_equal(sha256_file(source), spec["train_asr_sha256"], f"{dataset} ASR hash")
    rows: list[dict[str, Any]] = []
    counters = {
        "label_field_syntactically_skipped": 0,
        "label_value_materialized": 0,
        "dev_content_read_count": 0,
        "test_content_read_count": 0,
        "ocr_call_count": 0,
        "external_api_call_count": 0,
        "cross_dataset_input_count": 0,
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


def reconcile_completed_gpu_seconds(ledger: dict[str, Any]) -> int:
    total = 0
    for job in ledger["jobs"]:
        job_id = str(job["slurm_job_id"])
        if not job_id.isdigit():
            raise RuntimeError(f"HALT_RESOURCE_CAP: invalid prior job id {job_id}")
        completed = subprocess.run(
            ["sacct", "-X", "-n", "-P", "-j", job_id, "-o", "ElapsedRaw,AllocTRES,State"],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [line for line in completed.stdout.splitlines() if line.strip()]
        if len(rows) != 1:
            raise RuntimeError(f"HALT_RESOURCE_CAP: ambiguous sacct result for {job_id}")
        elapsed_text, alloc_tres, state = rows[0].split("|", 2)
        if not state.startswith(("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY")):
            raise RuntimeError(f"HALT_RESOURCE_CAP: prior job {job_id} is nonterminal")
        gpu_count = 0
        for token in alloc_tres.split(","):
            if token.startswith("gres/gpu="):
                gpu_count = int(token.split("=", 1)[1])
            elif token.startswith("gres/gpu:") and "=" in token:
                gpu_count = int(token.rsplit("=", 1)[1])
        if gpu_count != 1:
            raise RuntimeError(f"HALT_RESOURCE_CAP: prior job GPU count {gpu_count}")
        total += int(elapsed_text) * gpu_count
    return total


def preflight(cfg: dict[str, Any]) -> None:
    namespace = root_path(ARTIFACT_ROOT)
    if namespace.exists():
        raise FileExistsError(f"no-clobber namespace refusal: {ARTIFACT_ROOT}")
    self_test = run_self_tests(cfg)
    model_snapshot = verify_model_snapshot(cfg)

    staged: dict[str, bytes] = {}
    dataset_manifests: dict[str, Any] = {}
    aggregate_counters: dict[str, int] = {}
    for dataset in DATASETS:
        selected, counters = load_dataset_evidence(cfg, dataset)
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
            video_fs = video_path(dataset, row["id"])
            source_rows.append({
                "video_id": row["id"],
                "language": row["language"],
                "transcript_scalar_count": len(list(row["transcript"])),
                "transcript_sha256": sha256_bytes(row["transcript"].encode("utf-8")),
                "video_path": video_relative,
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
        "schema_version": "c04_gpu_ledger_v1",
        "run_id": RUN_ID,
        "cap_gpu_seconds": cfg["resources"]["small_cap_gpu_seconds"],
        "jobs": [],
        "watchdog_events": 0,
        "resubmit_authorized": False,
    }
    completed_seconds = reconcile_completed_gpu_seconds(gpu_ledger)
    remaining = cfg["resources"]["small_cap_gpu_seconds"] - completed_seconds
    if remaining <= cfg["resources"]["minimum_submit_remaining_seconds"]:
        raise RuntimeError("HALT_RESOURCE_CAP: insufficient remaining GPU seconds")
    ticket = {
        "schema_version": "c04_resource_ticket_v1",
        "run_id": RUN_ID,
        "single_use": True,
        "consumed": False,
        "completed_gpu_seconds": completed_seconds,
        "cap_gpu_seconds": cfg["resources"]["small_cap_gpu_seconds"],
        "remaining_seconds": remaining,
        "watchdog_seconds": remaining - cfg["resources"]["watchdog_reserve_seconds"],
        "issued_by_slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "no_submit_performed": True,
    }
    gpu_rel = cfg["paths"]["gpu_ledger"]
    ticket_rel = cfg["paths"]["resource_ticket"]
    staged[gpu_rel] = canonical_bytes(gpu_ledger)
    staged[ticket_rel] = canonical_bytes(ticket)

    preflight_manifest = {
        "schema_version": "c04_a0t_small_preflight_manifest_v1",
        "run_id": RUN_ID,
        "terminal_state": "PREFLIGHT_HASH_FREEZE_PENDING_PAYLOAD_REVIEW",
        "execution_authorized": False,
        "datasets": dataset_manifests,
        "model_snapshot": model_snapshot,
        "prompt_hashes": prompt_hashes(),
        "map_hashes": map_hashes,
        "access_zero_counters": aggregate_counters,
        "self_test": self_test,
        "staged_output_hashes": {
            relative: sha256_bytes(payload) for relative, payload in sorted(staged.items())
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    preflight_manifest["payload_sha256"] = sha256_obj(preflight_manifest)
    staged[cfg["paths"]["preflight_manifest"]] = canonical_bytes(preflight_manifest)
    staged[cfg["paths"]["access_ledger"]] = canonical_bytes({
        "schema_version": "c04_access_ledger_v1",
        "run_id": RUN_ID,
        "stage": "preflight",
        "authorized_train_evidence_read_count": sum(
            cfg["datasets"][dataset]["expected_train_n"] for dataset in DATASETS
        ),
        "counters": aggregate_counters,
        "teacher_calls": 0,
        "frame_decodes": 0,
        "label_value_materialized": aggregate_counters["label_value_materialized"],
    })

    created: list[Path] = []
    try:
        for relative, payload in sorted(staged.items()):
            exclusive_publish_bytes(relative, payload)
            created.append(root_path(relative))
    except Exception:
        for path in reversed(created):
            for target in (path, path.with_name(path.name + ".publish.lock")):
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
        raise


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8") + b"\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("self-test", "freeze"), required=True)
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("C04 preflight requires a reviewed SLURM allocation")
    cfg = load_json(args.config)
    verify_static_config(cfg, args.run_id)
    if args.mode == "self-test":
        result = run_self_tests(cfg)
        print(json.dumps(result, sort_keys=True))
        return 0
    preflight(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
