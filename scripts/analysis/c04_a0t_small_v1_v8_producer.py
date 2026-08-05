#!/usr/bin/env python
"""Prospective GPU producer and pre-label seal for C04-A0T-SMALL-v1.

The fixed dataset order is HateMM then MHC_zh on one allocation.  Each selected
train ID receives exactly prompt A and prompt B.  There is no retry, redraw,
label read, dev/test read, OCR, network/API call, cross-dataset input, job
submission, dependency, release, or resubmission path.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from c04_a0t_small_v1_v8_common import (  # noqa: E402
    ADDITIVE_INPUT_DIM,
    ARTIFACT_ROOT,
    CAMPAIGN_AGGREGATE_CAP_GPU_SECONDS,
    CAMPAIGN_PHASE_CAPS,
    CAMPAIGN_GPU_LEDGER_RELATIVE,
    CONFIG_RELATIVE,
    DATASETS,
    EPS,
    FRAME_PACK_METADATA_KEYS,
    LABEL_BEARING_ID_SUBSTRINGS,
    LE3_INPUT_DIM,
    MAX_NEW_TOKENS,
    NUM_FRAMES,
    PROMPT_FORMS,
    PROMPT_HASH_BINDING_LITERAL,
    PROMPT_HASH_KEYS,
    PROPOSITION_COSINE_MIN,
    Q_DIM,
    ROLE_DIM,
    RUN_ID,
    SCHEMA_VERSION,
    SELECT_N,
    SLOTS,
    SMALL_TRANCHE_CAP_GPU_SECONDS,
    SYSTEM_PROMPT,
    TEACHER_MAX_PIXELS,
    VISUAL_PATCH_TOKEN_HARD_CEILING,
    assert_campaign_aggregate_headroom,
    assert_teacher_visible_containment,
    assert_visual_token_ceiling,
    build_provisional_gpu_usage,
    build_slot_reliability,
    _write_fsynced,
    canonical_json_bytes,
    config_contract_sha256,
    cosine,
    exclusive_publish_bytes,
    exclusive_publish_json,
    exclusive_publish_jsonl,
    f32le_b64,
    forbidden_teacher_visible_tokens,
    frame_pack_binding,
    load_frame_pack_images,
    load_json,
    merkle_root,
    normalize_proposition,
    parse_teacher_response,
    project_train_asr_line,
    prompt_hashes,
    q_product,
    render_prompt,
    require_exact_keys,
    resolve_prompt_hashes,
    root_path,
    safe_vector,
    selection_digest,
    sha256_bytes,
    sha256_file,
    sha256_obj,
    strict_validate_frame_pack,
    teacher_visible_texts,
    train_asr_path,
    validate_schema,
    verify_bound_file_map,
    verify_gpu_execution_authorization,
    verify_historical_code_resource_authorization,
    verify_payload_review,
    verify_preflight_manifest,
    verify_prompt_hash_freeze_payload,
    video_path,
    visual_patch_tokens,
)


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


class AccessAudit:
    """Guard and record all producer-controlled evidence/model entrypoints."""

    def __init__(self) -> None:
        self.events: list[dict[str, Any]] = []

    def train_asr(self, cfg: dict[str, Any], dataset: str) -> Path:
        path = train_asr_path(cfg, dataset)
        self.events.append({
            "operation": "OPEN_TRAIN_ASR_PROJECTED_FIELDS_ONLY",
            "dataset": dataset,
            "resolved_path": path.as_posix(),
        })
        return path

    def train_video(self, cfg: dict[str, Any], dataset: str, video_id: str) -> Path:
        path = video_path(cfg, dataset, video_id)
        physical = Path(cfg["datasets"][dataset]["physical_train_video_root"])
        identity = path.stat()
        self.events.append({
            "operation": "OPEN_TRAIN_VIDEO",
            "dataset": dataset,
            "video_id_sha256": sha256_bytes(video_id.encode("utf-8")),
            "resolved_train_relative": path.relative_to(physical).as_posix(),
            "regular_file_device": identity.st_dev,
            "regular_file_inode": identity.st_ino,
        })
        return path

    def event(self, operation: str, dataset: str, video_id: str) -> None:
        self.events.append({
            "operation": operation,
            "dataset": dataset,
            "video_id_sha256": sha256_bytes(video_id.encode("utf-8")),
        })

    def snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "c04_guarded_access_audit_v8",
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
                # v8: every frame pack is frozen by the CPU preflight, so this
                # program has no decoder import and no image-encode call at all.
                # The CPU preflight proves this by parsing these bytes.
                "video_decode_entrypoint_present": False,
                "image_encode_entrypoint_present": False,
            },
            "offline_runtime_guards": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "model_loader_local_files_only": True,
            },
            "static_assertions_are_not_runtime_counters": True,
        }


def verify_authorization(cfg: dict[str, Any]) -> None:
    assert_equal(cfg["run"]["run_id"], RUN_ID, "config run id")
    assert_equal(cfg["run"]["implementation_version"], "v8_prospective", "implementation")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("HALT_INVALID_FREEZE: producer requires SLURM")
    if os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_DEPENDENCY"):
        raise RuntimeError("HALT_INVALID_FREEZE: array/dependency execution forbidden")
    required_true = (
        "teacher_authorized", "gpu_authorized", "slurm_authorized",
        "small_tranche_execution_authorized",
    )
    for key in required_true:
        if cfg["authorization"][key] is not True:
            raise RuntimeError(f"HALT_INVALID_FREEZE: authorization.{key} is false")
    required_false = (
        "preflight_materialization_authorized",
        "test_authorized", "dev_authorized", "ocr_authorized",
        "external_api_authorized", "network_authorized",
        "cross_dataset_authorized", "label_value_authorized_before_seal",
        "chain_authorized", "release_authorized", "resubmit_authorized",
        "post_job_reconciliation_authorized",
    )
    for key in required_false:
        assert_equal(cfg["authorization"][key], False, f"authorization.{key}")
    assert_equal(os.environ.get("HF_HUB_OFFLINE"), "1", "HF_HUB_OFFLINE")
    assert_equal(os.environ.get("TRANSFORMERS_OFFLINE"), "1", "TRANSFORMERS_OFFLINE")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("HALT_RESOURCE_CAP: exactly one visible GPU required")
    assert_equal(cfg["resources"]["gpu_count"], 1, "GPU count")
    assert_equal(cfg["resources"]["cpus"], 8, "CPU count")
    assert_equal(cfg["resources"]["ram_gb"], 64, "RAM")
    assert_equal(
        cfg["resources"]["small_cap_gpu_seconds"],
        SMALL_TRANCHE_CAP_GPU_SECONDS,
        "GPU cap",
    )
    assert_equal(
        cfg["resources"]["campaign_aggregate_cap_gpu_seconds"],
        CAMPAIGN_AGGREGATE_CAP_GPU_SECONDS,
        "campaign aggregate cap",
    )
    assert_equal(
        cfg["resources"]["guard_item_margin_seconds"], 300, "guard item margin"
    )
    assert_equal(
        cfg["resources"]["guard_seal_reserve_seconds"], 600, "guard seal reserve"
    )
    assert_equal(
        cfg["resources"]["campaign_first_tranche_phase_cap_gpu_seconds"],
        CAMPAIGN_PHASE_CAPS["FIRST_TRANCHE"],
        "campaign first-tranche phase cap",
    )
    assert_equal(
        cfg["paths"]["campaign_gpu_ledger"],
        CAMPAIGN_GPU_LEDGER_RELATIVE,
        "campaign ledger path",
    )
    # The amendment's 8 GPU-hour campaign ceiling, checked once at job start
    # against the machine-readable accumulator before any model or data work.
    assert_campaign_aggregate_headroom(
        cfg["resources"]["small_cap_gpu_seconds"], "the small tranche"
    )
    verify_bound_file_map(cfg["implementation_hashes"], "producer implementation")
    verify_bound_file_map(cfg["frozen_design_hashes"], "producer frozen design")
    resolved_prompt_hashes, prompt_hash_binding = resolve_prompt_hashes(cfg, False)
    assert_equal(
        prompt_hash_binding, PROMPT_HASH_BINDING_LITERAL, "producer prompt-hash binding"
    )
    assert_equal(resolved_prompt_hashes, prompt_hashes(), "producer prompt hashes")


def verify_model_snapshot(cfg: dict[str, Any]) -> None:
    snapshot = Path(cfg["model"]["snapshot_path"])
    if not snapshot.is_absolute() or not snapshot.is_dir():
        raise RuntimeError("HALT_REVIEW_LINEAGE: pinned model snapshot missing")
    for group in ("model", "processor"):
        lines = bytearray()
        for expected in cfg["model"]["files"][group]:
            relative = expected["path"]
            path = snapshot / relative
            if not path.is_file():
                raise RuntimeError(f"HALT_REVIEW_LINEAGE: model file missing {relative}")
            size = path.stat().st_size
            digest = sha256_file(path)
            assert_equal(size, expected["size"], f"{group} size {relative}")
            assert_equal(digest, expected["sha256"], f"{group} hash {relative}")
            lines.extend(f"{relative}\t{size}\t{digest}\n".encode("utf-8"))
        assert_equal(
            sha256_bytes(bytes(lines)),
            cfg["model"][f"{group}_tree_sha256"],
            f"{group} tree hash",
        )


def verify_claimed_resource(
    cfg: dict[str, Any],
    preflight: dict[str, Any],
    preflight_sha: str,
    payload_sha: str,
    gpu_auth_sha: str,
) -> tuple[dict[str, Any], "BudgetGuard"]:
    ticket_path = root_path(cfg["paths"]["resource_ticket"])
    ticket = load_json(cfg["paths"]["resource_ticket"])
    ticket_body = dict(ticket)
    ticket_claimed = ticket_body.pop("payload_sha256", None)
    assert_equal(sha256_obj(ticket_body), ticket_claimed, "ticket payload")
    assert_equal(ticket["run_id"], RUN_ID, "resource ticket run id")
    assert_equal(ticket["single_use"], True, "resource ticket single-use")
    assert_equal(ticket["consumed"], False, "resource ticket consumed flag")
    assert_equal(
        ticket["genesis_gpu_ledger_sha256"],
        preflight["staged_output_hashes"][cfg["paths"]["gpu_ledger"]],
        "ticket/preflight genesis ledger",
    )
    claim = load_json(cfg["paths"]["allocation_claim"])
    claim_body = dict(claim)
    claim_claimed = claim_body.pop("claim_sha256", None)
    assert_equal(sha256_obj(claim_body), claim_claimed, "allocation claim payload")
    assert_equal(claim["slurm_job_id"], os.environ["SLURM_JOB_ID"], "claim job id")
    assert_equal(claim["preflight_manifest_sha256"], preflight_sha, "claim preflight")
    assert_equal(claim["payload_review_sha256"], payload_sha, "claim payload review")
    assert_equal(
        claim["gpu_execution_authorization_sha256"], gpu_auth_sha, "claim GPU authorization"
    )
    assert_equal(claim["config_contract_sha256"], config_contract_sha256(cfg), "claim config")
    consumption = load_json(cfg["paths"]["resource_consumption"])
    assert_equal(consumption["slurm_job_id"], os.environ["SLURM_JOB_ID"], "consumption job")
    assert_equal(consumption["ticket_sha256"], sha256_file(ticket_path), "consumed ticket")
    assert_equal(
        consumption["allocation_claim_sha256"],
        sha256_file(root_path(cfg["paths"]["allocation_claim"])),
        "consumed claim",
    )
    ledger = load_json(cfg["paths"]["gpu_ledger"])
    ledger_body = dict(ledger)
    ledger_claimed = ledger_body.pop("payload_sha256", None)
    assert_equal(sha256_obj(ledger_body), ledger_claimed, "GPU ledger payload")
    matching = [
        job for job in ledger["jobs"]
        if str(job["slurm_job_id"]) == os.environ["SLURM_JOB_ID"]
    ]
    if len(matching) != 1 or matching[0]["status"] not in {
        "CLAIMED_ACTIVE", "SACCT_ACTIVE"
    }:
        raise RuntimeError("HALT_RESOURCE_CAP: no unique active ledger claim")
    if int(ledger["aggregate_accounted_gpu_seconds"]) > int(ledger["cap_gpu_seconds"]):
        raise RuntimeError("HALT_RESOURCE_CAP: ledger aggregate exceeds cap")
    watchdog_env = int(os.environ.get("C04_WATCHDOG_SECONDS", "0"))
    if watchdog_env <= 0 or watchdog_env > int(ticket["watchdog_seconds"]):
        raise RuntimeError("HALT_RESOURCE_CAP: invalid allocation-start watchdog remainder")
    return claim, BudgetGuard.at_job_start(
        cfg, watchdog_env, int(claim["allocation_entry_uptime_seconds"])
    )


def verify_frozen_prompt_hashes(
    cfg: dict[str, Any],
    preflight: dict[str, Any],
) -> dict[str, str]:
    """Bind this downstream stage to the literal hashes frozen by CPU preflight.

    The freeze artifact, the preflight manifest and the config must all carry
    the same literal values; the pending sentinel is rejected everywhere on
    this path.
    """
    relative = cfg["paths"]["prompt_hash_freeze"]
    payload = load_json(relative)
    frozen = verify_prompt_hash_freeze_payload(payload)
    assert_equal(
        sha256_file(root_path(relative)),
        preflight["staged_output_hashes"][relative],
        "frozen prompt-hash artifact",
    )
    attested = preflight["prompt_hash_freeze"]
    assert_equal(
        payload["payload_sha256"],
        attested["payload_sha256"],
        "frozen prompt-hash payload lineage",
    )
    assert_equal(attested["path"], relative, "frozen prompt-hash path")
    assert_equal(
        attested["sha256"], sha256_file(root_path(relative)), "frozen prompt-hash file"
    )
    assert_equal(attested["keys"], list(PROMPT_HASH_KEYS), "frozen prompt-hash keys")
    assert_equal(
        attested["config_binding_at_freeze"],
        payload["config_binding_at_freeze"],
        "frozen prompt-hash config binding",
    )
    assert_equal(attested["literal_hashes_written"], True, "frozen literal attestation")
    assert_equal(attested["sentinel_written"], False, "frozen sentinel attestation")
    assert_equal(
        attested["downstream_must_read_literal_values_here"],
        True,
        "frozen downstream attestation",
    )
    manifest_hashes = require_exact_keys(
        preflight["prompt_hashes"], set(PROMPT_HASH_KEYS), "preflight prompt hashes"
    )
    assert_equal(frozen, manifest_hashes, "frozen vs preflight prompt hashes")
    resolved, binding = resolve_prompt_hashes(cfg, False)
    assert_equal(binding, PROMPT_HASH_BINDING_LITERAL, "config prompt-hash binding")
    assert_equal(frozen, resolved, "frozen vs config prompt hashes")
    return frozen


def verify_execution_lineage(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], "BudgetGuard"]:
    verify_authorization(cfg)
    preflight, preflight_sha = verify_preflight_manifest(
        cfg, allow_claimed_gpu_ledger=True
    )
    verify_frozen_prompt_hashes(cfg, preflight)
    verify_historical_code_resource_authorization(cfg, preflight)
    _, payload_sha = verify_payload_review(cfg, preflight, preflight_sha)
    _, gpu_auth_sha = verify_gpu_execution_authorization(
        cfg, preflight, preflight_sha, payload_sha
    )
    verify_model_snapshot(cfg)
    claim, deadline = verify_claimed_resource(
        cfg, preflight, preflight_sha, payload_sha, gpu_auth_sha
    )
    return preflight, {
        "preflight_manifest_sha256": preflight_sha,
        "payload_review_sha256": payload_sha,
        "gpu_execution_authorization_sha256": gpu_auth_sha,
        "config_contract_sha256": config_contract_sha256(cfg),
        "allocation_claim_sha256": sha256_file(
            root_path(cfg["paths"]["allocation_claim"])
        ),
    }, claim, deadline


BUDGET_BREACH_EXIT_CODE = 40


class BudgetDeadlineReached(RuntimeError):
    """Raised only *before* a unit of work starts, never during or after one."""


class BudgetGuard:
    """The frozen 2 GPU-hour tranche ceiling, enforced in-job.

    v6 relied on two things: the wrapper's `timeout`, which kills mid-forward,
    and a monotonic deadline derived from the ledger claim.  The v6 payload
    review (Important I-3) asked for the ceiling to be a machine-checked,
    fail-closed guard.  This class is that guard, modelled on the C02 bounded
    extraction: ONE absolute deadline is computed at job start and never
    recomputed, and the guard may only ever STOP work before an item begins.
    It never truncates, rewrites or shortens an output -- a breach leaves the
    in-progress item entirely unwritten, every completed per-item checkpoint
    record intact, and no seal at all, because the seal is published only when
    all 800 records exist.
    """

    def __init__(
        self,
        monotonic_deadline: float,
        watchdog_seconds: int,
        entry_uptime_seconds: int,
        item_margin_seconds: int,
        seal_reserve_seconds: int,
    ) -> None:
        self.monotonic_deadline = monotonic_deadline
        self.watchdog_seconds = watchdog_seconds
        self.entry_uptime_seconds = entry_uptime_seconds
        self.item_margin_seconds = item_margin_seconds
        self.seal_reserve_seconds = seal_reserve_seconds
        self.started_monotonic = time.monotonic()

    @classmethod
    def at_job_start(
        cls,
        cfg: dict[str, Any],
        watchdog_seconds: int,
        allocation_entry_uptime_seconds: int,
    ) -> "BudgetGuard":
        """Anchor the deadline to allocation entry, ahead of the wrapper timeout.

        The anchor is `allocation_entry_marker`'s `/proc/uptime` reading, which
        the wrapper takes before its first Python call.  That is strictly
        earlier than the moment `timeout` starts counting, so this guard is
        always ahead of the wrapper's kill -- and it does not depend on
        `SLURM_JOB_START_TIME` being exported, whose absence would otherwise
        anchor the guard *after* the wrapper (the model tree alone is SHA-256'd
        before this point) and leave it unable to fire at all.

        The margin then buys back at least one worst-case item, so the guard
        stops cleanly before an item instead of the wrapper killing a forward
        mid-flight -- which would produce exit 124 and no breach record.
        """
        now_monotonic = time.monotonic()
        cap = int(cfg["resources"]["small_cap_gpu_seconds"])
        margin = int(cfg["resources"]["guard_item_margin_seconds"])
        seal_reserve = int(cfg["resources"]["guard_seal_reserve_seconds"])
        if margin <= 0 or margin >= watchdog_seconds:
            raise RuntimeError("HALT_RESOURCE_CAP: invalid guard item margin")
        if seal_reserve <= 0 or seal_reserve >= watchdog_seconds:
            raise RuntimeError("HALT_RESOURCE_CAP: invalid guard seal reserve")
        with Path("/proc/uptime").open("r", encoding="ascii") as handle:
            uptime_now = int(float(handle.read().split()[0]))
        elapsed_since_entry = uptime_now - int(allocation_entry_uptime_seconds)
        if elapsed_since_entry < 0:
            raise RuntimeError("HALT_RESOURCE_CAP: allocation entry is in the future")
        remaining = watchdog_seconds - elapsed_since_entry - margin
        if remaining <= 0:
            raise RuntimeError(
                "HALT_RESOURCE_CAP: no budget remains at job start under the "
                f"{cap}s tranche ceiling ({elapsed_since_entry}s already elapsed "
                f"since allocation entry, {margin}s item margin reserved)"
            )
        return cls(
            now_monotonic + remaining,
            watchdog_seconds,
            int(allocation_entry_uptime_seconds),
            margin,
            seal_reserve,
        )

    def remaining_seconds(self) -> float:
        return self.monotonic_deadline - time.monotonic()

    def check(self, unit: str) -> None:
        """Stop before `unit` begins if the ceiling has been reached."""
        if time.monotonic() >= self.monotonic_deadline:
            raise BudgetDeadlineReached(
                f"HALT_RESOURCE_CAP: the frozen tranche ceiling was reached before "
                f"starting {unit}; no output was truncated or altered"
            )

    def require_remaining(self, seconds: int, unit: str) -> None:
        """Stop before a whole PHASE begins unless it comfortably fits.

        The item margin sizes one item.  The post-loop canonicalization and seal
        is a much larger indivisible unit, and it was previously unguarded: an
        overrun there is killed by the wrapper mid-write, which yields exit 124,
        no seal, no breach record, and an unresumable single-use tranche.
        """
        remaining = self.remaining_seconds()
        if remaining < seconds:
            raise BudgetDeadlineReached(
                f"HALT_RESOURCE_CAP: {int(remaining)}s remain against the {seconds}s "
                f"reserved for {unit}; stopping before it begins, with no output "
                "truncated or altered"
            )

    def accounting_snapshot(self) -> dict[str, Any]:
        return {
            "deadline_basis": "ALLOCATION_ENTRY_UPTIME_MINUS_ITEM_MARGIN",
            "allocation_entry_uptime_seconds": self.entry_uptime_seconds,
            "ticket_watchdog_seconds": self.watchdog_seconds,
            "item_margin_seconds": self.item_margin_seconds,
            "seal_reserve_seconds": self.seal_reserve_seconds,
            "producer_elapsed_seconds": int(time.monotonic() - self.started_monotonic),
            "guard_may_only_stop_work_before_an_item": True,
            "guard_never_truncates_or_alters_an_output": True,
        }


def deadline_check(guard: BudgetGuard, unit: str = "a teacher forward") -> None:
    guard.check(unit)


def publish_budget_breach_record(
    cfg: dict[str, Any],
    lineage: dict[str, str],
    guard: BudgetGuard,
    completed_by_dataset: dict[str, int],
    teacher_calls: int,
    frame_packs_loaded: int,
    detail: str,
) -> str:
    """Write the accounting-only breach record.

    It carries no metric, no teacher output, no reliability rate and no
    CONTINUE/KILL verdict -- only resource accounting and the position at which
    the guard stopped.
    """
    record = {
        "schema_version": "c04_budget_breach_v8",
        "run_id": RUN_ID,
        "implementation_version": "v8_prospective",
        "lineage": lineage,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "terminal_state": "HALT_RESOURCE_CAP_TRANCHE_CEILING",
        "exit_code": BUDGET_BREACH_EXIT_CODE,
        "detail": detail,
        "small_cap_gpu_seconds": cfg["resources"]["small_cap_gpu_seconds"],
        "campaign_aggregate_cap_gpu_seconds": cfg["resources"][
            "campaign_aggregate_cap_gpu_seconds"
        ],
        "guard": guard.accounting_snapshot(),
        "completed_prompt_records_by_dataset": completed_by_dataset,
        "teacher_calls_this_invocation": teacher_calls,
        "frame_packs_created_this_invocation": 0,
        "frozen_frame_packs_loaded_this_invocation": frame_packs_loaded,
        "outputs_truncated_or_altered": 0,
        "seal_published": False,
        "no_performance_claim": True,
        "no_scientific_verdict_is_published_by_a_budget_breach": True,
        "requires_sacct_reconciliation": True,
    }
    record["payload_sha256"] = sha256_obj(record)
    exclusive_publish_json(cfg["paths"]["budget_breach"], record)
    return record["payload_sha256"]


def load_selected_inputs(
    cfg: dict[str, Any],
    preflight: dict[str, Any],
    dataset: str,
    audit: AccessAudit,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    spec = cfg["datasets"][dataset]
    allowlist = load_json(spec["allowlist_path"])
    source_manifest = load_json(spec["source_manifest_path"])
    assert_equal(allowlist["count"], SELECT_N, f"{dataset} allowlist count")
    assert_equal(source_manifest["count"], SELECT_N, f"{dataset} source count")
    assert_equal(
        sha256_file(root_path(spec["allowlist_path"])),
        preflight["datasets"][dataset]["allowlist_sha256"],
        f"{dataset} allowlist hash",
    )
    assert_equal(
        sha256_file(root_path(spec["source_manifest_path"])),
        preflight["datasets"][dataset]["source_manifest_sha256"],
        f"{dataset} source hash",
    )
    selected_ids = [row["video_id"] for row in allowlist["rows"]]
    if len(selected_ids) != SELECT_N or len(set(selected_ids)) != SELECT_N:
        raise RuntimeError(f"HALT_INVALID_FREEZE: {dataset} allowlist uniqueness")
    for rank, row in enumerate(allowlist["rows"]):
        assert_equal(row["rank"], rank, f"{dataset} allowlist rank")
        assert_equal(
            row["selection_sha256"],
            selection_digest(dataset, row["video_id"]),
            f"{dataset} selection digest",
        )
    if selected_ids != [
        row["video_id"]
        for row in sorted(
            allowlist["rows"], key=lambda row: (row["selection_sha256"], row["video_id"])
        )
    ]:
        raise RuntimeError(f"HALT_INVALID_FREEZE: {dataset} allowlist order")
    source_by_id = {row["video_id"]: row for row in source_manifest["rows"]}
    if set(source_by_id) != set(selected_ids):
        raise RuntimeError(f"HALT_INVALID_FREEZE: {dataset} source/allowlist mismatch")

    projected: dict[str, dict[str, Any]] = {}
    counters = {
        "label_field_syntactically_skipped": 0,
        "label_value_materialized": 0,
    }
    source_path = audit.train_asr(cfg, dataset)
    assert_equal(sha256_file(source_path), spec["train_asr_sha256"], f"{dataset} ASR hash")
    with source_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row, row_counts = project_train_asr_line(line)
            if row["id"] in source_by_id:
                if row["id"] in projected:
                    raise RuntimeError(f"HALT_INVALID_FREEZE: duplicate selected ID {row['id']}")
                projected[row["id"]] = row
            counters["label_field_syntactically_skipped"] += row_counts[
                "label_field_syntactically_skipped"
            ]
            counters["label_value_materialized"] += row_counts["label_value_materialized"]
    if set(projected) != set(selected_ids):
        raise RuntimeError(f"HALT_INVALID_FREEZE: missing selected evidence in {dataset}")
    output = []
    for video_id in selected_ids:
        row = projected[video_id]
        source = source_by_id[video_id]
        assert_equal(
            source["video_path"],
            f"data/video/{dataset}/All/{video_id}.mp4",
            f"{dataset}/{video_id} lexical video path",
        )
        assert_equal(source["language"], row["language"], "source language")
        assert_equal(
            source["transcript_scalar_count"],
            len(list(row["transcript"])),
            "source transcript scalar count",
        )
        assert_equal(
            sha256_bytes(row["transcript"].encode("utf-8")),
            source["transcript_sha256"],
            f"{dataset}/{video_id} transcript hash",
        )
        video_fs = audit.train_video(cfg, dataset, video_id)
        physical_root = Path(spec["physical_train_video_root"])
        identity = video_fs.stat()
        assert_equal(
            video_fs.relative_to(physical_root).as_posix(),
            source["resolved_train_relative"],
            f"{dataset}/{video_id} resolved train relative",
        )
        assert_equal(identity.st_dev, source["regular_file_device"], "video device")
        assert_equal(identity.st_ino, source["regular_file_inode"], "video inode")
        assert_equal(sha256_file(video_fs), source["video_sha256"], f"{dataset}/{video_id} video hash")
        output.append({
            "video_id": video_id,
            "language": row["language"],
            "transcript": row["transcript"],
            "source": source,
            "resolved_video_path": video_fs,
        })
    return output, counters


def frame_pack_binding_for(
    cfg: dict[str, Any],
    preflight: dict[str, Any],
    dataset: str,
    input_row: dict[str, Any],
) -> dict[str, Any]:
    """Re-derive the binding the CPU preflight wrote into this pack's manifest.

    The code/resource pin is read from the PREFLIGHT MANIFEST rather than from
    `lineage`, because `lineage` is embedded verbatim in every record's
    `provenance` and its key set is pinned by the prompt-record schema; adding a
    key there would move a reviewed contract for no gain.  The manifest's own
    pin is verified against the config by
    `verify_historical_code_resource_authorization(cfg, preflight)`.
    """
    return frame_pack_binding(
        cfg,
        dataset,
        input_row["video_id"],
        input_row["source"]["transcript_sha256"],
        input_row["source"]["video_sha256"],
        preflight["code_resource_authorization_sha256"],
    )


def load_frozen_frame_pack(
    cfg: dict[str, Any],
    preflight: dict[str, Any],
    dataset: str,
    input_row: dict[str, Any],
    audit: AccessAudit,
) -> dict[str, Any]:
    """Load the frame pack the CPU preflight froze.  This stage never creates one.

    v7 decoded videos inside the GPU allocation, so a slow or pathological
    decode spent budgeted A100 seconds -- and one MHC_zh item needs a pyav
    fallback that measured 32.9 s on its own.  In v8 every pack already exists
    and is pinned twice: by `staged_output_hashes` in the preflight manifest,
    and by the manifest binding re-derived here.  A missing pack HALTs rather
    than falling back to a decode, because a decode inside this window is
    exactly the cost the preflight projection did not budget for.
    """
    video_id = input_row["video_id"]
    pack_root = root_path(cfg["datasets"][dataset]["frame_pack_root"])
    binding = frame_pack_binding_for(cfg, preflight, dataset, input_row)
    if not (pack_root / video_id / "manifest.json").is_file():
        raise RuntimeError(
            f"HALT_INVALID_FREEZE: no frozen frame pack for {dataset}/{video_id}; "
            "the producer may not decode a video"
        )
    manifest, frame_paths, manifest_sha256 = strict_validate_frame_pack(
        pack_root, video_id, binding
    )
    manifest_relative = (
        f"{cfg['datasets'][dataset]['frame_pack_root']}/{video_id}/manifest.json"
    )
    assert_equal(
        manifest_sha256,
        preflight["staged_output_hashes"][manifest_relative],
        f"frozen frame-pack manifest lineage {dataset}/{video_id}",
    )
    frames = load_frame_pack_images(frame_paths)
    audit.event("LOAD_FROZEN_FRAME_PACK", dataset, video_id)
    return {
        "frames": frames,
        "backend": manifest["frame_backend"],
        "total_frame_indices": manifest["total_frame_indices"],
        "requested_indices": manifest["requested_indices"],
        "frame_decode_failed": manifest["frame_decode_failed"],
        "manifest_sha256": manifest_sha256,
        "frame_sha256": [row["sha256"] for row in manifest["frames"]],
        "preflight_visual_patch_tokens": int(
            preflight["visual_geometry"]["items"][dataset][video_id]["patch_tokens"]
        ),
    }


def build_messages(frames: list[Any], transcript: str, prompt_form: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": render_prompt(prompt_form, transcript)},
        ]},
    ]


def expected_prompt_provenance(
    cfg: dict[str, Any],
    lineage: dict[str, str],
    prompt_form: str,
) -> dict[str, Any]:
    return {
        **lineage,
        "prompt_sha256": prompt_hashes()[prompt_form],
        "model_snapshot_revision": cfg["model"]["snapshot_revision"],
        "model_tree_sha256": cfg["model"]["model_tree_sha256"],
        "processor_tree_sha256": cfg["model"]["processor_tree_sha256"],
        "teacher_max_pixels": TEACHER_MAX_PIXELS,
        "visual_patch_token_hard_ceiling": VISUAL_PATCH_TOKEN_HARD_CEILING,
        "guard_contract": {
            "label_values": "SYNTACTIC_SKIP_ONLY",
            "dev_test_paths": "REJECTED_BY_COMPONENT_AND_ROOT_GUARD",
            "cross_dataset_paths": "REJECTED_BY_DATASET_PHYSICAL_ROOT",
            "ocr": "NO_ENTRYPOINT_IN_IMPLEMENTATION",
            "external_api": "NO_CLIENT_AND_OFFLINE_LOCAL_FILES_ONLY",
        },
    }


def verify_frame_pack_reference(
    cfg: dict[str, Any],
    preflight: dict[str, Any],
    dataset: str,
    input_row: dict[str, Any],
    record_input: dict[str, Any],
) -> None:
    manifest, _, manifest_sha256 = strict_validate_frame_pack(
        root_path(cfg["datasets"][dataset]["frame_pack_root"]),
        input_row["video_id"],
        frame_pack_binding_for(cfg, preflight, dataset, input_row),
    )
    assert_equal(
        manifest_sha256,
        record_input["frame_pack_manifest_sha256"],
        "checkpoint frame-pack manifest",
    )
    assert_equal(
        [row["sha256"] for row in manifest["frames"]],
        record_input["frame_sha256"],
        "checkpoint frame hashes",
    )
    for field in FRAME_PACK_METADATA_KEYS:
        assert_equal(
            record_input[field], manifest[field], f"checkpoint frame metadata {field}"
        )


def load_checkpoint(
    path: Path,
    schema_path: str,
    dataset: str,
    input_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
    preflight: dict[str, Any],
    lineage: dict[str, str],
) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    input_by_id = {row["video_id"]: row for row in input_rows}
    rank_by_id = {row["video_id"]: rank for rank, row in enumerate(input_rows)}
    if not path.exists():
        return records
    if not path.is_dir():
        raise RuntimeError("HALT_INVALID_FREEZE: checkpoint root is not a directory")
    checkpoint_files = sorted(path.glob("*.json"))
    for checkpoint_file in checkpoint_files:
            with checkpoint_file.open("r", encoding="utf-8") as handle:
                record = json.load(handle)
            validate_schema(record, schema_path, f"checkpoint {dataset}:{checkpoint_file.name}")
            if record["dataset"] != dataset or record["run_id"] != RUN_ID:
                raise RuntimeError("HALT_INVALID_FREEZE: foreign checkpoint record")
            key = (record["video_id"], record["prompt_form"])
            if key in records:
                raise RuntimeError(f"HALT_INVALID_FREEZE: duplicate checkpoint key {key}")
            if record["video_id"] not in input_by_id:
                raise RuntimeError("HALT_INVALID_FREEZE: checkpoint ID outside allowlist")
            expected_sequence = (
                rank_by_id[record["video_id"]] * 2
                + PROMPT_FORMS.index(record["prompt_form"])
            )
            assert_equal(record["sequence_index"], expected_sequence, "checkpoint sequence")
            assert_equal(
                sha256_bytes(record["raw_output"].encode("utf-8")),
                record["raw_output_sha256"],
                "checkpoint raw output hash",
            )
            assert_equal(
                record["parsed"],
                parse_teacher_response(record["raw_output"], dataset),
                "checkpoint parsed replay",
            )
            assert_equal(
                record["provenance"],
                expected_prompt_provenance(cfg, lineage, record["prompt_form"]),
                "checkpoint provenance",
            )
            source = input_by_id[record["video_id"]]["source"]
            assert_equal(record["input"]["transcript_sha256"], source["transcript_sha256"], "checkpoint transcript")
            assert_equal(record["input"]["video_sha256"], source["video_sha256"], "checkpoint video")
            verify_frame_pack_reference(
                cfg,
                preflight,
                dataset,
                input_by_id[record["video_id"]],
                record["input"],
            )
            expected_filename = (
                f"{record['sequence_index']:03d}_{record['prompt_form']}.json"
            )
            assert_equal(checkpoint_file.name, expected_filename, "checkpoint filename")
            records[key] = record
    for video_id in input_by_id:
        pair = [
            records.get((video_id, form))
            for form in PROMPT_FORMS
            if (video_id, form) in records
        ]
        if len(pair) == 2:
            assert_equal(
                pair[0]["input"]["frame_pack_manifest_sha256"],
                pair[1]["input"]["frame_pack_manifest_sha256"],
                "A/B frame-pack identity",
            )
            assert_equal(
                pair[0]["input"]["frame_sha256"],
                pair[1]["input"]["frame_sha256"],
                "A/B frame payload identity",
            )
    return records


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    target = path / f"{record['sequence_index']:03d}_{record['prompt_form']}.json"
    payload = canonical_json_bytes(record) + b"\n"
    if target.exists():
        assert_equal(sha256_file(target), sha256_bytes(payload), "checkpoint idempotency")
        return
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", dir=str(path), prefix=target.name + ".tmp.", delete=False
        ) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temp_name, target)
        Path(temp_name).unlink()
        temp_name = ""
    except Exception:
        if temp_name:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass
        raise


def teacher_mean_embedding(text: str, tokenizer: Any, embedding: Any, torch: Any) -> list[float]:
    ids = tokenizer(text, add_special_tokens=False, return_tensors="pt")["input_ids"]
    if ids.numel() == 0:
        return [0.0] * int(embedding.embedding_dim)
    ids = ids.to(embedding.weight.device)
    with torch.no_grad():
        value = embedding(ids).mean(dim=1)[0].float().cpu().tolist()
    return [float(item) for item in value]


def apply_role(embedding: list[float], role_map: dict[str, Any]) -> list[float]:
    if len(embedding) != role_map["teacher_dim"]:
        raise RuntimeError("HALT_INVALID_FREEZE: teacher hidden dimension mismatch")
    projected = [
        float(sign) * embedding[index]
        for index, sign in zip(role_map["indices"], role_map["signs"])
    ]
    return safe_vector(projected)


def fixed_projection(matrix: Any, vector: list[float]) -> list[float]:
    import numpy as np

    array = np.asarray(vector, dtype=np.float32)
    value = matrix @ array
    norm = float(np.linalg.norm(value))
    normalized = value / max(norm, EPS) if norm > EPS else np.zeros_like(value)
    return [float(item) for item in normalized.tolist()] + [1.0 if norm <= EPS else 0.0]


def compose_features(
    slot_rows: dict[str, dict[str, Any]],
    fallback_applicability: dict[str, dict[str, Any]],
    role_maps: dict[str, dict[str, Any]],
    tokenizer: Any,
    embedding: Any,
    torch: Any,
    le3_matrix: Any,
    additive_matrix: Any,
) -> tuple[dict[str, Any], dict[str, dict[str, str]], dict[str, bool]]:
    control_renders: dict[str, dict[str, str]] = {}
    for control in ("FULL", "STATE_ONLY", "STATE_BLIND", "FALLBACK_COLLAPSE"):
        rendered = {}
        for slot in SLOTS:
            row = slot_rows[slot]
            if control == "FULL":
                text = f"{row['content']}<fallback={row['state']}>"
            elif control == "STATE_ONLY":
                text = f"NO_CONTENT_{slot}<fallback={row['state']}>"
            elif control == "STATE_BLIND":
                text = row["content"]
            else:
                if (
                    fallback_applicability[slot]["collapse"]
                    == "NA_DEGENERATE_EXACT"
                ):
                    text = f"{row['content']}<fallback={row['state']}>"
                else:
                    text = f"{row['content']}<fallback=unavailable>"
            rendered[slot] = text
        control_renders[control] = rendered

    features: dict[str, Any] = {}
    full_u: dict[str, list[float]] = {}
    collapse_exact_by_slot: dict[str, bool] = {}
    for control, rendered in control_renders.items():
        vectors = {}
        for slot in SLOTS:
            if (
                control == "FALLBACK_COLLAPSE"
                and fallback_applicability[slot]["collapse"]
                == "NA_DEGENERATE_EXACT"
            ):
                vectors[slot] = full_u[slot]
            else:
                mean = teacher_mean_embedding(rendered[slot], tokenizer, embedding, torch)
                vectors[slot] = apply_role(mean, role_maps[slot])
        q4 = q_product([vectors[slot] for slot in SLOTS])
        features[control] = {"q4": f32le_b64(q4)}
        if control == "FULL":
            full_u = vectors
            features[control]["slots"] = {
                slot: f32le_b64(vectors[slot]) for slot in SLOTS
            }
        if control == "FALLBACK_COLLAPSE":
            features[control]["slots"] = {
                slot: f32le_b64(vectors[slot]) for slot in SLOTS
            }
            for slot in SLOTS:
                is_degenerate = (
                    fallback_applicability[slot]["collapse"]
                    == "NA_DEGENERATE_EXACT"
                )
                collapse_exact_by_slot[slot] = (
                    control_renders["FALLBACK_COLLAPSE"][slot]
                    == control_renders["FULL"][slot]
                    and features["FALLBACK_COLLAPSE"]["slots"][slot]
                    == features["FULL"]["slots"][slot]
                )
                if is_degenerate and not collapse_exact_by_slot[slot]:
                    raise RuntimeError(
                        f"HALT_INVALID_FREEZE: {slot} degenerate collapse is not byte-exact"
                    )

    subsets = (
        ("S",), ("P",), ("T",), ("H",),
        ("S", "P"), ("S", "T"), ("S", "H"), ("P", "T"), ("P", "H"), ("T", "H"),
        ("S", "P", "T"), ("S", "P", "H"), ("S", "T", "H"), ("P", "T", "H"),
    )
    le3_input = []
    for subset in subsets:
        le3_input.extend(q_product([full_u[slot] for slot in subset]))
    if len(le3_input) != LE3_INPUT_DIM:
        raise RuntimeError("HALT_INVALID_FREEZE: LE3 input dimension")
    additive_input = []
    for slot in SLOTS:
        additive_input.extend(full_u[slot])
    if len(additive_input) != ADDITIVE_INPUT_DIM:
        raise RuntimeError("HALT_INVALID_FREEZE: additive input dimension")
    features["LOWER_ORDER_LE3"] = {"q": f32le_b64(fixed_projection(le3_matrix, le3_input))}
    features["ADDITIVE"] = {"q": f32le_b64(fixed_projection(additive_matrix, additive_input))}
    features["STANCE_ONLY"] = {"q": f32le_b64(q_product([full_u["T"]]))}
    features["HARM_ONLY"] = {"q": f32le_b64(q_product([full_u["H"]]))}
    return features, control_renders, collapse_exact_by_slot


def canonicalize_dataset(
    cfg: dict[str, Any],
    dataset: str,
    input_rows: list[dict[str, Any]],
    prompt_records: dict[tuple[str, str], dict[str, Any]],
    model: Any,
    processor: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import numpy as np
    import torch

    role_maps = {
        slot: load_json(cfg["maps"]["role_maps"][slot]) for slot in SLOTS
    }
    for slot, payload in role_maps.items():
        claimed = payload["payload_sha256"]
        body = dict(payload)
        body.pop("payload_sha256")
        assert_equal(sha256_obj(body), claimed, f"role payload {slot}")
    le3_path = root_path(cfg["maps"]["le3_payload_path"])
    additive_path = root_path(cfg["maps"]["additive_payload_path"])
    le3 = np.memmap(le3_path, dtype="<f4", mode="r", shape=(ROLE_DIM, LE3_INPUT_DIM))
    additive = np.memmap(
        additive_path, dtype="<f4", mode="r", shape=(ROLE_DIM, ADDITIVE_INPUT_DIM)
    )
    tokenizer = processor.tokenizer
    embedding = model.get_input_embeddings()
    prepared: list[dict[str, Any]] = []
    prompt_valid_count = 0
    for input_row in input_rows:
        video_id = input_row["video_id"]
        record_a = prompt_records[(video_id, "A")]
        record_b = prompt_records[(video_id, "B")]
        parsed_a = record_a["parsed"]
        parsed_b = record_b["parsed"]
        prompt_valid_count += int(parsed_a["form_valid"]) + int(parsed_b["form_valid"])
        proposition_cosine = None
        if parsed_a["slots"]["P"]["valid"] and parsed_b["slots"]["P"]["valid"]:
            left = teacher_mean_embedding(
                normalize_proposition(parsed_a["slots"]["P"]["content"]),
                tokenizer, embedding, torch,
            )
            right = teacher_mean_embedding(
                normalize_proposition(parsed_b["slots"]["P"]["content"]),
                tokenizer, embedding, torch,
            )
            proposition_cosine = cosine(left, right)
        slots = {
            slot: build_slot_reliability(
                slot, parsed_a, parsed_b,
                proposition_cosine if slot == "P" else None,
            )
            for slot in SLOTS
        }
        prepared.append({
            "input_row": input_row,
            "record_a": record_a,
            "record_b": record_b,
            "slots": slots,
        })

    state_counts = {
        slot: dict(Counter(row["slots"][slot]["state"] for row in prepared))
        for slot in SLOTS
    }
    fallback_applicability: dict[str, dict[str, Any]] = {}
    for slot in SLOTS:
        counts = {
            state: state_counts[slot].get(state, 0)
            for state in ("stable", "single_valid", "conflict", "missing")
        }
        observed = [state for state, count in counts.items() if count > 0]
        if len(observed) == 1:
            collapse = shuffle = noise = "NA_DEGENERATE_EXACT"
        elif min(counts[state] for state in observed) < 10:
            collapse = "APPLICABLE"
            shuffle = noise = "NA_LOW_SUPPORT"
        else:
            collapse = shuffle = noise = "APPLICABLE"
        fallback_applicability[slot] = {
            "state_counts": counts,
            "observed_states": observed,
            "nonempty_min_support": min(counts[state] for state in observed),
            "small_tranche_low_support_threshold": 10,
            "collapse": collapse,
            "shuffle": shuffle,
            "noise": noise,
            "state_only": "MANDATORY",
            "state_blind": "MANDATORY",
        }
    applicability_payload = {
        "schema_version": "c04_fallback_applicability_v8",
        "run_id": RUN_ID,
        "dataset": dataset,
        "sealed_tranche_size": SELECT_N,
        "frozen_before_feature_materialization": True,
        "slots": fallback_applicability,
    }
    applicability_payload["payload_sha256"] = sha256_obj(applicability_payload)

    records = []
    for prepared_row in prepared:
        input_row = prepared_row["input_row"]
        record_a = prepared_row["record_a"]
        record_b = prepared_row["record_b"]
        slots = prepared_row["slots"]
        video_id = input_row["video_id"]
        features, renders, collapse_exact = compose_features(
            slots,
            fallback_applicability,
            role_maps,
            tokenizer,
            embedding,
            torch,
            le3,
            additive,
        )
        record = {
            "schema_version": "c04_a0t_small_canonical_record_v8",
            "run_id": RUN_ID,
            "dataset": dataset,
            "video_id": video_id,
            "prompt_record_sha256": {
                "A": sha256_obj(record_a),
                "B": sha256_obj(record_b),
            },
            "slots": slots,
            "control_renders": renders,
            "fallback_applicability_sha256": applicability_payload["payload_sha256"],
            "collapse_exact_by_slot": collapse_exact,
            "features": features,
            "frame_decode_failed": record_a["input"]["frame_decode_failed"],
        }
        validate_schema(record, cfg["schemas"]["canonical_record"], f"canonical {dataset}/{video_id}")
        records.append(record)

    slot_rates: dict[str, Any] = {}
    rate_pass = True
    for slot in SLOTS:
        counts = {state: state_counts[slot].get(state, 0) for state in (
            "stable", "single_valid", "conflict", "missing"
        )}
        usable = counts["stable"] + counts["single_valid"]
        nonfallback = [
            record["slots"][slot]["content"]
            for record in records
            if record["slots"][slot]["state"] in ("stable", "single_valid")
        ]
        max_frequency = (
            max(Counter(nonfallback).values()) / len(nonfallback) if nonfallback else 1.0
        )
        slot_pass = (
            usable / SELECT_N >= 0.85
            and counts["missing"] / SELECT_N <= 0.10
            and counts["conflict"] / SELECT_N <= 0.20
            and max_frequency <= 0.90
        )
        rate_pass = rate_pass and slot_pass
        slot_rates[slot] = {
            "counts": counts,
            "usable_rate": usable / SELECT_N,
            "missing_rate": counts["missing"] / SELECT_N,
            "conflict_rate": counts["conflict"] / SELECT_N,
            "max_nonfallback_value_frequency": max_frequency,
            "passed": slot_pass,
        }
    joint = sum(
        all(record["slots"][slot]["state"] in ("stable", "single_valid") for slot in SLOTS)
        for record in records
    ) / SELECT_N
    rate_pass = rate_pass and joint >= 0.60
    reliability = {
        "prompt_parse_rate": prompt_valid_count / (2 * SELECT_N),
        "slots": slot_rates,
        "joint_all_four_usable_rate": joint,
        "joint_threshold": 0.60,
        "fallback_applicability": applicability_payload,
        "semantic_reliability_passed": rate_pass,
        "kill_state_if_failed": "KILL_C04_TEACHER_SEMANTIC_RELIABILITY",
    }
    return records, reliability


def load_jsonl_exact(path: Path, expected_count: int, label: str) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise RuntimeError(f"HALT_INVALID_FREEZE: torn {label} line {line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"HALT_INVALID_FREEZE: non-object {label} row")
            rows.append(value)
    assert_equal(len(rows), expected_count, f"{label} count")
    return rows


def idempotent_complete(
    cfg: dict[str, Any],
    preflight: dict[str, Any],
    lineage: dict[str, str],
    inputs: dict[str, list[dict[str, Any]]],
) -> bool:
    manifest_path = root_path(cfg["paths"]["seal_manifest"])
    if not manifest_path.exists():
        return False
    manifest = load_json(cfg["paths"]["seal_manifest"])
    body = dict(manifest)
    claimed = body.pop("payload_sha256", None)
    assert_equal(sha256_obj(body), claimed, "existing seal payload")
    assert_equal(manifest["run_id"], RUN_ID, "existing seal run id")
    assert_equal(manifest["implementation_version"], "v8_prospective", "seal implementation")
    assert_equal(manifest["lineage"], lineage, "existing seal lineage")
    if manifest["terminal_state"] not in {
        "SEALED_PRELABEL_RELIABILITY_PASS",
        "KILL_C04_TEACHER_SEMANTIC_RELIABILITY",
    }:
        raise RuntimeError("HALT_INVALID_FREEZE: nonterminal existing seal")
    assert_equal(manifest["labels_opened"], False, "existing seal labels")
    assert_equal(
        manifest["resource_final_state_required_before_any_downstream_review"],
        True,
        "existing seal terminal-resource gate",
    )
    for relative, expected in manifest["sealed_output_hashes"].items():
        assert_equal(sha256_file(root_path(relative)), expected, f"existing sealed output {relative}")
    access_path = root_path(cfg["paths"]["producer_access_ledger"])
    provisional_path = root_path(cfg["paths"]["provisional_gpu_usage"])
    assert_equal(sha256_file(access_path), manifest["access_ledger_sha256"], "access ledger")
    assert_equal(
        sha256_file(provisional_path),
        manifest["provisional_gpu_usage_sha256"],
        "provisional usage",
    )
    access = load_json(cfg["paths"]["producer_access_ledger"])
    provisional = load_json(cfg["paths"]["provisional_gpu_usage"])
    assert_equal(access["lineage"], lineage, "access lineage")
    assert_equal(provisional["lineage"], lineage, "provisional lineage")
    assert_equal(provisional["slurm_job_id"], os.environ["SLURM_JOB_ID"], "provisional job")
    assert_equal(provisional["allocated_gpu_count"], 1, "provisional GPU count")
    if provisional["provisional_gpu_seconds"] > cfg["resources"]["small_cap_gpu_seconds"]:
        raise RuntimeError("HALT_RESOURCE_CAP: sealed provisional usage exceeds cap")

    prompt_rows = load_jsonl_exact(
        root_path(cfg["paths"]["sealed_prompt_records"]), 800, "sealed prompt"
    )
    canonical_rows = load_jsonl_exact(
        root_path(cfg["paths"]["sealed_canonical_bank"]), 400, "sealed canonical"
    )
    expected_prompt_keys = []
    input_by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset in DATASETS:
        for input_row in inputs[dataset]:
            input_by_key[(dataset, input_row["video_id"])] = input_row
            expected_prompt_keys.extend(
                (dataset, input_row["video_id"], form) for form in PROMPT_FORMS
            )
    actual_prompt_keys = []
    prompt_by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in prompt_rows:
        validate_schema(row, cfg["schemas"]["prompt_record"], "sealed prompt")
        key = (row["dataset"], row["video_id"], row["prompt_form"])
        actual_prompt_keys.append(key)
        prompt_by_key[key] = row
        assert_equal(
            row["raw_output_sha256"],
            sha256_bytes(row["raw_output"].encode("utf-8")),
            "sealed raw output",
        )
        assert_equal(
            row["parsed"],
            parse_teacher_response(row["raw_output"], row["dataset"]),
            "sealed parsed replay",
        )
        assert_equal(
            row["provenance"],
            expected_prompt_provenance(cfg, lineage, row["prompt_form"]),
            "sealed prompt provenance",
        )
        verify_frame_pack_reference(
            cfg,
            preflight,
            row["dataset"],
            input_by_key[(row["dataset"], row["video_id"])],
            row["input"],
        )
    assert_equal(actual_prompt_keys, expected_prompt_keys, "sealed prompt order")
    for sequence_index, row in enumerate(prompt_rows):
        assert_equal(row["sequence_index"], sequence_index % (2 * SELECT_N), "sealed sequence")
    for dataset in DATASETS:
        checkpoint_records = load_checkpoint(
            root_path(cfg["datasets"][dataset]["checkpoint_path"]),
            cfg["schemas"]["prompt_record"],
            dataset,
            inputs[dataset],
            cfg,
            preflight,
            lineage,
        )
        for input_row in inputs[dataset]:
            for form in PROMPT_FORMS:
                key = (dataset, input_row["video_id"], form)
                assert_equal(
                    sha256_obj(checkpoint_records[(input_row["video_id"], form)]),
                    sha256_obj(prompt_by_key[key]),
                    "sealed/checkpoint prompt identity",
                )
            assert_equal(
                prompt_by_key[(dataset, input_row["video_id"], "A")]["input"][
                    "frame_pack_manifest_sha256"
                ],
                prompt_by_key[(dataset, input_row["video_id"], "B")]["input"][
                    "frame_pack_manifest_sha256"
                ],
                "sealed A/B frame-pack identity",
            )

    expected_canonical_keys = [
        (dataset, input_row["video_id"])
        for dataset in DATASETS
        for input_row in inputs[dataset]
    ]
    actual_canonical_keys = []
    for row in canonical_rows:
        validate_schema(row, cfg["schemas"]["canonical_record"], "sealed canonical")
        key = (row["dataset"], row["video_id"])
        actual_canonical_keys.append(key)
        assert_equal(
            row["prompt_record_sha256"]["A"],
            sha256_obj(prompt_by_key[(key[0], key[1], "A")]),
            "canonical prompt A",
        )
        assert_equal(
            row["prompt_record_sha256"]["B"],
            sha256_obj(prompt_by_key[(key[0], key[1], "B")]),
            "canonical prompt B",
        )
    assert_equal(actual_canonical_keys, expected_canonical_keys, "sealed canonical order")
    assert_equal(merkle_root(prompt_rows), manifest["prompt_merkle_root"], "prompt Merkle")
    assert_equal(
        merkle_root(canonical_rows), manifest["canonical_merkle_root"], "canonical Merkle"
    )
    reliability_pass = all(
        manifest["reliability"][dataset]["semantic_reliability_passed"]
        for dataset in DATASETS
    )
    expected_terminal = (
        "SEALED_PRELABEL_RELIABILITY_PASS"
        if reliability_pass
        else "KILL_C04_TEACHER_SEMANTIC_RELIABILITY"
    )
    assert_equal(manifest["terminal_state"], expected_terminal, "seal terminal/reliability")
    return True


def assert_teacher_visible_precondition(
    inputs: dict[str, list[dict[str, Any]]],
) -> tuple[tuple[str, ...], dict[str, Any]]:
    """Fail-closed precondition on every teacher-visible field of the tranche.

    Runs over all 400 identifiers and both prompt forms *before* the model is
    loaded, so it fires before any teacher forward rather than partway through
    one.  The same assertion is repeated per item immediately before each
    forward, where the assembled frame payload also becomes checkable.
    """
    selected = {dataset: [row["video_id"] for row in inputs[dataset]] for dataset in DATASETS}
    forbidden = forbidden_teacher_visible_tokens(selected)
    # Placeholder frames, so this pass runs through exactly the same assembly and
    # extraction path as the per-forward call site rather than comparing a list of
    # strings against one it built itself.  They are deliberately not str/bytes/Path,
    # which is the class the extractor rejects.
    placeholder_frames = [object() for _ in range(NUM_FRAMES)]
    checked = 0
    for dataset in DATASETS:
        for input_row in inputs[dataset]:
            for prompt_form in PROMPT_FORMS:
                messages = build_messages(
                    placeholder_frames, input_row["transcript"], prompt_form
                )
                assert_teacher_visible_containment(
                    dataset,
                    input_row["video_id"],
                    prompt_form,
                    input_row["transcript"],
                    teacher_visible_texts(messages),
                    forbidden,
                )
                checked += 1
    summary = {
        "banned_token_count": len(forbidden),
        "prompt_renderings_checked_before_model_load": checked,
        "label_bearing_id_prefixes_banned": {
            dataset: list(LABEL_BEARING_ID_SUBSTRINGS[dataset]) for dataset in DATASETS
        },
        "hatemm_identifiers_are_label_bearing": True,
        "checked_again_per_item_before_every_forward": True,
    }
    return forbidden, summary


def main() -> int:
    cfg = load_json(CONFIG_RELATIVE)
    preflight, lineage, allocation_claim, guard = verify_execution_lineage(cfg)
    audit = AccessAudit()
    inputs: dict[str, list[dict[str, Any]]] = {}
    access_counters: dict[str, dict[str, int]] = {}
    for dataset in DATASETS:
        inputs[dataset], access_counters[dataset] = load_selected_inputs(
            cfg, preflight, dataset, audit
        )
    forbidden_tokens, containment_summary = assert_teacher_visible_precondition(inputs)
    if idempotent_complete(cfg, preflight, lineage, inputs):
        return 0

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    torch.manual_seed(0)
    snapshot = cfg["model"]["snapshot_path"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        snapshot,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map=None,
        local_files_only=True,
    )
    model.to(torch.device("cuda")).eval()
    # THE TEACHER-INPUT CHANGE.  v7 built the processor with no max_pixels, so
    # frames entered the vision tower at native resolution and job 13852 died
    # trying to allocate 110.50 GiB of fp32 vision-attention scores on the first
    # MHC_zh item.  151200 = 360*420 is the cap every deployed Qwen2.5-VL
    # entrypoint in this repository already uses, so this makes the teacher's
    # visual input the same kind of input the rest of the project produces.
    processor = AutoProcessor.from_pretrained(
        snapshot, local_files_only=True, max_pixels=TEACHER_MAX_PIXELS
    )
    assert_equal(
        int(processor.image_processor.max_pixels),
        TEACHER_MAX_PIXELS,
        "processor max_pixels",
    )

    observed_visual_tokens: list[int] = []

    @torch.no_grad()
    def one_forward(
        dataset: str,
        video_id: str,
        frames: list[Any],
        transcript: str,
        prompt_form: str,
        expected_visual_tokens: int,
    ) -> str:
        deadline_check(guard, f"the teacher forward for {dataset} prompt {prompt_form}")
        messages = build_messages(frames, transcript, prompt_form)
        # Fail-closed: nothing the teacher can read may carry an identifier or a
        # label-bearing string, and the assembled payload must be exactly the
        # frozen template on the frozen transcript plus eight non-path frames.
        assert_teacher_visible_containment(
            dataset,
            video_id,
            prompt_form,
            transcript,
            teacher_visible_texts(messages),
            forbidden_tokens,
        )
        chat = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        prepared = processor(
            text=[chat], images=None, videos=[frames], return_tensors="pt"
        )
        # Fail-closed BEFORE the vision tower.  This is the check whose absence
        # cost the v7 allocation: the geometry is known here, on the CPU side of
        # the forward, and an item above the ceiling is refused instead of being
        # handed to an attention that would try to allocate more memory than the
        # card has.  The count is PRE-merge, which is what vision SDPA runs on.
        tokens = visual_patch_tokens(prepared["video_grid_thw"][0].tolist())
        assert_visual_token_ceiling(dataset, video_id, tokens)
        if tokens != expected_visual_tokens:
            raise RuntimeError(
                f"HALT_VISUAL_TOKEN_CEILING: {dataset}/{video_id} measured "
                f"{tokens} pre-merge visual tokens against the "
                f"{expected_visual_tokens} the CPU preflight froze for it"
            )
        observed_visual_tokens.append(tokens)
        tensors = prepared.to(model.device)
        output = model.generate(
            **tensors,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
        )
        generated = output[:, tensors["input_ids"].shape[1]:]
        return processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()

    all_prompt_rows: dict[str, dict[tuple[str, str], dict[str, Any]]] = {}
    total_teacher_calls = 0
    frame_packs_loaded = 0
    completed_by_dataset: dict[str, int] = {dataset: 0 for dataset in DATASETS}
    try:
        for dataset in DATASETS:
            checkpoint = root_path(cfg["datasets"][dataset]["checkpoint_path"])
            records = load_checkpoint(
                checkpoint,
                cfg["schemas"]["prompt_record"],
                dataset,
                inputs[dataset],
                cfg,
                preflight,
                lineage,
            )
            completed_by_dataset[dataset] = len(records)
            for rank, input_row in enumerate(inputs[dataset]):
                video_id = input_row["video_id"]
                needed = [form for form in PROMPT_FORMS if (video_id, form) not in records]
                if not needed:
                    continue
                # The item boundary: stop here rather than inside a decode or a
                # forward, so a breach can never leave a partial artifact.
                deadline_check(guard, f"item {dataset}/{rank}")
                decoded = load_frozen_frame_pack(
                    cfg, preflight, dataset, input_row, audit
                )
                frame_packs_loaded += 1
                for prompt_form in PROMPT_FORMS:
                    key = (video_id, prompt_form)
                    if key in records:
                        continue
                    sequence = rank * 2 + PROMPT_FORMS.index(prompt_form)
                    started = time.monotonic()
                    audit.event(f"LOCAL_TEACHER_PROMPT_{prompt_form}", dataset, video_id)
                    raw = one_forward(
                        dataset,
                        video_id,
                        decoded["frames"],
                        input_row["transcript"],
                        prompt_form,
                        decoded["preflight_visual_patch_tokens"],
                    )
                    transport_error = ""
                    total_teacher_calls += 1
                    parsed = parse_teacher_response(raw, dataset)
                    record = {
                        "schema_version": "c04_a0t_small_prompt_record_v8",
                        "run_id": RUN_ID,
                        "dataset": dataset,
                        "video_id": video_id,
                        "prompt_form": prompt_form,
                        "sequence_index": sequence,
                        "input": {
                            "transcript_sha256": input_row["source"]["transcript_sha256"],
                            "video_sha256": input_row["source"]["video_sha256"],
                            "frame_backend": decoded["backend"],
                            "total_frame_indices": decoded["total_frame_indices"],
                            "requested_indices": decoded["requested_indices"],
                            "frame_decode_failed": decoded["frame_decode_failed"],
                            "frame_pack_manifest_sha256": decoded["manifest_sha256"],
                            "frame_sha256": decoded["frame_sha256"],
                            "visual_patch_tokens": decoded[
                                "preflight_visual_patch_tokens"
                            ],
                        },
                        "provenance": expected_prompt_provenance(
                            cfg, lineage, prompt_form
                        ),
                        "raw_output": raw,
                        "raw_output_sha256": sha256_bytes(raw.encode("utf-8")),
                        "transport_error": transport_error,
                        "parsed": parsed,
                        "elapsed_seconds": time.monotonic() - started,
                        "retry_count": 0,
                    }
                    validate_schema(
                        record, cfg["schemas"]["prompt_record"],
                        f"prompt {dataset}/{video_id}/{prompt_form}",
                    )
                    append_checkpoint(checkpoint, record)
                    records[key] = record
                    completed_by_dataset[dataset] = len(records)
            expected_keys = {
                (row["video_id"], form)
                for row in inputs[dataset]
                for form in PROMPT_FORMS
            }
            if set(records) != expected_keys or len(records) != 2 * SELECT_N:
                raise RuntimeError(f"HALT_INVALID_C04_V4: incomplete {dataset} prompt cache")
            all_prompt_rows[dataset] = records
        guard.require_remaining(
            int(cfg["resources"]["guard_seal_reserve_seconds"]),
            "the canonicalization and seal phase",
        )
    except BudgetDeadlineReached as breach:
        # Accounting only.  No seal, no reliability rate, no verdict, and no
        # already-written record is touched: the guard stopped before an item.
        digest = publish_budget_breach_record(
            cfg,
            lineage,
            guard,
            completed_by_dataset,
            total_teacher_calls,
            frame_packs_loaded,
            str(breach),
        )
        print(f"{breach}", file=sys.stderr)
        print(
            f"HALT_RESOURCE_CAP: budget breach record published, payload_sha256={digest}",
            file=sys.stderr,
        )
        return BUDGET_BREACH_EXIT_CODE

    canonical_by_dataset: dict[str, list[dict[str, Any]]] = {}
    reliability_by_dataset: dict[str, Any] = {}
    for dataset in DATASETS:
        canonical_by_dataset[dataset], reliability_by_dataset[dataset] = canonicalize_dataset(
            cfg, dataset, inputs[dataset], all_prompt_rows[dataset], model, processor
        )

    prompt_rows = []
    canonical_rows = []
    for dataset in DATASETS:
        for input_row in inputs[dataset]:
            for form in PROMPT_FORMS:
                prompt_rows.append(all_prompt_rows[dataset][(input_row["video_id"], form)])
        canonical_rows.extend(canonical_by_dataset[dataset])
    if len(prompt_rows) != 800 or len(canonical_rows) != 400:
        raise RuntimeError("HALT_INVALID_C04_V4: global record count mismatch")

    terminal = (
        "SEALED_PRELABEL_RELIABILITY_PASS"
        if all(
            reliability_by_dataset[dataset]["semantic_reliability_passed"]
            for dataset in DATASETS
        )
        else "KILL_C04_TEACHER_SEMANTIC_RELIABILITY"
    )
    prompt_rel = cfg["paths"]["sealed_prompt_records"]
    canonical_rel = cfg["paths"]["sealed_canonical_bank"]
    access_rel = cfg["paths"]["producer_access_ledger"]
    provisional_rel = cfg["paths"]["provisional_gpu_usage"]
    with Path("/proc/uptime").open("r", encoding="ascii") as handle:
        current_uptime = int(float(handle.read().split()[0]))
    elapsed = max(
        0,
        current_uptime - int(allocation_claim["allocation_entry_uptime_seconds"]),
    )
    access = {
        "schema_version": "c04_access_ledger_v8",
        "run_id": RUN_ID,
        "implementation_version": "v8_prospective",
        "stage": "teacher_and_prelabel_seal",
        "lineage": lineage,
        "per_dataset_projected_field_counters": access_counters,
        "teacher_calls_this_invocation": total_teacher_calls,
        "sealed_prompt_record_count": 800,
        # v8 never creates a frame pack: every pack is frozen by the CPU
        # preflight and hash-pinned in the preflight manifest.  This counter
        # is the number LOADED, and the producer has no decode entrypoint.
        "frame_packs_created_this_invocation": 0,
        "frozen_frame_packs_loaded_this_invocation": frame_packs_loaded,
        "producer_may_decode_a_video": False,
        "label_value_materialized_from_projector": sum(
            access_counters[dataset]["label_value_materialized"] for dataset in DATASETS
        ),
        "guarded_runtime_evidence": audit.snapshot(),
        "teacher_visible_containment": containment_summary,
        "static_surface_assertions_are_not_runtime_counters": True,
    }
    provisional = build_provisional_gpu_usage(
        lineage,
        os.environ["SLURM_JOB_ID"],
        elapsed,
        guard.accounting_snapshot(),
        sha256_file(root_path(cfg["paths"]["gpu_ledger"])),
        sha256_file(root_path(CAMPAIGN_GPU_LEDGER_RELATIVE)),
    )
    prompt_payload = b"".join(canonical_json_bytes(row) + b"\n" for row in prompt_rows)
    canonical_payload = b"".join(
        canonical_json_bytes(row) + b"\n" for row in canonical_rows
    )
    access_payload = canonical_json_bytes(access) + b"\n"
    provisional_payload = canonical_json_bytes(provisional) + b"\n"
    sealed_hashes = {
        prompt_rel: sha256_bytes(prompt_payload),
        canonical_rel: sha256_bytes(canonical_payload),
    }
    seal = {
        "schema_version": "c04_a0t_small_seal_manifest_v8",
        "run_id": RUN_ID,
        "implementation_version": "v8_prospective",
        "lineage": lineage,
        "terminal_state": terminal,
        "labels_opened": False,
        "label_access_allowed_after_this_seal_only_if_reliability_passes": terminal
        == "SEALED_PRELABEL_RELIABILITY_PASS",
        "prompt_record_count": len(prompt_rows),
        "canonical_record_count": len(canonical_rows),
        "dataset_counts": {dataset: SELECT_N for dataset in DATASETS},
        "reliability": reliability_by_dataset,
        "prompt_merkle_root": merkle_root(prompt_rows),
        "canonical_merkle_root": merkle_root(canonical_rows),
        "sealed_output_hashes": sealed_hashes,
        "access_ledger_sha256": sha256_bytes(access_payload),
        "provisional_gpu_usage_sha256": sha256_bytes(provisional_payload),
        "resource_final_state_required_before_any_downstream_review": True,
        "no_performance_claim": True,
        "no_retry_redraw_prompt_rewrite": True,
    }
    seal["payload_sha256"] = sha256_obj(seal)
    seal_payload = canonical_json_bytes(seal) + b"\n"
    seal_dir = root_path(cfg["paths"]["seal_manifest"]).parent
    if seal_dir.exists():
        raise RuntimeError("HALT_INVALID_FREEZE: incomplete or colliding seal namespace")
    seal_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_seal = Path(
        tempfile.mkdtemp(prefix=seal_dir.name + ".tmp.", dir=str(seal_dir.parent))
    )
    staged_seal = {
        prompt_rel: prompt_payload,
        canonical_rel: canonical_payload,
        access_rel: access_payload,
        provisional_rel: provisional_payload,
        cfg["paths"]["seal_manifest"]: seal_payload,
    }
    try:
        for relative, payload in sorted(staged_seal.items()):
            final_path = root_path(relative)
            try:
                suffix = final_path.relative_to(seal_dir)
            except ValueError as error:
                raise RuntimeError(
                    f"HALT_INVALID_FREEZE: sealed path outside seal namespace {relative}"
                ) from error
            target = temp_seal / suffix
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_fsynced(target, payload)
        directory_fd = os.open(str(temp_seal), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temp_seal, seal_dir)
    except Exception:
        shutil.rmtree(temp_seal, ignore_errors=True)
        raise
    if not idempotent_complete(cfg, preflight, lineage, inputs):
        raise RuntimeError("HALT_INVALID_FREEZE: post-publication seal verification failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
