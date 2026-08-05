#!/usr/bin/env python
"""Persistent single-allocation GPU ledger for C04-A0T-SMALL-v1 impl-v2.

The wrapper invokes `claim` as its first Python operation and installs an EXIT
trap that invokes `mark-exit`.  Every claim reconciles all prior job rows with
sacct (terminal and active partial time), creates an allocation-entry marker,
consumes the one-use ticket, and reserves the remaining cap before model/data
work.  This module never submits, releases, chains, or resubmits a job.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from c04_a0t_small_v1_v2_common import (  # noqa: E402
    CONFIG_RELATIVE,
    RUN_ID,
    canonical_json_bytes,
    config_contract_sha256,
    exclusive_publish_json,
    load_json,
    require_exact_keys,
    root_path,
    sha256_file,
    sha256_obj,
    verify_closure_hash,
    verify_gpu_execution_authorization,
    verify_historical_code_resource_authorization,
    verify_payload_review,
    verify_preflight_manifest,
)

TERMINAL_PREFIXES = (
    "COMPLETED",
    "FAILED",
    "CANCELLED",
    "TIMEOUT",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "NODE_FAIL",
    "BOOT_FAIL",
    "DEADLINE",
)


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def uptime_seconds() -> int:
    with Path("/proc/uptime").open("r", encoding="ascii") as handle:
        return int(float(handle.read().split()[0]))


def atomic_replace_json(path: Path, value: dict[str, Any]) -> None:
    payload = canonical_json_bytes(value) + b"\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_name = ""
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=path.name + ".tmp.",
            delete=False,
        ) as handle:
            temp_name = handle.name
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    except Exception:
        if temp_name:
            try:
                Path(temp_name).unlink()
            except FileNotFoundError:
                pass
        raise


def validate_gpu_environment(cfg: dict[str, Any]) -> str:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    if not job_id.isdigit():
        raise RuntimeError("HALT_RESOURCE_CAP: numeric SLURM_JOB_ID required")
    if os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_DEPENDENCY"):
        raise RuntimeError("HALT_RESOURCE_CAP: arrays/dependencies forbidden")
    assert_equal(cfg["run"]["run_id"], RUN_ID, "config run id")
    assert_equal(cfg["run"]["implementation_version"], "v2_prospective", "implementation")
    assert_equal(cfg["resources"]["gpu_count"], 1, "GPU count")
    assert_equal(cfg["resources"]["cpus"], 8, "CPU count")
    assert_equal(cfg["resources"]["ram_gb"], 64, "RAM")
    assert_equal(cfg["resources"]["small_cap_gpu_seconds"], 7200, "GPU cap")
    assert_equal(cfg["resources"]["watchdog_reserve_seconds"], 120, "reserve")
    required_true = (
        "teacher_authorized",
        "gpu_authorized",
        "slurm_authorized",
        "small_tranche_execution_authorized",
    )
    for key in required_true:
        assert_equal(cfg["authorization"][key], True, f"authorization.{key}")
    for key in (
        "preflight_materialization_authorized",
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
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not visible or "," in visible:
        raise RuntimeError("HALT_RESOURCE_CAP: exactly one visible GPU required")
    return job_id


def verify_gpu_lineage(cfg: dict[str, Any]) -> tuple[str, str, str]:
    preflight, preflight_sha = verify_preflight_manifest(
        cfg, allow_claimed_gpu_ledger=True
    )
    verify_historical_code_resource_authorization(cfg, preflight)
    _, payload_sha = verify_payload_review(cfg, preflight, preflight_sha)
    _, gpu_auth_sha = verify_gpu_execution_authorization(
        cfg, preflight, preflight_sha, payload_sha
    )
    return preflight_sha, payload_sha, gpu_auth_sha


def parse_gpu_count(alloc_tres: str) -> int:
    count = 0
    for token in alloc_tres.split(","):
        if token.startswith("gres/gpu="):
            count = int(token.split("=", 1)[1])
        elif token.startswith("gres/gpu:") and "=" in token:
            count = int(token.rsplit("=", 1)[1])
    return count


def sacct_row(job_id: str) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "sacct", "-X", "-n", "-P", "-j", job_id,
            "-o", "JobIDRaw,ElapsedRaw,AllocTRES,State",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [line for line in completed.stdout.splitlines() if line.strip()]
    exact = [line for line in rows if line.split("|", 1)[0] == job_id]
    if len(exact) != 1:
        raise RuntimeError(f"HALT_RESOURCE_CAP: ambiguous sacct row for {job_id}")
    _, elapsed_text, alloc_tres, state = exact[0].split("|", 3)
    gpu_count = parse_gpu_count(alloc_tres)
    if gpu_count != 1:
        raise RuntimeError(f"HALT_RESOURCE_CAP: job {job_id} GPU count {gpu_count}")
    elapsed = int(elapsed_text)
    if elapsed < 0:
        raise RuntimeError(f"HALT_RESOURCE_CAP: negative elapsed for {job_id}")
    return {
        "elapsed_gpu_seconds": elapsed * gpu_count,
        "gpu_count": gpu_count,
        "state": state,
        "terminal": state.startswith(TERMINAL_PREFIXES),
    }


def validate_ledger(ledger: dict[str, Any], cfg: dict[str, Any]) -> None:
    require_exact_keys(
        ledger,
        {
            "schema_version",
            "run_id",
            "implementation_version",
            "cap_gpu_seconds",
            "ledger_revision",
            "state",
            "jobs",
            "aggregate_accounted_gpu_seconds",
            "aggregate_reconciled_terminal_gpu_seconds",
            "requires_terminal_reconciliation",
            "resubmit_authorized",
            "single_allocation_only",
            "code_resource_authorization_sha256",
            "config_contract_sha256",
            "payload_sha256",
        },
        "GPU ledger",
    )
    body = dict(ledger)
    claimed = body.pop("payload_sha256", None)
    if not isinstance(claimed, str) or sha256_obj(body) != claimed:
        raise RuntimeError("HALT_RESOURCE_CAP: ledger payload hash mismatch")
    assert_equal(ledger["schema_version"], "c04_gpu_ledger_v2", "ledger schema")
    assert_equal(ledger["run_id"], RUN_ID, "ledger run id")
    assert_equal(ledger["cap_gpu_seconds"], cfg["resources"]["small_cap_gpu_seconds"], "cap")
    assert_equal(ledger["resubmit_authorized"], False, "resubmit")
    assert_equal(ledger["single_allocation_only"], True, "single allocation")


def reconcile(ledger: dict[str, Any], current_job_id: str | None) -> dict[str, Any]:
    jobs = []
    terminal_total = 0
    accounted_total = 0
    any_nonterminal = False
    for existing in ledger["jobs"]:
        job = dict(existing)
        row = sacct_row(str(job["slurm_job_id"]))
        job["last_sacct_state"] = row["state"]
        job["last_sacct_gpu_seconds"] = row["elapsed_gpu_seconds"]
        if row["terminal"]:
            job["status"] = "SACCT_TERMINAL"
            job["accounted_gpu_seconds"] = row["elapsed_gpu_seconds"]
            job["reserved_gpu_seconds"] = 0
            job["requires_terminal_reconciliation"] = False
            terminal_total += row["elapsed_gpu_seconds"]
            accounted_total += row["elapsed_gpu_seconds"]
        else:
            any_nonterminal = True
            if current_job_id is None or str(job["slurm_job_id"]) != current_job_id:
                raise RuntimeError(
                    f"HALT_RESOURCE_CAP: active prior C04 allocation {job['slurm_job_id']}"
                )
            job["status"] = "SACCT_ACTIVE"
            job["accounted_gpu_seconds"] = row["elapsed_gpu_seconds"]
            job["requires_terminal_reconciliation"] = True
            accounted_total += max(
                row["elapsed_gpu_seconds"], int(job.get("reserved_gpu_seconds", 0))
            )
        jobs.append(job)
    ledger["jobs"] = jobs
    ledger["aggregate_reconciled_terminal_gpu_seconds"] = terminal_total
    ledger["aggregate_accounted_gpu_seconds"] = accounted_total
    ledger["requires_terminal_reconciliation"] = any_nonterminal
    if accounted_total > int(ledger["cap_gpu_seconds"]):
        raise RuntimeError("HALT_RESOURCE_CAP: reconciled/reserved aggregate exceeds cap")
    return ledger


def create_entry_marker(cfg: dict[str, Any], job_id: str, start_uptime: int) -> None:
    relative = cfg["paths"]["allocation_entry_marker"]
    path = root_path(relative)
    marker = {
        "schema_version": "c04_allocation_entry_marker_v2",
        "run_id": RUN_ID,
        "slurm_job_id": job_id,
        "allocation_entry_uptime_seconds": start_uptime,
        "claim_completed": False,
        "exit_marker_recorded": False,
    }
    if path.exists():
        existing = load_json(relative)
        assert_equal(existing["slurm_job_id"], job_id, "entry marker job id")
        assert_equal(
            existing["allocation_entry_uptime_seconds"],
            start_uptime,
            "entry marker start uptime",
        )
    else:
        exclusive_publish_json(relative, marker)


def claim(cfg: dict[str, Any], start_uptime: int) -> int:
    job_id = validate_gpu_environment(cfg)
    create_entry_marker(cfg, job_id, start_uptime)
    preflight_sha, payload_sha, gpu_auth_sha = verify_gpu_lineage(cfg)
    ledger_path = root_path(cfg["paths"]["gpu_ledger"])
    ticket_path = root_path(cfg["paths"]["resource_ticket"])
    lock_path = root_path(cfg["paths"]["gpu_ledger_lock"])
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        ledger = load_json(cfg["paths"]["gpu_ledger"])
        validate_ledger(ledger, cfg)
        ledger = reconcile(ledger, job_id)
        reconciled_body = dict(ledger)
        reconciled_body.pop("payload_sha256", None)
        ledger["payload_sha256"] = sha256_obj(reconciled_body)
        atomic_replace_json(ledger_path, ledger)
        matching = [
            job for job in ledger["jobs"] if str(job["slurm_job_id"]) == job_id
        ]
        consumption_path = root_path(cfg["paths"]["resource_consumption"])
        if matching:
            if len(matching) != 1 or not consumption_path.exists():
                raise RuntimeError("HALT_RESOURCE_CAP: corrupt same-allocation resume")
            job = matching[0]
            assert_equal(job["allocation_entry_uptime_seconds"], start_uptime, "resume uptime")
            assert_equal(job["preflight_manifest_sha256"], preflight_sha, "resume preflight")
            assert_equal(job["payload_review_sha256"], payload_sha, "resume payload review")
            assert_equal(
                job["gpu_execution_authorization_sha256"],
                gpu_auth_sha,
                "resume GPU authorization",
            )
            assert_equal(
                job["config_contract_sha256"],
                config_contract_sha256(cfg),
                "resume config contract",
            )
            assert_equal(
                job["claim_sha256"],
                load_json(cfg["paths"]["allocation_claim"])["claim_sha256"],
                "resume claim file",
            )
            remaining = int(job["watchdog_deadline_uptime_seconds"]) - uptime_seconds()
            if remaining <= 0:
                raise RuntimeError("HALT_RESOURCE_CAP: same-allocation watchdog exhausted")
            return remaining
        claim_path = root_path(cfg["paths"]["allocation_claim"])
        if not ledger["jobs"] and claim_path.exists():
            claim_record = load_json(cfg["paths"]["allocation_claim"])
            claim_body = dict(claim_record)
            claim_sha = claim_body.pop("claim_sha256", None)
            if not isinstance(claim_sha, str) or sha256_obj(claim_body) != claim_sha:
                raise RuntimeError("HALT_RESOURCE_CAP: interrupted claim hash mismatch")
            assert_equal(claim_record["slurm_job_id"], job_id, "interrupted claim job")
            assert_equal(
                claim_record["allocation_entry_uptime_seconds"],
                start_uptime,
                "interrupted claim uptime",
            )
            assert_equal(
                claim_record["preflight_manifest_sha256"],
                preflight_sha,
                "interrupted claim preflight",
            )
            assert_equal(
                claim_record["payload_review_sha256"],
                payload_sha,
                "interrupted claim payload review",
            )
            assert_equal(
                claim_record["gpu_execution_authorization_sha256"],
                gpu_auth_sha,
                "interrupted claim GPU authorization",
            )
            if not consumption_path.exists():
                exclusive_publish_json(cfg["paths"]["resource_consumption"], {
                    "schema_version": "c04_resource_ticket_consumption_v2",
                    "run_id": RUN_ID,
                    "slurm_job_id": job_id,
                    "ticket_sha256": claim_record["ticket_sha256"],
                    "allocation_claim_sha256": sha256_file(claim_path),
                    "consumed_once": True,
                })
            consumption = load_json(cfg["paths"]["resource_consumption"])
            assert_equal(consumption["slurm_job_id"], job_id, "interrupted consumption job")
            assert_equal(
                consumption["allocation_claim_sha256"],
                sha256_file(claim_path),
                "interrupted consumption claim",
            )
            ledger["ledger_revision"] = int(ledger["ledger_revision"]) + 1
            ledger["state"] = "CLAIMED_ACTIVE"
            ledger["jobs"].append({
                **claim_record,
                "status": "CLAIMED_ACTIVE",
                "gpu_count": 1,
                "accounted_gpu_seconds": 0,
                "requires_terminal_reconciliation": True,
            })
            ledger["aggregate_accounted_gpu_seconds"] = int(
                claim_record["reserved_gpu_seconds"]
            )
            ledger["requires_terminal_reconciliation"] = True
            ledger_body = dict(ledger)
            ledger_body.pop("payload_sha256", None)
            ledger["payload_sha256"] = sha256_obj(ledger_body)
            atomic_replace_json(ledger_path, ledger)
            remaining = int(claim_record["watchdog_deadline_uptime_seconds"]) - uptime_seconds()
            if remaining <= 0:
                raise RuntimeError("HALT_RESOURCE_CAP: interrupted claim watchdog exhausted")
            return remaining
        if ledger["jobs"]:
            raise RuntimeError("HALT_RESOURCE_CAP: single-use tranche already claimed")

        ticket = load_json(cfg["paths"]["resource_ticket"])
        require_exact_keys(
            ticket,
            {
                "schema_version",
                "run_id",
                "implementation_version",
                "single_use",
                "consumed",
                "authorized_slurm_allocation_count",
                "completed_gpu_seconds",
                "cap_gpu_seconds",
                "remaining_seconds",
                "watchdog_seconds",
                "issued_by_slurm_job_id",
                "no_submit_performed",
                "genesis_gpu_ledger_sha256",
                "code_resource_authorization_sha256",
                "config_contract_sha256",
                "payload_sha256",
            },
            "resource ticket",
        )
        ticket_body = dict(ticket)
        ticket_claimed = ticket_body.pop("payload_sha256", None)
        if not isinstance(ticket_claimed, str) or sha256_obj(ticket_body) != ticket_claimed:
            raise RuntimeError("HALT_RESOURCE_CAP: ticket payload hash mismatch")
        assert_equal(ticket["schema_version"], "c04_resource_ticket_v2", "ticket schema")
        assert_equal(ticket["single_use"], True, "ticket single use")
        assert_equal(ticket["consumed"], False, "ticket consumed flag")
        assert_equal(ticket["authorized_slurm_allocation_count"], 1, "allocation count")
        assert_equal(
            ticket["genesis_gpu_ledger_sha256"],
            sha256_file(ledger_path),
            "ticket genesis ledger",
        )
        assert_equal(
            ticket["code_resource_authorization_sha256"],
            cfg["review"]["code_resource_authorization_sha256"],
            "ticket code authorization",
        )
        assert_equal(ticket["config_contract_sha256"], config_contract_sha256(cfg), "ticket config")
        if consumption_path.exists():
            raise RuntimeError("HALT_RESOURCE_CAP: ticket consumption already exists")
        remaining = int(ticket["remaining_seconds"])
        watchdog = int(ticket["watchdog_seconds"])
        reserve = int(cfg["resources"]["watchdog_reserve_seconds"])
        if watchdog != remaining - reserve or remaining > int(ledger["cap_gpu_seconds"]):
            raise RuntimeError("HALT_RESOURCE_CAP: ticket watchdog/reserve mismatch")
        if watchdog <= int(cfg["resources"]["minimum_submit_remaining_seconds"]):
            raise RuntimeError("HALT_RESOURCE_CAP: insufficient watchdog budget")

        claim_record = {
            "schema_version": "c04_allocation_claim_v2",
            "run_id": RUN_ID,
            "slurm_job_id": job_id,
            "allocation_entry_uptime_seconds": start_uptime,
            "watchdog_deadline_uptime_seconds": start_uptime + watchdog,
            "reserved_gpu_seconds": remaining,
            "ticket_sha256": sha256_file(ticket_path),
            "preflight_manifest_sha256": preflight_sha,
            "payload_review_sha256": payload_sha,
            "gpu_execution_authorization_sha256": gpu_auth_sha,
            "config_contract_sha256": config_contract_sha256(cfg),
        }
        claim_record["claim_sha256"] = sha256_obj(claim_record)
        exclusive_publish_json(cfg["paths"]["allocation_claim"], claim_record)
        consumption = {
            "schema_version": "c04_resource_ticket_consumption_v2",
            "run_id": RUN_ID,
            "slurm_job_id": job_id,
            "ticket_sha256": sha256_file(ticket_path),
            "allocation_claim_sha256": sha256_file(
                root_path(cfg["paths"]["allocation_claim"])
            ),
            "consumed_once": True,
        }
        exclusive_publish_json(cfg["paths"]["resource_consumption"], consumption)
        ledger["ledger_revision"] = int(ledger["ledger_revision"]) + 1
        ledger["state"] = "CLAIMED_ACTIVE"
        ledger["jobs"].append({
            **claim_record,
            "status": "CLAIMED_ACTIVE",
            "gpu_count": 1,
            "accounted_gpu_seconds": 0,
            "requires_terminal_reconciliation": True,
        })
        ledger["aggregate_accounted_gpu_seconds"] = remaining
        ledger["requires_terminal_reconciliation"] = True
        ledger_body = dict(ledger)
        ledger_body.pop("payload_sha256", None)
        ledger["payload_sha256"] = sha256_obj(ledger_body)
        atomic_replace_json(ledger_path, ledger)

        marker = load_json(cfg["paths"]["allocation_entry_marker"])
        marker["claim_completed"] = True
        marker["allocation_claim_sha256"] = sha256_file(
            root_path(cfg["paths"]["allocation_claim"])
        )
        atomic_replace_json(root_path(cfg["paths"]["allocation_entry_marker"]), marker)
        current = uptime_seconds()
        active = start_uptime + watchdog - current
        if active <= 0:
            raise RuntimeError("HALT_RESOURCE_CAP: watchdog exhausted during allocation claim")
        return active


def mark_exit(cfg: dict[str, Any], start_uptime: int, exit_code: int) -> None:
    job_id = os.environ.get("SLURM_JOB_ID", "")
    marker_path = root_path(cfg["paths"]["allocation_entry_marker"])
    if not marker_path.exists():
        return
    marker = load_json(cfg["paths"]["allocation_entry_marker"])
    if marker.get("slurm_job_id") != job_id:
        raise RuntimeError("HALT_RESOURCE_CAP: foreign allocation entry marker")
    marker["exit_marker_recorded"] = True
    marker["wrapper_exit_code"] = exit_code
    marker["exit_uptime_seconds"] = uptime_seconds()
    marker["provisional_gpu_seconds"] = max(
        0, marker["exit_uptime_seconds"] - start_uptime
    )
    atomic_replace_json(marker_path, marker)

    ledger_path = root_path(cfg["paths"]["gpu_ledger"])
    lock_path = root_path(cfg["paths"]["gpu_ledger_lock"])
    if not ledger_path.exists():
        return
    with lock_path.open("a+b") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        ledger = load_json(cfg["paths"]["gpu_ledger"])
        validate_ledger(ledger, cfg)
        for job in ledger["jobs"]:
            if str(job["slurm_job_id"]) == job_id:
                elapsed = max(0, uptime_seconds() - start_uptime)
                job["status"] = "EXIT_RECORDED_PENDING_SACCT"
                job["wrapper_exit_code"] = exit_code
                job["provisional_gpu_seconds"] = elapsed
                job["accounted_gpu_seconds"] = elapsed
                job["requires_terminal_reconciliation"] = True
        ledger["ledger_revision"] = int(ledger["ledger_revision"]) + 1
        ledger["state"] = "EXIT_RECORDED_PENDING_SACCT"
        ledger["aggregate_accounted_gpu_seconds"] = sum(
            max(
                int(job.get("accounted_gpu_seconds", 0)),
                int(job.get("reserved_gpu_seconds", 0))
                if job.get("requires_terminal_reconciliation")
                else 0,
            )
            for job in ledger["jobs"]
        )
        if ledger["aggregate_accounted_gpu_seconds"] > ledger["cap_gpu_seconds"]:
            raise RuntimeError("HALT_RESOURCE_CAP: exit ledger aggregate exceeds cap")
        ledger_body = dict(ledger)
        ledger_body.pop("payload_sha256", None)
        ledger["payload_sha256"] = sha256_obj(ledger_body)
        atomic_replace_json(ledger_path, ledger)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("claim", "mark-exit"), required=True)
    parser.add_argument("--allocation-start-uptime-seconds", type=int, required=True)
    parser.add_argument("--exit-code", type=int, default=255)
    args = parser.parse_args()
    cfg = load_json(CONFIG_RELATIVE)
    if args.mode == "claim":
        active = claim(cfg, args.allocation_start_uptime_seconds)
        print(active)
        return 0
    mark_exit(cfg, args.allocation_start_uptime_seconds, args.exit_code)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
