#!/usr/bin/env python
"""Prospective GPU producer and pre-label seal for C04-A0T-SMALL-v1.

The fixed dataset order is HateMM then MHC_zh on one allocation.  Each selected
train ID receives exactly prompt A and prompt B.  There is no retry, redraw,
label read, dev/test read, OCR, network/API call, cross-dataset input, job
submission, dependency, release, or resubmission path.
"""
from __future__ import annotations

import argparse
import io
import json
import math
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

from c04_a0t_small_v1_v2_common import (  # noqa: E402
    ADDITIVE_INPUT_DIM,
    ARTIFACT_ROOT,
    CONFIG_RELATIVE,
    DATASETS,
    EPS,
    LE3_INPUT_DIM,
    MAX_NEW_TOKENS,
    NUM_FRAMES,
    PROMPTS,
    PROMPT_FORMS,
    PROPOSITION_COSINE_MIN,
    Q_DIM,
    ROLE_DIM,
    RUN_ID,
    SCHEMA_VERSION,
    SELECT_N,
    SLOTS,
    SYSTEM_PROMPT,
    build_slot_reliability,
    config_contract_sha256,
    canonical_json_bytes,
    exclusive_publish_bytes,
    exclusive_publish_json,
    exclusive_publish_jsonl,
    f32le_b64,
    load_json,
    merkle_root,
    normalize_proposition,
    parse_teacher_response,
    project_train_asr_line,
    prompt_hashes,
    q_product,
    root_path,
    safe_vector,
    selection_digest,
    sha256_bytes,
    sha256_file,
    sha256_obj,
    train_asr_path,
    validate_schema,
    verify_bound_file_map,
    verify_gpu_execution_authorization,
    verify_historical_code_resource_authorization,
    verify_payload_review,
    verify_preflight_manifest,
    video_path,
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
            "offline_runtime_guards": {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
                "model_loader_local_files_only": True,
            },
            "static_assertions_are_not_runtime_counters": True,
        }


def verify_authorization(cfg: dict[str, Any]) -> None:
    assert_equal(cfg["run"]["run_id"], RUN_ID, "config run id")
    assert_equal(cfg["run"]["implementation_version"], "v2_prospective", "implementation")
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
    assert_equal(cfg["resources"]["small_cap_gpu_seconds"], 7200, "GPU cap")
    verify_bound_file_map(cfg["implementation_hashes"], "producer implementation")
    verify_bound_file_map(cfg["frozen_design_hashes"], "producer frozen design")
    assert_equal(prompt_hashes(), cfg["prompt_hashes"], "prompt hashes")


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
) -> tuple[dict[str, Any], float]:
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
    return claim, time.monotonic() + watchdog_env


def verify_execution_lineage(
    cfg: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str], dict[str, Any], float]:
    verify_authorization(cfg)
    preflight, preflight_sha = verify_preflight_manifest(
        cfg, allow_claimed_gpu_ledger=True
    )
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


def deadline_check(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise RuntimeError("HALT_RESOURCE_CAP: producer deadline reached before model forward")


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


def black_frame() -> Any:
    from PIL import Image

    return Image.new("RGB", (336, 336), color=(0, 0, 0))


def decode_exact_frames(path: Path) -> dict[str, Any]:
    """Decode V2/V3 eight indices; failed requests are black, never substituted."""
    from PIL import Image

    requested: list[int] = []
    frames: list[Any] = []
    failed: list[bool] = []
    backend = "none"
    total = 0
    try:
        import decord
        from decord import VideoReader, cpu

        decord.bridge.set_bridge("native")
        reader = VideoReader(str(path), ctx=cpu(0))
        total = len(reader)
        backend = "decord"
        if total > 0:
            requested = [
                min(total - 1, math.floor((index + 0.5) * total / NUM_FRAMES))
                for index in range(NUM_FRAMES)
            ]
            for frame_index in requested:
                try:
                    frames.append(Image.fromarray(reader[frame_index].asnumpy()).convert("RGB"))
                    failed.append(False)
                except Exception:
                    frames.append(black_frame())
                    failed.append(True)
    except Exception:
        frames = []
        failed = []
        requested = []
        try:
            import av

            container = av.open(str(path))
            decoded = [frame.to_image().convert("RGB") for frame in container.decode(video=0)]
            container.close()
            total = len(decoded)
            backend = "pyav"
            if total > 0:
                requested = [
                    min(total - 1, math.floor((index + 0.5) * total / NUM_FRAMES))
                    for index in range(NUM_FRAMES)
                ]
                frames = [decoded[index] for index in requested]
                failed = [False] * NUM_FRAMES
        except Exception:
            total = 0
            backend = "none"
    if total == 0 or len(frames) != NUM_FRAMES:
        requested = [] if total == 0 else requested
        frames = [black_frame() for _ in range(NUM_FRAMES)]
        failed = [True] * NUM_FRAMES
    return {
        "frames": frames,
        "backend": backend,
        "total_frame_indices": total,
        "requested_indices": requested,
        "frame_decode_failed": failed,
        "any_frame_decode_failed": any(failed),
    }


def _write_fsynced(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def load_or_create_frame_pack(
    cfg: dict[str, Any],
    dataset: str,
    input_row: dict[str, Any],
    lineage: dict[str, str],
    audit: AccessAudit,
) -> dict[str, Any]:
    """Persist and reuse the exact same eight decoded frames across A/B and resume."""
    video_id = input_row["video_id"]
    pack_root = root_path(cfg["datasets"][dataset]["frame_pack_root"])
    final_dir = pack_root / video_id
    manifest_path = final_dir / "manifest.json"
    expected_binding = {
        "run_id": RUN_ID,
        "implementation_version": "v2_prospective",
        "dataset": dataset,
        "video_id": video_id,
        "transcript_sha256": input_row["source"]["transcript_sha256"],
        "video_sha256": input_row["source"]["video_sha256"],
        "preflight_manifest_sha256": lineage["preflight_manifest_sha256"],
        "payload_review_sha256": lineage["payload_review_sha256"],
        "gpu_execution_authorization_sha256": lineage[
            "gpu_execution_authorization_sha256"
        ],
        "config_contract_sha256": lineage["config_contract_sha256"],
        "model_snapshot_revision": cfg["model"]["snapshot_revision"],
        "model_tree_sha256": cfg["model"]["model_tree_sha256"],
        "processor_tree_sha256": cfg["model"]["processor_tree_sha256"],
        "prompt_hashes": cfg["prompt_hashes"],
    }
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        body = dict(manifest)
        claimed = body.pop("payload_sha256", None)
        assert_equal(sha256_obj(body), claimed, "frame-pack manifest payload")
        for key, value in expected_binding.items():
            assert_equal(manifest[key], value, f"frame-pack binding {key}")
        frames = []
        from PIL import Image

        for index, frame_row in enumerate(manifest["frames"]):
            assert_equal(frame_row["index"], index, "frame-pack index")
            path = final_dir / frame_row["filename"]
            assert_equal(path.parent, final_dir, "frame-pack lexical parent")
            assert_equal(sha256_file(path), frame_row["sha256"], "frame-pack frame hash")
            assert_equal(path.stat().st_size, frame_row["size"], "frame-pack frame size")
            with Image.open(path) as image:
                frames.append(image.convert("RGB").copy())
        if len(frames) != NUM_FRAMES:
            raise RuntimeError("HALT_INVALID_FREEZE: frame-pack count")
        audit.event("LOAD_SEALED_FRAME_PACK", dataset, video_id)
        return {
            "frames": frames,
            "backend": manifest["frame_backend"],
            "total_frame_indices": manifest["total_frame_indices"],
            "requested_indices": manifest["requested_indices"],
            "frame_decode_failed": manifest["frame_decode_failed"],
            "manifest_sha256": sha256_file(manifest_path),
            "frame_sha256": [row["sha256"] for row in manifest["frames"]],
            "created": False,
        }
    if final_dir.exists():
        raise RuntimeError("HALT_INVALID_FREEZE: incomplete frame-pack namespace")

    decoded = decode_exact_frames(input_row["resolved_video_path"])
    audit.event("DECODE_TRAIN_VIDEO_TO_FRAME_PACK", dataset, video_id)
    pack_root.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix=video_id + ".tmp.", dir=str(pack_root)))
    try:
        frame_rows = []
        for index, frame in enumerate(decoded["frames"]):
            buffer = io.BytesIO()
            frame.convert("RGB").save(buffer, format="PNG", optimize=False, compress_level=6)
            payload = buffer.getvalue()
            filename = f"{index:02d}.png"
            _write_fsynced(temp_dir / filename, payload)
            frame_rows.append({
                "index": index,
                "filename": filename,
                "size": len(payload),
                "sha256": sha256_bytes(payload),
            })
        manifest = {
            "schema_version": "c04_frame_pack_manifest_v2",
            **expected_binding,
            "frame_backend": decoded["backend"],
            "total_frame_indices": decoded["total_frame_indices"],
            "requested_indices": decoded["requested_indices"],
            "frame_decode_failed": decoded["frame_decode_failed"],
            "frames": frame_rows,
        }
        manifest["payload_sha256"] = sha256_obj(manifest)
        _write_fsynced(
            temp_dir / "manifest.json", canonical_json_bytes(manifest) + b"\n"
        )
        directory_fd = os.open(str(temp_dir), os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        os.rename(temp_dir, final_dir)
    except Exception:
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise
    return {
        **decoded,
        "manifest_sha256": sha256_file(final_dir / "manifest.json"),
        "frame_sha256": [row["sha256"] for row in frame_rows],
        "created": True,
    }


def build_messages(frames: list[Any], transcript: str, prompt_form: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": PROMPTS[prompt_form].format(transcript=transcript)},
        ]},
    ]


def expected_prompt_provenance(
    cfg: dict[str, Any],
    lineage: dict[str, str],
    prompt_form: str,
) -> dict[str, Any]:
    return {
        **lineage,
        "prompt_sha256": cfg["prompt_hashes"][prompt_form],
        "model_snapshot_revision": cfg["model"]["snapshot_revision"],
        "model_tree_sha256": cfg["model"]["model_tree_sha256"],
        "processor_tree_sha256": cfg["model"]["processor_tree_sha256"],
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
    dataset: str,
    input_row: dict[str, Any],
    record_input: dict[str, Any],
    lineage: dict[str, str],
) -> None:
    manifest_path = (
        root_path(cfg["datasets"][dataset]["frame_pack_root"])
        / input_row["video_id"]
        / "manifest.json"
    )
    assert_equal(
        sha256_file(manifest_path),
        record_input["frame_pack_manifest_sha256"],
        "checkpoint frame-pack manifest",
    )
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    body = dict(manifest)
    claimed = body.pop("payload_sha256", None)
    assert_equal(sha256_obj(body), claimed, "checkpoint frame-pack payload")
    expected = {
        "run_id": RUN_ID,
        "implementation_version": "v2_prospective",
        "dataset": dataset,
        "video_id": input_row["video_id"],
        "transcript_sha256": input_row["source"]["transcript_sha256"],
        "video_sha256": input_row["source"]["video_sha256"],
        **lineage,
        "model_snapshot_revision": cfg["model"]["snapshot_revision"],
        "model_tree_sha256": cfg["model"]["model_tree_sha256"],
        "processor_tree_sha256": cfg["model"]["processor_tree_sha256"],
        "prompt_hashes": cfg["prompt_hashes"],
    }
    for key, value in expected.items():
        assert_equal(manifest[key], value, f"checkpoint frame-pack {key}")
    assert_equal(
        [row["sha256"] for row in manifest["frames"]],
        record_input["frame_sha256"],
        "checkpoint frame hashes",
    )
    for frame_row in manifest["frames"]:
        path = manifest_path.parent / frame_row["filename"]
        assert_equal(sha256_file(path), frame_row["sha256"], "checkpoint frame file")


def load_checkpoint(
    path: Path,
    schema_path: str,
    dataset: str,
    input_rows: list[dict[str, Any]],
    cfg: dict[str, Any],
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
                dataset,
                input_by_id[record["video_id"]],
                record["input"],
                lineage,
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


def cosine(left: list[float], right: list[float]) -> float:
    ln = math.sqrt(sum(value * value for value in left))
    rn = math.sqrt(sum(value * value for value in right))
    if ln <= EPS or rn <= EPS:
        return 0.0
    return sum(a * b for a, b in zip(left, right)) / (ln * rn)


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
        "schema_version": "c04_fallback_applicability_v2",
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
            "schema_version": "c04_a0t_small_canonical_record_v2",
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
    assert_equal(manifest["implementation_version"], "v2_prospective", "seal implementation")
    assert_equal(manifest["lineage"], lineage, "existing seal lineage")
    if manifest["terminal_state"] not in {
        "SEALED_PRELABEL_RELIABILITY_PASS",
        "KILL_C04_TEACHER_SEMANTIC_RELIABILITY",
    }:
        raise RuntimeError("HALT_INVALID_FREEZE: nonterminal existing seal")
    assert_equal(manifest["labels_opened"], False, "existing seal labels")
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
            row["dataset"],
            input_by_key[(row["dataset"], row["video_id"])],
            row["input"],
            lineage,
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


def main() -> int:
    cfg = load_json(CONFIG_RELATIVE)
    preflight, lineage, allocation_claim, deadline = verify_execution_lineage(cfg)
    audit = AccessAudit()
    inputs: dict[str, list[dict[str, Any]]] = {}
    access_counters: dict[str, dict[str, int]] = {}
    for dataset in DATASETS:
        inputs[dataset], access_counters[dataset] = load_selected_inputs(
            cfg, preflight, dataset, audit
        )
    if idempotent_complete(cfg, lineage, inputs):
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
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True)

    @torch.no_grad()
    def one_forward(frames: list[Any], transcript: str, prompt_form: str) -> str:
        deadline_check(deadline)
        messages = build_messages(frames, transcript, prompt_form)
        chat = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        tensors = processor(
            text=[chat], images=None, videos=[frames], return_tensors="pt"
        ).to(model.device)
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
    frame_decode_count = 0
    for dataset in DATASETS:
        checkpoint = root_path(cfg["datasets"][dataset]["checkpoint_path"])
        records = load_checkpoint(
            checkpoint,
            cfg["schemas"]["prompt_record"],
            dataset,
            inputs[dataset],
            cfg,
            lineage,
        )
        for rank, input_row in enumerate(inputs[dataset]):
            video_id = input_row["video_id"]
            needed = [form for form in PROMPT_FORMS if (video_id, form) not in records]
            if not needed:
                continue
            decoded = load_or_create_frame_pack(
                cfg, dataset, input_row, lineage, audit
            )
            frame_decode_count += int(decoded["created"])
            for prompt_form in PROMPT_FORMS:
                key = (video_id, prompt_form)
                if key in records:
                    continue
                sequence = rank * 2 + PROMPT_FORMS.index(prompt_form)
                started = time.monotonic()
                audit.event(f"LOCAL_TEACHER_PROMPT_{prompt_form}", dataset, video_id)
                raw = one_forward(
                    decoded["frames"], input_row["transcript"], prompt_form
                )
                transport_error = ""
                total_teacher_calls += 1
                parsed = parse_teacher_response(raw, dataset)
                record = {
                    "schema_version": "c04_a0t_small_prompt_record_v2",
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
        expected_keys = {
            (row["video_id"], form)
            for row in inputs[dataset]
            for form in PROMPT_FORMS
        }
        if set(records) != expected_keys or len(records) != 2 * SELECT_N:
            raise RuntimeError(f"HALT_INVALID_C04_V4: incomplete {dataset} prompt cache")
        all_prompt_rows[dataset] = records

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
        "schema_version": "c04_access_ledger_v2",
        "run_id": RUN_ID,
        "implementation_version": "v2_prospective",
        "stage": "teacher_and_prelabel_seal",
        "lineage": lineage,
        "per_dataset_projected_field_counters": access_counters,
        "teacher_calls_this_invocation": total_teacher_calls,
        "sealed_prompt_record_count": 800,
        "frame_packs_created_this_invocation": frame_decode_count,
        "label_value_materialized_from_projector": sum(
            access_counters[dataset]["label_value_materialized"] for dataset in DATASETS
        ),
        "guarded_runtime_evidence": audit.snapshot(),
        "static_surface_assertions_are_not_runtime_counters": True,
    }
    provisional = {
        "schema_version": "c04_provisional_gpu_usage_v2",
        "run_id": RUN_ID,
        "implementation_version": "v2_prospective",
        "lineage": lineage,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "allocated_gpu_count": 1,
        "provisional_elapsed_seconds": elapsed,
        "provisional_gpu_seconds": elapsed,
        "requires_sacct_reconciliation": True,
        "allocation_claim_sha256": lineage["allocation_claim_sha256"],
        "gpu_ledger_sha256_at_seal": sha256_file(
            root_path(cfg["paths"]["gpu_ledger"])
        ),
    }
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
        "schema_version": "c04_a0t_small_seal_manifest_v2",
        "run_id": RUN_ID,
        "implementation_version": "v2_prospective",
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
    if not idempotent_complete(cfg, lineage, inputs):
        raise RuntimeError("HALT_INVALID_FREEZE: post-publication seal verification failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
