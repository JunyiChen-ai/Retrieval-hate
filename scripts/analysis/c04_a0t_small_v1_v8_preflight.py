#!/usr/bin/env python
"""CPU preflight/freeze for the prospective C04-A0T-SMALL-v1 tranche.

This program never loads a model weight, reads a label value, touches a
development/test path, or submits a job.  It materializes the ID-only 200+200
allowlists, source-content hashes, deterministic role/JL payloads, access
ledger, empty GPU ledger and a one-use resource ticket.

v8 adds two things v7 did on the GPU or not at all.  It DOES now decode video
and write the 400 immutable eight-frame packs -- work that used to consume
budgeted A100 seconds -- and it measures each item's real visual token count by
running the frozen frames through the Qwen2.5-VL image_processor on CPU, with
no model and with CUDA explicitly unavailable.  Those measurements feed a
pre-submit projection gate: either the tranche fits the remaining window by
measurement and the namespace is published, or this program HALTs before the
namespace exists and no GPU submission is authorized.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from c04_a0t_small_v1_v8_common import (  # noqa: E402
    ADDITIVE_INPUT_DIM,
    ADDITIVE_TAG,
    ARTIFACT_ROOT,
    CAMPAIGN_AGGREGATE_CAP_GPU_SECONDS,
    CAMPAIGN_PHASE_CAPS,
    CAMPAIGN_GPU_LEDGER_RELATIVE,
    CONFIG_RELATIVE,
    DATASETS,
    assert_campaign_aggregate_headroom,
    require_exact_keys,
    LE3_INPUT_DIM,
    LE3_TAG,
    PROMPT_HASH_BINDING_PENDING,
    PROMPT_HASH_KEYS,
    NUM_FRAMES,
    ROLE_DIM,
    RUN_ID,
    SELECT_N,
    SMALL_TRANCHE_CAP_GPU_SECONDS,
    TEACHER_MAX_PIXELS,
    VISUAL_PATCH_TOKEN_HARD_CEILING,
    assert_visual_token_ceiling,
    build_prompt_hash_freeze_payload,
    decode_exact_frames,
    frame_pack_binding,
    load_frame_pack_images,
    strict_validate_frame_pack,
    vision_sdpa_fp32_bytes,
    visual_patch_tokens,
    write_frame_pack,
    config_contract_sha256,
    dense_rademacher_payload,
    load_json,
    materialize_role_map,
    merkle_root,
    project_train_asr_line,
    prompt_hashes,
    model_hash_closure,
    resolve_prompt_hashes,
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
    verify_prompt_hash_freeze_payload,
    video_path,
)
from c04_a0t_small_v1_v8_gpu_ledger import (  # noqa: E402
    GPU_LEDGER_KEYS,
    RESOURCE_TICKET_KEYS,
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
            },
            "static_assertions_are_not_runtime_counters": True,
        }


def verify_static_config(cfg: dict[str, Any]) -> str:
    """Verify the static contract and return the config's prompt-hash binding.

    This program is the CPU-preflight freeze stage: it is the one stage that
    computes the prompt hashes, so it is the one stage whose config is allowed
    to still hold the pending sentinel for exactly the four prompt-hash keys.
    Every other value comparison remains exact equality.
    """
    assert_equal(cfg["run"]["run_id"], RUN_ID, "config run id")
    assert_equal(cfg["run"]["implementation_version"], "v8_prospective", "implementation")
    assert_equal(cfg["schema_version"], "c04_a0t_small_v1_v8_config_v1", "config schema")
    assert_equal(tuple(cfg["run"]["datasets"]), DATASETS, "dataset order")
    assert_equal(cfg["selection"]["count_per_dataset"], SELECT_N, "selection count")
    assert_equal(cfg["selection"]["sort"], "ascending_sha256_utf8_concatenation", "selection sort")
    assert_equal(cfg["resources"]["gpu_count"], 1, "GPU count")
    assert_equal(cfg["resources"]["cpus"], 8, "CPU count")
    assert_equal(cfg["resources"]["ram_gb"], 64, "RAM")
    assert_equal(
        cfg["resources"]["small_cap_gpu_seconds"],
        SMALL_TRANCHE_CAP_GPU_SECONDS,
        "small cap",
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
    # Verified, never created, here: the campaign accumulator outlives every
    # implementation namespace, so a stage that could recreate it could also
    # erase the spend it exists to remember.  A tranche whose 2 GPU-hours would
    # not fit under the 8 GPU-hour campaign ceiling is refused before the
    # no-clobber namespace is materialized.
    assert_campaign_aggregate_headroom(
        cfg["resources"]["small_cap_gpu_seconds"], "the eventual small tranche"
    )
    assert_equal(cfg["resources"]["watchdog_reserve_seconds"], 300, "reserve")
    assert_equal(cfg["resources"]["watchdog_term_then_kill_seconds"], 30, "kill after")
    # The teacher-input change is a config term, not an implicit default.
    assert_equal(
        cfg["teacher_contract"]["max_pixels"], TEACHER_MAX_PIXELS, "teacher max_pixels"
    )
    assert_equal(
        cfg["teacher_contract"]["visual_patch_token_hard_ceiling"],
        VISUAL_PATCH_TOKEN_HARD_CEILING,
        "visual patch token ceiling",
    )
    assert_equal(
        cfg["teacher_contract"]["max_pixels_is_a_teacher_input_change"],
        True,
        "max_pixels is declared a teacher-input change",
    )
    assert_equal(cfg["teacher_contract"]["num_frames"], NUM_FRAMES, "frame count")
    assert_equal(cfg["teacher_contract"]["ocr"], False, "OCR")
    # Every measured input the projection gate consumes must be present and
    # positive before anything is materialized; a gate reading a missing or
    # zeroed basis would "fit" vacuously.
    basis = cfg["projection_basis"]
    assert_equal(basis["source_job"], "13852", "projection basis job")
    assert_equal(basis["sacct_elapsed_seconds"], 1978, "projection basis elapsed")
    assert_equal(basis["hatemm_forward_count"], 2 * SELECT_N, "projection basis forwards")
    for key in (
        "hatemm_forward_seconds_mean",
        "hatemm_forward_seconds_sum",
        "non_forward_overhead_seconds",
    ):
        if not isinstance(basis[key], (int, float)) or basis[key] <= 0:
            raise RuntimeError(f"HALT_INVALID_FREEZE: projection basis {key}")
    if (
        abs(
            basis["sacct_elapsed_seconds"]
            - basis["hatemm_forward_seconds_sum"]
            - basis["non_forward_overhead_seconds"]
        )
        > 0.05
    ):
        raise RuntimeError(
            "HALT_INVALID_FREEZE: projection overhead is not the sacct elapsed minus "
            "the summed forward seconds"
        )
    if (
        abs(
            basis["hatemm_forward_seconds_sum"] / basis["hatemm_forward_count"]
            - basis["hatemm_forward_seconds_mean"]
        )
        > 0.001
    ):
        raise RuntimeError("HALT_INVALID_FREEZE: projection mean is not sum/count")
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
        "post_job_reconciliation_authorized",
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
    if cfg["authorization"]["implementation_authorized"] is not True:
        # Every downstream stage requires this flag true, so a false value would
        # let the preflight create the no-clobber namespace and only then wedge
        # the run.  Check it here, before anything is materialized.
        raise RuntimeError("HALT_INVALID_FREEZE: implementation authorization is false")
    verify_bound_file_map(cfg["frozen_design_hashes"], "frozen design")
    verify_bound_file_map(cfg["implementation_hashes"], "implementation")
    _, prompt_hash_binding = resolve_prompt_hashes(cfg, True)
    for dataset in DATASETS:
        source = train_asr_path(cfg, dataset)
        assert_equal(source.stat().st_size, cfg["datasets"][dataset]["train_asr_size"], "ASR size")
        assert_equal(sha256_file(source), cfg["datasets"][dataset]["train_asr_sha256"], "ASR hash")
    return prompt_hash_binding


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
    assert_equal(body["implementation_version"], "v8_prospective", "code/resource implementation")
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
    # Execute the claim rather than assert it: parse the producer's own bytes
    # and prove it carries no decoder and no image-encode call site.  A check
    # that cannot fail under any mutation of the thing it guards is not a check,
    # and "the producer never decodes a video" is now load-bearing for the
    # budget projection, not just for tidiness.
    checks.update(producer_has_no_decode_surface(cfg))
    checks["no_test_paths"] = all(
        "test" not in cfg["datasets"][dataset][field].lower()
        for dataset in DATASETS
        for field in ("train_asr", "video_root", "allowlist_path", "source_manifest_path")
    )
    if not all(checks.values()):
        failed = sorted(key for key, value in checks.items() if not value)
        raise RuntimeError(f"HALT_INVALID_FREEZE: self-test failure {failed}")
    return {
        "schema_version": "c04_a0t_small_v1_impl_v8_self_test_v1",
        "run_id": RUN_ID,
        "checks": checks,
        "all_passed": True,
    }


PRODUCER_FORBIDDEN_MODULES = ("decord", "av", "cv2", "torchvision", "imageio")
PRODUCER_FORBIDDEN_ATTRIBUTES = ("to_image", "asnumpy", "VideoReader")
# The producer imports the common module, and the common module is where the
# decoder lives (lazily, inside the two functions that need it).  Banning the
# decoder modules alone would therefore leave a transitive route open, so the
# frame-WRITING symbols are banned by name as well.
PRODUCER_FORBIDDEN_COMMON_SYMBOLS = (
    "decode_exact_frames",
    "_pyav_decode_exact",
    "write_frame_pack",
    "encode_frame_png",
    "black_frame",
)


def producer_has_no_decode_surface(cfg: dict[str, Any]) -> dict[str, bool]:
    """Static proof that the GPU producer cannot decode a video or encode a PNG.

    The v8 budget projection is only sound if the frame work really left the
    allocation.  This parses the producer's frozen bytes -- the same bytes whose
    SHA-256 is in `implementation_hashes` and is re-verified at every stage --
    and looks for a decoder import, a decoder attribute, or a PIL `save` call.
    """
    import ast

    relative = "scripts/analysis/c04_a0t_small_v1_v8_producer.py"
    if relative not in cfg["implementation_hashes"]:
        raise RuntimeError("HALT_INVALID_FREEZE: producer is not in implementation_hashes")
    tree = ast.parse(root_path(relative).read_text(encoding="utf-8"))
    modules: set[str] = set()
    attributes: set[str] = set()
    imported_symbols: set[str] = set()
    save_calls = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.add(node.module.split(".")[0])
            imported_symbols.update(alias.name for alias in node.names)
        elif isinstance(node, ast.Attribute):
            attributes.add(node.attr)
            if node.attr == "save":
                save_calls += 1
    return {
        "producer_imports_no_video_decoder": not (
            modules & set(PRODUCER_FORBIDDEN_MODULES)
        ),
        "producer_calls_no_decoder_attribute": not (
            attributes & set(PRODUCER_FORBIDDEN_ATTRIBUTES)
        ),
        "producer_has_no_image_save_call": save_calls == 0,
        "producer_imports_no_frame_writing_symbol": not (
            imported_symbols & set(PRODUCER_FORBIDDEN_COMMON_SYMBOLS)
        ),
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


PROJECTION_HALT = "HALT_RESOURCE_PROJECTION"


def build_cpu_image_processor(cfg: dict[str, Any]) -> Any:
    """The deployed processor geometry path, on CPU, with no model and no CUDA.

    This loads only the processor tree whose SHA-256 is already pinned in the
    config and re-verified by `verify_model_snapshot`.  No model weight is read,
    no CUDA context is created, and nothing here can emit a teacher response.
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES", "") not in ("", "-1", "NoDevFiles"):
        raise RuntimeError("HALT_INVALID_FREEZE: the CPU preflight must not see a GPU")
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(
        cfg["model"]["snapshot_path"],
        local_files_only=True,
        max_pixels=TEACHER_MAX_PIXELS,
    )
    image_processor = processor.image_processor
    assert_equal(
        int(image_processor.max_pixels), TEACHER_MAX_PIXELS, "processor max_pixels"
    )
    assert_equal(int(image_processor.patch_size), 14, "processor patch size")
    assert_equal(int(image_processor.temporal_patch_size), 2, "processor temporal patch")
    assert_equal(int(image_processor.merge_size), 2, "processor merge size")
    return image_processor


def measure_visual_geometry(image_processor: Any, frames: list[Any]) -> dict[str, Any]:
    """Run one item's real frames through the real processor and read the grid."""
    prepared = image_processor(images=None, videos=[frames], return_tensors="pt")
    grid = [int(value) for value in prepared["video_grid_thw"][0].tolist()]
    tokens = visual_patch_tokens(grid)
    return {
        "video_grid_thw": grid,
        "patch_tokens": tokens,
        "merged_tokens": tokens // 4,
        "frame_size": [int(frames[0].size[0]), int(frames[0].size[1])],
        "vision_sdpa_fp32_bytes": vision_sdpa_fp32_bytes(tokens),
    }


def project_gpu_window(
    cfg: dict[str, Any],
    geometry: dict[str, Any],
    frame_pack_load_seconds: float,
) -> dict[str, Any]:
    """The pre-submit gate: does the measured tranche fit the remaining window?

    Two states only.  `fits_by_measurement` true means the GPU submission may
    proceed; false means this preflight HALTs before the no-clobber namespace
    exists and the numbers go to the team lead.  There is no third state and no
    partial-tranche option -- a tranche that does not fit is not a tranche that
    should be started.

    The basis is deliberately pessimistic in three separate ways, each recorded:

      1. The per-forward time is v7's measured mean at NATIVE resolution.  Every
         v8 item is capped, and forward time regresses strongly on visual token
         count, so a v8 forward on the same item cannot be slower.
      2. The fixed overhead is v7's whole non-forward remainder, which still
         contains the 201 frame packs v7 built inside its allocation.  v8 builds
         none there.
      3. MHC_zh forwards are priced at the HateMM mean.  MHC_zh has no measured
         forward -- that is this projection's one genuinely unmeasured input,
         and it is stated rather than hidden.
    """
    resources = cfg["resources"]
    basis = cfg["projection_basis"]
    cap = int(resources["small_cap_gpu_seconds"])
    usable_window = (
        cap
        - int(resources["watchdog_reserve_seconds"])
        - int(resources["guard_item_margin_seconds"])
        - int(resources["guard_seal_reserve_seconds"])
    )
    forwards = 2 * SELECT_N * len(DATASETS)
    forward_seconds = forwards * float(basis["hatemm_forward_seconds_mean"])
    fixed_overhead = float(basis["non_forward_overhead_seconds"])
    projected = fixed_overhead + float(frame_pack_load_seconds) + forward_seconds
    max_tokens = max(
        item["patch_tokens"]
        for dataset in DATASETS
        for item in geometry["items"][dataset].values()
    )
    geometry_fits = max_tokens < VISUAL_PATCH_TOKEN_HARD_CEILING
    time_fits = projected <= usable_window
    secondary = float(basis["capped_regime_observation"]["their_mean_seconds"])
    return {
        "schema_version": "c04_gpu_window_projection_v1",
        "usable_teacher_window_seconds": usable_window,
        "window_derivation": (
            f"{cap} small_cap - {resources['watchdog_reserve_seconds']} watchdog_reserve "
            f"- {resources['guard_item_margin_seconds']} guard_item_margin "
            f"- {resources['guard_seal_reserve_seconds']} guard_seal_reserve"
        ),
        "teacher_forward_count": forwards,
        "per_forward_seconds_basis": float(basis["hatemm_forward_seconds_mean"]),
        "per_forward_basis_regime": "V7_NATIVE_RESOLUTION_UPPER_BOUND",
        "projected_forward_seconds": round(forward_seconds, 1),
        "projected_fixed_overhead_seconds": round(fixed_overhead, 1),
        "measured_frame_pack_load_seconds": round(float(frame_pack_load_seconds), 1),
        "projected_gpu_seconds": round(projected, 1),
        "projected_margin_seconds": round(usable_window - projected, 1),
        "projected_margin_fraction": round((usable_window - projected) / usable_window, 4),
        "affordable_mean_per_forward_seconds": round(
            (usable_window - fixed_overhead - float(frame_pack_load_seconds)) / forwards, 4
        ),
        "secondary_capped_regime_projection_seconds": round(
            fixed_overhead + float(frame_pack_load_seconds) + forwards * secondary, 1
        ),
        "secondary_basis_note": (
            "v7's 46 HateMM forwards whose items were already at or below the cap, "
            "where native geometry EQUALS capped geometry, measured "
            f"{secondary}s per forward; reported as corroboration, NOT as the gate"
        ),
        "measured_max_patch_tokens": max_tokens,
        "visual_patch_token_hard_ceiling": VISUAL_PATCH_TOKEN_HARD_CEILING,
        "measured_worst_vision_sdpa_gib": round(
            vision_sdpa_fp32_bytes(max_tokens) / (1024 ** 3), 3
        ),
        "geometry_fits_by_measurement": geometry_fits,
        "time_fits_by_measurement": time_fits,
        "fits_by_measurement": bool(geometry_fits and time_fits),
        "unmeasured_input": (
            "MHC_zh has no measured teacher forward; it is priced at the HateMM mean"
        ),
    }


def materialize_frame_packs(
    cfg: dict[str, Any],
    temp_namespace: Path,
    selected_by_dataset: dict[str, list[dict[str, Any]]],
    source_rows_by_dataset: dict[str, dict[str, dict[str, Any]]],
    code_authorization_sha256: str,
    audit: AccessAudit,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Decode, freeze, re-validate and MEASURE all 400 items, on CPU.

    Everything here used to happen inside the GPU allocation, except the
    measurement, which used to happen nowhere at all -- which is why v7 handed
    the vision tower a 43,056-token item and lost the allocation to a 110.50 GiB
    allocation failure.
    """
    image_processor = build_cpu_image_processor(cfg)
    manifest_hashes: dict[str, str] = {}
    items: dict[str, dict[str, Any]] = {dataset: {} for dataset in DATASETS}
    decode_seconds = 0.0
    write_seconds = 0.0
    load_seconds = 0.0
    measure_seconds = 0.0
    backends: dict[str, int] = {}
    decode_failed_items = 0
    for dataset in DATASETS:
        relative_root = cfg["datasets"][dataset]["frame_pack_root"]
        temp_pack_root = temp_namespace / Path(relative_root).relative_to(ARTIFACT_ROOT)
        for row in selected_by_dataset[dataset]:
            video_id = row["id"]
            source = source_rows_by_dataset[dataset][video_id]
            video_fs = video_path(cfg, dataset, video_id)
            started = time.monotonic()
            decoded = decode_exact_frames(video_fs)
            decode_seconds += time.monotonic() - started
            backends[decoded["backend"]] = backends.get(decoded["backend"], 0) + 1
            decode_failed_items += int(decoded["any_frame_decode_failed"])
            binding = frame_pack_binding(
                cfg,
                dataset,
                video_id,
                source["transcript_sha256"],
                source["video_sha256"],
                code_authorization_sha256,
            )
            started = time.monotonic()
            write_frame_pack(temp_pack_root, video_id, decoded, binding)
            write_seconds += time.monotonic() - started
            # Re-read what was just written, through the SAME validator the GPU
            # producer applies, and measure geometry on the PNG-round-tripped
            # frames rather than on the in-memory decode -- because the PNGs are
            # what the producer will actually hand the processor.
            started = time.monotonic()
            manifest, frame_paths, manifest_sha256 = strict_validate_frame_pack(
                temp_pack_root, video_id, binding
            )
            frames = load_frame_pack_images(frame_paths)
            load_seconds += time.monotonic() - started
            started = time.monotonic()
            geometry = measure_visual_geometry(image_processor, frames)
            measure_seconds += time.monotonic() - started
            assert_visual_token_ceiling(dataset, video_id, geometry["patch_tokens"])
            manifest_hashes[f"{relative_root}/{video_id}/manifest.json"] = manifest_sha256
            items[dataset][video_id] = {
                **geometry,
                "frame_backend": manifest["frame_backend"],
                "frame_decode_failed_count": sum(manifest["frame_decode_failed"]),
            }
            audit.event("FREEZE_AND_MEASURE_FRAME_PACK", dataset, video_id)
    per_dataset = {}
    for dataset in DATASETS:
        tokens = sorted(item["patch_tokens"] for item in items[dataset].values())
        per_dataset[dataset] = {
            "count": len(tokens),
            "patch_tokens_min": tokens[0],
            "patch_tokens_median": tokens[len(tokens) // 2],
            "patch_tokens_max": tokens[-1],
            "patch_tokens_mean": round(sum(tokens) / len(tokens), 1),
        }
    geometry_summary = {
        "schema_version": "c04_visual_geometry_v1",
        "teacher_max_pixels": TEACHER_MAX_PIXELS,
        "measured_through": "Qwen2.5-VL image_processor on CPU, no model, no CUDA",
        "measured_on": "the PNG-round-tripped frames the producer will load",
        "items": items,
        "per_dataset": per_dataset,
        "frame_backends": backends,
        "items_with_any_frame_decode_failure": decode_failed_items,
        "cpu_seconds": {
            "decode": round(decode_seconds, 1),
            "write": round(write_seconds, 1),
            "validate_and_load": round(load_seconds, 1),
            "measure": round(measure_seconds, 1),
        },
    }
    return manifest_hashes, geometry_summary


def preflight(cfg: dict[str, Any], prompt_hash_binding: str) -> None:
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
    selected_by_dataset: dict[str, list[dict[str, Any]]] = {}
    source_rows_by_dataset: dict[str, dict[str, dict[str, Any]]] = {}
    for dataset in DATASETS:
        selected, counters = load_dataset_evidence(cfg, dataset, audit)
        selected_by_dataset[dataset] = selected
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
        source_rows_by_dataset[dataset] = {row["video_id"]: row for row in source_rows}
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
        "schema_version": "c04_gpu_ledger_v8",
        "run_id": RUN_ID,
        "implementation_version": "v8_prospective",
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
    require_exact_keys(gpu_ledger, set(GPU_LEDGER_KEYS), "GPU ledger writer")
    completed_seconds = 0
    remaining = cfg["resources"]["small_cap_gpu_seconds"]
    if remaining <= cfg["resources"]["minimum_submit_remaining_seconds"]:
        raise RuntimeError("HALT_RESOURCE_CAP: insufficient remaining GPU seconds")
    ticket = {
        "schema_version": "c04_resource_ticket_v8",
        "run_id": RUN_ID,
        "implementation_version": "v8_prospective",
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
    require_exact_keys(ticket, set(RESOURCE_TICKET_KEYS), "resource ticket writer")
    staged[ticket_rel] = canonical_bytes(ticket)
    preflight_access = {
        "schema_version": "c04_access_ledger_v8",
        "run_id": RUN_ID,
        "stage": "preflight",
        "projected_field_counters": aggregate_counters,
        "guarded_runtime_evidence": audit.snapshot(),
        "no_teacher_forward_invoked_in_this_program": True,
        "model_weights_loaded_in_this_program": False,
        "cuda_visible_devices_at_freeze": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "frame_packs_frozen_by_this_program": SELECT_N * len(DATASETS),
        "static_surface_assertions_are_not_runtime_counters": True,
    }
    staged[cfg["paths"]["access_ledger"]] = canonical_bytes(preflight_access)

    prompt_hash_freeze = build_prompt_hash_freeze_payload(
        prompt_hash_binding,
        config_contract_sha256(cfg),
        code_authorization_sha256,
    )
    frozen_prompt_hashes = verify_prompt_hash_freeze_payload(prompt_hash_freeze)
    assert_equal(frozen_prompt_hashes, prompt_hashes(), "frozen prompt hashes")
    prompt_hash_rel = cfg["paths"]["prompt_hash_freeze"]
    staged[prompt_hash_rel] = canonical_bytes(prompt_hash_freeze)

    # ------------------------------------------------------------------
    # The temp namespace is created HERE, before the frame packs are written
    # into it and before the projection gate runs.  It is a disposable
    # directory, not the no-clobber namespace: on any failure below --
    # including a failed projection gate -- it is removed and nothing
    # irreversible has happened.  The namespace itself appears only at the
    # single atomic rename at the end.
    # ------------------------------------------------------------------
    namespace.parent.mkdir(parents=True, exist_ok=True)
    temp_namespace = Path(
        tempfile.mkdtemp(prefix=namespace.name + ".tmp.", dir=str(namespace.parent))
    )
    try:
        frame_pack_hashes, visual_geometry = materialize_frame_packs(
            cfg,
            temp_namespace,
            selected_by_dataset,
            source_rows_by_dataset,
            code_authorization_sha256,
            audit,
        )
        expected_packs = SELECT_N * len(DATASETS)
        if len(frame_pack_hashes) != expected_packs:
            raise RuntimeError(
                f"HALT_INVALID_FREEZE: froze {len(frame_pack_hashes)} frame packs, "
                f"expected {expected_packs}"
            )
        projection = project_gpu_window(
            cfg,
            visual_geometry,
            visual_geometry["cpu_seconds"]["validate_and_load"],
        )
        print(json.dumps({"visual_geometry": {
            key: value for key, value in visual_geometry.items() if key != "items"
        }}, sort_keys=True))
        print(json.dumps({"gpu_window_projection": projection}, sort_keys=True))
        if not projection["fits_by_measurement"]:
            raise RuntimeError(
                f"{PROJECTION_HALT}: the measured tranche does not fit the remaining "
                f"window -- projected {projection['projected_gpu_seconds']}s against "
                f"{projection['usable_teacher_window_seconds']}s usable, worst measured "
                f"geometry {projection['measured_max_patch_tokens']} patch tokens "
                f"against a {projection['visual_patch_token_hard_ceiling']} ceiling. "
                "No namespace was created; no GPU submission is authorized."
            )

        preflight_manifest = _build_preflight_manifest(
            cfg,
            staged,
            frame_pack_hashes,
            visual_geometry,
            projection,
            dataset_manifests,
            model_snapshot,
            frozen_prompt_hashes,
            prompt_hash_rel,
            prompt_hash_freeze,
            prompt_hash_binding,
            map_hashes,
            aggregate_counters,
            audit,
            code_authorization_sha256,
            self_test,
        )
        staged[cfg["paths"]["preflight_manifest"]] = canonical_bytes(preflight_manifest)
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


def _build_preflight_manifest(
    cfg: dict[str, Any],
    staged: dict[str, bytes],
    frame_pack_hashes: dict[str, str],
    visual_geometry: dict[str, Any],
    projection: dict[str, Any],
    dataset_manifests: dict[str, Any],
    model_snapshot: dict[str, Any],
    frozen_prompt_hashes: dict[str, str],
    prompt_hash_rel: str,
    prompt_hash_freeze: dict[str, Any],
    prompt_hash_binding: str,
    map_hashes: dict[str, Any],
    aggregate_counters: dict[str, int],
    audit: "AccessAudit",
    code_authorization_sha256: str,
    self_test: dict[str, Any],
) -> dict[str, Any]:
    preflight_manifest = {
        "schema_version": "c04_a0t_small_preflight_manifest_v8",
        "run_id": RUN_ID,
        "implementation_version": "v8_prospective",
        "terminal_state": "PREFLIGHT_HASH_FREEZE_PENDING_PAYLOAD_REVIEW",
        "execution_authorized": False,
        "datasets": dataset_manifests,
        "model_snapshot": model_snapshot,
        "prompt_hashes": frozen_prompt_hashes,
        "prompt_hash_freeze": {
            "path": prompt_hash_rel,
            "sha256": sha256_bytes(staged[prompt_hash_rel]),
            "payload_sha256": prompt_hash_freeze["payload_sha256"],
            "config_binding_at_freeze": prompt_hash_binding,
            "keys": list(PROMPT_HASH_KEYS),
            "literal_hashes_written": True,
            "sentinel_written": False,
            "downstream_must_read_literal_values_here": True,
        },
        "map_hashes": map_hashes,
        "visual_geometry": visual_geometry,
        "gpu_window_projection": projection,
        "projected_field_counters": aggregate_counters,
        "guarded_access_audit": audit.snapshot(),
        "code_resource_authorization_sha256": code_authorization_sha256,
        "config_contract_sha256": config_contract_sha256(cfg),
        "self_test": self_test,
        # The 400 frozen frame-pack manifests join the staged hashes.  Each
        # manifest pins its own eight PNG digests and the validator re-checks
        # every one, so pinning the manifest transitively pins the frames.
        "staged_output_hashes": {
            **{
                relative: sha256_bytes(payload)
                for relative, payload in sorted(staged.items())
            },
            **dict(sorted(frame_pack_hashes.items())),
        },
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    preflight_manifest["payload_sha256"] = sha256_obj(preflight_manifest)
    return preflight_manifest


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
    prompt_hash_binding = verify_static_config(cfg)
    verify_code_resource_authorization(cfg)
    if args.mode == "self-test":
        result = run_self_tests(cfg)
        result["config_prompt_hash_binding"] = prompt_hash_binding
        result["config_prompt_hashes_pending_freeze"] = (
            prompt_hash_binding == PROMPT_HASH_BINDING_PENDING
        )
        print(json.dumps(result, sort_keys=True))
        return 0
    preflight(cfg, prompt_hash_binding)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
