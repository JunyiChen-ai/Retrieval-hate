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
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from c04_a0t_small_v1_common import (  # noqa: E402
    ADDITIVE_INPUT_DIM,
    ARTIFACT_ROOT,
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
    validate_schema,
    video_path,
)


def assert_equal(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise RuntimeError(f"{label}: {actual!r} != {expected!r}")


def verify_authorization(cfg: dict[str, Any], run_id: str) -> None:
    assert_equal(run_id, RUN_ID, "CLI run id")
    assert_equal(cfg["run"]["run_id"], RUN_ID, "config run id")
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
    if "," in visible:
        raise RuntimeError("HALT_RESOURCE_CAP: more than one visible GPU")


def verify_preflight(cfg: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest_path = root_path(cfg["paths"]["preflight_manifest"])
    manifest = load_json(cfg["paths"]["preflight_manifest"])
    payload = dict(manifest)
    claimed = payload.pop("payload_sha256")
    assert_equal(sha256_obj(payload), claimed, "preflight payload hash")
    review = load_json(cfg["paths"]["payload_review"])
    assert_equal(review["run_id"], RUN_ID, "payload review run id")
    assert_equal(review["verdict"], "GO", "payload review verdict")
    assert_equal(review["preflight_manifest_sha256"], sha256_file(manifest_path), "reviewed preflight hash")
    assert_equal(review["map_hashes"], manifest["map_hashes"], "reviewed map hashes")
    assert_equal(review["prompt_hashes"], prompt_hashes(), "reviewed prompt hashes")
    for relative, expected in manifest["staged_output_hashes"].items():
        assert_equal(sha256_file(root_path(relative)), expected, f"preflight output {relative}")
    return manifest, review


def consume_resource_ticket(cfg: dict[str, Any]) -> tuple[dict[str, Any], float]:
    ticket_path = root_path(cfg["paths"]["resource_ticket"])
    ticket = load_json(cfg["paths"]["resource_ticket"])
    assert_equal(ticket["run_id"], RUN_ID, "resource ticket run id")
    assert_equal(ticket["single_use"], True, "resource ticket single-use")
    assert_equal(ticket["consumed"], False, "resource ticket consumed flag")
    if int(ticket["remaining_seconds"]) <= cfg["resources"]["minimum_submit_remaining_seconds"]:
        raise RuntimeError("HALT_RESOURCE_CAP: insufficient ticket remainder")
    watchdog_env = int(os.environ.get("C04_WATCHDOG_SECONDS", "0"))
    if watchdog_env <= 0 or watchdog_env > int(ticket["watchdog_seconds"]):
        raise RuntimeError("HALT_RESOURCE_CAP: invalid allocation-start watchdog remainder")
    consumption = {
        "schema_version": "c04_resource_ticket_consumption_v1",
        "run_id": RUN_ID,
        "ticket_sha256": sha256_file(ticket_path),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "consumed_once": True,
        "chain_release_resubmit_performed": False,
        "active_watchdog_seconds": watchdog_env,
    }
    consumption_path = root_path(cfg["paths"]["resource_consumption"])
    if consumption_path.exists():
        existing = load_json(cfg["paths"]["resource_consumption"])
        assert_equal(existing, consumption, "same-allocation resume consumption")
    else:
        exclusive_publish_json(cfg["paths"]["resource_consumption"], consumption)
    return ticket, time.monotonic() + watchdog_env


def deadline_check(deadline: float) -> None:
    if time.monotonic() >= deadline:
        raise RuntimeError("HALT_RESOURCE_CAP: producer deadline reached before model forward")


def load_selected_inputs(
    cfg: dict[str, Any],
    preflight: dict[str, Any],
    dataset: str,
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
        "dev_content_read_count": 0,
        "test_content_read_count": 0,
        "ocr_call_count": 0,
        "external_api_call_count": 0,
        "cross_dataset_input_count": 0,
    }
    source_path = root_path(spec["train_asr"])
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
            sha256_bytes(row["transcript"].encode("utf-8")),
            source["transcript_sha256"],
            f"{dataset}/{video_id} transcript hash",
        )
        video_fs = video_path(dataset, video_id)
        assert_equal(sha256_file(video_fs), source["video_sha256"], f"{dataset}/{video_id} video hash")
        output.append({
            "video_id": video_id,
            "language": row["language"],
            "transcript": row["transcript"],
            "source": source,
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


def build_messages(frames: list[Any], transcript: str, prompt_form: str) -> list[dict[str, Any]]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"type": "video", "video": frames},
            {"type": "text", "text": PROMPTS[prompt_form].format(transcript=transcript)},
        ]},
    ]


def load_checkpoint(path: Path, schema_path: str, dataset: str) -> dict[tuple[str, str], dict[str, Any]]:
    records: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.endswith("\n"):
                raise RuntimeError(f"HALT_INVALID_FREEZE: torn checkpoint line {line_number}")
            record = json.loads(line)
            validate_schema(record, schema_path, f"checkpoint {dataset}:{line_number}")
            if record["dataset"] != dataset or record["run_id"] != RUN_ID:
                raise RuntimeError("HALT_INVALID_FREEZE: foreign checkpoint record")
            key = (record["video_id"], record["prompt_form"])
            if key in records:
                raise RuntimeError(f"HALT_INVALID_FREEZE: duplicate checkpoint key {key}")
            records[key] = record
    return records


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(
            record, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        ) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


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
    role_maps: dict[str, dict[str, Any]],
    tokenizer: Any,
    embedding: Any,
    torch: Any,
    le3_matrix: Any,
    additive_matrix: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
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
                text = f"{row['content']}<fallback=unavailable>"
            rendered[slot] = text
        control_renders[control] = rendered

    features: dict[str, Any] = {}
    full_u: dict[str, list[float]] = {}
    for control, rendered in control_renders.items():
        vectors = {}
        for slot in SLOTS:
            mean = teacher_mean_embedding(rendered[slot], tokenizer, embedding, torch)
            vectors[slot] = apply_role(mean, role_maps[slot])
        q4 = q_product([vectors[slot] for slot in SLOTS])
        features[control] = {"q4": f32le_b64(q4)}
        if control == "FULL":
            full_u = vectors
            features[control]["slots"] = {
                slot: f32le_b64(vectors[slot]) for slot in SLOTS
            }

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
    return features, control_renders["FULL"]


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
    records = []
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
        features, renders = compose_features(
            slots, role_maps, tokenizer, embedding, torch, le3, additive
        )
        record = {
            "schema_version": "c04_a0t_small_canonical_record_v1",
            "run_id": RUN_ID,
            "dataset": dataset,
            "video_id": video_id,
            "prompt_record_sha256": {
                "A": sha256_obj(record_a),
                "B": sha256_obj(record_b),
            },
            "slots": slots,
            "full_renders": renders,
            "features": features,
            "frame_decode_failed": record_a["input"]["frame_decode_failed"],
        }
        validate_schema(record, cfg["schemas"]["canonical_record"], f"canonical {dataset}/{video_id}")
        records.append(record)

    state_counts = {
        slot: dict(Counter(record["slots"][slot]["state"] for record in records))
        for slot in SLOTS
    }
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
        "semantic_reliability_passed": rate_pass,
        "kill_state_if_failed": "KILL_C04_TEACHER_SEMANTIC_RELIABILITY",
    }
    return records, reliability


def idempotent_complete(cfg: dict[str, Any]) -> bool:
    manifest_path = root_path(cfg["paths"]["seal_manifest"])
    if not manifest_path.exists():
        return False
    manifest = load_json(cfg["paths"]["seal_manifest"])
    assert_equal(manifest["run_id"], RUN_ID, "existing seal run id")
    for relative, expected in manifest["sealed_output_hashes"].items():
        assert_equal(sha256_file(root_path(relative)), expected, f"existing sealed output {relative}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    cfg = load_json(args.config)
    verify_authorization(cfg, args.run_id)
    if idempotent_complete(cfg):
        return 0
    preflight, payload_review = verify_preflight(cfg)
    ticket, deadline = consume_resource_ticket(cfg)

    inputs: dict[str, list[dict[str, Any]]] = {}
    access_counters: dict[str, dict[str, int]] = {}
    for dataset in DATASETS:
        inputs[dataset], access_counters[dataset] = load_selected_inputs(
            cfg, preflight, dataset
        )

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
            checkpoint, cfg["schemas"]["prompt_record"], dataset
        )
        sequence = 0
        for input_row in inputs[dataset]:
            video_id = input_row["video_id"]
            needed = [form for form in PROMPT_FORMS if (video_id, form) not in records]
            if not needed:
                sequence += 2
                continue
            decoded = decode_exact_frames(video_path(dataset, video_id))
            frame_decode_count += 1
            for prompt_form in PROMPT_FORMS:
                key = (video_id, prompt_form)
                if key in records:
                    sequence += 1
                    continue
                started = time.monotonic()
                raw = one_forward(
                    decoded["frames"], input_row["transcript"], prompt_form
                )
                transport_error = ""
                total_teacher_calls += 1
                parsed = parse_teacher_response(raw, dataset)
                record = {
                    "schema_version": "c04_a0t_small_prompt_record_v1",
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
                    },
                    "provenance": {
                        "preflight_manifest_sha256": sha256_file(
                            root_path(cfg["paths"]["preflight_manifest"])
                        ),
                        "payload_review_sha256": sha256_file(
                            root_path(cfg["paths"]["payload_review"])
                        ),
                        "prompt_sha256": cfg["prompt_hashes"][prompt_form],
                        "model_snapshot_revision": cfg["model"]["snapshot_revision"],
                        "model_tree_sha256": cfg["model"]["model_tree_sha256"],
                        "processor_tree_sha256": cfg["model"]["processor_tree_sha256"],
                        "teacher_label_read": False,
                        "dev_test_read": False,
                        "ocr_used": False,
                        "external_api_used": False,
                        "cross_dataset_input_used": False,
                    },
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
                sequence += 1
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
    elapsed = int(math.ceil(ticket["watchdog_seconds"] - max(0.0, deadline - time.monotonic())))
    access = {
        "schema_version": "c04_access_ledger_v1",
        "run_id": RUN_ID,
        "stage": "teacher_and_prelabel_seal",
        "per_dataset": access_counters,
        "teacher_calls_this_invocation": total_teacher_calls,
        "sealed_prompt_record_count": 800,
        "frame_decode_count_this_invocation": frame_decode_count,
        "zero_counters": {
            "label_value_materialized": sum(
                access_counters[dataset]["label_value_materialized"] for dataset in DATASETS
            ),
            "dev_content_read_count": 0,
            "test_content_read_count": 0,
            "ocr_call_count": 0,
            "external_api_call_count": 0,
            "cross_dataset_input_count": 0,
            "chain_count": 0,
            "release_count": 0,
            "resubmit_count": 0,
        },
    }
    provisional = {
        "schema_version": "c04_provisional_gpu_usage_v1",
        "run_id": RUN_ID,
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "allocated_gpu_count": 1,
        "provisional_elapsed_seconds": elapsed,
        "provisional_gpu_seconds": elapsed,
        "requires_sacct_reconciliation": True,
        "watchdog_event": False,
    }
    sealed_hashes = {
        prompt_rel: sha256_bytes(
            b"".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8") + b"\n"
                for row in prompt_rows
            )
        ),
        canonical_rel: sha256_bytes(
            b"".join(
                json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                           allow_nan=False).encode("utf-8") + b"\n"
                for row in canonical_rows
            )
        ),
    }
    seal = {
        "schema_version": "c04_a0t_small_seal_manifest_v1",
        "run_id": RUN_ID,
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
        "preflight_manifest_sha256": sha256_file(root_path(cfg["paths"]["preflight_manifest"])),
        "payload_review_sha256": sha256_file(root_path(cfg["paths"]["payload_review"])),
        "access_ledger_sha256": sha256_bytes(
            json.dumps(access, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False).encode("utf-8") + b"\n"
        ),
        "provisional_gpu_usage_sha256": sha256_bytes(
            json.dumps(provisional, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
                       allow_nan=False).encode("utf-8") + b"\n"
        ),
        "no_performance_claim": True,
        "no_retry_redraw_prompt_rewrite": True,
    }
    seal["payload_sha256"] = sha256_obj(seal)

    created: list[Path] = []
    try:
        exclusive_publish_jsonl(prompt_rel, prompt_rows)
        created.append(root_path(prompt_rel))
        exclusive_publish_jsonl(canonical_rel, canonical_rows)
        created.append(root_path(canonical_rel))
        exclusive_publish_json(access_rel, access)
        created.append(root_path(access_rel))
        exclusive_publish_json(provisional_rel, provisional)
        created.append(root_path(provisional_rel))
        exclusive_publish_json(cfg["paths"]["seal_manifest"], seal)
        created.append(root_path(cfg["paths"]["seal_manifest"]))
    except Exception:
        for path in reversed(created):
            for target in (path, path.with_name(path.name + ".publish.lock")):
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
