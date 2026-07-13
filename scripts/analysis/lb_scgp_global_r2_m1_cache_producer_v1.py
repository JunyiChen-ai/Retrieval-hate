#!/usr/bin/env python
"""M1 Stage 2 - GPU certificate producer (candidate cache; sealed by Stage 3).

For one dataset, builds the deterministic evidence packs (Stage 1), loads a LOCAL
Qwen2.5-VL-7B-Instruct (HF_HUB_OFFLINE=1), and emits R=4 deterministic replicas per
unique evidence pack under the restricted scgp_global_cert_v2 schema.  Parse failures
become canonical all-unresolved records (no prompt/schema rescue); transport/no-
response failures retry up to the plan cap and then fall back to canonical unresolved.
Writes the per-dataset cache.jsonl (scgp_global_cache_replica_v2), a cache_manifest.json
(prompt/input_builder/model_processor hashes, call_count, Merkle root, consensus, zero
counters), and access_ledger.json.  No label is read; no training, no kNN, no accuracy.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from lb_scgp_global_r2_m1_cache_v1_common import (  # noqa: E402
    CACHE_MANIFEST_SCHEMA_VERSION,
    CERT_SCHEMA_VERSION,
    DATASET_RUN,
    MAX_NEW_TOKENS,
    MODEL_ID,
    NUM_FRAMES,
    REPLICA_SCHEMA_ID,
    REPLICAS,
    SYSTEM_PROMPT,
    TrainEvidenceAccessLedger,
    assert_equal,
    build_user_prompt,
    canonical_root_path,
    canonical_unresolved_observables,
    cert_v2_object,
    consensus_for_video,
    exclusive_publish_json,
    exclusive_publish_jsonl,
    input_builder_hash,
    make_replica_record,
    merkle_root,
    model_processor_hash,
    parse_certificate,
    payload_hash,
    prompt_hash,
    read_json,
    record_leaf_hash,
    require_slurm_cache,
    sha256_file,
    sha256_obj,
    validate_against_schema,
    verify_machine_cache,
)
import lb_scgp_global_r2_m1_evidence_pack_v1 as evpack  # noqa: E402

DEP_MODULES = ("torch", "transformers", "numpy", "jsonschema", "decord", "av")


def dependency_check() -> None:
    import importlib
    missing = []
    for name in DEP_MODULES:
        try:
            importlib.import_module(name)
        except Exception:  # noqa: BLE001
            missing.append(name)
    # decord OR av suffices for the frame loader; require at least one.
    if "decord" in missing and "av" in missing:
        raise RuntimeError("neither decord nor av is available for frame decoding")
    hard = [m for m in missing if m not in ("decord", "av")]
    if hard:
        raise RuntimeError(f"missing hard dependencies: {hard}")


def verify_config(cfg: dict[str, Any], dataset: str) -> None:
    run_id = DATASET_RUN[dataset]
    assert_equal(cfg["run"]["run_id"], run_id, "config run id")
    assert_equal(cfg["run"]["dataset"], dataset, "config dataset")
    assert_equal(cfg["run"]["schema_id"], REPLICA_SCHEMA_ID, "config schema id")
    for key in (
        "external_network_or_model_api_allowed",
        "ocr_calls_allowed",
        "label_read_allowed",
        "validation_or_test_allowed",
        "held_content_allowed",
        "teacher_or_cache_read_allowed",
        "query_z_or_labels_allowed",
        "training_allowed",
        "performance_evaluation_allowed",
    ):
        assert_equal(cfg["authorization"][key], False, f"authorization {key}")
    assert_equal(cfg["authorization"]["train_evidence_read_allowed"], True, "train_evidence_read_allowed")
    assert_equal(cfg["authorization"]["local_mllm_inference_allowed"], True, "local_mllm_inference_allowed")
    assert_equal(cfg["authorization"]["gpu_allowed"], True, "gpu_allowed")
    assert_equal(cfg["model"]["model_id"], MODEL_ID, "config model id")


def build_messages(frames: Any, title: str, transcript: str) -> list[dict[str, Any]]:
    user_text = build_user_prompt(title, transcript)
    content: list[dict[str, Any]] = []
    if frames:
        content.append({"type": "video", "video": frames})
    content.append({"type": "text", "text": user_text})
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dataset", required=True, choices=list(DATASET_RUN.keys()))
    args = parser.parse_args()

    require_slurm_cache()
    dependency_check()
    assert_equal(args.run_id, DATASET_RUN[args.dataset], "authorized run id")

    cfg = read_json(args.config)
    verify_config(cfg, args.dataset)
    verify_machine_cache(cfg, args.dataset)

    for rel in (cfg["run"]["artifact_path"], cfg["paths"]["cache_manifest_path"], cfg["paths"]["access_ledger_path"]):
        path, _ = canonical_root_path(rel)
        if path.exists() or path.with_name(path.name + ".publish.lock").exists():
            raise FileExistsError(f"M1 cache no-clobber refusal: {rel}")

    # ---- Stage 1: deterministic evidence packs (label-blind) --------------- #
    ledger = TrainEvidenceAccessLedger(evpack.evidence_allowlist(args.dataset))
    built = evpack.build_dataset_packs(args.dataset, ledger, hash_videos=True)
    order, packs, dedup = built["order"], built["packs"], built["dedup"]

    # ---- Stage 2: local MLLM inference over UNIQUE packs ------------------- #
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

    sys.path.insert(0, str(ROOT / "src"))
    from utils.generate_subclip_embedding_HF import load_video_frames  # noqa: E402

    torch.manual_seed(0)
    device = torch.device("cuda")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    @torch.no_grad()
    def one_call(frames: Any, title: str, transcript: str) -> str:
        messages = build_messages(frames, title, transcript)
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        videos = [frames] if frames else None
        inputs = processor(text=[chat], images=None, videos=videos, return_tensors="pt").to(device)
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
        new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(new_ids, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()

    # api_retry_cap is the HARD cap on TOTAL local model invocations for the dataset
    # (base 4*U_D plus transport retries); call_count counts only the 4*U_D base slots.
    retry_cap = int(cfg["run"]["budget"]["api_retry_cap"])
    call_count = 0
    total_invocations = 0
    parse_ok = 0
    unresolved_records = 0
    transport_fallback = 0
    unique_observables: dict[str, list[list[dict[str, Any]]]] = {}  # pack_sha -> [replica observables...]
    unique_flags: dict[str, list[list[str]]] = {}

    for pack_sha, vids in sorted(dedup.items()):
        rep_vid = sorted(vids)[0]
        pack = packs[rep_vid]
        video_fs, _ = canonical_root_path(pack["video_relpath"])
        frames = None
        if video_fs.exists():
            ledger.note_video_read(pack["video_relpath"], args.dataset)
            decoded, ok = load_video_frames(str(video_fs), NUM_FRAMES)
            frames = decoded if ok else None
        rep_obs: list[dict[str, Any]] = []
        rep_flags: list[list[str]] = []
        for _replica in range(REPLICAS):
            raw = None
            attempt = 0
            while True:
                total_invocations += 1
                try:
                    raw = one_call(frames, pack["title"], pack["asr_transcript"])
                    break
                except Exception:  # noqa: BLE001 - transport/no-response retry only
                    attempt += 1
                    # stop retrying at the total-invocation hard cap or per-slot bound
                    if total_invocations >= retry_cap or attempt > REPLICAS:
                        raw = None
                        break
            call_count += 1
            if raw is None:
                observables = canonical_unresolved_observables()
                flags = ["transport_failure"]
                transport_fallback += 1
                unresolved_records += 1
            else:
                observables, flags = parse_certificate(raw)
                if flags:
                    unresolved_records += 1
                else:
                    parse_ok += 1
            rep_obs.append(observables)
            rep_flags.append(flags)
        unique_observables[pack_sha] = rep_obs
        unique_flags[pack_sha] = rep_flags

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ---- assemble the cache bank (per video x replica) --------------------- #
    records: list[dict[str, Any]] = []
    leaves: list[str] = []
    consensus_rows: list[dict[str, Any]] = []
    for vid in order:
        pack = packs[vid]
        pack_sha = pack["evidence_pack_sha256"]
        rep_obs = unique_observables[pack_sha]
        rep_flags = unique_flags[pack_sha]
        vid_records = []
        for replica_index in range(REPLICAS):
            observables = rep_obs[replica_index]
            flags = rep_flags[replica_index]
            # cross-validate the observables against the Run1-FROZEN cert_v2 contract
            validate_against_schema(cert_v2_object(observables, flags), cfg["paths"]["cert_schema"],
                                    f"cert_v2 {vid}#{replica_index}")
            record = make_replica_record(vid, pack_sha, replica_index, observables, flags)
            validate_against_schema(record, cfg["paths"]["replica_schema"], f"replica {vid}#{replica_index}")
            records.append(record)
            vid_records.append(record)
            leaves.append(record_leaf_hash(record))
        consensus_rows.append({
            "video_id": vid,
            "evidence_pack_sha256": pack_sha,
            "consensus": consensus_for_video(vid_records),
        })

    cache_root = merkle_root(leaves)
    builder_sha = sha256_file(canonical_root_path("scripts/analysis/lb_scgp_global_r2_m1_evidence_pack_v1.py")[0])
    common_sha = sha256_file(canonical_root_path("scripts/analysis/lb_scgp_global_r2_m1_cache_v1_common.py")[0])
    ph = prompt_hash()
    ibh = input_builder_hash(builder_sha, common_sha)
    mph = model_processor_hash()

    zero_counters = dict(ledger.counters)
    expected_calls = REPLICAS * built["unique_pack_count"]
    if call_count != expected_calls:
        raise RuntimeError(f"call_count {call_count} != 4*U_D {expected_calls}")

    manifest = {
        "schema_version": CACHE_MANIFEST_SCHEMA_VERSION,
        "cert_schema_version": CERT_SCHEMA_VERSION,
        "run_id": args.run_id,
        "dataset": args.dataset,
        "terminal_state": "CACHE_PRODUCED_PENDING_SEAL",
        "no_success_claim": True,
        "model_id": MODEL_ID,
        "num_frames": NUM_FRAMES,
        "replicas": REPLICAS,
        "video_count": built["video_count"],
        "unique_pack_count": built["unique_pack_count"],
        "missing_video_count": built["missing_video_count"],
        "call_count": call_count,
        "total_invocations": total_invocations,
        "retry_used": total_invocations - call_count,
        "retry_cap": int(cfg["run"]["budget"]["api_retry_cap"]),
        "record_count": len(records),
        "parse_ok_records": parse_ok,
        "unresolved_records": unresolved_records,
        "transport_fallback_records": transport_fallback,
        "parse_rate": (parse_ok / len(records)) if records else 0.0,
        "train_id_allowlist_sha256": built["train_id_allowlist_sha256"],
        "cache_merkle_root": cache_root,
        "merkle_leaves": len(leaves),
        "prompt_hash": ph,
        "input_builder_hash": ibh,
        "model_processor_hash": mph,
        "consensus": consensus_rows,
        "gold_isolation": {
            "only_gold_supervision": "parent_video_binary_label",
            "train_labels_opened": False,
            "labels_enter_after_cache_seal_only": True,
        },
        "zero_counters": zero_counters,
        "authorized_train_evidence_read_count": ledger.authorized_train_evidence_read_count,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
        "hashes": {
            "config_sha256": sha256_file(canonical_root_path(args.config)[0]),
            "replica_schema_sha256": sha256_file(canonical_root_path(cfg["paths"]["replica_schema"])[0]),
            "evidence_pack_builder_sha256": builder_sha,
            "common_sha256": common_sha,
        },
    }
    manifest["payload_sha256"] = payload_hash(manifest)

    artifact_path, _ = canonical_root_path(cfg["run"]["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    created: list[str] = []
    try:
        exclusive_publish_jsonl(cfg["run"]["artifact_path"], records)
        created.append(cfg["run"]["artifact_path"])
        exclusive_publish_json(cfg["paths"]["cache_manifest_path"], manifest)
        created.append(cfg["paths"]["cache_manifest_path"])
        exclusive_publish_json(cfg["paths"]["access_ledger_path"], ledger.fields(args.dataset))
        created.append(cfg["paths"]["access_ledger_path"])
    except Exception:
        for rel in created:
            path, _ = canonical_root_path(rel)
            for target in (path, path.with_name(path.name + ".publish.lock")):
                try:
                    target.unlink()
                except FileNotFoundError:
                    pass
        raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
