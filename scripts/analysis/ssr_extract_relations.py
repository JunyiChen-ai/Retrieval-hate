#!/usr/bin/env python
"""Frozen four-call Qwen2.5-VL relation extraction for SSR B0.

The model receives no labels, predictions, ranks, margins, folds, seeds,
events, or intended loss/family role.  Invalid/missing calls are never repaired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts/analysis"))
from utils.generate_video_archive_HF import load_video_frames  # noqa: E402
from ssr_common import (  # noqa: E402
    PROMPT_P0, PROMPT_P1, RELATION_SCHEMA, SYSTEM_PROMPT,
    atomic_write_json, atomic_write_jsonl,
    build_prompt, calls_to_record, canonical_json, canonicalize_order,
    forbidden_payload_keys, head_tail, id_hash, load_config, read_jsonl,
    resolve, sha256_file, sha256_obj, sha256_text, strict_parse_relation,
)


def load_train_evidence(cfg, dataset):
    gt_path = resolve(cfg, "gt") / dataset / "train.jsonl"
    gt = {str(x["id"]): str(x.get("text") or "") for x in read_jsonl(gt_path)}
    asr_path = resolve(cfg, "asr") / dataset / "train_asrK4_whisper-large-v3.jsonl"
    asr = {}
    if asr_path.exists():
        for row in read_jsonl(asr_path):
            # Deliberately ignore the ASR file's copied parent label field.
            chunks = []
            for chunk in row.get("chunks") or []:
                if len(chunk) >= 3:
                    chunks.append("[{:.3f}-{:.3f}] {}".format(
                        float(chunk[0]), float(chunk[1]), str(chunk[2]).strip()))
            asr[str(row["id"])] = "\n".join(chunks)
    return gt_path, asr_path, gt, asr


def evidence_text(video_id, gt, asr, max_chars):
    source = gt.get(video_id, "")
    timed = asr.get(video_id, "")
    combined = (
        "[SOURCE TRANSCRIPT / OCR-LIKE METADATA; AUTOMATIC OR SOURCE-PROVIDED]\n" +
        (source or "(missing)") +
        "\n[TIMESTAMPED AUTOMATIC ASR]\n" + (timed or "(missing)"))
    return head_tail(combined, max_chars)


def frame_digest(frames):
    return sha256_obj([
        {"shape": list(np.asarray(x).shape),
         "pixels_sha256": hashlib.sha256(np.asarray(x).tobytes()).hexdigest()}
        for x in frames])


def build_serialized_input(dataset, pair_id, order, prompt_version,
                           first_id, second_id, evidence_first, evidence_second,
                           frames_first, frames_second):
    prompt = build_prompt(prompt_version, evidence_first, evidence_second)
    payload = {
        "dataset": dataset,
        "canonical_pair_id": pair_id,
        "prompt_version": prompt_version,
        "order": order,
        "video_a_id": first_id,
        "video_b_id": second_id,
        "evidence_a": evidence_first,
        "evidence_b": evidence_second,
        "frame_count_a": len(frames_first),
        "frame_count_b": len(frames_second),
        "frames_digest_a": frame_digest(frames_first),
        "frames_digest_b": frame_digest(frames_second),
        "user_prompt": prompt,
    }
    hits = forbidden_payload_keys(payload)
    if hits:
        raise RuntimeError("forbidden serialized payload keys: {}".format(hits))
    return payload


@torch.no_grad()
def generate(model, processor, device, payload, frames_a, frames_b, cfg):
    content = [
        {"type": "text", "text": "Video A full-video uniform frames:"},
        {"type": "video", "video": frames_a},
        {"type": "text", "text": "Video B full-video uniform frames:"},
        {"type": "video", "video": frames_b},
        {"type": "text", "text": payload["user_prompt"]},
    ]
    messages = [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
        {"role": "user", "content": content},
    ]
    rendered = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(
        text=[rendered], images=None, videos=[frames_a, frames_b],
        return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=int(cfg["max_new_tokens"]),
        do_sample=bool(cfg["do_sample"]),
        temperature=float(cfg["temperature"]),
        top_p=float(cfg["top_p"]),
    )
    new = out[:, inputs["input_ids"].shape[1]:]
    raw = processor.batch_decode(
        new, skip_special_tokens=True,
        clean_up_tokenization_spaces=False)[0].strip()
    return raw, sha256_text(rendered)


def decode_pair_frames(cfg, dataset, pair):
    root = resolve(cfg, "video") / dataset / "All"
    result = {}
    for key in ("video_a_id", "video_b_id"):
        vid = pair[key]
        path = root / "{}.mp4".format(vid)
        frames, ok = load_video_frames(str(path), int(cfg["mllm"]["num_frames"]))
        result[vid] = {"frames": frames, "ok": bool(ok), "path": path}
    return result


def run_call(model, processor, device, cfg, dataset, pair, order, pv,
             gt, asr, decoded, replay_index):
    low, high = pair["video_a_id"], pair["video_b_id"]
    first, second = (low, high) if order == "AB" else (high, low)
    first_dec, second_dec = decoded[first], decoded[second]
    rec = {
        "dataset": dataset, "canonical_pair_id": pair["canonical_pair_id"],
        "prompt_version": pv, "order": order, "replay_index": replay_index,
        "serialized_input": None, "serialized_input_sha256": None,
        "rendered_chat_sha256": None, "raw_output": None,
        "raw_output_sha256": None, "parsed": None, "parse_error": None,
        "frame_decode_ok": bool(first_dec["ok"] and second_dec["ok"]),
        "wall_s": None,
    }
    t0 = time.time()
    if not rec["frame_decode_ok"]:
        rec["parse_error"] = "frame_decode_failed"
        rec["wall_s"] = round(time.time() - t0, 3)
        return rec
    ev_first = evidence_text(first, gt, asr,
                             int(cfg["mllm"]["max_evidence_chars_per_video"]))
    ev_second = evidence_text(second, gt, asr,
                              int(cfg["mllm"]["max_evidence_chars_per_video"]))
    payload = build_serialized_input(
        dataset, pair["canonical_pair_id"], order, pv, first, second,
        ev_first, ev_second, first_dec["frames"], second_dec["frames"])
    rec["serialized_input"] = payload
    rec["serialized_input_sha256"] = sha256_obj(payload)
    try:
        raw, rendered_hash = generate(
            model, processor, device, payload, first_dec["frames"],
            second_dec["frames"], cfg["mllm"])
        parsed, err = strict_parse_relation(raw)
        rec.update(raw_output=raw, raw_output_sha256=sha256_text(raw),
                   rendered_chat_sha256=rendered_hash,
                   parsed=parsed, parse_error=err)
    except Exception as exc:
        rec["parse_error"] = "generation_failed:{!r}".format(exc)
    rec["wall_s"] = round(time.time() - t0, 3)
    return rec


def expected_payload(cfg, dataset, pair, order, pv, gt, asr, decoded):
    low, high = pair["video_a_id"], pair["video_b_id"]
    first, second = (low, high) if order == "AB" else (high, low)
    if not decoded[first]["ok"] or not decoded[second]["ok"]:
        return None
    ev_first = evidence_text(first, gt, asr,
                             int(cfg["mllm"]["max_evidence_chars_per_video"]))
    ev_second = evidence_text(second, gt, asr,
                              int(cfg["mllm"]["max_evidence_chars_per_video"]))
    return build_serialized_input(
        dataset, pair["canonical_pair_id"], order, pv, first, second,
        ev_first, ev_second, decoded[first]["frames"], decoded[second]["frames"])


def jsonable_component(component):
    if component is None:
        return None
    if hasattr(component, "to_dict"):
        return json.loads(json.dumps(component.to_dict(), default=str,
                                     sort_keys=True))
    return {"class": component.__class__.__name__, "repr": repr(component)}


def canonicalization_unit_check():
    obj = {k: v[0] for k, v in RELATION_SCHEMA.items()}
    obj["stance_a"], obj["stance_b"] = "endorse", "condemn"
    obj["mechanism_a"], obj["mechanism_b"] = "slur", "none"
    out = canonicalize_order(obj, "BA")
    return (out["stance_a"], out["stance_b"], out["mechanism_a"],
            out["mechanism_b"]) == ("condemn", "endorse", "none", "slur")


def smoke_pairs(cfg, dataset, pairs):
    n = int(cfg["mllm"]["smoke_pairs_per_dataset"])
    return sorted(
        pairs,
        key=lambda x: id_hash("smoke-v1", dataset, x["canonical_pair_id"]))[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True, choices=["MHC", "MHC_zh"])
    ap.add_argument("--mode", required=True, choices=["smoke", "full"])
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("SSR computation must run under SLURM")
    if os.environ.get("CONDA_DEFAULT_ENV") != "HateVideo":
        raise RuntimeError("expected conda HateVideo")
    cfg = load_config(args.config)
    pair_root = resolve(cfg, "artifacts") / "pairs" / args.dataset
    pairs = read_jsonl(pair_root / "pairs.jsonl")
    chosen = smoke_pairs(cfg, args.dataset, pairs) if args.mode == "smoke" else pairs
    if not chosen:
        raise RuntimeError("no pairs to extract")
    fold_art = json.load(open(resolve(cfg, "artifacts") / "folds" /
                              "{}.json".format(args.dataset), encoding="utf-8"))
    train_ids = {x["id"] for x in fold_art["records"]}
    if any(p["video_a_id"] not in train_ids or p["video_b_id"] not in train_ids
           for p in chosen):
        raise RuntimeError("non-train pair endpoint")
    gt_path, asr_path, gt, asr = load_train_evidence(cfg, args.dataset)

    from transformers import (AutoProcessor, Qwen2_5_VLForConditionalGeneration,
                              __version__ as transformers_version)
    device = torch.device(args.device)
    model_name = cfg["mllm"]["model"]
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map=None).to(device).eval()
    processor = AutoProcessor.from_pretrained(
        model_name, max_pixels=int(cfg["mllm"]["max_pixels"]))

    model_revision = (getattr(model.config, "_commit_hash", None) or
                      getattr(model.config, "_name_or_path", None))
    model_fingerprint = sha256_obj({
        "model_name": model_name, "revision": model_revision,
        "config": json.loads(json.dumps(model.config.to_dict(), default=str,
                                          sort_keys=True)),
    })
    processor_spec = {
        "class": processor.__class__.__name__,
        "tokenizer": jsonable_component(getattr(processor, "tokenizer", None)),
        "image_processor": jsonable_component(getattr(processor, "image_processor", None)),
        "video_processor": jsonable_component(getattr(processor, "video_processor", None)),
    }
    processor_fingerprint = sha256_obj(processor_spec)
    prompt_schema_decoding = {
        "system": SYSTEM_PROMPT, "P0": PROMPT_P0, "P1": PROMPT_P1,
        "schema": RELATION_SCHEMA,
        "decoding": {k: cfg["mllm"][k] for k in
                     ("do_sample", "temperature", "top_p", "max_new_tokens")},
    }
    prompt_schema_decoding_hash = sha256_obj(prompt_schema_decoding)
    source_hashes = {
        "config": cfg["computed_config_sha256"],
        "gt": sha256_file(gt_path),
        "asr": sha256_file(asr_path) if asr_path.exists() else None,
        "pairs": sha256_file(pair_root / "pairs.jsonl"),
        "input_builder": sha256_file(ROOT / "scripts/analysis/ssr_extract_relations.py"),
        "common": sha256_file(ROOT / "scripts/analysis/ssr_common.py"),
    }
    provenance = {
        "model_fingerprint": model_fingerprint,
        "model_revision": model_revision,
        "processor_fingerprint": processor_fingerprint,
        "prompt_schema_decoding_hash": prompt_schema_decoding_hash,
        "source_hashes": source_hashes,
    }
    provenance_hash = sha256_obj(provenance)

    if args.mode == "smoke":
        out = (resolve(cfg, "artifacts") / "relations" / "smoke" /
               args.run_id / str(os.environ["SLURM_JOB_ID"]) / args.dataset)
        replays = 2
    else:
        out = resolve(cfg, "artifacts") / "relations" / args.dataset
        replays = 1
    out.mkdir(parents=True, exist_ok=True)
    calls = []
    failures = []
    for index, pair in enumerate(chosen, 1):
        decoded = decode_pair_frames(cfg, args.dataset, pair)
        for replay in range(replays):
            for pv in cfg["mllm"]["prompt_versions"]:
                for order in cfg["mllm"]["orders"]:
                    shard = (out / "call_shards" / pair["canonical_pair_id"] /
                             "{}_{}_r{}.json".format(pv, order, replay))
                    expected = expected_payload(
                        cfg, args.dataset, pair, order, pv, gt, asr, decoded)
                    if shard.exists():
                        rec = json.load(open(shard, encoding="utf-8"))
                        if rec.get("provenance_sha256") != provenance_hash:
                            raise RuntimeError("resume provenance mismatch: {}".format(shard))
                        expected_hash = sha256_obj(expected) if expected is not None else None
                        if rec.get("serialized_input_sha256") != expected_hash:
                            raise RuntimeError("resume input hash mismatch: {}".format(shard))
                    else:
                        rec = run_call(
                            model, processor, device, cfg, args.dataset, pair,
                            order, pv, gt, asr, decoded, replay)
                        rec.update({
                            "provenance_sha256": provenance_hash,
                            "model_fingerprint": model_fingerprint,
                            "processor_fingerprint": processor_fingerprint,
                            "prompt_schema_decoding_hash": prompt_schema_decoding_hash,
                        })
                        atomic_write_json(shard, rec)
                    calls.append(rec)
                    if rec.get("parsed") is None:
                        failures.append({
                            "canonical_pair_id": pair["canonical_pair_id"],
                            "prompt_version": pv, "order": order,
                            "replay_index": replay,
                            "reason": rec.get("parse_error"),
                        })
        print(canonical_json({"mode": args.mode, "dataset": args.dataset,
                              "pair": index, "total": len(chosen),
                              "canonical_pair_id": pair["canonical_pair_id"]}),
              flush=True)

    if args.mode == "full":
        # Keep one append-safe canonical call per key, including resumed calls.
        final_by_key = {}
        for c in calls:
            if int(c.get("replay_index", 0)) == 0:
                key = (c["canonical_pair_id"], c["prompt_version"], c["order"])
                final_by_key[key] = c
        final_calls = [final_by_key[k] for k in sorted(final_by_key)]
        atomic_write_jsonl(out / "calls.jsonl", final_calls)
        by_pair = defaultdict(list)
        for c in final_calls:
            by_pair[c["canonical_pair_id"]].append(c)
        records = [calls_to_record(p["canonical_pair_id"], by_pair[p["canonical_pair_id"]])
                   for p in pairs]
        atomic_write_jsonl(out / "records.jsonl", records)
        atomic_write_jsonl(out / "failures.jsonl", failures)
        expected_keys = {(p["canonical_pair_id"], pv, order)
                         for p in pairs for pv in cfg["mllm"]["prompt_versions"]
                         for order in cfg["mllm"]["orders"]}
        observed_keys = {(c["canonical_pair_id"], c["prompt_version"], c["order"])
                         for c in final_calls}
        provenance_ok = all(c.get("provenance_sha256") == provenance_hash
                            for c in final_calls)
        payload_ok = all(not forbidden_payload_keys(c.get("serialized_input"))
                         for c in final_calls if c.get("serialized_input"))
        pass_status = bool(
            len(final_calls) == len(expected_keys) and observed_keys == expected_keys
            and provenance_ok and payload_ok and len(records) == len(pairs)
            and all(r["status"] in ("relation", "missing/no_edge") for r in records))
        replay_report = None
    else:
        atomic_write_jsonl(out / "replay_calls.jsonl", calls)
        replay0 = defaultdict(list)
        for c in calls:
            if int(c["replay_index"]) == 0:
                replay0[c["canonical_pair_id"]].append(c)
        smoke_records = [calls_to_record(
            p["canonical_pair_id"], replay0[p["canonical_pair_id"]]) for p in chosen]
        atomic_write_jsonl(out / "records.jsonl", smoke_records)
        by = defaultdict(dict)
        for c in calls:
            by[(c["canonical_pair_id"], c["prompt_version"], c["order"])][
                int(c["replay_index"])] = c
        comparisons = []
        for key, rr in sorted(by.items()):
            same_input = len(rr) == 2 and rr[0].get("serialized_input_sha256") == rr[1].get("serialized_input_sha256")
            same_output = len(rr) == 2 and rr[0].get("raw_output_sha256") == rr[1].get("raw_output_sha256")
            comparisons.append({"call_key": list(key), "same_input": same_input,
                                "same_output": same_output})
        replay_report = {
            "n_calls_expected": len(chosen) * 4 * 2,
            "n_calls_observed": len(calls),
            "identical_inputs": all(x["same_input"] for x in comparisons),
            "identical_outputs": all(x["same_output"] for x in comparisons),
            "comparisons": comparisons,
        }
        atomic_write_json(out / "replay_report.json", replay_report)
        atomic_write_jsonl(out / "failures.jsonl", failures)
        all_payloads_clean = all(not forbidden_payload_keys(c.get("serialized_input"))
                                 for c in calls if c.get("serialized_input"))
        all_valid_conform = all(c.get("parsed") is None or
                                set(c["parsed"]) == set(RELATION_SCHEMA)
                                for c in calls)
        all_calls_executed = all(c.get("raw_output") is not None for c in calls)
        failures_map_to_missing = all(
            not any(c.get("parsed") is None for c in replay0[p["canonical_pair_id"]])
            or next(r for r in smoke_records
                    if r["canonical_pair_id"] == p["canonical_pair_id"])["status"] == "missing/no_edge"
            for p in chosen)
        pass_status = bool(
            replay_report["n_calls_observed"] == replay_report["n_calls_expected"]
            and replay_report["identical_inputs"]
            and replay_report["identical_outputs"]
            and all_payloads_clean and all_valid_conform
            and all_calls_executed
            and failures_map_to_missing
            and canonicalization_unit_check())

    manifest = {
        "run_id": args.run_id, "mode": args.mode, "dataset": args.dataset,
        "status": "GO" if pass_status else "FAIL",
        "config_sha256": cfg["computed_config_sha256"],
        "n_pairs": len(chosen), "n_call_rows": len(calls),
        "parse_failures_are_missing_no_edge": True,
        "repair_prompt_issued": False,
        "all_endpoints_train_only": True,
        "validation_or_test_endpoint_count": 0,
        "serialized_payload_forbidden_key_count": sum(
            len(forbidden_payload_keys(c.get("serialized_input")))
            for c in calls if c.get("serialized_input")),
        "canonicalization_unit_pass": canonicalization_unit_check(),
        "replay": replay_report,
        "teacher": model_name, "model_revision": model_revision,
        "model_fingerprint": model_fingerprint,
        "transformers_version": transformers_version,
        "processor_class": processor.__class__.__name__,
        "processor_fingerprint": processor_fingerprint,
        "model_class": model.__class__.__name__,
        "decoding": {k: cfg["mllm"][k] for k in
                     ("do_sample", "temperature", "top_p", "max_new_tokens")},
        "source_artifacts": {
            "gt_train": {"path": str(gt_path.relative_to(ROOT)),
                         "sha256": sha256_file(gt_path)},
            "asr_train": {"path": str(asr_path.relative_to(ROOT)),
                          "sha256": sha256_file(asr_path) if asr_path.exists() else None},
            "pairs": {"path": str((pair_root / "pairs.jsonl").relative_to(ROOT)),
                      "sha256": sha256_file(pair_root / "pairs.jsonl")},
            "input_builder": sha256_file(ROOT / "scripts/analysis/ssr_extract_relations.py"),
        },
        "prompt_schema_decoding_hash": prompt_schema_decoding_hash,
        "provenance_sha256": provenance_hash,
        "only_gold_supervision": "video_level_binary_label_not_in_mllm_payload",
        "segment_gold_exists": False, "segment_gold_used": False,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_write_json(out / "manifest.json", manifest)
    print(canonical_json(manifest))
    if not pass_status:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
