#!/usr/bin/env python
"""M1 NON-LINEAGE smoke (M1_CACHE_CODE_REVIEW.md ruling (3)).

Throwaway pre-authorization validation. Exercises the FROZEN M1 producer path at
tiny N on HateMM (a NON-contract dataset: zero MHC train/val/test contact, zero MHC
isolation counters) to empirically settle the deferred rows before the single submit:
  (a) FIX-2 GPU guard require_slurm_cache() under a real 1-GPU allocation,
  (b) offline Qwen2.5-VL-7B load, (c) load_video_frames decode,
  (d) processor videos=[frames] (PIL list, images=None), (e) generate + parse,
  (f) R=4 greedy determinism (byte-compare the four replica raw outputs),
  (g) strict-JSON parse rate, GPU mem peak, per-video wall time -> full-run extrapolation.

Writes ONLY to slurm/tmp/ (throwaway). Touches NO artifacts/lb_scgp_global/v1/m1/,
writes no sealed cache, produces no ledger, reads no label. Imports the frozen common
module + the frozen producer's build_messages so the exact sealed code path is tested.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from lb_scgp_global_r2_m1_cache_v1_common import (  # noqa: E402
    MAX_ASR_CHARS, MAX_NEW_TOKENS, MAX_TITLE_CHARS, MODEL_ID, NUM_FRAMES, REPLICAS,
    build_user_prompt, parse_certificate, require_slurm_cache,
)
from lb_scgp_global_r2_m1_cache_producer_v1 import build_messages  # noqa: E402

HATEMM_GT = "data/gt/HateMM/train.jsonl"
HATEMM_ASR = "data/ASR/HateMM/train_asrK4_whisper-large-v3.jsonl"
HATEMM_VIDEO = "data/video/HateMM/All"
OUT = "slurm/tmp/m1_smoke_result.json"


def _truncate(text, limit):
    t = (text or "").strip()
    return (t[:limit] + " ...[truncated]") if len(t) > limit else t


def load_min(fs, key_text):
    """Read only id + a text field (never label) from a jsonl."""
    out = {}
    with open(ROOT / fs, encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            out[str(o["id"])] = o.get(key_text)
    return out


def load_asr_chunks(fs):
    out = {}
    with open(ROOT / fs, encoding="utf-8") as h:
        for line in h:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            vid = str(o["id"])
            parts = []
            chunks = o.get("chunks")
            if isinstance(chunks, list):
                for c in chunks:
                    if isinstance(c, (list, tuple)) and len(c) >= 3 and c[2] is not None:
                        parts.append(str(c[2]).strip())
            out[vid] = " ".join(p for p in parts if p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=10)
    args = ap.parse_args()

    result = {"dataset": "HateMM", "non_lineage": True, "model_id": MODEL_ID,
              "num_frames": NUM_FRAMES, "replicas": REPLICAS, "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")}

    # (a) FIX-2 GPU guard under the real allocation
    require_slurm_cache()
    result["gpu_guard"] = {"require_slurm_cache_passed": True,
                           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}

    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    sys.path.insert(0, str(ROOT / "src"))
    from utils.generate_subclip_embedding_HF import load_video_frames  # noqa: E402

    titles = load_min(HATEMM_GT, "text")
    asr = load_asr_chunks(HATEMM_ASR) if (ROOT / HATEMM_ASR).exists() else {}
    ids = sorted(titles.keys())[: args.n]

    torch.manual_seed(0)
    device = torch.device("cuda")
    t_load0 = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    result["model_load_seconds"] = round(time.time() - t_load0, 2)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def one_call(frames, title, transcript):
        messages = build_messages(frames, title, transcript)
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        videos = [frames] if frames else None
        inputs = processor(text=[chat], images=None, videos=videos, return_tensors="pt").to(device)
        out_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False, num_beams=1)
        new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(new_ids, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()

    per_video = []
    parse_ok = 0
    total_records = 0
    deterministic_videos = 0
    decode_ok = 0
    for vid in ids:
        title = _truncate(titles.get(vid, ""), MAX_TITLE_CHARS)
        transcript = _truncate(asr.get(vid, ""), MAX_ASR_CHARS)
        vpath = ROOT / HATEMM_VIDEO / f"{vid}.mp4"
        frames = None
        if vpath.exists():
            decoded, ok = load_video_frames(str(vpath), NUM_FRAMES)
            frames = decoded if ok else None
            if ok:
                decode_ok += 1
        t0 = time.time()
        raws = []
        for _r in range(REPLICAS):
            try:
                raws.append(one_call(frames, title, transcript))
            except Exception as e:  # noqa: BLE001
                raws.append(f"__CALL_ERROR__:{type(e).__name__}")
        dt = time.time() - t0
        # (f) determinism: all four raw outputs byte-identical?
        identical = len(set(raws)) == 1
        if identical:
            deterministic_videos += 1
        # (g) parse rate over the four replicas
        vid_parse_ok = 0
        for raw in raws:
            _obs, flags = parse_certificate(raw)
            total_records += 1
            if not flags:
                vid_parse_ok += 1
                parse_ok += 1
        per_video.append({
            "video_id": vid, "frames_decoded": bool(frames), "seconds_for_R_calls": round(dt, 2),
            "replicas_byte_identical": identical, "distinct_raw_outputs": len(set(raws)),
            "parse_ok_replicas": vid_parse_ok,
            "raw0_sha256_prefix": hashlib.sha256(raws[0].encode("utf-8")).hexdigest()[:16],
        })

    gpu_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    total_call_seconds = sum(v["seconds_for_R_calls"] for v in per_video)
    videos_done = len(per_video)
    avg_per_video = (total_call_seconds / videos_done) if videos_done else 0.0
    # full M1 = (549 + 579) unique packs x R=4 calls; per_video already covers R=4.
    full_video_count = 549 + 579
    extrapolated_full_seconds = avg_per_video * full_video_count

    result.update({
        "n_videos": videos_done,
        "frames_decode_ok": decode_ok,
        "parse_ok_records": parse_ok,
        "total_records": total_records,
        "parse_rate": round(parse_ok / total_records, 4) if total_records else 0.0,
        "deterministic_videos_all_R_identical": deterministic_videos,
        "gpu_peak_bytes": gpu_peak,
        "gpu_peak_gib": round(gpu_peak / (1024 ** 3), 2),
        "avg_seconds_per_video_R4": round(avg_per_video, 2),
        "extrapolated_full_run": {
            "full_video_count_MHC_plus_MHC_zh": full_video_count,
            "note": "per-video time covers all R=4 calls; extrapolation assumes no dedup (U_D=N) and one GPU",
            "estimated_seconds_single_gpu": round(extrapolated_full_seconds, 1),
            "estimated_hours_single_gpu": round(extrapolated_full_seconds / 3600.0, 2),
            "estimated_hours_per_dataset_parallel_2gpu": {
                "MHC": round(avg_per_video * 549 / 3600.0, 2),
                "MHC_zh": round(avg_per_video * 579 / 3600.0, 2),
            },
        },
        "per_video": per_video,
    })

    out_fs = ROOT / OUT
    out_fs.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fs, "w", encoding="utf-8") as h:
        json.dump(result, h, indent=2, ensure_ascii=True)
    # stdout summary for the monitor + record
    print("SMOKE_DONE", json.dumps({
        "gpu_guard_passed": result["gpu_guard"]["require_slurm_cache_passed"],
        "cuda_visible_devices": result["gpu_guard"]["cuda_visible_devices"],
        "n": videos_done, "decode_ok": decode_ok, "parse_rate": result["parse_rate"],
        "deterministic_videos": deterministic_videos,
        "gpu_peak_gib": result["gpu_peak_gib"],
        "avg_sec_per_video": result["avg_seconds_per_video_R4"],
        "extrap_hours_1gpu": result["extrapolated_full_run"]["estimated_hours_single_gpu"],
        "model_load_s": result["model_load_seconds"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
