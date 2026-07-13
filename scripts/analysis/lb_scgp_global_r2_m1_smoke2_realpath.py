#!/usr/bin/env python
"""M1 v2 NON-LINEAGE real-path smoke (M1_CACHE_V1_RESULT_TO_CLAIM_REVIEW.md §4.4 / §5.1).

Closes the v1-smoke gap: it calls the ACTUAL FROZEN entities on real symlinked mp4, NOT a
re-implementation. Specifically it runs the frozen `build_dataset_packs` (whose canonical_root_path
line burned v1) on HateMM (a NON-contract dataset whose mp4 are symlinks escaping the repo exactly
like MHC/MHC_zh), then decodes + runs R=4 inference on the first 5 packs via the frozen producer path.

HateMM is registered in the IN-MEMORY EXPECTED_TRAIN_N so the frozen (unmodified) build_dataset_packs
accepts the non-contract dataset; every line of its logic — including the fixed canonical_video_path
guard on all 744 symlinked mp4 — runs as frozen. Output ONLY to slurm/tmp/; touches no MHC data, no
label, no artifacts/lb_scgp_global/v1/m1/.
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

import lb_scgp_global_r2_m1_cache_v2_common as C          # noqa: E402
import lb_scgp_global_r2_m1_evidence_pack_v2 as ev        # noqa: E402
from lb_scgp_global_r2_m1_cache_producer_v2 import build_messages  # noqa: E402

OUT = "slurm/tmp/m1_smoke2_result.json"
HATEMM = "HateMM"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infer_n", type=int, default=5)
    args = ap.parse_args()

    result = {"dataset": HATEMM, "non_lineage": True, "model_id": C.MODEL_ID,
              "num_frames": C.NUM_FRAMES, "replicas": C.REPLICAS,
              "slurm_job_id": os.environ.get("SLURM_JOB_ID", "")}

    # (a) FIX-2 GPU guard under the real allocation (carried from v1 fix)
    C.require_slurm_cache()
    result["gpu_guard"] = {"require_slurm_cache_passed": True,
                           "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", "")}

    # register HateMM train count so the FROZEN build_dataset_packs accepts the non-contract dataset
    hatemm_ids = [json.loads(l)["id"] for l in open(ROOT / "data/gt/HateMM/train.jsonl") if l.strip()]
    C.EXPECTED_TRAIN_N[HATEMM] = len(hatemm_ids)
    result["hatemm_train_count_registered"] = len(hatemm_ids)

    # (b) THE BURN SURFACE: run the FROZEN build_dataset_packs on real symlinked HateMM mp4.
    ledger = ev.TrainEvidenceAccessLedger(ev.evidence_allowlist(HATEMM))
    t0 = time.time()
    built = ev.build_dataset_packs(HATEMM, ledger, hash_videos=True)   # frozen; v1 crashed here
    result["build_dataset_packs"] = {
        "completed_no_raise": True,
        "seconds": round(time.time() - t0, 2),
        "video_count": built["video_count"],
        "unique_pack_count": built["unique_pack_count"],
        "missing_video_count": built["missing_video_count"],
        "train_id_allowlist_sha256": built["train_id_allowlist_sha256"],
    }
    # audit: sample of the symlink followed-targets recorded by the frozen ledger
    vid_reads = [r for r in ledger.records if r.get("kind") == "train_video_read"]
    sym = [r for r in vid_reads if r.get("is_symlink")]
    escaped = [r for r in vid_reads if r.get("followed_target_in_repo") is False]
    result["symlink_audit"] = {
        "train_video_reads": len(vid_reads),
        "is_symlink": len(sym),
        "followed_target_escapes_repo": len(escaped),
        "sample_followed_target": (vid_reads[0]["followed_target"] if vid_reads else ""),
        "forbidden_zero_counters_all_zero": all(v == 0 for v in ledger.counters.values()),
    }

    # (c) decode + R=4 inference on the first infer_n packs via the frozen producer path
    import torch
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    sys.path.insert(0, str(ROOT / "src"))
    from utils.generate_subclip_embedding_HF import load_video_frames  # noqa: E402

    torch.manual_seed(0)
    device = torch.device("cuda")
    tload = time.time()
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        C.MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    model.to(device).eval()
    processor = AutoProcessor.from_pretrained(C.MODEL_ID)
    result["model_load_seconds"] = round(time.time() - tload, 2)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    @torch.no_grad()
    def one_call(frames, title, transcript):
        messages = build_messages(frames, title, transcript)
        chat = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        videos = [frames] if frames else None
        inputs = processor(text=[chat], images=None, videos=videos, return_tensors="pt").to(device)
        out_ids = model.generate(**inputs, max_new_tokens=C.MAX_NEW_TOKENS, do_sample=False, num_beams=1)
        new_ids = out_ids[:, inputs["input_ids"].shape[1]:]
        return processor.batch_decode(new_ids, skip_special_tokens=True,
                                      clean_up_tokenization_spaces=False)[0].strip()

    per_video = []
    parse_ok = 0
    total_records = 0
    deterministic = 0
    decode_ok = 0
    cert_valid = 0
    for vid in built["order"][: args.infer_n]:
        pack = built["packs"][vid]
        # decode via the frozen guard's returned location (follows the symlink at OS level)
        video_fs = C.canonical_video_path(pack["video_relpath"], HATEMM)
        frames = None
        if video_fs.exists():
            decoded, ok = load_video_frames(str(video_fs), C.NUM_FRAMES)
            frames = decoded if ok else None
            if ok:
                decode_ok += 1
        n_frames = len(frames) if frames else 0
        raws = []
        for _r in range(C.REPLICAS):
            try:
                raws.append(one_call(frames, pack["title"], pack["asr_transcript"]))
            except Exception as e:  # noqa: BLE001
                raws.append(f"__CALL_ERROR__:{type(e).__name__}")
        identical = len(set(raws)) == 1
        if identical:
            deterministic += 1
        vok = 0
        for raw in raws:
            obs, flags = C.parse_certificate(raw)
            total_records += 1
            if not flags:
                vok += 1
                parse_ok += 1
            # cross-validate observables against the Run1-frozen cert_v2 schema
            try:
                C.validate_against_schema(C.cert_v2_object(obs, flags),
                                          "schemas/lb_scgp_global_r2/scgp_global_cert_v2.schema.json", "cert")
                cert_valid += 1
            except Exception:  # noqa: BLE001
                pass
        per_video.append({
            "video_id": vid, "n_frames_decoded": n_frames, "replicas_byte_identical": identical,
            "distinct_raw_outputs": len(set(raws)), "parse_ok_replicas": vok,
            "raw0_sha256_prefix": hashlib.sha256(raws[0].encode("utf-8")).hexdigest()[:16],
        })

    gpu_peak = int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0
    result["inference"] = {
        "infer_n": len(per_video),
        "frames_decode_ok": decode_ok,
        "sixteen_frames_each": all(v["n_frames_decoded"] == C.NUM_FRAMES for v in per_video),
        "deterministic_videos_all_R_identical": deterministic,
        "parse_ok_records": parse_ok,
        "total_records": total_records,
        "parse_rate": round(parse_ok / total_records, 4) if total_records else 0.0,
        "cert_v2_schema_valid_records": cert_valid,
        "gpu_peak_gib": round(gpu_peak / (1024 ** 3), 2),
        "per_video": per_video,
    }

    out_fs = ROOT / OUT
    out_fs.parent.mkdir(parents=True, exist_ok=True)
    with open(out_fs, "w", encoding="utf-8") as h:
        json.dump(result, h, indent=2, ensure_ascii=True)
    print("SMOKE2_DONE", json.dumps({
        "gpu_guard_cvd": result["gpu_guard"]["cuda_visible_devices"],
        "build_dataset_packs_ok": result["build_dataset_packs"]["completed_no_raise"],
        "bdp_seconds": result["build_dataset_packs"]["seconds"],
        "video_count": result["build_dataset_packs"]["video_count"],
        "symlinks": result["symlink_audit"]["is_symlink"],
        "escapes": result["symlink_audit"]["followed_target_escapes_repo"],
        "forbidden0": result["symlink_audit"]["forbidden_zero_counters_all_zero"],
        "decode_ok": result["inference"]["frames_decode_ok"],
        "sixteen_each": result["inference"]["sixteen_frames_each"],
        "determ": result["inference"]["deterministic_videos_all_R_identical"],
        "parse_rate": result["inference"]["parse_rate"],
        "cert_valid": result["inference"]["cert_v2_schema_valid_records"],
        "gpu_peak_gib": result["inference"]["gpu_peak_gib"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
