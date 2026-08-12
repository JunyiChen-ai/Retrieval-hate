"""TEXT_MERGE step 1 -- re-extract the TEXT stream of the project's video-MLLM encoder
with the arm-specific transcript string.

Encoder, prompt scaffolding, frame sampler, pooling span and cache contract are imported
verbatim from src/utils/generate_VideoMLLM_embedding_HF.py (the script that produced the
banked `{split}_Qwen2.5-VL-7B-Instruct_HF.pt` caches). NOTHING about the encoder is changed:
the only thing that differs between arms is the string placed in the `Transcript: ` slot.

Per video the 8 frames are decoded ONCE and reused for every distinct arm prompt, and
identical prompts across arms are encoded once (deduplicated by text hash).

img_feats are NOT recomputed: they do not depend on the transcript, so they are copied
from the banked cache (identical for every arm, so any hardware drift there is common-mode).

Outputs, resumable:
  idea-stage/text_merge/feats/cache/<vid>.pt   {arm: FloatTensor[3584]}  + meta
Assembly into the loader contract is done by --assemble.
"""
import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "utils"))
from textmerge import ARMS, SPLIT_OUT, build  # noqa: E402

import generate_VideoMLLM_embedding_HF as GEN  # noqa: E402

BANKED_TAG = "Qwen2.5-VL-7B-Instruct_HF"
MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"
NUM_FRAMES = 8
MAX_PIXELS = 360 * 420
CACHE = os.path.join(HERE, "feats", "cache")
OUTDIR = os.path.join(ROOT, "data", "CLIP_Embedding", "HateMM")
OUT_TAG = "TEXTMERGE-%s"


def text_prompt(transcript, title):
    """Byte-identical to generate_VideoMLLM_embedding_HF.process_split's assembly."""
    return (GEN.TEXT_INSTRUCTION
            + "\n" + "Title: " + (title if title else "(none)")
            + "\n" + "Transcript: " + (transcript if transcript else "(none)"))


def free_vram_mib():
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.total,memory.used",
         "--format=csv,noheader,nounits"]).decode().strip().split("\n")[0]
    tot, used = [int(x) for x in out.split(",")]
    return tot - used


def wait_for_vram(need_mib, poll=60):
    while True:
        f = free_vram_mib()
        if f >= need_mib:
            print("[gpu] %d MiB free >= %d MiB required; proceeding" % (f, need_mib),
                  flush=True)
            return
        print("[gpu] waiting: %d MiB free < %d MiB required (%s)"
              % (f, need_mib, time.strftime("%H:%M:%S")), flush=True)
        time.sleep(poll)


# --------------------------------------------------------------------------- extract
def title_map():
    """Titles exactly as generate_VideoMLLM_embedding_HF.read_gt sees them."""
    t = {}
    for split in SPLIT_OUT:
        for it in GEN.read_gt(os.path.join(ROOT, "data", "gt", "HateMM",
                                           "%s.jsonl" % split)):
            t[it["id"]] = it["title"]
    return t


def run_extract(a):
    ids, gt, arm_text, defect = build(ROOT)
    titles = title_map()
    os.makedirs(CACHE, exist_ok=True)

    todo = [v for v in ids if not os.path.exists(os.path.join(CACHE, v + ".pt"))]
    if a.limit:
        todo = todo[: a.limit]
    print("[plan] %d videos total, %d still to encode" % (len(ids), len(todo)), flush=True)
    if not todo:
        print("[plan] nothing to do")
        return

    if a.dry_run:
        # Validates frame decode + prompt assembly + dedup + token counting on CPU,
        # without loading the 7B encoder and without writing any cache file.
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
        video_root = os.path.join(ROOT, "data", "video", "HateMM", "All")
        for vid in todo:
            frames, ok = GEN.load_video_frames(
                os.path.join(video_root, vid + ".mp4"), NUM_FRAMES)
            if not ok:
                print("[dry] %s DECODE_FAIL" % vid)
                continue
            seen, info = set(), []
            for arm in ARMS:
                p = text_prompt(arm_text[arm][vid], titles[vid])
                h = hashlib.sha256(p.encode("utf-8")).hexdigest()[:8]
                nt = n_prompt_tokens(processor, frames, p)
                info.append("%s=%s/%d%s" % (arm, h, nt, "" if h not in seen else "(dup)"))
                seen.add(h)
            print("[dry] %-22s %s" % (vid, "  ".join(info)), flush=True)
        return

    wait_for_vram(a.need_vram)
    print("[load] %s" % MODEL_ID, flush=True)
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa", device_map=None)
    model.to("cuda").eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)

    video_root = os.path.join(ROOT, "data", "video", "HateMM", "All")
    t0 = time.time()
    n_fwd = 0
    for k, vid in enumerate(todo):
        vpath = os.path.join(video_root, vid + ".mp4")
        frames, ok = GEN.load_video_frames(vpath, NUM_FRAMES)
        rec = {"id": vid, "ok": bool(ok), "n_tokens": {}, "prompt_sha": {}}
        vecs = {}
        if not ok:
            for arm in ARMS:
                vecs[arm] = torch.zeros(3584, dtype=torch.float32)
        else:
            title = titles[vid]
            seen = {}
            for arm in ARMS:
                p = text_prompt(arm_text[arm][vid], title)
                h = hashlib.sha256(p.encode("utf-8")).hexdigest()
                rec["prompt_sha"][arm] = h[:16]
                if h in seen:
                    vecs[arm] = seen[h][0]
                    rec["n_tokens"][arm] = seen[h][1]
                    continue
                ntok = n_prompt_tokens(processor, frames, p)
                v = GEN._encode(frames, p, processor, model, "cuda",
                                MAX_PIXELS, span="response")
                n_fwd += 1
                seen[h] = (v, ntok)
                vecs[arm] = v
                rec["n_tokens"][arm] = ntok
        torch.save({"vecs": vecs, "meta": rec}, os.path.join(CACHE, vid + ".pt"))
        if (k + 1) % 10 == 0:
            el = time.time() - t0
            print("[extract] %d/%d done  forwards=%d  %.1fs elapsed  %.2fs/video  eta %.1fmin"
                  % (k + 1, len(todo), n_fwd, el, el / (k + 1),
                     el / (k + 1) * (len(todo) - k - 1) / 60), flush=True)
    print("[extract] DONE %d videos, %d forwards, %.1f min"
          % (len(todo), n_fwd, (time.time() - t0) / 60), flush=True)


def n_prompt_tokens(processor, frames, prompt):
    """Length in tokens of the full multimodal prompt actually fed to the model."""
    messages = GEN._build_messages(frames, prompt)
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=[frames], return_tensors="pt")
    return int(inputs["input_ids"].shape[1])


# -------------------------------------------------------------------------- assemble
def run_assemble(a):
    ids, gt, arm_text, defect = build(ROOT)
    meta = {"per_video": {}, "arms": {}}
    vec = {}
    for vid in ids:
        p = os.path.join(CACHE, vid + ".pt")
        if not os.path.exists(p):
            raise SystemExit("missing cache for %s -- extraction incomplete" % vid)
        d = torch.load(p, map_location="cpu")
        vec[vid] = d["vecs"]
        meta["per_video"][vid] = d["meta"]

    for split, out in SPLIT_OUT.items():
        gtp = os.path.join(ROOT, "data", "gt", "HateMM", "%s.jsonl" % split)
        items = GEN.read_gt(gtp)
        sids = [it["id"] for it in items]
        banked = torch.load(os.path.join(OUTDIR, "%s_%s.pt" % (out, BANKED_TAG)),
                            map_location="cpu")
        bids = [i for s in banked["ids"] for i in s]
        assert bids == sids, "banked id order mismatch for %s" % split
        for arm in ARMS:
            tf = torch.stack([vec[v][arm].float() for v in sids], dim=0)
            obj = {"ids": [sids], "img_feats": banked["img_feats"].float(),
                   "text_feats": tf, "labels": banked["labels"]}
            path = os.path.join(OUTDIR, "%s_%s.pt" % (out, OUT_TAG % arm))
            torch.save(obj, path)
            print("[assemble] %s  N=%d  %s" % (arm, len(sids), path))
        # drift diagnostic: our re-extracted A0 vs the banked cache
        a0 = torch.stack([vec[v]["A0"].float() for v in sids], dim=0)
        cos = torch.nn.functional.cosine_similarity(a0, banked["text_feats"].float(), dim=1)
        meta["arms"].setdefault("A0_vs_banked_cos", {})[out] = {
            "mean": float(cos.mean()), "min": float(cos.min()), "max": float(cos.max())}
        print("[drift] %s A0-vs-banked text cos: mean %.6f min %.6f"
              % (out, cos.mean(), cos.min()))

    # truncation / length accounting
    ctx = 128000
    lens = {arm: [meta["per_video"][v]["n_tokens"].get(arm, 0) for v in ids
                  if meta["per_video"][v]["ok"]] for arm in ARMS}
    for arm in ARMS:
        L = sorted(lens[arm])
        meta["arms"].setdefault("tokens", {})[arm] = {
            "n": len(L), "min": L[0], "median": L[len(L) // 2], "max": L[-1],
            "over_context_%d" % ctx: sum(1 for x in L if x > ctx)}
        print("[tokens] %-7s min %d median %d max %d  over-context %d"
              % (arm, L[0], L[len(L) // 2], L[-1], sum(1 for x in L if x > ctx)))
    meta["n_failed_decode"] = sum(1 for v in ids if not meta["per_video"][v]["ok"])
    print("[decode] videos with no decodable frames (zero-vector guard): %d"
          % meta["n_failed_decode"])
    json.dump(meta, open(os.path.join(HERE, "extract_meta.json"), "w"), indent=1)
    print("[assemble] wrote", os.path.join(HERE, "extract_meta.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true",
                    help="decode + prompt + tokenize only; no model, no cache written")
    ap.add_argument("--need_vram", type=int, default=19000,
                    help="MiB of free VRAM required before the model is loaded")
    a = ap.parse_args()
    if a.assemble:
        run_assemble(a)
    else:
        run_extract(a)


if __name__ == "__main__":
    main()
