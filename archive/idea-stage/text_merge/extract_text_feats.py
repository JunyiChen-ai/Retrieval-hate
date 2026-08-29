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


class _NoLMHead(torch.nn.Module):
    """Replaces lm_head so the (seq x 152k, fp32-upcast) logits are never materialised.

    The pooled feature is read from the LAST HIDDEN STATE, which lm_head does not touch,
    so this is a memory optimisation with no effect on the vector. Verified numerically
    against the unmodified generate_VideoMLLM_embedding_HF._encode by --verify_lean.
    """

    def forward(self, x):
        return x[..., :1]


def encode_lean(frames, instruction, processor, model, device, span="response"):
    """Bit-equivalent, memory-lean version of GEN._encode.

    Same prompt, same processor call, same span arithmetic, same L2 norm; the only
    differences are (a) output_hidden_states=False plus a forward hook on the text model
    to grab the same final hidden state, and (b) the no-op lm_head.
    """
    messages = GEN._build_messages(frames, instruction)
    text = processor.apply_chat_template(messages, tokenize=False,
                                         add_generation_prompt=True)
    inputs = processor(text=[text], images=None, videos=[frames], return_tensors="pt")
    inputs = inputs.to(device)

    cap = {}

    def hook(_m, _i, out):
        cap["h"] = out.last_hidden_state if hasattr(out, "last_hidden_state") else out[0]

    h = model.model.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(**inputs, output_hidden_states=False, use_cache=False)
    finally:
        h.remove()
    last_hidden = cap["h"][0]
    input_ids = inputs["input_ids"][0]
    assert last_hidden.shape[0] == input_ids.numel(), (
        "hidden/input_ids length mismatch: %d vs %d"
        % (last_hidden.shape[0], input_ids.numel()))

    im_start_id = processor.tokenizer.convert_tokens_to_ids("<|im_start|>")
    positions = (input_ids == im_start_id).nonzero(as_tuple=True)[0]
    if span == "prefix":
        end = int(positions[-1].item()) if len(positions) else last_hidden.shape[0]
        pooled = last_hidden[: max(end, 1)].mean(dim=0)
    else:
        start = int(positions[-1].item()) if len(positions) else max(
            last_hidden.shape[0] - 4, 0)
        start = min(start, last_hidden.shape[0] - 1)
        pooled = last_hidden[start:].mean(dim=0)
    pooled = torch.nn.functional.normalize(pooled.float(), p=2, dim=0)
    return pooled.detach().cpu()


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

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    if a.offload_gib:
        # Fallback when another user holds most of the card: split the frozen encoder
        # between GPU and CPU. Same weights, same dtype, same prompts; every arm of
        # every video is extracted in this one process, so the placement is common-mode
        # across arms and cannot bias any paired comparison.
        print("[load] %s (GPU/CPU split, %d GiB on GPU)" % (MODEL_ID, a.offload_gib),
              flush=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map="auto",
            max_memory={0: "%dGiB" % a.offload_gib, "cpu": "45GiB"})
        model.eval()
    else:
        wait_for_vram(a.need_vram)
        print("[load] %s" % MODEL_ID, flush=True)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
            device_map=None)
        model.to("cuda").eval()
    model.lm_head = _NoLMHead()
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
                v = encode_lean(frames, p, processor, model, "cuda", span="response")
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


def run_verify_lean(a):
    """encode_lean must return the same vector as the production GEN._encode."""
    ids, gt, arm_text, defect = build(ROOT)
    titles = title_map()
    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto", max_memory={0: "%dGiB" % (a.offload_gib or 7), "cpu": "45GiB"})
    model.eval()
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)
    video_root = os.path.join(ROOT, "data", "video", "HateMM", "All")
    # shortest-transcript videos, so the production path (which materialises the
    # fp32 logits) still fits next to the other tenant on the card
    cand = sorted(ids, key=lambda v: len(arm_text["TMall"][v]))[: a.limit or 3]
    for vid in cand:
        frames, ok = GEN.load_video_frames(os.path.join(video_root, vid + ".mp4"),
                                           NUM_FRAMES)
        if not ok:
            continue
        p = text_prompt(arm_text["TMall"][vid], titles[vid])
        ref = GEN._encode(frames, p, processor, model, "cuda", MAX_PIXELS,
                          span="response")
        got = encode_lean(frames, p, processor, model, "cuda", span="response")
        print("[verify] %-22s max|diff| %.3e  cos %.10f  bitwise %s"
              % (vid, (ref - got).abs().max().item(),
                 torch.nn.functional.cosine_similarity(
                     ref.unsqueeze(0), got.unsqueeze(0)).item(),
                 bool(torch.equal(ref, got))), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--verify_lean", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dry_run", action="store_true",
                    help="decode + prompt + tokenize only; no model, no cache written")
    ap.add_argument("--offload_gib", type=int, default=0,
                    help="if >0, load with device_map=auto capped at this many GiB on "
                         "GPU (fallback when the card is shared)")
    ap.add_argument("--need_vram", type=int, default=19000,
                    help="MiB of free VRAM required before the model is loaded")
    a = ap.parse_args()
    if a.verify_lean:
        run_verify_lean(a)
    elif a.assemble:
        run_assemble(a)
    else:
        run_extract(a)


if __name__ == "__main__":
    main()
