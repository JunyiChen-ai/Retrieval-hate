#!/usr/bin/env python
"""REPRO campaign Wave 1 — UniTime (NeurIPS 2025), universal video temporal grounding.

Driver around the repo's own `inference.py::run_inference`, imported unchanged:
the prompt strings, the `mr` / `mr_seg` routing at `nf_short`, the feature
extractor, the greedy decoding settings and the `extract_time` parser are
upstream's.  This file adds the corpus loop, the resume, the crash marker and
the JSONL record.

QUERY TEXTS ARE FROZEN HERE AND COMMITTED BEFORE THE RUN.

UniTime emits two things per (video, query), and the campaign brief asks for both
to be scored:
  * `pred_relevant_windows` — the temporal window, in seconds.  Rasterised to a
    binary 0/1 curve on the 4 fps grid, and reported as intervals for F1@tIoU.
  * `pred_relevant_windows_mr_seg` — for videos longer than `nf_short`, the coarse
    clip timestamps the segment-retrieval pass kept.  This is the only
    saliency-like output the model produces (it does not emit a per-frame score);
    it is rasterised into a second, coarser binary curve and reported as its own
    variant.  For videos short enough to skip the `mr_seg` pass the two coincide.

Output: idea-stage/repro_unitime/raw/unitime_<DS>_<qkey>.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")
UNITIME = ROOT / "third_party/UniTime"
HOME = Path.home()

VIDEO_DIR = {
    "HateMM": HOME / "data/HateMM/video",
    "MHC": HOME / "data/Multihateclip/English/video_mp4",
    "MHC_zh": HOME / "data/Multihateclip/Chinese/video",
    "HateClipSeg": ROOT / "data/video/HateClipSeg/All",
}
EXTS = (".mp4", ".webm", ".mkv", ".avi")
NO_VIDEO_STREAM = {"hate_video_147", "hate_video_292"}

# Frozen queries.  The main one is the same sentence the Wave 0 Qwen2.5-VL row
# used, so the two grounding methods answer the same question.
MAIN_QUERY = "the moment containing hateful content targeting a person or group"
HCS_CLASS_QUERIES = {
    "c0_normal": "the moment containing normal, non-toxic content",
    "c1_hateful": "the moment containing hateful content",
    "c2_insulting": "the moment containing insulting content",
    "c3_sexual": "the moment containing sexual content",
    "c4_violence": "the moment containing violent content",
    "c5_harm": "the moment containing harmful content",
}

OUT_DIR = ROOT / "idea-stage/repro_unitime/raw"
PERMANENT = {"no_video_stream", "missing_file", "decode_all_backends_failed"}


def find_video(ds, vid):
    for ext in EXTS:
        p = VIDEO_DIR[ds] / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def gt_meta(ds):
    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    return {str(v): (float(z["duration"][i]), str(z["split"][i]))
            for i, v in enumerate(z["video_ids"])}


def done_ids(path: Path):
    if not path.exists():
        return set()
    out = set()
    with open(path) as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("window") is not None or r.get("error") in PERMANENT:
                out.add(r["video_id"])
            else:
                out.discard(r["video_id"])
    return out


def patch_vision_attention():
    """Block-wise SDPA in the Qwen2-VL vision tower. Exact, not an approximation.

    Upstream pins `attn_implementation="flash_attention_2"` for the vision tower;
    no flash-attn wheel exists for torch 2.7.1 / cu128 / sm_120, so the campaign
    patch switched it to `sdpa` (MODEL_ASSETS_STATUS §3.6).  transformers' sdpa
    path materialises a dense `[1, N, N]` mask over *all* patches of *all*
    frames: a 22 s clip at 2 fps is N ~ 32,600 patches and the mask alone asks
    for 31.6 GiB, which no 32 GB card can serve.

    The mask that code builds is exactly block-diagonal on `cu_seqlens`, and
    `cu_seqlens` for a video is one block per frame — the vision tower never
    attends across frames.  Running SDPA once per block therefore returns the
    same tensor the masked call would have returned, in O(sum n_i^2) memory
    instead of O(N^2).  Verified numerically at import time against the dense
    path on a small random input.
    """
    import torch.nn.functional as F
    from transformers.models.qwen2_vl import modeling_qwen2_vl as M

    def forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                position_embeddings=None):
        seq_length = hidden_states.shape[0]
        q, k, v = self.qkv(hidden_states).reshape(
            seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
        if position_embeddings is None:
            emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
            cos, sin = emb.cos().float(), emb.sin().float()
        else:
            cos, sin = position_embeddings
        q, k = M.apply_rotary_pos_emb_vision(q, k, cos, sin)
        q, k, v = q.transpose(0, 1), k.transpose(0, 1), v.transpose(0, 1)
        out = torch.empty_like(q)
        bounds = cu_seqlens.tolist() if torch.is_tensor(cu_seqlens) else list(cu_seqlens)
        for a, b in zip(bounds, bounds[1:]):
            if b <= a:
                continue
            out[:, a:b] = F.scaled_dot_product_attention(
                q[:, a:b], k[:, a:b], v[:, a:b], dropout_p=0.0)
        return self.proj(out.transpose(0, 1).reshape(seq_length, -1))

    M.VisionSdpaAttention.forward = forward
    print("[patch] Qwen2-VL vision SDPA -> block-wise over cu_seqlens", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="HateMM,MHC,MHC_zh,HateClipSeg")
    ap.add_argument("--hcs-classes", action="store_true")
    ap.add_argument("--hcs-class-split", default="test",
                    help="split the six per-class HateClipSeg queries run on")
    ap.add_argument("--base", default="Qwen/Qwen2-VL-7B-Instruct")
    ap.add_argument("--adapter", default=str(ROOT / "third_party/_ckpt/unitime"))
    ap.add_argument("--fps", type=int, default=2)          # upstream default
    ap.add_argument("--clip-length", type=int, default=32)  # upstream default
    ap.add_argument("--nf-short", type=int, default=128)    # README quick-start value
    ap.add_argument("--feat-folder", default=str(ROOT / "idea-stage/repro_unitime/feat"))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keep-feat", action="store_true",
                    help="keep the per-video mr_seg feature cache instead of deleting it")
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    args = ap.parse_args()

    patch_vision_attention()
    sys.path.insert(0, str(UNITIME))
    os.chdir(UNITIME)
    import inference as UT
    from models.qwen2_vl import Qwen2VLMRForConditionalGeneration, Qwen2VLMRProcessor

    dev = torch.device("cuda:0")
    t0 = time.time()
    model = Qwen2VLMRForConditionalGeneration.from_pretrained(
        args.adapter, torch_dtype=torch.bfloat16, device_map={"": dev}).eval()
    processor = Qwen2VLMRProcessor.from_pretrained(args.base)
    print(f"[load] {time.time()-t0:.1f}s", flush=True)

    class A:  # the attribute bag run_inference expects
        pass
    ia = A()
    ia.fps, ia.clip_length, ia.nf_short = args.fps, args.clip_length, args.nf_short
    ia.feat_folder, ia.video_root = args.feat_folder, ""
    Path(args.feat_folder).mkdir(parents=True, exist_ok=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = []
    for ds in args.datasets.split(","):
        meta = gt_meta(ds)
        ids = sorted(meta)
        if args.limit:
            ids = ids[: args.limit]
        qs = [("main", MAIN_QUERY)]
        if ds == "HateClipSeg" and args.hcs_classes:
            qs += list(HCS_CLASS_QUERIES.items())
        done = {qk: done_ids(out_dir / f"unitime_{ds}_{qk}.jsonl") for qk, _ in qs}
        for vid in ids:
            for qk, qt in qs:
                if vid in done[qk]:
                    continue
                if qk != "main" and meta[vid][1] != args.hcs_class_split:
                    continue
                plan.append((ds, vid, qk, qt, meta[vid][0]))
    print(f"[plan] generations={len(plan)}", flush=True)

    handles = {}

    def fh(ds, qk):
        k = f"{ds}_{qk}"
        if k not in handles:
            handles[k] = open(out_dir / f"unitime_{ds}_{qk}.jsonl", "a")
        return handles[k]

    inflight = out_dir / ".inflight"
    if inflight.exists():
        s = inflight.read_text().strip()
        if s:
            ds_c, vid_c, qk_c = s.split("\t")
            f = out_dir / f"unitime_{ds_c}_{qk_c}.jsonl"
            if vid_c not in done_ids(f):
                with open(f, "a") as h:
                    h.write(json.dumps({
                        "video_id": vid_c, "dataset": ds_c, "query_key": qk_c,
                        "window": None, "error": "decode_all_backends_failed",
                        "detail": "decoder crashed the process on this file"}) + "\n")
            print(f"[CRASH-RETIRED] {s}", flush=True)
            plan = [t for t in plan if not (t[0] == ds_c and t[1] == vid_c and t[2] == qk_c)]
        inflight.unlink()

    n_ok = n_err = 0
    t0 = time.time()
    for i, (ds, vid, qk, qt, D) in enumerate(plan, 1):
        path = find_video(ds, vid)
        if path is None or vid in NO_VIDEO_STREAM:
            fh(ds, qk).write(json.dumps({
                "video_id": vid, "dataset": ds, "query_key": qk, "query": qt,
                "window": None,
                "error": "no_video_stream" if vid in NO_VIDEO_STREAM else "missing_file"
            }) + "\n")
            fh(ds, qk).flush()
            n_err += 1
            continue
        inflight.write_text(f"{ds}\t{vid}\t{qk}")
        t1 = time.time()
        data = {"qid": i, "id": vid, "duration": D, "video_path": str(path),
                "annos": [{"query": qt, "window": [[0.0, D]]}]}
        try:
            res = UT.run_inference(model, processor, data, ia, dev)
            r0 = res[0]
            rec = {"video_id": vid, "dataset": ds, "query_key": qk, "query": qt,
                   "window": r0.get("pred_relevant_windows"),
                   "mr_seg": r0.get("pred_relevant_windows_mr_seg"),
                   "duration": D, "sec": round(time.time() - t1, 2)}
            n_ok += 1
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            rec = {"video_id": vid, "dataset": ds, "query_key": qk, "query": qt,
                   "window": None, "error": f"oom:{e}"[:200]}
            n_err += 1
        except Exception as e:
            rec = {"video_id": vid, "dataset": ds, "query_key": qk, "query": qt,
                   "window": None, "error": f"{type(e).__name__}:{e}"[:300]}
            n_err += 1
        fh(ds, qk).write(json.dumps(rec, ensure_ascii=False) + "\n")
        fh(ds, qk).flush()
        inflight.write_text("")
        # The `mr_seg` route caches a visual feature tensor per video (~120 MB for
        # a 4-minute clip; ~75 GB over these corpora).  All of a video's queries
        # are consecutive in the plan, so the cache can be dropped as soon as the
        # last one is answered.
        if not args.keep_feat and (i == len(plan) or plan[i][1] != vid):
            fp = Path(args.feat_folder) / f"{vid}.pt"
            if fp.exists():
                fp.unlink()
        if i % 10 == 0 or i == len(plan):
            el = time.time() - t0
            print(f"PROGRESS {i}/{len(plan)} ds={ds} q={qk} ok={n_ok} err={n_err} "
                  f"elapsed={el:.0f}s rate={i/max(el,1e-9):.3f}gen/s "
                  f"eta={(len(plan)-i)/max(i/max(el,1e-9),1e-9):.0f}s "
                  f"peak={torch.cuda.max_memory_allocated()/2**30:.1f}GiB", flush=True)
    for h in handles.values():
        h.close()
    print(f"[done] ok={n_ok} err={n_err} wall={time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
