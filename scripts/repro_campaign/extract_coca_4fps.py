#!/usr/bin/env python
"""REPRO campaign Wave 2 — CoCa ViT-L/14 features + captions for T3AL.

T3AL (Liberatori et al., CVPR 2024) consumes two pre-extracted artefacts per video:

  1. per-frame CoCa visual features taken **before** the visual projection, because
     `T3ALNet.forward` computes `image_features_pre @ self.model.visual.proj` itself;
  2. a caption file `./captions/<video>.txt`, used by the caption-refinement step,
     whose lines the upstream code parses as `int(line.split("-")[0].split(".")[0]) * 3`
     -> the feature index that caption belongs to.

The released features are THUMOS14 / ActivityNet only, so we extract our own with the
same backbone and the same preprocessing the repo uses
(`third_party/T3AL/src/data/components/utils.py::transform` = Resize((224, 224)) then
CenterCrop(224), i.e. an aspect-squashing resize, then CLIP mean/std).

Rate: the released features are one vector per *native* video frame (T3AL's evaluator
converts a predicted feature index to seconds by dividing by the video's own fps).  We
extract on the campaign's canonical 4 fps grid instead (freeze §1) — recorded as a
deviation in the T3AL section.  Captions are generated on a coarser grid (default 1 fps),
as upstream did (10 fps for THUMOS at 30 fps native, 1 fps for ActivityNet).

Outputs
  data/CLIP_Embedding/<DS>/coca_vitL14_4fps/<vid>.npy       float32 (T, 1024)
  idea-stage/repro_t3al/captions/<DS>/<vid>.txt             "<k>.jpg-<caption>"

Usage
  python scripts/repro_campaign/extract_coca_4fps.py --datasets HateClipSeg --limit 1
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")
HOME = Path.home()

VIDEO_DIR = {
    "HateMM": HOME / "data/HateMM/video",
    "MHC": HOME / "data/Multihateclip/English/video_mp4",
    "MHC_zh": HOME / "data/Multihateclip/Chinese/video",
    "HateClipSeg": ROOT / "data/video/HateClipSeg/All",
}
EXTS = (".mp4", ".webm", ".mkv", ".avi")
NO_VIDEO_STREAM = {"hate_video_147", "hate_video_292"}  # freeze D2

FPS = 4.0
SIZE = 224
MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

MODEL_NAME = "coca_ViT-L-14"
PRETRAINED = "mscoco_finetuned_laion2B-s13B-b90k"

FEAT_DIR = lambda ds: ROOT / f"data/CLIP_Embedding/{ds}/coca_vitL14_4fps"
CAP_DIR = lambda ds: ROOT / f"idea-stage/repro_t3al/captions/{ds}"


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


def frame_stream(path: Path, chunk: int = 64):
    """Yield uint8 (n, SIZE, SIZE, 3) RGB frames at 4 fps.

    `scale=224:224` with no `force_original_aspect_ratio` squashes the frame onto a
    square, which is what torchvision `Resize((224, 224))` does; the subsequent
    `CenterCrop(224)` in the upstream transform is then a no-op.
    """
    vf = f"fps={FPS:g},scale=w={SIZE}:h={SIZE}:flags=bicubic"
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-i", str(path), "-map", "0:v:0",
           "-vf", vf, "-pix_fmt", "rgb24", "-f", "rawvideo", "-"]
    nbytes = SIZE * SIZE * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=nbytes * chunk)
    try:
        while True:
            buf = proc.stdout.read(nbytes * chunk)
            if not buf:
                break
            n = len(buf) // nbytes
            if n == 0:
                break
            yield np.frombuffer(buf[: n * nbytes], dtype=np.uint8).reshape(n, SIZE, SIZE, 3)
    finally:
        proc.stdout.close()
        err = proc.stderr.read().decode("utf-8", "ignore")
        proc.wait()
        if proc.returncode not in (0, None) and err.strip():
            print(f"[ffmpeg] rc={proc.returncode} {path.name}: {err.strip()[:200]}", flush=True)


def patch_isin_device():
    """open_clip 2.24 `CoCa.generate` x transformers 4.49 device mismatch.

    `MinLengthLogitsProcessor` keeps its `eos_token_id` on the CPU while the
    vocabulary tensor it tests against is built on the scores' device, so
    `torch.isin` raises on any CUDA generation.  Moving the second operand onto
    the first's device is the whole fix; it changes no value.
    """
    import torch as _t
    import transformers.generation.logits_process as LP
    import transformers.pytorch_utils as PU
    orig = PU.isin_mps_friendly

    def fixed(elements, test_elements):
        if _t.is_tensor(test_elements):
            test_elements = test_elements.to(elements.device)
        return orig(elements, test_elements)

    PU.isin_mps_friendly = fixed
    LP.isin_mps_friendly = fixed

    # Second incompatibility, same pair of versions: transformers >= 4.42 makes
    # `StoppingCriteriaList.__call__` return a per-sequence bool *tensor*, while
    # open_clip 2.24 uses it as a scalar (`if stopping_criteria(...)`).  Restoring
    # the older "all sequences are done" scalar is what open_clip was written
    # against and is what the reduction below computes.
    from transformers.generation import stopping_criteria as SC
    orig_call = SC.StoppingCriteriaList.__call__

    def call(self, input_ids, scores, **kw):
        r = orig_call(self, input_ids, scores, **kw)
        return bool(r.all()) if _t.is_tensor(r) else bool(r)

    SC.StoppingCriteriaList.__call__ = call


def build_model(device, dtype):
    import open_clip
    patch_isin_device()
    model, _, _ = open_clip.create_model_and_transforms(
        model_name=MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(device=device, dtype=dtype).eval()
    return model


def visual_no_proj(model, x):
    """`model.visual` forward with the final projection skipped.

    Upstream's stored features are exactly this: `T3ALNet.forward` applies
    `@ self.model.visual.proj` to them.  Setting `visual.proj = None` for the call
    short-circuits the one `if self.proj is not None` line inside
    `open_clip.transformer.VisionTransformer.forward`, so nothing else changes.
    """
    proj = model.visual.proj
    model.visual.proj = None
    try:
        out = model.visual(x)
    finally:
        model.visual.proj = proj
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", default="HateMM,MHC,MHC_zh,HateClipSeg")
    ap.add_argument("--splits", default="val,test")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--caption-fps", type=float, default=1.0)
    ap.add_argument("--no-captions", action="store_true")
    ap.add_argument("--caption-beams", type=int, default=6)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshard", type=int, default=1)
    ap.add_argument("--mem-frac", type=float, default=0.0)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if args.mem_frac > 0 and dev == "cuda":
        torch.cuda.set_per_process_memory_fraction(args.mem_frac)
    dtype = torch.float16 if args.fp16 else torch.float32
    model = build_model(dev, dtype)
    mean = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(STD, device=dev).view(1, 3, 1, 1)
    step = max(1, int(round(FPS / args.caption_fps))) if args.caption_fps > 0 else 0

    splits = set(args.splits.split(","))
    t_all = time.time()
    for ds in args.datasets.split(","):
        meta = gt_meta(ds)
        ids = sorted(v for v, (_, sp) in meta.items()
                     if sp in splits and v not in NO_VIDEO_STREAM)
        ids = [v for i, v in enumerate(ids) if i % args.nshard == args.shard]
        if args.limit:
            ids = ids[: args.limit]
        fdir, cdir = FEAT_DIR(ds), CAP_DIR(ds)
        fdir.mkdir(parents=True, exist_ok=True)
        cdir.mkdir(parents=True, exist_ok=True)
        print(f"[plan] {ds} n={len(ids)}", flush=True)
        t0, nf = time.time(), 0
        for n, vid in enumerate(ids):
            fout = fdir / f"{vid}.npy"
            cout = cdir / f"{vid}.txt"
            need_f = not fout.exists()
            need_c = (not args.no_captions) and (not cout.exists())
            if not need_f and not need_c:
                continue
            path = find_video(ds, vid)
            if path is None:
                print(f"[MISS] {ds} {vid}", flush=True)
                continue
            feats, caps, idx = [], [], 0
            try:
                for arr in frame_stream(path, args.batch):
                    x = torch.from_numpy(arr.copy()).to(dev)
                    x = x.permute(0, 3, 1, 2).to(dtype).div_(255.0)
                    x = (x - mean.to(dtype)) / std.to(dtype)
                    with torch.no_grad():
                        if need_f:
                            feats.append(visual_no_proj(model, x).float().cpu().numpy())
                        if need_c and step:
                            sel = [i for i in range(x.shape[0]) if (idx + i) % step == 0]
                            if sel:
                                # Greedy (`top_k=1` keeps a single token, so the
                                # multinomial draw is deterministic).  Upstream's
                                # default is 6-beam search; see the T3AL section's
                                # deviation on caption decoding.
                                gen = model.generate(
                                    x[sel], generation_type="top_k", top_k=1,
                                    seq_len=20)
                                import open_clip
                                for j, g in zip(sel, gen):
                                    txt = open_clip.decode(g).split("<end_of_text>")[0]
                                    txt = txt.replace("<start_of_text>", "").strip()
                                    txt = txt.replace("\n", " ").replace("-", " ")
                                    caps.append((idx + j, txt))
                    idx += x.shape[0]
            except Exception as e:
                print(f"[FAIL] {ds} {vid}: {type(e).__name__}: {e}", flush=True)
                continue
            if need_f:
                if not feats:
                    print(f"[EMPTY] {ds} {vid}", flush=True)
                    continue
                F = np.concatenate(feats, 0).astype(np.float32)
                np.save(fout, F)
                nf += F.shape[0]
            if need_c:
                # T3AL parses `int(prefix) * 3` back into a feature index, so the
                # prefix we write is the feature index divided by three.
                lines = [f"{round(i / 3)}.jpg-{t}" for i, t in caps]
                cout.write_text("\n".join(lines) + ("\n" if lines else ""))
            if (n + 1) % 10 == 0 or n + 1 == len(ids):
                el = time.time() - t0
                print(f"[prog] {ds} {n+1}/{len(ids)} frames={nf} "
                      f"{el:.0f}s {(n+1)/max(el,1e-9):.3f} vid/s "
                      f"eta={(len(ids)-n-1)/max((n+1)/max(el,1e-9),1e-9)/60:.1f}min",
                      flush=True)
    print(f"[done] {time.time()-t_all:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
