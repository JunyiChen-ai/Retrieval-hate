#!/usr/bin/env python
"""REPRO campaign Wave 1 — LaGoVAD (ICLR 2026), all four corpora.

LaGoVAD's premise is that the anomaly is *defined at inference time by free
text*.  The repo ships only `src/end2end_inference.py`, a single-video demo whose
class list is hard-coded to XD-Violence; the campaign patch
(`scripts/repro_campaign/patches/LaGoVAD-PreVAD.patch`) exposes `--queries` so the
`cap_*` head — the free-text head — is reachable.  This driver is the corpus loop
around exactly that code path: the model class, the checkpoint, the CLIP ViT-B/16
visual tower, the every-8th-frame sampling and the 224x224 square resize are the
upstream demo's; only the loop, the resume, the long-video chunking and the npz
dump are ours.

DEFINITION TEXTS ARE FROZEN IN THIS FILE AND COMMITTED BEFORE THE RUN.
Every one of them is reported; none is selected after seeing a number.

Stages
  extract : CLIP ViT-B/16 image features of every 8th native frame,
            -> data/CLIP_Embedding/<DS>/lagovad_clipB16/<vid>.npy  (N, 512) fp32
  infer   : LaGoVAD forward with all frozen queries at once,
            -> idea-stage/repro_lagovad/curves/<DS>/<vid>.npz
            keys: one per variant, plus `rate` (samples/s) and `bin` (the
            binary head, kept for the record even though §3.1 of
            MODEL_ASSETS_STATUS shows it is constant across frames).

Both stages are idempotent (existing output is skipped, every write is
`.tmp` + os.replace) and print parseable PROGRESS lines.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")
LAGO = ROOT / "third_party/LaGoVAD-PreVAD"
HOME = Path.home()

VIDEO_DIR = {
    "HateMM": HOME / "data/HateMM/video",
    "MHC": HOME / "data/Multihateclip/English/video_mp4",
    "MHC_zh": HOME / "data/Multihateclip/Chinese/video",
    "HateClipSeg": ROOT / "data/video/HateClipSeg/All",
}
EXTS = (".mp4", ".webm", ".mkv", ".avi")
NO_VIDEO_STREAM = {"hate_video_147", "hate_video_292"}   # freeze §12 D2

FRAME_INTERVAL = 8          # upstream `FRAME_INTERVAL`
RESIZE = 224                # upstream CLIPProcessor size=(224, 224), center crop 224
MAX_POS = 1500              # model_config.max_position_embeddings = 1536

# ----------------------------------------------------------------- queries ---
# The main definition and the two sensitivity definitions.  All three are
# reported side by side (freeze: variants are all reported, never chosen).
QUERIES = {
    "main": ("Hateful content: speech, imagery or on-screen text that attacks, "
             "dehumanizes, demeans or incites hatred or violence against a person "
             "or a group because of a protected attribute such as race, ethnicity, "
             "national origin, religion, gender, sexual orientation or disability."),
    "sens_short": ("A person or a group is being attacked, insulted or dehumanized "
                   "because of their race, religion, gender, sexuality or nationality."),
    "sens_vad": ("Abnormal event: hateful or abusive behaviour directed at a person "
                 "or a group, including slurs, harassment, threats and violence."),
    # the reference definition, mirroring upstream's own `Normal` class text.  It
    # is reported as its own row and is the second term of the `_vsnormal` rows.
    "normal": ("Normal behavior, views or content that does not involve any hateful, "
               "abusive, offensive or otherwise unexpected event."),
}
# HateClipSeg's six released classes, one definition each (class order = freeze §4).
HCS_QUERIES = {
    "c0_normal": ("Normal, non-toxic content that does not attack, insult, sexualise "
                  "or harm anyone."),
    "c1_hateful": ("Hateful content that attacks or dehumanizes a person or a group "
                   "because of race, ethnicity, religion, gender, sexual orientation, "
                   "nationality or disability."),
    "c2_insulting": ("Insulting content: mocking, humiliating, name-calling or "
                     "otherwise demeaning a person or a group."),
    "c3_sexual": ("Sexual content: nudity, sexual acts, or sexually explicit language "
                  "and imagery."),
    "c4_violence": ("Violent content: physical fighting, assault, weapons, blood, "
                    "injury or killing."),
    "c5_harm": ("Harmful content: self-harm, suicide, dangerous acts, or content that "
                "encourages people to harm themselves or others."),
}
# Declared secondary reading, fixed here before the run: the similarity of a hate
# definition minus the similarity of the `normal` definition, i.e. the two-prompt
# contrast every other zero-shot row in this campaign uses.  Reported next to the
# raw rows, not instead of them.
CONTRASTS = ["main", "sens_short", "sens_vad"]

FEAT_SUBDIR = "lagovad_clipB16"
CURVE_DIR = ROOT / "idea-stage/repro_lagovad/curves"


def find_video(ds: str, vid: str):
    for ext in EXTS:
        p = VIDEO_DIR[ds] / f"{vid}{ext}"
        if p.exists():
            return p
    return None


def dataset_ids(ds: str) -> list[str]:
    z = np.load(ROOT / f"data/gt/frame_gt_4fps/{ds}.npz", allow_pickle=True)
    return sorted(str(v) for v in z["video_ids"])


def durations(ds: str) -> dict:
    return json.loads((ROOT / f"data/gt/frame_gt_4fps/durations_{ds}.json").read_text())


def atomic_save(path: Path, saver):
    tmp = path.with_suffix(path.suffix + ".tmp")
    saver(tmp)
    os.replace(tmp, path)


# ------------------------------------------------------------------ extract ---
def stream_frames(path: Path, batch: int):
    """Every 8th decoded frame, square-resized to 224, as uint8 RGB batches.

    Upstream writes the same selection to JPEG (`-vf select=not(mod(n,8))`,
    `-q:v 2`) and then lets CLIPProcessor resize to (224, 224) with
    `do_center_crop=True`, which for a 224 crop box is a plain square resize.
    Doing the resize inside ffmpeg (bicubic, as PIL's) avoids writing ~1.1 M JPEGs
    to disk; the only numerical difference from upstream is the absence of JPEG
    quantisation.  Checked on the smoke video: mean cosine of the two feature
    sets is reported by `--check-resize`.
    """
    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-vf", f"select=not(mod(n\\,{FRAME_INTERVAL})),scale={RESIZE}:{RESIZE}",
           "-vsync", "vfr", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    nbytes = RESIZE * RESIZE * 3
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            bufsize=nbytes * 8)
    buf = []
    try:
        while True:
            raw = proc.stdout.read(nbytes)
            if not raw or len(raw) < nbytes:
                break
            buf.append(np.frombuffer(raw, dtype=np.uint8).reshape(RESIZE, RESIZE, 3))
            if len(buf) == batch:
                yield np.stack(buf)
                buf = []
        if buf:
            yield np.stack(buf)
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass
        err = proc.stderr.read().decode("utf-8", "replace")[-400:]
        proc.stderr.close()
        proc.wait()
        if proc.returncode not in (0, None) and not buf:
            pass
        stream_frames.last_err = err


def stage_extract(args):
    from transformers import CLIPModel, CLIPImageProcessor

    dev = "cuda:0"
    clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(dev).eval()
    proc = CLIPImageProcessor.from_pretrained("openai/clip-vit-base-patch16")
    mean = torch.tensor(proc.image_mean, device=dev).view(1, 3, 1, 1)
    std = torch.tensor(proc.image_std, device=dev).view(1, 3, 1, 1)

    t0, n_done, n_skip, n_fail = time.time(), 0, 0, 0
    plan = []
    for ds in args.datasets.split(","):
        outd = ROOT / f"data/CLIP_Embedding/{ds}/{FEAT_SUBDIR}"
        outd.mkdir(parents=True, exist_ok=True)
        ids = dataset_ids(ds)
        if args.limit:
            ids = ids[: args.limit]
        for vid in ids:
            if (outd / f"{vid}.npy").exists():
                n_skip += 1
                continue
            plan.append((ds, vid, outd / f"{vid}.npy"))
    print(f"[plan] extract videos={len(plan)} already_done={n_skip}", flush=True)

    for i, (ds, vid, outp) in enumerate(plan, 1):
        if vid in NO_VIDEO_STREAM:
            n_fail += 1
            continue
        path = find_video(ds, vid)
        if path is None:
            print(f"[MISSING] {ds}/{vid}", flush=True)
            n_fail += 1
            continue
        feats = []
        try:
            for chunk in stream_frames(path, args.batch):
                x = torch.from_numpy(chunk).to(dev).permute(0, 3, 1, 2).float().div_(255.)
                x = (x - mean) / std
                with torch.inference_mode():
                    f = clip.get_image_features(pixel_values=x)
                feats.append(f.float().cpu().numpy())
        except Exception as e:
            print(f"[FAIL] {ds}/{vid}: {type(e).__name__}:{e}"[:200], flush=True)
            n_fail += 1
            continue
        if not feats:
            print(f"[EMPTY] {ds}/{vid}: {getattr(stream_frames,'last_err','')}"[:200],
                  flush=True)
            n_fail += 1
            continue
        arr = np.concatenate(feats).astype(np.float32)
        atomic_save(outp, lambda p: np.save(open(p, 'wb'), arr))
        n_done += 1
        if i % 25 == 0 or i == len(plan):
            el = time.time() - t0
            print(f"PROGRESS extract {i}/{len(plan)} ds={ds} vid={vid} n={len(arr)} "
                  f"elapsed={el:.0f}s rate={n_done/max(el,1e-9):.2f}vid/s "
                  f"eta={(len(plan)-i)/max(n_done/max(el,1e-9),1e-9):.0f}s fail={n_fail}",
                  flush=True)
    print(f"[done] extract done={n_done} skipped={n_skip} failed={n_fail} "
          f"wall={time.time()-t0:.0f}s", flush=True)


# -------------------------------------------------------------------- infer ---
def build_lagovad(cfg_path: Path, ckpt: Path):
    import importlib
    import yaml

    sys.path.insert(0, str(LAGO))
    sys.path.insert(0, str(LAGO / "src"))
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # upstream demo forces this; §3.1 of MODEL_ASSETS_STATUS records what it does
    cfg["model"]["init_args"]["model_config"]["init_args"]["verbalizer_type"] = None
    mp = cfg["model"]["class_path"].split(".")
    model_cls = getattr(importlib.import_module(".".join(mp[:-1])), mp[-1])
    cp = cfg["model"]["init_args"]["model_config"]["class_path"].split(".")
    model_cfg = getattr(importlib.import_module(".".join(cp[:-1])), cp[-1])(
        **cfg["model"]["init_args"]["model_config"]["init_args"])
    tp = cfg["model"]["init_args"]["training_config"]["class_path"].split(".")
    train_cfg = getattr(importlib.import_module(".".join(tp[:-1])), tp[-1])(
        **cfg["model"]["init_args"]["training_config"]["init_args"])
    model = model_cls(model_cfg, train_cfg).to("cuda:0")
    sd = torch.load(ckpt, map_location="cpu", weights_only=True)["state_dict"]
    miss, unexp = model.load_state_dict(sd, strict=False)
    print(f"[ckpt] missing={len(miss)} unexpected={len(unexp)}", flush=True)
    if unexp:
        print(f"[ckpt] unexpected keys: {unexp[:8]}", flush=True)
    model.eval()
    return model


def stage_infer(args):
    model = build_lagovad(Path(args.config), Path(args.ckpt))
    dur = {ds: durations(ds) for ds in args.datasets.split(",")}

    t0, n_done, n_skip, n_fail = time.time(), 0, 0, 0
    for ds in args.datasets.split(","):
        # LaGoVAD's co-attention fusion lets the *visual* stream attend over the
        # whole text set (`vis_attn(v_feat, t_feat, t_feat)`), so a query's score
        # depends on which other queries share the forward pass.  The queries are
        # therefore grouped deliberately, and the grouping is part of the frozen
        # design rather than an accident of batching:
        #   * each definition alone, so the `main` / `sens_*` / `normal` rows are
        #     not contaminated by their siblings;
        #   * each hate definition paired with the `normal` definition, which is
        #     the two-class VAD set-up upstream's own `Normal + anomaly` class
        #     list uses, giving the `_vsnormal` contrast rows;
        #   * HateClipSeg's six released classes in one forward, which is the
        #     multi-class set-up the model was built for.
        groups = [([k], [k]) for k in QUERIES]                       # solo rows
        groups += [(["normal", k], [f"{k}_pair", f"{k}_vsnormal"])
                   for k in CONTRASTS]                               # paired rows
        if ds == "HateClipSeg":
            groups.append((list(HCS_QUERIES), list(HCS_QUERIES)))    # 6-class row
        ALL = {**QUERIES, **HCS_QUERIES}
        outd = CURVE_DIR / ds
        outd.mkdir(parents=True, exist_ok=True)
        featd = ROOT / f"data/CLIP_Embedding/{ds}/{FEAT_SUBDIR}"
        ids = dataset_ids(ds)
        if args.limit:
            ids = ids[: args.limit]
        for i, vid in enumerate(ids, 1):
            outp = outd / f"{vid}.npz"
            if outp.exists() and not args.force:
                n_skip += 1
                continue
            fp = featd / f"{vid}.npy"
            if not fp.exists():
                n_fail += 1
                continue
            v = torch.from_numpy(np.load(fp)).float()
            D = float(dur[ds].get(vid, 0.0))
            if D <= 0 or len(v) == 0:
                n_fail += 1
                continue
            out, binv, T = {}, None, None
            try:
                for qkeys, outkeys in groups:
                    texts = [ALL[k] for k in qkeys]
                    sims, bins = [], []
                    for s in range(0, len(v), MAX_POS):
                        ch = v[s: s + MAX_POS].to("cuda:0")
                        batch = {"v_feat": ch[None, ...],
                                 "v_feat_l": torch.tensor([len(ch)], device="cuda:0")}
                        with torch.inference_mode():
                            o = model(batch=batch, query_captions=texts)
                        # RAW logits, not the demo's `.sigmoid()`.  SimScoreHead
                        # divides the cosine by a learned temperature, so the logit
                        # range is wide enough that sigmoid saturates to 0/1 in
                        # float32 and destroys the within-video ranking.  ROC-AUC
                        # and AP are rank metrics and the sigmoid is strictly
                        # increasing, so the raw logit is the same measurement
                        # without the saturation.
                        sims.append(o["cap_sim_mat"][0].float().cpu().numpy())   # T,S
                        bins.append(o["cap_bin_logits"][0].float().cpu().numpy())
                    sim = np.concatenate(sims)
                    T = len(sim)
                    if len(qkeys) == 2 and outkeys[-1].endswith("_vsnormal"):
                        out[outkeys[0]] = sim[:, 1].astype(np.float32)
                        out[outkeys[1]] = (sim[:, 1] - sim[:, 0]).astype(np.float32)
                    else:
                        for j, k in enumerate(outkeys):
                            out[k] = sim[:, j].astype(np.float32)
                    if binv is None:      # the binary head does not depend on the text
                        binv = np.concatenate(bins).astype(np.float32)
            except Exception as e:
                print(f"[FAIL] {ds}/{vid}: {type(e).__name__}:{e}"[:200], flush=True)
                n_fail += 1
                continue
            out["bin"] = binv
            out["rate"] = np.float32(T / D)
            atomic_save(outp, lambda p: np.savez(open(p, 'wb'), **out))
            n_done += 1
            if i % 50 == 0 or i == len(ids):
                el = time.time() - t0
                print(f"PROGRESS infer {ds} {i}/{len(ids)} done={n_done} skip={n_skip} "
                      f"fail={n_fail} elapsed={el:.0f}s", flush=True)
    print(f"[done] infer done={n_done} skipped={n_skip} failed={n_fail} "
          f"wall={time.time()-t0:.0f}s", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["extract", "infer"])
    ap.add_argument("--datasets", default="HateMM,MHC,MHC_zh,HateClipSeg")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--config", default=str(ROOT / "third_party/_ckpt/lagovad/config.yaml"))
    ap.add_argument("--ckpt", default=str(ROOT / "third_party/_ckpt/lagovad/best.ckpt"))
    args = ap.parse_args()
    if args.stage == "extract":
        stage_extract(args)
    else:
        stage_infer(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
