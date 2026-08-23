#!/usr/bin/env python
"""Repair + explicitly mark degenerate rows in the CLIP feature caches.

Never overwrites an existing cache. Writes `<split>_<model>_HF-degenfix1.pt` with

  * `img_feats` repaired for rows whose original vector was a silent zero-fill
    caused by a mid-stream decode abort (robust prefix decoder + the *same*
    CLIP encode path as `src/utils/generate_VideoCLIP_embedding_HF.py`),
  * a new `degen_flags` dict {video_id: CODE} marking every row that is
    degenerate and NOT silently sharing a vector any more,
  * a `degen_meta` block with provenance and SHA-256 of the source cache.

Codes
  BLACK_VIDEO                 source video track decodes but is all-black end to
                              end -> CLIP(black) constant, shared across ids.
                              Not repairable: the source has no picture.
  DECODE_TRUNCATED_REPAIRED   source mp4 truncated; original row was all zeros;
                              re-encoded from the decodable prefix.
  DECODE_FAIL_UNREPAIRABLE    source mp4 unreadable (0 decodable frames); row is
                              all zeros and stays zero, but is now flagged.
  DUP_SOURCE_VERIFIED:<md5>   two or more ids point at byte-identical (or
                              frame-identical) source video -> identical feature
                              is correct, it is a dataset-level duplicate.
  DUP_SOURCE_UNVERIFIED:<g>   identical feature, raw video not available locally
                              to confirm at the source.
"""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

MODEL = "openai_clip-vit-large-patch14-336_HF"
HF_MODEL = "openai/clip-vit-large-patch14-336"
NUM_FRAMES = 8
IMAGE_SIZE = 336
BATCH = 32
SUFFIX = "degenfix1"

# ---------------------------------------------------------------- ledger ----
# Frozen from the forensic scan (scripts/analysis/degen_feat_scan.py) plus
# source-file inspection.  See refine-logs/DEGEN_FEATURE_FIX_2026-08-09.md.
BLACK = {
    ("HateMM", "train"): [
        "hate_video_76", "hate_video_109", "hate_video_127", "hate_video_298",
        "hate_video_308", "non_hate_video_25", "non_hate_video_90",
        "non_hate_video_110", "non_hate_video_308", "non_hate_video_395",
        "non_hate_video_470",
    ],
    ("HateMM", "dev_seen"): ["non_hate_video_101", "hate_video_34"],
    ("HateMM", "test_seen"): ["non_hate_video_140", "hate_video_273", "hate_video_295"],
}
DUP_VERIFIED = {
    ("HateMM", "train"): {
        "hate_video_50": "84f69bdbe438", "non_hate_video_338": "84f69bdbe438",
        "hate_video_63": "84f69bdbe438",
        "hate_video_59": "e718f5935977", "hate_video_297": "e718f5935977",
    },
    ("MHC_zh", "train"): {
        "BV1ka4y1m7Ti": "e8e9e9370645", "BV1UT4y1p7WS": "e8e9e9370645",
    },
    ("HateClipSeg", "test_seen"): {
        "bit_1kW6COenXnwa": "217131428f50", "bit_25xl3W3FOAU8": "217131428f50",
        "bit_2AX22tKwbovE": "9d8913a23df0", "bit_IZ0vKg6zRxQC": "9d8913a23df0",
        "bit_JUPPGbicIM0r": "a715817bcb82", "bit_LGoOw2bdKb39": "a715817bcb82",
        "bit_gWeEH6fKQUgP": "a351c3fb60c8", "bit_mQ5gX03QAX9I": "a351c3fb60c8",
        "bit_xKSJF6iNYyde": "8efb00c1c327", "bit_zfVfpvwhregD": "8efb00c1c327",
        # different container md5, byte-identical decoded frames (verified)
        "bit_T8WbPXzZTebJ": "frames-identical", "bit_jubQcJCZQ3dQ": "frames-identical",
    },
}
DUP_UNVERIFIED = {
    ("ImpliHateVid", "train"): {
        "IM_476": "g1", "NH_933": "g1",       # label conflict 1 vs 0
        "EX_80": "g2", "EX_81": "g2",
        "EX_172": "g3", "EX_224": "g3",
        "EX_74": "g4", "EX_297": "g4",
    },
}
# id -> raw video path, rows whose original feature is an all-zero silent fill
REPAIR = {
    ("HateMM", "train"): {
        "hate_video_95": os.path.expanduser("~/data/HateMM/video/hate_video_95.mp4"),
    },
}
UNREPAIRABLE_ZERO = {
    ("HateClipSeg", "test_seen"): ["yt_NzvfkIYS5Yg"],
}


def log(m):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), m), flush=True)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


# --------------------------------------------------------------- decoding ---
def decode_prefix(video_path, num_frames):
    """Decode every frame PyAV can read, tolerating a mid-stream abort.

    The stock `_decode_with_pyav` lets the abort propagate, so a truncated file
    yields a zero vector.  Here the decodable prefix is kept and the frame grid
    is sampled over it.  Returns (frames, n_decoded, n_declared).
    """
    import av
    from PIL import Image  # noqa: F401  (frame.to_image needs PIL present)

    container = av.open(video_path)
    declared = container.streams.video[0].frames
    frames = []
    try:
        for frame in container.decode(video=0):
            frames.append(frame.to_image().convert("RGB"))
    except Exception as e:  # noqa: BLE001
        log("  prefix decode aborted after %d frames: %r" % (len(frames), e))
    finally:
        container.close()
    if not frames:
        return None, 0, declared
    idx = np.round(np.linspace(0, len(frames) - 1, num_frames)).astype(int)
    return [frames[i] for i in idx], len(frames), declared


def build_encoder(device):
    from transformers import CLIPImageProcessor, CLIPVisionModel

    pre = CLIPImageProcessor.from_pretrained(HF_MODEL)
    vis = CLIPVisionModel.from_pretrained(HF_MODEL).to(device).eval()
    return pre, vis


@torch.no_grad()
def encode_frames(frames, pre, vis, device):
    """Byte-for-byte the recipe in generate_VideoCLIP_embedding_HF.encode_frames."""
    from PIL import Image

    resized = [im.resize((IMAGE_SIZE, IMAGE_SIZE), Image.BICUBIC) for im in frames]
    pooled = []
    for s in range(0, len(resized), BATCH):
        inp = pre(images=resized[s:s + BATCH], return_tensors="pt")
        out = vis(pixel_values=inp["pixel_values"].to(device))
        pooled.append(out.pooler_output.detach().cpu().float())
    return torch.cat(pooled, 0).mean(0)


# ------------------------------------------------------------------ main ----
def flat_ids(d):
    raw = d["ids"]
    return list(raw[0]) if (len(raw) == 1 and isinstance(raw[0], list)) else list(raw)


def process(ds, split, root, device, encoder, dry):
    src = Path(root) / ds / ("%s_%s.pt" % (split, MODEL))
    if not src.exists():
        return None
    dst = src.with_name("%s_%s-%s.pt" % (split, MODEL, SUFFIX))
    d = torch.load(src, map_location="cpu")
    ids = flat_ids(d)
    idx = {v: i for i, v in enumerate(ids)}
    img = torch.as_tensor(d["img_feats"]).clone()
    if img.dim() == 3 and img.shape[0] == 1:
        img = img[0]

    flags = {}
    for v in BLACK.get((ds, split), []):
        flags[v] = "BLACK_VIDEO"
    for v, m in DUP_VERIFIED.get((ds, split), {}).items():
        flags[v] = "DUP_SOURCE_VERIFIED:" + m
    for v, g in DUP_UNVERIFIED.get((ds, split), {}).items():
        flags[v] = "DUP_SOURCE_UNVERIFIED:" + g
    for v in UNREPAIRABLE_ZERO.get((ds, split), []):
        flags[v] = "DECODE_FAIL_UNREPAIRABLE"

    repaired = {}
    for v, path in REPAIR.get((ds, split), {}).items():
        if v not in idx:
            raise SystemExit("HALT: repair id %s not in %s" % (v, src))
        before = img[idx[v]].clone()
        if float(before.abs().sum()) != 0.0:
            raise SystemExit("HALT: %s was expected to be an all-zero row" % v)
        log("repairing %s from %s" % (v, path))
        frames, ndec, ndecl = decode_prefix(path, NUM_FRAMES)
        if frames is None:
            flags[v] = "DECODE_FAIL_UNREPAIRABLE"
            log("  no decodable frames -> left as zero, flagged")
            continue
        if dry:
            log("  DRY: would encode %d frames (%d/%d decodable)" % (len(frames), ndec, ndecl))
            continue
        vec = encode_frames(frames, encoder[0], encoder[1], device)
        img[idx[v]] = vec
        flags[v] = "DECODE_TRUNCATED_REPAIRED"
        repaired[v] = {
            "frames_decoded": int(ndec), "frames_declared": int(ndecl),
            "coverage": round(ndec / max(ndecl, 1), 4),
            "new_norm": round(float(vec.norm()), 6),
            "old_norm": round(float(before.norm()), 6),
        }
        log("  ok: %d/%d frames decodable, new |v|=%.4f" % (ndec, ndecl, float(vec.norm())))

    if dry:
        return {"src": str(src), "dry": True, "n_flags": len(flags)}

    out = dict(d)
    out["img_feats"] = img
    out["degen_flags"] = flags
    out["degen_meta"] = {
        "produced": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "producer": "scripts/analysis/degen_feat_fix.py",
        "source_cache": str(src),
        "source_sha256": sha256_file(src),
        "repaired": repaired,
        "codes": {
            "BLACK_VIDEO": "source video track is all-black end to end; feature is "
                           "CLIP(black) and is shared with every other black video",
            "DECODE_TRUNCATED_REPAIRED": "source truncated; original row was a silent "
                                         "zero-fill; re-encoded from decodable prefix",
            "DECODE_FAIL_UNREPAIRABLE": "source unreadable; row stays all-zero, flagged",
            "DUP_SOURCE_VERIFIED": "raw video byte/frame-identical to another id",
            "DUP_SOURCE_UNVERIFIED": "feature identical to another id; raw video not "
                                     "available locally to confirm",
        },
    }
    torch.save(out, dst)
    rec = {
        "dataset": ds, "split": split,
        "src": str(src), "src_sha256": out["degen_meta"]["source_sha256"],
        "dst": str(dst), "dst_sha256": sha256_file(dst),
        "n": len(ids), "n_flagged": len(flags), "repaired": repaired,
        "flags": flags,
    }
    log("wrote %s (%d rows, %d flagged, %d repaired)"
        % (dst.name, len(ids), len(flags), len(repaired)))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(ROOT / "data/CLIP_Embedding"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=str(ROOT / "artifacts/degen_feat_fix/ledger.json"))
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()

    targets = sorted(set(list(BLACK) + list(DUP_VERIFIED) + list(DUP_UNVERIFIED)
                         + list(REPAIR) + list(UNREPAIRABLE_ZERO)))
    need_gpu = any(k in REPAIR for k in targets)
    enc = (None, None)
    if need_gpu and not a.dry:
        log("loading CLIP vision tower on %s" % a.device)
        enc = build_encoder(a.device)

    recs = []
    for ds, split in targets:
        r = process(ds, split, a.root, a.device, enc, a.dry)
        if r:
            recs.append(r)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(recs, open(a.out, "w"), indent=1)
    log("ledger -> %s" % a.out)


if __name__ == "__main__":
    main()
