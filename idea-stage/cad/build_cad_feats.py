"""CAD step 3 -- encode the rewritten transcripts and assemble the three arm caches.

Encoder path is imported verbatim from idea-stage/text_merge/extract_text_feats.py
(which is itself a bit-equivalent, memory-lean wrapper around
src/utils/generate_VideoMLLM_embedding_HF._encode, verified by --verify_lean).
NOTHING about the encoder changes: for an augmented row the SAME 8 frames of the SAME
original video are decoded, and the only thing that differs from the A0 row is the
string in the `Transcript: ` slot.

  python build_cad_feats.py            # encode rewritten transcripts (resumable)
  python build_cad_feats.py --assemble # write the three arm caches

Arms (all three share dev_seen / test_seen byte-for-byte with A0; only TRAIN differs):
  A0        744 original rows                                        (baseline)
  CAD       744 + N augmented rows: img_feats copied from the source hate video,
            text_feats = encode(original frames, REWRITTEN transcript), label 0
  CTRLRAND  744 + N control rows:   img_feats copied from the SAME source hate video,
            text_feats = the A0 text feature of a randomly drawn, distinct
            non-hate TRAIN video, label 0
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TM = os.path.join(ROOT, "idea-stage", "text_merge")
sys.path.insert(0, HERE)
sys.path.insert(0, TM)
sys.path.insert(0, os.path.join(ROOT, "src", "utils"))
from extract_text_feats import (encode_lean, text_prompt, MODEL_ID, NUM_FRAMES,  # noqa: E402
                                MAX_PIXELS, wait_for_vram, n_prompt_tokens)
import generate_VideoMLLM_embedding_HF as GEN  # noqa: E402
from gates import run as run_gates  # noqa: E402

A0_TAG = "TEXTMERGE-A0"                 # the frozen A0 cache re-extracted by TEXT_MERGE
OUT_PREFIX = "CAD"
ARMS = ["A0", "CAD", "CTRLRAND"]
SPLIT_OUT = {"train": "train", "val": "dev_seen", "test": "test_seen"}
CACHE = os.path.join(HERE, "feats", "cache")
OUTDIR = os.path.join(ROOT, "data", "CLIP_Embedding", "HateMM")
DONOR_SEED = 20260813


def titles():
    t = {}
    for split in SPLIT_OUT:
        for it in GEN.read_gt(os.path.join(ROOT, "data", "gt", "HateMM",
                                           "%s.jsonl" % split)):
            t[it["id"]] = it["title"]
    return t


# ------------------------------------------------------------------ encode
def run_encode(a):
    acc = run_gates(ROOT, verbose=False)["accepted"]
    ids = sorted(acc)
    ttl = titles()
    os.makedirs(CACHE, exist_ok=True)
    todo = [v for v in ids if not os.path.exists(os.path.join(CACHE, v + ".pt"))]
    print("[plan] %d accepted rewrites, %d still to encode" % (len(ids), len(todo)),
          flush=True)
    if not todo:
        return

    from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
    if a.offload_gib:
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
    from extract_text_feats import _NoLMHead
    model.lm_head = _NoLMHead()
    processor = AutoProcessor.from_pretrained(MODEL_ID, max_pixels=MAX_PIXELS)

    video_root = os.path.join(ROOT, "data", "video", "HateMM", "All")
    t0 = time.time()
    for k, vid in enumerate(todo):
        frames, ok = GEN.load_video_frames(os.path.join(video_root, vid + ".mp4"),
                                           NUM_FRAMES)
        if not ok:
            # An augmented row whose frames cannot be decoded would carry a zero image
            # vector; drop it instead (recorded, counted at assemble time).
            torch.save({"vec": None, "ok": False}, os.path.join(CACHE, vid + ".pt"))
            print("[encode] %s DECODE_FAIL -> dropped" % vid, flush=True)
            continue
        p = text_prompt(acc[vid]["rewritten"], ttl[vid])
        ntok = n_prompt_tokens(processor, frames, p)
        v = encode_lean(frames, p, processor, model, "cuda", span="response")
        torch.save({"vec": v, "ok": True, "n_tokens": ntok},
                   os.path.join(CACHE, vid + ".pt"))
        if (k + 1) % 10 == 0:
            el = time.time() - t0
            print("[encode] %d/%d  %.1fs elapsed  %.2fs/video  eta %.1fmin"
                  % (k + 1, len(todo), el, el / (k + 1),
                     el / (k + 1) * (len(todo) - k - 1) / 60), flush=True)
    print("[encode] DONE %d videos in %.1f min" % (len(todo), (time.time() - t0) / 60),
          flush=True)


# ------------------------------------------------------------------ assemble
def load_a0(split_out):
    return torch.load(os.path.join(OUTDIR, "%s_%s.pt" % (split_out, A0_TAG)),
                      map_location="cpu")


def run_assemble(a):
    gates = run_gates(ROOT, verbose=False)
    acc = gates["accepted"]
    tr = load_a0("train")
    tr_ids = [i for s in tr["ids"] for i in s]
    row = {v: i for i, v in enumerate(tr_ids)}
    labels = tr["labels"]
    img = tr["img_feats"].float()
    txt = tr["text_feats"].float()

    aug_ids, aug_vecs, decode_fail = [], [], []
    for vid in sorted(acc):
        p = os.path.join(CACHE, vid + ".pt")
        if not os.path.exists(p):
            raise SystemExit("missing encode cache for %s -- run encode first" % vid)
        d = torch.load(p, map_location="cpu")
        if not d.get("ok"):
            decode_fail.append(vid)
            continue
        aug_ids.append(vid)
        aug_vecs.append(d["vec"].float())
    n = len(aug_ids)
    print("[assemble] %d augmented rows (%d dropped: frame decode failure)"
          % (n, len(decode_fail)))

    # donor draw for CTRLRAND: distinct non-hate TRAIN videos, no replacement
    nonhate = [v for v in tr_ids if int(labels[row[v]]) == 0]
    rng = np.random.default_rng(DONOR_SEED)
    perm = rng.permutation(len(nonhate))
    if n > len(nonhate):
        raise SystemExit("more augmented rows than non-hate donors")
    donors = [nonhate[int(perm[i])] for i in range(n)]

    aug_txt = torch.stack(aug_vecs, dim=0) if n else torch.zeros(0, txt.shape[1])
    src_img = torch.stack([img[row[v]] for v in aug_ids], dim=0) if n \
        else torch.zeros(0, img.shape[1])
    ctl_txt = torch.stack([txt[row[v]] for v in donors], dim=0) if n \
        else torch.zeros(0, txt.shape[1])
    new_lab = torch.zeros(n, dtype=labels.dtype)

    packs = {
        "A0": (tr_ids, img, txt, labels),
        "CAD": (tr_ids + [v + "__cad" for v in aug_ids],
                torch.cat([img, src_img], 0), torch.cat([txt, aug_txt], 0),
                torch.cat([labels, new_lab], 0)),
        "CTRLRAND": (tr_ids + [v + "__ctrl" for v in aug_ids],
                     torch.cat([img, src_img], 0), torch.cat([txt, ctl_txt], 0),
                     torch.cat([labels, new_lab], 0)),
    }
    for arm, (i_, im, tx, lb) in packs.items():
        assert len(i_) == len(set(i_)), "duplicate ids in arm %s" % arm
        assert im.shape[0] == tx.shape[0] == lb.shape[0] == len(i_)
        torch.save({"ids": [i_], "img_feats": im, "text_feats": tx, "labels": lb},
                   os.path.join(OUTDIR, "train_%s-%s.pt" % (OUT_PREFIX, arm)))
        print("[assemble] train %-9s N=%d  hate=%d" % (arm, len(i_), int(lb.sum())))

    # dev_seen / test_seen: identical A0 copies for every arm (test untouched)
    for split_out in ("dev_seen", "test_seen"):
        d = load_a0(split_out)
        for arm in ARMS:
            torch.save(d, os.path.join(OUTDIR, "%s_%s-%s.pt"
                                       % (split_out, OUT_PREFIX, arm)))
        print("[assemble] %s copied to all %d arms (N=%d)"
              % (split_out, len(ARMS), sum(len(s) for s in d["ids"])))

    meta = {"n_augmented": n, "decode_fail": decode_fail,
            "aug_ids": aug_ids, "donors": dict(zip(aug_ids, donors)),
            "gate_counts": gates["counts"], "donor_seed": DONOR_SEED,
            "a0_tag": A0_TAG}
    with open(os.path.join(HERE, "assemble_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, ensure_ascii=False)
    print("[assemble] wrote", os.path.join(HERE, "assemble_meta.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--assemble", action="store_true")
    ap.add_argument("--offload_gib", type=int, default=0)
    ap.add_argument("--need_vram", type=int, default=19000)
    a = ap.parse_args()
    if a.assemble:
        run_assemble(a)
    else:
        run_encode(a)


if __name__ == "__main__":
    main()
