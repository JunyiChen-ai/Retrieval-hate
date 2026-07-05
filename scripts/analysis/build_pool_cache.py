#!/usr/bin/env python
"""P3 — build weighted-pool whole-video caches (EXP_p3_evidence_pooling §0.2).

Consumes:
  * the existing *_subclipK<K>_<model>.pt cache (per-window CLIP visual feats),
  * the whole-video cache (for text_feats / labels / ids — copied UNCHANGED),
  * the MLLM segment scores JSONL (build via score_segments_mllm.py),

and emits drop-in whole-video caches whose ONLY difference from the mean-pool
floor is the visual pooling WEIGHTS:

  <split>_p3pool_mean_HF.pt      floor : w_i = 1/K            (== sub-clip mean)
  <split>_p3pool_wsoftT1_HF.pt   PRIMARY : w = softmax(s/T=1)
  <split>_p3pool_wmild_HF.pt     SECONDARY : w = (1+s)/sum(1+s)

Each output = copy of the whole-video cache dict with img_feats REPLACED by the
pooled 16-frame sub-clip visual embedding. text_feats / labels / ids untouched.

Sanity (asserted here, so a broken build can never reach training):
  * uniform-weighted == sub-clip mean, bit-for-bit (torch.equal);
  * mean cache text/labels/ids identical to the whole-video source;
  * weighted caches share text/labels/ids with the mean cache.

NEVER overwrites the existing sub-clip or whole-video caches; only writes the
new p3pool_* tags.
"""
import argparse
import json
import os

import torch

SPLIT_TO_OUTNAME = {"train": "train", "val": "dev_seen", "test": "test_seen"}
MODEL_TAG = "openai_clip-vit-large-patch14-336_HF"


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Build P3 weighted-pool caches.")
    ap.add_argument("--dataset", type=str, default="MHC")
    ap.add_argument("--splits", type=str, default="train,val,test")
    ap.add_argument("--emb_dir", type=str, default="./data/CLIP_Embedding")
    ap.add_argument("--score_dir", type=str, default="./data/MLLM_scores")
    ap.add_argument("--num_subclips", type=int, default=4)
    ap.add_argument("--softmax_T", type=float, default=1.0)
    return ap.parse_args(argv)


def load_scores(path, K):
    """id -> LongTensor[K] of segment scores."""
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            s = list(o["scores"])[:K]
            if len(s) < K:
                s = s + [0] * (K - len(s))
            out[str(o["id"])] = torch.tensor(s, dtype=torch.float32)
    return out


def pooled(sub_VKD, weights_VK):
    """sub_VKD [V,K,D], weights_VK [V,K] (rows sum to 1) -> [V,D]."""
    return (weights_VK.unsqueeze(-1) * sub_VKD).sum(dim=1)


def build_split(dataset, split, args):
    K = args.num_subclips
    outname = SPLIT_TO_OUTNAME[split]
    ds_dir = os.path.join(args.emb_dir, dataset)
    wv_path = os.path.join(ds_dir, "{}_{}.pt".format(outname, MODEL_TAG))
    sc_path = os.path.join(ds_dir, "{}_subclipK{}_{}.pt".format(outname, K, MODEL_TAG))
    score_path = os.path.join(
        args.score_dir, dataset, "{}_segscoreK{}_qwen.jsonl".format(outname, K))
    for p in (wv_path, sc_path, score_path):
        if not os.path.exists(p):
            raise FileNotFoundError("missing input: {}".format(p))

    wv = torch.load(wv_path, map_location="cpu")
    sc = torch.load(sc_path, map_location="cpu")
    scores = load_scores(score_path, K)

    video_ids = list(sc["video_ids"])
    V = len(video_ids)
    parent = sc["subclip_parent"].long()
    # sub-clip rows must be contiguous K-per-parent in video_ids order.
    expect = torch.arange(V).repeat_interleave(K)
    assert torch.equal(parent, expect), \
        "sub-clip parent order not contiguous K-per-parent; refusing to reshape"
    D = sc["subclip_img_feats"].shape[1]
    sub = sc["subclip_img_feats"].float().view(V, K, D)

    # align scores to video_ids
    missing = [v for v in video_ids if v not in scores]
    if missing:
        print("[WARN] {}/{} videos have NO score record ({}...); using zeros "
              "(=> uniform weights => mean).".format(len(missing), V, missing[:3]))
    S = torch.stack([scores.get(v, torch.zeros(K)) for v in video_ids], dim=0)  # [V,K]

    # weight variants
    uniform = torch.full((V, K), 1.0 / K)
    w_soft = torch.softmax(S / args.softmax_T, dim=1)
    w_mild = (1.0 + S)
    w_mild = w_mild / w_mild.sum(dim=1, keepdim=True)

    img_mean = sub.view(V, K, D).mean(dim=1)
    img_unif = pooled(sub, uniform)
    img_soft = pooled(sub, w_soft)
    img_mild = pooled(sub, w_mild)

    # SANITY 1: equal weights reproduce mean bit-for-bit.
    assert torch.equal(img_unif, img_mean), \
        "uniform-weighted != sub-clip mean (bit-for-bit sanity FAILED)"

    # id order of the whole-video cache (nested [[...]]) must match sub-clip order
    wv_ids_flat = [i for sub_ in wv["ids"] for i in sub_]
    assert wv_ids_flat == video_ids, \
        "whole-video id order != sub-clip video_ids order for {}".format(split)

    def make_cache(img):
        d = dict(wv)                 # shallow copy
        d["img_feats"] = img.float().contiguous()
        return d

    out = {
        "mean": make_cache(img_mean),
        "wsoftT1": make_cache(img_soft),
        "wmild": make_cache(img_mild),
    }
    # SANITY 2: text/labels/ids identical across all three (only img differs).
    base = out["mean"]
    for tag, d in out.items():
        assert torch.equal(d["text_feats"], base["text_feats"]), tag
        assert torch.equal(d["labels"], base["labels"]), tag
        assert d["ids"] == base["ids"], tag

    for tag, d in out.items():
        op = os.path.join(ds_dir, "{}_p3pool_{}_HF.pt".format(outname, tag))
        torch.save(d, op)
        print("  wrote {}  img{}  |Δmean|max={:.4f}".format(
            op, tuple(d["img_feats"].shape),
            (d["img_feats"] - img_mean).abs().max().item()))

    # score report
    hi = (S >= 2).float().mean().item()
    print("[{}/{}] V={} K={} scores: mean {:.3f}  frac>=2 {:.3f}  "
          "per-video var mean {:.3f}".format(
              dataset, split, V, K, S.mean().item(), hi, S.var(dim=1).mean().item()))
    return {"V": V, "scores": S, "video_ids": video_ids, "labels": base["labels"]}


def main(args):
    for split in [s.strip() for s in args.splits.split(",") if s.strip()]:
        if split not in SPLIT_TO_OUTNAME:
            print("[WARN] unknown split {}".format(split))
            continue
        build_split(args.dataset, split, args)


if __name__ == "__main__":
    main(parse_args())
