#!/usr/bin/env python
"""P3 probe gate (EXP_p3_evidence_pooling §0.4) — probe BEFORE train.

TRAIN split only, raw frozen-encoder space, NO trained head. Leave-one-out kNN
vote replicating the training-time retrieval decision (cosine similarity,
'arithmetic' rank-weighted + similarity-signed vote, k=20 to match --topk 20).

Compares MEAN pooling vs WEIGHTED pooling (softmax T=1 primary, mild secondary)
of the video image embedding, for two raw representations:
  * PRIMARY gate: concat [l2n(img) | l2n(text)]  (what the head consumes)
  * diagnostic  : img-only l2n(img)              (most sensitive to the change)

Also reports the score-concentration check: within-video score variance / max,
true-Hateful vs benign.

Gate (EN): weighted >= mean on the PRIMARY concat LOO accuracy (k=20).
"""
import argparse
import os

import numpy as np
import torch
from sklearn.metrics import f1_score

SPLIT_TO_OUTNAME = {"train": "train", "val": "dev_seen", "test": "test_seen"}


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="P3 probe gate.")
    ap.add_argument("--dataset", type=str, default="MHC")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--emb_dir", type=str, default="./data/CLIP_Embedding")
    ap.add_argument("--score_dir", type=str, default="./data/MLLM_scores")
    ap.add_argument("--num_subclips", type=int, default=4)
    ap.add_argument("--ks", type=str, default="1,5,10,20")
    return ap.parse_args(argv)


def l2n(x, eps=1e-12):
    return x / (x.norm(dim=1, keepdim=True) + eps)


def loo_knn_vote(feats, labels, k):
    """Leave-one-out kNN vote mirroring compute_metrics_retrieval(use_sim,
    arithmetic): for each query, rank all OTHER items by cosine sim desc, take
    top-k, vote = sum_r w_r * sim_r * (2*y_r-1) / sum_r w_r with rank weights
    w = [k, k-1, ..., 1]; decision = vote >= 0. Returns (preds, votes)."""
    f = l2n(feats.float())
    S = f @ f.t()                       # cosine (rows l2-normed)
    S.fill_diagonal_(-2.0)              # exclude self
    n = f.shape[0]
    y = labels.long().numpy()
    ymap = y * 2 - 1
    w = np.arange(1, k + 1)[::-1].astype(np.float64)
    preds = np.zeros(n, dtype=np.int64)
    votes = np.zeros(n, dtype=np.float64)
    topk = torch.topk(S, k=min(k, n - 1), dim=1)
    idx = topk.indices.numpy()
    sim = topk.values.numpy()
    for i in range(n):
        nn = idx[i]
        sm = sim[i]
        length = len(nn)
        contrib = ymap[nn] * sm * w[:length]
        vote = contrib.sum() / w[:length].sum()
        votes[i] = vote
        preds[i] = 1 if vote >= 0.0 else 0
    return preds, votes


def evaluate(feats, labels, ks):
    y = labels.long().numpy()
    out = {}
    for k in ks:
        preds, _ = loo_knn_vote(feats, labels, k)
        acc = float((preds == y).mean())
        mf1 = float(f1_score(y, preds, average="macro", zero_division=0))
        out[k] = (acc, mf1)
    return out


def load_cache(ds_dir, outname, tag):
    p = os.path.join(ds_dir, "{}_p3pool_{}_HF.pt".format(outname, tag))
    return torch.load(p, map_location="cpu")


def load_scores_aligned(score_path, video_ids, K):
    import json
    by_id = {}
    with open(score_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            o = json.loads(line)
            s = list(o["scores"])[:K]
            if len(s) < K:
                s += [0] * (K - len(s))
            by_id[str(o["id"])] = s
    return torch.tensor([by_id.get(v, [0] * K) for v in video_ids], dtype=torch.float32)


def main(args):
    ks = [int(x) for x in args.ks.split(",")]
    outname = SPLIT_TO_OUTNAME[args.split]
    ds_dir = os.path.join(args.emb_dir, args.dataset)
    K = args.num_subclips

    mean = load_cache(ds_dir, outname, "mean")
    soft = load_cache(ds_dir, outname, "wsoftT1")
    mild = load_cache(ds_dir, outname, "wmild")
    labels = mean["labels"]
    text = mean["text_feats"].float()
    video_ids = [i for sub in mean["ids"] for i in sub]

    print("=" * 70)
    print("P3 PROBE  dataset={}  split={}  V={}  pos-frac={:.3f}".format(
        args.dataset, args.split, len(labels), float((labels > 0).float().mean())))
    print("=" * 70)

    reps = {
        "img_only": lambda c: l2n(c["img_feats"].float()),
        "concat_img_text": lambda c: torch.cat(
            [l2n(c["img_feats"].float()), l2n(text)], dim=1),
    }
    pools = {"mean": mean, "wsoftT1": soft, "wmild": mild}

    results = {}
    for rep_name, rep_fn in reps.items():
        results[rep_name] = {}
        for pool_name, cache in pools.items():
            results[rep_name][pool_name] = evaluate(rep_fn(cache), labels, ks)

    for rep_name in reps:
        print("\n--- representation: {} ---".format(rep_name))
        print("{:>10} | {}".format(
            "pool", "  ".join("k={:<2} acc/maF1".format(k) for k in ks)))
        for pool_name in pools:
            row = results[rep_name][pool_name]
            cells = "  ".join("{:.4f}/{:.4f}".format(row[k][0], row[k][1]) for k in ks)
            print("{:>10} | {}".format(pool_name, cells))
        # deltas vs mean
        for pool_name in ("wsoftT1", "wmild"):
            dcells = "  ".join(
                "{:+.4f}".format(results[rep_name][pool_name][k][0]
                                 - results[rep_name]["mean"][k][0]) for k in ks)
            print("{:>10} | Δacc vs mean: {}".format(pool_name, dcells))

    # ---- PRIMARY GATE ----
    kg = 20 if 20 in ks else max(ks)
    mean_acc = results["concat_img_text"]["mean"][kg][0]
    soft_acc = results["concat_img_text"]["wsoftT1"][kg][0]
    img_mean = results["img_only"]["mean"][kg][0]
    img_soft = results["img_only"]["wsoftT1"][kg][0]
    gate_pass = soft_acc >= mean_acc
    print("\n" + "=" * 70)
    print("PRIMARY GATE  concat LOO acc @k={}:  mean={:.4f}  weighted(soft)={:.4f}"
          "  Δ={:+.4f}  ->  {}".format(
              kg, mean_acc, soft_acc, soft_acc - mean_acc,
              "PASS" if gate_pass else "FAIL"))
    print("  diagnostic   img-only LOO acc @k={}: mean={:.4f}  weighted(soft)={:.4f}"
          "  Δ={:+.4f}".format(kg, img_mean, img_soft, img_soft - img_mean))

    # ---- score concentration ----
    score_path = os.path.join(
        args.score_dir, args.dataset, "{}_segscoreK{}_qwen.jsonl".format(outname, K))
    if os.path.exists(score_path):
        S = load_scores_aligned(score_path, video_ids, K)
        y = labels.long()
        hate = y == 1
        beni = y == 0
        var = S.var(dim=1)
        mx = S.max(dim=1).values
        print("\n  SCORE CONCENTRATION (true-Hateful vs benign train videos):")
        print("    within-video score VAR : hate {:.3f}  benign {:.3f}  Δ {:+.3f}".format(
            var[hate].mean().item(), var[beni].mean().item(),
            var[hate].mean().item() - var[beni].mean().item()))
        print("    within-video MAX score : hate {:.3f}  benign {:.3f}  Δ {:+.3f}".format(
            mx[hate].float().mean().item(), mx[beni].float().mean().item(),
            mx[hate].float().mean().item() - mx[beni].float().mean().item()))
        print("    mean segment score     : hate {:.3f}  benign {:.3f}".format(
            S[hate].mean().item(), S[beni].mean().item()))
        print("    frac windows score>=2  : hate {:.3f}  benign {:.3f}".format(
            (S[hate] >= 2).float().mean().item(), (S[beni] >= 2).float().mean().item()))
    print("=" * 70)
    return gate_pass


if __name__ == "__main__":
    main(parse_args())
