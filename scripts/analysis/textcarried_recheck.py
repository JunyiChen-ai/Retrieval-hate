#!/usr/bin/env python
"""
TEXT-CARRIED RECHECK — does HateMM's "text-only >= image-only" survive deletion of the
39 empty-transcript train rows?

CPU only. Zero GPU. Zero test-set touch: reads ONLY
  data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt
  data/gt/HateMM/train.jsonl
Nothing from dev_seen / test_seen is opened.

Protocol is byte-identical to the one that produced the historical 0.8471 / 0.8255 pair
(scripts/analysis/encoder_swap_geometry.py::{l2n,build_modality,loo_knn,auc}, K=20, reused
verbatim by scripts/analysis/hatemm_lora_stream_decomp.py). We re-implement the same three
functions here rather than importing, because encoder_swap_geometry.py hardcodes the old
/data/jehc223/RGCL BASE path; the numeric bodies below are copied unchanged from it.

Decision rule is FROZEN in refine-logs/TEXTCARRIED_RECHECK_2026-08-09.md (committed 3fe55db)
BEFORE this script was run.
"""
import json
import os

import numpy as np
import torch

ROOT = "/home/jehc223/Retrieval-hate"
CACHE = f"{ROOT}/data/CLIP_Embedding/HateMM/train_openai_clip-vit-large-patch14-336_HF.pt"
GT = f"{ROOT}/data/gt/HateMM/train.jsonl"
OUT = f"{ROOT}/scripts/analysis/textcarried_recheck_OUT.json"
K = 20
HIST = {"text": 0.8471, "img": 0.8255}   # historical 744-row train-LOO AUC
HIST_MARGIN = HIST["text"] - HIST["img"]
ALIGN_TOL = 0.0005


# ---- copied verbatim from scripts/analysis/encoder_swap_geometry.py ----
def l2n(x, eps=1e-8):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def build_modality(img, txt, mode):
    if mode == "img":
        return l2n(img)
    if mode == "text":
        return l2n(txt)
    if mode == "concat":
        return np.concatenate([l2n(img), l2n(txt)], axis=1)
    raise ValueError(mode)


def auc(y, score):
    order = np.argsort(score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(score) + 1)
    n1 = np.sum(y == 1)
    n0 = np.sum(y == 0)
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((np.sum(ranks[y == 1]) - n1 * (n1 + 1) / 2) / (n1 * n0))


def loo_knn(X, y, k):
    S = X @ X.T
    n = np.linalg.norm(X, axis=1, keepdims=True)
    S = S / (n + 1e-8) / (n.T + 1e-8)
    np.fill_diagonal(S, -np.inf)
    idx = np.argpartition(-S, kth=k, axis=1)[:, :k]
    scores = np.zeros(len(y))
    for i in range(len(y)):
        nn = idx[i]
        w = S[i, nn]
        signs = 2 * y[nn] - 1
        scores[i] = np.sum(w * signs)
    pred = (scores > 0).astype(int)
    return pred, scores
# ---- end verbatim block ----


def bal_acc(y, p):
    return float(np.mean([np.mean(p[y == c] == c) for c in (0, 1) if (y == c).sum()]))


def boot_delta(y, s_text, s_img, n_boot=2000, seed=0):
    """Paired stratified bootstrap over rows: distribution of AUC_text - AUC_img.
    Query-side resampling only (memory bank held fixed) -- a descriptive spread, not
    an inferential test; the frozen rule uses the point estimate."""
    rng = np.random.default_rng(seed)
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y == 0)
    ds = []
    for _ in range(n_boot):
        b = np.concatenate([rng.choice(pos, len(pos), replace=True),
                            rng.choice(neg, len(neg), replace=True)])
        ds.append(auc(y[b], s_text[b]) - auc(y[b], s_img[b]))
    ds = np.asarray(ds)
    return float(np.percentile(ds, 2.5)), float(np.percentile(ds, 97.5)), float(np.mean(ds > 0))


def main():
    d = torch.load(CACHE, map_location="cpu", weights_only=False)
    img = d["img_feats"].float().numpy()
    txt = d["text_feats"].float().numpy()
    y = d["labels"].long().numpy()
    ids = d["ids"][0] if isinstance(d["ids"], list) and len(d["ids"]) == 1 else d["ids"]
    ids = np.asarray(ids)
    n = len(y)

    # --- identify the empty-transcript rows two independent ways ---
    gt_text = {}
    with open(GT) as f:
        for line in f:
            r = json.loads(line)
            gt_text[r["id"]] = r["text"]
    gt_empty = np.array([gt_text[i].strip() == "" for i in ids])

    # feature-side: the modal duplicated text row (the [BOS,EOS] constant vector)
    _, inv, cnt = np.unique(np.round(txt, 6), axis=0, return_inverse=True, return_counts=True)
    big = int(np.argmax(cnt))
    feat_empty = inv == big

    agree = bool((gt_empty == feat_empty).all())
    keep = ~gt_empty
    res = {
        "n_total": int(n),
        "n_empty_gt": int(gt_empty.sum()),
        "n_empty_featdup": int(feat_empty.sum()),
        "empty_id_sets_agree": agree,
        "n_unique_text_rows_744": int(len(cnt)),
        "empty_ids": sorted(ids[gt_empty].tolist()),
        "empty_hate_count": int(y[gt_empty].sum()),
        "empty_p_hate": float(y[gt_empty].mean()),
        "full_p_hate": float(y.mean()),
        "clean_p_hate": float(y[keep].mean()),
        "K": K,
        "historical": {"text": HIST["text"], "img": HIST["img"], "margin": HIST_MARGIN},
        "conditions": {},
    }
    assert agree, "empty-row identification disagrees between gt text and feature duplicates"

    for cond, m in (("full744", np.ones(n, bool)), ("clean705", keep)):
        c = {"n": int(m.sum()), "p_hate": float(y[m].mean())}
        sc = {}
        for mode in ("img", "text", "concat"):
            X = build_modality(img[m], txt[m], mode)
            pred, s = loo_knn(X, y[m], K)
            sc[mode] = s
            c[mode] = {
                "auc": auc(y[m], s),
                "acc": float(np.mean(pred == y[m])),
                "balacc": bal_acc(y[m], pred),
            }
        c["delta_text_minus_img"] = c["text"]["auc"] - c["img"]["auc"]
        lo, hi, pgt0 = boot_delta(y[m], sc["text"], sc["img"])
        c["delta_boot95"] = [lo, hi]
        c["delta_boot_frac_gt0"] = pgt0
        res["conditions"][cond] = c

    f744 = res["conditions"]["full744"]
    c705 = res["conditions"]["clean705"]
    res["alignment"] = {
        "text_diff_vs_hist": f744["text"]["auc"] - HIST["text"],
        "img_diff_vs_hist": f744["img"]["auc"] - HIST["img"],
        "tol": ALIGN_TOL,
        "pass": bool(abs(f744["text"]["auc"] - HIST["text"]) <= ALIGN_TOL
                     and abs(f744["img"]["auc"] - HIST["img"]) <= ALIGN_TOL),
    }
    res["paired"] = {
        "d_text_705_minus_744": c705["text"]["auc"] - f744["text"]["auc"],
        "d_img_705_minus_744": c705["img"]["auc"] - f744["img"]["auc"],
        "d_concat_705_minus_744": c705["concat"]["auc"] - f744["concat"]["auc"],
        "margin_744": f744["delta_text_minus_img"],
        "margin_705": c705["delta_text_minus_img"],
        "margin_shrink": c705["delta_text_minus_img"] - f744["delta_text_minus_img"],
    }
    d705 = c705["delta_text_minus_img"]
    res["verdict"] = {
        "threshold_half_original_margin": HIST_MARGIN / 2,
        "delta705": d705,
        "label": "SURVIVES" if (d705 > 0 and d705 >= HIST_MARGIN / 2) else "RETRACTED",
        "branch": ("text>img and margin >= half original" if (d705 > 0 and d705 >= HIST_MARGIN / 2)
                   else ("text advantage reversed/absent" if d705 <= 0
                         else "margin survives in sign but < half original")),
    }

    with open(OUT, "w") as f:
        json.dump(res, f, indent=1)

    print(f"n=744 empty(gt)={res['n_empty_gt']} empty(featdup)={res['n_empty_featdup']} "
          f"agree={agree} P(hate|empty)={res['empty_p_hate']:.3f} base={res['full_p_hate']:.3f}")
    for cond in ("full744", "clean705"):
        c = res["conditions"][cond]
        print(f"[{cond}] n={c['n']} pos={c['p_hate']:.3f} "
              f"img={c['img']['auc']:.4f} text={c['text']['auc']:.4f} "
              f"concat={c['concat']['auc']:.4f} "
              f"delta(text-img)={c['delta_text_minus_img']:+.4f} "
              f"boot95=[{c['delta_boot95'][0]:+.4f},{c['delta_boot95'][1]:+.4f}] "
              f"P(d>0)={c['delta_boot_frac_gt0']:.3f}")
    print("alignment vs historical:", res["alignment"])
    print("paired:", json.dumps(res["paired"], indent=1))
    print("VERDICT:", res["verdict"]["label"], "|", res["verdict"]["branch"],
          f"| delta705={d705:+.4f} threshold={HIST_MARGIN/2:+.4f}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
