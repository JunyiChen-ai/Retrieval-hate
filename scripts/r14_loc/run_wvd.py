#!/usr/bin/env python
"""R14-WVD pilot runner. Implements `idea-stage/R14_WVD_FREEZE.md` exactly.

2x2x2 factorial (objective x text substrate x representation), 5-fold video-grouped CV
inside the HateClipSeg TRAIN split only, 5 seeds. Val and test are never used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "scripts" / "r11_seg"))
sys.path.insert(0, str(ROOT / "scripts" / "r14_loc"))
import run_pilot as RP  # noqa: E402
from recon_decode import blocks_of, match_f1  # noqa: E402

K = 30
SEEDS = [4200, 4201, 4202, 4203, 4204]
NFOLD = 5
FOLD_SEED = 4210
BOOT_SEED = 4299
NBOOT = 10000
EPOCHS = 40
LAM = 1.0
DELTA = 0.020
OUT = ROOT / "idea-stage/r14_loc/out"


# ------------------------------------------------------------------ features
def l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def build_features():
    g = np.load(ROOT / "idea-stage/r11_seg/out/grid_labels.npz", allow_pickle=True)
    vids = [str(v) for v in g["video_ids"]]
    y_win = g["y_win"].astype(np.int64)
    bounds = g["bounds"]

    ct = torch.load(
        ROOT / "data/CLIP_Embedding/HateClipSeg/test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt",
        map_location="cpu")
    assert list(ct["video_ids"]) == vids
    V = l2(ct["subclip_img_feats"].float().numpy().reshape(len(vids), K, -1))

    t = np.load(ROOT / "idea-stage/r11_seg/out/text_feats.npz")
    assert [str(x) for x in t["video_ids"]] == vids
    Tc, Oc = l2(t["asr_feat"]), l2(t["ocr_feat"])
    Tm = t["asr_mask"].astype(np.float32)[..., None]
    Om = t["ocr_mask"].astype(np.float32)[..., None]

    h = np.load(OUT / "hate_text_feats.npz")
    assert [str(x) for x in h["video_ids"]] == vids
    Th, Oh = l2(h["asr_feat"]), l2(h["ocr_feat"])
    assert np.array_equal(h["asr_mask"], t["asr_mask"]) and np.array_equal(h["ocr_mask"], t["ocr_mask"])

    a = np.load(ROOT / "idea-stage/r11_seg/out/audio_feats.npz", allow_pickle=True)
    assert [str(x) for x in a["video_ids"]] == vids
    A = l2(a["w2v"])

    chans = {
        "B0": [("V", V), ("T", Tc), ("O", Oc), ("A", A)],
        "B1": [("V", V), ("T", Th), ("O", Oh), ("A", A)],
    }
    return vids, y_win, bounds, chans, Tm, Om


def relative_block(x):
    """Label-free video-relative features for one channel: leave-one-out residual and the
    within-video rank of cosine similarity to the leave-one-out centroid."""
    s = x.sum(1, keepdims=True)
    loo = (s - x) / (K - 1)
    resid = x - loo
    num = (x * loo).sum(-1)
    den = np.linalg.norm(x, axis=-1) * np.linalg.norm(loo, axis=-1)
    cos = num / np.maximum(den, 1e-8)
    rk = cos.argsort(axis=1).argsort(axis=1).astype(np.float32) / (K - 1.0)
    return resid.astype(np.float32), rk[..., None].astype(np.float32)


def assemble(chans, Tm, Om, B, C):
    parts = [x for _, x in chans[B]] + [Tm, Om]
    if C == "C1":
        for _, x in chans[B]:
            r, rk = relative_block(x)
            parts += [r, rk]
    return np.concatenate(parts, axis=-1).astype(np.float32)


# ------------------------------------------------------------------ metrics
def wv_auc_per_video(scores, y):
    out = {}
    for j in range(len(scores)):
        yy = y[j]
        if yy.min() == yy.max():
            continue
        s = scores[j]
        r = s.argsort().argsort() + 1
        p = int((yy == 1).sum()); n = int((yy == 0).sum())
        out[j] = float((r[yy == 1].sum() - p * (p + 1) / 2) / (p * n))
    return out


def macro_f1_windows(y, pred):
    f = []
    for c in (0, 1):
        tp = ((pred == c) & (y == c)).sum(); fp = ((pred == c) & (y != c)).sum()
        fn = ((pred != c) & (y == c)).sum()
        pr = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
        f.append(2 * pr * rc / max(pr + rc, 1e-9))
    return float(np.mean(f))


def decode_frozen(scores, bounds, ids, thr_per_video):
    """Frozen decoder: no smoothing, merge gaps <= 5 s, drop intervals < 12 s.
    Threshold is prevalence-matched on the training folds of the video's own held-out fold."""
    preds = {}
    for j, v in enumerate(ids):
        s = scores[j]; wb = bounds[j]; thr = thr_per_video[j]
        out, cur = [], None
        for k in range(K):
            if s[k] >= thr:
                cur = [wb[k, 0], wb[k, 1]] if cur is None else [cur[0], wb[k, 1]]
            else:
                if cur:
                    out.append(cur); cur = None
        if cur:
            out.append(cur)
        m = []
        for iv in out:
            if m and iv[0] - m[-1][1] <= 5.0:
                m[-1][1] = iv[1]
            else:
                m.append(list(iv))
        preds[v] = [tuple(x) for x in m if x[1] - x[0] >= 12.0]
    return preds


# ------------------------------------------------------------------ training
def wv_rank_loss(logits, y):
    """Equal-weight-per-video within-video pairwise ranking over gold-opposite pairs."""
    s = logits[..., 1] - logits[..., 0]
    tot, nv = 0.0, 0
    for i in range(s.shape[0]):
        pos = s[i][y[i] == 1]; neg = s[i][y[i] == 0]
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        d = pos[:, None] - neg[None, :]
        tot = tot + F.softplus(-d).mean()
        nv += 1
    if nv == 0:
        return logits.sum() * 0.0
    return tot / nv


def fit_cell(X, y, tr_idx, ho_idx, A, seed, dev):
    mu = X[tr_idx].reshape(-1, X.shape[-1]).mean(0)
    sd = np.maximum(X[tr_idx].reshape(-1, X.shape[-1]).std(0), 1e-6)
    Xn = ((X - mu) / sd).astype(np.float32)
    torch.manual_seed(seed); np.random.seed(seed)
    m = RP.PerWin(Xn.shape[-1]).to(dev)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
    Xtr = torch.tensor(Xn[tr_idx]).to(dev); ytr = torch.tensor(y[tr_idx]).to(dev)
    Xho = torch.tensor(Xn[ho_idx]).to(dev)
    for _ in range(EPOCHS):
        m.train(); opt.zero_grad()
        lo = m(Xtr)
        loss = F.cross_entropy(lo.reshape(-1, 2), ytr.reshape(-1))
        if A == "A1":
            loss = loss + LAM * wv_rank_loss(lo, ytr)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p_ho = torch.softmax(m(Xho), -1)[..., 1].cpu().numpy()
        p_tr = torch.softmax(m(Xtr), -1)[..., 1].cpu().numpy()
    return p_ho, p_tr


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    tr_ids, va_ids, te_ids = split["train"], split["val"], split["test"]
    assert not (set(tr_ids) & set(va_ids)) and not (set(tr_ids) & set(te_ids)) and not (set(va_ids) & set(te_ids))
    print(f"[guard] split disjoint OK  train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)}", flush=True)

    vids, y_all, bounds, chans, Tm, Om = build_features()
    idx = {v: i for i, v in enumerate(vids)}
    tr_pos = np.array([idx[v] for v in tr_ids])
    banned = set(va_ids) | set(te_ids)
    assert not (set(tr_ids) & banned)
    assert all(vids[i] not in banned for i in tr_pos)
    print(f"[guard] pilot touches {len(tr_pos)} train videos only; 0 val/test ids in any tensor", flush=True)

    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    golds = {v: blocks_of(gold[v]["segments"]) for v in tr_ids}

    rng = np.random.default_rng(FOLD_SEED)
    order = np.array(sorted(range(len(tr_pos)), key=lambda i: tr_ids[i]))
    perm = rng.permutation(len(order))
    fold_of = np.empty(len(order), dtype=int)
    fold_of[order[perm]] = np.arange(len(order)) % NFOLD
    print(f"[folds] sizes = {[int((fold_of == f).sum()) for f in range(NFOLD)]}", flush=True)

    y = y_all[tr_pos]
    bnd = bounds[tr_pos]
    results = {}
    per_video_auc = {}

    for B in ("B0", "B1"):
        for C in ("C0", "C1"):
            X = assemble(chans, Tm, Om, B, C)[tr_pos]
            for A in ("A0", "A1"):
                cell = f"{A}_{B}_{C}"
                aucs_seed, f1_seed, mf1_seed, bv_seed = [], [], [], []
                pv_seed = []
                for seed in SEEDS:
                    P = np.zeros((len(tr_pos), K))
                    THR = np.zeros(len(tr_pos))
                    for f in range(NFOLD):
                        ho = np.where(fold_of == f)[0]; trn = np.where(fold_of != f)[0]
                        p_ho, p_tr = fit_cell(X, y, trn, ho, A, seed + 7 * f,
                                              "cuda" if torch.cuda.is_available() else "cpu")
                        P[ho] = p_ho
                        rate = float(y[trn].mean())
                        THR[ho] = float(np.quantile(p_tr, 1.0 - rate))
                    pv = wv_auc_per_video(P, y)
                    pv_seed.append(pv)
                    aucs_seed.append(float(np.mean(list(pv.values()))))
                    preds = decode_frozen(P, bnd, tr_ids, THR)
                    f1_seed.append([match_f1(preds, golds, t) for t in (0.3, 0.5, 0.7)])
                    mf1_seed.append(macro_f1_windows(y.ravel(), (P >= THR[:, None]).astype(int).ravel()))
                    bv_seed.append(float(np.var(P.mean(1)) / np.var(P)))
                results[cell] = dict(
                    wv_auc=float(np.mean(aucs_seed)), wv_auc_sd=float(np.std(aucs_seed)),
                    f1=[float(np.mean([x[i] for x in f1_seed])) for i in range(3)],
                    win_macro_f1=float(np.mean(mf1_seed)),
                    between_video_var_share=float(np.mean(bv_seed)))
                per_video_auc[cell] = {v: float(np.mean([pv.get(i, np.nan) for pv in pv_seed]))
                                       for i, v in enumerate(tr_ids) if i in pv_seed[0]}
                r = results[cell]
                print(f"  {cell:14s} wv-AUC={r['wv_auc']:.4f} (sd {r['wv_auc_sd']:.4f})  "
                      f"F1@.3/.5/.7 = {100*r['f1'][0]:.1f}/{100*r['f1'][1]:.1f}/{100*r['f1'][2]:.1f}  "
                      f"winMacroF1={100*r['win_macro_f1']:.1f}  betwVar={r['between_video_var_share']:.3f}",
                      flush=True)

    # ------------------------------------------------------- main effects + CIs
    common = sorted(set.intersection(*[set(per_video_auc[c]) for c in per_video_auc]))
    boot = np.random.default_rng(BOOT_SEED)

    def effect(pairs):
        d = np.zeros(len(common))
        for hi, lo in pairs:
            d += np.array([per_video_auc[hi][v] - per_video_auc[lo][v] for v in common])
        return d / len(pairs)

    fx = {
        "A (objective  BCE+WVR - BCE)": [(f"A1_{B}_{C}", f"A0_{B}_{C}") for B in ("B0", "B1") for C in ("C0", "C1")],
        "B (text  HATETXT - CLIPTXT)": [(f"{A}_B1_{C}", f"{A}_B0_{C}") for A in ("A0", "A1") for C in ("C0", "C1")],
        "C (repr  ABS+REL - ABS)": [(f"{A}_{B}_C1", f"{A}_{B}_C0") for A in ("A0", "A1") for B in ("B0", "B1")],
    }
    print(f"\nmain effects on video-macro wv-AUC (n={len(common)} OOF videos with label variation),"
          f" video-clustered paired bootstrap {NBOOT}x, delta={DELTA}")
    eff_out = {}
    for name, pairs in fx.items():
        d = effect(pairs)
        bs = np.array([d[boot.integers(0, len(d), len(d))].mean() for _ in range(NBOOT)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        go = (d.mean() >= DELTA) and (lo > 0 or hi < 0) and (lo > 0)
        eff_out[name] = dict(delta=float(d.mean()), ci=[float(lo), float(hi)], go=bool(go))
        print(f"  {name:32s} D = {d.mean():+.4f}  CI [{lo:+.4f}, {hi:+.4f}]  {'GO' if go else 'no'}")

    kill = all((e["delta"] < 0.010) or (e["ci"][0] <= 0 <= e["ci"][1]) for e in eff_out.values())
    print(f"\nKILL rule (all three main effects < +0.010 or CI contains 0): {'FIRES' if kill else 'does not fire'}")

    (OUT / "results.json").write_text(json.dumps(
        dict(cells=results, effects=eff_out, kill=bool(kill), n_common=len(common)), indent=2))
    np.savez_compressed(OUT / "per_video_auc.npz",
                        **{c: np.array([per_video_auc[c][v] for v in common]) for c in per_video_auc},
                        video_ids=np.array(common))
    print(f"wrote {OUT/'results.json'}")


if __name__ == "__main__":
    main()
