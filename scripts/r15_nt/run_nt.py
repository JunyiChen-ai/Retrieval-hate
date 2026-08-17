#!/usr/bin/env python
"""R15-NT pilot runner. Implements `idea-stage/R15_NT_FREEZE.md` exactly.

Part 1: seven-arm matched channel-composition panel (freeze section 2), 5-fold video-grouped
CV inside the HateClipSeg TRAIN split only, 5 seeds, no epoch selection.
Part 2: fixed-score falsification panel R15-FS (freeze section 4) on the dumped OOF
seed-averaged per-window scores of arm ALL. No fitting.

Val and test are never opened. Single submission.
"""
from __future__ import annotations

import builtins
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata

ROOT = Path("/home/jehc223/Retrieval-hate")

# ------------------------------------------------------------------ path guard
_real_open = builtins.open


def _guarded_open(file, *a, **kw):
    if "test.jsonl" in str(file):
        raise RuntimeError(f"[path-guard] refusing to open {file}")
    return _real_open(file, *a, **kw)


builtins.open = _guarded_open

sys.path.insert(0, str(ROOT / "scripts" / "r11_seg"))
sys.path.insert(0, str(ROOT / "scripts" / "r14_loc"))
import run_pilot as RP  # noqa: E402
from recon_decode import blocks_of, match_f1  # noqa: E402

K = 30
SEEDS = [4300, 4301, 4302, 4303, 4304]
NFOLD = 5
FOLD_SEED = 4310
BOOT_SEED = 4399
NBOOT = 10000
EPOCHS = 40
DELTA = 0.010
OUT = ROOT / "idea-stage/r15_nt/out"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

ARMS = ["ALL", "AUD", "VIS", "TXT", "ALLCENT", "AUDCENT", "AUDVIS0"]


# ------------------------------------------------------------------ features
def l2(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-8)


def cent_block(x, mask=None):
    """Leave-one-out within-video centering, per video, per channel block.

    cent(x)_k = x_k - mean_{j != k} x_j  computed as  x_k - (n*mean - x_k)/(n-1).
    For a masked block the mean runs over non-empty windows only and empty windows are
    left at the zero vector.  When a video has n_valid <= 1 there is no other window to
    subtract; the leave-one-out mean is taken as the zero vector (documented plumbing
    convention, no design choice is available here).
    """
    x = x.astype(np.float64)
    if mask is None:
        S = x.sum(1, keepdims=True)
        loo = (S - x) / (K - 1.0)
        return (x - loo).astype(np.float32)
    m = mask.astype(bool)
    mf = m.astype(np.float64)[..., None]
    xm = x * mf                                        # empty windows are already zero
    S = xm.sum(1, keepdims=True)
    n = mf.sum(1, keepdims=True)                       # (V,1,1) valid count per video
    denom = np.maximum(n - 1.0, 1.0)
    loo = (S - xm) / denom
    loo = np.where(n > 1.0, loo, 0.0)
    out = (xm - loo) * mf                              # empty windows stay at zero
    return out.astype(np.float32)


def build_blocks(tr_pos, vids):
    """Return the frozen channel blocks, already restricted to the train videos."""
    ct = torch.load(
        ROOT / "data/CLIP_Embedding/HateClipSeg/test_seen_subclipK30_openai_clip-vit-large-patch14-336_HF.pt",
        map_location="cpu")
    assert list(ct["video_ids"]) == vids
    V = l2(ct["subclip_img_feats"].float().numpy().reshape(len(vids), K, -1))[tr_pos]

    t = np.load(ROOT / "idea-stage/r11_seg/out/text_feats.npz")
    assert [str(x) for x in t["video_ids"]] == vids
    Tc = l2(t["asr_feat"])[tr_pos]
    Oc = l2(t["ocr_feat"])[tr_pos]
    Tmask = t["asr_mask"].astype(bool)[tr_pos]
    Omask = t["ocr_mask"].astype(bool)[tr_pos]
    Tm = Tmask.astype(np.float32)[..., None]
    Om = Omask.astype(np.float32)[..., None]

    a = np.load(ROOT / "idea-stage/r11_seg/out/audio_feats.npz", allow_pickle=True)
    assert [str(x) for x in a["video_ids"]] == vids
    A = l2(a["w2v"])[tr_pos]

    blocks = dict(V=V.astype(np.float32), T=Tc.astype(np.float32), O=Oc.astype(np.float32),
                  A=A.astype(np.float32), M=np.concatenate([Tm, Om], axis=-1))
    blocks["cV"] = cent_block(V)
    blocks["cA"] = cent_block(A)
    blocks["cT"] = cent_block(Tc, Tmask)
    blocks["cO"] = cent_block(Oc, Omask)
    return blocks


ARM_SPEC = {
    "ALL":     ["V", "T", "O", "A", "M"],
    "AUD":     ["A"],
    "VIS":     ["V"],
    "TXT":     ["T", "O", "M"],
    "ALLCENT": ["cV", "cT", "cO", "cA", "M"],
    "AUDCENT": ["A", "cV", "cT", "cO", "M"],
    "AUDVIS0": ["A", "T", "O", "M"],
}
ARM_WIDTH = {"ALL": 3586, "AUD": 1024, "VIS": 1024, "TXT": 1538,
             "ALLCENT": 3586, "AUDCENT": 3586, "AUDVIS0": 2562}


def assemble(blocks, arm):
    X = np.concatenate([blocks[b] for b in ARM_SPEC[arm]], axis=-1).astype(np.float32)
    assert X.shape[-1] == ARM_WIDTH[arm], (arm, X.shape[-1], ARM_WIDTH[arm])
    return X


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


def _auc_mid(s, yy):
    """Mann-Whitney statistic with midranks for ties (freeze section: 'a broadcast predictor
    scores exactly 0.500 by construction').  Identical to `wv_auc_per_video` whenever the
    scores in a video are all distinct; the frozen argsort-of-argsort ranking silently breaks
    ties by array index, which is fine for continuous model scores but wrong for the FS-B
    pooled read-outs, where pooling produces exact ties by construction."""
    r = rankdata(s, method="average")
    p = int((yy == 1).sum()); n = int((yy == 0).sum())
    return float((r[yy == 1].sum() - p * (p + 1) / 2) / (p * n))


def wv_auc_per_video_ties(scores, y):
    out = {}
    for j in range(len(scores)):
        yy = np.asarray(y[j])
        if len(yy) == 0 or yy.min() == yy.max():
            continue
        out[j] = _auc_mid(np.asarray(scores[j]), yy)
    return out


def wv_auc_per_video_ragged(score_list, y_list):
    """Same statistic, for read-outs where the per-video support differs (FS-A)."""
    out = {}
    for j, (s, yy) in enumerate(zip(score_list, y_list)):
        yy = np.asarray(yy)
        if len(yy) == 0 or yy.min() == yy.max():
            continue
        out[j] = _auc_mid(np.asarray(s), yy)
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
    """Frozen decoder: no smoothing, merge gaps <= 5 s, drop intervals < 12 s."""
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
def fit_cell(X, y, tr_idx, ho_idx, seed, dev):
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
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        p_ho = torch.softmax(m(Xho), -1)[..., 1].cpu().numpy()
        p_tr = torch.softmax(m(Xtr), -1)[..., 1].cpu().numpy()
    return p_ho, p_tr


# ------------------------------------------------------------------ inference
def boot_ci(d, rng, levels=(95.0,)):
    """Video-clustered paired bootstrap over the per-video paired differences."""
    n = len(d)
    bs = np.array([d[rng.integers(0, n, n)].mean() for _ in range(NBOOT)])
    out = {}
    for lv in levels:
        a = (100.0 - lv) / 2.0
        lo, hi = np.percentile(bs, [a, 100.0 - a])
        out[f"ci{lv:g}"] = [float(lo), float(hi)]
    return out


def main():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"[env] device={DEV} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}", flush=True)

    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    tr_ids, va_ids, te_ids = split["train"], split["val"], split["test"]
    assert not (set(tr_ids) & set(va_ids)) and not (set(tr_ids) & set(te_ids)) \
        and not (set(va_ids) & set(te_ids))
    print(f"[guard] split disjoint OK  train={len(tr_ids)} val={len(va_ids)} test={len(te_ids)}",
          flush=True)

    g = np.load(ROOT / "idea-stage/r11_seg/out/grid_labels.npz", allow_pickle=True)
    vids = [str(v) for v in g["video_ids"]]
    y_all = g["y_win"].astype(np.int64)
    bounds_all = g["bounds"]
    idx = {v: i for i, v in enumerate(vids)}
    tr_pos = np.array([idx[v] for v in tr_ids])
    banned = set(va_ids) | set(te_ids)
    assert not (set(tr_ids) & banned)
    assert all(vids[i] not in banned for i in tr_pos)
    print(f"[guard] pilot touches {len(tr_pos)} train videos only; 0 val/test ids in any tensor",
          flush=True)
    print("[guard] path guard active: any path containing 'test.jsonl' raises", flush=True)

    blocks = build_blocks(tr_pos, vids)
    y = y_all[tr_pos]
    bnd = bounds_all[tr_pos]

    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    golds = {v: blocks_of(gold[v]["segments"]) for v in tr_ids}
    gold_raw = {v: gold[v]["segments"] for v in tr_ids}

    rng = np.random.default_rng(FOLD_SEED)
    order = np.array(sorted(range(len(tr_pos)), key=lambda i: tr_ids[i]))
    perm = rng.permutation(len(order))
    fold_of = np.empty(len(order), dtype=int)
    fold_of[order[perm]] = np.arange(len(order)) % NFOLD
    print(f"[folds] sizes = {[int((fold_of == f).sum()) for f in range(NFOLD)]}", flush=True)

    n_var = int(sum(1 for j in range(len(y)) if y[j].min() != y[j].max()))
    assert n_var == 193, f"expected 193 train videos with within-video label variation, got {n_var}"
    print(f"[endpoint] OOF train videos with within-video label variation = {n_var}  (ASSERT 193 OK)",
          flush=True)

    # ------------------------------------------------------------- part 1
    results = {}
    per_video_auc = {}
    score_dump = {}
    print(f"\n=== PART 1: seven-arm matched channel-composition panel "
          f"({len(ARMS)} arms x {NFOLD} folds x {len(SEEDS)} seeds = "
          f"{len(ARMS)*NFOLD*len(SEEDS)} head fits) ===", flush=True)
    for arm in ARMS:
        X = assemble(blocks, arm)
        aucs_seed, f1_seed, mf1_seed, bv_seed, pv_seed = [], [], [], [], []
        P_seed = []
        for seed in SEEDS:
            P = np.zeros((len(tr_pos), K))
            THR = np.zeros(len(tr_pos))
            for f in range(NFOLD):
                ho = np.where(fold_of == f)[0]; trn = np.where(fold_of != f)[0]
                p_ho, p_tr = fit_cell(X, y, trn, ho, seed + 7 * f, DEV)
                P[ho] = p_ho
                rate = float(y[trn].mean())
                THR[ho] = float(np.quantile(p_tr, 1.0 - rate))
                print(f"  [progress] arm={arm:8s} seed={seed} fold={f} done "
                      f"t={time.time()-t0:.1f}s", flush=True)
            pv = wv_auc_per_video(P, y)
            pv_seed.append(pv)
            P_seed.append(P.copy())
            aucs_seed.append(float(np.mean(list(pv.values()))))
            preds = decode_frozen(P, bnd, tr_ids, THR)
            f1_seed.append([match_f1(preds, golds, t) for t in (0.3, 0.5, 0.7)])
            mf1_seed.append(macro_f1_windows(y.ravel(), (P >= THR[:, None]).astype(int).ravel()))
            bv_seed.append(float(np.var(P.mean(1)) / np.var(P)))
        results[arm] = dict(
            width=int(X.shape[-1]),
            wv_auc=float(np.mean(aucs_seed)), wv_auc_sd=float(np.std(aucs_seed)),
            wv_auc_per_seed=[float(x) for x in aucs_seed],
            f1=[float(np.mean([x[i] for x in f1_seed])) for i in range(3)],
            win_macro_f1=float(np.mean(mf1_seed)),
            between_video_var_share=float(np.mean(bv_seed)))
        per_video_auc[arm] = {v: float(np.mean([pv[i] for pv in pv_seed]))
                              for i, v in enumerate(tr_ids) if i in pv_seed[0]}
        score_dump[arm] = np.mean(np.stack(P_seed, 0), axis=0)
        r = results[arm]
        print(f"  {arm:8s} w={r['width']:5d}  wv-AUC={r['wv_auc']:.4f} (sd {r['wv_auc_sd']:.4f})  "
              f"F1@.3/.5/.7 = {100*r['f1'][0]:.1f}/{100*r['f1'][1]:.1f}/{100*r['f1'][2]:.1f}  "
              f"winMacroF1={100*r['win_macro_f1']:.1f}  betwVar={r['between_video_var_share']:.3f}",
              flush=True)

    common = sorted(set.intersection(*[set(per_video_auc[a]) for a in ARMS]))
    assert len(common) == 193, f"common OOF video set = {len(common)}, expected 193"
    print(f"\n[endpoint] common OOF video set for contrasts n={len(common)}", flush=True)

    np.save(OUT / "oof_scores_ALL.npy", score_dump["ALL"])
    np.save(OUT / "oof_scores_AUD.npy", score_dump["AUD"])
    np.save(OUT / "oof_score_video_ids.npy", np.array(tr_ids))
    np.save(OUT / "oof_y_win.npy", y)
    np.save(OUT / "oof_fold_of.npy", fold_of)

    # contrasts
    boot = np.random.default_rng(BOOT_SEED)
    CONTRASTS = [
        ("AUD-ALL", "AUD", "ALL", "G0", (95.0,)),
        ("AUDCENT-ALL", "AUDCENT", "ALL", "P1a", (95.0,)),
        ("AUDCENT-AUD", "AUDCENT", "AUD", "P1b", (95.0,)),
        ("ALLCENT-ALL", "ALLCENT", "ALL", "P2", (95.0, 97.5)),
        ("ALLCENT-AUD", "ALLCENT", "AUD", "P2", (95.0, 97.5)),
        ("AUDVIS0-ALL", "AUDVIS0", "ALL", "P2", (95.0, 97.5)),
        ("AUDVIS0-AUD", "AUDVIS0", "AUD", "P2", (95.0, 97.5)),
        ("VIS-ALL", "VIS", "ALL", "desc", (95.0,)),
        ("TXT-ALL", "TXT", "ALL", "desc", (95.0,)),
        ("AUD-VIS", "AUD", "VIS", "desc", (95.0,)),
        ("AUD-TXT", "AUD", "TXT", "desc", (95.0,)),
    ]
    print(f"\ncontrasts on video-macro wv-AUC (n={len(common)}), video-clustered paired bootstrap "
          f"{NBOOT}x seed {BOOT_SEED}, delta={DELTA}", flush=True)
    contrasts = {}
    for name, hi_a, lo_a, tag, levels in CONTRASTS:
        d = np.array([per_video_auc[hi_a][v] - per_video_auc[lo_a][v] for v in common])
        ci = boot_ci(d, boot, levels)
        rec = dict(delta=float(d.mean()), tag=tag, **ci)
        c95 = rec["ci95"]
        rec["pass95"] = bool(d.mean() >= DELTA and c95[0] > 0)
        if "ci97.5" in rec:
            c975 = rec["ci97.5"]
            rec["pass97.5"] = bool(d.mean() >= DELTA and c975[0] > 0)
        contrasts[name] = rec
        extra = ("  CI97.5 [%+.4f, %+.4f]" % tuple(rec["ci97.5"])) if "ci97.5" in rec else ""
        print(f"  [{tag:4s}] {name:14s} D = {rec['delta']:+.4f}  "
              f"CI95 [{c95[0]:+.4f}, {c95[1]:+.4f}]{extra}  "
              f"{'PASS' if rec['pass95'] else 'no'}", flush=True)

    g0 = contrasts["AUD-ALL"]
    g0_refuted = bool(g0["ci95"][1] < DELTA)
    p1 = bool(contrasts["AUDCENT-ALL"]["pass95"] and contrasts["AUDCENT-AUD"]["pass95"])
    p2_allcent = bool(contrasts["ALLCENT-ALL"].get("pass97.5") and
                      contrasts["ALLCENT-AUD"].get("pass97.5"))
    p2_audvis0 = bool(contrasts["AUDVIS0-ALL"].get("pass97.5") and
                      contrasts["AUDVIS0-AUD"].get("pass97.5"))
    kill = not (p1 or p2_allcent or p2_audvis0)
    print(f"\n[G0] premise refuted under matched protocol (CI95 upper < +0.010): {g0_refuted}",
          flush=True)
    print(f"[P1] AUDCENT GO: {p1}", flush=True)
    print(f"[P2] ALLCENT conditional GO: {p2_allcent}   AUDVIS0 conditional GO: {p2_audvis0}",
          flush=True)
    print(f"[KILL] temporal-informativeness / within-video nuisance-suppression family KILLED: "
          f"{kill}", flush=True)

    # ------------------------------------------------------------- part 2
    print("\n=== PART 2: R15-FS fixed-score falsification panel (arm ALL OOF seed-averaged "
          "scores, no fitting) ===", flush=True)
    S = score_dump["ALL"]
    base_pv_frozen = wv_auc_per_video(S, y)
    base_pv = wv_auc_per_video_ties(S, y)
    agree = max(abs(base_pv[j] - base_pv_frozen[j]) for j in base_pv)
    print(f"  [impl-note] midrank read-out vs frozen argsort read-out on the ALL scores: "
          f"max |diff| = {agree:.3e} (0 => no ties in the model scores; Part 2 uses the "
          f"midrank form because pooling creates exact ties by construction)", flush=True)
    base_auc = float(np.mean(list(base_pv.values())))
    print(f"  baseline (unpooled, s=0) wv-AUC = {base_auc:.4f}  n={len(base_pv)}", flush=True)
    fs = {"baseline_wv_auc": base_auc, "baseline_n": len(base_pv),
          "midrank_vs_frozen_maxdiff": float(agree)}

    def paired_from(pv_new, pv_ref):
        keys = sorted(set(pv_new) & set(pv_ref))
        d = np.array([pv_new[j] - pv_ref[j] for j in keys])
        return keys, d

    # --- FS-A: evidence/label offset
    print("  --- FS-A  evidence/label offset (score shifted by s, labels unchanged; "
          "convention: shifted[k] = score[k+s] paired with y[k], off-end windows dropped)",
          flush=True)
    fs["FS_A"] = {}
    for s in (-2, -1, 1, 2):
        sl, yl = [], []
        for j in range(len(S)):
            ks = np.arange(K)
            src = ks + s
            keep = (src >= 0) & (src < K)
            sl.append(S[j][src[keep]])
            yl.append(y[j][ks[keep]])
        pv = wv_auc_per_video_ragged(sl, yl)
        keys, d = paired_from(pv, base_pv)
        ci = boot_ci(d, boot, (95.0, 98.75))
        # per-fold sign of the point estimate
        sgn = np.sign(d.mean())
        nsame = 0
        for f in range(NFOLD):
            sel = [i for i, j in enumerate(keys) if fold_of[j] == f]
            if sel and np.sign(d[sel].mean()) == sgn:
                nsame += 1
        rec = dict(wv_auc=float(np.mean(list(pv.values()))), n=len(keys),
                   delta=float(d.mean()), folds_same_sign=int(nsame), **ci)
        rec["pass"] = bool(d.mean() >= DELTA and rec["ci98.75"][0] > 0 and nsame >= 4)
        fs["FS_A"][str(s)] = rec
        print(f"    s={s:+d}  wv-AUC={rec['wv_auc']:.4f}  D={rec['delta']:+.4f}  "
              f"CI95 [{rec['ci95'][0]:+.4f}, {rec['ci95'][1]:+.4f}]  "
              f"CI98.75 [{rec['ci98.75'][0]:+.4f}, {rec['ci98.75'][1]:+.4f}]  "
              f"folds_same_sign={nsame}/5  n={rec['n']}  {'PASS' if rec['pass'] else 'no'}",
              flush=True)
    d10 = any(v["pass"] for v in fs["FS_A"].values())
    fs["D10_survives"] = bool(d10)
    print(f"    [D10] evidence/label offset survives: {d10}", flush=True)

    # --- FS-B1: gold-segment region pooling (ORACLE DIAGNOSTIC)
    print("  --- FS-B1  ORACLE DIAGNOSTIC: pool score inside gold segments "
          "(window assigned by midpoint), broadcast back", flush=True)
    S_b1 = np.zeros_like(S)
    for j, v in enumerate(tr_ids):
        segs = gold_raw[v]
        starts = np.array([sg[0] for sg in segs], dtype=float)
        mids = (bnd[j][:, 0] + bnd[j][:, 1]) / 2.0
        part = np.clip(np.searchsorted(starts, mids, side="right") - 1, 0, len(segs) - 1)
        for p in np.unique(part):
            m = part == p
            S_b1[j][m] = S[j][m].mean()
    pv_b1 = wv_auc_per_video_ties(S_b1, y)
    keys, d = paired_from(pv_b1, base_pv)
    ci = boot_ci(d, boot, (95.0,))
    fs["FS_B1"] = dict(oracle=True, wv_auc=float(np.mean(list(pv_b1.values()))),
                       n=len(keys), delta=float(d.mean()), **ci)
    print(f"    FS-B1 wv-AUC={fs['FS_B1']['wv_auc']:.4f}  D={fs['FS_B1']['delta']:+.4f}  "
          f"CI95 [{fs['FS_B1']['ci95'][0]:+.4f}, {fs['FS_B1']['ci95'][1]:+.4f}]  "
          f"n={fs['FS_B1']['n']}", flush=True)

    # --- FS-B2: label-run region pooling (ORACLE DIAGNOSTIC, D4 GATE)
    print("  --- FS-B2  ORACLE DIAGNOSTIC (D4 GATE, bar D >= +0.015): pool score inside maximal "
          "runs of identical y_win, broadcast back", flush=True)
    S_b2 = np.zeros_like(S)
    for j in range(len(S)):
        run = np.concatenate([[0], np.cumsum(np.diff(y[j]) != 0)])
        for p in np.unique(run):
            m = run == p
            S_b2[j][m] = S[j][m].mean()
    pv_b2 = wv_auc_per_video_ties(S_b2, y)
    keys, d = paired_from(pv_b2, base_pv)
    ci = boot_ci(d, boot, (95.0,))
    fs["FS_B2"] = dict(oracle=True, wv_auc=float(np.mean(list(pv_b2.values()))),
                       n=len(keys), delta=float(d.mean()), gate=0.015, **ci)
    fs["FS_B2"]["pass"] = bool(d.mean() >= 0.015)
    fs["D4_survives"] = fs["FS_B2"]["pass"]
    print(f"    FS-B2 wv-AUC={fs['FS_B2']['wv_auc']:.4f}  D={fs['FS_B2']['delta']:+.4f}  "
          f"CI95 [{fs['FS_B2']['ci95'][0]:+.4f}, {fs['FS_B2']['ci95'][1]:+.4f}]  "
          f"n={fs['FS_B2']['n']}  [D4] survives: {fs['FS_B2']['pass']}", flush=True)

    # --- FS-C: label-free running mean, descriptive
    print("  --- FS-C  descriptive, no gate: centred running mean of width 3", flush=True)
    S_c = np.zeros_like(S)
    for j in range(len(S)):
        pad = np.concatenate([[S[j][0]], S[j], [S[j][-1]]])
        S_c[j] = (pad[:-2] + pad[1:-1] + pad[2:]) / 3.0
    pv_c = wv_auc_per_video_ties(S_c, y)
    keys, d = paired_from(pv_c, base_pv)
    ci = boot_ci(d, boot, (95.0,))
    fs["FS_C"] = dict(wv_auc=float(np.mean(list(pv_c.values()))), n=len(keys),
                      delta=float(d.mean()), **ci)
    print(f"    FS-C  wv-AUC={fs['FS_C']['wv_auc']:.4f}  D={fs['FS_C']['delta']:+.4f}  "
          f"CI95 [{fs['FS_C']['ci95'][0]:+.4f}, {fs['FS_C']['ci95'][1]:+.4f}]  "
          f"n={fs['FS_C']['n']}", flush=True)

    (OUT / "results.json").write_text(json.dumps(dict(
        arms=results, contrasts=contrasts, n_common=len(common),
        g0_premise_refuted=g0_refuted, p1_go=p1,
        p2_allcent_conditional_go=p2_allcent, p2_audvis0_conditional_go=p2_audvis0,
        kill=kill, fs=fs, seeds=SEEDS, fold_seed=FOLD_SEED, boot_seed=BOOT_SEED,
        nboot=NBOOT, delta=DELTA), indent=2))
    np.savez_compressed(OUT / "per_video_auc.npz",
                        **{a: np.array([per_video_auc[a][v] for v in common]) for a in ARMS},
                        video_ids=np.array(common))
    print(f"\nwrote {OUT/'results.json'}", flush=True)
    print(f"total wall = {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
