"""R14 reconnaissance (descriptive, NOT a gate): does the score->interval decode carry the
leverage on HateClipSeg proposal-level F1@tIoU, on REAL model scores?

Train a per-window independent head (R11's A2_PERWIN) on the train split, score the val split,
decode to intervals, and compare naive vs tuned decode + a broadcast control.

Train + val only. The test split is never opened.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import uniform_filter1d

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "r11_seg"))
import run_pilot as RP  # noqa: E402

K = 30


def blocks_of(segs):
    out, cur = [], None
    for s, e, mh in segs:
        if sum(mh[1:]) > 0:
            cur = [s, e] if cur is None else [cur[0], e]
        else:
            if cur:
                out.append(tuple(cur))
                cur = None
    if cur:
        out.append(tuple(cur))
    return out


def match_f1(preds, golds, tiou):
    tp = npd = ng = 0
    for v in preds:
        P, Gd = preds[v], golds[v]
        npd += len(P); ng += len(Gd); used = set()
        for p in P:
            best, bi = -1, -1
            for i, q in enumerate(Gd):
                if i in used:
                    continue
                inter = max(0.0, min(p[1], q[1]) - max(p[0], q[0]))
                uni = max(p[1], q[1]) - min(p[0], q[0])
                iou = inter / uni if uni > 0 else 0.0
                if iou > best:
                    best, bi = iou, i
            if best >= tiou:
                tp += 1; used.add(bi)
    P_ = tp / max(npd, 1); R_ = tp / max(ng, 1)
    return 2 * P_ * R_ / max(P_ + R_, 1e-9)


def decode(scores, bounds, ids, thr, w, gap, minlen):
    preds = {}
    for j, v in enumerate(ids):
        s = scores[j].astype(float)
        if w > 1:
            s = uniform_filter1d(s, size=w, mode="nearest")
        wb = bounds[j]
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
            if m and iv[0] - m[-1][1] <= gap:
                m[-1][1] = iv[1]
            else:
                m.append(list(iv))
        preds[v] = [tuple(x) for x in m if x[1] - x[0] >= minlen]
    return preds


def wv_auc(scores, y):
    a = []
    for j in range(len(scores)):
        yy = y[j]
        if yy.min() == yy.max():
            continue
        s = scores[j]
        r = s.argsort().argsort() + 1
        p = int((yy == 1).sum()); n = int((yy == 0).sum())
        a.append((r[yy == 1].sum() - p * (p + 1) / 2) / (p * n))
    return float(np.mean(a))


def main():
    D = RP.load_all()
    g = np.load(ROOT / "idea-stage/r11_seg/out/grid_labels.npz", allow_pickle=True)
    bounds = g["bounds"]
    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    vids = D["vids"]; tr, va = D["tr"], D["va"]
    X = RP.zscore(D["ALL"], tr)
    y = D["y_win"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    probs = np.zeros((len(vids), K), dtype=np.float64)
    nseed = 5
    for seed in [3101, 3102, 3103, 3104, 3105]:
        torch.manual_seed(seed); np.random.seed(seed)
        m = RP.PerWin(X.shape[-1]).to(dev)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
        Xtr = torch.tensor(X[tr]).to(dev); ytr = torch.tensor(y[tr]).to(dev)
        Xva = torch.tensor(X[va]).to(dev)
        best, bstate = -1, None
        for ep in range(40):
            m.train(); opt.zero_grad()
            lo = m(Xtr)
            loss = F.cross_entropy(lo.reshape(-1, 2), ytr.reshape(-1))
            loss.backward(); opt.step()
            m.eval()
            with torch.no_grad():
                pv = torch.softmax(m(Xva), -1)[..., 1].cpu().numpy()
            auc = wv_auc(pv, y[va])
            if auc > best:
                best, bstate = auc, {k: v.clone() for k, v in m.state_dict().items()}
        m.load_state_dict(bstate); m.eval()
        with torch.no_grad():
            probs[va] += torch.softmax(m(torch.tensor(X[va]).to(dev)), -1)[..., 1].cpu().numpy() / nseed
            # video-level broadcast control: mean over windows of the same head
    ids = [vids[i] for i in va]
    golds = {v: blocks_of(gold[v]["segments"]) for v in ids}
    S = probs[va]
    print(f"\nval n={len(ids)}  window base rate={y[va].mean():.4f}")
    print(f"wv-AUC (val) = {wv_auc(S, y[va]):.4f}")

    # error autocorrelation of the residual (score - within-video mean) vs label residual
    ac = []
    for j in range(len(ids)):
        e = S[j] - S[j].mean() - (y[va][j] - y[va][j].mean())
        e = e - e.mean()
        if e.std() < 1e-9:
            continue
        ac.append(np.corrcoef(e[:-1], e[1:])[0, 1])
    print(f"lag-1 autocorrelation of per-window error: mean={np.nanmean(ac):.3f}")
    bv = np.var(S.mean(1)) / np.var(S)
    print(f"between-video share of score variance = {bv:.3f}")

    def sweep(sc, tag, grid):
        rows = []
        for (w, gap, ml) in grid:
            for thr in np.arange(0.02, 1.0, 0.02):
                p = decode(sc, bounds[va], ids, thr, w, gap, ml)
                rows.append((match_f1(p, golds, 0.5), match_f1(p, golds, 0.3),
                             match_f1(p, golds, 0.7), (w, gap, ml, round(float(thr), 2))))
        rows.sort(reverse=True)
        b = rows[0]
        print(f"  {tag:28s} F1@0.3={100*b[1]:5.1f}  F1@0.5={100*b[0]:5.1f}  F1@0.7={100*b[2]:5.1f}   cfg={b[3]}")
        return b

    print("\nproposal-level F1@tIoU on val (decoder swept; descriptive, no gate):")
    sweep(S, "naive (threshold only)", [(1, 0, 0)])
    grid = [(w, gp, ml) for w in (1, 3, 5, 7) for gp in (0, 5, 12, 25) for ml in (0, 5, 12)]
    sweep(S, "tuned (smooth+merge+minlen)", grid)
    Sn = np.stack([(s - s.mean()) / max(s.std(), 1e-8) for s in S])
    Sn = 1 / (1 + np.exp(-Sn))
    sweep(Sn, "per-video z-normalised", grid)
    Sb = np.stack([np.full(K, s.mean()) for s in S])
    sweep(Sb, "broadcast control", grid)
    # oracle window labels through the same decode = representation ceiling
    sweep(y[va].astype(float), "ORACLE window labels", grid)


if __name__ == "__main__":
    main()
