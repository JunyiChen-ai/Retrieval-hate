"""R14-WVD post-hoc diagnostic (descriptive, NOT a gate; written and run after the frozen verdict).

Question: how much of the measured within-video discrimination ceiling at the 8 s grid is a
window-impurity artifact rather than a substrate limit? Re-fits the baseline cell A0_B0_C0 under
the frozen protocol and reads its out-of-fold scores back against window purity.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/home/jehc223/Retrieval-hate")
sys.path.insert(0, str(ROOT / "scripts" / "r14_loc"))
import run_wvd as W  # noqa: E402


def main():
    split = json.loads((ROOT / "data/gt/HateClipSeg/p11_split.json").read_text())
    tr_ids = split["train"]
    vids, y_all, bounds, chans, Tm, Om = W.build_features()
    idx = {v: i for i, v in enumerate(vids)}
    tr_pos = np.array([idx[v] for v in tr_ids])
    g = np.load(ROOT / "idea-stage/r11_seg/out/grid_labels.npz", allow_pickle=True)
    frac = g["frac_off"][tr_pos]
    y = y_all[tr_pos]

    rng = np.random.default_rng(W.FOLD_SEED)
    order = np.array(sorted(range(len(tr_pos)), key=lambda i: tr_ids[i]))
    perm = rng.permutation(len(order))
    fold_of = np.empty(len(order), dtype=int)
    fold_of[order[perm]] = np.arange(len(order)) % W.NFOLD

    X = W.assemble(chans, Tm, Om, "B0", "C0")[tr_pos]
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = np.zeros((len(tr_pos), W.K))
    for seed in W.SEEDS:
        for f in range(W.NFOLD):
            ho = np.where(fold_of == f)[0]; trn = np.where(fold_of != f)[0]
            p_ho, _ = W.fit_cell(X, y, trn, ho, "A0", seed + 7 * f, dev)
            P[ho] += p_ho / len(W.SEEDS)

    pure = (frac <= 0.02) | (frac >= 0.98)
    print(f"window purity: {100*pure.mean():.1f}% of train windows are pure "
          f"(frac_off <=0.02 or >=0.98); mixed {100*(1-pure.mean()):.1f}%")

    def wv(scores, labels, mask=None):
        a = []
        for j in range(len(scores)):
            s, yy = scores[j], labels[j]
            if mask is not None:
                m = mask[j]
                s, yy = s[m], yy[m]
            if len(yy) == 0 or yy.min() == yy.max():
                continue
            r = s.argsort().argsort() + 1
            p = int((yy == 1).sum()); n = int((yy == 0).sum())
            a.append((r[yy == 1].sum() - p * (p + 1) / 2) / (p * n))
        return float(np.mean(a)), len(a)

    a1, n1 = wv(P, y)
    a2, n2 = wv(P, y, pure)
    print(f"\nwv-AUC, all windows, binary majority label      = {a1:.4f}  (n={n1} videos)")
    print(f"wv-AUC, PURE windows only                       = {a2:.4f}  (n={n2} videos)")

    # rank correlation between the model score and the continuous gold coverage of each window
    rs = []
    for j in range(len(P)):
        if np.std(frac[j]) < 1e-9:
            continue
        rp = P[j].argsort().argsort(); rf = frac[j].argsort().argsort()
        rs.append(np.corrcoef(rp, rf)[0, 1])
    print(f"within-video Spearman(model score, gold window coverage) = {np.mean(rs):.4f} "
          f"(n={len(rs)} videos)")

    # how well does the SAME head do at the video level, for scale
    vs = P.mean(1); vy = (y.sum(1) > 0).astype(int)
    o = vs.argsort().argsort() + 1
    p = int(vy.sum()); n = len(vy) - p
    print(f"video-level AUC of the same head (mean-pooled)  = "
          f"{(o[vy == 1].sum() - p*(p+1)/2)/(p*n):.4f}  ({p} toxic / {n} clean videos)")

    # ceiling of a coverage-ranked oracle under the same read-out
    ao, _ = wv(frac, y)
    print(f"wv-AUC of an ORACLE ranking windows by gold coverage = {ao:.4f}")


if __name__ == "__main__":
    main()
