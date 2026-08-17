"""R14 reconnaissance 2 (descriptive): decompose the proposal-level gap into a video-level
component and a within-video component. Train+val only.
"""
import json, sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "r11_seg"))
sys.path.insert(0, str(ROOT / "scripts" / "r14_loc"))
import run_pilot as RP  # noqa: E402
from recon_decode import blocks_of, match_f1, decode, wv_auc, K  # noqa: E402


def main():
    D = RP.load_all()
    g = np.load(ROOT / "idea-stage/r11_seg/out/grid_labels.npz", allow_pickle=True)
    bounds = g["bounds"]
    gold = json.loads((ROOT / "data/gt/HateClipSeg/gold_segments.json").read_text())
    vids = D["vids"]; tr, va = D["tr"], D["va"]
    y = D["y_win"]
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    def fit(chan_key):
        X = RP.zscore(D["ALL"] if chan_key == "ALL" else D["chans"][chan_key].astype(np.float32), tr)
        P = np.zeros((len(vids), K))
        for seed in [3101, 3102, 3103, 3104, 3105]:
            torch.manual_seed(seed); np.random.seed(seed)
            m = RP.PerWin(X.shape[-1]).to(dev)
            opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=1e-2)
            Xtr = torch.tensor(X[tr]).to(dev); ytr = torch.tensor(y[tr]).to(dev)
            Xva = torch.tensor(X[va]).to(dev)
            best, bstate = -1, None
            for ep in range(40):
                m.train(); opt.zero_grad()
                loss = F.cross_entropy(m(Xtr).reshape(-1, 2), ytr.reshape(-1))
                loss.backward(); opt.step()
                m.eval()
                with torch.no_grad():
                    pv = torch.softmax(m(Xva), -1)[..., 1].cpu().numpy()
                a = wv_auc(pv, y[va])
                if a > best:
                    best, bstate = a, {k: v.clone() for k, v in m.state_dict().items()}
            m.load_state_dict(bstate); m.eval()
            with torch.no_grad():
                P[va] += torch.softmax(m(Xva), -1)[..., 1].cpu().numpy() / 5.0
        return P[va]

    ids = [vids[i] for i in va]
    golds = {v: blocks_of(gold[v]["segments"]) for v in ids}
    yv = y[va].astype(float)
    grid = [(w, gp, ml) for w in (1, 3, 5, 7) for gp in (0, 5, 12, 25) for ml in (0, 5, 12)]

    def best_f1(sc, tag):
        rows = []
        for (w, gp, ml) in grid:
            for thr in np.arange(0.02, 1.0, 0.02):
                p = decode(sc, bounds[va], ids, thr, w, gp, ml)
                rows.append((match_f1(p, golds, 0.5), match_f1(p, golds, 0.3), match_f1(p, golds, 0.7)))
        rows.sort(reverse=True)
        b = rows[0]
        print(f"  {tag:34s} F1@0.3={100*b[1]:5.1f}  F1@0.5={100*b[0]:5.1f}  F1@0.7={100*b[2]:5.1f}")
        return b[0]

    S = fit("ALL")
    print(f"\nALL per-window head: wv-AUC={wv_auc(S, y[va]):.4f}")
    lvl = S.mean(1, keepdims=True); res = S - lvl
    glvl = yv.mean(1, keepdims=True); gres = yv - glvl

    print("\n2x2 substitution (video-level term x within-video residual), val:")
    best_f1(np.clip(lvl + res, 0, 1), "model level + model residual")
    best_f1(np.clip(glvl + res, 0, 1), "GOLD level  + model residual")
    best_f1(np.clip(lvl + gres, 0, 1), "model level + GOLD residual")
    best_f1(np.clip(glvl + gres, 0, 1), "GOLD level  + GOLD residual")

    print("\nsingle-channel per-window heads (within-video discrimination):")
    for c in ["V", "T", "O", "A"]:
        Sc = fit(c)
        a = wv_auc(Sc, y[va])
        print(f"  channel {c}: wv-AUC={a:.4f}", end="  ")
        best_f1(Sc, f"channel {c}")


if __name__ == "__main__":
    main()
