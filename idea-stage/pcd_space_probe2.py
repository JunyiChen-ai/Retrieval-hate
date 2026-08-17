"""PCD feasibility probe 2: whitened clause scores, TRAINED readout. TRAIN + VAL ONLY.

Tests the spec's actual construction at its cheapest: per-channel whitening, per-pair
violation/exemption projections, then a small trained readout on TRAIN, evaluated on VAL.
Comparators in the same frame: single anchor, random directions (matched count),
and a full logistic probe on the raw frozen features (information ceiling).
No test file is opened.
"""
import os
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import CLIPModel, CLIPTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CLIP_ID = "openai/clip-vit-large-patch14-336"
TAG = "openai_clip-vit-large-patch14-336_HF"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from pcd_space_probe import PAIRS, load, enc_sentences  # noqa: E402


def whiten_fit(X, lam=0.1):
    mu = X.mean(0, keepdims=True)
    Z = X - mu
    C = (Z.T @ Z) / max(len(Z) - 1, 1)
    C = (1 - lam) * C + lam * (np.trace(C) / C.shape[0]) * np.eye(C.shape[0])
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 1e-8)
    W = V @ np.diag(w ** -0.5) @ V.T
    return mu, W


def main():
    model = CLIPModel.from_pretrained(CLIP_ID).to(DEV).eval()
    tok = CLIPTokenizer.from_pretrained(CLIP_ID)
    Wv, Wt = model.visual_projection, model.text_projection
    vio = enc_sentences(model, tok, [p[0] for p in PAIRS]).numpy()
    exm = enc_sentences(model, tok, [p[1] for p in PAIRS]).numpy()
    anc = enc_sentences(model, tok, ["a hateful video",
                                     "a video that attacks a group of people"]).numpy().mean(0)
    saf = enc_sentences(model, tok, ["a harmless video",
                                     "an ordinary video about everyday life"]).numpy().mean(0)
    K = len(PAIRS)
    rng = np.random.default_rng(20260810)

    print(f"{'dataset':>13} {'arm':>22} {'val ROC':>9}")
    for ds in ["ImpliHateVid", "HateMM", "MHC", "MHC_zh"]:
        packs = {}
        for split in ["train", "dev_seen"]:
            _, img, txt, y = load(ds, split)
            with torch.no_grad():
                uv = Wv(img.to(DEV)).cpu().numpy().astype(np.float64)
                ut = Wt(txt.to(DEV)).cpu().numpy().astype(np.float64)
            packs[split] = (uv, ut, y)
        muv, Ww = whiten_fit(packs["train"][0])
        mut, Wx = whiten_fit(packs["train"][1])

        def feats(split, dirs_v, dirs_e):
            uv, ut, y = packs[split]
            zv, zt = (uv - muv) @ Ww, (ut - mut) @ Wx
            zv /= np.linalg.norm(zv, axis=1, keepdims=True)
            zt /= np.linalg.norm(zt, axis=1, keepdims=True)
            out = []
            for z, W_, mu_ in ((zv, Ww, muv), (zt, Wx, mut)):
                for D in (dirs_v, dirs_e):
                    Dw = D @ W_
                    Dw = Dw / np.linalg.norm(Dw, axis=1, keepdims=True)
                    out.append(z @ Dw.T)
            return np.concatenate(out, 1), y

        arms = {
            "clause_2K(vio+exm)": (vio, exm),
            "anchor_pair(K=1)": (anc[None], saf[None]),
            "random_2K": (rng.normal(size=(K, 768)), rng.normal(size=(K, 768))),
        }
        for nm, (dv, de) in arms.items():
            Xtr, ytr = feats("train", dv, de)
            Xva, yva = feats("dev_seen", dv, de)
            lr = LogisticRegression(max_iter=5000, C=1.0).fit(Xtr, ytr)
            print(f"{ds:>13} {nm:>22} {roc_auc_score(yva, lr.decision_function(Xva)):>9.4f}")
        # information ceiling: full frozen joint-space features
        uvtr, uttr, ytr = packs["train"]
        uvva, utva, yva = packs["dev_seen"]
        ftr = np.concatenate([uvtr, uttr], 1)
        fva = np.concatenate([uvva, utva], 1)
        lr = LogisticRegression(max_iter=5000, C=0.1).fit(ftr, ytr)
        print(f"{ds:>13} {'FULL joint-space LR':>22} "
              f"{roc_auc_score(yva, lr.decision_function(fva)):>9.4f}")


if __name__ == "__main__":
    main()
