"""PCD feasibility probe 3: UN-whitened clause scores + trained readout, 5 random draws.
TRAIN + VAL ONLY. Fair counterpart to probe 2 (which whitened). No test file is opened.
"""
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from transformers import CLIPModel, CLIPTokenizer

CLIP_ID = "openai/clip-vit-large-patch14-336"
DEV = "cuda" if torch.cuda.is_available() else "cpu"
from pcd_space_probe import PAIRS, load, enc_sentences  # noqa: E402


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

    def nz(X):
        n = np.linalg.norm(X, axis=1, keepdims=True)
        return np.nan_to_num(X / np.maximum(n, 1e-12))

    print(f"{'dataset':>13} {'arm':>26} {'val ROC':>9}")
    for ds in ["ImpliHateVid", "HateMM", "MHC", "MHC_zh"]:
        packs = {}
        for split in ["train", "dev_seen"]:
            _, img, txt, y = load(ds, split)
            with torch.no_grad():
                uv = nz(Wv(img.to(DEV)).cpu().numpy().astype(np.float64))
                ut = nz(Wt(txt.to(DEV)).cpu().numpy().astype(np.float64))
            packs[split] = (uv, ut, y)

        def feats(split, dv, de):
            uv, ut, y = packs[split]
            dv, de = nz(dv), nz(de)
            return np.concatenate([uv @ dv.T, uv @ de.T, ut @ dv.T, ut @ de.T], 1), y

        def run(dv, de):
            Xtr, ytr = feats("train", dv, de)
            Xva, yva = feats("dev_seen", dv, de)
            lr = LogisticRegression(max_iter=5000, C=1.0).fit(Xtr, ytr)
            return roc_auc_score(yva, lr.decision_function(Xva))

        print(f"{ds:>13} {'clause_2K (vio+exm)':>26} {run(vio, exm):>9.4f}")
        print(f"{ds:>13} {'clause_K (vio only, x2)':>26} {run(vio, vio):>9.4f}")
        print(f"{ds:>13} {'anchor_pair (K=1)':>26} {run(anc[None], saf[None]):>9.4f}")
        rs = [run(rng.normal(size=(K, 768)), rng.normal(size=(K, 768))) for _ in range(5)]
        print(f"{ds:>13} {'random_2K (5 draws)':>26} {np.mean(rs):>9.4f}"
              f"  [{min(rs):.4f},{max(rs):.4f}]")


if __name__ == "__main__":
    main()
