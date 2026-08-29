"""Round-9 zero-cost diagnostics (descriptive only; no candidate arm is scored here).

D1 - encoder-adaptation error population.
    Question: when the encoder is adapted (LoRA trained on the same train split) instead of
    frozen, does the head's error population CHANGE, or does the same set of items stay wrong?
    Arms are two ALREADY-DEPLOYED encoders from the contrast-line table, not candidates.
    Test labels are read for the comparison and this is declared as a disclosed diagnostic
    (same standing as IDEA_REPORT s10.6); nothing is selected or tuned on them.

D2 - train-side confident-error census.
    Question: how many train+val items does a frozen-feature head confidently disagree with,
    out-of-fold?  Descriptive counts only, no intervention, no test contact.

Usage:
    python diag.py --mode d1 --out d1.json
    python diag.py --mode d2 --out d2.json
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
from r4_harness import load_split, train_head, Head, macro_f1  # noqa: E402

FROZEN = "Qwen2.5-VL-7B-Instruct_HF"

D1_CELLS = [
    ("HateMM", FROZEN, "Qwen2.5-VL-7B-Instruct-LoRA_HF"),
    ("HateMM", FROZEN, "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF"),
    ("MHC", FROZEN, "Qwen2.5-VL-7B-Instruct-LoRA_HF"),
    ("MHC_zh", FROZEN, "Qwen2.5-VL-7B-Instruct-LoRA_HF"),
]
D2_SETS = ["HateMM", "MHC", "MHC_zh", "ImpliHateVid"]


def run_arm(dataset, tag, seeds):
    tr, va, te = (load_split(dataset, tag, s) for s in ("train", "val", "test"))
    probs, f1s = [], []
    for s in seeds:
        r = train_head(tr, va, te, s)
        probs.append(r["test_prob"])
        f1s.append(r["test_macro_f1"])
    return {
        "ids": te["ids"],
        "y": te["y"].numpy(),
        "prob": np.stack(probs),          # [S, N]
        "f1": np.array(f1s),
    }


def d1(seeds):
    out = []
    for ds, a_tag, b_tag in D1_CELLS:
        A = run_arm(ds, a_tag, seeds)
        B = run_arm(ds, b_tag, seeds)
        assert A["ids"] == B["ids"]
        y = A["y"]
        # per-item majority error across seeds
        eA = ((A["prob"] >= 0.5).astype(int) != y[None, :]).mean(0) > 0.5
        eB = ((B["prob"] >= 0.5).astype(int) != y[None, :]).mean(0) > 0.5
        inter = int((eA & eB).sum())
        union = int((eA | eB).sum())
        rec = {
            "dataset": ds, "frozen": a_tag, "adapted": b_tag, "n_test": int(len(y)),
            "f1_frozen_mean": float(A["f1"].mean()), "f1_frozen_std": float(A["f1"].std()),
            "f1_adapted_mean": float(B["f1"].mean()), "f1_adapted_std": float(B["f1"].std()),
            "n_err_frozen": int(eA.sum()), "n_err_adapted": int(eB.sum()),
            "n_err_both": inter, "n_err_either": union,
            "jaccard": float(inter / union) if union else None,
            "fixed_by_adapt": int((eA & ~eB).sum()),
            "broken_by_adapt": int((~eA & eB).sum()),
            "prob_corr": float(np.corrcoef(A["prob"].mean(0), B["prob"].mean(0))[0, 1]),
            # expected overlap if the two error sets were independent given their sizes
            "exp_overlap_indep": float(eA.sum() * eB.sum() / len(y)),
        }
        print(json.dumps(rec), flush=True)
        out.append(rec)
    return out


def d2(seeds, n_folds=5):
    out = []
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for ds in D2_SETS:
        tr, va, te = (load_split(ds, FROZEN, s) for s in ("train", "val", "test"))
        ids = list(tr["ids"]) + list(va["ids"])
        img = torch.cat([tr["img"], va["img"]])
        txt = torch.cat([tr["txt"], va["txt"]])
        y = torch.cat([tr["y"], va["y"]]).numpy()
        oof = np.zeros((len(seeds), len(y)))
        for si, s in enumerate(seeds):
            skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=s)
            for f, (itr, iva) in enumerate(skf.split(np.zeros(len(y)), y)):
                # inner: use the fold's own held-out part as BOTH val (epoch selection) and
                # scored part -> declared: epoch selection is optimistic for the census,
                # which only needs a confident-error count, not a clean generalisation number.
                pack_tr = {"img": img[itr], "txt": txt[itr], "y": torch.tensor(y[itr])}
                pack_va = {"img": img[iva], "txt": txt[iva], "y": torch.tensor(y[iva])}
                r = train_head(pack_tr, pack_va, pack_va, s, device=dev)
                oof[si, iva] = r["test_prob"]
        p = oof.mean(0)
        wrong = (p >= 0.5).astype(int) != y
        conf = np.where(y == 1, 1 - p, p)  # confidence in the WRONG direction
        rec = {
            "dataset": ds, "n": int(len(y)),
            "oof_macro_f1": float(f1_score(y, (p >= 0.5).astype(int), average="macro")),
            "oof_roc": float(roc_auc_score(y, p)),
            "n_wrong": int(wrong.sum()),
            "n_conf_wrong_0.9": int((wrong & (conf > 0.9)).sum()),
            "n_conf_wrong_0.8": int((wrong & (conf > 0.8)).sum()),
            "n_conf_wrong_0.7": int((wrong & (conf > 0.7)).sum()),
            "rate_conf_wrong_0.9": float((wrong & (conf > 0.9)).mean()),
            "pos_share_conf_wrong_0.9": float(y[(wrong & (conf > 0.9))].mean())
            if (wrong & (conf > 0.9)).sum() else None,
            "ids_conf_wrong_0.9": [ids[i] for i in np.where(wrong & (conf > 0.9))[0]],
        }
        print(json.dumps({k: v for k, v in rec.items() if k != "ids_conf_wrong_0.9"}),
              flush=True)
        out.append(rec)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["d1", "d2"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(300, 315)))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    res = d1(a.seeds) if a.mode == "d1" else d2(a.seeds[:5])
    json.dump({"mode": a.mode, "seeds": a.seeds, "rows": res},
              open(os.path.join(HERE, a.out), "w"), indent=2)
    print("WROTE", a.out)
