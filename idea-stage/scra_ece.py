"""Theory input for SCRA memo Proposition 3: L1 calibration error of the deployed head.

ECE_1 with equal-mass bins is a LOWER bound on E|eta - eta_hat| (Jensen), which is the quantity
that enters the certificate's slack. Val split only; no test labels.
"""
import json
import os
import sys

import numpy as np
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from r4_harness import load_split  # noqa: E402
from scra_shift_probe import CELLS, SEEDS, load_inputs, train_deployed  # noqa: E402


def ece_l1(y, p, nbin=10):
    order = np.argsort(p)
    bins = np.array_split(order, nbin)
    e = 0.0
    for b in bins:
        if len(b) == 0:
            continue
        e += (len(b) / len(y)) * abs(y[b].mean() - p[b].mean())
    return float(e)


if __name__ == "__main__":
    out = []
    for ds, mt, tag in CELLS:
        tr = load_split(ds, mt, "train")
        va = load_split(ds, mt, "val")
        te = load_inputs(ds, mt, "test")
        yv = va["y"].numpy()
        rows = []
        for s in SEEDS:
            p = train_deployed(tr, va, te, s)["val"]
            rows.append({"seed": s, "ece10": ece_l1(yv, p), "ece5": ece_l1(yv, p, 5),
                         "brier": float(np.mean((p - yv) ** 2)),
                         "auc": float(roc_auc_score(yv, p))})
        r = {"dataset": ds, "tag": tag, "rows": rows,
             "ece10_mean": float(np.mean([x["ece10"] for x in rows])),
             "ece5_mean": float(np.mean([x["ece5"] for x in rows])),
             "brier_mean": float(np.mean([x["brier"] for x in rows])),
             "pos_rate_val": float(yv.mean())}
        print(json.dumps(r), flush=True)
        out.append(r)
    json.dump(out, open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                     "scra_ece.json"), "w"), indent=2)
