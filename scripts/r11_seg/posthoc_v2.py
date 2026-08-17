#!/usr/bin/env python
"""R11-SEG v2 POST-HOC descriptive diagnostic. Not in the freeze, not a gate.

Question: the coverage-budget decode (C1) lost 7.5 macro-F1 while the same decode with the
GOLD coverage gained 9.0. Is that the coverage ESTIMATOR failing, or the IDEA failing?

Method: swap the ridge budget for (i) the model's own causal running-mean probability
(a self-consistent, well-calibrated coverage estimate), (ii) a constant train-mean budget,
(iii) gold coverage corrupted with Gaussian noise at increasing sd, which traces out how
accurate a coverage predictor would have to be for the decode to break even against C0.

Writes idea-stage/r11_seg/out/posthoc_v2.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

import run_pilot as R
import run_v2 as V

OUT = Path("/home/jehc223/Retrieval-hate/idea-stage/r11_seg/out")
K = R.K


def main() -> None:
    D = R.load_all()
    tr, te = D["tr"], D["te"]
    y_ts, wot, y = D["y_ts"], D["win_of_ts"], D["y_win"]
    P = np.load(OUT / "v2_probs_ALL_A2_PERWIN.npy").mean(0)
    T = V.fit_transition(y, tr)
    cov = y.mean(1)
    rep = {}

    def sc(pred):
        return R.macro_f1_acc(R.ts_counts(y_ts, wot, pred, te).sum(0))

    rep["C0_unconstrained"] = sc((P >= 0.5).astype(int))
    selfb = np.cumsum(P, axis=1) / np.arange(1, K + 1)
    rep["self_budget_test_r"] = float(np.corrcoef(selfb[:, -1], cov[te])[0, 1])

    for nm, b in [("SELF", selfb),
                  ("CONST_train_mean", np.full((len(te), K), cov[tr].mean())),
                  ("GOLD_oracle", cov[te][:, None].repeat(K, 1))]:
        pred = np.stack([V.decode(P[j], "covbud", b[j], T) for j in range(len(te))])
        rep[f"C1_budget_{nm}"] = sc(pred)

    rng = np.random.default_rng(R.BOOT_SEED)
    sweep = {}
    for s in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30):
        vals, rs = [], []
        for _ in range(5):
            noisy = np.clip(cov[te] + rng.normal(0, s, len(te)), 0, 1)
            rs.append(1.0 if s == 0 else float(np.corrcoef(noisy, cov[te])[0, 1]))
            b = noisy[:, None].repeat(K, 1)
            vals.append(sc(np.stack([V.decode(P[j], "covbud", b[j], T)
                                     for j in range(len(te))]))[0])
        sweep[f"sd_{s:.2f}"] = dict(mean_r=float(np.mean(rs)), macro_f1=float(np.mean(vals)))
    rep["gold_plus_noise_sweep"] = sweep

    (OUT / "posthoc_v2.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
