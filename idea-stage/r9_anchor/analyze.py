"""R9-1 ANCHOR-INT analyzer. Renders the frozen decision rule (R9_PILOT_FREEZE.md §5, §6).
Run exactly once on the complete grid."""
import json
import os

import numpy as np
from sklearn.metrics import f1_score

HERE = os.path.dirname(os.path.abspath(__file__))
B = 2000
RNG = np.random.default_rng(90210)


def mf1(y, p, idx=None):
    if idx is not None:
        y, p = y[idx], p[idx]
    return f1_score(y, (p >= 0.5).astype(int), average="macro")


def main():
    d = json.load(open(os.path.join(HERE, "results.json")))
    alphas = d["alphas"]
    out = {}
    passes = 0
    for ds, seeds in d["cells"].items():
        y = np.array(d["labels"][ds])
        P = {}   # alpha -> [S, N]
        V = {}   # alpha -> [S]
        for a in alphas:
            rows = sorted([r for r in d["rows"] if r["dataset"] == ds and r["alpha"] == a],
                          key=lambda r: r["seed"])
            assert [r["seed"] for r in rows] == sorted(seeds), (ds, a)
            P[a] = np.array([r["test_prob"] for r in rows])
            V[a] = np.array([r["val_macro_f1"] for r in rows])
        val_mean = {a: float(V[a].mean()) for a in alphas}
        test_mean = {a: float(np.mean([mf1(y, P[a][s]) for s in range(P[a].shape[0])]))
                     for a in alphas}
        # clause: alpha* by mean val macro-F1, ties -> smaller alpha
        best_val = max(val_mean.values())
        astar = min(a for a in alphas if val_mean[a] >= best_val - 1e-12)
        endpoint = 0.0 if test_mean[0.0] >= test_mean[1.0] else 1.0
        delta = test_mean[astar] - test_mean[endpoint]

        # paired bootstrap over test items, averaging the per-seed paired difference
        n = len(y)
        S = P[astar].shape[0]
        boots = np.empty(B)
        for b in range(B):
            idx = RNG.integers(0, n, n)
            if len(np.unique(y[idx])) < 2:
                boots[b] = np.nan
                continue
            boots[b] = np.mean([mf1(y, P[astar][s], idx) - mf1(y, P[endpoint][s], idx)
                                for s in range(S)])
        boots = boots[~np.isnan(boots)]
        lo, hi = np.percentile(boots, [2.5, 97.5])

        # clause 4: majority-error mechanism vs the adapted endpoint
        def err(a):
            return ((P[a] >= 0.5).astype(int) != y[None, :]).mean(0) > 0.5
        e0, e1, ea = err(0.0), err(1.0), err(astar)
        frozen_ok, frozen_bad = ~e0, e0

        def rates(e):
            brk = float(e[frozen_ok].mean()) if frozen_ok.sum() else 0.0
            rep = float((~e[frozen_bad]).mean()) if frozen_bad.sum() else 0.0
            return brk, rep
        brk_a, rep_a = rates(ea)
        brk_1, rep_1 = rates(e1)
        c1 = astar not in (0.0, 1.0)
        c2 = delta >= 0.005
        c3 = lo > 0
        c4 = (brk_1 > 0 and brk_a <= 0.75 * brk_1) and (rep_1 > 0 and rep_a >= 0.80 * rep_1)
        ok = bool(c1 and c2 and c3 and c4)
        passes += ok
        out[ds] = {
            "val_mean": val_mean, "test_mean": test_mean, "alpha_star": astar,
            "endpoint": endpoint, "delta": float(delta),
            "ci95": [float(lo), float(hi)],
            "break_rate_alpha": brk_a, "break_rate_adapted": brk_1,
            "repair_rate_alpha": rep_a, "repair_rate_adapted": rep_1,
            "clauses": {"interior": bool(c1), "size": bool(c2), "ci": bool(c3),
                        "mechanism": bool(c4)},
            "dataset_pass": ok,
        }
        print(f"\n== {ds}  (n_test={n}, seeds={S})")
        print("  val  :", {a: round(val_mean[a], 4) for a in alphas})
        print("  test :", {a: round(test_mean[a], 4) for a in alphas})
        print(f"  alpha*={astar} endpoint={endpoint} delta={delta:+.4f} "
              f"CI95=[{lo:+.4f},{hi:+.4f}]")
        print(f"  break rate alpha*={brk_a:.4f} vs adapted={brk_1:.4f} | "
              f"repair rate alpha*={rep_a:.4f} vs adapted={rep_1:.4f}")
        print(f"  clauses interior={c1} size={c2} ci={c3} mechanism={c4} -> "
              f"{'PASS' if ok else 'FAIL'}")
    verdict = "PASS" if passes >= 2 else "KILL"
    print(f"\nDATASETS PASSING: {passes}/3  ->  VERDICT {verdict}")
    json.dump({"per_dataset": out, "n_pass": passes, "verdict": verdict},
              open(os.path.join(HERE, "verdict.json"), "w"), indent=2)


if __name__ == "__main__":
    main()
