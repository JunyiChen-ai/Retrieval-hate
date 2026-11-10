"""Tree policies against the single level per budget, from the stage_a_all_trials.py trial files (README section 7).
Per corpus and budget B: mean over the trials that have every listed variant of test AP / ROC / within and of the
calls used; paired difference of each policy to revision 6 at the same budget (r6|none|eig) and to the single level
with the same weight (the 2^d equal windows with 2^d <= B).
    python experiments/20261110_prior_weight/budget_table.py [--weights loc] [--budgets 4 6 8 12 16]
Output <root>/budget_table_<weights>.json and a printed table.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METS = ("pooled_ap", "pooled_roc", "within_roc")
POLICIES = ("eig", "planfill_eig", "planfill_eer", "level")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/20261110_prior_weight/all_trials")
    ap.add_argument("--weights", default="loc")
    ap.add_argument("--budgets", nargs="+", type=int, default=[4, 6, 8, 12, 16])
    a = ap.parse_args()
    root = (ROOT / a.root).resolve()
    out = {}
    for cd in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = {}
        for f in sorted(cd.glob("seed*/trial*.json")):
            d = json.loads(f.read_text())
            for var, r in d["variants"].items():
                rows.setdefault(var, {})[(f.parent.name, f.stem)] = [r[m] for m in METS] + [r["mean_calls"]]
        out[cd.name] = {}
        print("==", cd.name)
        for B in a.budgets:
            suf = "" if B == 8 else "|%d" % B
            names = {p: "r6|%s|%s%s" % (a.weights, p, suf) for p in POLICIES}
            base = "r6|none|eig" + suf
            have = [n for n in list(names.values()) + [base] if n in rows]
            if base not in rows or names["level"] not in rows:
                continue
            keys = sorted(set.intersection(*[set(rows[n]) for n in have]))
            if not keys:
                continue
            b = np.array([rows[base][k] for k in keys])
            lev = np.array([rows[names["level"]][k] for k in keys])
            out[cd.name][str(B)] = {"n_trials": len(keys), "r6": dict(zip(METS, b.mean(0)[:3].tolist()))}
            print("  B %2d (n %d) revision 6 %s" % (B, len(keys), "/".join("%.3f" % x for x in b.mean(0)[:3])))
            for p, n in names.items():
                if n not in rows:
                    continue
                v = np.array([rows[n][k] for k in keys])
                out[cd.name][str(B)][p] = {
                    "mean": dict(zip(METS, v.mean(0)[:3].tolist())), "mean_calls": float(v.mean(0)[3]),
                    "vs_r6": dict(zip(METS, (v - b).mean(0)[:3].tolist())),
                    "vs_level": dict(zip(METS, (v - lev).mean(0)[:3].tolist())),
                    "vs_level_se": dict(zip(METS, ((v - lev).std(0)[:3] / np.sqrt(len(keys))).tolist()))}
                print("       %-13s %s calls %5.2f | vs r6 %s | vs level %s" % (
                    p, "/".join("%.3f" % x for x in v.mean(0)[:3]), v.mean(0)[3],
                    "/".join("%+.3f" % x for x in (v - b).mean(0)[:3]),
                    "/".join("%+.3f" % x for x in (v - lev).mean(0)[:3])))
    (root / f"budget_table_{a.weights}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
