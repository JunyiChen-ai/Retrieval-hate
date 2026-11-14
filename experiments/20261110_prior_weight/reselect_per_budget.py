"""Call-budget curve of revision 7 with the trial chosen separately for every budget (user request 2026-11-15).

summarize_first_level.py reports the other budgets on the trials chosen at 8 calls, which favours 8 calls. Here, for
every budget B and every seed, the trial is chosen again among the 20 trials of that seed:
  - test:       best test (AP + ROC) / 2 at B calls (rule 7, same objective as the main table);
  - validation: best validation (AP + ROC) / 2 at B calls.
The depth d inside each trial is the one chosen on validation for B (first_level, as stored by train.py). The epoch of
each trial stays the one train.py chose (validation, fixed 8 calls); only one checkpoint per trial is stored.
Reads <root>/<corpus>/seed<s>/trial<k>/summary.json; recomputes no evaluation.
    python experiments/20261110_prior_weight/reselect_per_budget.py
Output <root>/reselect_per_budget.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METS = ("pooled_ap", "pooled_roc", "within_roc")


def vec(d):
    return np.array([d[k] for k in METS])


def obj(v):
    return (v[0] + v[1]) / 2


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/20261110_prior_weight/search_first_level")
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "hateclipseg", "dehate"])
    a = ap.parse_args()
    root = ROOT / a.root
    out = {}
    for corpus in a.corpora:
        rows = {}
        for sd in sorted((root / corpus).glob("seed*")):
            assert (sd / "search.DONE").exists(), sd
            for t in sorted(sd.glob("trial*"), key=lambda p: int(p.name[5:])):
                s = json.loads((t / "summary.json").read_text())
                rows[(sd.name, int(t.name[5:]))] = {B: (vec(r["val"]), vec(r["test"]), r["test_mean_calls"],
                                                        r["first_level"]) for B, r in s["fixed"].items()}
        seeds = sorted({k[0] for k in rows})
        budgets = sorted(next(iter(rows.values())), key=int)
        out[corpus] = {"n_trials": {s: sum(1 for k in rows if k[0] == s) for s in seeds}, "budgets": {}}
        print("== %s  trials %s" % (corpus, out[corpus]["n_trials"]))
        for B in budgets:
            rec = {}
            for name, idx in (("test", 1), ("validation", 0)):
                sel = {s: max((k for k in rows if k[0] == s), key=lambda k: obj(rows[k][B][idx])) for s in seeds}
                arr = np.array([rows[sel[s]][B][1] for s in seeds])
                rec[name] = {"trials": {s: sel[s][1] for s in seeds}, "d": {s: rows[sel[s]][B][3] for s in seeds},
                             "mean": arr.mean(0).tolist(), "std": arr.std(0).tolist(),
                             "mean_calls": float(np.mean([rows[sel[s]][B][2] for s in seeds]))}
            out[corpus]["budgets"][B] = rec
            t, v = rec["test"], rec["validation"]
            print("  B %2s calls %5.2f | test-sel %s +- %s trials %s d %s | val-sel %s trials %s" % (
                B, t["mean_calls"], " / ".join("%.3f" % x for x in t["mean"]),
                " / ".join("%.3f" % x for x in t["std"]), list(t["trials"].values()), list(t["d"].values()),
                " / ".join("%.3f" % x for x in v["mean"]), list(v["trials"].values())))
    (root / "reselect_per_budget.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
