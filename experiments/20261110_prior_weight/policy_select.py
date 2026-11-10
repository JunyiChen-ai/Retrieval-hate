"""Question policy chosen on validation (developmental, README section 9; revision-6 checkpoints, nothing retrained).

For every trial of stage_a_all_trials.py and a set of candidate policies under one table and weight mode, the policy
with the largest validation (AP + ROC) / 2 at the budget is chosen (variants "<table>|<mode>|<policy>[|B]@val": the
replay on the validation videos; the per-second validation labels are used for this choice only, no gradient), and
the test metrics of the chosen policy are taken from the test replay of the same trial. Reported per corpus:
  - how often each candidate is chosen, and how often the choice is the candidate with the best test (AP + ROC) / 2;
  - the paired difference (all trials) of the chosen policy to revision 6, to every fixed candidate and to the single
    level;
  - the best trial per seed by test (AP + ROC) / 2 (rule 7's selection), 3-seed mean.
    python experiments/20261110_prior_weight/policy_select.py [--candidates eig planfill_eer level] [--budget 8]
Output <root>/policy_select_<tag>.json and a printed table. With --root on replays of revision-7 trials
(stage_a_all_trials.py --src ... --out-root ...) the column "r6" (variant "r6|none|eig") is the revision-7
checkpoint without its weight, not revision 6.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METS = ("pooled_ap", "pooled_roc", "within_roc")


def fmt(x):
    return "/".join("%+.3f" % v for v in x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/20261110_prior_weight/all_trials")
    ap.add_argument("--table", default="r6")
    ap.add_argument("--mode", default="loc")
    ap.add_argument("--candidates", nargs="+", default=["eig", "planfill_eer", "level"])
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--tag", default="b8")
    a = ap.parse_args()
    root = (ROOT / a.root).resolve()
    suf = "" if a.budget == 8 else "|%d" % a.budget
    names = {p: "%s|%s|%s%s" % (a.table, a.mode, p, suf) for p in a.candidates}
    out = {}
    for cd in sorted(p for p in root.iterdir() if p.is_dir()):
        rows = []                                # (seed, trial, {policy: test metrics}, {policy: val criterion}, r6)
        for sd in sorted(cd.glob("seed*")):
            for f in sorted(sd.glob("trial*.json"), key=lambda p: int(p.stem[5:])):
                d = json.loads(f.read_text())["variants"]
                if any(n not in d or n + "@val" not in d for n in names.values()) or "r6|none|eig" + suf not in d:
                    continue
                rows.append((sd.name, f.stem,
                             {p: [d[n][m] for m in METS] for p, n in names.items()},
                             {p: d[n + "@val"]["pooled_ap"] + d[n + "@val"]["pooled_roc"] for p, n in names.items()},
                             [d["r6|none|eig" + suf][m] for m in METS]))
        if not rows:
            continue
        chosen = [max(a.candidates, key=lambda p: r[3][p]) for r in rows]       # ties: the first candidate
        test_best = [max(a.candidates, key=lambda p: r[2][p][0] + r[2][p][1]) for r in rows]
        sel = np.array([r[2][c] for r, c in zip(rows, chosen)])
        r6 = np.array([r[4] for r in rows])
        fixed = {p: np.array([r[2][p] for r in rows]) for p in a.candidates}
        oracle = np.array([r[2][c] for r, c in zip(rows, test_best)])
        res = {"n_trials": len(rows),
               "chosen_share": {p: float(np.mean([c == p for c in chosen])) for p in a.candidates},
               "chosen_is_test_best": float(np.mean([c == t for c, t in zip(chosen, test_best)])),
               "test_best_share": {p: float(np.mean([t == p for t in test_best])) for p in a.candidates},
               "all_trials_mean": dict(zip(METS, sel.mean(0).tolist())),
               "paired_vs_r6": dict(zip(METS, (sel - r6).mean(0).tolist())),
               "paired_vs": {p: dict(zip(METS, (sel - fixed[p]).mean(0).tolist())) for p in a.candidates},
               "paired_vs_test_best_policy": dict(zip(METS, (sel - oracle).mean(0).tolist()))}
        best = {}
        for name, arr in [("chosen", sel), ("r6", r6)] + list(fixed.items()):
            per_seed = {}
            for i, r in enumerate(rows):
                if r[0] not in per_seed or arr[i][0] + arr[i][1] > per_seed[r[0]][1][0] + per_seed[r[0]][1][1]:
                    per_seed[r[0]] = (r[1] + ("" if name != "chosen" else ":" + chosen[i]), arr[i])
            m = np.array([v[1] for v in per_seed.values()])
            best[name] = {"trials": {s: v[0] for s, v in per_seed.items()},
                          "mean": dict(zip(METS, m.mean(0).tolist())), "std": dict(zip(METS, m.std(0).tolist()))}
        res["selected"] = best
        out[cd.name] = res
        print("== %s  n %d | chosen %s | choice = test-best policy in %.0f%% of trials (test-best shares %s)" % (
            cd.name, len(rows), " ".join("%s %.0f%%" % (p, 100 * s) for p, s in res["chosen_share"].items()),
            100 * res["chosen_is_test_best"],
            " ".join("%s %.0f%%" % (p, 100 * s) for p, s in res["test_best_share"].items())))
        print("  chosen, all trials %s | vs r6 %s | vs policy with the best test value %s" % (
            "/".join("%.3f" % v for v in sel.mean(0)), fmt((sel - r6).mean(0)), fmt((sel - oracle).mean(0))))
        for p in a.candidates:
            print("  chosen vs %-14s %s" % (p, fmt((sel - fixed[p]).mean(0))))
        for name, b in best.items():
            print("  best trial per seed  %-14s %s +- %s  %s" % (
                name, "/".join("%.3f" % b["mean"][m] for m in METS), "/".join("%.3f" % b["std"][m] for m in METS),
                ",".join(b["trials"][s][5:] for s in sorted(b["trials"]))))
    (root / ("policy_select_%s.json" % a.tag)).write_text(json.dumps(
        {"table": a.table, "mode": a.mode, "candidates": a.candidates, "budget": a.budget, "corpora": out},
        indent=1) + "\n")


if __name__ == "__main__":
    main()
