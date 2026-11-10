"""Rule 14(g) for the weight c on the first-level candidate's own configuration, and the level-only reference of R3
(README section 10; results in section 12).

Reads the first-level search (<search>/<corpus>/seed<s>/trial<k>/summary.json) and the replays of the same
checkpoints (stage_a_all_trials.py --src <search> --out-root <replays>: variants r6|none|<policy>[@val],
r6|saved|<policy>[@val] for the policies eig / first1 / first2 / first3, and r6|saved|level; "r6" is the
revision-7 checkpoint's own table, "none" = c = 1, "saved" = the c stored with the checkpoint). At 8 calls, per
corpus:
  - reproduction: the replay with the stored weight against the search's own numbers (summary "depths"): largest
    absolute difference over trials, depths and metrics on test and on validation, and in how many trials the replay
    chooses the same d;
  - c = 1 with the depth kept (the d the candidate chose) and c = 1 with the depth chosen again on validation under
    c = 1 (the rule of train.py: largest validation AP + ROC, ties to the smaller depth); the candidate minus each,
    on the test-selected trials of the search, paired over all trials and over trials 0-9, and with each variant's
    own best trial per seed;
  - the level-only policy with the stored weight (second reference of R3) against the chosen depth and against
    depths["3"].
    python experiments/20261110_prior_weight/c_ablation_first_level.py [--corpora hatemm hateclipseg dehate]
Output <replays>/c_ablation_<corpus>.json. No evaluation is recomputed here.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METS = ("pooled_ap", "pooled_roc", "within_roc")
POL = {0: "eig", 1: "first1", 2: "first2", 3: "first3"}


def vec(d):
    return np.array([d[k] for k in METS])


def obj(v):
    return v[0] + v[1]


def g(x):
    return " / ".join("%.3f" % v for v in x)


def f(x):
    return " / ".join("%+.3f" % v for v in x)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", default="runs/20261110_prior_weight/search_first_level")
    ap.add_argument("--replays", default="runs/20261110_prior_weight/r7_trials")
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "hateclipseg", "dehate"])
    a = ap.parse_args()
    search, replays = ROOT / a.search, ROOT / a.replays
    for corpus in a.corpora:
        rows, missing = {}, 0
        for sd in sorted((search / corpus).glob("seed*")):
            if not (sd / "search.DONE").exists():
                print("==", corpus, sd.name, "not finished, skipped")
                continue
            for t in sorted(sd.glob("trial*"), key=lambda p: int(p.name[5:])):
                s = json.loads((t / "summary.json").read_text())
                F = s["fixed"]["8"]
                rp = replays / corpus / sd.name / f"{t.name}.json"
                V = json.loads(rp.read_text())["variants"] if rp.exists() else {}
                names = ["r6|%s|%s%s" % (m, p, x) for m in ("none", "saved") for p in POL.values()
                         for x in ("", "@val")] + ["r6|saved|level"]
                if any(n not in V for n in names):
                    missing += 1
                    continue
                rows[(sd.name, int(t.name[5:]))] = dict(
                    ch=vec(F["test"]), d=F["first_level"], c=s["prior_weight"]["c"],
                    dep={d: (vec(F["depths"][str(d)]["val"]), vec(F["depths"][str(d)]["test"])) for d in POL},
                    none={d: (vec(V["r6|none|%s@val" % p]), vec(V["r6|none|%s" % p])) for d, p in POL.items()},
                    saved={d: (vec(V["r6|saved|%s@val" % p]), vec(V["r6|saved|%s" % p])) for d, p in POL.items()},
                    level=vec(V["r6|saved|level"]))
        seeds = sorted({k[0] for k in rows})
        if missing or len(seeds) < 3:
            print("==", corpus, "skipped: replays missing for %d trials, %d seeds complete" % (missing, len(seeds)))
            continue
        keys = list(rows)
        R = [rows[k] for k in keys]

        def pick(r, table):                      # the depth chosen on validation under this table's weight
            return max(sorted(POL), key=lambda d: obj(r[table][d][0]))

        rep_test = max(float(np.abs(r["saved"][d][1] - r["dep"][d][1]).max()) for r in R for d in POL)
        rep_val = max(float(np.abs(r["saved"][d][0] - r["dep"][d][0]).max()) for r in R for d in POL)
        same_d = int(sum(pick(r, "saved") == r["d"] for r in R))
        for r in R:
            r["kept"] = r["none"][r["d"]][1]
            r["d1"] = pick(r, "none")
            r["re"] = r["none"][r["d1"]][1]
            r["dep3"] = r["dep"][3][1]

        def best(name):
            return {s: max((k for k in keys if k[0] == s), key=lambda k: obj(rows[k][name])) for s in seeds}

        def mean(sel, name):
            return np.array([rows[sel[s]][name] for s in seeds]).mean(0)

        def paired(x, y, ks):
            return np.mean([rows[k][x] - rows[k][y] for k in ks], 0)

        bt = best("ch")
        k09 = [k for k in keys if k[1] < 10]
        out = {"n_trials": len(keys), "budget": 8,
               "reproduction": {"max_abs_diff_test": rep_test, "max_abs_diff_val": rep_val,
                                "same_chosen_d": same_d},
               "test_selected": {"trials": {s: bt[s][1] for s in seeds}, "d": {s: rows[bt[s]]["d"] for s in seeds},
                                 "c": {s: rows[bt[s]]["c"] for s in seeds},
                                 "d_under_c1": {s: rows[bt[s]]["d1"] for s in seeds}},
               "d_under_c1_share": {str(d): float(np.mean([r["d1"] == d for r in R])) for d in POL},
               "d_changed_share": float(np.mean([r["d1"] != r["d"] for r in R]))}
        print("== %s  %d trials | replay with the stored weight against the search: largest difference test %.4f, "
              "validation %.4f; same chosen d in %d of %d trials" % (corpus, len(keys), rep_test, rep_val, same_d,
                                                                     len(keys)))
        print("  test-selected trials %s | d %s | c %s | d chosen under c = 1 %s" % (
            [bt[s][1] for s in seeds], [rows[bt[s]]["d"] for s in seeds],
            ["%.2f" % rows[bt[s]]["c"] for s in seeds], [rows[bt[s]]["d1"] for s in seeds]))
        print("  d chosen under c = 1, share of trials: %s; differs from the candidate's d in %.0f%%" % (
            {d: "%.0f%%" % (100 * v) for d, v in out["d_under_c1_share"].items()}, 100 * out["d_changed_share"]))
        for name, label in (("ch", "candidate (stored c, chosen d)"), ("kept", "c = 1, d kept"),
                            ("re", "c = 1, d chosen again"), ("level", "level only, stored c"),
                            ("dep3", "depths[3], stored c")):
            own = best(name)
            out[name] = {"label": label, "test_selected": mean(bt, name).tolist(),
                         "own_best": mean(own, name).tolist(), "own_best_trials": {s: own[s][1] for s in seeds},
                         "all_trials": np.mean([r[name] for r in R], 0).tolist()}
            if name == "ch":
                print("  %-32s test-selected %s | all trials %s" % (label, g(out[name]["test_selected"]),
                                                                    g(out[name]["all_trials"])))
                continue
            out[name]["candidate_minus"] = {
                "test_selected": (mean(bt, "ch") - mean(bt, name)).tolist(),
                "own_best": (mean(bt, "ch") - mean(own, name)).tolist(),
                "all_trials": paired("ch", name, keys).tolist(), "trials0to9": paired("ch", name, k09).tolist()}
            print("  %-32s test-selected %s | own best %s (trials %s) | all trials %s" % (
                label, g(out[name]["test_selected"]), g(out[name]["own_best"]), [own[s][1] for s in seeds],
                g(out[name]["all_trials"])))
            print("  %-32s candidate minus it: test-selected %s | own best %s | all trials %s | trials 0-9 %s" % (
                "", *[f(out[name]["candidate_minus"][k]) for k in ("test_selected", "own_best", "all_trials",
                                                                  "trials0to9")]))
        (replays / f"c_ablation_{corpus}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
