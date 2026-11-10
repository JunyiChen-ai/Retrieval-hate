"""Reading of the first-level search (README section 10, R1-R4; results in section 12).

Reads <root>/<corpus>/seed<s>/trial<k>/summary.json of search_first_level.py (the evaluator's numbers as stored by
train.py) and prints / writes, per corpus, at the primary budget of 8 calls:
  - the test-selected trials (best trial per seed by test (AP + ROC) / 2 of the depth chosen on validation), 3-seed
    mean and seed standard deviation, the chosen d, c and the mean calls; the differences to the naive control (one
    rule: runs/20261109_tree_within/naive_questions/<corpus>.json, uniform8, alpha best on test, the baseline with
    the higher (AP + ROC) / 2) and to revision 6 (test-selected trials of runs/20261101_r6_length_table);
  - on the same trials depths["0"] (greedy) and depths["3"] (level first), R3;
  - the validation-selected trials and the mean over all trials;
  - the share of every chosen d per budget, how often the chosen d is the d with the best test value (R4);
  - every fixed d: its own best trial per seed, the all-trial mean, the paired difference chosen minus fixed over all
    trials and over trials 0-9 (hyper-parameters independent of the objective);
  - c over the trials; the other budgets on the test-selected trials.
    python experiments/20261110_prior_weight/summarize_first_level.py [--corpora hatemm hateclipseg dehate]
Output <root>/summary_<corpus>.json. No evaluation is recomputed here.
"""
import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
METS = ("pooled_ap", "pooled_roc", "within_roc")
R6 = {"hatemm": (.7017, .8890, .7827), "hateclipseg": (.6915, .6797, .5630), "dehate": (.2780, .7698, .6810)}


def vec(d):
    return np.array([d[k] for k in METS])


def obj(v):
    return (v[0] + v[1]) / 2


def g(x):
    return " / ".join("%.3f" % v for v in x)


def f(x):
    return " / ".join("%+.3f" % v for v in x)


def control(corpus):
    d = json.loads((ROOT / "runs/20261109_tree_within/naive_questions" / f"{corpus}.json").read_text())["methods"]
    best = None
    for name, m in d.items():
        u = m["uniform8"]
        a = u["best_alpha_on_test"]
        mean = vec(u["mean"][a])
        std = np.array([[s[a][k] for k in METS] for s in u["per_seed"]]).std(0)
        if best is None or obj(mean) > obj(best[1]):
            best = (name, mean, std, a, u["mean_calls"])
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/20261110_prior_weight/search_first_level")
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "hateclipseg", "dehate"])
    ap.add_argument("--budget", default="8")
    a = ap.parse_args()
    root = ROOT / a.root
    pb = a.budget
    for corpus in a.corpora:
        rows = {}
        for sd in sorted((root / corpus).glob("seed*")):
            if not (sd / "search.DONE").exists():
                print("==", corpus, sd.name, "not finished, skipped")
                continue
            for t in sorted(sd.glob("trial*"), key=lambda p: int(p.name[5:])):
                s = json.loads((t / "summary.json").read_text())
                F = s["fixed"][pb]
                rows[(sd.name, int(t.name[5:]))] = dict(
                    ch=vec(F["test"]), chv=vec(F["val"]), d=F["first_level"], calls=F["test_mean_calls"],
                    c=s["prior_weight"]["c"],
                    dep={int(k): (vec(v["val"]), vec(v["test"])) for k, v in F["depths"].items()},
                    fl={B: r["first_level"] for B, r in s["fixed"].items()},
                    fx={B: (vec(r["test"]), vec(r["depths"]["0"]["test"])) for B, r in s["fixed"].items()})
        seeds = sorted({k[0] for k in rows})
        if len(seeds) < 3:
            continue

        def best(key):
            return {s: max((k for k in rows if k[0] == s), key=key) for s in seeds}

        def three(sel, get):
            arr = np.array([get(rows[sel[s]]) for s in seeds])
            return arr.mean(0), arr.std(0)

        bt = best(lambda k: obj(rows[k]["ch"]))
        mean, std = three(bt, lambda r: r["ch"])
        cname, cmean, cstd, calpha, ccalls = control(corpus)
        r6 = np.array(R6[corpus])
        depths = sorted(rows[bt[seeds[0]]]["dep"])
        allc = np.array([r["ch"] for r in rows.values()])
        tb = [max(r["dep"], key=lambda d: obj(r["dep"][d][1])) for r in rows.values()]
        bv = best(lambda k: obj(rows[k]["chv"]))
        cs = np.array([r["c"] for r in rows.values()])
        out = {"n_trials": {s: sum(1 for k in rows if k[0] == s) for s in seeds}, "budget": int(pb),
               "test_selected": {"trials": {s: bt[s][1] for s in seeds}, "d": {s: rows[bt[s]]["d"] for s in seeds},
                                 "c": {s: rows[bt[s]]["c"] for s in seeds}, "mean": mean.tolist(), "std": std.tolist(),
                                 "mean_calls": float(np.mean([rows[bt[s]]["calls"] for s in seeds])),
                                 "per_seed": {s: rows[bt[s]]["ch"].tolist() for s in seeds}},
               "control": {"baseline": cname, "alpha": calpha, "mean": cmean.tolist(), "std": cstd.tolist(),
                           "mean_calls": ccalls, "lead": (mean - cmean).tolist()},
               "revision6": {"mean": r6.tolist(), "lead": (mean - r6).tolist()},
               "same_trials_depth": {str(d): three(bt, lambda r: r["dep"][d][1])[0].tolist() for d in depths},
               "validation_selected": {"trials": {s: bv[s][1] for s in seeds},
                                       "mean": three(bv, lambda r: r["ch"])[0].tolist()},
               "all_trials_mean": allc.mean(0).tolist(),
               "chosen_share": {B: {str(d): float(np.mean([r["fl"][B] == d for r in rows.values()]))
                                    for d in sorted({r["fl"][B] for r in rows.values()})}
                                for B in sorted(rows[bt[seeds[0]]]["fl"], key=int)},
               "chosen_is_test_best": float(np.mean([r["d"] == t for r, t in zip(rows.values(), tb)])),
               "test_best_share": {str(d): float(np.mean([t == d for t in tb])) for d in depths},
               "c": {"median": float(np.median(cs)), "min": float(cs.min()), "max": float(cs.max()),
                     "below_.05": int((cs < .05).sum()), "n": len(cs)}, "fixed_d": {}, "other_budgets": {}}
        print("== %s  trials %s" % (corpus, out["n_trials"]))
        print("  test-selected, chosen d, %s calls: %s +- %s | trials %s d %s c %s | calls %.2f" % (
            pb, g(mean), g(std), [bt[s][1] for s in seeds], [rows[bt[s]]["d"] for s in seeds],
            ["%.2f" % rows[bt[s]]["c"] for s in seeds], out["test_selected"]["mean_calls"]))
        print("  minus control %s (%s alpha %s: %s, std %s, %.2f calls) | minus revision 6 %s" % (
            f(mean - cmean), cname, calpha, g(cmean), g(cstd), ccalls, f(mean - r6)))
        for d in depths:
            m = np.array(out["same_trials_depth"][str(d)])
            print("  same trials, depths[%d]: %s | chosen minus it %s" % (d, g(m), f(mean - m)))
        print("  validation-selected %s (trials %s) | all-trial mean %s" % (
            g(out["validation_selected"]["mean"]), [bv[s][1] for s in seeds], g(allc.mean(0))))
        print("  chosen d share per budget: %s" % {B: {d: "%.0f%%" % (100 * v) for d, v in sh.items()}
                                                   for B, sh in out["chosen_share"].items()})
        print("  chosen = d with the best test value in %.0f%% of trials (shares %s)" % (
            100 * out["chosen_is_test_best"], {d: "%.0f%%" % (100 * v) for d, v in out["test_best_share"].items()}))
        k09 = [k for k in rows if k[1] < 10]
        for d in depths:
            bd = best(lambda k: obj(rows[k]["dep"][d][1]))
            m, sdv = three(bd, lambda r: r["dep"][d][1])
            al = np.array([r["dep"][d][1] for r in rows.values()])
            p09 = np.mean([rows[k]["ch"] - rows[k]["dep"][d][1] for k in k09], 0)
            out["fixed_d"][str(d)] = {"best_per_seed": m.tolist(), "std": sdv.tolist(),
                                      "trials": {s: bd[s][1] for s in seeds}, "all_trials_mean": al.mean(0).tolist(),
                                      "chosen_minus_all": (allc - al).mean(0).tolist(),
                                      "chosen_minus_trials0to9": p09.tolist()}
            print("  fixed d = %d: best per seed %s +- %s (trials %s) | all trials %s | chosen minus it: all %s, "
                  "trials 0-9 %s" % (d, g(m), g(sdv), [bd[s][1] for s in seeds], g(al.mean(0)),
                                     f((allc - al).mean(0)), f(p09)))
        print("  c: median %.2f, %.3f to %.2f, below .05 in %d of %d trials" % (
            np.median(cs), cs.min(), cs.max(), (cs < .05).sum(), len(cs)))
        for B in sorted(rows[bt[seeds[0]]]["fx"], key=int):
            m = three(bt, lambda r: r["fx"][B][0])[0]
            z = three(bt, lambda r: r["fx"][B][1])[0]
            out["other_budgets"][B] = {"chosen": m.tolist(), "d": {s: rows[bt[s]]["fl"][B] for s in seeds},
                                       "d0": z.tolist()}
            print("  budget %2s (test-selected trials): chosen %s (d %s) | d = 0 %s" % (
                B, g(m), [rows[bt[s]]["fl"][B] for s in seeds], g(z)))
        (root / f"summary_{corpus}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
