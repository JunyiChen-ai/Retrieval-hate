"""Reading of the section-8 search (revision 6 + the location weight c, greedy EIG policy; README section 8, Q1-Q3).

Reads <root>/<corpus>/seed<s>/trial<k>/summary.json of search.py (the evaluator's numbers as stored by train.py) and
prints / writes, per corpus, at the primary budget of 8 calls:
  - the test-selected trials (best trial per seed by test (AP + ROC) / 2), 3-seed mean and seed standard deviation,
    c and the mean calls; the differences to revision 6 (Q1, Q3) and to the naive control (same rule as in
    summarize_first_level.py);
  - the validation-selected trials (largest val_criterion per seed) and the mean over all trials;
  - trials 0-9 of every seed (same seeded hyper-parameters as revision 6's trials 0-9 and as the first-level
    search's): paired difference to revision 6's trials 0-9 (runs/20261101_r6_length_table), and the largest
    absolute difference to depths["0"] of the first-level search's trials 0-9 with the number of trials that have
    the same selected epoch (the two searches train the same networks there; the first-level search selects the
    epoch under the validated depth);
  - against the first-level search (validated first level): its test-selected mean and its fixed d = 0 own best;
  - c over the trials and its gain in the validation answers' negative log marginal likelihood;
  - the other budgets on the test-selected trials.
    python experiments/20261110_prior_weight/summarize_search.py [--corpora hatemm hateclipseg dehate]
Output <root>/summary_<corpus>.json. No evaluation is recomputed here.
"""
import argparse
import json

import numpy as np

from summarize_first_level import METS, R6, ROOT, control, f, g, obj, vec


def trials(root):
    out = {}
    for sd in sorted(root.glob("seed*")):
        if not (sd / "search.DONE").exists():
            print("==", root.name, sd.name, "not finished, skipped")
            continue
        for t in sorted(sd.glob("trial*"), key=lambda p: int(p.name[5:])):
            out[(sd.name, int(t.name[5:]))] = json.loads((t / "summary.json").read_text())
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="runs/20261110_prior_weight/search")
    ap.add_argument("--first-level", default="runs/20261110_prior_weight/search_first_level")
    ap.add_argument("--r6", default="runs/20261101_r6_length_table")
    ap.add_argument("--corpora", nargs="+", default=["hatemm", "hateclipseg", "dehate"])
    a = ap.parse_args()
    root = ROOT / a.root
    for corpus in a.corpora:
        S = trials(root / corpus)
        seeds = sorted({k[0] for k in S})
        if len(seeds) < 3:
            continue
        rows = {k: dict(te=vec(s["fixed"]["8"]["test"]), calls=s["fixed"]["8"]["test_mean_calls"],
                        crit=s["val_criterion"], c=s["prior_weight"]["c"], ep=s["best_epoch"],
                        gain=s["prior_weight"]["nll_all_before"] - s["prior_weight"]["nll_all_after"],
                        fx={B: vec(r["test"]) for B, r in s["fixed"].items()}) for k, s in S.items()}

        def best(key):
            return {s: max((k for k in rows if k[0] == s), key=key) for s in seeds}

        def three(sel, get=lambda r: r["te"]):
            arr = np.array([get(rows[sel[s]]) for s in seeds])
            return arr.mean(0), arr.std(0)

        bt, bv = best(lambda k: obj(rows[k]["te"])), best(lambda k: rows[k]["crit"])
        mean, std = three(bt)
        cname, cmean, cstd, calpha, ccalls = control(corpus)
        r6 = np.array(R6[corpus])
        allm = np.array([r["te"] for r in rows.values()]).mean(0)
        cs = np.array([r["c"] for r in rows.values()])
        gains = np.array([r["gain"] for r in rows.values()])
        out = {"n_trials": {s: sum(1 for k in rows if k[0] == s) for s in seeds}, "budget": 8,
               "test_selected": {"trials": {s: bt[s][1] for s in seeds}, "c": {s: rows[bt[s]]["c"] for s in seeds},
                                 "mean": mean.tolist(), "std": std.tolist(),
                                 "mean_calls": float(np.mean([rows[bt[s]]["calls"] for s in seeds])),
                                 "per_seed": {s: rows[bt[s]]["te"].tolist() for s in seeds}},
               "revision6": {"mean": r6.tolist(), "lead": (mean - r6).tolist()},
               "control": {"baseline": cname, "alpha": calpha, "mean": cmean.tolist(), "std": cstd.tolist(),
                           "mean_calls": ccalls, "lead": (mean - cmean).tolist()},
               "validation_selected": {"trials": {s: bv[s][1] for s in seeds}, "mean": three(bv)[0].tolist()},
               "all_trials_mean": allm.tolist(),
               "c": {"median": float(np.median(cs)), "min": float(cs.min()), "max": float(cs.max()),
                     "below_.05": int((cs < .05).sum()), "n": len(cs), "nll_gain_median": float(np.median(gains)),
                     "nll_gain_min": float(gains.min())}, "other_budgets": {}}
        print("== %s  trials %s" % (corpus, out["n_trials"]))
        print("  test-selected, 8 calls: %s +- %s | trials %s c %s | calls %.2f" % (
            g(mean), g(std), [bt[s][1] for s in seeds], ["%.2f" % rows[bt[s]]["c"] for s in seeds],
            out["test_selected"]["mean_calls"]))
        print("  minus revision 6 %s (Q1: AP, ROC not more than .005 below: %s; Q3: within above: %s)" % (
            f(mean - r6), bool((mean - r6)[:2].min() >= -.005), bool((mean - r6)[2] > 0)))
        print("  minus control %s (%s alpha %s: %s, std %s, %.2f calls)" % (
            f(mean - cmean), cname, calpha, g(cmean), g(cstd), ccalls))
        print("  validation-selected %s (trials %s) | all-trial mean %s" % (
            g(out["validation_selected"]["mean"]), [bv[s][1] for s in seeds], g(allm)))

        k09 = [k for k in rows if k[1] < 10]
        R = {k: json.loads(p.read_text()) for k in k09
             if (p := ROOT / a.r6 / corpus / k[0] / f"trial{k[1]}" / "summary.json").exists()}
        if len(R) == len(k09):
            same_hp = sum(abs(R[k]["cfg"]["lr"] - S[k]["cfg"]["lr"]) < 1e-12 for k in k09)
            d = np.mean([rows[k]["te"] - vec(R[k]["fixed"]["8"]["test"]) for k in k09], 0)
            out["trials0to9_minus_revision6"] = {"paired": d.tolist(), "same_lr": int(same_hp), "n": len(k09)}
            print("  trials 0-9 minus revision 6's trials 0-9 (paired, %d trials, same lr in %d): %s" % (
                len(k09), same_hp, f(d)))
        F = trials(ROOT / a.first_level / corpus) if (ROOT / a.first_level / corpus).exists() else {}
        if all(k in F for k in k09):
            diff = [float(np.abs(rows[k]["te"] - vec(F[k]["fixed"]["8"]["depths"]["0"]["test"])).max()) for k in k09]
            same_ep = sum(F[k]["best_epoch"] == rows[k]["ep"] for k in k09)
            out["trials0to9_vs_first_level_d0"] = {"max_abs_diff": max(diff), "same_epoch": int(same_ep),
                                                   "exact": int(sum(x < 5e-7 for x in diff)), "n": len(k09)}
            print("  trials 0-9 against the first-level search's depths[0]: same epoch in %d of %d, identical test "
                  "numbers in %d, largest difference %.4f" % (same_ep, len(k09), out[
                      "trials0to9_vs_first_level_d0"]["exact"], max(diff)))
        fl = root.parent / a.first_level.split("/")[-1] / f"summary_{corpus}.json"
        if fl.exists():
            fs = json.loads(fl.read_text())
            cand, d0 = np.array(fs["test_selected"]["mean"]), np.array(fs["fixed_d"]["0"]["best_per_seed"])
            out["first_level_search"] = {"chosen_d_mean": cand.tolist(), "minus_this": (cand - mean).tolist(),
                                         "fixed_d0_best_per_seed": d0.tolist()}
            print("  first-level search: validated d %s (minus this search %s) | its fixed d = 0, own best %s" % (
                g(cand), f(cand - mean), g(d0)))
        print("  c: median %.2f, %.3f to %.2f, below .05 in %d of %d trials | gain in the validation answers' "
              "negative log marginal likelihood: median %.3f, minimum %.3f" % (
                  np.median(cs), cs.min(), cs.max(), (cs < .05).sum(), len(cs), np.median(gains), gains.min()))
        for B in sorted(rows[bt[seeds[0]]]["fx"], key=int):
            m = three(bt, lambda r: r["fx"][B])[0]
            out["other_budgets"][B] = m.tolist()
            print("  budget %2s (test-selected trials): %s" % (B, g(m)))
        (root / f"summary_{corpus}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
