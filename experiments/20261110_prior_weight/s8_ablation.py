"""Ablations of the weight c on the section-8 search (README section 8, "Fixed before the search"; results in
section 13): rule 14(g) reading 2.

Reads the section-8 search (<search>/<corpus>/seed<s>/trial<k>/summary.json) and the replays of the same checkpoints
(stage_a_all_trials.py --src <search> --out-root <replays>; "r6" is the checkpoint's own table):
  r6|none|eig          (i) c = 1 at inference (the checkpoint was still chosen under c)
  r6|saved_post|eig    (iii) questions chosen under weight 1, posterior under the stored c
  r6|saved_ask|eig     (iii) questions chosen under the stored c, posterior under weight 1
  r6|saved|level       the 8 windows of one level only, stored c (goal b)
  r6|saved|first3      the 8 windows of depth 3 first, then greedy, stored c
  r6|saved|eig         the candidate again (reproduction of the search's numbers)
  r6|saved|eig@raw     the candidate on unrounded test scores (diagnostic of the 6-decimal score files)
  r6|none|eig@val, r6|saved|eig@val   validation replays (reproduction of the validation numbers)
At 8 calls, per corpus: the candidate minus each variant on the test-selected trials of the search, with each
variant's own best trial per seed, paired over all trials and over trials 0-9; (ii) revision 6's own search is
taken from summarize_first_level.R6.
    python experiments/20261110_prior_weight/s8_ablation.py [--corpora hatemm hateclipseg dehate]
Output <replays>/s8_ablation_<corpus>.json. No evaluation is recomputed here.
"""
import argparse
import json

import numpy as np

from summarize_first_level import R6, ROOT, f, g, vec

VARS = {"none": ("r6|none|eig", "(i) c = 1 on the same checkpoints"),
        "post": ("r6|saved_post|eig", "(iii) revision-6 questions, weighted posterior"),
        "ask": ("r6|saved_ask|eig", "(iii) weighted questions, revision-6 posterior"),
        "level": ("r6|saved|level", "single level only, stored c"),
        "first3": ("r6|saved|first3", "depth-3 windows first, stored c"),
        "raw": ("r6|saved|eig@raw", "candidate, unrounded test scores")}


def obj(v):
    return v[0] + v[1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--search", default="runs/20261110_prior_weight/search")
    ap.add_argument("--replays", default="runs/20261110_prior_weight/s8_trials")
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
                rp = replays / corpus / sd.name / f"{t.name}.json"
                V = json.loads(rp.read_text())["variants"] if rp.exists() else {}
                need = [n for n, _ in VARS.values()] + ["r6|saved|eig", "r6|none|eig@val", "r6|saved|eig@val"]
                if any(n not in V for n in need):
                    missing += 1
                    continue
                r = dict(ch=vec(s["fixed"]["8"]["test"]), chv=vec(s["fixed"]["8"]["val"]), c=s["prior_weight"]["c"],
                         saved=vec(V["r6|saved|eig"]), savedv=vec(V["r6|saved|eig@val"]))
                r.update({k: vec(V[n]) for k, (n, _) in VARS.items()})
                rows[(sd.name, int(t.name[5:]))] = r
        seeds = sorted({k[0] for k in rows})
        if missing or len(seeds) < 3:
            print("==", corpus, "skipped: replays missing for %d trials, %d seeds complete" % (missing, len(seeds)))
            continue
        keys = list(rows)
        R = [rows[k] for k in keys]
        k09 = [k for k in keys if k[1] < 10]

        def best(name):
            return {s: max((k for k in keys if k[0] == s), key=lambda k: obj(rows[k][name])) for s in seeds}

        def mean(sel, name):
            return np.array([rows[sel[s]][name] for s in seeds]).mean(0)

        bt = best("ch")
        cand = mean(bt, "ch")
        rep_test = max(float(np.abs(r["saved"] - r["ch"]).max()) for r in R)
        rep_val = max(float(np.abs(r["savedv"] - r["chv"]).max()) for r in R)
        out = {"n_trials": len(keys), "budget": 8,
               "reproduction": {"max_abs_diff_test": rep_test, "max_abs_diff_val": rep_val},
               "candidate": {"test_selected": cand.tolist(), "trials": {s: bt[s][1] for s in seeds},
                             "c": {s: rows[bt[s]]["c"] for s in seeds},
                             "all_trials": np.mean([r["ch"] for r in R], 0).tolist()},
               "revision6": {"mean": list(R6[corpus]), "candidate_minus": (cand - np.array(R6[corpus])).tolist()}}
        print("== %s  %d trials | replay with the stored weight against the search: largest difference test %.4f, "
              "validation %.4f" % (corpus, len(keys), rep_test, rep_val))
        print("  candidate: test-selected %s (trials %s, c %s) | all trials %s" % (
            g(cand), [bt[s][1] for s in seeds], ["%.2f" % rows[bt[s]]["c"] for s in seeds],
            g(out["candidate"]["all_trials"])))
        print("  (ii) revision 6's own search %s | candidate minus it %s" % (g(R6[corpus]), f(cand - np.array(R6[corpus]))))
        for name, (_, label) in VARS.items():
            own = best(name)
            out[name] = {"label": label, "test_selected": mean(bt, name).tolist(), "own_best": mean(own, name).tolist(),
                         "own_best_trials": {s: own[s][1] for s in seeds},
                         "all_trials": np.mean([r[name] for r in R], 0).tolist(),
                         "candidate_minus": {
                             "test_selected": (cand - mean(bt, name)).tolist(),
                             "own_best": (cand - mean(own, name)).tolist(),
                             "all_trials": np.mean([r["ch"] - r[name] for r in R], 0).tolist(),
                             "trials0to9": np.mean([rows[k]["ch"] - rows[k][name] for k in k09], 0).tolist()}}
            print("  %-46s test-selected %s | own best %s (trials %s) | all trials %s" % (
                label, g(out[name]["test_selected"]), g(out[name]["own_best"]), [own[s][1] for s in seeds],
                g(out[name]["all_trials"])))
            print("  %-46s candidate minus it: test-selected %s | own best %s | all trials %s | trials 0-9 %s" % (
                "", *[f(out[name]["candidate_minus"][k]) for k in ("test_selected", "own_best", "all_trials",
                                                                  "trials0to9")]))
        (replays / f"s8_ablation_{corpus}.json").write_text(json.dumps(out, indent=1) + "\n")


if __name__ == "__main__":
    main()
