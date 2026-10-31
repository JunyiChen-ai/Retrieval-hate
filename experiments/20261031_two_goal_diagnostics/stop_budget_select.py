"""Goal 1, option 2: can the stopping rule pick its own budget instead of a hand-set mean of 5 calls? (CPU, stored
policy_lab runs, no new VLM inference or tree passes.)

The qmixE1 rule (README of 20261031_fewer_calls) keeps one threshold c; until now c was set so that the validation
mean number of calls is 5 (a hand-set target). Here c is chosen on validation from the per-second labels (allowed
without gradient, user ruling 2026-09-29), with no target:
  - argmax: c maximising the validation objective J = (pooled AP + pooled ROC) / 2 (the rule-7 objective);
  - one_se: the one-standard-error rule (Breiman et al. 1984, CART; Hastie et al., ESL 7.10): the c with the fewest
    validation mean calls whose J is within one bootstrap standard error (validation videos resampled) of the best J;
  - paired_one_se: the same with the standard error of the paired difference J(c) - J(best) on the same resampled
    videos (both scored on every resample), scanned from the fewest calls up; the first c within one paired standard
    error of the best is taken.
Every c at which some validation video's number of calls changes is a candidate (thinned evenly to at most
--max-candidates, a computing resolution). Maximum 8 calls (the stored 16-call runs truncated, as rule_scan --cap 8).
Test: the chosen c applied per video, scored by the unified evaluator, against fixed 8 calls of the same trial.
Rule-10 developmental evidence.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import socket
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import policy, stop_rules
import hier_evidence_common as hc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from rule_scan import truncate                            # noqa: E402  (same truncation as the cap-8 scan)

RULE = "qmixE1"
MAX_CALLS = 8


def objective(scores, gt, hate):
    m = hc.frame_metrics(scores, gt, hate)
    return (m["pooled_ap"] + m["pooled_roc"]) / 2, m


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--root", default="runs/20261031_two_goal_diagnostics/policy_lab")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--max-candidates", type=int, default=400)
    ap.add_argument("--paired-bootstrap", type=int, default=200)
    ap.add_argument("--paired-candidates", type=int, default=100, help="thinning of the paired scan (resolution)")
    ap.add_argument("--out-root", default="runs/20261031_two_goal_diagnostics/stop_budget_select")
    a = ap.parse_args()
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    val, test = list(ids["val"]), list(ids["test"])
    hate_val = {v for v in val if labels[v] == 1}
    hate_test = {v for v in test if labels[v] == 1}
    gt_val = {v: np.asarray(gt["val"][v]) for v in val}
    out = (ROOT / a.out_root / a.corpus).resolve()
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "rule": RULE, "max_calls": MAX_CALLS, "bootstrap": a.bootstrap, "trials": {}}
    for d in sorted((ROOT / a.root / a.corpus / "r5").glob("*_eig")):
        with (d / "runs.pkl").open("rb") as fh:
            runs = pickle.load(fh)["runs"]
        runs = {s: {v: truncate(r, MAX_CALLS) for v, r in runs[s].items()} for s in runs}
        vals = {s: {v: stop_rules.rule_values(r, len(r["scores"][0])) for v, r in runs[s].items()} for s in runs}
        stop_rules.add_quantile_rules(vals)
        rv = {s: {v: np.asarray(vals[s][v][RULE], dtype=np.float64) for v in vals[s]} for s in vals}
        # calls at threshold c = number of running-minimum values >= c (policy.stop_calls)
        pm = {v: np.minimum.accumulate(rv["val"][v]) if len(rv["val"][v]) else rv["val"][v] for v in val}
        cands = np.unique(np.concatenate([x for x in pm.values() if len(x)] + [np.zeros(1)]))
        cands = np.append(cands, np.inf)                 # inf: no calls
        if len(cands) > a.max_candidates:
            cands = cands[np.unique(np.linspace(0, len(cands) - 1, a.max_candidates).round().astype(int))]
        curve = []
        for c in cands:
            calls = {v: int((pm[v] >= c).sum()) for v in val}
            J, m = objective({v: runs["val"][v]["scores"][calls[v]] for v in val}, gt_val, hate_val)
            curve.append({"c": float(c), "val_mean_calls": float(np.mean(list(calls.values()))), "J": float(J),
                          "pooled_ap": float(m["pooled_ap"]), "pooled_roc": float(m["pooled_roc"]),
                          "within_roc": float(m["within_roc"])})
        b = int(np.argmax([r["J"] for r in curve]))
        calls_b = {v: int((pm[v] >= curve[b]["c"]).sum()) for v in val}
        boot = []
        for _ in range(a.bootstrap):
            pick = rng.choice(len(val), len(val), replace=True)
            keys = [f"{val[i]}#{k}" for k, i in enumerate(pick)]
            sc = {k: runs["val"][val[i]]["scores"][calls_b[val[i]]] for k, i in zip(keys, pick)}
            g = {k: gt_val[val[i]] for k, i in zip(keys, pick)}
            h = {k for k, i in zip(keys, pick) if val[i] in hate_val}
            boot.append(objective(sc, g, h)[0])
        se = float(np.std(boot, ddof=1))
        ok = [i for i, r in enumerate(curve) if r["J"] >= curve[b]["J"] - se]
        one_se = min(ok, key=lambda i: (curve[i]["val_mean_calls"], -curve[i]["J"]))
        picks = [rng.choice(len(val), len(val), replace=True) for _ in range(a.paired_bootstrap)]

        def boot_J(i, pick):
            calls = {v: int((pm[v] >= curve[i]["c"]).sum()) for v in val}
            keys = [f"{val[j]}#{k}" for k, j in enumerate(pick)]
            return objective({k: runs["val"][val[j]]["scores"][calls[val[j]]] for k, j in zip(keys, pick)},
                             {k: gt_val[val[j]] for k, j in zip(keys, pick)},
                             {k for k, j in zip(keys, pick) if val[j] in hate_val})[0]
        jb = np.array([boot_J(b, p) for p in picks])
        sub = np.unique(np.linspace(0, len(curve) - 1, min(len(curve), a.paired_candidates)).round().astype(int))
        paired, paired_se = b, 0.0
        for i in sorted(sub, key=lambda i: (curve[i]["val_mean_calls"], -curve[i]["J"])):
            if curve[i]["val_mean_calls"] >= curve[b]["val_mean_calls"]:
                break
            diff = np.array([boot_J(i, p) for p in picks]) - jb
            s = float(np.std(diff, ddof=1))
            if curve[i]["J"] - curve[b]["J"] >= -s:
                paired, paired_se = int(i), s
                break
        name = d.name[:-len("_eig")]
        od = out / name
        od.mkdir(parents=True, exist_ok=True)

        def score_test(label, calls):
            path = od / f"scores_test_{label}.jsonl"
            hc.write_scores(str(path), {v: runs["test"][v]["scores"][calls[v]] for v in test})
            m = hc.run_evaluator(a.corpus, "test", str(path), str(od / f"metrics_test_{label}.json"))["results"]["score_av"]
            c = np.array([calls[v] for v in test])
            pos = np.array([labels[v] == 1 for v in test])
            return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"],
                    "mean_calls": float(c.mean()), "mean_calls_positive": float(c[pos].mean()),
                    "mean_calls_negative": float(c[~pos].mean())}
        row = {"val_best_J": curve[b]["J"], "val_J_se": se, "val_paired_se_at_choice": paired_se, "curve": curve,
               "fixed8": score_test("fixed8", {v: min(MAX_CALLS, len(runs["test"][v]["eig"])) for v in test})}
        for label, i in (("argmax", b), ("one_se", one_se), ("paired_one_se", paired)):
            c = curve[i]["c"]
            calls = {v: policy.stop_calls(list(rv["test"][v]), c) for v in test}
            row[label] = score_test(f"{RULE}_{label}", calls)
            row[label].update(threshold=c if np.isfinite(c) else "inf", val_mean_calls=curve[i]["val_mean_calls"],
                              val_J=curve[i]["J"])
            row[label]["minus_fixed8"] = {m: row[label][m] - row["fixed8"][m]
                                          for m in ("pooled_ap", "pooled_roc", "within_roc")}
        report["trials"][name] = row
        print(a.corpus, name, "SE %.4f" % se, " | ".join(
            "%s %.2f calls %s" % (k, row[k]["mean_calls"], " ".join("%+.4f" % x for x in row[k]["minus_fixed8"].values()))
            for k in ("argmax", "one_se", "paired_one_se")), flush=True)
    T = list(report["trials"].values())
    mets = ("pooled_ap", "pooled_roc", "within_roc", "mean_calls")
    report["mean"] = {k: {m: float(np.mean([t[k][m] for t in T])) for m in mets} for k in ("fixed8", "argmax", "one_se", "paired_one_se")}
    for k in ("argmax", "one_se", "paired_one_se"):
        report["mean"][k]["minus_fixed8"] = {m: report["mean"][k][m] - report["mean"]["fixed8"][m] for m in mets[:3]}
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["mean"], indent=1))
    print("DONE", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
