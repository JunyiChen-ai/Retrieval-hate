"""Goal 1, option 1 (user ruling 2026-11-01): the mean number of calls is a budget the user sets. Curve of the qmixE1
rule over budgets, against fixed k calls, on the stored r5 policy runs (CPU, no new VLM inference or tree passes).

For each r5 test-selected trial (EIG order, copy likelihood, the stored 16-call runs truncated to 8 = the maximum and
the fixed budget compared with): qmixE1's threshold calibrated on validation mean calls (no labels, policy.calibrate)
to each budget, applied to test; and fixed k calls for k = 0..8. Every row is scored by the unified evaluator.
Reported per budget: AP / ROC / within, test mean calls (all, positive, negative videos), and the differences to fixed
8 and to fixed k at the nearest whole number of calls. Rule-10 developmental evidence.
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
METS = ("pooled_ap", "pooled_roc", "within_roc")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--root", default="runs/20261031_two_goal_diagnostics/policy_lab")
    ap.add_argument("--budgets", nargs="+", type=float, default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--out-root", default="runs/20261031_two_goal_diagnostics/stop_budget_curve")
    a = ap.parse_args()
    labels, ids, _, _ = hc.load_fixed_cohort(a.corpus)
    val, test = list(ids["val"]), list(ids["test"])
    pos = np.array([labels[v] == 1 for v in test])
    out = (ROOT / a.out_root / a.corpus).resolve()
    out.mkdir(parents=True, exist_ok=True)
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
              "rule": RULE, "max_calls": MAX_CALLS, "budgets": a.budgets, "trials": {}}
    for d in sorted((ROOT / a.root / a.corpus / "r5").glob("*_eig")):
        with (d / "runs.pkl").open("rb") as fh:
            runs = pickle.load(fh)["runs"]
        runs = {s: {v: truncate(r, MAX_CALLS) for v, r in runs[s].items()} for s in runs}
        vals = {s: {v: stop_rules.rule_values(r, len(r["scores"][0])) for v, r in runs[s].items()} for s in runs}
        stop_rules.add_quantile_rules(vals)
        name = d.name[:-len("_eig")]
        od = out / name
        od.mkdir(parents=True, exist_ok=True)

        def score(label, calls):
            path = od / f"scores_test_{label}.jsonl"
            hc.write_scores(str(path), {v: runs["test"][v]["scores"][calls[v]] for v in test})
            m = hc.run_evaluator(a.corpus, "test", str(path), str(od / f"metrics_test_{label}.json"))["results"]["score_av"]
            c = np.array([calls[v] for v in test])
            return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"],
                    "mean_calls": float(c.mean()), "mean_calls_positive": float(c[pos].mean()),
                    "mean_calls_negative": float(c[~pos].mean())}
        row = {"fixed": {}, "rule": {}}
        for k in range(MAX_CALLS + 1):
            row["fixed"][str(k)] = score(f"fixed{k}", {v: min(k, len(runs["test"][v]["eig"])) for v in test})
        for b in a.budgets:
            thr, vm = policy.calibrate([vals["val"][v][RULE] for v in val], b)
            calls = {v: policy.stop_calls(vals["test"][v][RULE], thr) for v in test}
            r_ = score(f"{RULE}_{b:g}", calls)
            r_.update(threshold=thr, validation_mean_calls=vm)
            row["rule"][f"{b:g}"] = r_
        report["trials"][name] = row
        print(a.corpus, name, " | ".join("%s: %.2f calls %s" % (b, r_["mean_calls"], " ".join(
            "%+.4f" % (r_[m] - row["fixed"][str(MAX_CALLS)][m]) for m in METS)) for b, r_ in row["rule"].items()),
            flush=True)
    T = list(report["trials"].values())
    keys = METS + ("mean_calls", "mean_calls_positive", "mean_calls_negative")
    mean = {"fixed": {k: {m: float(np.mean([t["fixed"][k][m] for t in T])) for m in keys} for k in T[0]["fixed"]},
            "rule": {}}
    for b in T[0]["rule"]:
        r_ = {m: float(np.mean([t["rule"][b][m] for t in T])) for m in keys}
        f8 = mean["fixed"][str(MAX_CALLS)]
        fk = mean["fixed"][str(int(round(r_["mean_calls"])))]
        r_["minus_fixed8"] = {m: r_[m] - f8[m] for m in METS}
        r_["minus_fixed_nearest_k"] = {"k": int(round(r_["mean_calls"])), **{m: r_[m] - fk[m] for m in METS}}
        mean["rule"][b] = r_
    report["mean"] = mean
    (out / "summary.json").write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(mean["rule"], indent=1))
    print("DONE", out / "summary.json", flush=True)


if __name__ == "__main__":
    main()
