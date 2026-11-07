"""Review control E4 (README, round 1): does the qmixE1 stopping rule spend the budget the user sets?

GPT 6 Astra's round-1 point 4: policy.calibrate takes the smallest threshold whose validation mean calls is <= B, and
because the rule's values jump, the validation mean can stay well below B (HateMM seed 3407, validation-selected
trial, B = 7: 4.72 calls). For each revision-6 trial (policy_replay.py runs, validation and test, cap 8) and B in
--budgets:
  deterministic  the method as reported: calibrate's threshold; validation and test mean calls and test metrics;
  randomized     the two neighbouring thresholds c_lo (validation mean m_lo <= B) and c_hi (the next lower
                 threshold, m_hi > B); each video uses c_hi with probability q = (B - m_lo) / (m_hi - m_lo), else
                 c_lo, so the expected validation mean is exactly B; q depends only on validation call counts (no
                 labels, no scores). Test: expected mean calls (exact) and, over --draws random draws, the realized
                 mean calls and the metrics (mean and standard deviation over draws).
Metrics by the unified evaluator; one score per video (the posterior after its number of calls), nothing mixed.
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import pickle
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import policy, stop_rules                                      # noqa: E402
from qtl.replay import truncate                                         # noqa: E402
import hier_evidence_common as hc                                       # noqa: E402

RULE = "qmixE1"
METS = ("pooled_ap", "pooled_roc", "within_roc")


def score(corpus, od, label, sc):
    path = od / f"scores_test_{label}.jsonl"
    hc.write_scores(str(path), sc)
    m = hc.run_evaluator(corpus, "test", str(path), str(od / f"metrics_test_{label}.json"))["results"]["score_av"]
    return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--replay-dirs", nargs="+", required=True)
    ap.add_argument("--tag", required=True, help="test_selected / val_selected (which trials the replay dirs are)")
    ap.add_argument("--budgets", nargs="+", type=float, default=[1, 2, 3, 4, 5, 6, 7])
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--draws", type=int, default=5)
    ap.add_argument("--out-root", default="runs/20261108_astra_review/e4")
    a = ap.parse_args()
    _, ids, _, _ = hc.load_fixed_cohort(a.corpus)
    val, test = list(ids["val"]), list(ids["test"])
    rows = {}
    for rd in a.replay_dirs:
        rd = Path(rd)
        rp = pickle.load(open(rd / "runs_L0.pkl", "rb"))
        runs = {s: rp["runs"][s] for s in ("val", "test")}
        tr = {s: {v: truncate(r, a.cap) for v, r in runs[s].items()} for s in runs}
        vals = {s: {v: stop_rules.rule_values(r, len(r["scores"][0])) for v, r in tr[s].items()} for s in tr}
        stop_rules.add_quantile_rules(vals)
        rv = {s: {v: vals[s][v][RULE] for v in vals[s]} for s in vals}
        cands = np.unique(np.concatenate([np.asarray(rv["val"][v], dtype=np.float64) for v in val if len(rv["val"][v])]
                                         + [np.zeros(1)]))[::-1]
        vmean = np.array([np.mean([policy.stop_calls(rv["val"][v], c) for v in val]) for c in cands])
        od = (ROOT / a.out_root / a.tag / a.corpus / rd.name).resolve()
        od.mkdir(parents=True, exist_ok=True)
        row = {}
        for B in a.budgets:
            thr, vm = policy.calibrate([rv["val"][v] for v in val], B)
            calls = {v: policy.stop_calls(rv["test"][v], thr) for v in test}
            det = score(a.corpus, od, f"det_{B:g}", {v: tr["test"][v]["scores"][calls[v]] for v in test})
            det.update(threshold=thr, validation_mean_calls=vm, test_mean_calls=float(np.mean(list(calls.values()))))
            i_lo = int(np.where(vmean <= B + 1e-9)[0].max()) if (vmean <= B + 1e-9).any() else None
            res = {"deterministic": det}
            if i_lo is not None and i_lo + 1 < len(cands) and vmean[i_lo] < B - 1e-9:
                c_lo, c_hi, m_lo, m_hi = cands[i_lo], cands[i_lo + 1], vmean[i_lo], vmean[i_lo + 1]
                q = float((B - m_lo) / (m_hi - m_lo))
                lo = {v: policy.stop_calls(rv["test"][v], c_lo) for v in test}
                hi = {v: policy.stop_calls(rv["test"][v], c_hi) for v in test}
                draws = []
                for d in range(a.draws):
                    rng = np.random.default_rng(1000 * int(rp["seed"]) + 10 * int(B) + d)
                    pick = {v: (hi[v] if rng.random() < q else lo[v]) for v in test}
                    m = score(a.corpus, od, f"rand_{B:g}_d{d}", {v: tr["test"][v]["scores"][pick[v]] for v in test})
                    m["test_mean_calls"] = float(np.mean(list(pick.values())))
                    draws.append(m)
                res["randomized"] = {
                    "c_lo": float(c_lo), "c_hi": float(c_hi), "val_mean_lo": float(m_lo), "val_mean_hi": float(m_hi),
                    "q": q, "validation_expected_calls": float((1 - q) * m_lo + q * m_hi),
                    "test_expected_calls": float(np.mean([(1 - q) * lo[v] + q * hi[v] for v in test])),
                    "test_realized_calls_mean": float(np.mean([d_["test_mean_calls"] for d_ in draws])),
                    **{m_: float(np.mean([d_[m_] for d_ in draws])) for m_ in METS},
                    **{m_ + "_sd": float(np.std([d_[m_] for d_ in draws])) for m_ in METS}}
            row["%g" % B] = res
            print(a.corpus, a.tag, rd.name, "B", B, "val %.2f test %.2f" % (vm, det["test_mean_calls"]),
                  "rand val %.2f test %.2f" % (res["randomized"]["validation_expected_calls"],
                                               res["randomized"]["test_expected_calls"])
                  if "randomized" in res else "", flush=True)
        rows[rd.name] = row
    out = {"corpus": a.corpus, "tag": a.tag, "created": datetime.now().astimezone().isoformat(), "cap": a.cap,
           "trials": rows}
    (ROOT / a.out_root / a.tag / a.corpus / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print("DONE", a.corpus, a.tag, flush=True)


if __name__ == "__main__":
    main()
