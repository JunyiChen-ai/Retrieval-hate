"""Reported call curves of a finished method (README section 6): fixed k calls and qmixE1 budgets, unified evaluator.

For each given trial: the batched EIG policy (cpolicy.run_batch with the checkpoint's own answer model; the copy
likelihood only with --copy) up to --max-calls on validation and test; test scores written for fixed k in --fixed and
for qmixE1 at --budgets (threshold calibrated on validation mean calls, no labels; per-video cap --cap) and scored by
the unified evaluator (hc.run_evaluator, scripts/reproduction_baselines/eval_baseline_scores.py). Writes
<out>/<corpus>/<trial name>/{scores_test_*.jsonl, metrics_test_*.json, curve.json} and <out>/<corpus>/summary.json
(3-trial means and standard deviations, mean calls of all / positive / negative videos).
"""
import argparse
from datetime import datetime
import json
from pathlib import Path
import socket
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from qtl import data as qdata, policy, stop_rules
from qtl.backbones import make_model
from qtl.checkpoint import load_trial
from qtl.copy_noise import bucket_of, estimate_pi_neg
import hier_evidence_common as hc
from sim_monotone import run_source, truncate                              # noqa: E402

RULE = "qmixE1"
METS = ("pooled_ap", "pooled_roc", "within_roc")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--trial-dirs", nargs="+", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--copy", action="store_true")
    ap.add_argument("--max-calls", type=int, default=32)
    ap.add_argument("--fixed", nargs="+", type=int, default=[0, 1, 2, 3, 4, 5, 6, 7, 8, 12, 16, 24, 32])
    ap.add_argument("--budgets", nargs="+", type=float, default=[1, 2, 3, 4, 5, 6, 7, 8])
    ap.add_argument("--cap", type=int, default=8)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out-root", default="runs/20261101_r5_completion/final_curves")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    test, val = list(ids["test"]), list(ids["val"])
    pos = np.array([labels[v] == 1 for v in test])
    qdata.configure_source("soft_both", 8)
    answers, answer_T = qdata.load_answers(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, val + test, ["bert"])
    root = (ROOT / a.out_root / a.tag / a.corpus).resolve()
    rows = {}
    for td in a.trial_dirs:
        trial = Path(td).resolve()
        summary, cfg, model, am, chain = load_trial(trial, a.device, answers, ids, model_factory=make_model)
        copy_fn = None
        if a.copy:
            pi, *_ = estimate_pi_neg(answers, labels, ids["train"], answer_T, am, cfg["categories"])
            copy_fn = lambda length, p=pi: float(p[bucket_of(length)])  # noqa: E731
        name = f'{trial.parents[2].name}_seed{summary["seed"]}_{trial.name}'
        od = root / name
        od.mkdir(parents=True, exist_ok=True)
        runs = run_source(model, store, ids, am, chain, answers, cfg, a.max_calls, a.device, copy_fn)

        def score(label, calls):
            path = od / f"scores_test_{label}.jsonl"
            hc.write_scores(str(path), {v: runs["test"][v]["scores"][calls[v]] for v in test})
            m = hc.run_evaluator(a.corpus, "test", str(path), str(od / f"metrics_test_{label}.json"))["results"]["score_av"]
            c = np.array([calls[v] for v in test])
            return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"],
                    "mean_calls": float(c.mean()), "mean_calls_positive": float(c[pos].mean()),
                    "mean_calls_negative": float(c[~pos].mean())}
        row = {"trial": str(trial), "seed": summary["seed"], "fixed": {}, "rule": {}}
        for k in a.fixed:
            row["fixed"][str(k)] = score(f"fixed{k}", {v: min(k, len(runs["test"][v]["eig"])) for v in test})
        tr = {s: {v: truncate(r, a.cap) for v, r in runs[s].items()} for s in runs}
        vals = {s: {v: stop_rules.rule_values(r, len(r["scores"][0])) for v, r in tr[s].items()} for s in tr}
        stop_rules.add_quantile_rules(vals)
        for b in a.budgets:
            thr, vm = policy.calibrate([vals["val"][v][RULE] for v in val], b)
            r_ = score(f"{RULE}_{b:g}", {v: policy.stop_calls(vals["test"][v][RULE], thr) for v in test})
            r_.update(threshold=thr, validation_mean_calls=vm)
            row["rule"]["%g" % b] = r_
        (od / "curve.json").write_text(json.dumps(row, indent=1) + "\n")
        rows[name] = row
        print(a.corpus, name, "fixed8", [round(row["fixed"]["8"][m], 4) for m in METS],
              "budget5", [round(row["rule"]["5"][m], 4) for m in METS] if "5" in row["rule"] else "", flush=True)
    R = list(rows.values())
    keys = METS + ("mean_calls", "mean_calls_positive", "mean_calls_negative")
    mean = {part: {k: {m: float(np.mean([r[part][k][m] for r in R])) for m in keys} for k in R[0][part]}
            for part in ("fixed", "rule")}
    sd = {part: {k: {m: float(np.std([r[part][k][m] for r in R])) for m in METS} for k in R[0][part]}
          for part in ("fixed", "rule")}
    out = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "corpus": a.corpus,
           "tag": a.tag, "copy": a.copy, "cap": a.cap, "trials": rows, "mean": mean, "sd": sd}
    (root / "summary.json").write_text(json.dumps(out, indent=1) + "\n")
    print("DONE", a.corpus, a.tag, flush=True)


if __name__ == "__main__":
    main()
