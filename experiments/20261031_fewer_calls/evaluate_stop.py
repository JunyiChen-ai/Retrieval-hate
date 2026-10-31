"""Goal 1 (fewer VLM calls), evaluator outputs for the stopping rule (README section 2).

For each finished trial: the policy as in the complete method (EIG order, copy likelihood in EIG and posterior,
`cpolicy.run_batch`), up to --max-calls on validation and test. The maximum is the fixed budget the rule is compared
with (8): the rule only decides when to stop earlier, so no video gets more calls than in the fixed setting. The stopping statistic of each rule is computed per
(video, call) by `qtl.stop_rules`; its threshold c is calibrated on validation so that the validation mean number of
calls equals --target (policy.calibrate, no labels) and is then applied to test. Writes, per trial, test scores and
the unified evaluator's metrics for: each rule at the target, fixed 8 calls (the comparison point of goal 1) and
fixed <target> calls (equal-calls reference); plus calls per video. A summary over trials is written per corpus.
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import socket
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import cpolicy, data as qdata, policy
from qtl.checkpoint import load_trial
from qtl.copy_noise import bucket_of, estimate_pi_neg
from qtl.inside_outside import make_model
from qtl.stop_rules import rule_values, add_quantile_rules
import hier_evidence_common as hc


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--trial-dirs", nargs="+", required=True)
    ap.add_argument("--rules", nargs="+", default=["qmixE1", "qmixE3"])
    ap.add_argument("--target", type=float, default=5.0)
    ap.add_argument("--max-calls", type=int, default=8, help="maximum calls = the fixed budget compared with")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", required=True, help="output subdirectory, e.g. r5")
    ap.add_argument("--out-root", default="runs/20261031_fewer_calls")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    qdata.configure_source("soft_both", 8)
    answers, answer_T = qdata.load_answers(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, ids["val"] + ids["test"], ["bert"])
    out_root = (ROOT / a.out_root / a.corpus / a.tag).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    fixed_k = (a.max_calls, int(round(a.target)))
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(),
              "target": a.target, "max_calls": a.max_calls, "rules": a.rules, "trials": {}}
    for trial in a.trial_dirs:
        trial = Path(trial).resolve()
        summary, cfg, model, am, chain = load_trial(trial, a.device, answers, ids, model_factory=make_model)
        assert cfg["answer_source"] == "soft_both" and int(cfg.get("soft_levels", 8)) == 8
        pi, *_ = estimate_pi_neg(answers, labels, ids["train"], answer_T, am, cfg["categories"])
        copy_fn = lambda length, p=pi: float(p[bucket_of(length)])  # noqa: E731
        name = f'seed{summary["seed"]}_{trial.name}'
        od = out_root / name
        od.mkdir(parents=True, exist_ok=True)
        runs = {}
        for split in ("val", "test"):
            vids = sorted(ids[split], key=lambda v: store.T[v])
            runs[split] = {}
            for i in range(0, len(vids), int(cfg["eval_chunk"])):
                with torch.no_grad():
                    runs[split].update(cpolicy.run_batch(
                        model, store, vids[i:i + int(cfg["eval_chunk"])], am, chain, answers, cfg["categories"],
                        a.max_calls, a.device, order="eig", record_voi=True, copy_pi=copy_fn, copy_mode="both"))
        vals = {s: {v: rule_values(r, len(r["scores"][0])) for v, r in runs[s].items()} for s in runs}
        add_quantile_rules(vals)

        def evaluate(label, calls):
            scores = {v: runs["test"][v]["scores"][calls[v]] for v in ids["test"]}
            path = od / f"scores_test_{label}.jsonl"
            hc.write_scores(str(path), scores)
            m = hc.run_evaluator(a.corpus, "test", str(path), str(od / f"metrics_test_{label}.json"))
            m = m["results"]["score_av"]
            c = np.array([calls[v] for v in ids["test"]])
            pos = np.array([labels[v] == 1 for v in ids["test"]])
            return {"pooled_ap": m["pr_auc"], "pooled_roc": m["roc_auc"], "within_roc": m["per_video"]["macro_auc"],
                    "mean_calls": float(c.mean()), "mean_calls_positive": float(c[pos].mean()),
                    "mean_calls_negative": float(c[~pos].mean()),
                    "calls_quantiles": [float(x) for x in np.percentile(c, [0, 25, 50, 75, 100])],
                    "videos_at_max_calls": int((c >= a.max_calls).sum())}
        row = {"source": str(trial), "fixed": {}, "rules": {}}
        for k in fixed_k:
            row["fixed"][str(k)] = evaluate(f"fixed{k}", {v: min(k, len(runs["test"][v]["eig"])) for v in ids["test"]})
        for rule in a.rules:
            thr, vm = policy.calibrate([vals["val"][v][rule] for v in ids["val"]], a.target)
            calls = {v: policy.stop_calls(vals["test"][v][rule], thr) for v in ids["test"]}
            r_ = evaluate(f"{rule}_{a.target:g}", calls)
            r_.update(threshold=thr if math.isfinite(thr) else str(thr), validation_mean_calls=vm,
                      validation_videos_at_max_calls=int(sum(policy.stop_calls(vals["val"][v][rule], thr) >= a.max_calls
                                                             for v in ids["val"])))
            (od / f"calls_test_{rule}_{a.target:g}.json").write_text(json.dumps(calls) + "\n")
            row["rules"][rule] = r_
        report["trials"][name] = row
        f8 = row["fixed"][str(a.max_calls)]
        for rule, r_ in row["rules"].items():
            print(name, rule, "%.2f calls" % r_["mean_calls"],
                  " ".join("%+.4f" % (r_[m] - f8[m]) for m in ("pooled_ap", "pooled_roc", "within_roc")), flush=True)
        del model, am, chain
    mets = ("pooled_ap", "pooled_roc", "within_roc", "mean_calls")
    T = list(report["trials"].values())
    report["mean"] = {f"fixed{fixed_k[0]}": {m: float(np.mean([t["fixed"][str(fixed_k[0])][m] for t in T])) for m in mets},
                      f"fixed{fixed_k[1]}": {m: float(np.mean([t["fixed"][str(fixed_k[1])][m] for t in T])) for m in mets}}
    for rule in a.rules:
        report["mean"][rule] = {m: float(np.mean([t["rules"][rule][m] for t in T])) for m in mets}
        report["mean"][rule][f"minus_fixed{fixed_k[0]}"] = {m: report["mean"][rule][m] - report["mean"][f"fixed{fixed_k[0]}"][m]
                                                for m in mets[:3]}
    path = out_root / f"summary_{socket.gethostname()}.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(report["mean"], indent=1))
    print(path)


if __name__ == "__main__":
    main()
