"""Goal 1 diagnostic: question-selection orders on trained checkpoints (no retraining, CPU or GPU).

For each source trial (r5 or inside-outside checkpoints), runs the batched policy with copy likelihood (copy_neg,
both) under each selection order, up to --max-calls, on validation and test, and reports with the shared core
evaluator (screening; quoted rows are re-run through the unified evaluator):
  - fixed-k test curves;
  - stopping on the order's own selection value (EIG in bits, or the expected squared-error reduction for "eer")
    and on qmixSG_rt10, thresholds calibrated on validation mean calls (no labels) to several targets.
Rule-10 developmental evidence (test labels read; nothing trained or selected for the method).
"""
import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import pickle
import socket
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from qtl import cpolicy, data as qdata, policy
from qtl.answer_calib import fit_validation_answer_model
from qtl.checkpoint import load_trial
from qtl.copy_noise import bucket_of, estimate_pi_neg
from qtl.inside_outside import make_model
from qtl.stop_rules import rule_values, add_quantile_rules, add_relaxed_rules
import hier_evidence_common as hc

FIXED = (0, 1, 2, 3, 4, 5, 6, 8, 12, 16)
TARGETS = (3, 4, 5, 6, 8)


def metrics(scores, gt, hate):
    m = hc.frame_metrics(scores, gt, hate)
    return [m["pooled_ap"], m["pooled_roc"], m["within_roc"]]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", choices=("hatemm", "hateclipseg", "dehate"), required=True)
    ap.add_argument("--trial-dirs", nargs="+", required=True)
    ap.add_argument("--orders", nargs="+", default=["eig", "eer"])
    ap.add_argument("--max-calls", type=int, default=16)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", required=True, help="Output subdirectory name, e.g. r5 or io")
    ap.add_argument("--answer-model", choices=("train", "val3"), default="train",
                    help="train = the checkpoint's anchored model; val3 = three-state length model fitted on "
                         "validation span labels (src/qtl/answer_calib.py), used for selection and likelihood")
    ap.add_argument("--out-root", default="runs/20261031_two_goal_diagnostics/policy_lab")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    labels, ids, gt, _ = hc.load_fixed_cohort(a.corpus)
    hate = {s: {v for v in ids[s] if labels[v] == 1} for s in ("val", "test")}
    qdata.configure_source("soft_both", 8)
    answers, answer_T = qdata.load_answers(a.corpus, "soft_both")
    store = qdata.Store(a.corpus, ids["val"] + ids["test"], ["bert"])
    out_root = (ROOT / a.out_root / a.corpus / a.tag).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    report = {"host": socket.gethostname(), "created": datetime.now().astimezone().isoformat(), "trials": {}}
    for trial in a.trial_dirs:
        trial = Path(trial).resolve()
        summary, cfg, model, am, chain = load_trial(trial, a.device, answers, ids, model_factory=make_model)
        pi, *_ = estimate_pi_neg(answers, labels, ids["train"], answer_T, am, cfg["categories"])
        copy_fn = lambda length, p=pi: float(p[bucket_of(length)])  # noqa: E731
        name = f'seed{summary["seed"]}_{trial.name}'
        report["trials"][name] = {"source": str(trial), "orders": {}, "answer_model": a.answer_model}
        if a.answer_model == "val3":
            am, counts = fit_validation_answer_model(answers, labels, gt["val"], ids["val"], store.T, am.len_mu,
                                                     am.len_sd, cfg["categories"])
            am = am.to(a.device)
            report["trials"][name]["val3_state_counts"] = counts
        for order in a.orders:
            runs = {}
            for split in ("val", "test"):
                vids = sorted(ids[split], key=lambda v: store.T[v])
                runs[split] = {}
                for i in range(0, len(vids), int(cfg["eval_chunk"])):
                    with torch.no_grad():
                        runs[split].update(cpolicy.run_batch(
                            model, store, vids[i:i + int(cfg["eval_chunk"])], am, chain, answers, cfg["categories"],
                            a.max_calls, a.device, order=order, record_voi=True, copy_pi=copy_fn, copy_mode="both"))
            dest = out_root / f"{name}_{order}"
            dest.mkdir(parents=True, exist_ok=True)
            with (dest / "runs.pkl").open("wb") as fh:
                pickle.dump({"runs": runs, "T": {v: store.T[v] for v in ids["val"] + ids["test"]}}, fh)
            row = {"fixed": {}, "stop": {}}

            def summ(calls):
                c = np.array([calls[v] for v in ids["test"]])
                pos = np.array([labels[v] == 1 for v in ids["test"]])
                return {"metrics": metrics({v: runs["test"][v]["scores"][calls[v]] for v in ids["test"]},
                                           gt["test"], hate["test"]),
                        "mean_calls": float(c.mean()), "mean_calls_positive": float(c[pos].mean()),
                        "mean_calls_negative": float(c[~pos].mean())}
            for k in FIXED:
                row["fixed"][k] = summ({v: min(k, len(runs["test"][v]["eig"])) for v in ids["test"]})
            vals = {s: {v: rule_values(r, len(r["scores"][0])) for v, r in runs[s].items()} for s in runs}
            add_quantile_rules(vals)
            add_relaxed_rules(vals)
            for rule in ("eig", "qmixSG_rt10"):
                row["stop"][rule] = {}
                for t in TARGETS:
                    thr, vm = policy.calibrate([vals["val"][v][rule] for v in ids["val"]], float(t))
                    r_ = summ({v: policy.stop_calls(vals["test"][v][rule], thr) for v in ids["test"]})
                    r_.update(threshold=thr if math.isfinite(thr) else str(thr), validation_mean_calls=vm)
                    row["stop"][rule][t] = r_
            report["trials"][name]["orders"][order] = row
            print(name, order, "fixed4", row["fixed"][4]["metrics"], "fixed8", row["fixed"][8]["metrics"], flush=True)
        del model, am, chain
    path = out_root / f"summary_{socket.gethostname()}.json"
    path.write_text(json.dumps(report, indent=1) + "\n")
    print(path)


if __name__ == "__main__":
    main()
