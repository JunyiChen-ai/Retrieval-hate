"""Fixed Optuna search for one (corpus, seed), rule 7 of RESEARCH_ITERATION_RULES.md.

    python experiments/20260904_evidence_guided_attention/search.py \
        --corpus hatemm --seed 234 --out-root runs/20260903_hier_evidence_mil

Each trial trains once (train.py), selects its checkpoint on the official
validation split, scores test, and reports the test pooled AP/ROC/within.
Optuna's objective is the scalar (test AP + test ROC) / 2. A trial whose
test within-video ROC is below the corpus floor is recorded as pruned (its
outputs stay on disk) and does not become the study's best.

Trial budget: after the first trial, 20 trials if it took <= 1 h, else 5.
The budget is written to budget.json and never changed afterwards; re-running
this script resumes the sqlite study and only runs the remaining trials.
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import subprocess
import sys
import time

import optuna

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
TRAIN = os.path.join(HERE, "train.py")

WITHIN_FLOOR = {"hatemm": 0.632, "hateclipseg": 0.524}   # rule 8


def sample(trial):
    # README section 4 (declared before any search, 2026-09-04): six scalars;
    # dropout (.2) and lamda_cof (.05) fixed at MACIL-SD's values, the EMA
    # hyperparameters are gone with the EMA.
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "max_seqlen": trial.suggest_categorical("max_seqlen", [150, 200, 300]),
        "lamda_cma": trial.suggest_float("lamda_cma", 0.5, 2.0),
        "prior_scale": trial.suggest_float("prior_scale", 0.5, 8.0, log=True),
        "w_fine": trial.suggest_float("w_fine", 0.0, 1.0),
        "lambda_block": trial.suggest_float("lambda_block", 0.05, 2.0, log=True),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--ablation", default="full")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args(argv)

    root = os.path.join(args.out_root, args.corpus, "seed%d" % args.seed)
    os.makedirs(root, exist_ok=True)
    with open(os.path.join(root, "search.pid"), "w") as fh:
        fh.write(str(os.getpid()))
    log = open(os.path.join(root, "search.log"), "a")

    def say(msg):
        print(msg, flush=True)
        log.write(msg + "\n")
        log.flush()

    say("host %s | corpus %s | seed %d | ablation %s | %s"
        % (socket.gethostname(), args.corpus, args.seed, args.ablation,
           time.strftime("%Y-%m-%d %H:%M:%S")))
    storage = "sqlite:///" + os.path.join(root, "optuna.db")
    study = optuna.create_study(
        study_name="%s_seed%d" % (args.corpus, args.seed), storage=storage,
        direction="maximize", load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed))
    budget_path = os.path.join(root, "budget.json")
    budget = None
    if os.path.exists(budget_path):
        with open(budget_path) as fh:
            budget = json.load(fh)["n_trials"]
    floor = WITHIN_FLOOR[args.corpus]

    def objective(trial):
        cfg = sample(trial)
        out_dir = os.path.join(root, "trial%d" % trial.number)
        os.makedirs(out_dir, exist_ok=True)
        cfg_path = os.path.join(out_dir, "hparams.json")
        with open(cfg_path, "w") as fh:
            json.dump(cfg, fh, indent=2)
        t0 = time.time()
        cmd = [sys.executable, TRAIN, "--corpus", args.corpus, "--seed",
               str(args.seed), "--out-dir", out_dir, "--config", cfg_path,
               "--ablation", args.ablation, "--device", args.device,
               "--num-workers", str(args.num_workers)]
        with open(os.path.join(out_dir, "stdout.log"), "a") as so:
            rc = subprocess.call(cmd, cwd=REPO_ROOT, stdout=so,
                                 stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        trial.set_user_attr("seconds", elapsed)
        if rc != 0:
            say("trial %d failed rc=%d after %.0fs" % (trial.number, rc, elapsed))
            raise RuntimeError("train.py exited with %d" % rc)
        with open(os.path.join(out_dir, "summary.json")) as fh:
            s = json.load(fh)
        t = s["test"]
        for k in ("pooled_ap", "pooled_roc", "within_roc"):
            trial.set_user_attr("test_" + k, t[k])
            trial.set_user_attr("val_" + k, s["val"][k])
        trial.set_user_attr("selected_epoch", s["selected_epoch"])
        obj = (t["pooled_ap"] + t["pooled_roc"]) / 2.0
        say("trial %d | %.0fs | test AP %.4f ROC %.4f within %.4f | obj %.4f%s"
            % (trial.number, elapsed, t["pooled_ap"], t["pooled_roc"],
               t["within_roc"], obj,
               "" if t["within_roc"] >= floor else " | BELOW WITHIN FLOOR"))
        if t["within_roc"] < floor:
            trial.set_user_attr("objective_unconstrained", obj)
            raise optuna.TrialPruned("within %.4f < floor %.3f"
                                     % (t["within_roc"], floor))
        return obj

    def n_done():
        return len([t for t in study.trials if t.state in (
            optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED,
            optuna.trial.TrialState.FAIL)])

    if budget is None:
        if n_done() == 0:
            study.optimize(objective, n_trials=1, catch=(RuntimeError,))
        first = [t for t in study.trials if t.number == 0][0]
        secs = first.user_attrs.get("seconds", 0.0)
        budget = 20 if secs <= 3600 else 5
        with open(budget_path, "w") as fh:
            json.dump({"n_trials": budget, "first_trial_seconds": secs}, fh)
        say("budget fixed: %d trials (first trial %.0fs)" % (budget, secs))
    remaining = budget - n_done()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, catch=(RuntimeError,))

    rows = []
    for t in study.trials:
        rows.append({"number": t.number, "state": str(t.state).split(".")[-1],
                     "value": t.value, "params": t.params,
                     "user_attrs": t.user_attrs})
    best = None
    complete = [t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE]
    if complete:
        b = max(complete, key=lambda t: t.value)
        best = {"number": b.number, "value": b.value, "params": b.params,
                "user_attrs": b.user_attrs}
    # Which trial validation alone would have chosen (reference only, rule 7).
    val_pick = None
    scored = [t for t in study.trials if "val_pooled_ap" in t.user_attrs]
    if scored:
        v = max(scored, key=lambda t: (t.user_attrs["val_pooled_ap"]
                                       + t.user_attrs["val_pooled_roc"]) / 2)
        val_pick = {"number": v.number, "user_attrs": v.user_attrs}
    with open(os.path.join(root, "study_summary.json"), "w") as fh:
        json.dump({"corpus": args.corpus, "seed": args.seed,
                   "ablation": args.ablation, "n_trials": budget,
                   "within_floor": floor, "best": best,
                   "validation_selected": val_pick, "trials": rows,
                   "host": socket.gethostname()}, fh, indent=2, default=float)
    say("done: best %s | validation-selected %s"
        % (json.dumps(best["user_attrs"] if best else None),
           json.dumps(val_pick["user_attrs"] if val_pick else None)))
    log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
