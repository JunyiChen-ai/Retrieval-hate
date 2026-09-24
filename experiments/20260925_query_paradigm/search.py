"""Fixed Optuna search for one (corpus, seed), rule 7 (README section 4).

    python experiments/20260925_query_paradigm/search.py --corpus hatemm --seed 234 \
        --out-root runs/20260925_query_paradigm

Each trial trains once (train.py): validation selects the checkpoint, test is scored at the primary operating
point (fixed 8 EIG questions per video, README section 2.4). Objective = (test pooled AP + test
pooled ROC) / 2. Trial budget after the first trial: 20 if it took <= 1 h, else 5 (budget.json, never changed).
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


def sample(trial):
    # README section 4 (declared before the search): three training scalars; everything else fixed at train.DEFAULTS
    return {
        "lr": trial.suggest_float("lr", 1e-4, 1e-3, log=True),
        "lamda_cma": trial.suggest_float("lamda_cma", 0.5, 2.0),
        "dropout": trial.suggest_float("dropout", 0.1, 0.5),
        # revision 1 (README section 7.3): learning rate of the answer model and the chain; revision 2 (README
        # section 9): the chain only (the answer model is fixed), range lowered so that the chain can stay at its start
        "lr_answer": trial.suggest_float("lr_answer", 1e-4, 1e-1, log=True),
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--extra-config", default=None, help="JSON dict of fixed settings (arms) for every trial")
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

    say("host %s | corpus %s | seed %d | %s" % (socket.gethostname(), args.corpus, args.seed,
                                                time.strftime("%Y-%m-%d %H:%M:%S")))
    study = optuna.create_study(study_name="%s_seed%d" % (args.corpus, args.seed),
                                storage="sqlite:///" + os.path.join(root, "optuna.db"), direction="maximize",
                                load_if_exists=True, sampler=optuna.samplers.TPESampler(seed=args.seed))
    extra = json.loads(args.extra_config) if args.extra_config else {}
    if extra:
        sys.path.insert(0, HERE)
        from train import DEFAULTS
        assert set(extra) <= set(DEFAULTS), sorted(set(extra) - set(DEFAULTS))
    budget_path = os.path.join(root, "budget.json")
    budget = json.load(open(budget_path))["n_trials"] if os.path.exists(budget_path) else None

    def objective(trial):
        sampled = sample(trial)
        assert not set(extra) & set(sampled)
        cfg = dict(extra)
        cfg.update(sampled)
        out_dir = os.path.join(root, "trial%d" % trial.number)
        os.makedirs(out_dir, exist_ok=True)
        cfg_path = os.path.join(out_dir, "hparams.json")
        json.dump(cfg, open(cfg_path, "w"), indent=2)
        t0 = time.time()
        cmd = [sys.executable, TRAIN, "--corpus", args.corpus, "--seed", str(args.seed), "--out-dir", out_dir,
               "--config", cfg_path, "--device", args.device, "--num-workers", str(args.num_workers)]
        with open(os.path.join(out_dir, "stdout.log"), "a") as so:
            rc = subprocess.call(cmd, cwd=REPO_ROOT, stdout=so, stderr=subprocess.STDOUT)
        elapsed = time.time() - t0
        trial.set_user_attr("seconds", elapsed)
        if rc != 0:
            say("trial %d FAILED rc=%d after %.0fs" % (trial.number, rc, elapsed))
            raise RuntimeError("train.py exited with %d" % rc)
        s = json.load(open(os.path.join(out_dir, "summary.json")))
        t = s["test"]
        pb = str(s["cfg"]["primary_budget"])
        v = s["fixed"][pb]["val"]
        for k in ("pooled_ap", "pooled_roc", "within_roc"):
            trial.set_user_attr("test_" + k, t[k])
            trial.set_user_attr("val_" + k, v[k])
        trial.set_user_attr("test_mean_calls", s["test_mean_calls"])
        trial.set_user_attr("best_epoch", s["best_epoch"])
        obj = (t["pooled_ap"] + t["pooled_roc"]) / 2.0
        say("trial %d | %.0fs | test AP %.4f ROC %.4f within %.4f | calls %.2f | obj %.4f" % (
            trial.number, elapsed, t["pooled_ap"], t["pooled_roc"], t["within_roc"], s["test_mean_calls"], obj))
        return obj

    def n_done():
        return len([t for t in study.trials if t.state in (optuna.trial.TrialState.COMPLETE,
                                                           optuna.trial.TrialState.FAIL)])

    if budget is None:
        if n_done() == 0:
            study.optimize(objective, n_trials=1, catch=(RuntimeError,))
        secs = [t for t in study.trials if t.number == 0][0].user_attrs.get("seconds", 0.0)
        budget = 20 if secs <= 3600 else 5
        json.dump({"n_trials": budget, "first_trial_seconds": secs}, open(budget_path, "w"))
        say("budget fixed: %d trials (first trial %.0fs)" % (budget, secs))
    remaining = budget - n_done()
    if remaining > 0:
        study.optimize(objective, n_trials=remaining, catch=(RuntimeError,))
    complete = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    best = val_pick = None
    if complete:
        b = max(complete, key=lambda t: t.value)
        best = {"number": b.number, "value": b.value, "params": b.params, "user_attrs": b.user_attrs}
        vp = max(complete, key=lambda t: t.user_attrs["val_pooled_ap"] + t.user_attrs["val_pooled_roc"])
        val_pick = {"number": vp.number, "user_attrs": vp.user_attrs}
    rows = [{"number": t.number, "state": str(t.state).split(".")[-1], "value": t.value, "params": t.params,
             "user_attrs": t.user_attrs} for t in study.trials]
    json.dump({"corpus": args.corpus, "seed": args.seed, "extra": extra, "n_trials": budget, "best": best,
               "validation_selected": val_pick, "trials": rows, "host": socket.gethostname()},
              open(os.path.join(root, "study_summary.json"), "w"), indent=2, default=float)
    say("DONE best %s | validation-selected %s" % (json.dumps(best and best["user_attrs"]),
                                                   json.dumps(val_pick and val_pick["user_attrs"])))


if __name__ == "__main__":
    main()
