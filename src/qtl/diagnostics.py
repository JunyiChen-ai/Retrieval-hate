"""Full-epoch, locked-configuration structural diagnostics; not seed confirmation."""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys


def inspect_trial(path, seed, cfg):
    summary = json.loads((path / "summary.json").read_text())
    if summary["seed"] != seed or summary["cfg"] != cfg:
        raise RuntimeError("Diagnostic seed/config does not match its locked specification")
    if len(summary["history"]) != cfg["max_epoch"]:
        raise RuntimeError("Diagnostic did not complete all training epochs")
    sources = {}
    for budget in (0, 8, 32):
        source = path / f"metrics_test_fixed{budget}.json"
        result = json.loads(source.read_text())["results"]["score_av"]
        values = (result["pr_auc"], result["roc_auc"], result["per_video"]["macro_auc"])
        if result["n_videos_missing_from_scores"] or not all(math.isfinite(x) for x in values):
            raise RuntimeError("Incomplete evaluator output")
        sources[str(budget)] = str(source)
    return {"seed": seed, "run": str(path), "metrics": sources}


def main(train_entry, experiment, arm_changes, expected_config):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg"))
    ap.add_argument("--arm", required=True, choices=tuple(arm_changes))
    ap.add_argument("--config", required=True)
    ap.add_argument("--source-trial", required=True)
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    cfg = json.loads(Path(a.config).read_text())
    if cfg["max_epoch"] != 50 or any(cfg.get(k) != v for k, v in expected_config.items()):
        raise ValueError("Source configuration does not match the declared full model")
    if set(arm_changes[a.arm]) - set(cfg):
        raise ValueError("Diagnostic arm contains an unknown configuration key")
    cfg.update(arm_changes[a.arm])
    source_trial = Path(a.source_trial).resolve()
    run = repo / "runs" / experiment / "diagnostics" / a.corpus / a.arm
    run.mkdir(parents=True, exist_ok=True)
    if (run / "identity.json").exists():
        raise RuntimeError("Existing run identity: inspect before resuming")
    identity = {"host": socket.gethostname(), "pid": os.getpid(), "pgid": os.getpgrp(),
                "sid": os.getsid(0), "start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
                "run": str(run), "repo": str(repo), "command": sys.argv,
                "started": datetime.now().astimezone().isoformat()}
    (run / "identity.json").write_text(json.dumps(identity, indent=2))
    (run / "run.pid").write_text(str(os.getpid()) + "\n")
    config = run / "config.json"
    config.write_text(json.dumps(cfg, indent=2))
    result = {"host": identity["host"], "run": str(run), "arm": a.arm,
              "purpose": "locked-configuration structural diagnosis, not Optuna confirmation", "trials": []}
    with (run / "run.log").open("a", buffering=1) as log:
        log.write("host %s | %s\n" % (identity["host"], identity["started"]))
        try:
            for seed in (234, 2025, 3407):
                if a.arm == "full" and seed == 234:
                    row = inspect_trial(source_trial, seed, cfg)
                    row["reused"] = True
                else:
                    trial = run / f"seed{seed}"
                    trial.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, str(Path(train_entry).resolve()),
                               "--corpus", a.corpus, "--seed", str(seed), "--out-dir", str(trial),
                               "--config", str(config), "--num-workers", "4"]
                    log.write("START seed %d\n" % seed)
                    with (trial / "stdout.log").open("a") as output:
                        rc = subprocess.call(command, cwd=repo, stdout=output, stderr=subprocess.STDOUT)
                    if rc:
                        raise RuntimeError("Training exited %d for seed %d" % (rc, seed))
                    row = inspect_trial(trial, seed, cfg)
                    row["command"] = command
                    row["reused"] = False
                result["trials"].append(row)
                log.write("COMPLETE seed %d reused=%s\n" % (seed, row["reused"]))
            result["status"] = "success"
        except Exception as exc:
            result.update(status="failed", error=str(exc))
            log.write("FAILED %s\n" % exc)
        result["ended"] = datetime.now().astimezone().isoformat()
        (run / "completion.json").write_text(json.dumps(result, indent=2))
        log.write("FINISHED %s\n" % result["status"])
    if result["status"] != "success":
        raise SystemExit(1)
