"""Shared detached QTL study owner; promoted from the initial backbone launcher.

The original launcher stays frozen until its currently active search ends.
"""
import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import socket
import subprocess
import sys


def main(search_entry, default_root):
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg", "dehate"))
    ap.add_argument("--seed", type=int, default=234)
    ap.add_argument("--out-root", default=default_root)
    ap.add_argument("--extra-config")
    a = ap.parse_args()
    repo = Path(__file__).resolve().parents[2]
    os.chdir(repo)
    run = (Path(a.out_root) / a.corpus / f"seed{a.seed}").resolve()
    run.mkdir(parents=True, exist_ok=True)
    if (run / "identity.json").exists():
        raise RuntimeError("Existing run identity: inspect before resuming this output directory")
    identity = {"host": socket.gethostname(), "pid": os.getpid(), "pgid": os.getpgrp(),
                "sid": os.getsid(0), "start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
                "run": str(run), "repo": str(repo), "command": sys.argv,
                "started": datetime.now().astimezone().isoformat()}
    (run / "identity.json").write_text(json.dumps(identity, indent=2))
    (run / "run.pid").write_text(str(os.getpid()) + "\n")
    command = [sys.executable, str(Path(search_entry).resolve()),
               "--corpus", a.corpus, "--seed", str(a.seed), "--out-root", str(Path(a.out_root).resolve())]
    if a.extra_config:
        command += ["--extra-config", a.extra_config]
    result = {"host": identity["host"], "run": str(run), "command": command}
    with (run / "run.log").open("a", buffering=1) as log:
        log.write("host %s | %s\n" % (identity["host"], identity["started"]))
        try:
            rc = subprocess.call(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            if rc:
                raise RuntimeError("search exited with code %d" % rc)
            summary = json.loads((run / "study_summary.json").read_text())
            complete = [t for t in summary["trials"] if t["state"] == "COMPLETE"]
            if len(complete) != summary["n_trials"]:
                raise RuntimeError("Study does not have its full completed trial budget")
            for trial in complete:
                metric = json.loads((run / f'trial{trial["number"]}' / "metrics_test_fixed8.json").read_text())
                r = metric["results"]["score_av"]
                if r["n_videos_missing_from_scores"] or not all(math.isfinite(x) for x in
                        (r["pr_auc"], r["roc_auc"], r["per_video"]["macro_auc"])):
                    raise RuntimeError("Incomplete/invalid evaluator output")
            result.update(status="success", n_complete=len(complete), n_trials=summary["n_trials"])
        except Exception as exc:
            result.update(status="failed", error=str(exc))
            log.write("FAILED %s\n" % exc)
        result["ended"] = datetime.now().astimezone().isoformat()
        (run / "completion.json").write_text(json.dumps(result, indent=2))
        log.write("FINISHED %s\n" % result["status"])
    if result["status"] != "success":
        raise SystemExit(1)

