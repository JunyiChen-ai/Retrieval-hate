"""Detached owner: validate inputs, optionally complete a full study, then integrate evaluation."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import socket
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--corpus", required=True, choices=("hatemm", "hateclipseg", "dehate"))
    ap.add_argument("--seed", type=int, choices=(234, 2025, 3407))
    ap.add_argument("--train-first", action="store_true")
    ap.add_argument("--source-root", default="runs/20261001_inside_outside_backbone")
    ap.add_argument("--out-root", default="runs/20261002_complete_io_method")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--threads", type=int, default=2)
    a = ap.parse_args()
    os.chdir(ROOT)
    out_root = Path(a.out_root).resolve()
    run = out_root / a.corpus / (f"seed{a.seed}" if a.seed is not None else "integrated")
    run.mkdir(parents=True, exist_ok=True)
    if (run / "identity.json").exists():
        raise RuntimeError("Existing run identity; diagnose before resuming, never duplicate a run")
    identity = {"host": socket.gethostname(), "pid": os.getpid(), "pgid": os.getpgrp(), "sid": os.getsid(0),
                "start_ticks": Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19],
                "run": str(run), "repo": str(ROOT), "command": sys.argv,
                "started": datetime.now().astimezone().isoformat()}
    (run / "identity.json").write_text(json.dumps(identity, indent=2))
    (run / "run.pid").write_text(str(os.getpid()) + "\n")
    (run / "config.json").write_text(json.dumps(vars(a), indent=2))
    result = {"host": identity["host"], "run": str(run), "corpus": a.corpus, "seed": a.seed}
    with (run / "run.log").open("a", buffering=1) as log:
        log.write(f'host {identity["host"]} | {identity["started"]}\n')
        def call(command):
            log.write("command " + json.dumps(command) + "\n")
            subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
        try:
            setup = out_root / "setup"
            setup.mkdir(exist_ok=True)
            check_path = setup / f"inputs_{a.corpus}_{socket.gethostname()}.json"
            with check_path.with_suffix(".lock").open("a") as lock:
                fcntl.flock(lock, fcntl.LOCK_EX)
                if not check_path.exists():
                    call([sys.executable, str(EXPERIMENT / "check_inputs.py"), "--corpus", a.corpus,
                          "--out", str(check_path)])
                checked = json.loads(check_path.read_text())
                assert checked["status"] == "parsed_and_covered" and checked["host"] == socket.gethostname()
            if a.train_first:
                assert a.corpus == "dehate" and a.seed is not None
                call([sys.executable, str(EXPERIMENT / "search.py"), "--corpus", a.corpus,
                      "--seed", str(a.seed), "--out-root", str(out_root), "--device", a.device])
                studies = [run]
            else:
                seeds = (a.seed,) if a.seed is not None else (234, 2025, 3407)
                studies = [Path(a.source_root).resolve() / a.corpus / f"seed{s}" for s in seeds]
            call([sys.executable, str(EXPERIMENT / "evaluate.py"), "--corpus", a.corpus,
                  "--studies", *map(str, studies), "--out-dir", str(run / "results"),
                  "--device", a.device, "--threads", str(a.threads)])
            report = json.loads((run / "results/summary.json").read_text())
            result.update(status="success", evaluation_summary=str(run / "results/summary.json"),
                          evaluated_trials=len(report["trials"]), studies=list(map(str, studies)))
        except Exception as exc:
            result.update(status="failed", error=str(exc))
            log.write(f"FAILED {exc}\n")
        result["ended"] = datetime.now().astimezone().isoformat()
        (run / "completion.json").write_text(json.dumps(result, indent=2))
        log.write(f'FINISHED {result["status"]}\n')
    if result["status"] != "success":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
