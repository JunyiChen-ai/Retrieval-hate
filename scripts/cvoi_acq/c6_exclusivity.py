"""C6 GPU-exclusivity guard (deviation D2, 2026-08-09).

External to the pinned cost modules: this file never imports or alters
`cost_driver.py` / `cost_overhead_driver.py` / `cost_audit.py` / `costs.py`, and it
computes no candidate metric and no cost value. It only observes GPU co-tenancy so a
timing run can be declared binding or VOID.

Subcommands:
  probe   one sample; exit 0 exclusive, 3 contended, 4 probe failure.
  watch   append samples to a JSONL file until terminated.
  verify  turn a sample file plus a run window into an EXCLUSIVE_OK / VOID verdict.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import subprocess
import sys
import time
from pathlib import Path

QUERY = ["nvidia-smi", "--query-compute-apps=pid,used_memory",
         "--format=csv,noheader,nounits"]


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _pgid(pid: int):
    try:
        out = subprocess.run(["ps", "-o", "pgid=", "-p", str(pid)],
                             text=True, capture_output=True, check=True).stdout.strip()
        return int(out) if out else None
    except Exception:
        return None


def _user(pid: int):
    try:
        return subprocess.run(["ps", "-o", "user=", "-p", str(pid)],
                              text=True, capture_output=True, check=True).stdout.strip() or None
    except Exception:
        return None


def sample(own_pgid: int | None) -> dict:
    """One co-tenancy sample. `probe_ok=False` is treated as contention by callers."""
    row = {"utc": _now(), "probe_ok": True, "apps": [], "foreign": [], "error": None}
    try:
        res = subprocess.run(QUERY, text=True, capture_output=True, check=True)
    except Exception as exc:
        row["probe_ok"] = False
        row["error"] = type(exc).__name__ + ":" + str(exc)
        return row
    for line in res.stdout.splitlines():
        line = line.strip()
        if not line or line.lower().startswith("no running"):
            continue
        parts = [p.strip() for p in line.split(",")]
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        mib = None
        if len(parts) > 1:
            try:
                mib = int(parts[1])
            except ValueError:
                mib = None
        pgid = _pgid(pid)
        app = {"pid": pid, "used_mib": mib, "pgid": pgid, "user": _user(pid),
               "own": (own_pgid is not None and pgid == own_pgid)}
        row["apps"].append(app)
        if not app["own"]:
            row["foreign"].append(app)
    row["exclusive"] = row["probe_ok"] and not row["foreign"]
    return row


def cmd_probe(a) -> int:
    row = sample(a.own_pgid)
    print(json.dumps(row, sort_keys=True))
    if not row["probe_ok"]:
        return 4
    return 0 if row["exclusive"] else 3


def cmd_watch(a) -> int:
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a") as fh:
        while True:
            fh.write(json.dumps(sample(a.own_pgid), sort_keys=True) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            time.sleep(a.interval)
    return 0


def cmd_verify(a) -> int:
    rows = [json.loads(x) for x in Path(a.samples).open() if x.strip()]
    window = [r for r in rows if a.start_utc <= r["utc"] <= a.end_utc]
    foreign = [r for r in window if r.get("foreign")]
    failed = [r for r in window if not r.get("probe_ok", False)]
    reasons = []
    if not window:
        reasons.append("no_cotenancy_samples_in_window")
    if len(window) < a.min_samples:
        reasons.append("insufficient_cotenancy_samples")
    if foreign:
        reasons.append("foreign_compute_process_observed")
    if failed:
        reasons.append("probe_failure_in_window")
    verdict = {
        "schema": "cvoi-c6-cotenancy-verdict/1",
        "deviation": "D2",
        "status": "EXCLUSIVE_OK" if not reasons else "VOID_CONTENDED",
        "reasons": sorted(set(reasons)),
        "run_start_utc": a.start_utc,
        "run_end_utc": a.end_utc,
        "sample_interval_s": a.interval,
        "n_samples_total": len(rows),
        "n_samples_in_window": len(window),
        "n_samples_with_foreign": len(foreign),
        "n_samples_probe_failed": len(failed),
        "foreign_pids": sorted({q["pid"] for r in foreign for q in r["foreign"]}),
        "foreign_users": sorted({q.get("user") for r in foreign for q in r["foreign"] if q.get("user")}),
        "max_foreign_used_mib": max([q.get("used_mib") or 0 for r in foreign for q in r["foreign"]], default=0),
        "candidate_metric_computed": False,
        "test_contact_count": 0,
    }
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    tmp = outp.with_suffix(outp.suffix + ".tmp")
    tmp.write_text(json.dumps(verdict, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, outp)
    print(json.dumps(verdict, sort_keys=True))
    return 0 if verdict["status"] == "EXCLUSIVE_OK" else 3


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("probe")
    p.add_argument("--own-pgid", type=int, default=None)
    p.set_defaults(fn=cmd_probe)

    w = sub.add_parser("watch")
    w.add_argument("--own-pgid", type=int, default=None)
    w.add_argument("--interval", type=float, default=15.0)
    w.add_argument("--out", required=True)
    w.set_defaults(fn=cmd_watch)

    v = sub.add_parser("verify")
    v.add_argument("--samples", required=True)
    v.add_argument("--start-utc", required=True)
    v.add_argument("--end-utc", required=True)
    v.add_argument("--interval", type=float, default=15.0)
    v.add_argument("--min-samples", type=int, default=4)
    v.add_argument("--out", required=True)
    v.set_defaults(fn=cmd_verify)

    a = ap.parse_args()
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
