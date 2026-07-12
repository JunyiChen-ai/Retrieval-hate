#!/usr/bin/env python3
"""Fresh validator for the prospective v6 analytic feasibility supplement."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/jehc223/RGCL")
V6 = ROOT / "refine-logs" / "lb_scgp" / "v6"
OUT_DIR = V6 / "results"
TARGETS = [
    V6 / "runtime" / "analytic_feasibility_witness.py",
    V6 / "runtime" / "analytic_witness_replay.py",
    V6 / "runtime" / "validate_analytic_v6.py",
]
SBATCHES = [
    V6 / "runtime" / "validate_analytic_v6.sbatch",
    V6 / "runtime" / "analytic_feasibility_witness.sbatch",
]


def cjson(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hobj(obj: Any) -> str:
    return hashlib.sha256(cjson(obj).encode()).hexdigest()


def hfile(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_sbatch(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    cpus = None
    mem = None
    for line in text.splitlines():
        if line.startswith("#SBATCH --cpus-per-task="):
            cpus = int(line.split("=", 1)[1])
        if line.startswith("#SBATCH --mem="):
            raw = line.split("=", 1)[1].strip().upper()
            mem = int(raw[:-1]) if raw.endswith("G") else int(raw)
    return {
        "path": str(path.relative_to(ROOT)),
        "has_time_directive": "--time" in text,
        "cpus_per_task": cpus,
        "mem": mem,
        "conda_hatevideo": "conda activate HateVideo" in text,
        "sha256": hfile(path),
        "ok": bool("--time" not in text and cpus is not None and cpus <= 4 and mem is not None and mem <= 24 and "conda activate HateVideo" in text),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    ok = True
    compile_results = []
    for path in TARGETS:
        try:
            py_compile.compile(str(path), doraise=True)
            compile_results.append({"path": str(path.relative_to(ROOT)), "ok": True, "sha256": hfile(path)})
        except Exception as exc:
            ok = False
            compile_results.append({
                "path": str(path.relative_to(ROOT)),
                "ok": False,
                "error": "{}: {}".format(type(exc).__name__, str(exc)),
                "sha256": hfile(path),
            })
    sbatch_results = [parse_sbatch(path) for path in SBATCHES]
    ok = ok and all(row["ok"] for row in sbatch_results)
    self_check = {"ok": False, "not_run": True}
    if ok:
        proc = subprocess.run(
            [sys.executable, str(V6 / "runtime" / "analytic_feasibility_witness.py"), "--self-check"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            self_check = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception as exc:
            self_check = {
                "ok": False,
                "parse_error": "{}: {}".format(type(exc).__name__, str(exc)),
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
                "returncode": proc.returncode,
            }
        self_check["returncode"] = proc.returncode
        ok = ok and proc.returncode == 0 and bool(self_check.get("ok"))
    hashes = []
    for path in sorted((V6 / "runtime").glob("analytic*.py")) + sorted((V6 / "runtime").glob("*analytic*.sbatch")):
        hashes.append({"path": str(path.relative_to(ROOT)), "sha256": hfile(path)})
    out = {
        "schema_version": 1,
        "task": "lb_scgp_v6_analytic_validator",
        "slurm_job_id": job_id,
        "python": sys.version,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "ok": ok,
        "compile_results": compile_results,
        "sbatch_results": sbatch_results,
        "self_check": self_check,
        "hashes": hashes,
    }
    out["payload_sha256"] = hobj({k: v for k, v in out.items() if k != "payload_sha256"})
    out_path = OUT_DIR / "validation_analytic_{}.json".format(job_id)
    with out_path.open("xb") as handle:
        handle.write((cjson(out) + "\n").encode())
    print(cjson({"status": "OK" if ok else "FAIL", "path": str(out_path), "payload_sha256": out["payload_sha256"]}))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
