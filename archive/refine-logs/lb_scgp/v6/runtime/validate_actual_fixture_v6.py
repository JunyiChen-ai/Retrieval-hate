#!/usr/bin/env python3
"""Fresh validator for the v6 actual fixture oracle."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/jehc223/RGCL")
V6 = ROOT / "refine-logs" / "lb_scgp" / "v6"
OUT_DIR = V6 / "results"
PY_FILES = [
    V6 / "runtime" / "actual_fixture_oracle.py",
    V6 / "runtime" / "actual_fixture_replay.py",
    V6 / "runtime" / "validate_actual_fixture_v6.py",
]
SBATCHES = [
    V6 / "runtime" / "validate_actual_fixture_v6.sbatch",
    V6 / "runtime" / "actual_fixture_oracle.sbatch",
]


def cjson(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hobj(obj: Any) -> str:
    return hashlib.sha256(cjson(obj).encode("utf-8")).hexdigest()


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
    cpus_match = re.search(r"^#SBATCH --cpus-per-task=(\d+)$", text, re.M)
    mem_match = re.search(r"^#SBATCH --mem=(\d+)G$", text, re.M)
    cpus = int(cpus_match.group(1)) if cpus_match else -1
    mem = int(mem_match.group(1)) if mem_match else -1
    return {
        "path": str(path.relative_to(ROOT)),
        "sha256": hfile(path),
        "has_no_time_directive": "--time" not in text,
        "conda_hatevideo": "conda activate HateVideo" in text,
        "cpus_per_task": cpus,
        "mem_gb": mem,
        "within_added_job_bound": 1 <= cpus <= 8 and 1 <= mem <= 64,
        "ok": bool("--time" not in text and "conda activate HateVideo" in text and 1 <= cpus <= 8 and 1 <= mem <= 64),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    ok = True
    compile_results = []
    for path in PY_FILES:
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
            [sys.executable, str(V6 / "runtime" / "actual_fixture_oracle.py"), "--self-check"],
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
    out = {
        "schema_version": 1,
        "task": "lb_scgp_v6_actual_fixture_validator",
        "slurm_job_id": job_id,
        "python": sys.version,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "ok": ok,
        "compile_results": compile_results,
        "sbatch_results": sbatch_results,
        "self_check": self_check,
        "hashes": [{"path": str(path.relative_to(ROOT)), "sha256": hfile(path)} for path in PY_FILES + SBATCHES],
    }
    out["payload_sha256"] = hobj({k: v for k, v in out.items() if k != "payload_sha256"})
    out_path = OUT_DIR / "actual_fixture_validation_{}.json".format(job_id)
    with out_path.open("xb") as handle:
        handle.write((cjson(out) + "\n").encode("utf-8"))
    print(cjson({"status": "OK" if ok else "FAIL", "path": str(out_path.relative_to(ROOT)), "payload_sha256": out["payload_sha256"]}))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
