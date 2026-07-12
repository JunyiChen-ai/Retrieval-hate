#!/usr/bin/env python3
"""LB-SCGP v6 solver availability probe.

This script performs no model, data, or solver computation.  It only imports
candidate numerical packages inside the SLURM job environment and records
versions/available CVXPY backends for later v6 certificate work.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import sys
from pathlib import Path


ROOT = Path("/data/jehc223/RGCL")
OUT_DIR = ROOT / "refine-logs" / "lb_scgp" / "v6" / "results"


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def package_record(import_name: str, dist_name: str | None = None) -> dict:
    record = {"import": import_name, "available": False}
    try:
        module = importlib.import_module(import_name)
    except Exception as exc:  # import availability is the point of the probe
        record["error"] = "{}: {}".format(type(exc).__name__, str(exc))
        return record
    record["available"] = True
    version = getattr(module, "__version__", None)
    if version is None:
        try:
            version = importlib.metadata.version(dist_name or import_name)
        except Exception:
            version = None
    record["version"] = version
    return record


def exclusive_json(path: Path, obj: dict) -> None:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode() + b"\n"
    with path.open("xb") as handle:
        handle.write(payload)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    packages = [
        ("numpy", None),
        ("scipy", None),
        ("cvxpy", None),
        ("scs", None),
        ("clarabel", None),
        ("osqp", None),
        ("ecos", None),
        ("cvxopt", None),
        ("mpmath", None),
        ("qpsolvers", None),
        ("quadprog", None),
        ("mosek", None),
        ("gurobipy", None),
    ]
    records = [package_record(name, dist) for name, dist in packages]
    cvxpy_solvers = []
    cvxpy_error = None
    try:
        import cvxpy as cp

        cvxpy_solvers = list(cp.installed_solvers())
    except Exception as exc:
        cvxpy_error = "{}: {}".format(type(exc).__name__, str(exc))
    obj = {
        "schema_version": 1,
        "task": "lb_scgp_v6_solver_probe",
        "slurm_job_id": job_id,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": sys.version,
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": records,
        "cvxpy_installed_solvers": cvxpy_solvers,
        "cvxpy_error": cvxpy_error,
        "source_hashes": {
            "solver_probe.py": sha256_file(
                ROOT / "refine-logs/lb_scgp/v6/runtime/solver_probe.py"
            ),
            "solver_probe.sbatch": sha256_file(
                ROOT / "refine-logs/lb_scgp/v6/runtime/solver_probe.sbatch"
            ),
        },
    }
    obj["payload_sha256"] = hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    out = OUT_DIR / "solver_probe_{}.json".format(job_id)
    exclusive_json(out, obj)
    print(json.dumps({"status": "OK", "path": str(out), "payload_sha256": obj["payload_sha256"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
