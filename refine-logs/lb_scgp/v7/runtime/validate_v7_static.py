#!/usr/bin/env python3
"""Static machine validator for the LB-SCGP G0 v7 repair."""

from __future__ import annotations

import os
import py_compile
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from v7_common import (
    DESIGN,
    OUT_DIR,
    ROOT,
    RUNTIME,
    cjson,
    current_source_hashes,
    existing_hashes_unchanged,
    hfile,
    hobj,
    payload_hash,
    read_json,
    write_json_exclusive,
)


PY_FILES = [
    RUNTIME / "v7_common.py",
    RUNTIME / "v7_actual_certificate.py",
    RUNTIME / "v7_independent_replay.py",
    RUNTIME / "validate_v7_static.py",
]

SBATCHES = [
    RUNTIME / "validate_v7_static.sbatch",
    RUNTIME / "v7_actual_certificate.sbatch",
    RUNTIME / "v7_independent_replay.sbatch",
]


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
        "within_v7_bound": 1 <= cpus <= 4 and 1 <= mem <= 24,
        "ok": bool("--time" not in text and "conda activate HateVideo" in text and 1 <= cpus <= 4 and 1 <= mem <= 24),
    }


def compile_file(path: Path) -> dict[str, Any]:
    try:
        py_compile.compile(str(path), doraise=True)
        return {"path": str(path.relative_to(ROOT)), "ok": True, "sha256": hfile(path)}
    except Exception as exc:
        return {
            "path": str(path.relative_to(ROOT)),
            "ok": False,
            "error": "{}: {}".format(type(exc).__name__, str(exc)),
            "sha256": hfile(path),
        }


def replay_import_boundary() -> dict[str, Any]:
    text = (RUNTIME / "v7_independent_replay.py").read_text(encoding="utf-8")
    common_text = (RUNTIME / "v7_common.py").read_text(encoding="utf-8")
    forbidden = ["scipy", "cvxpy", "torch", "sklearn"]
    pattern = r"^\s*(from|import)\s+({})(\.|\s|$)"
    hits = [word for word in forbidden if re.search(pattern.format(re.escape(word)), text, re.M)]
    common_hits = [word for word in forbidden if re.search(pattern.format(re.escape(word)), common_text, re.M)]
    return {
        "replay_forbidden_import_hits": hits,
        "common_forbidden_import_hits": common_hits,
        "ok": not hits and not common_hits,
    }


def design_check() -> dict[str, Any]:
    design = read_json(DESIGN)
    ok = (
        design.get("eta") == 1e-12
        and design.get("eta_status") == "fixed_before_any_v7_numerical_run_and_not_to_be_changed"
        and design.get("immutable_contract", {}).get("topk") == 20
        and design.get("immutable_contract", {}).get("tau") == 1e-7
        and design.get("immutable_contract", {}).get("violation") == 1e-6
        and design.get("immutable_contract", {}).get("relative") == 1e-7
        and design.get("immutable_contract", {}).get("max_independent_orientations") == 8
        and design.get("immutable_contract", {}).get("max_pivots") == 32
    )
    return {
        "ok": ok,
        "path": str(DESIGN.relative_to(ROOT)),
        "sha256": hfile(DESIGN),
        "eta": design.get("eta"),
        "thread_session": design.get("thread_session"),
    }


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    compile_results = [compile_file(path) for path in PY_FILES]
    sbatch_results = [parse_sbatch(path) for path in SBATCHES]
    replay_boundary = replay_import_boundary()
    design = design_check()
    self_check = {"ok": False, "not_run": True}
    if all(row["ok"] for row in compile_results) and all(row["ok"] for row in sbatch_results) and replay_boundary["ok"] and design["ok"]:
        proc = subprocess.run(
            [sys.executable, str(RUNTIME / "v7_actual_certificate.py"), "--self-check"],
            cwd=str(ROOT),
            text=True,
            capture_output=True,
            check=False,
        )
        try:
            self_check = read_json_from_stdout(proc.stdout)
        except Exception as exc:
            self_check = {
                "ok": False,
                "parse_error": "{}: {}".format(type(exc).__name__, str(exc)),
                "stdout_tail": proc.stdout[-2000:],
                "stderr_tail": proc.stderr[-2000:],
                "returncode": proc.returncode,
            }
        self_check["returncode"] = proc.returncode
    unchanged = existing_hashes_unchanged()
    ok = bool(
        design["ok"]
        and all(row["ok"] for row in compile_results)
        and all(row["ok"] for row in sbatch_results)
        and replay_boundary["ok"]
        and self_check.get("ok") is True
        and unchanged.get("ok") is True
    )
    out = {
        "schema_version": 1,
        "task": "lb_scgp_g0_v7_static_validator",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", "no_slurm_job"),
        "python": sys.version,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "ok": ok,
        "design_check": design,
        "compile_results": compile_results,
        "sbatch_results": sbatch_results,
        "replay_import_boundary": replay_boundary,
        "self_check": self_check,
        "existing_hashes_unchanged": unchanged,
        "source_hashes": current_source_hashes(),
    }
    out["payload_sha256"] = payload_hash(out)
    out_path = OUT_DIR / "v7_static_validation_{}.json".format(os.environ.get("SLURM_JOB_ID", "no_slurm_job"))
    write_json_exclusive(out_path, out)
    print(cjson({"status": "OK" if ok else "FAIL", "path": str(out_path.relative_to(ROOT)), "payload_sha256": out["payload_sha256"]}))
    return 0 if ok else 2


def read_json_from_stdout(stdout: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("empty stdout")
    import json

    return json.loads(lines[-1])


if __name__ == "__main__":
    raise SystemExit(main())
