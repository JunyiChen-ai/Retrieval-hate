#!/usr/bin/env python
"""SLURM-only Run1 validator for LB-SCGP Global-R2 M0."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from lb_scgp_global_r2_common import (  # noqa: E402
    RUN1,
    canonical_json,
    read_json,
    require_slurm_cpu,
    sha256_file,
)


def run_command(args: list[str], cwd: Path = ROOT) -> dict[str, object]:
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return {
        "args": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "status": "PASS" if proc.returncode == 0 else "FAIL",
    }


def scan_trailing_whitespace(paths: list[str]) -> dict[str, object]:
    bad = []
    for rel in paths:
        path = ROOT / rel
        with open(path, "rb") as handle:
            for idx, raw in enumerate(handle, 1):
                if raw.rstrip(b"\n\r").endswith((b" ", b"\t")):
                    bad.append({"path": rel, "line": idx})
    return {"status": "PASS" if not bad else "FAIL", "bad_lines": bad}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    require_slurm_cpu()
    if args.run_id != RUN1:
        raise RuntimeError(f"unauthorized run id: {args.run_id}")

    cfg = read_json(args.config)
    if cfg["run"]["run_id"] != args.run_id:
        raise RuntimeError("config run_id mismatch")
    files = cfg["implementation_files"]

    checks = []
    for rel in [
        args.config,
        cfg["paths"]["experiment_machine"],
        cfg["paths"]["cert_schema"],
        cfg["paths"]["contract_schema"],
    ]:
        checks.append(run_command(["jq", "-e", ".", rel]))
    for rel in [cfg["paths"]["wrapper"], cfg["paths"]["slurm_script"]]:
        checks.append(run_command(["bash", "-n", rel]))
    checks.append(
        run_command(
            [
                sys.executable,
                "-m",
                "py_compile",
                "scripts/analysis/lb_scgp_global_r2_common.py",
                "scripts/analysis/lb_scgp_global_r2_contract_freeze.py",
                "scripts/analysis/lb_scgp_global_r2_validate.py",
            ]
        )
    )
    checks.append(
        run_command(
            [
                "git",
                "diff",
                "--check",
                "--",
                "refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md",
            ]
        )
    )
    whitespace = scan_trailing_whitespace(files)
    status = "PASS" if all(item["status"] == "PASS" for item in checks) and whitespace["status"] == "PASS" else "FAIL"
    result = {
        "schema_version": "lb_scgp_global_r2_run1_validation_v1",
        "run_id": args.run_id,
        "status": status,
        "checks": checks,
        "tracked_git_diff_check_scope": ["refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md"],
        "new_file_whitespace_scan": whitespace,
        "jq_path": run_command(["bash", "-lc", "command -v jq"]),
        "validator_sha256": sha256_file("scripts/analysis/lb_scgp_global_r2_validate.py"),
    }
    out = Path(args.json_out)
    out.write_text(canonical_json(result) + "\n", encoding="utf-8")
    if status != "PASS":
        print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
