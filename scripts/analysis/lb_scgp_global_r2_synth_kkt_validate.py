#!/usr/bin/env python
"""Run2 preflight and SLURM-side validator."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from lb_scgp_global_r2_common import (  # noqa: E402
    canonical_json,
    canonical_root_path,
    old_protected_hash_manifest,
    payload_hash,
    read_json,
    require_slurm_cpu,
    sha256_file,
)


RUN1 = "LBSCGP-GLOBAL-G0-M0-CONTRACT-FREEZE-v1"
RUN2 = "LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1"
SCHEMA_ID = "scgp_global_synth_kkt_payload_v1"


def run_command(args: list[str], cwd: Path = ROOT) -> dict[str, Any]:
    proc = subprocess.run(args, cwd=cwd, text=True, capture_output=True)
    return {
        "args": args,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
        "status": "PASS" if proc.returncode == 0 else "FAIL",
    }


def scan_trailing_whitespace(paths: list[str]) -> dict[str, Any]:
    bad = []
    for rel in paths:
        path, _ = canonical_root_path(rel)
        if not path.exists() or not path.is_file():
            bad.append({"path": rel, "line": None, "reason": "missing"})
            continue
        with open(path, "rb") as handle:
            for idx, raw in enumerate(handle, 1):
                if raw.rstrip(b"\n\r").endswith((b" ", b"\t")):
                    bad.append({"path": rel, "line": idx, "reason": "trailing whitespace"})
    return {"status": "PASS" if not bad else "FAIL", "bad_lines": bad}


def check_sbatch(path: str) -> dict[str, Any]:
    fs_path, _ = canonical_root_path(path)
    text = fs_path.read_text(encoding="utf-8")
    active_sbatch = [
        line.strip()
        for line in text.splitlines()
        if line.startswith("#SBATCH")
    ]
    checks = {
        "no_time_flag": not any(line.startswith("#SBATCH --time") for line in active_sbatch),
        "cpu_8": "#SBATCH --cpus-per-task=8" in text,
        "mem_64g": "#SBATCH --mem=64G" in text,
        "no_gpu_directive": not any("--gres" in line or "--gpus" in line for line in active_sbatch),
        "conda_hatevideo": "conda activate HateVideo" in text,
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def check_run2_absent() -> dict[str, Any]:
    squeue = run_command(
        [
            "bash",
            "-lc",
            "squeue -h -u \"$USER\" -o '%i|%j|%T' | rg 'lbscgp_global_r2_run2_synth_kkt|LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1' || true",
        ]
    )
    sacct = run_command(
        [
            "bash",
            "-lc",
            "sacct -u \"$USER\" --starttime 2026-07-12 --format=JobID,JobName%80,State -P | rg 'lbscgp_global_r2_run2_synth_kkt|LBSCGP-GLOBAL-G0-M0-SYNTH-KKT-v1' || true",
        ]
    )
    no_squeue = squeue["stdout"].strip() == ""
    no_sacct = sacct["stdout"].strip() == ""
    return {
        "status": "PASS" if no_squeue and no_sacct else "FAIL",
        "squeue": squeue,
        "sacct": sacct,
        "no_existing_run2_squeue": no_squeue,
        "no_existing_run2_sacct": no_sacct,
    }


def semantic_static_checks(cfg: dict[str, Any]) -> dict[str, Any]:
    artifact_path, _ = canonical_root_path(cfg["run"]["artifact_path"])
    lock_path = artifact_path.with_name(artifact_path.name + ".publish.lock")
    run1 = read_json(cfg["dependencies"]["run1_artifact_path"])
    run1_payload = payload_hash(run1)
    old_hash, old_count = old_protected_hash_manifest()
    checks = {
        "run_id": cfg["run"]["run_id"] == RUN2,
        "schema_id": cfg["run"]["schema_id"] == SCHEMA_ID,
        "artifact_absent": not artifact_path.exists(),
        "lock_absent": not lock_path.exists(),
        "run1_frozen": run1.get("terminal_state") == "FROZEN",
        "run1_payload_hash": run1.get("payload_sha256") == cfg["dependencies"]["run1_payload_sha256"] == run1_payload,
        "run1_artifact_hash": sha256_file(cfg["dependencies"]["run1_artifact_path"])
        == cfg["dependencies"]["run1_artifact_sha256"],
        "run1_lock_hash": sha256_file(cfg["dependencies"]["run1_publish_lock_path"])
        == cfg["dependencies"]["run1_publish_lock_sha256"],
        "old_protected_hash": old_hash
        == "243e89b69b169b222dd97f9df092d511f823fb26201e91fd89cb581710940462",
        "old_protected_count": old_count == 278,
        "slurm_policy": cfg["run"]["slurm"]
        == {"cpu": 8, "ram_gb": 64, "gpu": 0, "env": "HateVideo", "no_time_flag": True},
        "no_forbidden_authorization": not any(
            cfg["authorization"][key]
            for key in [
                "mllm_calls_allowed",
                "ocr_calls_allowed",
                "performance_evaluation_allowed",
                "query_labels_allowed",
                "query_z_allowed",
                "run3_or_later_allowed",
                "training_allowed",
                "validation_or_test_allowed",
            ]
        ),
    }
    return {"status": "PASS" if all(checks.values()) else "FAIL", "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--phase", choices=["preflight", "slurm"], default="slurm")
    args = parser.parse_args()

    if args.phase == "slurm":
        require_slurm_cpu()
    if args.run_id != RUN2:
        raise RuntimeError(f"unauthorized run id: {args.run_id}")

    cfg = read_json(args.config)
    files = cfg["implementation_files"]
    checks: list[dict[str, Any]] = []
    for rel in [
        args.config,
        cfg["paths"]["schema"],
        cfg["paths"]["experiment_machine"],
        cfg["dependencies"]["run1_artifact_path"],
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
                "scripts/analysis/lb_scgp_global_r2_synth_kkt.py",
                "scripts/analysis/lb_scgp_global_r2_synth_kkt_validate.py",
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
                "configs/lb_scgp_global_r2/m0_synth_kkt_v1.json",
                "schemas/lb_scgp_global_r2/scgp_global_synth_kkt_payload_v1.schema.json",
                "scripts/analysis/lb_scgp_global_r2_synth_kkt.py",
                "scripts/analysis/lb_scgp_global_r2_synth_kkt_validate.py",
                "scripts/wrappers/lb_scgp_global_r2_run2_synth_kkt.sh",
                "scripts/slurm/lb_scgp_global_r2_m0_synth_kkt.sbatch",
                "refine-logs/lb_scgp_global/M0_SYNTH_KKT_IMPLEMENTATION.md",
                "refine-logs/lb_scgp_global/M0_SYNTH_KKT_EXECUTION.md",
                "refine-logs/lb_scgp_global/EXPERIMENT_TRACKER.md",
            ]
        )
    )
    static = semantic_static_checks(cfg)
    sbatch = check_sbatch(cfg["paths"]["slurm_script"])
    whitespace = scan_trailing_whitespace(files)
    run2_absent = check_run2_absent() if args.phase == "preflight" else {"status": "PASS", "skipped": "inside submitted job"}
    diff_name_status = run_command(["git", "diff", "--name-status"])
    jq_path = run_command(["bash", "-lc", "command -v jq"])
    status = (
        "PASS"
        if all(item["status"] == "PASS" for item in checks)
        and static["status"] == "PASS"
        and sbatch["status"] == "PASS"
        and whitespace["status"] == "PASS"
        and run2_absent["status"] == "PASS"
        else "FAIL"
    )
    result = {
        "schema_version": "lb_scgp_global_r2_run2_synth_kkt_validation_v1",
        "run_id": RUN2,
        "phase": args.phase,
        "status": status,
        "checks": checks,
        "semantic_static_checks": static,
        "sbatch_policy_check": sbatch,
        "new_file_whitespace_scan": whitespace,
        "run2_absence_check": run2_absent,
        "diff_name_status": diff_name_status,
        "jq_path": jq_path,
        "validator_sha256": sha256_file("scripts/analysis/lb_scgp_global_r2_synth_kkt_validate.py"),
    }
    out = Path(args.json_out)
    out.write_text(canonical_json(result) + "\n", encoding="utf-8")
    if status != "PASS":
        print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
