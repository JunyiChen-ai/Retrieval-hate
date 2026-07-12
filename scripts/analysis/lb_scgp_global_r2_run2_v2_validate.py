#!/usr/bin/env python
"""SLURM-only Run2 preflight validator."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path("/data/jehc223/RGCL")
sys.path.insert(0, str(ROOT / "scripts/analysis"))

from lb_scgp_global_r2_run2_v2_common import (  # noqa: E402
    PAYLOAD_SCHEMA_ID,
    RUN2,
    canonical_json,
    canonical_root_path,
    old_protected_hash_manifest,
    read_json,
    require_slurm_run2,
    schema_requires_no_additional_properties,
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


def schema_strict_check(paths: list[str]) -> dict[str, object]:
    errors = {}
    for rel in paths:
        schema = read_json(rel)
        bad = schema_requires_no_additional_properties(schema)
        if bad:
            errors[rel] = bad[:20]
    return {"status": "PASS" if not errors else "FAIL", "errors": errors}


def verify_run1_hashes(cfg: dict[str, object]) -> dict[str, object]:
    mismatches = []
    for rel, expected in cfg["hash_bindings"]["run1_frozen"].items():
        actual = sha256_file(ROOT / rel)
        if actual != expected:
            mismatches.append({"path": rel, "expected": expected, "actual": actual})
    old_hash, old_count = old_protected_hash_manifest()
    expected_old = cfg["hash_bindings"]["old_protected_pre_snapshot"]
    if old_hash != expected_old["manifest_sha256"] or old_count != expected_old["path_count"]:
        mismatches.append(
            {
                "path": "old_protected_scope",
                "expected": expected_old,
                "actual": {"manifest_sha256": old_hash, "path_count": old_count},
            }
        )
    return {"status": "PASS" if not mismatches else "FAIL", "mismatches": mismatches}


def no_clobber_check(cfg: dict[str, object]) -> dict[str, object]:
    bad = []
    for rel in [
        cfg["paths"]["artifact_path"],
        cfg["paths"]["source_manifest_path"],
        cfg["paths"]["access_ledger_path"],
        cfg["paths"]["semantic_verification_path"],
    ]:
        path, _ = canonical_root_path(rel)
        lock = path.with_name(path.name + ".publish.lock")
        if path.exists() or lock.exists():
            bad.append(rel)
    for rel in cfg["dirty_policy"]["allowed_new_files_after_run"]:
        path, _ = canonical_root_path(rel)
        if path.exists():
            bad.append(rel)
    return {"status": "PASS" if not bad else "FAIL", "existing_forbidden_outputs": bad}


def python_dependency_check() -> dict[str, object]:
    script = (
        "import importlib.util, json, sys\n"
        "missing=[name for name in ['jsonschema'] if importlib.util.find_spec(name) is None]\n"
        "print(json.dumps({'checked':['jsonschema'],'missing':missing}))\n"
        "sys.exit(1 if missing else 0)\n"
    )
    result = run_command([sys.executable, "-c", script])
    result["check"] = "python_dependency_jsonschema"
    return result


def resource_and_run_check(cfg: dict[str, object], run_id: str) -> dict[str, object]:
    errors = []
    if run_id != RUN2 or cfg["run"]["run_id"] != RUN2:
        errors.append("run_id")
    if cfg["run"]["schema_id"] != PAYLOAD_SCHEMA_ID:
        errors.append("schema_id")
    if cfg["run"]["artifact_path"] != "artifacts/lb_scgp_global/v2/m0/synth_kkt/manifest.json":
        errors.append("artifact_path")
    if cfg["run"]["slurm"] != {"cpu": 8, "ram_gb": 64, "gpu": 0, "env": "HateVideo", "no_time_flag": True}:
        errors.append("slurm_resource")
    if cfg["authorization"]["authorized_run_ids"] != [RUN2]:
        errors.append("authorization")
    return {"status": "PASS" if not errors else "FAIL", "errors": errors}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--json-out", required=True)
    args = parser.parse_args()

    require_slurm_run2()
    cfg = read_json(args.config)
    files = cfg["implementation_files"]
    checks = []
    for rel in [
        args.config,
        cfg["paths"]["experiment_machine"],
        cfg["paths"]["payload_schema"],
        cfg["paths"]["case_schema"],
        cfg["paths"]["cert_schema"],
        "artifacts/lb_scgp_global/v1/m0/contract_freeze.json",
    ]:
        checks.append(run_command(["jq", "-e", ".", rel]))
    checks.append(schema_strict_check([cfg["paths"]["payload_schema"], cfg["paths"]["case_schema"], cfg["paths"]["cert_schema"]]))
    for rel in [cfg["paths"]["wrapper"], cfg["paths"]["slurm_script"]]:
        checks.append(run_command(["bash", "-n", rel]))
    checks.append(python_dependency_check())
    checks.append(run_command([sys.executable, "-m", "py_compile"] + [path for path in files if path.endswith(".py")]))
    checks.append(run_command(["git", "diff", "--check", "--"] + files + [cfg["paths"]["experiment_tracker"]]))
    checks.append(scan_trailing_whitespace(files))
    checks.append(verify_run1_hashes(cfg))
    checks.append(no_clobber_check(cfg))
    checks.append(resource_and_run_check(cfg, args.run_id))
    status = "PASS" if all(item["status"] == "PASS" for item in checks) else "FAIL"
    result = {
        "schema_version": "lb_scgp_global_r2_run2_v2_validation_v1",
        "run_id": args.run_id,
        "status": status,
        "checks": checks,
        "validator_sha256": sha256_file("scripts/analysis/lb_scgp_global_r2_run2_v2_validate.py"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID", ""),
    }
    Path(args.json_out).write_text(canonical_json(result) + "\n", encoding="utf-8")
    if status != "PASS":
        print(json.dumps(result, indent=2, sort_keys=True), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
