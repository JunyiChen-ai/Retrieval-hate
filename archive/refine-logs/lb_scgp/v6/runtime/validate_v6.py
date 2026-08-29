#!/usr/bin/env python3
"""Validate v6-only LB-SCGP review artifacts under SLURM."""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/jehc223/RGCL")
V6 = ROOT / "refine-logs" / "lb_scgp" / "v6"
REPORT = ROOT / "refine-logs" / "lb_scgp" / "G0_V6_NUMERICAL_CERTIFICATE_REVIEW.md"


def cjson(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def hobj(obj: Any) -> str:
    return hashlib.sha256(cjson(obj).encode()).hexdigest()


def hfile(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_payload(obj: dict[str, Any]) -> bool:
    expected = obj.get("payload_sha256")
    if not expected:
        return True
    body = {k: v for k, v in obj.items() if k != "payload_sha256"}
    return hobj(body) == expected


def main() -> int:
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    py_files = sorted((V6 / "runtime").glob("*.py"))
    compile_results = []
    ok = True
    for path in py_files:
        try:
            py_compile.compile(str(path), doraise=True)
            compile_results.append({"path": str(path.relative_to(ROOT)), "ok": True})
        except Exception as exc:
            ok = False
            compile_results.append({
                "path": str(path.relative_to(ROOT)),
                "ok": False,
                "error": "{}: {}".format(type(exc).__name__, str(exc)),
            })
    json_results = []
    for path in sorted(V6.rglob("*.json")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                obj = json.load(handle)
            payload_ok = validate_payload(obj) if isinstance(obj, dict) else True
            ok = ok and payload_ok
            json_results.append({
                "path": str(path.relative_to(ROOT)),
                "json_ok": True,
                "payload_ok": payload_ok,
                "sha256": hfile(path),
            })
        except Exception as exc:
            ok = False
            json_results.append({
                "path": str(path.relative_to(ROOT)),
                "json_ok": False,
                "payload_ok": False,
                "error": "{}: {}".format(type(exc).__name__, str(exc)),
            })
    hash_rows = []
    for path in sorted(V6.rglob("*")):
        if path.is_file():
            hash_rows.append({"path": str(path.relative_to(ROOT)), "sha256": hfile(path)})
    if REPORT.exists():
        hash_rows.append({"path": str(REPORT.relative_to(ROOT)), "sha256": hfile(REPORT)})
    out = {
        "schema_version": 1,
        "task": "lb_scgp_v6_artifact_validation",
        "slurm_job_id": job_id,
        "python": sys.version,
        "ok": ok,
        "compile_results": compile_results,
        "json_results": json_results,
        "hashes": hash_rows,
    }
    out["payload_sha256"] = hobj(out)
    out_path = V6 / "results" / "validation_{}.json".format(job_id)
    with out_path.open("xb") as handle:
        handle.write((cjson(out) + "\n").encode())
    print(cjson({"status": "OK" if ok else "FAIL", "path": str(out_path), "payload_sha256": out["payload_sha256"]}))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
