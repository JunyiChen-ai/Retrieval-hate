#!/usr/bin/env python3
"""Fresh v6 scientific-repair validator.

This validator is intentionally prospective and nonformal.  It compiles the
new v6 repair scripts, verifies the machine-readable design contract, checks
SLURM wrappers for the requested resource/conda constraints, and records a
machine gate before any nonformal sanity oracle is submitted.
"""

from __future__ import annotations

import hashlib
import json
import os
import py_compile
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path("/data/jehc223/RGCL")
V6 = ROOT / "refine-logs" / "lb_scgp" / "v6"
RUNTIME = V6 / "runtime"
RESULTS = V6 / "results"
DESIGN = V6 / "G0_V6_SCIENTIFIC_REPAIR_DESIGN_TEST_MACHINE.json"
TRACKER = V6 / "G0_V6_PROSPECTIVE_TRACKER_FINDING.md"
V5_CONFIG = ROOT / "configs" / "lb_scgp" / "lb_scgp_v5.json"
HANDOFF = ROOT / "refine-logs" / "lb_scgp" / "G0_V6_SCIENTIFIC_REPAIR_HANDOFF.md"

PY_FILES = [
    RUNTIME / "validate_scientific_repair_v6.py",
    RUNTIME / "scientific_repair_sanity.py",
    RUNTIME / "scientific_repair_replay.py",
]
SBATCH_FILES = [
    RUNTIME / "validate_scientific_repair_v6.sbatch",
    RUNTIME / "scientific_repair_sanity.sbatch",
]


def cjson(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256_obj(obj: Any) -> str:
    return hashlib.sha256(cjson(obj).encode("utf-8")).hexdigest()


def payload_sha256(obj: dict[str, Any]) -> str:
    return sha256_obj({k: v for k, v in obj.items() if k != "payload_sha256"})


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def slurm_resource_check(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    cpus_match = re.search(r"^#SBATCH --cpus-per-task=(\d+)$", text, re.M)
    mem_match = re.search(r"^#SBATCH --mem=(\d+)G$", text, re.M)
    cpus = int(cpus_match.group(1)) if cpus_match else -1
    mem_gb = int(mem_match.group(1)) if mem_match else -1
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "has_no_time_directive": "--time" not in text,
        "conda_hatevideo": "conda activate HateVideo" in text,
        "cpus_per_task": cpus,
        "mem_gb": mem_gb,
        "within_added_job_bound": 1 <= cpus <= 8 and 1 <= mem_gb <= 64,
        "sha256": sha256_file(path),
    }


def main() -> int:
    job_id = os.environ.get("SLURM_JOB_ID", "no_slurm_job")
    ok = True
    failures: list[str] = []

    v5 = read_json(V5_CONFIG)
    solver = v5["solver"]
    supervision = v5["supervision"]
    counters = v5["counters"]
    design = read_json(DESIGN)

    threshold_expected = {
        "topk": 20,
        "max_cycles": 500,
        "dykstra_set_violation_tolerance": 1e-6,
        "dykstra_relative_change_tolerance": 1e-7,
        "tie_tolerance": 1e-7,
        "max_independent_orientations": 8,
        "max_pivots": 32,
    }
    threshold_observed = {
        "topk": solver["topk"],
        "max_cycles": solver["max_dykstra_cycles"],
        "dykstra_set_violation_tolerance": solver["dykstra_set_violation_tolerance"],
        "dykstra_relative_change_tolerance": solver["dykstra_relative_change_tolerance"],
        "tie_tolerance": solver["tie_tolerance"],
        "max_independent_orientations": solver["max_independent_orientations"],
        "max_pivots": solver["max_pivots"],
    }
    thresholds_ok = threshold_observed == threshold_expected
    if not thresholds_ok:
        ok = False
        failures.append("immutable_threshold_mismatch")

    design_thresholds_ok = design.get("immutable_thresholds") == threshold_expected
    if not design_thresholds_ok:
        ok = False
        failures.append("design_threshold_mismatch")

    supervision_expected = {
        "only_gold_supervision": "parent_video_binary_label",
        "segment_gold_exists": False,
        "segment_gold_used": False,
    }
    supervision_ok = all(supervision.get(k) == v for k, v in supervision_expected.items())
    zero_counters_ok = all(int(counters.get(k, -1)) == 0 for k in [
        "mllm_call_count",
        "ocr_call_count",
        "teacher_cache_read_count",
        "teacher_cache_write_count",
        "outer_held_label_read_count",
        "outer_held_content_read_count",
        "val_content_read_count",
        "test_content_read_count",
        "val_test_teacher_artifact_count",
    ])
    if not supervision_ok or not zero_counters_ok:
        ok = False
        failures.append("supervision_or_counter_contract_failed")

    compile_results = []
    for path in PY_FILES:
        try:
            py_compile.compile(str(path), doraise=True)
            compile_results.append({"path": str(path.relative_to(ROOT)), "ok": True, "sha256": sha256_file(path)})
        except Exception as exc:  # pragma: no cover - emitted as machine JSON
            ok = False
            failures.append("py_compile_failed:" + str(path.relative_to(ROOT)))
            compile_results.append({
                "path": str(path.relative_to(ROOT)),
                "ok": False,
                "error": "{}: {}".format(type(exc).__name__, str(exc)),
            })

    sbatch_results = [slurm_resource_check(path) for path in SBATCH_FILES]
    for item in sbatch_results:
        if not (
            item["exists"]
            and item["has_no_time_directive"]
            and item["conda_hatevideo"]
            and item["within_added_job_bound"]
        ):
            ok = False
            failures.append("sbatch_contract_failed:" + item["path"])

    required_cases = set(design.get("required_adversarial_sanity_cases", []))
    expected_cases = {
        "top20_stable_outsider_shuffle",
        "zero_orientation_scalar_converged_top20_stable_true",
        "zero_orientation_scalar_converged_top20_stable_false",
        "known_local_one_boundary_orientation",
        "known_bounded_one_boundary_orientation",
        "near_threshold_below_1e-6",
        "near_threshold_above_1e-6",
        "relative_change_without_feasibility",
        "canonical_tie_below_1e-7",
        "canonical_tie_at_1e-7",
        "canonical_tie_above_1e-7",
        "duplicate_id_negative",
        "unresolved_tie_map_negative",
        "orientation_over_budget",
        "pivot_over_budget",
        "self_exclusion_top20",
        "psd_unitdiag_box_trust_stress",
        "no_segment_zero_counter_manifest",
    }
    cases_ok = required_cases == expected_cases
    if not cases_ok:
        ok = False
        failures.append("required_case_set_mismatch")

    hash_rows = []
    for path in [DESIGN, TRACKER, HANDOFF] + PY_FILES + SBATCH_FILES:
        if path.exists():
            hash_rows.append({"path": str(path.relative_to(ROOT)), "sha256": sha256_file(path)})

    out = {
        "schema_version": 1,
        "task": "lb_scgp_v6_scientific_repair_validator",
        "slurm_job_id": job_id,
        "status": "OK" if ok else "FAIL",
        "ok": ok,
        "failures": failures,
        "python": sys.version,
        "thresholds": {
            "observed_from_v5_config": threshold_observed,
            "expected_immutable": threshold_expected,
            "thresholds_ok": thresholds_ok,
            "design_thresholds_ok": design_thresholds_ok,
        },
        "supervision": {
            "supervision_ok": supervision_ok,
            "zero_counters_ok": zero_counters_ok,
            "only_gold_supervision": supervision.get("only_gold_supervision"),
            "segment_gold_exists": supervision.get("segment_gold_exists"),
            "segment_gold_used": supervision.get("segment_gold_used"),
        },
        "phase_i_semantics": design.get("phase_i_semantics"),
        "case_set_ok": cases_ok,
        "compile_results": compile_results,
        "sbatch_results": sbatch_results,
        "hashes": hash_rows,
    }
    out["payload_sha256"] = payload_sha256(out)
    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS / "scientific_repair_validation_{}.json".format(job_id)
    with out_path.open("xb") as handle:
        handle.write((cjson(out) + "\n").encode("utf-8"))
    print(cjson({"status": out["status"], "path": str(out_path.relative_to(ROOT)), "payload_sha256": out["payload_sha256"]}))
    return 0 if ok else 2


if __name__ == "__main__":
    raise SystemExit(main())
