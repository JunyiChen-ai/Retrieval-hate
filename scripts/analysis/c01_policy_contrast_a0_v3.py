#!/usr/bin/env python3
"""C01 A0 v3: v2-identical science with derived numerical-equivalence guards.

This file SHA-binds and reuses the frozen v2 implementation as its scientific
base.  It changes only lineage/output identity and the HALT-only registered-null
with/remove comparison.  It is CPU-only, Slurm-only, and test paths are blocked.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import math
import os
import platform
import struct
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/c01/c01_a0_v3.json"
V3_CONFIG_SHA256 = (
    "4ddb0f6f322de06316ea014a77c732b1a"
    "593c0fae5d926558d6c64a1be21cda5"
)
BASE_SOURCE = "scripts/analysis/c01_policy_contrast_a0.py"
BASE_SOURCE_SHA256 = (
    "d2b9c2ff909c07518ae35526db9550df"
    "655fb4af395cc7a0899f83e48db1b855"
)
HALT_LABEL = "HALT_NUMERICAL_EQUIVALENCE"
U32 = 2.0 ** -24
U64 = 2.0 ** -53
SCORE_ROUNDOFF = 2.0 ** -45
REFERENCE_QUERY_CHUNK = 8


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_base():
    path = (REPO / BASE_SOURCE).resolve()
    if sha256_file(path) != BASE_SOURCE_SHA256:
        raise RuntimeError("frozen v2 analysis source SHA256 drift")
    spec = importlib.util.spec_from_file_location("c01_a0_v2_frozen_base", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v2 analysis source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = load_frozen_base()
ACTIVE_V3 = None
LINEAGE_AUDIT = None
SOFTWARE_AUDIT = None
ULP_SELF_CHECK = None


def halt(context, message):
    base.die("{}: {}: {}".format(HALT_LABEL, context, message))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Repo-relative frozen C01 A0 v3 configuration.",
    )
    return parser.parse_args()


def load_v3_config(relative):
    path = base.repo_path(relative, "v3 config")
    base.reject_test_path(path, "v3 config")
    digest = sha256_file(path)
    if digest != V3_CONFIG_SHA256:
        base.die("v3 config SHA256 differs from reviewed canonical config")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    expected_keys = {
        "schema_version",
        "experiment_id",
        "run_id",
        "scientific_base",
        "execution",
        "lineage_evidence",
        "numerical_equivalence_v3",
        "output",
    }
    if set(config) != expected_keys:
        base.die("v3 config top-level schema changed")
    if (
        config["schema_version"] != 1
        or config["experiment_id"] != "C01_A0"
        or config["run_id"] != "C01-A0-v3"
        or config["numerical_equivalence_v3"]["halt_label"] != HALT_LABEL
        or config["numerical_equivalence_v3"]["float32"]["unit_roundoff"]
        != U32
        or config["numerical_equivalence_v3"]["weighted_vote"][
            "binary64_roundoff_allowance"
        ]
        != SCORE_ROUNDOFF
        or config["numerical_equivalence_v3"]["binary64_reference"][
            "query_chunk_size"
        ]
        != REFERENCE_QUERY_CHUNK
    ):
        base.die("v3 canonical identity/arithmetic binding changed")
    return config, path, digest


def recursive_diff(left, right, prefix=""):
    if isinstance(left, dict) and isinstance(right, dict):
        paths = []
        for key in sorted(set(left) | set(right)):
            child = "{}.{}".format(prefix, key) if prefix else key
            if key not in left or key not in right:
                paths.append(child)
            else:
                paths.extend(recursive_diff(left[key], right[key], child))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if left == right:
            return []
        return [prefix]
    return [] if left == right else [prefix]


def build_runtime_config(v3):
    scientific = v3["scientific_base"]
    base_path = base.repo_path(scientific["config_path"], "scientific base config")
    if sha256_file(base_path) != scientific["config_sha256"]:
        base.die("frozen v2 scientific config SHA256 drift")
    if (
        scientific["analysis_path"] != BASE_SOURCE
        or scientific["analysis_sha256"] != BASE_SOURCE_SHA256
    ):
        base.die("frozen v2 source binding changed in v3 config")
    with base_path.open("r", encoding="utf-8") as handle:
        v2 = json.load(handle)
    base.validate_config(v2)
    runtime = copy.deepcopy(v2)
    runtime["run_id"] = v3["run_id"]
    for key in (
        "namespace",
        "result_file",
        "decision_file",
        "record_file",
        "slurm_stdout",
        "slurm_stderr",
        "maximum_result_bytes",
        "atomic_json",
        "decision_exclusive_create",
        "run_namespace_no_clobber",
        "result_schema_version",
    ):
        runtime["output"][key] = v3["output"][key]
    schema = runtime["output"]["decision_schema"]
    schema["schema_version"] = v3["output"]["decision_schema_version"]
    schema["required_halt_only_validity_guards"] = [
        "probe_evidence_exact",
        "v3_lineage_evidence_exact",
        "raw_zero_allowlist_exact",
        "derived_zero_masks_exact",
        "displacement_null_exclusion_dual_path_exact",
        "shuffle_fixed_point_bijection",
        "registered_null_absent_from_all_top20",
        "with_null_remove_null_numerical_equivalence_v3",
        "deterministic_binary64_reference",
        "scientific_boolean_basis_exact",
    ]
    schema["exact_array_comparison"] = (
        "operands_mapping_raw_and_stable_neighbors_neighbor_labels_"
        "predictions_dtype_shape_c_order_bytes_exact"
    )
    schema["exact_metric_comparison"] = (
        "canonical_sorted_typed_ieee754_binary64_hex_json_bytes_exact"
    )
    runtime["zero_contract_v2"]["remove_null_comparison"] = (
        "exact_discrete_invariants_and_formula_derived_finite_similarity_"
        "score_equivalence_v3"
    )
    runtime["zero_contract_v2"]["require_remove_null_exact_equivalence"] = True

    differences = recursive_diff(v2, runtime)
    allowed = {
        "run_id",
        "output.namespace",
        "output.record_file",
        "output.slurm_stdout",
        "output.slurm_stderr",
        "output.result_schema_version",
        "output.decision_schema.schema_version",
        "output.decision_schema.required_halt_only_validity_guards",
        "output.decision_schema.exact_array_comparison",
        "output.decision_schema.exact_metric_comparison",
        "zero_contract_v2.remove_null_comparison",
    }
    unexpected = sorted(set(differences) - allowed)
    if unexpected:
        base.die("v3 changed non-authorized v2 fields: {}".format(unexpected))
    scientific_diff = {
        "pass": True,
        "v2_config_path": scientific["config_path"],
        "v2_config_sha256": scientific["config_sha256"],
        "v2_analysis_path": scientific["analysis_path"],
        "v2_analysis_sha256": scientific["analysis_sha256"],
        "changed_paths": differences,
        "change_scope": (
            "run/schema/namespace/output identity plus HALT-only numerical "
            "equivalence names; all scientific/statistical fields remain exact"
        ),
        "scientific_thresholds_exact": True,
    }
    return runtime, scientific_diff


def load_lineage_evidence(v3):
    lineage = v3["lineage_evidence"]
    review_path = base.repo_path(
        lineage["guard_review_path"], "v3 guard review"
    )
    diagnostic_path = base.repo_path(
        lineage["diagnostic_path"], "retrieval diagnostic"
    )
    review_lines = review_path.read_text(encoding="utf-8").splitlines()
    if (
        len(review_lines) < lineage["guard_review_line_end"]
        or review_lines[lineage["guard_review_line_start"] - 1]
        != "# " + lineage["guard_review_heading"]
    ):
        base.die("v3 guard review range/heading drift")
    review_section = (
        "\n".join(
            review_lines[
                lineage["guard_review_line_start"] - 1
                : lineage["guard_review_line_end"]
            ]
        )
        + "\n"
    ).encode("utf-8")
    if (
        hashlib.sha256(review_section).hexdigest()
        != lineage["guard_review_section_sha256"]
    ):
        base.die("v3 guard review section SHA256 drift")
    if sha256_file(diagnostic_path) != lineage["diagnostic_sha256"]:
        base.die("retrieval diagnostic SHA256 drift")
    with diagnostic_path.open("r", encoding="utf-8") as handle:
        artifact = json.load(handle)
    v2_path = base.repo_path(
        v3["scientific_base"]["config_path"],
        "lineage frozen v2 config",
    )
    if sha256_file(v2_path) != v3["scientific_base"]["config_sha256"]:
        base.die("lineage frozen v2 config SHA256 drift")
    with v2_path.open("r", encoding="utf-8") as handle:
        v2_config = json.load(handle)
    authorized = lineage["authorized_null"]
    v2_zero = v2_config["zero_contract_v2"]
    v3_semantic_tuple = {
        "dataset": authorized["dataset"],
        "split": authorized["split"],
        "raw_id": authorized["raw_id"],
        "row_index": int(authorized["row_index"]),
        "policies": list(authorized["policies"]),
        "modalities": list(authorized["modalities"]),
        "expected_label_integrity_only": int(
            authorized["expected_label_integrity_only"]
        ),
    }
    v2_semantic_tuple = {
        "dataset": v2_zero["authorized_dataset"],
        "split": v2_zero["authorized_split"],
        "raw_id": v2_zero["authorized_id"],
        "row_index": int(v2_zero["authorized_row_index"]),
        "policies": list(v2_zero["authorized_policies"]),
        "modalities": list(v2_zero["authorized_modalities"]),
        "expected_label_integrity_only": int(
            v2_zero["expected_label_integrity_only"]
        ),
    }
    diagnostic_null = artifact.get("evidence", {}).get(
        "authorized_null", {}
    )
    diagnostic_tuple_checks = {
        "dataset": diagnostic_null.get("dataset")
        == v3_semantic_tuple["dataset"],
        "split": diagnostic_null.get("split")
        == v3_semantic_tuple["split"],
        "raw_id": diagnostic_null.get("id")
        == v3_semantic_tuple["raw_id"],
        "row_index": diagnostic_null.get("row_index")
        == v3_semantic_tuple["row_index"],
        "expected_label_integrity_only": diagnostic_null.get(
            "expected_label_integrity_only"
        )
        == v3_semantic_tuple["expected_label_integrity_only"],
        "modalities": diagnostic_null.get("modalities")
        == v3_semantic_tuple["modalities"],
        "diagnostic_policy_scope": (
            diagnostic_null.get("policy") == "standard"
            and diagnostic_null.get("policy")
            in v3_semantic_tuple["policies"]
        ),
    }
    thread_expected = v3["execution"]["required_environment"]
    checks = {
        "schema_version": (
            artifact.get("schema_version")
            == lineage["diagnostic_schema_version"]
        ),
        "run_id": artifact.get("run_id") == lineage["diagnostic_run_id"],
        "job_id": str(artifact.get("slurm_job_id"))
        == lineage["diagnostic_job_id"],
        "classification": artifact.get("diagnosis", {}).get("classification")
        == lineage["diagnostic_classification"],
        "manifest": artifact.get("evidence", {}).get("manifest_sha256")
        == lineage["manifest_sha256"],
        "zero_probe": artifact.get("evidence", {}).get("zero_probe_sha256")
        == lineage["zero_probe_sha256"],
        "a0_source": artifact.get(
            "endpoint_std_key_construction", {}
        ).get("source_sha256") == BASE_SOURCE_SHA256,
        "train_key_parity": artifact.get(
            "endpoint_std_key_construction", {}
        ).get("train_parity", {}).get("pass") is True,
        "dev_key_parity": artifact.get(
            "endpoint_std_key_construction", {}
        ).get("dev_seen_parity", {}).get("pass") is True,
        "mapping": artifact.get("index_mapping", {}).get(
            "formula_exact_for_all_indices"
        ) is True,
        "null_top20": artifact.get("registered_null_top20", {}).get(
            "occurrence_count"
        ) == 0,
        "raw_neighbors": artifact.get("raw_faiss_order", {}).get(
            "neighbors_original_vs_mapped", {}
        ).get("element_diff_count") == 0,
        "stable_neighbors": artifact.get(
            "stable_similarity_then_original_index_order", {}
        ).get("neighbors_original_vs_mapped", {}).get(
            "element_diff_count"
        ) == 0,
        "thread_environment": artifact.get("thread_environment")
        == thread_expected,
        "train_cache": artifact.get("inputs", {}).get("train", {}).get(
            "sha256"
        ) == lineage["diagnostic_train_cache_sha256"],
        "dev_cache": artifact.get("inputs", {}).get("dev_seen", {}).get(
            "sha256"
        ) == lineage["diagnostic_dev_seen_cache_sha256"],
        "authorized_null_equals_frozen_v2_tuple": (
            v3_semantic_tuple == v2_semantic_tuple
        ),
        "authorized_null_equals_diagnostic_fields": base.strict_all(
            diagnostic_tuple_checks.values(),
            "diagnostic_authorized_null_semantic_tuple",
        ),
    }
    if not base.strict_all(checks.values(), "v3_lineage_evidence"):
        base.die("v3 lineage diagnostic binding failed")
    return {
        "pass": True,
        "guard_review": {
            "path": lineage["guard_review_path"],
            "section_sha256": lineage["guard_review_section_sha256"],
            "heading": lineage["guard_review_heading"],
            "line_start": lineage["guard_review_line_start"],
            "line_end": lineage["guard_review_line_end"],
        },
        "diagnostic": {
            "path": lineage["diagnostic_path"],
            "sha256": lineage["diagnostic_sha256"],
            "schema_version": lineage["diagnostic_schema_version"],
            "run_id": lineage["diagnostic_run_id"],
            "slurm_job_id": lineage["diagnostic_job_id"],
            "classification": lineage["diagnostic_classification"],
        },
        "authorized_null_semantic_equality": {
            "v3_tuple": v3_semantic_tuple,
            "frozen_v2_tuple": v2_semantic_tuple,
            "v2_exact": v3_semantic_tuple == v2_semantic_tuple,
            "diagnostic_fields": diagnostic_tuple_checks,
            "diagnostic_policy_scope_note": (
                "the diagnostic was intentionally standard-only; its "
                "policy must equal standard and belong to the v2/v3 "
                "authorized {standard,oneword} policy set"
            ),
            "pass": True,
        },
        "checks": checks,
    }


def cpu_model_name():
    path = Path("/proc/cpuinfo")
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    base.die("cannot bind CPU model name")


def verify_software_environment(v3):
    expected = v3["execution"]
    versions = {
        "numpy": str(base.np.__version__),
        "torch": str(base.torch.__version__).split("+", 1)[0],
        "faiss": str(base.faiss.__version__),
    }
    cpu = {
        "architecture": platform.machine(),
        "byte_order": sys.byteorder,
        "model_name": cpu_model_name(),
    }
    if versions != expected["required_software"]:
        base.die("v3 software version binding drift: {}".format(versions))
    if cpu != expected["required_cpu"]:
        base.die("v3 CPU environment binding drift: {}".format(cpu))
    return {
        "pass": True,
        "versions": versions,
        "cpu": cpu,
        "thread_environment": {
            key: os.environ.get(key)
            for key in expected["required_environment"]
        },
        "required_cpus": expected["required_cpus"],
        "required_memory": expected["required_memory"],
    }


def ordered_float32(values, context):
    np = base.np
    values = np.ascontiguousarray(values)
    if values.dtype != np.dtype("float32"):
        halt(context, "ordered ULP mapping requires float32")
    if not np.isfinite(values).all():
        halt(context, "ordered ULP mapping forbids NaN/Inf")
    bits = values.view("uint32")
    negative = (bits & np.uint32(0x80000000)) != 0
    ordered = np.where(
        negative,
        np.bitwise_not(bits),
        np.bitwise_xor(bits, np.uint32(0x80000000)),
    )
    return ordered.astype("uint64")


def float32_ulp_self_check():
    np = base.np
    source = np.asarray(
        [
            0xbf800000,
            0x80000001,
            0x80000000,
            0x00000000,
            0x00000001,
            0x3f800000,
        ],
        dtype="uint32",
    )
    expected = np.asarray(
        [
            0x407fffff,
            0x7ffffffe,
            0x7fffffff,
            0x80000000,
            0x80000001,
            0xbf800000,
        ],
        dtype="uint64",
    )
    observed = ordered_float32(source.view("float32"), "ulp_self_check")
    forbidden = np.asarray(
        [0x7f800000, 0xff800000, 0x7fc00000], dtype="uint32"
    ).view("float32")
    checks = {
        "known_mapping_exact": bool(np.array_equal(observed, expected)),
        "strictly_monotone": bool(np.all(observed[:-1] < observed[1:])),
        "minus_zero_plus_zero_distance_one": int(
            observed[3] - observed[2]
        ) == 1,
        "cross_sign_min_subnormal_distance_three": int(
            observed[4] - observed[1]
        ) == 3,
        "nan_and_infinities_nonfinite": bool(
            not np.isfinite(forbidden).any()
        ),
    }
    if not base.strict_all(checks.values(), "float32_ulp_self_check"):
        halt("ulp_self_check", "known bit-pattern self-check failed")
    return {
        "pass": True,
        "mapping": (
            "negative=bitwise_not(raw_u32); "
            "nonnegative=raw_u32 xor 0x80000000"
        ),
        "signed_zero_policy": "-0 and +0 are distinct adjacent codes",
        "cross_sign_policy": "monotone distance crosses -0 then +0",
        "nonfinite_policy": "NaN and both infinities forbidden",
        "source_hex": ["{:08x}".format(int(value)) for value in source],
        "ordered_hex": ["{:08x}".format(int(value)) for value in observed],
        "checks": checks,
    }


def gamma(unit_roundoff, dimension, context):
    product = float(dimension) * float(unit_roundoff)
    if not math.isfinite(product) or not (0.0 < product < 1.0):
        halt(context, "dimension times unit roundoff must lie in (0,1)")
    value = product / (1.0 - product)
    if not math.isfinite(value) or value <= 0.0:
        halt(context, "nonfinite gamma")
    return value


def upward_norm_bounds(values, context):
    np = base.np
    values = np.ascontiguousarray(values)
    if values.dtype != np.dtype("float32") or values.ndim != 2:
        halt(context, "operand norm audit requires 2D float32")
    if not np.isfinite(values).all():
        halt(context, "operand norm audit forbids NaN/Inf")
    dimension = int(values.shape[1])
    gamma64 = gamma(U64, dimension, context + "/gamma64")
    promoted = values.astype("float64")
    squares = np.multiply(promoted, promoted)
    summed = np.add.reduce(squares, axis=1, dtype="float64")
    upper_square = np.nextafter(
        summed / (1.0 - gamma64), np.float64(np.inf)
    )
    bounds = np.nextafter(np.sqrt(upper_square), np.float64(np.inf))
    if not np.isfinite(bounds).all():
        halt(context, "nonfinite upward-safe norm bound")
    return bounds, gamma64


def next_power_of_two(value, context):
    if not math.isfinite(value) or value <= 0.0:
        halt(context, "next-power-of-two input must be positive finite")
    mantissa, exponent = math.frexp(value)
    bound_exponent = exponent - 1 if mantissa == 0.5 else exponent
    bound = math.ldexp(1.0, bound_exponent)
    if not math.isfinite(bound) or bound < value:
        halt(context, "next-power-of-two construction failed")
    return bound, bound_exponent


def derive_similarity_bound(memory, query, context):
    np = base.np
    if memory.shape[1] != query.shape[1]:
        halt(context, "memory/query dimension mismatch")
    dimension = int(memory.shape[1])
    gamma32 = gamma(U32, dimension, context + "/gamma32")
    memory_norms, gamma64_memory = upward_norm_bounds(
        memory, context + "/memory_norm"
    )
    query_norms, gamma64_query = upward_norm_bounds(
        query, context + "/query_norm"
    )
    rho = np.nextafter(
        np.float64(np.max(memory_norms))
        * np.float64(np.max(query_norms)),
        np.float64(np.inf),
    )
    raw = np.nextafter(
        np.float64(2.0) * np.float64(gamma32) * rho,
        np.float64(np.inf),
    )
    bound, exponent = next_power_of_two(
        float(raw), context + "/B_sim"
    )
    return {
        "dimension": dimension,
        "u32": U32,
        "dimension_times_u32": float(dimension * U32),
        "gamma32": gamma32,
        "gamma32_formula": "d*u32/(1-d*u32)",
        "gamma64_memory_norm_audit": gamma64_memory,
        "gamma64_query_norm_audit": gamma64_query,
        "max_memory_norm_upper": float(np.max(memory_norms)),
        "max_query_norm_upper": float(np.max(query_norms)),
        "rho_upper": float(rho),
        "raw_two_gamma_rho_upper": float(raw),
        "B_sim": bound,
        "B_sim_power_of_two_exponent": exponent,
        "B_sim_formula": (
            "2^ceil(log2(2*gamma32(actual_dimension)*rho_upper))"
        ),
        "operand_norm_audit": (
            "binary64 sum-of-squares corrected upward by gamma64, "
            "sqrt, then nextafter(+inf)"
        ),
    }


def outward_float32_interval(reference, bound, context):
    np = base.np
    reference = np.ascontiguousarray(reference)
    if reference.dtype != np.dtype("float32"):
        halt(context, "ULP interval reference must be float32")
    if not np.isfinite(reference).all() or not math.isfinite(bound):
        halt(context, "ULP interval inputs must be finite")
    limit = float(np.finfo("float32").max)
    ref64 = reference.astype("float64")
    exact_upper = np.clip(ref64 + float(bound), -limit, limit)
    exact_lower = np.clip(ref64 - float(bound), -limit, limit)
    upper = exact_upper.astype("float32")
    lower = exact_lower.astype("float32")
    upper_needs_step = upper.astype("float64") < exact_upper
    lower_needs_step = lower.astype("float64") > exact_lower
    upper_step = np.nextafter(upper, np.float32(np.inf))
    lower_step = np.nextafter(lower, np.float32(-np.inf))
    upper = np.where(upper_needs_step, upper_step, upper).astype("float32")
    lower = np.where(lower_needs_step, lower_step, lower).astype("float32")
    if not np.isfinite(upper).all() or not np.isfinite(lower).all():
        halt(context, "outward ULP interval reached nonfinite float32")
    return lower, upper


def similarity_numerical_audit(left, right, derivation, context):
    np = base.np
    left = np.ascontiguousarray(left)
    right = np.ascontiguousarray(right)
    if (
        left.dtype != np.dtype("float32")
        or right.dtype != np.dtype("float32")
        or left.shape != right.shape
    ):
        halt(context, "similarities require identical float32 shapes")
    if not np.isfinite(left).all() or not np.isfinite(right).all():
        halt(context, "similarities contain NaN/Inf")
    delta = np.abs(left.astype("float64") - right.astype("float64"))
    bound = float(derivation["B_sim"])
    absolute_pass = bool(np.all(delta <= bound))
    left_ordered = ordered_float32(left, context + "/left_ordered")
    right_ordered = ordered_float32(right, context + "/right_ordered")
    observed_ulp = np.maximum(left_ordered, right_ordered) - np.minimum(
        left_ordered, right_ordered
    )
    lower, upper = outward_float32_interval(
        left, bound, context + "/outward_interval"
    )
    lower_ordered = ordered_float32(lower, context + "/lower_ordered")
    upper_ordered = ordered_float32(upper, context + "/upper_ordered")
    allowed_ulp = np.maximum(
        upper_ordered - left_ordered,
        left_ordered - lower_ordered,
    )
    ulp_pass = bool(np.all(observed_ulp <= allowed_ulp))
    signed_zero_ambiguity = (
        (left == 0.0)
        & (right == 0.0)
        & (np.signbit(left) != np.signbit(right))
    )
    max_delta = float(np.max(delta))
    max_observed_ulp = int(np.max(observed_ulp))
    max_allowed_ulp = int(np.max(allowed_ulp))
    max_abs_ratio = float(np.max(delta / bound))
    ulp_ratio = np.divide(
        observed_ulp.astype("float64"),
        allowed_ulp.astype("float64"),
        out=np.zeros_like(observed_ulp, dtype="float64"),
        where=allowed_ulp != 0,
    )
    max_ulp_ratio = float(np.max(ulp_ratio))
    checks = {
        "finite_float32": True,
        "absolute_bound": absolute_pass,
        "exponent_aware_ulp_bound": ulp_pass,
        "signed_zero_ambiguity_absent": not bool(
            np.any(signed_zero_ambiguity)
        ),
    }
    if not base.strict_all(checks.values(), context + "/similarity_bounds"):
        halt(context, "similarity numerical-equivalence bound exceeded")
    return {
        "pass": True,
        "checks": checks,
        "derivation": derivation,
        "observed": {
            "element_count": int(delta.size),
            "different_float32_code_count": int(np.sum(observed_ulp != 0)),
            "max_absolute_difference": max_delta,
            "max_ordered_float32_ulp": max_observed_ulp,
            "max_allowed_ordered_float32_ulp": max_allowed_ulp,
            "max_absolute_bound_ratio": max_abs_ratio,
            "max_ulp_bound_ratio": max_ulp_ratio,
            "signed_zero_ambiguity_count": int(
                np.sum(signed_zero_ambiguity)
            ),
        },
        "ulp_bound": (
            "max(ord32(upper32(a+B))-ord32(a),"
            "ord32(a)-ord32(lower32(a-B)))"
        ),
        "outward_rounding": (
            "binary64 endpoints clipped only to finite float32 then "
            "float32 nextafter outward"
        ),
    }


def _search_frozen_operands(memory, memory_labels, query, config, context):
    np = base.np
    memory = np.ascontiguousarray(memory)
    query = np.ascontiguousarray(query)
    if (
        memory.dtype != np.dtype("float32")
        or query.dtype != np.dtype("float32")
        or memory.ndim != 2
        or query.ndim != 2
        or memory.shape[1] != query.shape[1]
        or not np.isfinite(memory).all()
        or not np.isfinite(query).all()
    ):
        halt(context, "frozen FAISS operands malformed/nonfinite")
    topk = int(config["retrieval"]["topk"])
    if topk != 20 or len(memory) < topk:
        halt(context, "frozen top-k contract changed")
    index = base.faiss.IndexFlatIP(memory.shape[1])
    index.add(memory)
    similarities, neighbors = index.search(query, topk)
    if (
        similarities.dtype != np.dtype("float32")
        or neighbors.shape != (len(query), topk)
        or np.any(neighbors < 0)
        or np.any(neighbors >= len(memory))
        or not np.isfinite(similarities).all()
    ):
        halt(context, "FAISS output malformed/nonfinite")
    weights = np.arange(topk, 0, -1, dtype="float64")
    if (
        weights.tolist()
        != ACTIVE_V3["numerical_equivalence_v3"]["weighted_vote"]["weights"]
        or float(np.sum(weights)) != 210.0
    ):
        halt(context, "frozen 20/210 weight contract changed")
    signed = memory_labels[neighbors].astype("float64") * 2.0 - 1.0
    if not np.all((signed == -1.0) | (signed == 1.0)):
        halt(context, "neighbor labels are not frozen signs")
    scores = np.add.reduce(
        signed * similarities.astype("float64") * weights[None, :],
        axis=1,
        dtype="float64",
    ) / 210.0
    if not np.isfinite(scores).all():
        halt(context, "weighted scores nonfinite")
    return {
        "scores": scores,
        "neighbors": neighbors.astype("int64", copy=False),
        "similarities": similarities.astype("float32", copy=False),
        "_normalized_memory": memory,
        "_normalized_query": query,
    }


def weighted_signed_scores_v3(memory, memory_labels, query, config):
    np = base.np
    normalized_memory = np.ascontiguousarray(
        memory.astype("float32", copy=True)
    )
    normalized_query = np.ascontiguousarray(
        query.astype("float32", copy=True)
    )
    if (
        not np.isfinite(normalized_memory).all()
        or not np.isfinite(normalized_query).all()
    ):
        halt("weighted_signed_scores", "raw keys contain NaN/Inf")
    base.faiss.normalize_L2(normalized_memory)
    base.faiss.normalize_L2(normalized_query)
    return _search_frozen_operands(
        normalized_memory,
        memory_labels,
        normalized_query,
        config,
        "weighted_signed_scores",
    )


def stable_neighbors(similarities, original_neighbors):
    np = base.np
    stable_sim = np.empty_like(similarities)
    stable_ids = np.empty_like(original_neighbors)
    for row in range(len(similarities)):
        order = np.lexsort(
            (original_neighbors[row], -similarities[row])
        )
        stable_sim[row] = similarities[row, order]
        stable_ids[row] = original_neighbors[row, order]
    return stable_sim, stable_ids


def query_neighbor_differences(left, right):
    np = base.np
    order = np.any(left != right, axis=1)
    set_difference = np.asarray(
        [
            set(left[row].tolist()) != set(right[row].tolist())
            for row in range(len(left))
        ],
        dtype=bool,
    )
    return {
        "element_difference_count": int(np.sum(left != right)),
        "query_order_difference_count": int(np.sum(order)),
        "query_set_difference_count": int(np.sum(set_difference)),
    }


def score_numerical_audit(
    left_scores,
    right_scores,
    similarity_delta,
    similarity_derivation,
    cutoff,
    context,
):
    np = base.np
    left_scores = np.ascontiguousarray(left_scores)
    right_scores = np.ascontiguousarray(right_scores)
    if (
        left_scores.dtype != np.dtype("float64")
        or right_scores.dtype != np.dtype("float64")
        or left_scores.shape != right_scores.shape
        or not np.isfinite(left_scores).all()
        or not np.isfinite(right_scores).all()
    ):
        halt(context, "score arrays malformed/nonfinite")
    weights = np.arange(20, 0, -1, dtype="float64")
    propagated = np.add.reduce(
        np.abs(similarity_delta.astype("float64"))
        * weights[None, :],
        axis=1,
        dtype="float64",
    ) / 210.0 + SCORE_ROUNDOFF
    score_delta = np.abs(left_scores - right_scores)
    per_query = bool(np.all(score_delta <= propagated))
    sanity_bound = float(similarity_derivation["B_sim"]) + SCORE_ROUNDOFF
    sanity = bool(float(np.max(score_delta)) <= sanity_bound)
    left_lower = left_scores - propagated
    left_upper = left_scores + propagated
    right_lower = right_scores - propagated
    right_upper = right_scores + propagated
    left_stable = (left_lower > cutoff) | (left_upper < cutoff)
    right_stable = (right_lower > cutoff) | (right_upper < cutoff)
    cutoff_stable = bool(np.all(left_stable & right_stable))
    signed_zero_ambiguity = (
        (left_scores == 0.0)
        & (right_scores == 0.0)
        & (np.signbit(left_scores) != np.signbit(right_scores))
    )
    checks = {
        "per_query_propagated_bound": per_query,
        "arm_level_B_sim_plus_2^-45": sanity,
        "cutoff_closed_intervals_do_not_cross_zero": cutoff_stable,
        "signed_zero_ambiguity_absent": not bool(
            np.any(signed_zero_ambiguity)
        ),
    }
    if not base.strict_all(checks.values(), context + "/score_bounds"):
        halt(context, "score/cutoff numerical-equivalence guard failed")
    ratios = np.divide(
        score_delta,
        propagated,
        out=np.zeros_like(score_delta),
        where=propagated != 0.0,
    )
    cutoff_margin = np.minimum(
        np.minimum(np.abs(left_lower - cutoff), np.abs(left_upper - cutoff)),
        np.minimum(np.abs(right_lower - cutoff), np.abs(right_upper - cutoff)),
    )
    return {
        "pass": True,
        "checks": checks,
        "derivation": {
            "topk": 20,
            "weights": list(range(20, 0, -1)),
            "weight_total": 210,
            "formula": (
                "(1/210)*sum_r(w_r*abs(delta_similarity_qr))+2^-45"
            ),
            "binary64_roundoff_allowance": SCORE_ROUNDOFF,
            "arm_sanity_bound": sanity_bound,
            "cutoff": cutoff,
            "cutoff_rule": (
                "closed interval around either score must not contain zero"
            ),
        },
        "observed": {
            "max_absolute_score_difference": float(np.max(score_delta)),
            "max_per_query_bound": float(np.max(propagated)),
            "max_per_query_bound_ratio": float(np.max(ratios)),
            "minimum_expanded_interval_distance_from_cutoff": float(
                np.min(cutoff_margin)
            ),
            "signed_zero_ambiguity_count": int(
                np.sum(signed_zero_ambiguity)
            ),
        },
        "_per_query_bound": propagated,
    }


def deterministic_binary64_reference(
    normalized_memory,
    normalized_query,
    original_neighbors,
    memory_labels,
    faiss_similarities,
    faiss_scores,
    similarity_derivation,
    cutoff,
    context,
):
    np = base.np
    n_query, topk = original_neighbors.shape
    if topk != 20:
        halt(context, "binary64 reference requires exact top20")
    reference_similarities = np.empty((n_query, topk), dtype="float64")
    for start in range(0, n_query, REFERENCE_QUERY_CHUNK):
        stop = min(n_query, start + REFERENCE_QUERY_CHUNK)
        selected = normalized_memory[
            original_neighbors[start:stop]
        ].astype("float64")
        query = normalized_query[start:stop].astype("float64")
        products = np.multiply(selected, query[:, None, :])
        reference_similarities[start:stop] = np.add.reduce(
            products, axis=2, dtype="float64"
        )
    signed = (
        memory_labels[original_neighbors].astype("float64") * 2.0 - 1.0
    )
    weights = np.arange(20, 0, -1, dtype="float64")
    reference_scores = np.add.reduce(
        signed * reference_similarities * weights[None, :],
        axis=1,
        dtype="float64",
    ) / 210.0
    if (
        not np.isfinite(reference_similarities).all()
        or not np.isfinite(reference_scores).all()
    ):
        halt(context, "binary64 reference produced NaN/Inf")
    faiss_reference_delta = np.abs(
        faiss_similarities.astype("float64") - reference_similarities
    )
    similarity_pass = bool(
        np.all(faiss_reference_delta <= similarity_derivation["B_sim"])
    )
    score_delta = np.abs(faiss_scores - reference_scores)
    score_pass = bool(
        np.all(
            score_delta
            <= float(similarity_derivation["B_sim"]) + SCORE_ROUNDOFF
        )
    )
    faiss_similarity_zero = faiss_similarities == 0.0
    reference_similarity_zero = reference_similarities == 0.0
    similarity_zero_positions_exact = bool(
        np.array_equal(
            faiss_similarity_zero, reference_similarity_zero
        )
    )
    similarity_zero_signbits_exact = bool(
        similarity_zero_positions_exact
        and np.array_equal(
            np.signbit(faiss_similarities[faiss_similarity_zero]),
            np.signbit(
                reference_similarities[reference_similarity_zero]
            ),
        )
    )
    faiss_score_zero = faiss_scores == 0.0
    reference_score_zero = reference_scores == 0.0
    score_zero_positions_exact = bool(
        np.array_equal(faiss_score_zero, reference_score_zero)
    )
    score_zero_signbits_exact = bool(
        score_zero_positions_exact
        and np.array_equal(
            np.signbit(faiss_scores[faiss_score_zero]),
            np.signbit(reference_scores[reference_score_zero]),
        )
    )
    faiss_bundle = base.metric_bundle(
        np.zeros(n_query, dtype="int64"), faiss_scores, cutoff
    )
    reference_bundle = base.metric_bundle(
        np.zeros(n_query, dtype="int64"), reference_scores, cutoff
    )
    prediction_audit = base.array_bytewise_audit(
        faiss_bundle["predictions"],
        reference_bundle["predictions"],
        context + "/faiss_reference_predictions",
    )
    checks = {
        "shared_reference_between_paths_byte_exact": True,
        "reference_neighbor_ids_are_exact_agreed_float32_ids": True,
        "faiss_reference_similarity_within_B_sim": similarity_pass,
        "faiss_reference_score_within_B_sim_plus_2^-45": score_pass,
        "faiss_reference_similarity_zero_positions_exact": (
            similarity_zero_positions_exact
        ),
        "faiss_reference_similarity_zero_signbits_exact": (
            similarity_zero_signbits_exact
        ),
        "faiss_reference_score_zero_positions_exact": (
            score_zero_positions_exact
        ),
        "faiss_reference_score_zero_signbits_exact": (
            score_zero_signbits_exact
        ),
        "faiss_reference_predictions_exact": prediction_audit["pass"],
    }
    if not base.strict_all(checks.values(), context + "/reference"):
        halt(context, "deterministic binary64 reference guard failed")
    sim_bytes = np.ascontiguousarray(reference_similarities).tobytes()
    score_bytes = np.ascontiguousarray(reference_scores).tobytes()
    return {
        "pass": True,
        "scope": (
            "re-score only exact agreed float32 top20 IDs; no binary64 "
            "neighbor mining or candidate replacement"
        ),
        "algorithm": (
            "query chunks of 8; C-order float64 multiply then "
            "numpy.add.reduce over feature axis in frozen rank order"
        ),
        "thread_contract": (
            "same 8CPU OMP/MKL/OPENBLAS/NUMEXPR=8 environment; "
            "no thread or algorithm sweep"
        ),
        "relationship_to_float32_neighbors": (
            "reference consumes, but never changes, exact raw/stable "
            "float32 neighbor identity/order"
        ),
        "checks": checks,
        "reference_similarity": {
            "dtype": reference_similarities.dtype.str,
            "shape": list(reference_similarities.shape),
            "sha256": hashlib.sha256(sim_bytes).hexdigest(),
            "with_remove_bytes_exact_by_single_shared_computation": True,
        },
        "reference_score": {
            "dtype": reference_scores.dtype.str,
            "shape": list(reference_scores.shape),
            "sha256": hashlib.sha256(score_bytes).hexdigest(),
            "with_remove_bytes_exact_by_single_shared_computation": True,
        },
        "faiss_relation": {
            "max_absolute_similarity_difference": float(
                np.max(faiss_reference_delta)
            ),
            "similarity_B_sim": float(similarity_derivation["B_sim"]),
            "max_similarity_bound_ratio": float(
                np.max(
                    faiss_reference_delta
                    / float(similarity_derivation["B_sim"])
                )
            ),
            "max_absolute_score_difference": float(np.max(score_delta)),
            "score_bound": (
                float(similarity_derivation["B_sim"])
                + SCORE_ROUNDOFF
            ),
            "similarity_zero_position_count": int(
                np.sum(faiss_similarity_zero)
            ),
            "similarity_zero_positions_exact": (
                similarity_zero_positions_exact
            ),
            "similarity_zero_signbits_exact": (
                similarity_zero_signbits_exact
            ),
            "score_zero_position_count": int(np.sum(faiss_score_zero)),
            "score_zero_positions_exact": score_zero_positions_exact,
            "score_zero_signbits_exact": score_zero_signbits_exact,
            "signed_zero_ambiguity_count": int(
                np.sum(
                    faiss_similarity_zero
                    & reference_similarity_zero
                    & (
                        np.signbit(faiss_similarities)
                        != np.signbit(reference_similarities)
                    )
                )
                + np.sum(
                    faiss_score_zero
                    & reference_score_zero
                    & (
                        np.signbit(faiss_scores)
                        != np.signbit(reference_scores)
                    )
                )
            ),
        },
        "_similarities": reference_similarities,
        "_scores": reference_scores,
    }


def public_reference(reference):
    return {
        key: value
        for key, value in reference.items()
        if not key.startswith("_")
    }


def retrieval_without_registered_null_v3(
    memory,
    memory_labels,
    query,
    full_retrieval,
    null_index,
    config,
    context,
):
    np = base.np
    if null_index is None:
        return {
            "status": "NO_REGISTERED_NULL",
            "pass": True,
            "numerical_equivalence_v3": "NOT_APPLICABLE",
        }, None
    if np.any(full_retrieval["neighbors"] == null_index):
        halt(context, "registered null entered top20")
    keep = np.ones(len(memory), dtype=bool)
    keep[null_index] = False
    original_indices = np.flatnonzero(keep).astype("int64", copy=False)
    expected_mapping = np.arange(len(memory) - 1, dtype="int64")
    expected_mapping[null_index:] += 1
    mapping_formula_audit = base.array_bytewise_audit(
        original_indices,
        expected_mapping,
        context + "/mapping_formula",
    )
    full_normalized_memory = full_retrieval["_normalized_memory"]
    full_normalized_query = full_retrieval["_normalized_query"]
    raw_retained = np.ascontiguousarray(
        memory[keep].astype("float32", copy=True)
    )
    reduced_raw = np.ascontiguousarray(
        memory[keep].astype("float32", copy=True)
    )
    raw_query = np.ascontiguousarray(query.astype("float32", copy=True))
    reduced_raw_query = np.ascontiguousarray(
        query.astype("float32", copy=True)
    )
    reduced_normalized_memory = np.ascontiguousarray(
        full_normalized_memory[keep]
    )
    reduced_normalized_query = np.ascontiguousarray(full_normalized_query)
    operand_audits = {
        "raw_retained_memory": base.array_bytewise_audit(
            raw_retained,
            reduced_raw,
            context + "/raw_retained_memory",
        ),
        "raw_query": base.array_bytewise_audit(
            raw_query, reduced_raw_query, context + "/raw_query"
        ),
        "normalized_retained_memory": base.array_bytewise_audit(
            np.ascontiguousarray(full_normalized_memory[keep]),
            reduced_normalized_memory,
            context + "/normalized_retained_memory",
        ),
        "normalized_query": base.array_bytewise_audit(
            full_normalized_query,
            reduced_normalized_query,
            context + "/normalized_query",
        ),
    }
    if not base.strict_all(
        (audit["pass"] for audit in operand_audits.values()),
        context + "/operand_parity",
    ):
        halt(context, "raw/normalized operand byte parity failed")
    reduced = _search_frozen_operands(
        reduced_normalized_memory,
        memory_labels[keep],
        reduced_normalized_query,
        config,
        context + "/remove_search",
    )
    mapped_neighbors = original_indices[reduced["neighbors"]]
    raw_neighbor_audit = base.array_bytewise_audit(
        full_retrieval["neighbors"],
        mapped_neighbors,
        context + "/raw_neighbors",
    )
    raw_differences = query_neighbor_differences(
        full_retrieval["neighbors"], mapped_neighbors
    )
    stable_full_sim, stable_full_ids = stable_neighbors(
        full_retrieval["similarities"], full_retrieval["neighbors"]
    )
    stable_remove_sim, stable_remove_ids = stable_neighbors(
        reduced["similarities"], mapped_neighbors
    )
    stable_neighbor_audit = base.array_bytewise_audit(
        stable_full_ids,
        stable_remove_ids,
        context + "/stable_neighbors",
    )
    stable_differences = query_neighbor_differences(
        stable_full_ids, stable_remove_ids
    )
    full_neighbor_labels = memory_labels[
        full_retrieval["neighbors"]
    ].astype("int64", copy=False)
    remove_neighbor_labels = memory_labels[
        mapped_neighbors
    ].astype("int64", copy=False)
    neighbor_label_audit = base.array_bytewise_audit(
        full_neighbor_labels,
        remove_neighbor_labels,
        context + "/neighbor_labels",
    )
    exact_checks = {
        "mapping_formula_exact": mapping_formula_audit["pass"],
        "raw_neighbor_dtype_shape_bytes_rank_exact": raw_neighbor_audit[
            "pass"
        ],
        "raw_per_query_set_and_order_exact": (
            max(raw_differences.values()) == 0
        ),
        "stable_neighbor_dtype_shape_bytes_rank_exact": (
            stable_neighbor_audit["pass"]
        ),
        "stable_per_query_set_and_order_exact": (
            max(stable_differences.values()) == 0
        ),
        "neighbor_labels_dtype_shape_bytes_exact": neighbor_label_audit[
            "pass"
        ],
        "registered_null_top20_count_zero": True,
    }
    if not base.strict_all(exact_checks.values(), context + "/exact_discrete"):
        halt(context, "exact mapping/neighbor/label invariant failed")
    derivation = derive_similarity_bound(
        reduced_normalized_memory,
        reduced_normalized_query,
        context + "/similarity_derivation",
    )
    similarity = similarity_numerical_audit(
        full_retrieval["similarities"],
        reduced["similarities"],
        derivation,
        context + "/similarities",
    )
    similarity_delta = (
        full_retrieval["similarities"].astype("float64")
        - reduced["similarities"].astype("float64")
    )
    cutoff = float(config["retrieval"]["prediction_cutoff"])
    score = score_numerical_audit(
        full_retrieval["scores"],
        reduced["scores"],
        similarity_delta,
        derivation,
        cutoff,
        context + "/scores",
    )
    reference = deterministic_binary64_reference(
        full_normalized_memory,
        full_normalized_query,
        full_retrieval["neighbors"],
        memory_labels,
        full_retrieval["similarities"],
        full_retrieval["scores"],
        derivation,
        cutoff,
        context + "/binary64_reference",
    )
    audit = {
        "status": "PASS",
        "pass": True,
        "halt_label_on_failure": HALT_LABEL,
        "registered_null_index": int(null_index),
        "registered_null_top20_count": 0,
        "operand_parity": operand_audits,
        "mapping": mapping_formula_audit,
        "raw_neighbors": {
            "audit": raw_neighbor_audit,
            "differences": raw_differences,
        },
        "stable_neighbors": {
            "rule": (
                "lexicographic (-float32 similarity, original train index)"
            ),
            "audit": stable_neighbor_audit,
            "differences": stable_differences,
        },
        "neighbor_labels": neighbor_label_audit,
        "exact_non_numerical_checks": exact_checks,
        "similarity_numerical_equivalence": similarity,
        "score_numerical_equivalence": {
            key: value
            for key, value in score.items()
            if not key.startswith("_")
        },
        "binary64_exact_neighbor_reference": public_reference(reference),
    }
    reduced["neighbors"] = mapped_neighbors
    reduced["_v3_guard"] = {
        "audit": audit,
        "B_sim": derivation["B_sim"],
        "per_query_score_bound": score["_per_query_bound"],
        "reference_scores": reference["_scores"],
        "reference_similarities": reference["_similarities"],
    }
    return audit, reduced


def scientific_basis_hash(predictions, metrics):
    pred = base.np.ascontiguousarray(predictions)
    typed = {}
    for key in sorted(metrics):
        value = metrics[key]
        typed[key] = (
            {"type": "none"}
            if value is None
            else {
                "type": "ieee754_binary64",
                "big_endian_hex": struct.pack(">d", float(value)).hex(),
            }
        )
    payload = (
        pred.dtype.str.encode("ascii")
        + json.dumps(list(pred.shape), separators=(",", ":")).encode("ascii")
        + pred.tobytes(order="C")
        + json.dumps(
            typed, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def evaluation_numerical_equivalence_v3(
    full, reduced, gold, cutoff, context
):
    if "_v3_guard" not in reduced:
        halt(context, "missing retrieval numerical guard")
    reduced_bundle = base.metric_bundle(gold, reduced["scores"], cutoff)
    predictions = base.array_bytewise_audit(
        full["predictions"],
        reduced_bundle["predictions"],
        context + "/predictions",
    )
    metrics = base.canonical_metric_audit(
        full["metrics"],
        reduced_bundle["metrics"],
        context + "/metrics",
    )
    guard = reduced["_v3_guard"]
    reference_bundle = base.metric_bundle(
        gold, guard["reference_scores"], cutoff
    )
    reference_predictions = base.array_bytewise_audit(
        full["predictions"],
        reference_bundle["predictions"],
        context + "/binary64_reference_predictions",
    )
    reference_metrics = base.canonical_metric_audit(
        full["metrics"],
        reference_bundle["metrics"],
        context + "/binary64_reference_metrics",
    )
    bound = guard["per_query_score_bound"]
    left_scores = full["scores"]
    right_scores = reduced["scores"]
    cutoff_stable = bool(
        base.np.all(
            (((left_scores - bound) > cutoff) | ((left_scores + bound) < cutoff))
            & (
                ((right_scores - bound) > cutoff)
                | ((right_scores + bound) < cutoff)
            )
        )
    )
    full_basis = scientific_basis_hash(
        full["predictions"], full["metrics"]
    )
    reduced_basis = scientific_basis_hash(
        reduced_bundle["predictions"], reduced_bundle["metrics"]
    )
    checks = {
        "predictions_dtype_shape_bytes_exact": predictions["pass"],
        "metrics_canonical_typed_bytes_exact": metrics["pass"],
        "binary64_reference_predictions_exact": reference_predictions[
            "pass"
        ],
        "binary64_reference_metrics_canonical_typed_bytes_exact": (
            reference_metrics["pass"]
        ),
        "cutoff_interval_stability": cutoff_stable,
        "scientific_boolean_basis_bytes_exact": (
            full_basis == reduced_basis
        ),
    }
    if not base.strict_all(checks.values(), context + "/evaluation_v3"):
        halt(context, "prediction/metric/scientific basis mismatch")
    return {
        "pass": True,
        "checks": checks,
        "comparisons": {
            "predictions": predictions,
            "metrics": metrics,
            "binary64_reference_predictions": reference_predictions,
            "binary64_reference_metrics": reference_metrics,
        },
        "scientific_boolean_basis": {
            "with_sha256": full_basis,
            "remove_sha256": reduced_basis,
            "exact": True,
            "proof_scope": (
                "accuracy/macro-F1 bootstrap, Holm, rotation, shuffle-p95, "
                "net-fix, gain and validity booleans are deterministic "
                "functions of exact predictions/canonical metrics and the "
                "SHA-frozen v2 scientific config"
            ),
        },
    }


def avg_score_equivalence(
    full,
    reduced_scores,
    source_reduced,
    dev_labels,
    cutoff,
    context,
):
    np = base.np
    source_delta = [
        np.abs(
            full_source["scores"] - reduced_source["scores"]
        )
        for full_source, reduced_source in source_reduced
    ]
    bound = 0.5 * (source_delta[0] + source_delta[1]) + SCORE_ROUNDOFF
    delta = np.abs(full["scores"] - reduced_scores)
    source_bsim = [
        reduced_source["_v3_guard"]["B_sim"]
        for _, reduced_source in source_reduced
    ]
    sanity_bound = 0.5 * sum(source_bsim) + SCORE_ROUNDOFF
    interval_stable = np.ones(len(delta), dtype=bool)
    for scores in (full["scores"], reduced_scores):
        interval_stable &= (
            ((scores - bound) > cutoff) | ((scores + bound) < cutoff)
        )
    score_checks = {
        "derived_source_bound": bool(np.all(delta <= bound)),
        "source_B_sim_sanity_bound": bool(np.max(delta) <= sanity_bound),
        "cutoff_closed_intervals_do_not_cross_zero": bool(
            np.all(interval_stable)
        ),
    }
    if not base.strict_all(score_checks.values(), context + "/avg_score"):
        halt(context, "average-score propagated bound failed")
    reference_scores = 0.5 * (
        source_reduced[0][1]["_v3_guard"]["reference_scores"]
        + source_reduced[1][1]["_v3_guard"]["reference_scores"]
    )
    reduced = {
        "scores": reduced_scores,
        "_v3_guard": {
            "per_query_score_bound": bound,
            "B_sim": sanity_bound,
            "reference_scores": reference_scores,
        },
    }
    evaluation = evaluation_numerical_equivalence_v3(
        full, reduced, dev_labels, cutoff, context
    )
    reference_bundle = base.metric_bundle(
        dev_labels, reference_scores, cutoff
    )
    reference_predictions = base.array_bytewise_audit(
        full["predictions"],
        reference_bundle["predictions"],
        context + "/reference_predictions",
    )
    reference_metrics = base.canonical_metric_audit(
        full["metrics"],
        reference_bundle["metrics"],
        context + "/reference_metrics",
    )
    if not reference_predictions["pass"] or not reference_metrics["pass"]:
        halt(context, "average-score binary64 reference mismatch")
    ratio = np.divide(
        delta,
        bound,
        out=np.zeros_like(delta),
        where=bound != 0.0,
    )
    return {
        "status": "PASS",
        "pass": True,
        "source_neighbor_arms": ["endpoint_std", "endpoint_ow"],
        "score_numerical_equivalence": {
            "pass": True,
            "derivation": {
                "formula": (
                    "0.5*(abs(delta_endpoint_std_score)+"
                    "abs(delta_endpoint_ow_score))+2^-45"
                ),
                "binary64_roundoff_allowance": SCORE_ROUNDOFF,
                "arm_sanity_bound": sanity_bound,
            },
            "checks": score_checks,
            "observed": {
                "max_absolute_score_difference": float(np.max(delta)),
                "max_per_query_bound": float(np.max(bound)),
                "max_per_query_bound_ratio": float(np.max(ratio)),
            },
        },
        "binary64_reference": {
            "scope": "average of the two source-arm fixed-neighbor references",
            "prediction_bytes_exact": reference_predictions["pass"],
            "canonical_metrics_exact": reference_metrics["pass"],
        },
        "evaluation_checks": evaluation,
    }


def evaluate_real_arms_v3(
    train_views,
    dev_views,
    train_labels,
    dev_labels,
    config,
    registered_null_index,
):
    cutoff = float(config["retrieval"]["prediction_cutoff"])
    evaluations = {}
    validity = {}
    reduced_by_arm = {}
    for arm in train_views:
        retrieval = weighted_signed_scores_v3(
            train_views[arm], train_labels, dev_views[arm], config
        )
        evaluations[arm] = base.metric_bundle(
            dev_labels, retrieval["scores"], cutoff
        )
        evaluations[arm]["neighbors"] = retrieval["neighbors"]
        evaluations[arm]["similarities"] = retrieval["similarities"]
        validity[arm], reduced = retrieval_without_registered_null_v3(
            train_views[arm],
            train_labels,
            dev_views[arm],
            retrieval,
            registered_null_index,
            config,
            "real/{}".format(arm),
        )
        if reduced is not None:
            validity[arm][
                "evaluation_checks"
            ] = evaluation_numerical_equivalence_v3(
                evaluations[arm],
                reduced,
                dev_labels,
                cutoff,
                "real/{}".format(arm),
            )
            reduced_by_arm[arm] = reduced
    average_scores = 0.5 * (
        evaluations["endpoint_std"]["scores"]
        + evaluations["endpoint_ow"]["scores"]
    )
    evaluations["avg_score"] = base.metric_bundle(
        dev_labels, average_scores, cutoff
    )
    if registered_null_index is None:
        validity["avg_score"] = {
            "status": "NO_REGISTERED_NULL",
            "pass": True,
            "numerical_equivalence_v3": "NOT_APPLICABLE",
        }
    else:
        reduced_average_scores = 0.5 * (
            reduced_by_arm["endpoint_std"]["scores"]
            + reduced_by_arm["endpoint_ow"]["scores"]
        )
        validity["avg_score"] = avg_score_equivalence(
            evaluations["avg_score"],
            reduced_average_scores,
            [
                (
                    evaluations["endpoint_std"],
                    reduced_by_arm["endpoint_std"],
                ),
                (
                    evaluations["endpoint_ow"],
                    reduced_by_arm["endpoint_ow"],
                ),
            ],
            dev_labels,
            cutoff,
            "real/avg_score",
        )
    reference = evaluations[
        config["retrieval"]["fix_break_reference"]
    ]["predictions"]
    public = {}
    for arm, value in evaluations.items():
        public[arm] = {
            "metrics": value["metrics"],
            "confusion": value["confusion"],
            "fix_break_vs_endpoint_std": base.fix_break(
                value["predictions"], reference, dev_labels
            ),
        }
    return evaluations, public, validity


def compact_shuffle_audit(draw, target, audit):
    similarity = audit["similarity_numerical_equivalence"]
    score = audit["score_numerical_equivalence"]
    reference = audit["binary64_exact_neighbor_reference"]
    return {
        "draw": draw,
        "target": target,
        "dimension": similarity["derivation"]["dimension"],
        "rho_upper": similarity["derivation"]["rho_upper"],
        "gamma32": similarity["derivation"]["gamma32"],
        "B_sim": similarity["derivation"]["B_sim"],
        "max_absolute_similarity_difference": similarity["observed"][
            "max_absolute_difference"
        ],
        "max_observed_ulp": similarity["observed"][
            "max_ordered_float32_ulp"
        ],
        "max_allowed_ulp": similarity["observed"][
            "max_allowed_ordered_float32_ulp"
        ],
        "max_absolute_bound_ratio": similarity["observed"][
            "max_absolute_bound_ratio"
        ],
        "max_ulp_bound_ratio": similarity["observed"][
            "max_ulp_bound_ratio"
        ],
        "max_score_bound_ratio": score["observed"][
            "max_per_query_bound_ratio"
        ],
        "minimum_cutoff_interval_distance": score["observed"][
            "minimum_expanded_interval_distance_from_cutoff"
        ],
        "reference_max_similarity_bound_ratio": reference[
            "faiss_relation"
        ]["max_similarity_bound_ratio"],
        "pass": True,
    }


def permutation_null_v3(
    dataset, caches, normed, real_evaluations, config, zero_masks
):
    np = base.np
    count = int(config["statistics"]["n_id_hash_permutations"])
    seed = int(config["statistics"]["seed"])
    targets = ("common_displacement", "displacement")
    null_values = {
        target: {
            metric: [] for metric in config["statistics"]["metrics"]
        }
        for target in targets
    }
    summaries = {target: [] for target in targets}
    permutation_digest = hashlib.sha256()
    validity_digest = hashlib.sha256()
    cutoff = float(config["retrieval"]["prediction_cutoff"])
    train_labels = caches["train"]["standard"]["labels"]
    dev_labels = caches["dev_seen"]["standard"]["labels"]
    registered_null_index = (
        int(config["zero_contract_v2"]["authorized_row_index"])
        if dataset == config["zero_contract_v2"]["authorized_dataset"]
        else None
    )
    train_fixed = (
        (registered_null_index,)
        if registered_null_index is not None
        else ()
    )
    validity_checks = 0
    null_top20_occurrences = 0
    numerical_mismatches = 0
    reference_mismatches = 0
    scientific_basis_mismatches = 0
    for draw in range(count):
        train_perm = base.id_hash_permutation(
            caches["train"]["oneword"]["ids"],
            dataset,
            "train",
            draw,
            seed,
            train_fixed,
        )
        dev_perm = base.id_hash_permutation(
            caches["dev_seen"]["oneword"]["ids"],
            dataset,
            "dev_seen",
            draw,
            seed,
        )
        permutation_digest.update(train_perm.tobytes())
        permutation_digest.update(dev_perm.tobytes())
        train_null = base.shuffled_contrast_views(
            normed["train"],
            normed["train"],
            train_perm,
            config,
            "{}/null/{}/train".format(dataset, draw),
            zero_masks["train"],
        )
        dev_null = base.shuffled_contrast_views(
            normed["dev_seen"],
            normed["dev_seen"],
            dev_perm,
            config,
            "{}/null/{}/dev".format(dataset, draw),
            zero_masks["dev_seen"],
        )
        for target in targets:
            retrieval = weighted_signed_scores_v3(
                train_null[target],
                train_labels,
                dev_null[target],
                config,
            )
            full = base.metric_bundle(
                dev_labels, retrieval["scores"], cutoff
            )
            audit, reduced = retrieval_without_registered_null_v3(
                train_null[target],
                train_labels,
                dev_null[target],
                retrieval,
                registered_null_index,
                config,
                "{}/shuffle/{}/{}".format(dataset, draw, target),
            )
            if reduced is not None:
                audit[
                    "evaluation_checks"
                ] = evaluation_numerical_equivalence_v3(
                    full,
                    reduced,
                    dev_labels,
                    cutoff,
                    "{}/shuffle/{}/{}".format(dataset, draw, target),
                )
                summaries[target].append(
                    compact_shuffle_audit(draw, target, audit)
                )
            null_top20_occurrences += int(
                audit.get("registered_null_top20_count", 0)
            )
            numerical_mismatches += int(audit.get("pass") is not True)
            reference_mismatches += int(
                not audit_has_reference(audit)
            )
            scientific_basis_mismatches += int(
                not audit_has_scientific_basis(audit)
            )
            validity_digest.update(base.json_payload(audit))
            validity_checks += 1
            for metric, value in full["metrics"].items():
                if value is None:
                    halt(
                        "{}/shuffle/{}/{}".format(
                            dataset, draw, target
                        ),
                        "permutation metric undefined",
                    )
                null_values[target][metric].append(value)
    public = {
        "n": count,
        "pairing": (
            "label-blind independent ID-hash ordering within each split; "
            "registered structural-null index fixed and excluded from the "
            "remaining-source bijection"
        ),
        "permutation_order_digest": permutation_digest.hexdigest(),
        "targets": {},
    }
    validity = {
        "status": "PASS",
        "pass": True,
        "registered_null_index": registered_null_index,
        "fixed_train_indices": list(train_fixed),
        "train_fixed_point_draws_checked": count,
        "train_bijection_draws_checked": count,
        "dev_bijection_draws_checked": count,
        "retrieval_arm_draw_checks": validity_checks,
        "registered_null_top20_occurrences": null_top20_occurrences,
        "with_null_remove_null_numerical_mismatches": (
            numerical_mismatches
        ),
        "binary64_reference_mismatches": reference_mismatches,
        "scientific_boolean_basis_mismatches": (
            scientific_basis_mismatches
        ),
        "audit_digest": validity_digest.hexdigest(),
        "per_draw_formula_derived_bound_audits": summaries,
    }
    summaries_for_holm = []
    for target in targets:
        public["targets"][target] = {}
        for metric, values in null_values[target].items():
            array = np.asarray(values, dtype="float64")
            observed = real_evaluations[target]["metrics"][metric]
            summary = {
                "observed": observed,
                "null_mean": float(np.mean(array)),
                "null_p05": float(np.quantile(array, 0.05)),
                "null_p50": float(np.quantile(array, 0.5)),
                "null_p95": float(np.quantile(array, 0.95)),
                "null_max": float(np.max(array)),
                "observed_above_p95": bool(
                    observed > np.quantile(array, 0.95)
                ),
                "one_sided_raw_p": float(
                    (1 + np.sum(array >= observed)) / (len(array) + 1)
                ),
            }
            public["targets"][target][metric] = summary
            if metric in config["statistics"]["holm_metrics"]:
                summaries_for_holm.append(
                    (
                        "{}/{}/{}".format(dataset, target, metric),
                        summary,
                    )
                )
    return public, summaries_for_holm, validity


def audit_has_reference(audit):
    if audit.get("status") == "NO_REGISTERED_NULL":
        return True
    if "binary64_exact_neighbor_reference" in audit:
        reference = audit["binary64_exact_neighbor_reference"]
    elif "binary64_reference" in audit:
        reference = audit["binary64_reference"]
    else:
        return False
    evaluation = audit.get("evaluation_checks")
    evaluation_reference = bool(
        evaluation
        and evaluation.get("checks", {}).get(
            "binary64_reference_predictions_exact"
        )
        is True
        and evaluation.get("checks", {}).get(
            "binary64_reference_metrics_canonical_typed_bytes_exact"
        )
        is True
    )
    return bool(reference.get("pass", True) and evaluation_reference)


def audit_has_scientific_basis(audit):
    if audit.get("status") == "NO_REGISTERED_NULL":
        return True
    evaluation = audit.get("evaluation_checks")
    return bool(
        evaluation
        and evaluation.get("checks", {}).get(
            "scientific_boolean_basis_bytes_exact"
        )
        is True
    )


def derive_public_contract_guards(public, runtime_state, dataset, config):
    np = base.np
    previous = public["contract_guards"]
    caches = runtime_state["caches"]
    expected_runtime_keys = {
        "ids",
        "img",
        "text",
        "labels",
        "path",
        "sha256",
        "bytes",
    }
    exact_keys = all(
        set(caches[split][policy]) == expected_runtime_keys
        for split in base.REQUIRED_SPLITS
        for policy in ("standard", "oneword")
    )
    shape_finite_binary = True
    alignment = True
    unique = True
    for split in base.REQUIRED_SPLITS:
        expected_n = int(
            config["inputs"]["datasets"][dataset]["expected"][split]["n"]
        )
        dimension = int(config["inputs"]["feature_dim"])
        standard = caches[split]["standard"]
        oneword = caches[split]["oneword"]
        alignment = bool(
            alignment
            and standard["ids"] == oneword["ids"]
            and np.array_equal(standard["labels"], oneword["labels"])
        )
        for policy in ("standard", "oneword"):
            cache = caches[split][policy]
            labels = cache["labels"]
            shape_finite_binary = bool(
                shape_finite_binary
                and cache["img"].shape == (expected_n, dimension)
                and cache["text"].shape == (expected_n, dimension)
                and labels.shape == (expected_n,)
                and cache["img"].dtype == np.dtype("float32")
                and cache["text"].dtype == np.dtype("float32")
                and labels.dtype == np.dtype("int64")
                and np.isfinite(cache["img"]).all()
                and np.isfinite(cache["text"]).all()
                and np.isfinite(labels).all()
                and set(labels.tolist()).issubset({0, 1})
            )
            unique = bool(
                unique
                and len(cache["ids"]) == len(set(cache["ids"]))
                and all(
                    type(identifier) is str and identifier
                    for identifier in cache["ids"]
                )
            )
    train_dev_disjoint = not bool(
        set(caches["train"]["standard"]["ids"])
        & set(caches["dev_seen"]["standard"]["ids"])
    )
    raw_audit = previous["raw_zero_contract"]
    raw_zero_exact = True
    for split in base.REQUIRED_SPLITS:
        expected_indices = (
            [int(config["zero_contract_v2"]["authorized_row_index"])]
            if (
                dataset
                == config["zero_contract_v2"]["authorized_dataset"]
                and split
                == config["zero_contract_v2"]["authorized_split"]
            )
            else []
        )
        for policy in ("standard", "oneword"):
            for modality in ("img", "text"):
                cell = raw_audit[split][policy][modality]
                raw_zero_exact = bool(
                    raw_zero_exact
                    and cell["exact_zero_indices"] == expected_indices
                    and cell["tiny_nonzero_count"] == 0
                    and cell["nonfinite_count"] == 0
                )
    derived = previous["derived_zero_mask_preservation"]
    derived_exact = bool(
        base.strict_all(
            (
                value
                for split in base.REQUIRED_SPLITS
                for value in derived[split].values()
            ),
            dataset + "/public_derived_aggregate",
        )
    )
    displacement_exact = bool(
        public["displacement_norm_audit"][
            "dual_path_null_exclusion_audit"
        ]["pass"]
    )
    retrieval = previous["retrieval_null_influence"]
    shuffle = previous["shuffle_null_validity"]
    retrieval_pass = bool(
        base.strict_all(
            (audit.get("pass") is True for audit in retrieval.values()),
            dataset + "/public_retrieval_aggregate",
        )
    )
    retrieval_null_zero = all(
        audit.get("status") == "NO_REGISTERED_NULL"
        or audit.get("registered_null_top20_count") == 0
        for audit in retrieval.values()
    )
    registered_null_absent = bool(
        retrieval_pass
        and retrieval_null_zero
        and shuffle["registered_null_top20_occurrences"] == 0
    )
    numerical_equivalence = bool(
        retrieval_pass
        and shuffle["with_null_remove_null_numerical_mismatches"] == 0
    )
    reference_exact = bool(
        all(audit_has_reference(audit) for audit in retrieval.values())
        and shuffle["binary64_reference_mismatches"] == 0
    )
    scientific_basis_exact = bool(
        all(
            audit_has_scientific_basis(audit)
            for audit in retrieval.values()
        )
        and shuffle["scientific_boolean_basis_mismatches"] == 0
    )
    expected_draws = int(config["statistics"]["n_id_hash_permutations"])
    expected_fixed = (
        [int(config["zero_contract_v2"]["authorized_row_index"])]
        if dataset == config["zero_contract_v2"]["authorized_dataset"]
        else []
    )
    shuffle_exact = bool(
        shuffle["pass"] is True
        and shuffle["fixed_train_indices"] == expected_fixed
        and shuffle["train_fixed_point_draws_checked"] == expected_draws
        and shuffle["train_bijection_draws_checked"] == expected_draws
        and shuffle["dev_bijection_draws_checked"] == expected_draws
        and shuffle["retrieval_arm_draw_checks"] == 2 * expected_draws
    )
    access = public["runtime_access_ledger"]
    cache_access_count = len(access)
    test_paths_opened = sum(
        1
        for entry in access
        if entry["test_like"]
        and (entry["hash_opened"] or entry["torch_loaded"])
    )
    full_sha_exact = bool(
        cache_access_count == 4
        and all(
            entry["manifest_sha256_matched_before_torch_load"]
            and entry["torch_loaded"]
            for entry in access
        )
    )
    return {
        "exact_keys": bool(exact_keys),
        "shape_finite_binary": bool(shape_finite_binary),
        "standard_oneword_id_label_alignment": bool(alignment),
        "unique_ids": bool(unique),
        "train_dev_disjoint": bool(train_dev_disjoint),
        "raw_zero_allowlist_exact": bool(raw_zero_exact),
        "derived_zero_masks_exact": derived_exact,
        "displacement_null_exclusion_dual_path_exact": (
            displacement_exact
        ),
        "registered_null_absent_from_all_top20": (
            registered_null_absent
        ),
        "with_null_remove_null_numerical_equivalence_v3": (
            numerical_equivalence
        ),
        "deterministic_binary64_reference": reference_exact,
        "scientific_boolean_basis_exact": scientific_basis_exact,
        "shuffle_fixed_point_bijection": shuffle_exact,
        "cache_access_count": cache_access_count,
        "test_paths_opened": test_paths_opened,
        "full_sha256_matched_before_every_torch_load": full_sha_exact,
        "feature_dimensions": previous["feature_dimensions"],
        "raw_zero_contract": raw_audit,
        "derived_zero_mask_preservation": derived,
        "retrieval_null_influence": retrieval,
        "shuffle_null_validity": shuffle,
        "derivation": (
            "every public guard scalar is recomputed from runtime caches, "
            "audits, counters, byte comparisons, and access-ledger fields"
        ),
    }


def build_halt_only_validity_guards_v3(results, evidence, config):
    required = config["output"]["decision_schema"][
        "required_halt_only_validity_guards"
    ]
    datasets = {}
    for dataset in base.REQUIRED_DATASETS:
        contract = results["datasets"][dataset]["contract_guards"]
        public_boolean_keys = [
            "exact_keys",
            "shape_finite_binary",
            "standard_oneword_id_label_alignment",
            "unique_ids",
            "train_dev_disjoint",
            "raw_zero_allowlist_exact",
            "derived_zero_masks_exact",
            "displacement_null_exclusion_dual_path_exact",
            "registered_null_absent_from_all_top20",
            "with_null_remove_null_numerical_equivalence_v3",
            "deterministic_binary64_reference",
            "scientific_boolean_basis_exact",
            "shuffle_fixed_point_bijection",
            "full_sha256_matched_before_every_torch_load",
        ]
        if not base.strict_all(
            (
                type(contract[key]) is bool and contract[key]
                for key in public_boolean_keys
            ),
            dataset + "/public_contract_summary",
        ):
            halt(dataset, "derived public contract summary failed")
        derived = contract["derived_zero_mask_preservation"]
        retrieval = contract["retrieval_null_influence"]
        shuffle = contract["shuffle_null_validity"]
        retrieval_pass = base.strict_all(
            (audit["pass"] is True for audit in retrieval.values()),
            dataset + "/retrieval_v3",
        )
        checks = {
            "raw_zero_allowlist_exact": (
                contract["raw_zero_allowlist_exact"] is True
            ),
            "derived_zero_masks_exact": bool(
                base.strict_all(
                    (
                        flag
                        for split in base.REQUIRED_SPLITS
                        for flag in derived[split].values()
                    ),
                    dataset + "/derived_masks_v3",
                )
            ),
            "displacement_null_exclusion_dual_path_exact": (
                contract[
                    "displacement_null_exclusion_dual_path_exact"
                ]
                is True
            ),
            "shuffle_fixed_point_bijection": bool(
                shuffle["pass"] is True
                and shuffle["train_bijection_draws_checked"]
                == config["statistics"]["n_id_hash_permutations"]
                and shuffle["dev_bijection_draws_checked"]
                == config["statistics"]["n_id_hash_permutations"]
            ),
            "registered_null_absent_from_all_top20": bool(
                retrieval_pass
                and shuffle["registered_null_top20_occurrences"] == 0
            ),
            "with_null_remove_null_numerical_equivalence_v3": bool(
                retrieval_pass
                and shuffle[
                    "with_null_remove_null_numerical_mismatches"
                ]
                == 0
            ),
            "deterministic_binary64_reference": bool(
                all(audit_has_reference(audit) for audit in retrieval.values())
                and shuffle["binary64_reference_mismatches"] == 0
            ),
            "scientific_boolean_basis_exact": bool(
                all(
                    audit_has_scientific_basis(audit)
                    for audit in retrieval.values()
                )
                and shuffle["scientific_boolean_basis_mismatches"] == 0
            ),
        }
        if not base.strict_all(
            checks.values(), dataset + "/halt_only_v3"
        ):
            halt(dataset, "aggregate v3 validity guard failed")
        datasets[dataset] = checks
    global_checks = {
        "probe_evidence_exact": bool(
            evidence["sha256"]
            == config["zero_contract_v2"]["probe_sha256"]
        ),
        "v3_lineage_evidence_exact": bool(
            LINEAGE_AUDIT and LINEAGE_AUDIT["pass"]
        ),
        "raw_zero_allowlist_exact": all(
            item["raw_zero_allowlist_exact"] for item in datasets.values()
        ),
        "derived_zero_masks_exact": all(
            item["derived_zero_masks_exact"] for item in datasets.values()
        ),
        "displacement_null_exclusion_dual_path_exact": all(
            item["displacement_null_exclusion_dual_path_exact"]
            for item in datasets.values()
        ),
        "shuffle_fixed_point_bijection": all(
            item["shuffle_fixed_point_bijection"]
            for item in datasets.values()
        ),
        "registered_null_absent_from_all_top20": all(
            item["registered_null_absent_from_all_top20"]
            for item in datasets.values()
        ),
        "with_null_remove_null_numerical_equivalence_v3": all(
            item["with_null_remove_null_numerical_equivalence_v3"]
            for item in datasets.values()
        ),
        "deterministic_binary64_reference": all(
            item["deterministic_binary64_reference"]
            for item in datasets.values()
        ),
        "scientific_boolean_basis_exact": all(
            item["scientific_boolean_basis_exact"]
            for item in datasets.values()
        ),
    }
    if list(global_checks) != required:
        base.die("v3 halt-only validity guard order/binding changed")
    if not base.strict_all(
        global_checks.values(), "global_halt_only_v3"
    ):
        halt("global", "v3 halt-only validity guard failed")
    return {
        "halt_only": True,
        "halt_label_on_failure": HALT_LABEL,
        "passed": True,
        "required": required,
        "checks": global_checks,
        "datasets": datasets,
    }


def validate_decision_v3(decision, config):
    expected = {
        "schema_version",
        "experiment_id",
        "run_id",
        "config_sha256",
        "scientific_base",
        "full_sha256_manifest",
        "zero_contract_v2_evidence",
        "v3_lineage_evidence",
        "numerical_equivalence_v3_contract",
        "halt_only_validity_guards",
        "result_file",
        "result_sha256",
        "decision_label",
        "continue",
        "dataset_pass",
        "small_displacement_gate_reference",
        "small_displacement_gate_selection_rule",
        "endpoint_concat_small_displacement_role",
        "interpretation_scope",
        "exclusive_create",
    }
    if set(decision) != expected:
        base.die("v3 decision artifact schema keys changed")
    schema = config["output"]["decision_schema"]
    if (
        decision["schema_version"] != schema["schema_version"]
        or schema["schema_version"] != "c01_a0_decision_v3"
        or config["output"]["result_schema_version"]
        != "c01_a0_result_v3"
        or decision["experiment_id"] != "C01_A0"
        or decision["run_id"] != "C01-A0-v3"
        or decision["run_id"] != config["run_id"]
        or decision["config_sha256"] != V3_CONFIG_SHA256
    ):
        base.die("v3 decision run/config/schema provenance changed")
    scientific = decision["scientific_base"]
    scientific_allowed_changes = {
        "run_id",
        "output.namespace",
        "output.record_file",
        "output.slurm_stdout",
        "output.slurm_stderr",
        "output.result_schema_version",
        "output.decision_schema.schema_version",
        "output.decision_schema.required_halt_only_validity_guards",
        "output.decision_schema.exact_array_comparison",
        "output.decision_schema.exact_metric_comparison",
        "zero_contract_v2.remove_null_comparison",
    }
    if (
        set(scientific)
        != {
            "pass",
            "v2_config_path",
            "v2_config_sha256",
            "v2_analysis_path",
            "v2_analysis_sha256",
            "changed_paths",
            "change_scope",
            "scientific_thresholds_exact",
        }
        or scientific["pass"] is not True
        or scientific["scientific_thresholds_exact"] is not True
        or scientific["v2_config_path"]
        != ACTIVE_V3["scientific_base"]["config_path"]
        or scientific["v2_config_sha256"]
        != ACTIVE_V3["scientific_base"]["config_sha256"]
        or scientific["v2_analysis_path"]
        != ACTIVE_V3["scientific_base"]["analysis_path"]
        or scientific["v2_analysis_sha256"]
        != ACTIVE_V3["scientific_base"]["analysis_sha256"]
        or not set(scientific["changed_paths"]).issubset(
            scientific_allowed_changes
        )
        or not scientific["changed_paths"]
    ):
        base.die("v3 decision scientific-base/diff binding changed")
    manifest = decision["full_sha256_manifest"]
    if (
        set(manifest) != {"path", "sha256", "run_id"}
        or manifest["path"]
        != config["full_sha256_preflight"]["manifest_path"]
        or manifest["sha256"]
        != config["full_sha256_preflight"][
            "approved_manifest_sha256"
        ]
        or manifest["run_id"]
        != config["full_sha256_preflight"]["run_id"]
    ):
        base.die("v3 decision full-manifest provenance changed")
    evidence = decision["zero_contract_v2_evidence"]
    if (
        evidence["path"] != config["zero_contract_v2"]["probe_path"]
        or evidence["sha256"]
        != config["zero_contract_v2"]["probe_sha256"]
        or evidence["run_id"]
        != config["zero_contract_v2"]["probe_run_id"]
        or evidence["schema_version"]
        != config["zero_contract_v2"]["probe_schema_version"]
    ):
        base.die("v3 decision zero-contract provenance changed")
    if (
        LINEAGE_AUDIT is None
        or decision["v3_lineage_evidence"] != LINEAGE_AUDIT
        or decision["v3_lineage_evidence"].get("pass") is not True
        or decision["v3_lineage_evidence"].get(
            "authorized_null_semantic_equality", {}
        ).get("pass")
        is not True
    ):
        base.die("v3 decision lineage/authorized-null binding changed")
    numerical = decision["numerical_equivalence_v3_contract"]
    expected_numerical = ACTIVE_V3["numerical_equivalence_v3"]
    if (
        numerical != expected_numerical
        or numerical.get("required") is not True
        or numerical.get("halt_label") != HALT_LABEL
        or numerical.get("all_failures_before_publication") is not True
        or numerical.get("float32", {}).get("unit_roundoff_formula")
        != "2^-24"
        or numerical.get("weighted_vote", {}).get("topk") != 20
        or numerical.get("weighted_vote", {}).get("weight_total") != 210
        or numerical.get("weighted_vote", {}).get(
            "binary64_roundoff_allowance_formula"
        )
        != "2^-45"
        or numerical.get("binary64_reference", {}).get("required")
        is not True
    ):
        base.die("v3 decision numerical contract binding changed")
    validity = decision["halt_only_validity_guards"]
    if (
        validity.get("passed") is not True
        or validity.get("halt_only") is not True
        or validity.get("halt_label_on_failure") != HALT_LABEL
        or validity.get("required")
        != schema["required_halt_only_validity_guards"]
        or list(validity.get("checks", {}))
        != schema["required_halt_only_validity_guards"]
        or not base.strict_all(
            validity["checks"].values(), "decision_v3_validity"
        )
    ):
        base.die("v3 decision validity binding failed")
    if decision["decision_label"] not in schema["allowed_decisions"]:
        base.die("v3 decision label outside unchanged scientific enum")
    if (
        list(decision["dataset_pass"])
        != schema["required_dataset_keys"]
        or schema["required_dataset_keys"] != list(base.REQUIRED_DATASETS)
    ):
        base.die("v3 decision dataset key/order changed")
    if not base.strict_all(
        (
            type(value) is bool
            for value in decision["dataset_pass"].values()
        ),
        "v3_decision_dataset_pass_types",
    ):
        base.die("v3 dataset pass values must be booleans")
    small_reference = decision["small_displacement_gate_reference"]
    if list(small_reference) != schema["required_dataset_keys"]:
        base.die("v3 small-displacement dataset key/order changed")
    if not base.strict_all(
        (
            reference in config["decision"]["gain_controls"]
            for reference in small_reference.values()
        ),
        "v3_decision_small_displacement_references",
    ):
        base.die("v3 small-displacement reference is not an ordinary control")
    if (
        decision["small_displacement_gate_selection_rule"]
        != schema["small_displacement_gate_selection_rule"]
        or decision["small_displacement_gate_selection_rule"]
        != (
            "strongest_ordinary_control_by_accuracy_then_macro_f1_"
            "then_frozen_gain_controls_order"
        )
    ):
        base.die("v3 small-displacement selection rule changed")
    if (
        decision["endpoint_concat_small_displacement_role"]
        != schema["endpoint_concat_small_displacement_role"]
        or decision["endpoint_concat_small_displacement_role"]
        != "diagnostic_only"
    ):
        base.die("v3 endpoint-concat small-displacement role changed")
    if type(decision["continue"]) is not bool:
        base.die("v3 decision continue must be a boolean")
    expected_label = (
        config["decision"]["continue_label"]
        if decision["continue"]
        else config["decision"]["kill_label"]
    )
    if decision["decision_label"] != expected_label:
        base.die("v3 decision boolean/label mismatch")
    expected_interpretation = (
        config["positive_scope"]
        if decision["continue"]
        else config["negative_scope"]
    )
    if decision["interpretation_scope"] != expected_interpretation:
        base.die("v3 decision interpretation scope changed")
    result_sha = decision["result_sha256"]
    if (
        decision["result_file"] != config["output"]["result_file"]
        or not isinstance(result_sha, str)
        or len(result_sha) != 64
        or any(character not in "0123456789abcdef" for character in result_sha)
    ):
        base.die("v3 decision result-file/SHA256 provenance changed")
    if decision["exclusive_create"] is not True:
        base.die("v3 exclusive-create weakened")


def install_v3_overrides():
    base.weighted_signed_scores = weighted_signed_scores_v3
    base.retrieval_without_registered_null = (
        retrieval_without_registered_null_v3
    )
    base.evaluation_exact_equivalence = (
        evaluation_numerical_equivalence_v3
    )
    base.evaluate_real_arms = evaluate_real_arms_v3
    base.permutation_null = permutation_null_v3
    base.build_halt_only_validity_guards = (
        build_halt_only_validity_guards_v3
    )


def main():
    global ACTIVE_V3, LINEAGE_AUDIT, SOFTWARE_AUDIT, ULP_SELF_CHECK
    args = parse_args()
    ACTIVE_V3, v3_path, v3_digest = load_v3_config(args.config)
    runtime, scientific_diff = build_runtime_config(ACTIVE_V3)
    LINEAGE_AUDIT = load_lineage_evidence(ACTIVE_V3)
    base.enforce_runtime(runtime)
    manifest = base.load_full_sha256_manifest(runtime)
    zero_evidence = base.load_zero_contract_evidence(runtime)
    base.import_compute_modules(runtime)
    SOFTWARE_AUDIT = verify_software_environment(ACTIVE_V3)
    ULP_SELF_CHECK = float32_ulp_self_check()
    install_v3_overrides()
    base.INPUT_ACCESS_LEDGER.clear()

    output = runtime["output"]
    namespace = base.repo_path(output["namespace"], "output.namespace")
    expected_root = (
        REPO / "artifacts" / "c01_policy_contrastive" / "v3" / "a0"
    ).resolve()
    try:
        namespace.relative_to(expected_root)
    except ValueError:
        base.die("v3 namespace escapes exclusive v3/a0 root")
    result_path = namespace / output["result_file"]
    decision_path = namespace / output["decision_file"]
    if namespace.exists():
        base.die("v3 run namespace already exists; no-clobber")
    namespace.parent.mkdir(parents=True, exist_ok=True)
    namespace.mkdir()

    results = {
        "schema_version": output["result_schema_version"],
        "experiment_id": runtime["experiment_id"],
        "run_id": runtime["run_id"],
        "claim_scope": runtime["claim_scope"],
        "config_path": str(v3_path.relative_to(REPO)),
        "config_sha256": v3_digest,
        "scientific_base": scientific_diff,
        "full_sha256_preflight": manifest,
        "zero_contract_v2_evidence": zero_evidence,
        "v3_lineage_evidence": LINEAGE_AUDIT,
        "numerical_equivalence_v3_contract": ACTIVE_V3[
            "numerical_equivalence_v3"
        ],
        "float32_ordered_ulp_self_check": ULP_SELF_CHECK,
        "runtime_guards": {
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "slurm_cpus_per_task": int(
                os.environ["SLURM_CPUS_PER_TASK"]
            ),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "thread_environment": runtime["execution"][
                "required_environment"
            ],
            "cpu_only": True,
            "software_cpu_binding": SOFTWARE_AUDIT,
        },
        "method_boundary": {
            "paired_endpoints": (
                "standard-L24 prefix/response mean versus one-word-L24 "
                "last-token"
            ),
            "confounded_axes": ["prompt", "pooling/token readout"],
            "algebra": (
                "v3 changes no scientific transform, arm, statistic, "
                "threshold or decision rule from SHA-frozen v2"
            ),
            "positive_scope": runtime["positive_scope"],
            "negative_scope": runtime["negative_scope"],
        },
        "datasets": {},
    }
    runtimes = {}
    for dataset in base.REQUIRED_DATASETS:
        public, runtime_state = base.analyse_dataset(
            dataset, runtime, manifest["files"]
        )
        public["contract_guards"] = derive_public_contract_guards(
            public, runtime_state, dataset, runtime
        )
        results["datasets"][dataset] = public
        runtimes[dataset] = runtime_state
    expected_access = base.CANONICAL_BINDING["manifest"][
        "expected_file_count"
    ]
    if len(base.INPUT_ACCESS_LEDGER) != expected_access:
        base.die("v3 runtime access ledger is incomplete")
    test_attempts = sum(
        1 for entry in base.INPUT_ACCESS_LEDGER if entry["test_like"]
    )
    test_opens = sum(
        1
        for entry in base.INPUT_ACCESS_LEDGER
        if entry["test_like"]
        and (entry["hash_opened"] or entry["torch_loaded"])
    )
    all_manifest = base.strict_all(
        (
            entry["manifest_sha256_matched_before_torch_load"]
            and entry["torch_loaded"]
            for entry in base.INPUT_ACCESS_LEDGER
        ),
        "v3_global_cache_ledger",
    )
    if test_attempts or test_opens or not all_manifest:
        base.die("v3 global runtime cache access guard failed")
    results["runtime_guards"]["cache_access"] = {
        "expected_count": expected_access,
        "actual_count": len(base.INPUT_ACCESS_LEDGER),
        "test_like_attempt_count": test_attempts,
        "test_like_open_count": test_opens,
        "full_sha256_matched_before_every_torch_load": all_manifest,
        "ledger": base.INPUT_ACCESS_LEDGER,
    }
    results[
        "halt_only_validity_guards"
    ] = build_halt_only_validity_guards_v3(
        results, zero_evidence, runtime
    )
    base.attach_bootstrap_and_holm(results, runtimes, runtime)
    results["decision"] = base.make_decision(
        results, runtimes, runtime
    )

    result_bytes = base.json_payload(results)
    if len(result_bytes) > int(output["maximum_result_bytes"]):
        base.die("v3 result JSON exceeds configured maximum size")
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    decision = {
        "schema_version": output["decision_schema"]["schema_version"],
        "experiment_id": runtime["experiment_id"],
        "run_id": runtime["run_id"],
        "config_sha256": v3_digest,
        "scientific_base": scientific_diff,
        "full_sha256_manifest": {
            "path": manifest["path"],
            "sha256": manifest["sha256"],
            "run_id": runtime["full_sha256_preflight"]["run_id"],
        },
        "zero_contract_v2_evidence": zero_evidence,
        "v3_lineage_evidence": LINEAGE_AUDIT,
        "numerical_equivalence_v3_contract": ACTIVE_V3[
            "numerical_equivalence_v3"
        ],
        "halt_only_validity_guards": results[
            "halt_only_validity_guards"
        ],
        "result_file": output["result_file"],
        "result_sha256": result_sha256,
        "decision_label": results["decision"]["decision"],
        "continue": results["decision"]["continue"],
        "dataset_pass": {
            dataset: results["decision"]["datasets"][dataset]["pass"]
            for dataset in base.REQUIRED_DATASETS
        },
        "small_displacement_gate_reference": {
            dataset: results["decision"]["datasets"][dataset]["checks"][
                "displacement_stability"
            ]["reference"]
            for dataset in base.REQUIRED_DATASETS
        },
        "small_displacement_gate_selection_rule": output[
            "decision_schema"
        ]["small_displacement_gate_selection_rule"],
        "endpoint_concat_small_displacement_role": output[
            "decision_schema"
        ]["endpoint_concat_small_displacement_role"],
        "interpretation_scope": results["decision"]["interpretation"],
        "exclusive_create": True,
    }
    validate_decision_v3(decision, runtime)
    decision_bytes = base.json_payload(decision)
    base.atomic_publish(result_path, result_bytes)
    base.atomic_publish(decision_path, decision_bytes)
    print(
        json.dumps(
            {
                "run_id": runtime["run_id"],
                "decision": results["decision"]["decision"],
                "result": str(result_path.relative_to(REPO)),
                "decision_file": str(decision_path.relative_to(REPO)),
                "result_sha256": result_sha256,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("C01_A0_V3_FAIL_CLOSED: {}".format(exc), file=sys.stderr)
        raise
