#!/usr/bin/env python3
"""C01 A0 v4: frozen v3 science with a typed fail-closed audit union.

V4 SHA-imports the complete frozen v3 implementation.  It changes only
run/output lineage and the HALT-only audit schema that distinguishes direct
retrieval, the derived average-score control, and no-registered-null cases.
"""

import argparse
import copy
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = "configs/c01/c01_a0_v4.json"
V4_CONFIG_SHA256 = (
    "2d9488e6f9af6be00d500d1c2f13912f"
    "d4be0ab9439608d33b0857178efe7ca6"
)
V3_SOURCE = "scripts/analysis/c01_policy_contrast_a0_v3.py"
V3_SOURCE_SHA256 = (
    "40b35eee2fb6fdbdb21fe9b4acfdcebf"
    "003c121c76492b898fbd2ea9b8c34dfb"
)
HALT_LABEL = "HALT_NUMERICAL_EQUIVALENCE"
SCHEMA_HALT_LABEL = "HALT_AUDIT_SCHEMA_V4"

DIRECT_KIND = "REGISTERED_NULL_RETRIEVAL"
DERIVED_KIND = "DERIVED_AVERAGE_SCORE_CONTROL"
NO_NULL_KIND = "NO_REGISTERED_NULL"
DIRECT_SEMANTICS = "DIRECT_COUNT_FROM_TOP20_NEIGHBOR_IDS"
DERIVED_SEMANTICS = (
    "NOT_DIRECTLY_APPLICABLE_PROVEN_BY_SOURCE_RETRIEVAL_ARMS"
)
NO_NULL_SEMANTICS = "NOT_APPLICABLE_NO_REGISTERED_NULL"
DERIVED_SOURCE_ARMS = ["endpoint_std", "endpoint_ow"]


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_frozen_v3():
    path = (REPO / V3_SOURCE).resolve()
    if sha256_file(path) != V3_SOURCE_SHA256:
        raise RuntimeError("frozen v3 analysis source SHA256 drift")
    spec = importlib.util.spec_from_file_location("c01_a0_v3_frozen", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load frozen v3 analysis source")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


v3 = load_frozen_v3()
base = v3.base
ORIGINAL_RETRIEVAL = v3.retrieval_without_registered_null_v3
ORIGINAL_AVG_SCORE = v3.avg_score_equivalence
ORIGINAL_PERMUTATION = v3.permutation_null_v3
ACTIVE_V4 = None
FROZEN_V3_CONFIG = None
V4_LINEAGE_AUDIT = None
SOFTWARE_AUDIT = None
ULP_SELF_CHECK = None
SCHEMA_SELF_TEST = None


def halt_schema(context, message):
    base.die("{}: {}: {}".format(SCHEMA_HALT_LABEL, context, message))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG,
        help="Repo-relative frozen C01 A0 v4 configuration.",
    )
    return parser.parse_args()


def require_exact_keys(value, expected, context):
    if type(value) is not dict:
        halt_schema(context, "expected object")
    if set(value) != set(expected):
        halt_schema(
            context,
            "keys differ: expected={} actual={}".format(
                sorted(expected), sorted(value)
            ),
        )


def load_v4_config(relative):
    path = base.repo_path(relative, "v4 config")
    base.reject_test_path(path, "v4 config")
    digest = sha256_file(path)
    if digest != V4_CONFIG_SHA256:
        base.die("v4 config SHA256 differs from reviewed canonical config")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    require_exact_keys(
        config,
        {
            "schema_version",
            "experiment_id",
            "run_id",
            "frozen_v3",
            "runtime_failure_evidence",
            "execution",
            "numerical_equivalence_v3",
            "audit_schema_v4",
            "output",
        },
        "v4_config",
    )
    schema = config["audit_schema_v4"]
    variants = schema["variants"]
    if (
        config["schema_version"] != 1
        or config["experiment_id"] != "C01_A0"
        or config["run_id"] != "C01-A0-v4"
        or config["frozen_v3"]["analysis_path"] != V3_SOURCE
        or config["frozen_v3"]["analysis_sha256"] != V3_SOURCE_SHA256
        or config["frozen_v3"]["scientific_thresholds_exact"] is not True
        or config["numerical_equivalence_v3"]["halt_label"] != HALT_LABEL
        or schema["required"] is not True
        or schema["halt_label"] != SCHEMA_HALT_LABEL
        or schema["union_discriminator"] != "audit_kind"
        or set(variants) != {DIRECT_KIND, DERIVED_KIND, NO_NULL_KIND}
        or variants[DIRECT_KIND]["semantics"] != DIRECT_SEMANTICS
        or variants[DERIVED_KIND]["semantics"] != DERIVED_SEMANTICS
        or variants[NO_NULL_KIND]["semantics"] != NO_NULL_SEMANTICS
        or variants[DERIVED_KIND]["required_source_arms"]
        != DERIVED_SOURCE_ARMS
    ):
        base.die("v4 canonical identity/audit binding changed")
    return config, path, digest


def load_frozen_v3_config(v4):
    frozen = v4["frozen_v3"]
    path = base.repo_path(frozen["config_path"], "frozen v3 config")
    if sha256_file(path) != frozen["config_sha256"]:
        base.die("frozen v3 config SHA256 drift")
    config, checked_path, digest = v3.load_v3_config(
        frozen["config_path"]
    )
    if checked_path != path or digest != frozen["config_sha256"]:
        base.die("frozen v3 config loader provenance mismatch")
    for name in ("wrapper", "record"):
        item_path = base.repo_path(
            frozen[name + "_path"], "frozen v3 " + name
        )
        if sha256_file(item_path) != frozen[name + "_sha256"]:
            base.die("frozen v3 {} SHA256 drift".format(name))
    if (
        v4["execution"] != config["execution"]
        or v4["numerical_equivalence_v3"]
        != config["numerical_equivalence_v3"]
    ):
        base.die("v4 changed frozen v3 execution/numerical contract")
    return config


def load_v4_lineage_evidence(v4, frozen_v3):
    failure = v4["runtime_failure_evidence"]
    review_path = base.repo_path(
        failure["review_path"], "v4 failure review"
    )
    lines = review_path.read_text(encoding="utf-8").splitlines()
    start = failure["review_line_start"]
    end = failure["review_line_end"]
    if (
        len(lines) < end
        or lines[start - 1] != "## " + failure["review_heading"]
    ):
        base.die("v4 failure review range/heading drift")
    section = ("\n".join(lines[start - 1 : end]) + "\n").encode("utf-8")
    if hashlib.sha256(section).hexdigest() != failure[
        "review_section_sha256"
    ]:
        base.die("v4 failure review section SHA256 drift")
    for stream in ("stdout", "stderr"):
        path = base.repo_path(
            failure[stream + "_path"], "job 13735 " + stream
        )
        if sha256_file(path) != failure[stream + "_sha256"]:
            base.die("job 13735 {} SHA256 drift".format(stream))
    failed_namespace = base.repo_path(
        failure["failed_namespace"], "failed v3 namespace"
    )
    if (
        not failed_namespace.is_dir()
        or list(failed_namespace.iterdir())
        or (failed_namespace / "C01_A0_OUT.json").exists()
        or (failed_namespace / "C01_A0_DECISION.json").exists()
        or failure["result_absent"] is not True
        or failure["decision_absent"] is not True
        or failure["scientific_or_numerical_failure_observed"] is not False
    ):
        base.die("job 13735 empty/no-decision failure boundary drift")
    prior = v3.load_lineage_evidence(frozen_v3)
    if prior["pass"] is not True:
        base.die("frozen v3 lineage no longer passes")
    return {
        "pass": True,
        "frozen_v3": {
            "config_path": v4["frozen_v3"]["config_path"],
            "config_sha256": v4["frozen_v3"]["config_sha256"],
            "analysis_path": v4["frozen_v3"]["analysis_path"],
            "analysis_sha256": v4["frozen_v3"]["analysis_sha256"],
            "wrapper_path": v4["frozen_v3"]["wrapper_path"],
            "wrapper_sha256": v4["frozen_v3"]["wrapper_sha256"],
            "record_path": v4["frozen_v3"]["record_path"],
            "record_sha256": v4["frozen_v3"]["record_sha256"],
        },
        "job_13735_failure": {
            "job_id": failure["job_id"],
            "status": failure["status"],
            "review_path": failure["review_path"],
            "review_section_sha256": failure["review_section_sha256"],
            "stdout_sha256": failure["stdout_sha256"],
            "stderr_sha256": failure["stderr_sha256"],
            "namespace_empty": True,
            "result_absent": True,
            "decision_absent": True,
            "failure_scope": failure["failure_scope"],
            "scientific_or_numerical_failure_observed": False,
        },
        "frozen_v3_lineage_evidence": prior,
    }


def build_runtime_config(v4, frozen_v3):
    runtime_v3, scientific_diff = v3.build_runtime_config(frozen_v3)
    runtime = copy.deepcopy(runtime_v3)
    runtime["run_id"] = v4["run_id"]
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
        runtime["output"][key] = v4["output"][key]
    schema = runtime["output"]["decision_schema"]
    schema["schema_version"] = v4["output"]["decision_schema_version"]
    schema["required_halt_only_validity_guards"] = [
        "probe_evidence_exact",
        "v4_lineage_evidence_exact",
        "audit_schema_union_v4",
        "raw_zero_allowlist_exact",
        "derived_zero_masks_exact",
        "displacement_null_exclusion_dual_path_exact",
        "shuffle_fixed_point_bijection",
        "registered_null_absent_from_all_top20",
        "with_null_remove_null_numerical_equivalence_v3",
        "deterministic_binary64_reference",
        "scientific_boolean_basis_exact",
    ]
    differences = v3.recursive_diff(runtime_v3, runtime)
    allowed = {
        "run_id",
        "output.namespace",
        "output.record_file",
        "output.slurm_stdout",
        "output.slurm_stderr",
        "output.result_schema_version",
        "output.decision_schema.schema_version",
        "output.decision_schema.required_halt_only_validity_guards",
    }
    if set(differences) != allowed:
        base.die(
            "v4 runtime differs from frozen v3 outside exact lineage/schema "
            "repair: {}".format(sorted(differences))
        )
    scientific_diff = copy.deepcopy(scientific_diff)
    scientific_diff["change_scope"] = (
        "frozen v2/v3 scientific fields and v3 numerical equivalence exact; "
        "v4 changes only run/output lineage and HALT-only typed audit schema"
    )
    scientific_diff["scientific_thresholds_exact"] = True
    return runtime, scientific_diff


EVALUATION_KEYS = {
    "pass",
    "checks",
    "comparisons",
    "scientific_boolean_basis",
}
EVALUATION_CHECK_KEYS = {
    "predictions_dtype_shape_bytes_exact",
    "metrics_canonical_typed_bytes_exact",
    "binary64_reference_predictions_exact",
    "binary64_reference_metrics_canonical_typed_bytes_exact",
    "cutoff_interval_stability",
    "scientific_boolean_basis_bytes_exact",
}
DIRECT_REFERENCE_KEYS = {
    "pass",
    "scope",
    "algorithm",
    "thread_contract",
    "relationship_to_float32_neighbors",
    "checks",
    "reference_similarity",
    "reference_score",
    "faiss_relation",
}
DIRECT_REFERENCE_CHECK_KEYS = {
    "shared_reference_between_paths_byte_exact",
    "reference_neighbor_ids_are_exact_agreed_float32_ids",
    "faiss_reference_similarity_within_B_sim",
    "faiss_reference_score_within_B_sim_plus_2^-45",
    "faiss_reference_similarity_zero_positions_exact",
    "faiss_reference_similarity_zero_signbits_exact",
    "faiss_reference_score_zero_positions_exact",
    "faiss_reference_score_zero_signbits_exact",
    "faiss_reference_predictions_exact",
}
DIRECT_KEYS_BEFORE_EVALUATION = {
    "status",
    "pass",
    "halt_label_on_failure",
    "registered_null_index",
    "registered_null_top20_count",
    "operand_parity",
    "mapping",
    "raw_neighbors",
    "stable_neighbors",
    "neighbor_labels",
    "exact_non_numerical_checks",
    "similarity_numerical_equivalence",
    "score_numerical_equivalence",
    "binary64_exact_neighbor_reference",
}
DIRECT_KEYS = DIRECT_KEYS_BEFORE_EVALUATION | {
    "evaluation_checks",
    "audit_kind",
    "registered_null_top20_semantics",
}
DERIVED_KEYS_BEFORE_TYPING = {
    "status",
    "pass",
    "source_neighbor_arms",
    "score_numerical_equivalence",
    "binary64_reference",
    "evaluation_checks",
}
DERIVED_KEYS = DERIVED_KEYS_BEFORE_TYPING | {
    "audit_kind",
    "registered_null_top20_count",
    "registered_null_top20_semantics",
    "source_registered_null_top20",
}
NO_NULL_KEYS_BEFORE_TYPING = {
    "status",
    "pass",
    "numerical_equivalence_v3",
}
NO_NULL_KEYS = NO_NULL_KEYS_BEFORE_TYPING | {
    "audit_kind",
    "registered_null_top20_count",
    "registered_null_top20_semantics",
}


def validate_evaluation(evaluation, context):
    require_exact_keys(evaluation, EVALUATION_KEYS, context)
    require_exact_keys(
        evaluation["checks"], EVALUATION_CHECK_KEYS, context + "/checks"
    )
    if evaluation["pass"] is not True or not all(
        type(value) is bool and value
        for value in evaluation["checks"].values()
    ):
        halt_schema(context, "evaluation checks are not exact true booleans")
    require_exact_keys(
        evaluation["scientific_boolean_basis"],
        {"with_sha256", "remove_sha256", "exact", "proof_scope"},
        context + "/scientific_boolean_basis",
    )
    basis = evaluation["scientific_boolean_basis"]
    if (
        basis["exact"] is not True
        or type(basis["with_sha256"]) is not str
        or type(basis["remove_sha256"]) is not str
        or basis["with_sha256"] != basis["remove_sha256"]
    ):
        halt_schema(context, "scientific basis is not byte exact")


def validate_direct_reference(reference, context):
    require_exact_keys(reference, DIRECT_REFERENCE_KEYS, context)
    require_exact_keys(
        reference["checks"],
        DIRECT_REFERENCE_CHECK_KEYS,
        context + "/checks",
    )
    if reference["pass"] is not True or not all(
        type(value) is bool and value
        for value in reference["checks"].values()
    ):
        halt_schema(context, "binary64 direct reference did not pass")


def validate_direct_before_evaluation(audit, context):
    require_exact_keys(audit, DIRECT_KEYS_BEFORE_EVALUATION, context)
    if (
        audit["status"] != "PASS"
        or audit["pass"] is not True
        or audit["halt_label_on_failure"] != HALT_LABEL
        or type(audit["registered_null_index"]) is not int
        or type(audit["registered_null_top20_count"]) is not int
        or audit["registered_null_top20_count"] != 0
    ):
        halt_schema(
            context, "invalid pre-evaluation direct audit types/values"
        )
    validate_direct_reference(
        audit["binary64_exact_neighbor_reference"],
        context + "/binary64_reference",
    )


def validate_audit(audit, context):
    if type(audit) is not dict:
        halt_schema(context, "audit is not an object")
    if "audit_kind" not in audit:
        halt_schema(context, "missing audit_kind discriminator")
    kind = audit["audit_kind"]
    if kind == DIRECT_KIND:
        require_exact_keys(audit, DIRECT_KEYS, context)
        if (
            audit["status"] != "PASS"
            or audit["pass"] is not True
            or audit["halt_label_on_failure"] != HALT_LABEL
            or type(audit["registered_null_index"]) is not int
            or type(audit["registered_null_top20_count"]) is not int
            or audit["registered_null_top20_count"] != 0
            or audit["registered_null_top20_semantics"]
            != DIRECT_SEMANTICS
        ):
            halt_schema(context, "invalid direct-retrieval audit types/values")
        validate_direct_reference(
            audit["binary64_exact_neighbor_reference"],
            context + "/binary64_reference",
        )
        validate_evaluation(
            audit["evaluation_checks"], context + "/evaluation"
        )
    elif kind == DERIVED_KIND:
        require_exact_keys(audit, DERIVED_KEYS, context)
        if (
            audit["status"] != "PASS"
            or audit["pass"] is not True
            or audit["registered_null_top20_count"] is not None
            or audit["registered_null_top20_semantics"]
            != DERIVED_SEMANTICS
            or audit["source_neighbor_arms"] != DERIVED_SOURCE_ARMS
        ):
            halt_schema(context, "invalid derived-control audit types/values")
        proof = audit["source_registered_null_top20"]
        require_exact_keys(
            proof,
            {"source_arms", "counts_by_arm", "all_source_counts_zero"},
            context + "/source_proof",
        )
        require_exact_keys(
            proof["counts_by_arm"],
            DERIVED_SOURCE_ARMS,
            context + "/source_proof/counts",
        )
        if (
            proof["source_arms"] != DERIVED_SOURCE_ARMS
            or proof["all_source_counts_zero"] is not True
            or not all(
                type(value) is int and value == 0
                for value in proof["counts_by_arm"].values()
            )
        ):
            halt_schema(context, "invalid derived source-count proof")
        reference = audit["binary64_reference"]
        require_exact_keys(
            reference,
            {"scope", "prediction_bytes_exact", "canonical_metrics_exact"},
            context + "/binary64_reference",
        )
        if (
            reference["prediction_bytes_exact"] is not True
            or reference["canonical_metrics_exact"] is not True
        ):
            halt_schema(context, "derived binary64 reference did not pass")
        validate_evaluation(
            audit["evaluation_checks"], context + "/evaluation"
        )
    elif kind == NO_NULL_KIND:
        require_exact_keys(audit, NO_NULL_KEYS, context)
        if (
            audit["status"] != "NO_REGISTERED_NULL"
            or audit["pass"] is not True
            or audit["numerical_equivalence_v3"] != "NOT_APPLICABLE"
            or type(audit["registered_null_top20_count"]) is not int
            or audit["registered_null_top20_count"] != 0
            or audit["registered_null_top20_semantics"]
            != NO_NULL_SEMANTICS
        ):
            halt_schema(context, "invalid no-registered-null audit")
    else:
        halt_schema(context, "unknown audit_kind {!r}".format(kind))
    return {
        "audit_kind": kind,
        "pass": True,
        "registered_null_top20_count": audit[
            "registered_null_top20_count"
        ],
        "registered_null_absence_pass": True,
        "numerical_equivalence_pass": True,
        "reference_pass": True,
        "scientific_basis_pass": True,
    }


def summarize_retrieval_audits(retrieval, context):
    if type(retrieval) is not dict or not retrieval:
        halt_schema(context, "retrieval audit mapping is empty/non-object")
    summaries = {}
    for arm, audit in retrieval.items():
        if type(arm) is not str or not arm:
            halt_schema(context, "invalid retrieval arm key")
        summaries[arm] = validate_audit(audit, context + "/" + arm)
    for arm, audit in retrieval.items():
        if summaries[arm]["audit_kind"] != DERIVED_KIND:
            continue
        proof = audit["source_registered_null_top20"]
        for source in audit["source_neighbor_arms"]:
            if source not in summaries:
                halt_schema(context, "derived audit source arm is absent")
            source_summary = summaries[source]
            if (
                source_summary["audit_kind"] != DIRECT_KIND
                or source_summary["registered_null_top20_count"]
                != proof["counts_by_arm"][source]
            ):
                halt_schema(
                    context,
                    "derived audit source kind/count linkage failed",
                )
    return {
        "pass": True,
        "audit_count": len(summaries),
        "summaries": summaries,
        "registered_null_absence_pass": all(
            item["registered_null_absence_pass"]
            for item in summaries.values()
        ),
        "numerical_equivalence_pass": all(
            item["numerical_equivalence_pass"]
            for item in summaries.values()
        ),
        "reference_pass": all(
            item["reference_pass"] for item in summaries.values()
        ),
        "scientific_basis_pass": all(
            item["scientific_basis_pass"] for item in summaries.values()
        ),
    }


def enrich_no_registered_null(audit, context):
    if "audit_kind" in audit:
        validate_audit(audit, context)
        return audit
    require_exact_keys(audit, NO_NULL_KEYS_BEFORE_TYPING, context)
    audit["audit_kind"] = NO_NULL_KIND
    audit["registered_null_top20_count"] = 0
    audit["registered_null_top20_semantics"] = NO_NULL_SEMANTICS
    validate_audit(audit, context)
    return audit


def retrieval_without_registered_null_v4(
    memory,
    memory_labels,
    query,
    full_retrieval,
    null_index,
    config,
    context,
):
    audit, reduced = ORIGINAL_RETRIEVAL(
        memory,
        memory_labels,
        query,
        full_retrieval,
        null_index,
        config,
        context,
    )
    if audit["status"] == "NO_REGISTERED_NULL":
        if reduced is not None:
            halt_schema(context, "no-null audit returned reduced retrieval")
        return enrich_no_registered_null(audit, context), None
    validate_direct_before_evaluation(audit, context)
    audit["audit_kind"] = DIRECT_KIND
    audit["registered_null_top20_semantics"] = DIRECT_SEMANTICS
    if (
        reduced is None
        or "_v3_guard" not in reduced
        or reduced["_v3_guard"]["audit"] is not audit
    ):
        halt_schema(context, "direct audit/reduced guard identity mismatch")
    return audit, reduced


def avg_score_equivalence_v4(
    full,
    reduced_scores,
    source_reduced,
    dev_labels,
    cutoff,
    context,
):
    if type(source_reduced) is not list or len(source_reduced) != 2:
        halt_schema(context, "average-score source list changed")
    counts = {}
    for index, arm in enumerate(DERIVED_SOURCE_ARMS):
        reduced = source_reduced[index][1]
        if (
            type(reduced) is not dict
            or "_v3_guard" not in reduced
            or "audit" not in reduced["_v3_guard"]
        ):
            halt_schema(context, "average-score source audit missing")
        source_audit = reduced["_v3_guard"]["audit"]
        summary = validate_audit(
            source_audit, context + "/source/" + arm
        )
        if summary["audit_kind"] != DIRECT_KIND:
            halt_schema(context, "average-score source is not retrieval")
        counts[arm] = summary["registered_null_top20_count"]
    audit = ORIGINAL_AVG_SCORE(
        full,
        reduced_scores,
        source_reduced,
        dev_labels,
        cutoff,
        context,
    )
    require_exact_keys(audit, DERIVED_KEYS_BEFORE_TYPING, context)
    audit["audit_kind"] = DERIVED_KIND
    audit["registered_null_top20_count"] = None
    audit["registered_null_top20_semantics"] = DERIVED_SEMANTICS
    audit["source_registered_null_top20"] = {
        "source_arms": list(DERIVED_SOURCE_ARMS),
        "counts_by_arm": counts,
        "all_source_counts_zero": all(
            type(value) is int and value == 0
            for value in counts.values()
        ),
    }
    validate_audit(audit, context)
    return audit


def evaluate_real_arms_v4(
    train_views,
    dev_views,
    train_labels,
    dev_labels,
    config,
    registered_null_index,
):
    evaluations, public, validity = v3.evaluate_real_arms_v3(
        train_views,
        dev_views,
        train_labels,
        dev_labels,
        config,
        registered_null_index,
    )
    if "avg_score" not in validity:
        halt_schema("real", "average-score audit is absent")
    if validity["avg_score"]["status"] == "NO_REGISTERED_NULL":
        enrich_no_registered_null(validity["avg_score"], "real/avg_score")
    summarize_retrieval_audits(validity, "real/audit_union")
    return evaluations, public, validity


def audit_has_reference_v4(audit):
    return validate_audit(audit, "audit/reference")["reference_pass"]


def audit_has_scientific_basis_v4(audit):
    return validate_audit(audit, "audit/scientific_basis")[
        "scientific_basis_pass"
    ]


def permutation_null_v4(
    dataset, caches, normed, real_evaluations, config, zero_masks
):
    public, holm, validity = ORIGINAL_PERMUTATION(
        dataset,
        caches,
        normed,
        real_evaluations,
        config,
        zero_masks,
    )
    expected = {
        "status",
        "pass",
        "registered_null_index",
        "fixed_train_indices",
        "train_fixed_point_draws_checked",
        "train_bijection_draws_checked",
        "dev_bijection_draws_checked",
        "retrieval_arm_draw_checks",
        "registered_null_top20_occurrences",
        "with_null_remove_null_numerical_mismatches",
        "binary64_reference_mismatches",
        "scientific_boolean_basis_mismatches",
        "audit_digest",
        "per_draw_formula_derived_bound_audits",
    }
    require_exact_keys(validity, expected, dataset + "/shuffle_validity")
    integer_fields = (
        "train_fixed_point_draws_checked",
        "train_bijection_draws_checked",
        "dev_bijection_draws_checked",
        "retrieval_arm_draw_checks",
        "registered_null_top20_occurrences",
        "with_null_remove_null_numerical_mismatches",
        "binary64_reference_mismatches",
        "scientific_boolean_basis_mismatches",
    )
    if (
        validity["status"] != "PASS"
        or validity["pass"] is not True
        or not all(type(validity[key]) is int for key in integer_fields)
    ):
        halt_schema(dataset, "invalid typed shuffle aggregate")
    return public, holm, validity


def _evaluation_fixture():
    checks = {key: True for key in EVALUATION_CHECK_KEYS}
    return {
        "pass": True,
        "checks": checks,
        "comparisons": {},
        "scientific_boolean_basis": {
            "with_sha256": "0" * 64,
            "remove_sha256": "0" * 64,
            "exact": True,
            "proof_scope": "schema fixture only",
        },
    }


def _direct_fixture():
    reference_checks = {
        key: True for key in DIRECT_REFERENCE_CHECK_KEYS
    }
    return {
        "status": "PASS",
        "pass": True,
        "halt_label_on_failure": HALT_LABEL,
        "registered_null_index": 355,
        "registered_null_top20_count": 0,
        "operand_parity": {},
        "mapping": {},
        "raw_neighbors": {},
        "stable_neighbors": {},
        "neighbor_labels": {},
        "exact_non_numerical_checks": {},
        "similarity_numerical_equivalence": {},
        "score_numerical_equivalence": {},
        "binary64_exact_neighbor_reference": {
            "pass": True,
            "scope": "fixture",
            "algorithm": "fixture",
            "thread_contract": "fixture",
            "relationship_to_float32_neighbors": "fixture",
            "checks": reference_checks,
            "reference_similarity": {},
            "reference_score": {},
            "faiss_relation": {},
        },
        "evaluation_checks": _evaluation_fixture(),
        "audit_kind": DIRECT_KIND,
        "registered_null_top20_semantics": DIRECT_SEMANTICS,
    }


def _derived_fixture():
    return {
        "status": "PASS",
        "pass": True,
        "source_neighbor_arms": list(DERIVED_SOURCE_ARMS),
        "score_numerical_equivalence": {},
        "binary64_reference": {
            "scope": "fixture",
            "prediction_bytes_exact": True,
            "canonical_metrics_exact": True,
        },
        "evaluation_checks": _evaluation_fixture(),
        "audit_kind": DERIVED_KIND,
        "registered_null_top20_count": None,
        "registered_null_top20_semantics": DERIVED_SEMANTICS,
        "source_registered_null_top20": {
            "source_arms": list(DERIVED_SOURCE_ARMS),
            "counts_by_arm": {
                "endpoint_std": 0,
                "endpoint_ow": 0,
            },
            "all_source_counts_zero": True,
        },
    }


def _no_null_fixture():
    return {
        "status": "NO_REGISTERED_NULL",
        "pass": True,
        "numerical_equivalence_v3": "NOT_APPLICABLE",
        "audit_kind": NO_NULL_KIND,
        "registered_null_top20_count": 0,
        "registered_null_top20_semantics": NO_NULL_SEMANTICS,
    }


def expect_schema_rejection(name, value):
    try:
        summarize_retrieval_audits(value, "self_test/" + name)
    except RuntimeError as exc:
        if SCHEMA_HALT_LABEL not in str(exc):
            raise
        return True
    halt_schema("self_test/" + name, "invalid fixture was accepted")


def audit_schema_self_test(config):
    registered = {
        "endpoint_std": _direct_fixture(),
        "endpoint_ow": _direct_fixture(),
        "avg_score": _derived_fixture(),
    }
    valid_registered = summarize_retrieval_audits(
        registered, "self_test/HateMM_valid"
    )["pass"]
    missing_count = copy.deepcopy(registered)
    del missing_count["avg_score"]["registered_null_top20_count"]
    wrong_count_type = copy.deepcopy(registered)
    wrong_count_type["avg_score"]["registered_null_top20_count"] = 0
    wrong_source = copy.deepcopy(registered)
    wrong_source["avg_score"]["source_registered_null_top20"][
        "counts_by_arm"
    ]["endpoint_std"] = 1
    no_null = {"endpoint_std": _no_null_fixture()}
    valid_no_null = summarize_retrieval_audits(
        no_null, "self_test/MHC_NO_REGISTERED_NULL_valid"
    )["pass"]
    no_null_missing = copy.deepcopy(no_null)
    del no_null_missing["endpoint_std"]["registered_null_top20_count"]
    cases = {
        "HateMM_registered_null_direct_and_derived_valid": (
            valid_registered is True
        ),
        "HateMM_avg_missing_count_rejected": expect_schema_rejection(
            "HateMM_avg_missing_count", missing_count
        ),
        "HateMM_avg_integer_count_rejected": expect_schema_rejection(
            "HateMM_avg_integer_count", wrong_count_type
        ),
        "HateMM_avg_nonzero_source_count_rejected": (
            expect_schema_rejection(
                "HateMM_avg_nonzero_source_count", wrong_source
            )
        ),
        "MHC_NO_REGISTERED_NULL_valid": valid_no_null is True,
        "MHC_NO_REGISTERED_NULL_missing_count_rejected": (
            expect_schema_rejection(
                "MHC_NO_REGISTERED_NULL_missing_count", no_null_missing
            )
        ),
    }
    required = config["audit_schema_v4"]["fail_closed_self_test"][
        "required_cases"
    ]
    if list(cases) != required or not all(
        type(value) is bool and value for value in cases.values()
    ):
        halt_schema("self_test", "required schema cases did not all pass")
    return {
        "pass": True,
        "fail_closed": True,
        "required_cases": required,
        "cases": cases,
    }


def derive_public_contract_guards_v4(
    public, runtime_state, dataset, config
):
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
            dataset + "/public_derived_aggregate_v4",
        )
    )
    displacement_exact = bool(
        public["displacement_norm_audit"][
            "dual_path_null_exclusion_audit"
        ]["pass"]
    )
    retrieval = previous["retrieval_null_influence"]
    typed = summarize_retrieval_audits(
        retrieval, dataset + "/public_typed_retrieval_aggregate"
    )
    shuffle = previous["shuffle_null_validity"]
    registered_null_absent = bool(
        typed["registered_null_absence_pass"]
        and shuffle["registered_null_top20_occurrences"] == 0
    )
    numerical_equivalence = bool(
        typed["numerical_equivalence_pass"]
        and shuffle["with_null_remove_null_numerical_mismatches"] == 0
    )
    reference_exact = bool(
        typed["reference_pass"]
        and shuffle["binary64_reference_mismatches"] == 0
    )
    scientific_basis_exact = bool(
        typed["scientific_basis_pass"]
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
        "displacement_null_exclusion_dual_path_exact": displacement_exact,
        "audit_schema_union_v4": typed["pass"],
        "registered_null_absent_from_all_top20": registered_null_absent,
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
        "typed_retrieval_aggregate_v4": typed,
        "shuffle_null_validity": shuffle,
        "derivation": (
            "all audit variants pass exact-key/type validation before "
            "audit_kind-dispatched aggregation; the derived average-score "
            "control inherits null absence only from its two linked direct "
            "retrieval source-arm integer-zero counts"
        ),
    }


def build_halt_only_validity_guards_v4(results, evidence, config):
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
            "audit_schema_union_v4",
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
            dataset + "/public_contract_summary_v4",
        ):
            v3.halt(dataset, "derived public v4 contract summary failed")
        typed = summarize_retrieval_audits(
            contract["retrieval_null_influence"],
            dataset + "/halt_typed_retrieval_aggregate",
        )
        derived = contract["derived_zero_mask_preservation"]
        shuffle = contract["shuffle_null_validity"]
        checks = {
            "audit_schema_union_v4": (
                contract["audit_schema_union_v4"] is True
                and typed["pass"] is True
            ),
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
                    dataset + "/derived_masks_v4",
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
                typed["registered_null_absence_pass"]
                and shuffle["registered_null_top20_occurrences"] == 0
            ),
            "with_null_remove_null_numerical_equivalence_v3": bool(
                typed["numerical_equivalence_pass"]
                and shuffle[
                    "with_null_remove_null_numerical_mismatches"
                ]
                == 0
            ),
            "deterministic_binary64_reference": bool(
                typed["reference_pass"]
                and shuffle["binary64_reference_mismatches"] == 0
            ),
            "scientific_boolean_basis_exact": bool(
                typed["scientific_basis_pass"]
                and shuffle["scientific_boolean_basis_mismatches"] == 0
            ),
        }
        if not base.strict_all(
            checks.values(), dataset + "/halt_only_v4"
        ):
            v3.halt(dataset, "aggregate v4 validity guard failed")
        datasets[dataset] = checks
    global_checks = {
        "probe_evidence_exact": bool(
            evidence["sha256"]
            == config["zero_contract_v2"]["probe_sha256"]
        ),
        "v4_lineage_evidence_exact": bool(
            V4_LINEAGE_AUDIT and V4_LINEAGE_AUDIT["pass"]
        ),
        "audit_schema_union_v4": all(
            item["audit_schema_union_v4"] for item in datasets.values()
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
        base.die("v4 halt-only validity guard order/binding changed")
    if not base.strict_all(
        global_checks.values(), "global_halt_only_v4"
    ):
        v3.halt("global", "v4 halt-only validity guard failed")
    return {
        "halt_only": True,
        "halt_label_on_failure": HALT_LABEL,
        "schema_halt_label_on_failure": SCHEMA_HALT_LABEL,
        "passed": True,
        "required": required,
        "checks": global_checks,
        "datasets": datasets,
    }


def validate_decision_v4(decision, config):
    expected = {
        "schema_version",
        "experiment_id",
        "run_id",
        "config_sha256",
        "scientific_base",
        "full_sha256_manifest",
        "zero_contract_v2_evidence",
        "v4_lineage_evidence",
        "numerical_equivalence_v3_contract",
        "audit_schema_v4_contract",
        "audit_schema_self_test",
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
    require_exact_keys(decision, expected, "decision_v4")
    schema = config["output"]["decision_schema"]
    if (
        decision["schema_version"] != "c01_a0_decision_v4"
        or decision["schema_version"] != schema["schema_version"]
        or config["output"]["result_schema_version"]
        != "c01_a0_result_v4"
        or decision["experiment_id"] != "C01_A0"
        or decision["run_id"] != "C01-A0-v4"
        or decision["run_id"] != config["run_id"]
        or decision["config_sha256"] != V4_CONFIG_SHA256
        or decision["v4_lineage_evidence"] != V4_LINEAGE_AUDIT
        or decision["v4_lineage_evidence"]["pass"] is not True
        or decision["numerical_equivalence_v3_contract"]
        != ACTIVE_V4["numerical_equivalence_v3"]
        or decision["audit_schema_v4_contract"]
        != ACTIVE_V4["audit_schema_v4"]
        or decision["audit_schema_self_test"] != SCHEMA_SELF_TEST
        or decision["audit_schema_self_test"]["pass"] is not True
    ):
        base.die("v4 decision provenance/contract binding failed")
    scientific = decision["scientific_base"]
    require_exact_keys(
        scientific,
        {
            "pass",
            "v2_config_path",
            "v2_config_sha256",
            "v2_analysis_path",
            "v2_analysis_sha256",
            "changed_paths",
            "change_scope",
            "scientific_thresholds_exact",
        },
        "decision_v4/scientific_base",
    )
    if (
        scientific["pass"] is not True
        or scientific["scientific_thresholds_exact"] is not True
        or scientific["v2_config_path"]
        != FROZEN_V3_CONFIG["scientific_base"]["config_path"]
        or scientific["v2_config_sha256"]
        != FROZEN_V3_CONFIG["scientific_base"]["config_sha256"]
        or scientific["v2_analysis_path"]
        != FROZEN_V3_CONFIG["scientific_base"]["analysis_path"]
        or scientific["v2_analysis_sha256"]
        != FROZEN_V3_CONFIG["scientific_base"]["analysis_sha256"]
    ):
        base.die("v4 decision scientific base changed")
    manifest = decision["full_sha256_manifest"]
    require_exact_keys(
        manifest, {"path", "sha256", "run_id"}, "decision_v4/manifest"
    )
    if (
        manifest["path"]
        != config["full_sha256_preflight"]["manifest_path"]
        or manifest["sha256"]
        != config["full_sha256_preflight"][
            "approved_manifest_sha256"
        ]
        or manifest["run_id"]
        != config["full_sha256_preflight"]["run_id"]
    ):
        base.die("v4 decision manifest provenance changed")
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
        base.die("v4 decision zero-contract provenance changed")
    validity = decision["halt_only_validity_guards"]
    if (
        validity["passed"] is not True
        or validity["halt_only"] is not True
        or validity["halt_label_on_failure"] != HALT_LABEL
        or validity["schema_halt_label_on_failure"] != SCHEMA_HALT_LABEL
        or validity["required"]
        != schema["required_halt_only_validity_guards"]
        or list(validity["checks"])
        != schema["required_halt_only_validity_guards"]
        or not base.strict_all(
            validity["checks"].values(), "decision_v4_validity"
        )
    ):
        base.die("v4 decision validity binding failed")
    if (
        decision["decision_label"] not in schema["allowed_decisions"]
        or list(decision["dataset_pass"])
        != schema["required_dataset_keys"]
        or not all(
            type(value) is bool
            for value in decision["dataset_pass"].values()
        )
        or type(decision["continue"]) is not bool
    ):
        base.die("v4 decision scientific output types changed")
    expected_label = (
        config["decision"]["continue_label"]
        if decision["continue"]
        else config["decision"]["kill_label"]
    )
    expected_scope = (
        config["positive_scope"]
        if decision["continue"]
        else config["negative_scope"]
    )
    if (
        decision["decision_label"] != expected_label
        or decision["interpretation_scope"] != expected_scope
        or decision["small_displacement_gate_selection_rule"]
        != schema["small_displacement_gate_selection_rule"]
        or decision["endpoint_concat_small_displacement_role"]
        != schema["endpoint_concat_small_displacement_role"]
        or decision["exclusive_create"] is not True
    ):
        base.die("v4 decision label/scope/exclusive contract changed")
    result_sha = decision["result_sha256"]
    if (
        decision["result_file"] != config["output"]["result_file"]
        or type(result_sha) is not str
        or len(result_sha) != 64
        or any(character not in "0123456789abcdef" for character in result_sha)
    ):
        base.die("v4 decision result-file/SHA256 provenance changed")


def install_v4_overrides():
    v3.install_v3_overrides()
    v3.retrieval_without_registered_null_v3 = (
        retrieval_without_registered_null_v4
    )
    v3.avg_score_equivalence = avg_score_equivalence_v4
    v3.audit_has_reference = audit_has_reference_v4
    v3.audit_has_scientific_basis = audit_has_scientific_basis_v4
    base.retrieval_without_registered_null = (
        retrieval_without_registered_null_v4
    )
    base.evaluate_real_arms = evaluate_real_arms_v4
    base.permutation_null = permutation_null_v4
    base.build_halt_only_validity_guards = (
        build_halt_only_validity_guards_v4
    )


def main():
    global ACTIVE_V4
    global FROZEN_V3_CONFIG
    global V4_LINEAGE_AUDIT
    global SOFTWARE_AUDIT
    global ULP_SELF_CHECK
    global SCHEMA_SELF_TEST
    args = parse_args()
    ACTIVE_V4, v4_path, v4_digest = load_v4_config(args.config)
    FROZEN_V3_CONFIG = load_frozen_v3_config(ACTIVE_V4)
    runtime, scientific_diff = build_runtime_config(
        ACTIVE_V4, FROZEN_V3_CONFIG
    )
    V4_LINEAGE_AUDIT = load_v4_lineage_evidence(
        ACTIVE_V4, FROZEN_V3_CONFIG
    )
    base.enforce_runtime(runtime)
    SCHEMA_SELF_TEST = audit_schema_self_test(ACTIVE_V4)
    manifest = base.load_full_sha256_manifest(runtime)
    zero_evidence = base.load_zero_contract_evidence(runtime)
    base.import_compute_modules(runtime)
    SOFTWARE_AUDIT = v3.verify_software_environment(ACTIVE_V4)
    ULP_SELF_CHECK = v3.float32_ulp_self_check()
    v3.ACTIVE_V3 = FROZEN_V3_CONFIG
    v3.LINEAGE_AUDIT = V4_LINEAGE_AUDIT[
        "frozen_v3_lineage_evidence"
    ]
    v3.SOFTWARE_AUDIT = SOFTWARE_AUDIT
    v3.ULP_SELF_CHECK = ULP_SELF_CHECK
    install_v4_overrides()
    base.INPUT_ACCESS_LEDGER.clear()

    output = runtime["output"]
    namespace = base.repo_path(output["namespace"], "output.namespace")
    expected_root = (
        REPO / "artifacts" / "c01_policy_contrastive" / "v4" / "a0"
    ).resolve()
    try:
        namespace.relative_to(expected_root)
    except ValueError:
        base.die("v4 namespace escapes exclusive v4/a0 root")
    result_path = namespace / output["result_file"]
    decision_path = namespace / output["decision_file"]
    if namespace.exists():
        base.die("v4 run namespace already exists; no-clobber")
    namespace.parent.mkdir(parents=True, exist_ok=True)
    namespace.mkdir()

    results = {
        "schema_version": output["result_schema_version"],
        "experiment_id": runtime["experiment_id"],
        "run_id": runtime["run_id"],
        "claim_scope": runtime["claim_scope"],
        "config_path": str(v4_path.relative_to(REPO)),
        "config_sha256": v4_digest,
        "scientific_base": scientific_diff,
        "full_sha256_preflight": manifest,
        "zero_contract_v2_evidence": zero_evidence,
        "v4_lineage_evidence": V4_LINEAGE_AUDIT,
        "numerical_equivalence_v3_contract": ACTIVE_V4[
            "numerical_equivalence_v3"
        ],
        "audit_schema_v4_contract": ACTIVE_V4["audit_schema_v4"],
        "audit_schema_self_test": SCHEMA_SELF_TEST,
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
                "v4 changes no scientific transform, arm, statistic, "
                "threshold or numerical-equivalence rule from frozen v3"
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
        public["contract_guards"] = derive_public_contract_guards_v4(
            public, runtime_state, dataset, runtime
        )
        results["datasets"][dataset] = public
        runtimes[dataset] = runtime_state
    expected_access = base.CANONICAL_BINDING["manifest"][
        "expected_file_count"
    ]
    if len(base.INPUT_ACCESS_LEDGER) != expected_access:
        base.die("v4 runtime access ledger is incomplete")
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
        "v4_global_cache_ledger",
    )
    if test_attempts or test_opens or not all_manifest:
        base.die("v4 global runtime cache access guard failed")
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
    ] = build_halt_only_validity_guards_v4(
        results, zero_evidence, runtime
    )
    base.attach_bootstrap_and_holm(results, runtimes, runtime)
    results["decision"] = base.make_decision(results, runtimes, runtime)

    result_bytes = base.json_payload(results)
    if len(result_bytes) > int(output["maximum_result_bytes"]):
        base.die("v4 result JSON exceeds configured maximum size")
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    decision = {
        "schema_version": output["decision_schema"]["schema_version"],
        "experiment_id": runtime["experiment_id"],
        "run_id": runtime["run_id"],
        "config_sha256": v4_digest,
        "scientific_base": scientific_diff,
        "full_sha256_manifest": {
            "path": manifest["path"],
            "sha256": manifest["sha256"],
            "run_id": runtime["full_sha256_preflight"]["run_id"],
        },
        "zero_contract_v2_evidence": zero_evidence,
        "v4_lineage_evidence": V4_LINEAGE_AUDIT,
        "numerical_equivalence_v3_contract": ACTIVE_V4[
            "numerical_equivalence_v3"
        ],
        "audit_schema_v4_contract": ACTIVE_V4["audit_schema_v4"],
        "audit_schema_self_test": SCHEMA_SELF_TEST,
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
    validate_decision_v4(decision, runtime)
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
        print("C01_A0_V4_FAIL_CLOSED: {}".format(exc), file=sys.stderr)
        raise
