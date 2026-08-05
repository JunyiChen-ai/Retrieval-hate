#!/usr/bin/env python3
"""C01 A0: strict train-memory -> dev-query endpoint-contrast screen.

This script is intentionally CPU-only and Slurm-only.  It opens exactly the
registered MHC_zh/HateMM train/dev_seen standard-L24 and one-word-L24 caches.
Any test-like split or input path fails before torch.load.

The primary object is a block-normalized [common, displacement] reparameterization
of the paired endpoints.  Because the two existing caches also differ in pooling,
this screen is only a readout-policy endpoint audit, not a prompt-only or safety-
disentanglement experiment.  See refine-logs/C01_A0_RECORD.md.
"""

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
REQUIRED_DATASETS = ("MHC_zh", "HateMM")
REQUIRED_SPLITS = ("train", "dev_seen")
REQUIRED_ARMS = {
    "endpoint_std",
    "endpoint_ow",
    "avg_score",
    "endpoint_concat",
    "common",
    "displacement",
    "common_displacement",
    "common_interaction",
    "shuffled_pair_null",
}
CANONICAL_BINDING = {
    "run_id": "C01-A0-v2",
    "claim_scope": (
        "A parameter-free, block-normalized paired-endpoint contrast may preserve "
        "useful policy/readout-conditioned structure beyond either L24 endpoint "
        "and equally normalized endpoint-pair or orthogonal-rotation controls."
    ),
    "positive_scope": (
        "A CONTINUE decision authorizes only a later extraction of same-pooling "
        "neutral/policy caches and a capacity-matched fold-head pilot. It does "
        "not establish safety, stance, or discourse disentanglement."
    ),
    "negative_scope": (
        "A KILL decision retires only the current standard-L24 versus one-word-L24 "
        "endpoint-contrast route. It does not falsify policy contrast under "
        "same-pooling caches."
    ),
    "namespace": "artifacts/c01_policy_contrastive/v2/a0/C01-A0-v2",
    "datasets": ["MHC_zh", "HateMM"],
    "splits": ["train", "dev_seen"],
    "arms": [
        "endpoint_std",
        "endpoint_ow",
        "avg_score",
        "endpoint_concat",
        "common",
        "displacement",
        "common_displacement",
        "common_interaction",
        "shuffled_pair_null",
    ],
    "primary_arm": "common_displacement",
    "secondary_arm": "common_interaction",
    "metrics": ["accuracy", "macro_f1", "roc_auc"],
    "holm_metrics": ["accuracy", "macro_f1"],
    "primary_controls": [
        "endpoint_std",
        "endpoint_ow",
        "avg_score",
        "endpoint_concat",
        "common",
        "displacement",
    ],
    "secondary_controls": [
        "endpoint_concat",
        "common",
        "common_displacement",
    ],
    "gain_controls": [
        "endpoint_std",
        "endpoint_ow",
        "avg_score",
        "endpoint_concat",
        "common",
    ],
    "minimum_gain": 0.02,
    "minimum_bootstrap_lower_bound": 0.0,
    "minimum_net_fixes": {"MHC_zh": 2, "HateMM": 3},
    "bootstrap_count": 2000,
    "bootstrap_lower_quantile": 0.05,
    "bootstrap_upper_quantile": 0.95,
    "holm_alpha": 0.05,
    "seed": 20260728,
    "permutation_count": 256,
    "permutation_hash": "sha256",
    "permutation_pairing": (
        "independent label-blind ID-hash ordering within dataset and split; "
        "the registered structural-null train index is fixed and excluded "
        "from the remaining-source bijection"
    ),
    "angles_degrees": [8.3, 17.6, 29.1, 60.4, 72.7, 83.8],
    "rotation_arm_prefix": "orthrot_",
    "rotation_distribution": (
        "angles frozen ex ante as label-blind pseudo-random draws over (0,90) "
        "degrees; 45 degrees excluded because it is the primary "
        "common/displacement transform"
    ),
    "rotation_upper_bound": (
        "maximum observed metric over every frozen angle; no angle selection "
        "or tuning is permitted"
    ),
    "output_budget_bytes": 100000000,
    "input_contract": {
        "cache_root": "data/CLIP_Embedding",
        "feature_dim": 3584,
        "exact_contract_keys": ["ids", "img_feats", "text_feats", "labels"],
        "standard_suffix": "ro_L24",
        "oneword_suffix": "ro_ow_L24",
    },
    "execution": {
        "required_cpus": 8,
        "required_environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
        },
    },
    "retrieval_operator": {
        "protocol": "strict_train_memory_to_dev_query",
        "topk": 20,
        "similarity": "signed_cosine",
        "rank_weights": "descending_integer",
        "prediction_cutoff": 0.0,
        "fix_break_reference": "endpoint_std",
    },
    "transform_thresholds": {
        "normalization_epsilon": 1e-12,
        "tiny_displacement_epsilon": 0.001,
        "small_displacement_train_quantile": 0.1,
        "max_tiny_displacement_fraction": 0.05,
        "max_small_displacement_fix_fraction": 0.5,
    },
    "transform_semantics": {
        "endpoint_order": ["standard", "oneword"],
        "common": "L2(L2(standard)+L2(oneword))",
        "displacement": "L2(L2(oneword)-L2(standard))",
        "common_interaction": "L2(common*displacement)",
        "small_displacement_gate_reference": (
            "strongest_ordinary_control_by_accuracy_then_macro_f1_then_"
            "frozen_gain_controls_order"
        ),
        "small_displacement_endpoint_concat_role": "diagnostic_only",
        "displacement_registered_null_exclusion": (
            "with_null_masked_vs_physically_remove_null_dual_path_exact"
        ),
        "pair_block_rule": (
            "L2-concatenate two individually L2-normalized blocks within each "
            "modality, then L2-concatenate image and text modality blocks"
        ),
    },
    "history_parity_tolerance": 1e-12,
    "manifest": {
        "required": True,
        "schema_version": "c01_full_sha256_manifest_v1",
        "run_id": "C01-HASH-v1",
        "source_set_id": "c01_l24_train_dev_eight_files_v1",
        "path": (
            "artifacts/c01_policy_contrastive/v1/hash_preflight/"
            "C01-HASH-v1/full_sha256_manifest.json"
        ),
        "approved_sha256": (
            "083275d39a1026bde3b6583bd5608d41"
            "cec5b431da9ffda87ae8ab1046cf2305"
        ),
        "producer_script": "scripts/analysis/c01_hash_inputs.py",
        "expected_file_count": 8,
        "required_thread_environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "1",
            "MKL_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "NUMEXPR_NUM_THREADS": "1",
        },
        "access_ledger_entry_keys": [
            "ordinal",
            "path",
            "dataset",
            "split",
            "policy",
            "open_attempted",
            "opened",
            "bytes_read",
            "test_like",
        ],
        "slurm_job_id_pattern": r"[1-9][0-9]*",
    },
        "zero_contract_v2": {
        "probe_path": (
            "artifacts/c01_policy_contrastive/v2/zero_contract_probe/"
            "C01-ZERO-PROBE-v1/zero_contract_probe.json"
        ),
        "probe_sha256": (
            "bee4964ce7e4ca81cfdb72c3859f7819"
            "6568badf982aef587bc14ee6dbe63526"
        ),
        "probe_schema_version": "c01_zero_contract_probe_v1",
        "probe_run_id": "C01-ZERO-PROBE-v1",
        "authorized_dataset": "HateMM",
        "authorized_split": "train",
        "authorized_id": "hate_video_95",
        "authorized_row_index": 355,
        "expected_label_integrity_only": 1,
        "authorized_policies": ["standard", "oneword"],
        "authorized_modalities": ["img", "text"],
        "input_state": "exact_numeric_zero",
        "normalization_output_state": "exact_numeric_zero",
        "remove_null_comparison": "dtype_shape_c_order_bytes_exact",
        "remove_null_metric_comparison": (
            "canonical_typed_ieee754_binary64_bytes_exact"
        ),
        "signed_zero_comparison_policy": "positive_and_negative_zero_are_distinct",
        "nonfinite_comparison_policy": "nan_and_inf_forbidden",
        "halt_only": True,
        "required": True,
        "require_fixed_null_in_shuffle": True,
        "require_derived_mask_preservation": True,
        "require_null_absent_from_all_top20": True,
        "require_remove_null_exact_equivalence": True,
        "require_displacement_null_exclusion_dual_path_exact": True,
    },
    "decision_labels": [
        "CONTINUE_SAME_POOLING_CACHE_ONLY",
        "KILL_CURRENT_ENDPOINT_ROUTE_ONLY",
    ],
    "decision_config_labels": {
        "continue_label": "CONTINUE_SAME_POOLING_CACHE_ONLY",
        "kill_label": "KILL_CURRENT_ENDPOINT_ROUTE_ONLY",
    },
    "decision_schema_version": "c01_a0_decision_v2",
    "decision_schema_bindings": {
        "small_displacement_gate_selection_rule": (
            "strongest_ordinary_control_by_accuracy_then_macro_f1_then_"
            "frozen_gain_controls_order"
        ),
        "endpoint_concat_small_displacement_role": "diagnostic_only",
        "required_halt_only_validity_guards": [
            "probe_evidence_exact",
            "raw_zero_allowlist_exact",
            "derived_zero_masks_exact",
            "displacement_null_exclusion_dual_path_exact",
            "shuffle_fixed_point_bijection",
            "registered_null_absent_from_all_top20",
            "with_null_remove_null_dtype_shape_bytes_equivalence",
        ],
        "validity_guards_halt_only": True,
        "exact_array_comparison": "dtype_shape_c_order_bytes_sha256",
        "exact_metric_comparison": (
            "canonical_sorted_typed_ieee754_binary64_hex_json_sha256"
        ),
        "exact_zero_semantics": (
            "all_components_numeric_equal_zero_with_exact_registered_row_mask"
        ),
        "zero_contract_probe_other_modality_state_enum": [
            "normal_nonzero",
            "exact_zero",
            "tiny_nonzero",
            "nonfinite",
        ],
        "zero_contract_probe_exact_zero_semantics": (
            "observation_only_structural_interpretation_requires_external_"
            "evidence_and_review"
        ),
    },
    "result_schema_version": "c01_a0_result_v2",
    "output_paths": {
        "result_file": "C01_A0_OUT.json",
        "decision_file": "C01_A0_DECISION.json",
        "record_file": "refine-logs/C01_A0_RECORD.md",
        "slurm_stdout": "slurm/logs/c01_a0_v2_%j.out",
        "slurm_stderr": "slurm/logs/c01_a0_v2_%j.err",
    },
    "required_true_flags": {
        "execution.require_slurm": True,
        "execution.cpu_only": True,
        "orthogonal_rotation_control.same_block_l2": True,
        "history_parity.require_endpoint_accuracy_match": True,
        "full_sha256_preflight.required": True,
        "full_sha256_preflight.require_complete": True,
        "full_sha256_preflight.require_exact_64hex_before_torch_load": True,
        "decision.require_accuracy_gain_over_deployed_r0_context": True,
        "decision.require_primary_bootstrap_holm_reject": True,
        "decision.require_primary_above_all_rotation_controls": True,
        "decision.require_rotation_bootstrap_holm_reject": True,
        "decision.require_primary_and_displacement_above_shuffle_p95": True,
        "decision.require_shuffle_holm_reject": True,
        "decision.require_no_small_displacement_dominance": True,
        "output.atomic_json": True,
        "output.decision_exclusive_create": True,
        "output.run_namespace_no_clobber": True,
        "output.decision_schema.exclusive_create": True,
        "zero_contract_v2.required": True,
        "zero_contract_v2.require_fixed_null_in_shuffle": True,
        "zero_contract_v2.require_derived_mask_preservation": True,
        "zero_contract_v2.require_null_absent_from_all_top20": True,
        "zero_contract_v2.require_remove_null_exact_equivalence": True,
        "zero_contract_v2.require_displacement_null_exclusion_dual_path_exact": True,
    },
}
CANONICAL_CACHE_PROVENANCE = {
    ("MHC_zh", "train", "standard"): (
        579,
        16619920,
        "1d33fe5d69083479",
        "Qwen2.5-VL-7B-Instruct-LoRA_HF",
    ),
    ("MHC_zh", "train", "oneword"): (
        579,
        16619941,
        "3ad1309dc7500182",
        "Qwen2.5-VL-7B-Instruct-LoRA_HF",
    ),
    ("MHC_zh", "dev_seen", "standard"): (
        78,
        2240677,
        "a4cf072837e6fe6b",
        "Qwen2.5-VL-7B-Instruct-LoRA_HF",
    ),
    ("MHC_zh", "dev_seen", "oneword"): (
        78,
        2240698,
        "17c4efb2f7a0c2c0",
        "Qwen2.5-VL-7B-Instruct-LoRA_HF",
    ),
    ("HateMM", "train", "standard"): (
        744,
        21358913,
        "6a44cce4f65d4a60",
        "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
    ),
    ("HateMM", "train", "oneword"): (
        744,
        21358934,
        "60054f3be1204ca7",
        "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
    ),
    ("HateMM", "dev_seen", "standard"): (
        107,
        3073494,
        "92a17d42627cb4b1",
        "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
    ),
    ("HateMM", "dev_seen", "oneword"): (
        107,
        3073579,
        "07c19096a054845a",
        "Qwen2.5-VL-7B-Instruct-LoRA-curric_HF",
    ),
}
CANONICAL_HISTORY_PARITY = {
    "MHC_zh": {
        "source": "refine-logs/READOUT_SCREEN_OUT.json",
        "endpoint_std_accuracy": 0.8589743589743589,
        "endpoint_ow_accuracy": 0.8589743589743589,
        "deployed_r0_accuracy_context_only": 0.8589743589743589,
    },
    "HateMM": {
        "source": "refine-logs/READOUT_SCREEN_OUT.json",
        "endpoint_std_accuracy": 0.8411214953271028,
        "endpoint_ow_accuracy": 0.8411214953271028,
        "deployed_r0_accuracy_context_only": 0.8504672897196262,
    },
}
INPUT_ACCESS_LEDGER = []

# Imported only after the CPU/thread/Slurm guards pass.
np = None
torch = None
faiss = None


def die(message):
    raise RuntimeError(message)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default="configs/c01/c01_a0_v2.json",
        help="Repo-relative C01 A0 configuration.",
    )
    return parser.parse_args()


def repo_path(relative, field):
    if not isinstance(relative, str) or not relative:
        die("{} must be a non-empty repo-relative path".format(field))
    candidate = (REPO / relative).resolve()
    try:
        candidate.relative_to(REPO)
    except ValueError:
        die("{} escapes repository root: {}".format(field, relative))
    return candidate


def has_test_token(value):
    tokens = [t for t in re.split(r"[^a-z0-9]+", str(value).lower()) if t]
    return "test" in tokens


def reject_test_path(path, field):
    if has_test_token(str(path)):
        die("test-touch blocked in {}: {}".format(field, path))


def sha256_file(path, ledger_entry=None):
    digest = hashlib.sha256()
    if ledger_entry is not None:
        ledger_entry["hash_open_attempted"] = True
    with path.open("rb") as handle:
        if ledger_entry is not None:
            ledger_entry["hash_opened"] = True
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    if ledger_entry is not None:
        ledger_entry["bytes_hashed"] = path.stat().st_size
    return digest.hexdigest()


def stable_seed(base, *parts):
    payload = "|".join([str(base)] + [str(p) for p in parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & 0x7FFFFFFF


def rotation_arm_name(prefix, angle):
    token = ("{:.6f}".format(float(angle))).rstrip("0").rstrip(".").replace(".", "p")
    return "{}{}".format(prefix, token)


def strict_all(values, context):
    materialized = list(values)
    if not materialized:
        die("{} produced an empty binding/check family".format(context))
    return all(materialized)


def nested_value(config, dotted):
    value = config
    for part in dotted.split("."):
        if not isinstance(value, dict) or part not in value:
            die("missing frozen config flag {}".format(dotted))
        value = value[part]
    return value


def boolean_bindings(value, prefix="config"):
    found = []
    if isinstance(value, dict):
        for key, child in value.items():
            found.extend(boolean_bindings(child, prefix + "." + str(key)))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            found.extend(boolean_bindings(child, prefix + "[{}]".format(index)))
    elif isinstance(value, bool):
        found.append((prefix, value))
    return found


def binding_from_config(config):
    statistics = config["statistics"]
    decision = config["decision"]
    output = config["output"]
    preflight = config["full_sha256_preflight"]
    return {
        "run_id": config["run_id"],
        "claim_scope": config["claim_scope"],
        "positive_scope": config["positive_scope"],
        "negative_scope": config["negative_scope"],
        "namespace": output["namespace"],
        "datasets": config["inputs"]["allowed_datasets"],
        "splits": config["inputs"]["allowed_splits"],
        "arms": config["retrieval"]["arms"],
        "primary_arm": config["retrieval"]["primary_arm"],
        "secondary_arm": config["retrieval"]["secondary_arm"],
        "metrics": statistics["metrics"],
        "holm_metrics": statistics["holm_metrics"],
        "primary_controls": statistics["bootstrap_comparisons"][
            "primary_vs_controls"
        ],
        "secondary_controls": statistics["bootstrap_comparisons"][
            "secondary_vs_controls"
        ],
        "gain_controls": decision["gain_controls"],
        "minimum_gain": decision["minimum_gain_over_strongest_control"],
        "minimum_bootstrap_lower_bound": decision[
            "minimum_bootstrap_lower_bound"
        ],
        "minimum_net_fixes": decision["minimum_net_fixes"],
        "bootstrap_count": statistics["n_bootstrap"],
        "bootstrap_lower_quantile": statistics["bootstrap_lower_quantile"],
        "bootstrap_upper_quantile": statistics["bootstrap_upper_quantile"],
        "holm_alpha": statistics["holm_alpha"],
        "seed": statistics["seed"],
        "permutation_count": statistics["n_id_hash_permutations"],
        "permutation_hash": statistics["permutation_hash"],
        "permutation_pairing": statistics["permutation_pairing"],
        "angles_degrees": config["orthogonal_rotation_control"]["angles_degrees"],
        "rotation_arm_prefix": config["orthogonal_rotation_control"]["arm_prefix"],
        "rotation_distribution": config["orthogonal_rotation_control"][
            "distribution"
        ],
        "rotation_upper_bound": config["orthogonal_rotation_control"][
            "upper_bound"
        ],
        "output_budget_bytes": output["maximum_result_bytes"],
        "input_contract": {
            key: config["inputs"][key]
            for key in (
                "cache_root",
                "feature_dim",
                "exact_contract_keys",
                "standard_suffix",
                "oneword_suffix",
            )
        },
        "execution": {
            "required_cpus": config["execution"]["required_cpus"],
            "required_environment": config["execution"]["required_environment"],
        },
        "retrieval_operator": {
            key: config["retrieval"][key]
            for key in (
                "protocol",
                "topk",
                "similarity",
                "rank_weights",
                "prediction_cutoff",
                "fix_break_reference",
            )
        },
        "transform_thresholds": {
            key: config["transforms"][key]
            for key in (
                "normalization_epsilon",
                "tiny_displacement_epsilon",
                "small_displacement_train_quantile",
                "max_tiny_displacement_fraction",
                "max_small_displacement_fix_fraction",
            )
        },
        "transform_semantics": {
            key: config["transforms"][key]
            for key in (
                "endpoint_order",
                "common",
                "displacement",
                "common_interaction",
                "small_displacement_gate_reference",
                "small_displacement_endpoint_concat_role",
                "displacement_registered_null_exclusion",
                "pair_block_rule",
            )
        },
        "history_parity_tolerance": config["history_parity"][
            "absolute_tolerance"
        ],
        "manifest": {
            "required": preflight["required"],
            "schema_version": preflight["schema_version"],
            "run_id": preflight["run_id"],
            "source_set_id": preflight["source_set_id"],
            "path": preflight["manifest_path"],
            "approved_sha256": preflight["approved_manifest_sha256"],
            "producer_script": preflight["producer_script"],
            "expected_file_count": preflight["expected_file_count"],
            "required_thread_environment": preflight[
                "required_thread_environment"
            ],
            "access_ledger_entry_keys": preflight[
                "access_ledger_entry_keys"
            ],
            "slurm_job_id_pattern": preflight["slurm_job_id_pattern"],
        },
        "zero_contract_v2": config["zero_contract_v2"],
        "decision_labels": output["decision_schema"]["allowed_decisions"],
        "decision_config_labels": {
            "continue_label": decision["continue_label"],
            "kill_label": decision["kill_label"],
        },
        "decision_schema_version": output["decision_schema"]["schema_version"],
        "decision_schema_bindings": {
            key: output["decision_schema"][key]
            for key in (
                "small_displacement_gate_selection_rule",
                "endpoint_concat_small_displacement_role",
                "required_halt_only_validity_guards",
                "validity_guards_halt_only",
                "exact_array_comparison",
                "exact_metric_comparison",
                "exact_zero_semantics",
                "zero_contract_probe_other_modality_state_enum",
                "zero_contract_probe_exact_zero_semantics",
            )
        },
        "result_schema_version": output["result_schema_version"],
        "output_paths": {
            key: output[key]
            for key in (
                "result_file",
                "decision_file",
                "record_file",
                "slurm_stdout",
                "slurm_stderr",
            )
        },
        "required_true_flags": {
            dotted: nested_value(config, dotted)
            for dotted in CANONICAL_BINDING["required_true_flags"]
        },
    }


def load_config(path):
    reject_test_path(path, "config path")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    return config


def validate_config(config):
    if config.get("schema_version") != 1 or config.get("experiment_id") != "C01_A0":
        die("unsupported C01 A0 config schema/experiment")
    expected_top_level = {
        "schema_version",
        "experiment_id",
        "run_id",
        "claim_scope",
        "positive_scope",
        "negative_scope",
        "execution",
        "inputs",
        "transforms",
        "retrieval",
        "orthogonal_rotation_control",
        "statistics",
        "history_parity",
        "full_sha256_preflight",
        "zero_contract_v2",
        "decision",
        "output",
    }
    if set(config) != expected_top_level:
        die("top-level config schema changed: {}".format(sorted(set(config))))
    false_flags = [
        path for path, value in boolean_bindings(config) if value is not True
    ]
    if false_flags:
        die("all config boolean flags are frozen true; false flags={}".format(false_flags))
    actual_binding = binding_from_config(config)
    if actual_binding != CANONICAL_BINDING:
        die(
            "frozen canonical binding mismatch; config cannot weaken or alter "
            "run/science/statistics/output bindings"
        )
    if config["decision"]["required_datasets"] != CANONICAL_BINDING["datasets"]:
        die("decision required_datasets binding changed")
    if config["decision"]["required_metrics"] != CANONICAL_BINDING["holm_metrics"]:
        die("decision required_metrics binding changed")
    if (
        config["output"]["decision_schema"]["required_dataset_keys"]
        != CANONICAL_BINDING["datasets"]
    ):
        die("decision schema dataset key binding changed")

    inputs = config["inputs"]
    if tuple(inputs["allowed_datasets"]) != REQUIRED_DATASETS:
        die("allowed_datasets must be exactly {}".format(REQUIRED_DATASETS))
    if tuple(inputs["allowed_splits"]) != REQUIRED_SPLITS:
        die("allowed_splits must be exactly {}".format(REQUIRED_SPLITS))
    if set(inputs["datasets"]) != set(REQUIRED_DATASETS):
        die("dataset config must contain exactly MHC_zh and HateMM")
    for split in inputs["allowed_splits"]:
        if has_test_token(split):
            die("test split is forbidden: {}".format(split))
    if int(inputs["feature_dim"]) != 3584:
        die("C01 A0 is frozen to L24 hidden size 3584")
    if inputs["standard_suffix"] != "ro_L24" or inputs["oneword_suffix"] != "ro_ow_L24":
        die("C01 A0 permits only standard/one-word L24 cache suffixes")
    if set(inputs["exact_contract_keys"]) != {"ids", "img_feats", "text_feats", "labels"}:
        die("cache contract keys changed")

    retrieval = config["retrieval"]
    if retrieval["protocol"] != "strict_train_memory_to_dev_query":
        die("only strict train-memory -> dev-query is admissible")
    if int(retrieval["topk"]) != 20 or float(retrieval["prediction_cutoff"]) != 0.0:
        die("deployed top20/cutoff contract changed")
    if set(retrieval["arms"]) != REQUIRED_ARMS:
        die("registered arm set changed")
    if retrieval["primary_arm"] != "common_displacement":
        die("primary arm must remain common_displacement")
    if retrieval["secondary_arm"] != "common_interaction":
        die("secondary arm must remain common_interaction")

    statistics = config["statistics"]
    if int(statistics["n_id_hash_permutations"]) < 200:
        die("at least 200 label-blind ID-hash permutations are mandatory")
    if int(statistics["n_bootstrap"]) < 1000:
        die("at least 1000 paired bootstrap resamples are mandatory")
    if not (0.0 < float(statistics["bootstrap_lower_quantile"]) < 0.5):
        die("invalid bootstrap lower quantile")
    if not (0.5 < float(statistics["bootstrap_upper_quantile"]) < 1.0):
        die("invalid bootstrap upper quantile")

    transforms = config["transforms"]
    epsilon = float(transforms["normalization_epsilon"])
    tiny = float(transforms["tiny_displacement_epsilon"])
    if not (0.0 < epsilon < tiny):
        die("require 0 < normalization_epsilon < tiny_displacement_epsilon")
    if not (0.0 < float(transforms["small_displacement_train_quantile"]) < 0.5):
        die("small-displacement quantile must be fixed in (0, 0.5)")

    rotations = config["orthogonal_rotation_control"]
    angles = [float(x) for x in rotations["angles_degrees"]]
    if len(angles) < 4 or len(set(angles)) != len(angles):
        die("rotation control requires at least four unique frozen angles")
    if any(not (0.0 < x < 90.0) or abs(x - 45.0) < 1e-12 for x in angles):
        die("rotation angles must lie in (0,90) and exclude primary 45 degrees")
    if not bool(rotations["same_block_l2"]):
        die("rotation controls must use the identical block-L2 rule")

    observed_provenance = {}
    for dataset in REQUIRED_DATASETS:
        dataset_cfg = inputs["datasets"][dataset]
        for split in REQUIRED_SPLITS:
            expected = dataset_cfg["expected"][split]
            for policy in ("standard", "oneword"):
                observed_provenance[(dataset, split, policy)] = (
                    int(expected["n"]),
                    int(expected["{}_bytes".format(policy)]),
                    expected["{}_provenance_sha16".format(policy)],
                    dataset_cfg["base_tag"],
                )
    if observed_provenance != CANONICAL_CACHE_PROVENANCE:
        die("frozen cache size/sha16/base-tag provenance binding changed")
    observed_history = {
        dataset: inputs["datasets"][dataset]["historical_strict_devtrain"]
        for dataset in REQUIRED_DATASETS
    }
    if observed_history != CANONICAL_HISTORY_PARITY:
        die("frozen historical parity binding changed")

    output = config["output"]
    namespace = repo_path(output["namespace"], "output.namespace")
    expected_root = (REPO / "artifacts" / "c01_policy_contrastive" / "v2" / "a0").resolve()
    try:
        namespace.relative_to(expected_root)
    except ValueError:
        die("output namespace must be exclusive under {}".format(expected_root))
    for name in ("result_file", "decision_file"):
        value = output[name]
        if Path(value).name != value or has_test_token(value):
            die("invalid output filename {}: {}".format(name, value))


def expected_manifest_records(config):
    records = []
    inputs = config["inputs"]
    for dataset in REQUIRED_DATASETS:
        dataset_cfg = inputs["datasets"][dataset]
        for split in REQUIRED_SPLITS:
            expected = dataset_cfg["expected"][split]
            for policy in ("standard", "oneword"):
                suffix = (
                    inputs["standard_suffix"]
                    if policy == "standard"
                    else inputs["oneword_suffix"]
                )
                records.append(
                    {
                        "ordinal": len(records),
                        "dataset": dataset,
                        "split": split,
                        "policy": policy,
                        "path": "{}/{}/{}_{}-{}.pt".format(
                            inputs["cache_root"],
                            dataset,
                            split,
                            dataset_cfg["base_tag"],
                            suffix,
                        ),
                        "bytes": int(expected["{}_bytes".format(policy)]),
                        "provenance_sha16": expected[
                            "{}_provenance_sha16".format(policy)
                        ],
                    }
                )
    if len(records) != CANONICAL_BINDING["manifest"]["expected_file_count"]:
        die("canonical manifest record family is incomplete")
    return records


def load_full_sha256_manifest(config):
    preflight = config["full_sha256_preflight"]
    path = repo_path(preflight["manifest_path"], "full_sha256_preflight.manifest_path")
    reject_test_path(path, "full SHA256 manifest")
    if not path.is_file():
        die(
            "required full-SHA256 manifest is absent; run the registered Slurm "
            "hash preflight before A0"
        )
    raw = path.read_bytes()
    manifest_digest = hashlib.sha256(raw).hexdigest()
    if manifest_digest != preflight["approved_manifest_sha256"]:
        die("full-SHA256 manifest is not the exact approved artifact")
    manifest = json.loads(raw.decode("utf-8"))
    expected_top = {
        "schema_version",
        "run_id",
        "source_set_id",
        "complete",
        "producer",
        "slurm_job_id",
        "thread_environment",
        "expected_file_count",
        "files",
        "access_ledger",
        "access_summary",
    }
    if not isinstance(manifest, dict) or set(manifest) != expected_top:
        die("full-SHA256 manifest schema keys changed")
    if (
        manifest["schema_version"] != preflight["schema_version"]
        or manifest["run_id"] != preflight["run_id"]
        or manifest["source_set_id"] != preflight["source_set_id"]
        or manifest["producer"] != preflight["producer_script"]
        or manifest["complete"] is not True
        or manifest["expected_file_count"] != preflight["expected_file_count"]
    ):
        die("full-SHA256 manifest frozen identity/completeness mismatch")
    slurm_job_id = manifest["slurm_job_id"]
    if (
        not isinstance(slurm_job_id, str)
        or not re.fullmatch(preflight["slurm_job_id_pattern"], slurm_job_id)
    ):
        die("full-SHA256 manifest has an invalid Slurm job ID")
    if manifest["thread_environment"] != preflight["required_thread_environment"]:
        die("full-SHA256 manifest thread environment mismatch")
    expected = expected_manifest_records(config)
    files = manifest["files"]
    if not isinstance(files, list) or len(files) != len(expected):
        die("full-SHA256 manifest file count mismatch")
    manifest_by_path = {}
    required_file_keys = {
        "ordinal",
        "dataset",
        "split",
        "policy",
        "path",
        "bytes",
        "provenance_sha16",
        "sha256",
    }
    for registered, observed in zip(expected, files):
        if not isinstance(observed, dict) or set(observed) != required_file_keys:
            die("full-SHA256 manifest file-entry schema changed")
        observed_without_sha = {
            key: observed[key] for key in registered
        }
        if observed_without_sha != registered:
            die("full-SHA256 manifest file identity/provenance mismatch")
        digest = observed["sha256"]
        if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
            die("manifest lacks exact 64-hex SHA256 for {}".format(observed["path"]))
        if not digest.startswith(observed["provenance_sha16"]):
            die("manifest full SHA256 does not match registered sha16 provenance")
        if observed["path"] in manifest_by_path:
            die("duplicate path in full-SHA256 manifest")
        manifest_by_path[observed["path"]] = digest
    access_summary = manifest["access_summary"]
    if (
        not isinstance(access_summary, dict)
        or access_summary.get("opened_count") != len(expected)
        or access_summary.get("test_like_attempt_count") != 0
        or access_summary.get("test_like_open_count") != 0
    ):
        die("hash-preflight access summary is incomplete or test-tainted")
    access_ledger = manifest["access_ledger"]
    if not isinstance(access_ledger, list) or len(access_ledger) != len(expected):
        die("hash-preflight access ledger is incomplete")
    required_ledger_keys = set(preflight["access_ledger_entry_keys"])
    if len(required_ledger_keys) != len(preflight["access_ledger_entry_keys"]):
        die("hash-preflight access ledger key binding contains duplicates")
    for registered, observed in zip(expected, access_ledger):
        if not isinstance(observed, dict) or set(observed) != required_ledger_keys:
            die("hash-preflight access ledger entry schema changed")
        if (
            type(observed["ordinal"]) is not int
            or type(observed["bytes_read"]) is not int
            or observed["open_attempted"] is not True
            or observed["opened"] is not True
            or observed["test_like"] is not False
        ):
            die("hash-preflight access ledger entry types/flags changed")
        expected_ledger_entry = {
            "ordinal": registered["ordinal"],
            "path": registered["path"],
            "dataset": registered["dataset"],
            "split": registered["split"],
            "policy": registered["policy"],
            "open_attempted": True,
            "opened": True,
            "bytes_read": registered["bytes"],
            "test_like": False,
        }
        if observed != expected_ledger_entry:
            die(
                "hash-preflight access ledger identity/order/access mismatch "
                "at ordinal {}".format(registered["ordinal"])
            )
    return {
        "path": str(path.relative_to(REPO)),
        "sha256": manifest_digest,
        "slurm_job_id": slurm_job_id,
        "thread_environment": manifest["thread_environment"],
        "files": manifest_by_path,
        "access_ledger": access_ledger,
        "access_summary": access_summary,
    }


def load_zero_contract_evidence(config):
    contract = config["zero_contract_v2"]
    path = repo_path(contract["probe_path"], "zero_contract_v2.probe_path")
    reject_test_path(path, "zero-contract probe artifact")
    if not path.is_file():
        die("required zero-contract probe artifact is absent")
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != contract["probe_sha256"]:
        die("zero-contract probe artifact SHA256 mismatch")
    artifact = json.loads(raw.decode("utf-8"))
    if (
        artifact.get("schema_version") != contract["probe_schema_version"]
        or artifact.get("run_id") != contract["probe_run_id"]
        or artifact.get("status") != "DIAGNOSTIC_ONLY"
        or artifact.get("hash_manifest", {}).get("sha256")
        != config["full_sha256_preflight"]["approved_manifest_sha256"]
    ):
        die("zero-contract probe identity/provenance mismatch")
    assessment = artifact.get("v2_repair_preregistered_assessment", {})
    if (
        assessment.get("allow_zero_block_in_a0") is not False
        or assessment.get("endpoint_zero_masks_exact_match") is not True
        or assessment.get("non_structural_tiny_absent") is not True
        or assessment.get("nonfinite_rows_absent") is not True
    ):
        die("zero-contract probe necessary-condition record mismatch")
    for dataset in REQUIRED_DATASETS:
        for split in REQUIRED_SPLITS:
            split_record = artifact["datasets"][dataset]["splits"][split]
            if (
                split_record.get("endpoint_ids_exact_match") is not True
                or split_record.get("endpoint_labels_exact_match") is not True
            ):
                die("zero-contract probe endpoint alignment mismatch")
            for modality in ("img", "text"):
                comparison = split_record["endpoint_comparison"][modality]
                expected_ids = (
                    [contract["authorized_id"]]
                    if dataset == contract["authorized_dataset"]
                    and split == contract["authorized_split"]
                    else []
                )
                if (
                    comparison.get("zero_mask_exact_match") is not True
                    or comparison.get("matched_zero_ids") != expected_ids
                    or comparison.get("standard_only_zero_ids") != []
                    or comparison.get("oneword_only_zero_ids") != []
                ):
                    die("zero-contract probe zero-mask record mismatch")
                for policy in ("standard", "oneword"):
                    cell = split_record["policies"][policy]["modalities"][modality]
                    expected_rows = (
                        [{
                            "id": contract["authorized_id"],
                            "label": contract["expected_label_integrity_only"],
                            "other_modality_state": "exact_zero",
                            "row_index": contract["authorized_row_index"],
                        }]
                        if expected_ids
                        else []
                    )
                    if (
                        cell.get("zero_rows") != expected_rows
                        or cell.get("zero_row_count") != len(expected_rows)
                        or cell.get("tiny_nonzero_rows") != []
                        or cell.get("tiny_nonzero_row_count") != 0
                        or cell.get("nonfinite_rows") != []
                        or cell.get("nonfinite_row_count") != 0
                    ):
                        die("zero-contract probe per-cell record mismatch")
    return {
        "path": str(path.relative_to(REPO)),
        "sha256": digest,
        "schema_version": artifact["schema_version"],
        "run_id": artifact["run_id"],
        "authorized_tuple": {
            key: contract[key]
            for key in (
                "authorized_dataset",
                "authorized_split",
                "authorized_id",
                "authorized_row_index",
                "expected_label_integrity_only",
                "authorized_policies",
                "authorized_modalities",
            )
        },
    }


def enforce_runtime(config):
    execution = config["execution"]
    if not execution.get("require_slurm") or not os.environ.get("SLURM_JOB_ID"):
        die("C01 A0 computation is Slurm-only; SLURM_JOB_ID is required")
    if not execution.get("cpu_only"):
        die("C01 A0 must remain CPU-only")
    for key, expected in execution["required_environment"].items():
        actual = os.environ.get(key)
        if actual != expected:
            die("thread/device guard failed: {}={!r}, expected {!r}".format(key, actual, expected))
    cpus = int(os.environ.get("SLURM_CPUS_PER_TASK", "0"))
    if cpus != int(execution["required_cpus"]):
        die("SLURM_CPUS_PER_TASK={} does not match required_cpus".format(cpus))


def import_compute_modules(config):
    global np, torch, faiss
    import numpy as imported_numpy
    import torch as imported_torch
    import faiss as imported_faiss

    np = imported_numpy
    torch = imported_torch
    faiss = imported_faiss
    threads = int(config["execution"]["required_cpus"])
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    faiss.omp_set_num_threads(threads)
    if torch.cuda.is_available():
        die("CUDA is visible despite the CPU-only guard")


def normalize_ids(raw, context):
    if not isinstance(raw, (list, tuple)) or len(raw) != 1:
        die("{} ids must be the extractor's one-sublist contract".format(context))
    ids = raw[0]
    if not isinstance(ids, (list, tuple)):
        die("{} ids[0] is not a list".format(context))
    normalized = list(ids)
    if (
        any(type(x) is not str or not x for x in normalized)
        or len(set(normalized)) != len(normalized)
    ):
        die("{} ids must be raw non-empty unique strings".format(context))
    return normalized


def load_cache(config, dataset, split, policy, manifest_files):
    inputs = config["inputs"]
    if dataset not in REQUIRED_DATASETS or split not in REQUIRED_SPLITS:
        die("blocked dataset/split request: {}/{}".format(dataset, split))
    if has_test_token(split):
        die("test split request blocked: {}".format(split))
    suffix = inputs["standard_suffix"] if policy == "standard" else inputs["oneword_suffix"]
    if policy not in ("standard", "oneword"):
        die("unknown endpoint policy {}".format(policy))
    ds_cfg = inputs["datasets"][dataset]
    filename = "{}_{}-{}.pt".format(split, ds_cfg["base_tag"], suffix)
    expected_name = filename
    cache_root = repo_path(inputs["cache_root"], "inputs.cache_root")
    path = (cache_root / dataset / filename).resolve()
    reject_test_path(path, "cache input")
    try:
        path.relative_to((cache_root / dataset).resolve())
    except ValueError:
        die("cache path escapes dataset directory: {}".format(path))
    if path.name != expected_name or not path.is_file():
        die("missing or non-exact cache path: {}".format(path))

    expected = ds_cfg["expected"][split]
    prefix_key = "{}_provenance_sha16".format(policy)
    bytes_key = "{}_bytes".format(policy)
    stat_bytes = path.stat().st_size
    if stat_bytes != int(expected[bytes_key]):
        die("{} byte-size mismatch: {} != {}".format(path, stat_bytes, expected[bytes_key]))
    relative_path = str(path.relative_to(REPO))
    ledger = {
        "ordinal": len(INPUT_ACCESS_LEDGER),
        "dataset": dataset,
        "split": split,
        "policy": policy,
        "path": relative_path,
        "test_like": has_test_token(relative_path),
        "hash_open_attempted": False,
        "hash_opened": False,
        "bytes_hashed": 0,
        "manifest_sha256_matched_before_torch_load": False,
        "torch_load_attempted": False,
        "torch_loaded": False,
    }
    INPUT_ACCESS_LEDGER.append(ledger)
    if ledger["test_like"]:
        die("test-like path reached the A0 access ledger")
    digest = sha256_file(path, ledger)
    if not digest.startswith(expected[prefix_key]):
        die("{} sha16 provenance mismatch".format(path))
    if relative_path not in manifest_files:
        die("{} absent from full-SHA256 preflight manifest".format(relative_path))
    if digest != manifest_files[relative_path]:
        die("{} exact full-SHA256 manifest mismatch".format(relative_path))
    ledger["manifest_sha256_matched_before_torch_load"] = True

    ledger["torch_load_attempted"] = True
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    ledger["torch_loaded"] = True
    if not isinstance(payload, dict) or set(payload) != set(inputs["exact_contract_keys"]):
        die("{} violates exact cache contract".format(path))
    context = "{}/{}/{}".format(dataset, split, policy)
    ids = normalize_ids(payload["ids"], context)
    img = payload["img_feats"]
    text = payload["text_feats"]
    labels = payload["labels"]
    if not strict_all(
        (torch.is_tensor(x) for x in (img, text, labels)),
        context + "/tensor_contract",
    ):
        die("{} cache tensors are malformed".format(context))
    img = img.detach().cpu().float().numpy()
    text = text.detach().cpu().float().numpy()
    labels = labels.detach().cpu().numpy()
    n = int(expected["n"])
    dim = int(inputs["feature_dim"])
    if img.shape != (n, dim) or text.shape != (n, dim) or labels.shape != (n,):
        die("{} shape mismatch: img={} text={} labels={}".format(
            context, img.shape, text.shape, labels.shape
        ))
    if len(ids) != n:
        die("{} ID count mismatch".format(context))
    if not np.isfinite(img).all() or not np.isfinite(text).all() or not np.isfinite(labels).all():
        die("{} contains NaN/Inf".format(context))
    rounded = np.rint(labels).astype("int64")
    if not np.array_equal(labels, rounded) or not set(rounded.tolist()).issubset({0, 1}):
        die("{} labels are not binary integers".format(context))
    return {
        "ids": ids,
        "img": img.astype("float32", copy=False),
        "text": text.astype("float32", copy=False),
        "labels": rounded,
        "path": relative_path,
        "sha256": digest,
        "bytes": stat_bytes,
    }


def l2_rows(array, epsilon, context, zero_mask=None):
    norms = np.linalg.norm(array, axis=1)
    if not np.isfinite(norms).all():
        die("{} has non-finite row norms".format(context))
    if zero_mask is None:
        zero_mask = np.zeros(len(array), dtype=bool)
    zero_mask = np.asarray(zero_mask, dtype=bool)
    if zero_mask.shape != (len(array),):
        die("{} zero-mask shape mismatch".format(context))
    exact_zero = np.all(array == 0, axis=1)
    if not np.array_equal(exact_zero, zero_mask):
        die("{} exact-zero mask diverged from authorized mask".format(context))
    bad = np.flatnonzero((norms <= epsilon) & ~zero_mask)
    if len(bad):
        die("{} has {} rows at/below epsilon {}; first={}".format(
            context, len(bad), epsilon, int(bad[0])
        ))
    normalized = np.zeros_like(array, dtype="float32")
    keep = ~zero_mask
    normalized[keep] = (array[keep] / norms[keep, None]).astype("float32")
    if not np.array_equal(np.all(normalized == 0, axis=1), zero_mask):
        die("{} normalization did not preserve the exact-zero mask".format(context))
    return normalized, norms.astype("float64")


def fuse_modalities(img, text, epsilon, context, zero_mask=None):
    img, _ = l2_rows(img, epsilon, context + "/img", zero_mask)
    text, _ = l2_rows(text, epsilon, context + "/text", zero_mask)
    fused, _ = l2_rows(
        np.concatenate([img, text], axis=1),
        epsilon,
        context + "/fused",
        zero_mask,
    )
    return fused


def paired_key(img_a, img_b, text_a, text_b, epsilon, context, zero_mask=None):
    img_a, _ = l2_rows(img_a, epsilon, context + "/img_a", zero_mask)
    img_b, _ = l2_rows(img_b, epsilon, context + "/img_b", zero_mask)
    text_a, _ = l2_rows(text_a, epsilon, context + "/text_a", zero_mask)
    text_b, _ = l2_rows(text_b, epsilon, context + "/text_b", zero_mask)
    img_pair, _ = l2_rows(
        np.concatenate([img_a, img_b], axis=1),
        epsilon,
        context + "/img_pair",
        zero_mask,
    )
    text_pair, _ = l2_rows(
        np.concatenate([text_a, text_b], axis=1),
        epsilon,
        context + "/text_pair",
        zero_mask,
    )
    return fuse_modalities(
        img_pair, text_pair, epsilon, context + "/pair_fused", zero_mask
    )


def contrast_blocks(normed, epsilon, context, zero_mask=None):
    common = {}
    displacement = {}
    displacement_norm = {}
    common_interaction = {}
    for modality in ("img", "text"):
        standard = normed["standard"][modality]
        oneword = normed["oneword"][modality]
        common[modality], _ = l2_rows(
            standard + oneword,
            epsilon,
            context + "/" + modality + "/common",
            zero_mask,
        )
        raw_d = oneword - standard
        displacement[modality], displacement_norm[modality] = l2_rows(
            raw_d,
            epsilon,
            context + "/" + modality + "/displacement",
            zero_mask,
        )
        common_interaction[modality], _ = l2_rows(
            common[modality] * displacement[modality],
            epsilon,
            context + "/" + modality + "/common_interaction",
            zero_mask,
        )
    return common, displacement, common_interaction, displacement_norm


def orthogonal_blocks(normed, angle_degrees, epsilon, context, zero_mask=None):
    theta = math.radians(float(angle_degrees))
    cosine, sine = math.cos(theta), math.sin(theta)
    first, second = {}, {}
    for modality in ("img", "text"):
        standard = normed["standard"][modality]
        oneword = normed["oneword"][modality]
        first[modality], _ = l2_rows(
            cosine * standard + sine * oneword,
            epsilon,
            context + "/" + modality + "/rot_first",
            zero_mask,
        )
        second[modality], _ = l2_rows(
            -sine * standard + cosine * oneword,
            epsilon,
            context + "/" + modality + "/rot_second",
            zero_mask,
        )
    return first, second


def prepare_views(standard, oneword, config, context, zero_mask):
    epsilon = float(config["transforms"]["normalization_epsilon"])
    normed = {"standard": {}, "oneword": {}}
    for policy, cache in (("standard", standard), ("oneword", oneword)):
        for modality in ("img", "text"):
            normed[policy][modality], _ = l2_rows(
                cache[modality],
                epsilon,
                context + "/" + policy + "/" + modality,
                zero_mask,
            )

    common, displacement, common_interaction, d_norm = contrast_blocks(
        normed, epsilon, context, zero_mask
    )
    views = {
        "endpoint_std": fuse_modalities(
            normed["standard"]["img"], normed["standard"]["text"], epsilon,
            context + "/endpoint_std", zero_mask,
        ),
        "endpoint_ow": fuse_modalities(
            normed["oneword"]["img"], normed["oneword"]["text"], epsilon,
            context + "/endpoint_ow", zero_mask,
        ),
        "endpoint_concat": paired_key(
            normed["standard"]["img"], normed["oneword"]["img"],
            normed["standard"]["text"], normed["oneword"]["text"],
            epsilon, context + "/endpoint_concat", zero_mask,
        ),
        "common": fuse_modalities(
            common["img"], common["text"], epsilon, context + "/common", zero_mask
        ),
        "displacement": fuse_modalities(
            displacement["img"], displacement["text"], epsilon,
            context + "/displacement", zero_mask,
        ),
        "common_displacement": paired_key(
            common["img"], displacement["img"], common["text"], displacement["text"],
            epsilon, context + "/common_displacement", zero_mask,
        ),
        "common_interaction": paired_key(
            common["img"], common_interaction["img"],
            common["text"], common_interaction["text"],
            epsilon, context + "/common_interaction", zero_mask,
        ),
    }

    rotation_cfg = config["orthogonal_rotation_control"]
    rotation_names = []
    for angle in rotation_cfg["angles_degrees"]:
        first, second = orthogonal_blocks(
            normed, angle, epsilon, context + "/orthrot_{}".format(angle),
            zero_mask,
        )
        name = rotation_arm_name(rotation_cfg["arm_prefix"], angle)
        views[name] = paired_key(
            first["img"], second["img"], first["text"], second["text"],
            epsilon, context + "/" + name, zero_mask,
        )
        rotation_names.append(name)

    # Algebra guards: theta=0 is endpoint_concat and theta=45 is the primary
    # [common, displacement] object after identical per-block L2 normalization.
    rot0_a, rot0_b = orthogonal_blocks(
        normed, 0.0, epsilon, context + "/guard_rot0", zero_mask
    )
    rot45_a, rot45_b = orthogonal_blocks(
        normed, 45.0, epsilon, context + "/guard_rot45", zero_mask
    )
    endpoint0 = paired_key(
        rot0_a["img"], rot0_b["img"], rot0_a["text"], rot0_b["text"],
        epsilon, context + "/guard_endpoint0", zero_mask,
    )
    endpoint45 = paired_key(
        rot45_a["img"], rot45_b["img"], rot45_a["text"], rot45_b["text"],
        epsilon, context + "/guard_endpoint45", zero_mask,
    )
    algebra_guard = {
        "endpoint_concat_vs_theta0_max_abs": float(np.max(np.abs(
            views["endpoint_concat"] - endpoint0
        ))),
        "common_displacement_vs_theta45_max_abs": float(np.max(np.abs(
            views["common_displacement"] - endpoint45
        ))),
    }
    if max(algebra_guard.values()) > 2e-6:
        die("{} algebra parity guard failed: {}".format(context, algebra_guard))
    derived_masks = {
        arm: bool(np.array_equal(np.all(value == 0, axis=1), zero_mask))
        for arm, value in views.items()
    }
    if not strict_all(derived_masks.values(), context + "/derived_zero_masks"):
        die("{} derived exact-zero mask preservation failed".format(context))
    return views, normed, d_norm, rotation_names, algebra_guard, derived_masks


def shuffled_contrast_views(
    standard_normed, oneword_normed, permutation, config, context, zero_mask
):
    epsilon = float(config["transforms"]["normalization_epsilon"])
    paired = {
        "standard": standard_normed["standard"],
        "oneword": {
            "img": oneword_normed["oneword"]["img"][permutation],
            "text": oneword_normed["oneword"]["text"][permutation],
        },
    }
    common, displacement, _, _ = contrast_blocks(
        paired, epsilon, context, zero_mask
    )
    return {
        "displacement": fuse_modalities(
            displacement["img"], displacement["text"], epsilon,
            context + "/displacement", zero_mask,
        ),
        "common_displacement": paired_key(
            common["img"], displacement["img"], common["text"], displacement["text"],
            epsilon, context + "/common_displacement", zero_mask,
        ),
    }


def weighted_signed_scores(memory, memory_labels, query, config):
    topk = int(config["retrieval"]["topk"])
    if len(memory) < topk:
        die("memory has fewer than topk rows")
    memory = np.ascontiguousarray(memory.astype("float32", copy=True))
    query = np.ascontiguousarray(query.astype("float32", copy=True))
    faiss.normalize_L2(memory)
    faiss.normalize_L2(query)
    index = faiss.IndexFlatIP(memory.shape[1])
    index.add(memory)
    similarities, neighbors = index.search(query, topk)
    if (
        neighbors.shape != (len(query), topk)
        or np.any(neighbors < 0)
        or np.any(neighbors >= len(memory))
    ):
        die("FAISS returned malformed or out-of-range top20 neighbor indices")
    weights = np.arange(topk, 0, -1, dtype="float64")
    signed = memory_labels[neighbors].astype("float64") * 2.0 - 1.0
    scores = np.sum(signed * similarities.astype("float64") * weights[None, :], axis=1)
    scores /= float(np.sum(weights))
    if not np.isfinite(scores).all():
        die("non-finite kNN scores")
    return {
        "scores": scores,
        "neighbors": neighbors.astype("int64", copy=False),
        "similarities": similarities.astype("float32", copy=False),
    }


def roc_auc(gold, scores):
    positives = int(np.sum(gold == 1))
    negatives = int(np.sum(gold == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype="float64")
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop
    rank_sum = float(np.sum(ranks[gold == 1]))
    return (rank_sum - positives * (positives + 1) / 2.0) / (positives * negatives)


def metric_bundle(gold, scores, cutoff):
    predictions = (scores >= cutoff).astype("int64")
    accuracy = float(np.mean(predictions == gold))
    f1s = []
    confusion = {}
    for label in (0, 1):
        tp = int(np.sum((predictions == label) & (gold == label)))
        fp = int(np.sum((predictions == label) & (gold != label)))
        fn = int(np.sum((predictions != label) & (gold == label)))
        denominator = 2 * tp + fp + fn
        f1s.append(0.0 if denominator == 0 else 2.0 * tp / denominator)
        confusion[str(label)] = {"tp": tp, "fp": fp, "fn": fn}
    return {
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": float(np.mean(f1s)),
            "roc_auc": roc_auc(gold, scores),
        },
        "predictions": predictions,
        "scores": scores,
        "confusion": confusion,
    }


def fix_break(candidate_predictions, reference_predictions, gold, mask=None):
    if mask is None:
        mask = np.ones(len(gold), dtype=bool)
    candidate_ok = candidate_predictions == gold
    reference_ok = reference_predictions == gold
    fixed = int(np.sum(mask & candidate_ok & ~reference_ok))
    broken = int(np.sum(mask & ~candidate_ok & reference_ok))
    return {"fixed": fixed, "broken": broken, "net": fixed - broken, "n": int(np.sum(mask))}


def summarize_distribution(values):
    quantiles = (0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
    return {
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "quantiles": {
            str(q): float(np.quantile(values, q)) for q in quantiles
        },
    }


def array_bytewise_audit(left, right, context):
    left = np.asarray(left)
    right = np.asarray(right)
    if left.dtype.kind == "f" and not np.isfinite(left).all():
        die("{} left array contains non-finite floats".format(context))
    if right.dtype.kind == "f" and not np.isfinite(right).all():
        die("{} right array contains non-finite floats".format(context))
    left_bytes = np.ascontiguousarray(left).tobytes(order="C")
    right_bytes = np.ascontiguousarray(right).tobytes(order="C")
    audit = {
        "dtype_exact": left.dtype.str == right.dtype.str,
        "shape_exact": left.shape == right.shape,
        "c_order_bytes_exact": left_bytes == right_bytes,
        "left": {
            "dtype": left.dtype.str,
            "shape": list(left.shape),
            "nbytes": len(left_bytes),
            "sha256": hashlib.sha256(left_bytes).hexdigest(),
        },
        "right": {
            "dtype": right.dtype.str,
            "shape": list(right.shape),
            "nbytes": len(right_bytes),
            "sha256": hashlib.sha256(right_bytes).hexdigest(),
        },
        "signed_zero_policy": "+0.0 and -0.0 are distinct C-order byte strings",
        "nonfinite_policy": "forbidden_for_float_arrays",
    }
    audit["pass"] = bool(
        audit["dtype_exact"]
        and audit["shape_exact"]
        and audit["c_order_bytes_exact"]
    )
    return audit


def float64_bytewise_audit(left, right, context):
    left = float(left)
    right = float(right)
    if not math.isfinite(left) or not math.isfinite(right):
        die("{} scalar comparison contains NaN/Inf".format(context))
    left_bytes = struct.pack(">d", left)
    right_bytes = struct.pack(">d", right)
    return {
        "pass": left_bytes == right_bytes,
        "encoding": "IEEE-754 binary64 big-endian bytes",
        "left_hex": left_bytes.hex(),
        "right_hex": right_bytes.hex(),
        "left_sha256": hashlib.sha256(left_bytes).hexdigest(),
        "right_sha256": hashlib.sha256(right_bytes).hexdigest(),
        "signed_zero_policy": "+0.0 and -0.0 have distinct encodings",
        "nonfinite_policy": "NaN/Inf forbidden",
    }


def canonical_metric_audit(left, right, context):
    def encode(metrics, side):
        if not isinstance(metrics, dict):
            die("{} {} metrics are not a dict".format(context, side))
        typed = {}
        for name in sorted(metrics):
            value = metrics[name]
            if value is None:
                typed[name] = {"type": "none"}
                continue
            if not isinstance(value, float) or not math.isfinite(value):
                die("{} {} metric {} is non-float or non-finite".format(
                    context, side, name
                ))
            typed[name] = {
                "type": "ieee754_binary64",
                "big_endian_hex": struct.pack(">d", value).hex(),
            }
        payload = json.dumps(
            typed, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode("utf-8")
        return typed, payload

    left_typed, left_payload = encode(left, "left")
    right_typed, right_payload = encode(right, "right")
    return {
        "pass": left_payload == right_payload,
        "serialization": (
            "canonical sorted compact JSON over typed IEEE-754 binary64 "
            "big-endian hex values"
        ),
        "left_sha256": hashlib.sha256(left_payload).hexdigest(),
        "right_sha256": hashlib.sha256(right_payload).hexdigest(),
        "left_typed": left_typed,
        "right_typed": right_typed,
        "signed_zero_policy": "+0.0 and -0.0 have distinct encodings",
        "nonfinite_policy": "NaN/Inf forbidden; explicit None remains typed none",
    }


def retrieval_without_registered_null(
    memory, memory_labels, query, full_retrieval, null_index, config, context
):
    if null_index is None:
        return {"status": "NO_REGISTERED_NULL", "pass": True}, None
    if np.any(full_retrieval["neighbors"] == null_index):
        die("{} registered null entered dev top20".format(context))
    keep = np.ones(len(memory), dtype=bool)
    keep[null_index] = False
    reduced = weighted_signed_scores(
        memory[keep], memory_labels[keep], query, config
    )
    original_indices = np.flatnonzero(keep)
    mapped_neighbors = original_indices[reduced["neighbors"]]
    comparisons = {
        "neighbors": array_bytewise_audit(
            full_retrieval["neighbors"], mapped_neighbors,
            context + "/neighbors",
        ),
        "similarities": array_bytewise_audit(
            full_retrieval["similarities"], reduced["similarities"],
            context + "/similarities",
        ),
        "scores": array_bytewise_audit(
            full_retrieval["scores"], reduced["scores"],
            context + "/scores",
        ),
    }
    checks = {
        name + "_dtype_shape_bytes_exact": comparison["pass"]
        for name, comparison in comparisons.items()
    }
    if not strict_all(checks.values(), context + "/remove_null_exact"):
        die("{} with-null/remove-null retrieval mismatch".format(context))
    audit = {
        "status": "PASS",
        "pass": True,
        "registered_null_index": int(null_index),
        "registered_null_top20_count": 0,
        "checks": checks,
        "comparisons": comparisons,
    }
    reduced["neighbors"] = mapped_neighbors
    return audit, reduced


def evaluation_exact_equivalence(full, reduced, gold, cutoff, context):
    reduced_bundle = metric_bundle(gold, reduced["scores"], cutoff)
    comparisons = {
        "scores": array_bytewise_audit(
            full["scores"], reduced["scores"], context + "/scores"
        ),
        "predictions": array_bytewise_audit(
            full["predictions"], reduced_bundle["predictions"],
            context + "/predictions",
        ),
        "metrics": canonical_metric_audit(
            full["metrics"], reduced_bundle["metrics"], context + "/metrics"
        ),
    }
    checks = {
        "scores_dtype_shape_bytes_exact": comparisons["scores"]["pass"],
        "predictions_dtype_shape_bytes_exact": comparisons["predictions"]["pass"],
        "metrics_canonical_typed_bytes_exact": comparisons["metrics"]["pass"],
    }
    if not strict_all(checks.values(), context + "/evaluation_exact"):
        die("{} with-null/remove-null evaluation mismatch".format(context))
    return {"checks": checks, "comparisons": comparisons, "pass": True}


def evaluate_real_arms(
    train_views, dev_views, train_labels, dev_labels, config,
    registered_null_index,
):
    cutoff = float(config["retrieval"]["prediction_cutoff"])
    evaluations = {}
    validity = {}
    reduced_by_arm = {}
    for arm in train_views:
        retrieval = weighted_signed_scores(
            train_views[arm], train_labels, dev_views[arm], config
        )
        evaluations[arm] = metric_bundle(
            dev_labels, retrieval["scores"], cutoff
        )
        evaluations[arm]["neighbors"] = retrieval["neighbors"]
        evaluations[arm]["similarities"] = retrieval["similarities"]
        validity[arm], reduced = retrieval_without_registered_null(
            train_views[arm], train_labels, dev_views[arm], retrieval,
            registered_null_index, config, "real/{}".format(arm),
        )
        if reduced is not None:
            validity[arm]["evaluation_checks"] = evaluation_exact_equivalence(
                evaluations[arm], reduced, dev_labels, cutoff,
                "real/{}".format(arm),
            )
            reduced_by_arm[arm] = reduced
    average_scores = 0.5 * (
        evaluations["endpoint_std"]["scores"] + evaluations["endpoint_ow"]["scores"]
    )
    evaluations["avg_score"] = metric_bundle(dev_labels, average_scores, cutoff)
    if registered_null_index is None:
        validity["avg_score"] = {"status": "NO_REGISTERED_NULL", "pass": True}
    else:
        reduced_average_scores = 0.5 * (
            reduced_by_arm["endpoint_std"]["scores"]
            + reduced_by_arm["endpoint_ow"]["scores"]
        )
        reduced_average = {"scores": reduced_average_scores}
        validity["avg_score"] = {
            "status": "PASS",
            "pass": True,
            "source_neighbor_arms": ["endpoint_std", "endpoint_ow"],
            "evaluation_checks": evaluation_exact_equivalence(
                evaluations["avg_score"], reduced_average, dev_labels, cutoff,
                "real/avg_score",
            ),
        }

    reference = evaluations[config["retrieval"]["fix_break_reference"]]["predictions"]
    public = {}
    for arm, value in evaluations.items():
        public[arm] = {
            "metrics": value["metrics"],
            "confusion": value["confusion"],
            "fix_break_vs_endpoint_std": fix_break(
                value["predictions"], reference, dev_labels
            ),
        }
    return evaluations, public, validity


def metric_value(metric, gold, scores, cutoff):
    return metric_bundle(gold, scores, cutoff)["metrics"][metric]


def paired_bootstrap(candidate, control, gold, config, seed):
    count = int(config["statistics"]["n_bootstrap"])
    lower_q = float(config["statistics"]["bootstrap_lower_quantile"])
    upper_q = float(config["statistics"]["bootstrap_upper_quantile"])
    cutoff = float(config["retrieval"]["prediction_cutoff"])
    rng = np.random.default_rng(seed)
    deltas = {metric: [] for metric in config["statistics"]["metrics"]}
    for _ in range(count):
        sampled = rng.integers(0, len(gold), size=len(gold))
        sampled_gold = gold[sampled]
        for metric in deltas:
            cand = metric_value(metric, sampled_gold, candidate["scores"][sampled], cutoff)
            ctrl = metric_value(metric, sampled_gold, control["scores"][sampled], cutoff)
            if cand is not None and ctrl is not None:
                deltas[metric].append(cand - ctrl)
    output = {}
    for metric, values in deltas.items():
        array = np.asarray(values, dtype="float64")
        if len(array) != count:
            die("bootstrap {} produced class-degenerate resamples".format(metric))
        output[metric] = {
            "observed_delta": (
                candidate["metrics"][metric] - control["metrics"][metric]
            ),
            "bootstrap_mean": float(np.mean(array)),
            "lower": float(np.quantile(array, lower_q)),
            "upper": float(np.quantile(array, upper_q)),
            "one_sided_raw_p": float((1 + np.sum(array <= 0.0)) / (len(array) + 1)),
            "n": int(len(array)),
        }
    return output


def holm_adjust(entries, alpha):
    """Mutate entries [(id, summary), ...] with global Holm adjusted p-values."""
    ordered = sorted(entries, key=lambda x: x[1]["one_sided_raw_p"])
    total = len(ordered)
    running = 0.0
    for rank, (_, summary) in enumerate(ordered):
        adjusted = min(1.0, (total - rank) * summary["one_sided_raw_p"])
        running = max(running, adjusted)
        summary["holm_adjusted_p"] = float(running)
        summary["holm_reject"] = bool(running <= alpha)


def id_hash_permutation(ids, dataset, split, draw, seed, fixed_indices=()):
    fixed = tuple(sorted(set(int(index) for index in fixed_indices)))
    if any(index < 0 or index >= len(ids) for index in fixed):
        die("registered fixed permutation index is out of bounds")
    fixed_set = set(fixed)
    keyed = []
    for index, item_id in enumerate(ids):
        if index in fixed_set:
            continue
        payload = "{}|{}|{}|{}|{}".format(
            seed, draw, dataset, split, item_id
        ).encode("utf-8")
        keyed.append((hashlib.sha256(payload).digest(), str(item_id), index))
    movable_destinations = [
        index for index in range(len(ids)) if index not in fixed_set
    ]
    movable_sources = [x[2] for x in sorted(keyed)]
    order = np.arange(len(ids), dtype="int64")
    order[movable_destinations] = movable_sources
    checks = {
        "bijection": bool(np.array_equal(
            np.sort(order), np.arange(len(ids), dtype="int64")
        )),
        "fixed_indices_unchanged": bool(
            all(int(order[index]) == index for index in fixed)
        ),
        "movable_destinations_exclude_fixed_sources": bool(
            not fixed_set.intersection(
                int(order[index]) for index in movable_destinations
            )
        ),
    }
    if not strict_all(
        checks.values(), "{}/{}/draw_{}/permutation".format(dataset, split, draw)
    ):
        die("label-blind fixed-point permutation contract failed")
    return order


def permutation_null(
    dataset, caches, normed, real_evaluations, config, zero_masks
):
    count = int(config["statistics"]["n_id_hash_permutations"])
    seed = int(config["statistics"]["seed"])
    targets = ("common_displacement", "displacement")
    null_values = {
        target: {metric: [] for metric in config["statistics"]["metrics"]}
        for target in targets
    }
    permutation_digest = hashlib.sha256()
    cutoff = float(config["retrieval"]["prediction_cutoff"])
    train_labels = caches["train"]["standard"]["labels"]
    dev_labels = caches["dev_seen"]["standard"]["labels"]
    registered_null_index = (
        int(config["zero_contract_v2"]["authorized_row_index"])
        if dataset == config["zero_contract_v2"]["authorized_dataset"]
        else None
    )
    train_fixed = (
        (registered_null_index,) if registered_null_index is not None else ()
    )
    validity_digest = hashlib.sha256()
    validity_checks = 0
    for draw in range(count):
        train_perm = id_hash_permutation(
            caches["train"]["oneword"]["ids"], dataset, "train", draw, seed,
            train_fixed,
        )
        dev_perm = id_hash_permutation(
            caches["dev_seen"]["oneword"]["ids"], dataset, "dev_seen", draw, seed
        )
        permutation_digest.update(train_perm.tobytes())
        permutation_digest.update(dev_perm.tobytes())
        train_null = shuffled_contrast_views(
            normed["train"], normed["train"], train_perm, config,
            "{}/null/{}/train".format(dataset, draw),
            zero_masks["train"],
        )
        dev_null = shuffled_contrast_views(
            normed["dev_seen"], normed["dev_seen"], dev_perm, config,
            "{}/null/{}/dev".format(dataset, draw),
            zero_masks["dev_seen"],
        )
        for target in targets:
            retrieval = weighted_signed_scores(
                train_null[target], train_labels, dev_null[target], config
            )
            full = metric_bundle(dev_labels, retrieval["scores"], cutoff)
            audit, reduced = retrieval_without_registered_null(
                train_null[target], train_labels, dev_null[target], retrieval,
                registered_null_index, config,
                "{}/shuffle/{}/{}".format(dataset, draw, target),
            )
            if reduced is not None:
                audit["evaluation_checks"] = evaluation_exact_equivalence(
                    full, reduced, dev_labels, cutoff,
                    "{}/shuffle/{}/{}".format(dataset, draw, target),
                )
            validity_digest.update(json_payload(audit))
            validity_checks += 1
            metrics = full["metrics"]
            for metric, value in metrics.items():
                if value is None:
                    die("permutation null produced undefined {}".format(metric))
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
        "registered_null_top20_occurrences": 0,
        "with_null_remove_null_exact_mismatches": 0,
        "audit_digest": validity_digest.hexdigest(),
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
                "observed_above_p95": bool(observed > np.quantile(array, 0.95)),
                "one_sided_raw_p": float((1 + np.sum(array >= observed)) / (len(array) + 1)),
            }
            public["targets"][target][metric] = summary
            if metric in config["statistics"]["holm_metrics"]:
                summaries_for_holm.append((
                    "{}/{}/{}".format(dataset, target, metric), summary
                ))
    return public, summaries_for_holm, validity


def select_strongest_ordinary_control(evaluations, controls):
    control_order = list(controls)
    if not control_order:
        die("strongest ordinary control selection requires a non-empty family")
    if len(control_order) != len(set(control_order)):
        die("strongest ordinary control family contains duplicates")
    if "common_displacement" in control_order:
        die("primary arm cannot enter strongest ordinary control selection")
    missing = [name for name in control_order if name not in evaluations]
    if missing:
        die("strongest ordinary control evaluations missing: {}".format(missing))
    for name in control_order:
        metrics = evaluations[name].get("metrics", {})
        if "accuracy" not in metrics or "macro_f1" not in metrics:
            die("strongest ordinary control metrics missing for {}".format(name))
    return max(
        control_order,
        key=lambda name: (
            evaluations[name]["metrics"]["accuracy"],
            evaluations[name]["metrics"]["macro_f1"],
            -control_order.index(name),
        ),
    )


def displacement_audit(dataset, d_norm, evaluations, labels, config, zero_masks):
    transforms = config["transforms"]
    tiny_epsilon = float(transforms["tiny_displacement_epsilon"])
    quantile = float(transforms["small_displacement_train_quantile"])
    primary = evaluations["common_displacement"]["predictions"]
    strongest_control_name = select_strongest_ordinary_control(
        evaluations, config["decision"]["gain_controls"]
    )
    for split in REQUIRED_SPLITS:
        mask = np.asarray(zero_masks[split], dtype=bool)
        if mask.shape != d_norm[split]["img"].shape:
            die("{}/{} displacement null-mask shape mismatch".format(dataset, split))
    registered_null_indices = np.flatnonzero(zero_masks["train"])
    expected_null_count = (
        1 if dataset == config["zero_contract_v2"]["authorized_dataset"] else 0
    )
    if len(registered_null_indices) != expected_null_count:
        die("{} displacement registered-null count mismatch".format(dataset))

    def concentration(reference_name, binding_role, threshold, small_mask):
        reference = evaluations[reference_name]["predictions"]
        full = fix_break(primary, reference, labels)
        small = fix_break(primary, reference, labels, small_mask)
        large = fix_break(primary, reference, labels, ~small_mask)
        fixed_fraction = (
            0.0 if full["fixed"] == 0 else small["fixed"] / full["fixed"]
        )
        dominated = bool(
            full["fixed"] > 0
            and fixed_fraction
            > float(transforms["max_small_displacement_fix_fraction"])
        )
        return {
            "reference": reference_name,
            "binding_role": binding_role,
            "train_derived_quantile": quantile,
            "train_derived_threshold": threshold,
            "small_dev_rows": int(np.sum(small_mask)),
            "small_dev_fraction": float(np.mean(small_mask)),
            "all": full,
            "small": small,
            "non_small": large,
            "fraction_of_primary_fixes_in_small_rows": float(fixed_fraction),
            "small_rows_dominate_fixes": dominated,
        }

    def route(mode):
        if mode not in ("with_null_masked", "physically_remove_null"):
            die("unknown displacement null-exclusion route")

        def exclude(values, split):
            mask = zero_masks[split]
            if mode == "with_null_masked":
                return np.ma.array(values, mask=mask).compressed()
            return np.delete(values, np.flatnonzero(mask), axis=0)

        route_output = {"train": {}, "dev_seen": {}}
        train_min_full = np.minimum(
            d_norm["train"]["img"], d_norm["train"]["text"]
        )
        train_min = exclude(train_min_full, "train")
        threshold = float(np.quantile(train_min, quantile))
        if not math.isfinite(threshold):
            die("{} {} produced non-finite displacement threshold".format(
                dataset, mode
            ))
        dev_min = np.minimum(
            d_norm["dev_seen"]["img"], d_norm["dev_seen"]["text"]
        )
        if np.any(zero_masks["dev_seen"]):
            die("{} dev registered-null mask must be empty".format(dataset))
        small_mask = dev_min <= threshold
        max_tiny_fraction = 0.0
        for split in REQUIRED_SPLITS:
            for modality in ("img", "text"):
                values = exclude(d_norm[split][modality], split)
                if not len(values):
                    die("{} {} {}/{} displacement family is empty".format(
                        dataset, mode, split, modality
                    ))
                fraction = float(np.mean(values <= tiny_epsilon))
                max_tiny_fraction = max(max_tiny_fraction, fraction)
                route_output[split][modality] = {
                    "distribution": summarize_distribution(values),
                    "source_rows": int(len(d_norm[split][modality])),
                    "registered_null_rows_excluded": int(np.sum(zero_masks[split])),
                    "scientific_rows": int(len(values)),
                    "tiny_count": int(np.sum(values <= tiny_epsilon)),
                    "tiny_fraction": fraction,
                }
        tiny_ok = bool(
            max_tiny_fraction
            <= float(transforms["max_tiny_displacement_fraction"])
        )
        binding = concentration(
            strongest_control_name, "binding_gate", threshold, small_mask
        )
        diagnostic = concentration(
            "endpoint_concat",
            transforms["small_displacement_endpoint_concat_role"],
            threshold,
            small_mask,
        )
        final_bool = bool(
            tiny_ok
            and (
                not config["decision"][
                    "require_no_small_displacement_dominance"
                ]
                or not binding["small_rows_dominate_fixes"]
            )
        )
        route_output.update({
            "train_joint_scientific_rows": int(len(train_min)),
            "train_derived_quantile": quantile,
            "train_derived_threshold": threshold,
            "small_dev_mask": small_mask,
            "small_gain_gate_vs_strongest_ordinary_control": binding,
            "small_gain_diagnostic_vs_endpoint_concat": diagnostic,
            "maximum_tiny_fraction": max_tiny_fraction,
            "tiny_fraction_within_limit": tiny_ok,
            "scientific_gate_final_bool": final_bool,
        })
        return route_output

    masked = route("with_null_masked")
    removed = route("physically_remove_null")
    tiny_fraction_comparisons = {}
    tiny_count_exact = True
    for split in REQUIRED_SPLITS:
        tiny_fraction_comparisons[split] = {}
        for modality in ("img", "text"):
            comparison = float64_bytewise_audit(
                masked[split][modality]["tiny_fraction"],
                removed[split][modality]["tiny_fraction"],
                "{}/{}/{}/tiny_fraction".format(dataset, split, modality),
            )
            tiny_fraction_comparisons[split][modality] = comparison
            tiny_count_exact = bool(
                tiny_count_exact
                and masked[split][modality]["tiny_count"]
                == removed[split][modality]["tiny_count"]
                and masked[split][modality]["scientific_rows"]
                == removed[split][modality]["scientific_rows"]
            )
    comparisons = {
        "threshold": float64_bytewise_audit(
            masked["train_derived_threshold"],
            removed["train_derived_threshold"],
            dataset + "/displacement_threshold",
        ),
        "small_dev_mask": array_bytewise_audit(
            masked["small_dev_mask"],
            removed["small_dev_mask"],
            dataset + "/small_dev_mask",
        ),
        "tiny_fractions": tiny_fraction_comparisons,
        "maximum_tiny_fraction": float64_bytewise_audit(
            masked["maximum_tiny_fraction"],
            removed["maximum_tiny_fraction"],
            dataset + "/maximum_tiny_fraction",
        ),
        "tiny_counts_and_denominators_exact": tiny_count_exact,
        "small_rows_dominate_fixes_exact": (
            masked["small_gain_gate_vs_strongest_ordinary_control"][
                "small_rows_dominate_fixes"
            ]
            is removed["small_gain_gate_vs_strongest_ordinary_control"][
                "small_rows_dominate_fixes"
            ]
        ),
        "scientific_gate_final_bool_exact": (
            masked["scientific_gate_final_bool"]
            is removed["scientific_gate_final_bool"]
        ),
    }
    tiny_fraction_exact = strict_all(
        (
            comparison["pass"]
            for split in REQUIRED_SPLITS
            for comparison in comparisons["tiny_fractions"][split].values()
        ),
        dataset + "/displacement_tiny_fraction_dual_path",
    )
    dual_path_pass = bool(
        comparisons["threshold"]["pass"]
        and comparisons["small_dev_mask"]["pass"]
        and tiny_fraction_exact
        and comparisons["maximum_tiny_fraction"]["pass"]
        and comparisons["tiny_counts_and_denominators_exact"]
        and comparisons["small_rows_dominate_fixes_exact"]
        and comparisons["scientific_gate_final_bool_exact"]
    )
    if not dual_path_pass:
        die("{} displacement null-exclusion dual-path mismatch".format(dataset))

    output = {
        "definition": "raw ||L2(oneword)-L2(standard)||; joint row statistic=min(image,text)",
        "normalization_epsilon_fail_closed": float(transforms["normalization_epsilon"]),
        "tiny_epsilon": tiny_epsilon,
        "registered_null_exclusion": {
            "policy": "exclude from every displacement scientific reduction",
            "registered_train_indices": registered_null_indices.tolist(),
            "with_null_route": "masked before every quantile/distribution/tiny reduction",
            "remove_null_route": "physical row deletion before every reduction",
            "dual_path_required_exact": True,
        },
        "train": masked["train"],
        "dev_seen": masked["dev_seen"],
        "small_gain_gate_vs_strongest_ordinary_control": masked[
            "small_gain_gate_vs_strongest_ordinary_control"
        ],
        "small_gain_diagnostic_vs_endpoint_concat": masked[
            "small_gain_diagnostic_vs_endpoint_concat"
        ],
        "maximum_tiny_fraction": masked["maximum_tiny_fraction"],
        "tiny_fraction_within_limit": masked["tiny_fraction_within_limit"],
        "scientific_gate_final_bool": masked["scientific_gate_final_bool"],
        "dual_path_null_exclusion_audit": {
            "pass": dual_path_pass,
            "comparisons": comparisons,
            "masked_route_summary": {
                "train_joint_scientific_rows": masked[
                    "train_joint_scientific_rows"
                ],
                "train_derived_threshold": masked["train_derived_threshold"],
                "maximum_tiny_fraction": masked["maximum_tiny_fraction"],
                "small_rows_dominate_fixes": masked[
                    "small_gain_gate_vs_strongest_ordinary_control"
                ]["small_rows_dominate_fixes"],
                "scientific_gate_final_bool": masked[
                    "scientific_gate_final_bool"
                ],
            },
            "physically_removed_route_summary": {
                "train_joint_scientific_rows": removed[
                    "train_joint_scientific_rows"
                ],
                "train_derived_threshold": removed["train_derived_threshold"],
                "maximum_tiny_fraction": removed["maximum_tiny_fraction"],
                "small_rows_dominate_fixes": removed[
                    "small_gain_gate_vs_strongest_ordinary_control"
                ]["small_rows_dominate_fixes"],
                "scientific_gate_final_bool": removed[
                    "scientific_gate_final_bool"
                ],
            }
        },
    }
    return output


def validate_raw_zero_contract(dataset, caches, config):
    contract = config["zero_contract_v2"]
    masks = {}
    audit = {}
    epsilon = float(config["transforms"]["normalization_epsilon"])
    for split in REQUIRED_SPLITS:
        n = len(caches[split]["standard"]["ids"])
        expected_mask = np.zeros(n, dtype=bool)
        if (
            dataset == contract["authorized_dataset"]
            and split == contract["authorized_split"]
        ):
            index = int(contract["authorized_row_index"])
            if index >= n:
                die("authorized zero-contract row index is out of bounds")
            if caches[split]["standard"]["ids"][index] != contract["authorized_id"]:
                die("authorized zero-contract raw ID mismatch")
            if (
                int(caches[split]["standard"]["labels"][index])
                != int(contract["expected_label_integrity_only"])
            ):
                die("authorized zero-contract label integrity mismatch")
            expected_mask[index] = True
        masks[split] = expected_mask
        audit[split] = {}
        for policy in ("standard", "oneword"):
            audit[split][policy] = {}
            for modality in ("img", "text"):
                array = caches[split][policy][modality]
                if not np.isfinite(array).all():
                    die("{}/{}/{}/{} contains non-finite rows".format(
                        dataset, split, policy, modality
                    ))
                exact_zero = np.all(array == 0, axis=1)
                norms = np.linalg.norm(array, axis=1)
                tiny_nonzero = (~exact_zero) & (norms <= epsilon)
                if (
                    not np.array_equal(exact_zero, expected_mask)
                    or np.any(tiny_nonzero)
                ):
                    die("{}/{}/{}/{} violates the exact zero/tiny allowlist".format(
                        dataset, split, policy, modality
                    ))
                audit[split][policy][modality] = {
                    "exact_zero_indices": np.flatnonzero(exact_zero).tolist(),
                    "tiny_nonzero_count": int(np.sum(tiny_nonzero)),
                    "nonfinite_count": 0,
                }
    return masks, audit


def analyse_dataset(dataset, config, manifest_files):
    caches = {}
    access = []
    for split in REQUIRED_SPLITS:
        caches[split] = {}
        for policy in ("standard", "oneword"):
            loaded = load_cache(config, dataset, split, policy, manifest_files)
            caches[split][policy] = loaded
            access.append({
                "dataset": dataset,
                "split": split,
                "policy": policy,
                "path": loaded["path"],
                "sha256": loaded["sha256"],
                "bytes": loaded["bytes"],
            })
        standard = caches[split]["standard"]
        oneword = caches[split]["oneword"]
        if standard["ids"] != oneword["ids"]:
            die("{}/{} standard/oneword ID order mismatch".format(dataset, split))
        if not np.array_equal(standard["labels"], oneword["labels"]):
            die("{}/{} standard/oneword label mismatch".format(dataset, split))
    if set(caches["train"]["standard"]["ids"]) & set(caches["dev_seen"]["standard"]["ids"]):
        die("{} train/dev IDs overlap".format(dataset))
    zero_masks, raw_zero_audit = validate_raw_zero_contract(dataset, caches, config)

    views, normed, d_norm, rotation_names, algebra = {}, {}, {}, None, {}
    derived_zero_audit = {}
    for split in REQUIRED_SPLITS:
        (
            split_views,
            split_normed,
            split_d_norm,
            names,
            split_algebra,
            split_derived_masks,
        ) = prepare_views(
            caches[split]["standard"], caches[split]["oneword"], config,
            "{}/{}".format(dataset, split), zero_masks[split],
        )
        views[split] = split_views
        normed[split] = split_normed
        d_norm[split] = split_d_norm
        algebra[split] = split_algebra
        derived_zero_audit[split] = split_derived_masks
        if rotation_names is None:
            rotation_names = names
        elif rotation_names != names:
            die("rotation arm mismatch across splits")

    train_labels = caches["train"]["standard"]["labels"]
    dev_labels = caches["dev_seen"]["standard"]["labels"]
    registered_null_index = (
        int(config["zero_contract_v2"]["authorized_row_index"])
        if dataset == config["zero_contract_v2"]["authorized_dataset"]
        else None
    )
    evaluations, public_arms, retrieval_audit = evaluate_real_arms(
        views["train"], views["dev_seen"], train_labels, dev_labels, config,
        registered_null_index,
    )
    history = config["inputs"]["datasets"][dataset]["historical_strict_devtrain"]
    tolerance = float(config["history_parity"]["absolute_tolerance"])
    history_checks = {
        "endpoint_std_accuracy": {
            "expected": history["endpoint_std_accuracy"],
            "observed": evaluations["endpoint_std"]["metrics"]["accuracy"],
        },
        "endpoint_ow_accuracy": {
            "expected": history["endpoint_ow_accuracy"],
            "observed": evaluations["endpoint_ow"]["metrics"]["accuracy"],
        },
    }
    for check in history_checks.values():
        check["absolute_error"] = abs(check["observed"] - check["expected"])
        check["pass"] = bool(check["absolute_error"] <= tolerance)
    if (
        config["history_parity"]["require_endpoint_accuracy_match"]
        and not strict_all(
            (x["pass"] for x in history_checks.values()),
            dataset + "/historical_parity",
        )
    ):
        die("{} historical endpoint parity failed".format(dataset))

    null_public, null_holm, null_validity = permutation_null(
        dataset, caches, normed, evaluations, config, zero_masks
    )
    d_audit = displacement_audit(
        dataset, d_norm, evaluations, dev_labels, config, zero_masks
    )
    feature_dims = {arm: int(value.shape[1]) for arm, value in views["train"].items()}
    dataset_access = [
        entry for entry in INPUT_ACCESS_LEDGER if entry["dataset"] == dataset
    ]
    if len(dataset_access) != 4:
        die("{} access ledger must contain exactly four cache reads".format(dataset))
    test_open_count = sum(
        1
        for entry in dataset_access
        if entry["test_like"] and (entry["hash_opened"] or entry["torch_loaded"])
    )
    exact_hash_before_load = strict_all(
        (
            entry["manifest_sha256_matched_before_torch_load"]
            and entry["torch_loaded"]
            for entry in dataset_access
        ),
        dataset + "/cache_access_ledger",
    )
    public = {
        "n_train": int(len(train_labels)),
        "n_dev": int(len(dev_labels)),
        "access_ledger": access,
        "contract_guards": {
            "exact_keys": True,
            "shape_finite_binary": True,
            "standard_oneword_id_label_alignment": True,
            "unique_ids": True,
            "train_dev_disjoint": True,
            "raw_zero_allowlist_exact": True,
            "derived_zero_masks_exact": True,
            "displacement_null_exclusion_dual_path_exact": d_audit[
                "dual_path_null_exclusion_audit"
            ]["pass"],
            "registered_null_absent_from_all_top20": True,
            "with_null_remove_null_dtype_shape_bytes_equivalence": True,
            "shuffle_fixed_point_bijection": True,
            "cache_access_count": len(dataset_access),
            "test_paths_opened": test_open_count,
            "full_sha256_matched_before_every_torch_load": exact_hash_before_load,
            "feature_dimensions": feature_dims,
            "raw_zero_contract": raw_zero_audit,
            "derived_zero_mask_preservation": derived_zero_audit,
            "retrieval_null_influence": retrieval_audit,
            "shuffle_null_validity": null_validity,
        },
        "runtime_access_ledger": dataset_access,
        "historical_parity": history_checks,
        "historical_deployed_r0_accuracy_context_only": history[
            "deployed_r0_accuracy_context_only"
        ],
        "algebra_guards": algebra,
        "arms": public_arms,
        "rotation_arm_names": rotation_names,
        "shuffled_pair_null": null_public,
        "displacement_norm_audit": d_audit,
        "paired_bootstrap": {},
    }
    runtime = {
        "caches": caches,
        "evaluations": evaluations,
        "labels": dev_labels,
        "rotation_names": rotation_names,
        "null_holm": null_holm,
    }
    return public, runtime


def build_halt_only_validity_guards(results, evidence, config):
    required = config["output"]["decision_schema"][
        "required_halt_only_validity_guards"
    ]
    datasets = {}
    for dataset in REQUIRED_DATASETS:
        contract = results["datasets"][dataset]["contract_guards"]
        derived = contract["derived_zero_mask_preservation"]
        retrieval = contract["retrieval_null_influence"]
        shuffle = contract["shuffle_null_validity"]
        checks = {
            "raw_zero_allowlist_exact": (
                contract["raw_zero_allowlist_exact"] is True
            ),
            "derived_zero_masks_exact": bool(strict_all(
                (
                    flag
                    for split in REQUIRED_SPLITS
                    for flag in derived[split].values()
                ),
                dataset + "/aggregate_derived_zero_masks",
            )),
            "displacement_null_exclusion_dual_path_exact": (
                contract["displacement_null_exclusion_dual_path_exact"] is True
            ),
            "shuffle_fixed_point_bijection": bool(
                shuffle["pass"] is True
                and shuffle["train_bijection_draws_checked"]
                == config["statistics"]["n_id_hash_permutations"]
                and shuffle["dev_bijection_draws_checked"]
                == config["statistics"]["n_id_hash_permutations"]
            ),
            "registered_null_absent_from_all_top20": bool(
                strict_all(
                    (audit["pass"] is True for audit in retrieval.values()),
                    dataset + "/aggregate_real_retrieval",
                )
                and shuffle["registered_null_top20_occurrences"] == 0
            ),
            "with_null_remove_null_dtype_shape_bytes_equivalence": bool(
                strict_all(
                    (audit["pass"] is True for audit in retrieval.values()),
                    dataset + "/aggregate_real_remove_null",
                )
                and shuffle["with_null_remove_null_exact_mismatches"] == 0
            ),
        }
        if not strict_all(checks.values(), dataset + "/halt_only_guards"):
            die("{} halt-only validity guard aggregate failed".format(dataset))
        datasets[dataset] = checks
    global_checks = {
        "probe_evidence_exact": bool(
            evidence["sha256"] == config["zero_contract_v2"]["probe_sha256"]
        ),
        "raw_zero_allowlist_exact": bool(all(
            values["raw_zero_allowlist_exact"] for values in datasets.values()
        )),
        "derived_zero_masks_exact": bool(all(
            values["derived_zero_masks_exact"] for values in datasets.values()
        )),
        "displacement_null_exclusion_dual_path_exact": bool(all(
            values["displacement_null_exclusion_dual_path_exact"]
            for values in datasets.values()
        )),
        "shuffle_fixed_point_bijection": bool(all(
            values["shuffle_fixed_point_bijection"] for values in datasets.values()
        )),
        "registered_null_absent_from_all_top20": bool(all(
            values["registered_null_absent_from_all_top20"]
            for values in datasets.values()
        )),
        "with_null_remove_null_dtype_shape_bytes_equivalence": bool(all(
            values["with_null_remove_null_dtype_shape_bytes_equivalence"]
            for values in datasets.values()
        )),
    }
    if list(global_checks) != required:
        die("halt-only validity guard order/binding changed")
    if not strict_all(global_checks.values(), "global_halt_only_validity_guards"):
        die("global halt-only validity guard failed")
    return {
        "halt_only": True,
        "passed": True,
        "required": required,
        "checks": global_checks,
        "datasets": datasets,
    }


def attach_bootstrap_and_holm(results, runtimes, config):
    specs = {
        "primary_vs_controls": (
            config["retrieval"]["primary_arm"],
            config["statistics"]["bootstrap_comparisons"]["primary_vs_controls"],
        ),
        "secondary_vs_controls": (
            config["retrieval"]["secondary_arm"],
            config["statistics"]["bootstrap_comparisons"]["secondary_vs_controls"],
        ),
        "primary_vs_orthogonal_rotations": (
            config["retrieval"]["primary_arm"],
            None,
        ),
    }
    holm_entries = {family: [] for family in specs}
    for dataset in REQUIRED_DATASETS:
        runtime = runtimes[dataset]
        evaluations = runtime["evaluations"]
        labels = runtime["labels"]
        for family, (candidate_name, controls) in specs.items():
            if controls is None:
                controls = runtime["rotation_names"]
            results["datasets"][dataset]["paired_bootstrap"][family] = {}
            for control_name in controls:
                summary = paired_bootstrap(
                    evaluations[candidate_name],
                    evaluations[control_name],
                    labels,
                    config,
                    stable_seed(
                        config["statistics"]["seed"], dataset, family,
                        candidate_name, control_name,
                    ),
                )
                results["datasets"][dataset]["paired_bootstrap"][family][control_name] = summary
                for metric in config["statistics"]["holm_metrics"]:
                    holm_entries[family].append((
                        "{}/{}/{}/{}".format(dataset, candidate_name, control_name, metric),
                        summary[metric],
                    ))
    alpha = float(config["statistics"]["holm_alpha"])
    for family, entries in holm_entries.items():
        holm_adjust(entries, alpha)
    shuffle_entries = []
    for dataset in REQUIRED_DATASETS:
        shuffle_entries.extend(runtimes[dataset]["null_holm"])
    holm_adjust(shuffle_entries, alpha)
    results["holm_families"] = {
        family: {
            "alpha": alpha,
            "n_hypotheses": len(entries),
            "all_rejected": bool(strict_all(
                (x[1]["holm_reject"] for x in entries),
                family + "/holm_entries",
            )),
        }
        for family, entries in holm_entries.items()
    }
    results["holm_families"]["real_vs_shuffled_pair"] = {
        "alpha": alpha,
        "n_hypotheses": len(shuffle_entries),
        "all_rejected": bool(strict_all(
            (x[1]["holm_reject"] for x in shuffle_entries),
            "real_vs_shuffled_pair/holm_entries",
        )),
    }


def make_decision(results, runtimes, config):
    decision_cfg = config["decision"]
    primary_name = config["retrieval"]["primary_arm"]
    required_metrics = decision_cfg["required_metrics"]
    dataset_decisions = {}
    for dataset in decision_cfg["required_datasets"]:
        public = results["datasets"][dataset]
        runtime = runtimes[dataset]
        evaluations = runtime["evaluations"]
        primary = evaluations[primary_name]
        checks = {}

        controls = decision_cfg["gain_controls"]
        strongest = {}
        for metric in required_metrics:
            control_value = max(evaluations[x]["metrics"][metric] for x in controls)
            reference_value = control_value
            if metric == "accuracy" and decision_cfg[
                "require_accuracy_gain_over_deployed_r0_context"
            ]:
                reference_value = max(
                    reference_value,
                    public["historical_deployed_r0_accuracy_context_only"],
                )
            gain = primary["metrics"][metric] - reference_value
            strongest[metric] = {
                "primary": primary["metrics"][metric],
                "reference": reference_value,
                "gain": gain,
                "pass": bool(
                    gain >= float(decision_cfg["minimum_gain_over_strongest_control"])
                ),
            }
        checks["gain_over_strongest_control"] = {
            "metrics": strongest,
            "pass": bool(strict_all(
                (x["pass"] for x in strongest.values()),
                dataset + "/strongest_control_metrics",
            )),
        }

        primary_bootstrap = public["paired_bootstrap"]["primary_vs_controls"]
        required_bootstrap = []
        for control_name, summaries in primary_bootstrap.items():
            for metric in required_metrics:
                summary = summaries[metric]
                required_bootstrap.append(
                    summary["lower"] > float(decision_cfg["minimum_bootstrap_lower_bound"])
                    and (
                        not decision_cfg["require_primary_bootstrap_holm_reject"]
                        or summary["holm_reject"]
                    )
                )
        checks["primary_paired_bootstrap"] = {
            "pass": bool(strict_all(
                required_bootstrap, dataset + "/primary_bootstrap_requirements"
            )),
            "n_required": len(required_bootstrap),
        }

        rotation_bootstrap = public["paired_bootstrap"][
            "primary_vs_orthogonal_rotations"
        ]
        rotation_checks = []
        rotation_bounds = {}
        for metric in required_metrics:
            upper = max(
                evaluations[name]["metrics"][metric] for name in runtime["rotation_names"]
            )
            rotation_bounds[metric] = {
                "primary": primary["metrics"][metric],
                "frozen_rotation_upper_bound": upper,
                "pass": bool(primary["metrics"][metric] > upper),
            }
        for summaries in rotation_bootstrap.values():
            for metric in required_metrics:
                rotation_checks.append(
                    summaries[metric]["lower"] > 0.0
                    and (
                        not decision_cfg["require_rotation_bootstrap_holm_reject"]
                        or summaries[metric]["holm_reject"]
                    )
                )
        checks["orthogonal_rotation_controls"] = {
            "distribution_upper_bounds": rotation_bounds,
            "all_paired_corrected": bool(strict_all(
                rotation_checks, dataset + "/rotation_corrected_checks"
            )),
            "pass": bool(
                strict_all(
                    (x["pass"] for x in rotation_bounds.values()),
                    dataset + "/rotation_bounds",
                )
                and strict_all(
                    rotation_checks, dataset + "/rotation_corrected_checks"
                )
            ),
        }

        shuffle_checks = []
        for target in (primary_name, "displacement"):
            for metric in required_metrics:
                summary = public["shuffled_pair_null"]["targets"][target][metric]
                shuffle_checks.append(
                    summary["observed_above_p95"]
                    and (
                        not decision_cfg["require_shuffle_holm_reject"]
                        or summary["holm_reject"]
                    )
                )
        checks["real_vs_shuffled_pair"] = {
            "pass": bool(strict_all(
                shuffle_checks, dataset + "/shuffle_requirements"
            )),
            "n_required": len(shuffle_checks),
        }

        strongest_control_name = select_strongest_ordinary_control(
            evaluations, controls
        )
        net = fix_break(
            primary["predictions"],
            evaluations[strongest_control_name]["predictions"],
            runtime["labels"],
        )
        checks["net_fixes"] = {
            "reference": strongest_control_name,
            "counts": net,
            "minimum": int(decision_cfg["minimum_net_fixes"][dataset]),
            "pass": bool(net["net"] >= int(decision_cfg["minimum_net_fixes"][dataset])),
        }

        d_audit = public["displacement_norm_audit"]
        small_gain_gate = d_audit[
            "small_gain_gate_vs_strongest_ordinary_control"
        ]
        endpoint_concat_diagnostic = d_audit[
            "small_gain_diagnostic_vs_endpoint_concat"
        ]
        if small_gain_gate["reference"] != strongest_control_name:
            die(
                "small-displacement gate reference diverged from strongest "
                "ordinary control"
            )
        if (
            endpoint_concat_diagnostic["reference"] != "endpoint_concat"
            or endpoint_concat_diagnostic["binding_role"] != "diagnostic_only"
        ):
            die("endpoint_concat small-displacement audit must remain diagnostic-only")
        checks["displacement_stability"] = {
            "tiny_fraction_within_limit": d_audit["tiny_fraction_within_limit"],
            "registered_null_excluded_from_scientific_reductions": True,
            "masked_vs_physically_removed_exact": d_audit[
                "dual_path_null_exclusion_audit"
            ]["pass"],
            "reference": small_gain_gate["reference"],
            "selection_rule": config["transforms"][
                "small_displacement_gate_reference"
            ],
            "small_rows_dominate_fixes": small_gain_gate[
                "small_rows_dominate_fixes"
            ],
            "endpoint_concat_diagnostic_role": endpoint_concat_diagnostic[
                "binding_role"
            ],
            "pass": bool(
                d_audit["dual_path_null_exclusion_audit"]["pass"]
                and d_audit["scientific_gate_final_bool"]
            ),
        }
        checks["history_and_contract"] = {
            "pass": bool(
                strict_all(
                    (x["pass"] for x in public["historical_parity"].values()),
                    dataset + "/history_contract_checks",
                )
                and public["contract_guards"]["test_paths_opened"] == 0
                and public["contract_guards"][
                    "full_sha256_matched_before_every_torch_load"
                ]
            )
        }
        passed = bool(strict_all(
            (check["pass"] for check in checks.values()),
            dataset + "/decision_checks",
        ))
        dataset_decisions[dataset] = {
            "pass": passed,
            "checks": checks,
        }

    overall = bool(strict_all(
        (x["pass"] for x in dataset_decisions.values()),
        "overall_dataset_decisions",
    ))
    label = (
        decision_cfg["continue_label"] if overall else decision_cfg["kill_label"]
    )
    return {
        "decision": label,
        "continue": overall,
        "datasets": dataset_decisions,
        "interpretation": (
            config["positive_scope"] if overall else config["negative_scope"]
        ),
        "forbidden_interpretation": (
            "This A0 cannot establish prompt-only causality, safety/stance disentanglement, "
            "or an end-to-end performance gain because standard and one-word L24 caches "
            "also differ in pooling/readout policy."
        ),
    }


def ensure_json_finite(value, path="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            ensure_json_finite(child, path + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            ensure_json_finite(child, path + "[{}]".format(index))
    elif isinstance(value, float) and not math.isfinite(value):
        die("non-finite JSON value at {}".format(path))


def json_payload(value):
    ensure_json_finite(value)
    return (json.dumps(
        value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False
    ) + "\n").encode("utf-8")


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_publish(path, payload):
    if not path.parent.is_dir() or path.exists():
        die("exclusive publication target is not fresh: {}".format(path))
    descriptor, temporary = tempfile.mkstemp(
        prefix="." + path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(str(temporary_path), str(path))
        temporary_path.unlink()
        fsync_directory(path.parent)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def validate_decision_artifact(decision, config):
    expected_keys = {
        "schema_version",
        "experiment_id",
        "run_id",
        "config_sha256",
        "full_sha256_manifest",
        "zero_contract_v2_evidence",
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
    if set(decision) != expected_keys:
        die("decision artifact schema keys changed")
    schema = config["output"]["decision_schema"]
    if decision["schema_version"] != schema["schema_version"]:
        die("decision schema_version mismatch")
    evidence = decision["zero_contract_v2_evidence"]
    if (
        evidence["path"] != config["zero_contract_v2"]["probe_path"]
        or evidence["sha256"] != config["zero_contract_v2"]["probe_sha256"]
        or evidence["run_id"] != config["zero_contract_v2"]["probe_run_id"]
        or evidence["schema_version"]
        != config["zero_contract_v2"]["probe_schema_version"]
    ):
        die("decision zero-contract evidence binding changed")
    validity = decision["halt_only_validity_guards"]
    if (
        validity.get("halt_only") is not True
        or validity.get("passed") is not True
        or validity.get("required")
        != schema["required_halt_only_validity_guards"]
        or list(validity.get("checks", {}))
        != schema["required_halt_only_validity_guards"]
        or not strict_all(
            validity["checks"].values(), "decision_halt_only_validity_guards"
        )
    ):
        die("decision halt-only validity guard binding failed")
    if decision["decision_label"] not in schema["allowed_decisions"]:
        die("decision label outside frozen enum")
    if list(decision["dataset_pass"]) != schema["required_dataset_keys"]:
        die("decision dataset key/order binding changed")
    if (
        list(decision["small_displacement_gate_reference"])
        != schema["required_dataset_keys"]
    ):
        die("small-displacement gate reference dataset key/order changed")
    if not strict_all(
        (
            reference in config["decision"]["gain_controls"]
            for reference in decision["small_displacement_gate_reference"].values()
        ),
        "decision_small_displacement_gate_references",
    ):
        die("small-displacement gate reference is not an ordinary control")
    if (
        decision["small_displacement_gate_selection_rule"]
        != schema["small_displacement_gate_selection_rule"]
    ):
        die("small-displacement gate selection rule changed")
    if (
        decision["endpoint_concat_small_displacement_role"]
        != schema["endpoint_concat_small_displacement_role"]
        or decision["endpoint_concat_small_displacement_role"] != "diagnostic_only"
    ):
        die("endpoint_concat small-displacement role changed")
    if not strict_all(
        (isinstance(value, bool) for value in decision["dataset_pass"].values()),
        "decision_dataset_pass_types",
    ):
        die("decision dataset pass values must be booleans")
    if not isinstance(decision["continue"], bool):
        die("decision continue must be boolean")
    if decision["exclusive_create"] is not True:
        die("decision exclusive-create binding weakened")
    expected_label = (
        config["decision"]["continue_label"]
        if decision["continue"]
        else config["decision"]["kill_label"]
    )
    if decision["decision_label"] != expected_label:
        die("decision boolean/label mismatch")


def main():
    args = parse_args()
    config_path = repo_path(args.config, "config")
    config = load_config(config_path)
    validate_config(config)
    enforce_runtime(config)
    full_sha_manifest = load_full_sha256_manifest(config)
    zero_contract_evidence = load_zero_contract_evidence(config)
    import_compute_modules(config)

    output_cfg = config["output"]
    namespace = repo_path(output_cfg["namespace"], "output.namespace")
    result_path = namespace / output_cfg["result_file"]
    decision_path = namespace / output_cfg["decision_file"]
    if namespace.exists():
        if result_path.exists() and not decision_path.exists():
            die(
                "partial result without decision detected; fail-closed and choose "
                "a new run_id/namespace"
            )
        die("run namespace already exists; no-clobber requires a new run_id/namespace")
    namespace.parent.mkdir(parents=True, exist_ok=True)
    namespace.mkdir()

    config_digest = sha256_file(config_path)
    results = {
        "schema_version": output_cfg["result_schema_version"],
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "claim_scope": config["claim_scope"],
        "config_path": str(config_path.relative_to(REPO)),
        "config_sha256": config_digest,
        "full_sha256_preflight": full_sha_manifest,
        "zero_contract_v2_evidence": zero_contract_evidence,
        "runtime_guards": {
            "slurm_job_id": os.environ["SLURM_JOB_ID"],
            "slurm_cpus_per_task": int(os.environ["SLURM_CPUS_PER_TASK"]),
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "thread_environment": config["execution"]["required_environment"],
            "cpu_only": True,
        },
        "method_boundary": {
            "paired_endpoints": "standard-L24 prefix/response mean versus one-word-L24 last-token",
            "confounded_axes": ["prompt", "pooling/token readout"],
            "algebra": (
                "Without per-block L2, endpoint_concat and common_displacement are "
                "orthogonal reparameterizations. Any difference here is attributable "
                "only to the frozen block normalization/reweighting."
            ),
            "positive_scope": config["positive_scope"],
            "negative_scope": config["negative_scope"],
        },
        "datasets": {},
    }
    runtimes = {}
    for dataset in REQUIRED_DATASETS:
        public, runtime = analyse_dataset(
            dataset, config, full_sha_manifest["files"]
        )
        results["datasets"][dataset] = public
        runtimes[dataset] = runtime
    if len(INPUT_ACCESS_LEDGER) != CANONICAL_BINDING["manifest"]["expected_file_count"]:
        die("A0 runtime access ledger is incomplete")
    test_like_attempts = sum(
        1 for entry in INPUT_ACCESS_LEDGER if entry["test_like"]
    )
    test_like_opens = sum(
        1
        for entry in INPUT_ACCESS_LEDGER
        if entry["test_like"] and (entry["hash_opened"] or entry["torch_loaded"])
    )
    all_manifest_matched = strict_all(
        (
            entry["manifest_sha256_matched_before_torch_load"]
            and entry["torch_loaded"]
            for entry in INPUT_ACCESS_LEDGER
        ),
        "global_cache_access_ledger",
    )
    if test_like_attempts != 0 or test_like_opens != 0 or not all_manifest_matched:
        die("global runtime cache access guard failed")
    results["runtime_guards"]["cache_access"] = {
        "expected_count": CANONICAL_BINDING["manifest"]["expected_file_count"],
        "actual_count": len(INPUT_ACCESS_LEDGER),
        "test_like_attempt_count": test_like_attempts,
        "test_like_open_count": test_like_opens,
        "full_sha256_matched_before_every_torch_load": all_manifest_matched,
        "ledger": INPUT_ACCESS_LEDGER,
    }
    results["halt_only_validity_guards"] = build_halt_only_validity_guards(
        results, zero_contract_evidence, config
    )
    attach_bootstrap_and_holm(results, runtimes, config)
    results["decision"] = make_decision(results, runtimes, config)

    result_bytes = json_payload(results)
    if len(result_bytes) > int(output_cfg["maximum_result_bytes"]):
        die("result JSON exceeds configured maximum size")
    result_sha256 = hashlib.sha256(result_bytes).hexdigest()
    decision = {
        "schema_version": output_cfg["decision_schema"]["schema_version"],
        "experiment_id": config["experiment_id"],
        "run_id": config["run_id"],
        "config_sha256": config_digest,
        "full_sha256_manifest": {
            "path": full_sha_manifest["path"],
            "sha256": full_sha_manifest["sha256"],
            "run_id": config["full_sha256_preflight"]["run_id"],
        },
        "zero_contract_v2_evidence": zero_contract_evidence,
        "halt_only_validity_guards": results["halt_only_validity_guards"],
        "result_file": output_cfg["result_file"],
        "result_sha256": result_sha256,
        "decision_label": results["decision"]["decision"],
        "continue": results["decision"]["continue"],
        "dataset_pass": {
            dataset: results["decision"]["datasets"][dataset]["pass"]
            for dataset in REQUIRED_DATASETS
        },
        "small_displacement_gate_reference": {
            dataset: results["decision"]["datasets"][dataset]["checks"][
                "displacement_stability"
            ]["reference"]
            for dataset in REQUIRED_DATASETS
        },
        "small_displacement_gate_selection_rule": output_cfg["decision_schema"][
            "small_displacement_gate_selection_rule"
        ],
        "endpoint_concat_small_displacement_role": output_cfg["decision_schema"][
            "endpoint_concat_small_displacement_role"
        ],
        "interpretation_scope": results["decision"]["interpretation"],
        "exclusive_create": True,
    }
    validate_decision_artifact(decision, config)
    decision_bytes = json_payload(decision)

    atomic_publish(result_path, result_bytes)
    atomic_publish(decision_path, decision_bytes)
    print(json.dumps({
        "run_id": config["run_id"],
        "decision": results["decision"]["decision"],
        "result": str(result_path.relative_to(REPO)),
        "decision_file": str(decision_path.relative_to(REPO)),
        "result_sha256": result_sha256,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("C01_A0_FAIL_CLOSED: {}".format(exc), file=sys.stderr)
        raise
