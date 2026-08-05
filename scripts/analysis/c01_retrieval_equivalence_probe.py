#!/usr/bin/env python3
"""Read-only C01 diagnostic for HateMM endpoint_std with/remove-null retrieval."""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RUN_ID = "C01-RETRIEVAL-EQUIV-PROBE-v1"
SCHEMA_VERSION = "c01_retrieval_equivalence_probe_v1"
MANIFEST_PATH = (
    "artifacts/c01_policy_contrastive/v1/hash_preflight/"
    "C01-HASH-v1/full_sha256_manifest.json"
)
MANIFEST_SHA256 = (
    "083275d39a1026bde3b6583bd5608d41"
    "cec5b431da9ffda87ae8ab1046cf2305"
)
ZERO_PROBE_PATH = (
    "artifacts/c01_policy_contrastive/v2/zero_contract_probe/"
    "C01-ZERO-PROBE-v1/zero_contract_probe.json"
)
ZERO_PROBE_SHA256 = (
    "bee4964ce7e4ca81cfdb72c3859f7819"
    "6568badf982aef587bc14ee6dbe63526"
)
OUTPUT_DIR = (
    "artifacts/c01_policy_contrastive/v2/retrieval_equivalence_probe/"
    + RUN_ID
)
OUTPUT_FILE = "retrieval_equivalence_probe.json"
TRAIN_PATH = (
    "data/CLIP_Embedding/HateMM/"
    "train_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF-ro_L24.pt"
)
DEV_PATH = (
    "data/CLIP_Embedding/HateMM/"
    "dev_seen_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF-ro_L24.pt"
)
NULL_INDEX = 355
NULL_ID = "hate_video_95"
NULL_LABEL = 1
TOPK = 20
EPSILON = 1e-12
EXAMPLE_LIMIT = 12
DEFAULT_CONFIG = "configs/c01/c01_retrieval_equivalence_probe_v1.json"
CANONICAL_CONFIG = {
    "schema_version": 1,
    "run_id": RUN_ID,
    "evidence": {
        "manifest_path": MANIFEST_PATH,
        "manifest_sha256": MANIFEST_SHA256,
        "zero_probe_path": ZERO_PROBE_PATH,
        "zero_probe_sha256": ZERO_PROBE_SHA256,
    },
    "authorized_null": {
        "dataset": "HateMM",
        "split": "train",
        "id": NULL_ID,
        "row_index": NULL_INDEX,
        "expected_label_integrity_only": NULL_LABEL,
        "policy": "standard",
        "modalities": ["img", "text"],
    },
    "inputs": {
        "train_path": TRAIN_PATH,
        "dev_seen_path": DEV_PATH,
        "test_hardblock": True,
        "feature_vectors_serialized": False,
    },
    "retrieval": {
        "arm": "endpoint_std",
        "topk": TOPK,
        "similarity": "faiss_indexflatip_float32",
        "rank_weights": "descending_integer",
        "prediction_cutoff": 0.0,
        "raw_order": True,
        "stable_order": "negative_similarity_then_original_train_index",
        "a0_key_construction": (
            "raw_modality_L2_then_fuse_modalities_reL2_each_block_"
            "then_concat_then_final_L2"
        ),
    },
    "execution": {
        "required_cpus": 8,
        "required_memory": "32G",
        "required_environment": {
            "CUDA_VISIBLE_DEVICES": "",
            "OMP_NUM_THREADS": "8",
            "MKL_NUM_THREADS": "8",
            "OPENBLAS_NUM_THREADS": "8",
            "NUMEXPR_NUM_THREADS": "8",
        },
        "cpu_only": True,
        "slurm_only": True,
    },
    "output": {
        "namespace": OUTPUT_DIR,
        "file": OUTPUT_FILE,
        "maximum_bytes": 2_000_000,
        "exclusive_no_clobber": True,
    },
}


def die(message):
    raise RuntimeError(message)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    return parser.parse_args()


def has_test_token(value):
    return "test" in [
        token
        for token in re.split(r"[^a-z0-9]+", str(value).lower())
        if token
    ]


def repo_path(relative, field):
    if not isinstance(relative, str) or not relative:
        die("{} must be a non-empty relative path".format(field))
    path = (REPO / relative).resolve()
    try:
        path.relative_to(REPO)
    except ValueError:
        die("{} escapes repository".format(field))
    if has_test_token(path):
        die("test-like path forbidden in {}: {}".format(field, path))
    return path


def sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_config(relative):
    path = repo_path(relative, "config")
    with path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    if config != CANONICAL_CONFIG:
        die("probe config differs from frozen canonical binding")
    return config, path


def runtime_guard(config):
    execution = config["execution"]
    expected = execution["required_environment"]
    if not os.environ.get("SLURM_JOB_ID"):
        die("Slurm-only probe requires SLURM_JOB_ID")
    if os.environ.get("SLURM_CPUS_PER_TASK") != str(
        execution["required_cpus"]
    ):
        die("probe requires exactly eight Slurm CPUs")
    for key, value in expected.items():
        if os.environ.get(key) != value:
            die("{} mismatch".format(key))
    return expected


def load_evidence():
    manifest_path = repo_path(MANIFEST_PATH, "manifest")
    zero_path = repo_path(ZERO_PROBE_PATH, "zero probe")
    manifest_raw = manifest_path.read_bytes()
    zero_raw = zero_path.read_bytes()
    if sha256_bytes(manifest_raw) != MANIFEST_SHA256:
        die("manifest whole-file SHA256 mismatch")
    if sha256_bytes(zero_raw) != ZERO_PROBE_SHA256:
        die("zero-probe whole-file SHA256 mismatch")
    manifest = json.loads(manifest_raw.decode("utf-8"))
    zero = json.loads(zero_raw.decode("utf-8"))
    if (
        manifest.get("schema_version") != "c01_full_sha256_manifest_v1"
        or manifest.get("run_id") != "C01-HASH-v1"
        or manifest.get("complete") is not True
        or zero.get("schema_version") != "c01_zero_contract_probe_v1"
        or zero.get("run_id") != "C01-ZERO-PROBE-v1"
        or zero.get("status") != "DIAGNOSTIC_ONLY"
        or zero.get("hash_manifest", {}).get("sha256") != MANIFEST_SHA256
    ):
        die("evidence identity/completeness mismatch")
    train_record = zero["datasets"]["HateMM"]["splits"]["train"]
    dev_record = zero["datasets"]["HateMM"]["splits"]["dev_seen"]
    expected_zero = [{
        "id": NULL_ID,
        "label": NULL_LABEL,
        "other_modality_state": "exact_zero",
        "row_index": NULL_INDEX,
    }]
    for modality in ("img", "text"):
        if train_record["endpoint_comparison"][modality] != {
            "matched_zero_ids": [NULL_ID],
            "oneword_only_zero_ids": [],
            "standard_only_zero_ids": [],
            "zero_mask_exact_match": True,
        }:
            die("zero-probe endpoint mask mismatch")
        cell = train_record["policies"]["standard"]["modalities"][modality]
        if (
            cell["zero_rows"] != expected_zero
            or cell["tiny_nonzero_row_count"] != 0
            or cell["nonfinite_row_count"] != 0
        ):
            die("zero-probe registered tuple mismatch")
        dev_cell = dev_record["policies"]["standard"]["modalities"][modality]
        if (
            dev_cell["zero_row_count"] != 0
            or dev_cell["tiny_nonzero_row_count"] != 0
            or dev_cell["nonfinite_row_count"] != 0
        ):
            die("zero-probe dev anomaly mismatch")
    by_path = {entry["path"]: entry for entry in manifest["files"]}
    if TRAIN_PATH not in by_path or DEV_PATH not in by_path:
        die("required caches absent from approved manifest")
    return {
        "manifest": manifest,
        "manifest_by_path": by_path,
        "manifest_path": manifest_path,
        "zero_path": zero_path,
    }


def load_cache(relative, record, torch):
    path = repo_path(relative, "cache")
    if path.stat().st_size != int(record["bytes"]):
        die("{} byte-size mismatch".format(relative))
    digest = sha256_file(path)
    if digest != record["sha256"]:
        die("{} full SHA256 mismatch before torch.load".format(relative))
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != {
        "ids", "img_feats", "text_feats", "labels"
    }:
        die("{} exact cache schema mismatch".format(relative))
    raw_ids = payload["ids"]
    if (
        not isinstance(raw_ids, (list, tuple))
        or len(raw_ids) != 1
        or not isinstance(raw_ids[0], (list, tuple))
    ):
        die("{} ID container mismatch".format(relative))
    ids = list(raw_ids[0])
    if (
        any(type(item) is not str or not item for item in ids)
        or len(ids) != len(set(ids))
    ):
        die("{} raw IDs are invalid".format(relative))
    tensors = (payload["img_feats"], payload["text_feats"], payload["labels"])
    if not all(torch.is_tensor(value) for value in tensors):
        die("{} tensor contract mismatch".format(relative))
    img = payload["img_feats"].detach().cpu().float().numpy()
    text = payload["text_feats"].detach().cpu().float().numpy()
    labels = payload["labels"].detach().cpu().numpy()
    if (
        img.shape != (len(ids), 3584)
        or text.shape != (len(ids), 3584)
        or labels.shape != (len(ids),)
        or not all(math.isfinite(float(value)) for value in labels)
    ):
        die("{} shape/finite contract mismatch".format(relative))
    rounded = labels.round().astype("int64")
    if not (labels == rounded).all() or not set(rounded.tolist()) <= {0, 1}:
        die("{} labels are not binary integers".format(relative))
    return {
        "ids": ids,
        "img": img.astype("float32", copy=False),
        "text": text.astype("float32", copy=False),
        "labels": rounded,
        "path": relative,
        "sha256": digest,
        "bytes": path.stat().st_size,
    }


def normalize_rows(values, allowed_zero, np, context):
    if not np.isfinite(values).all():
        die("{} contains NaN/Inf".format(context))
    zero = np.all(values == 0, axis=1)
    if not np.array_equal(zero, allowed_zero):
        die("{} exact-zero mask mismatch".format(context))
    norms = np.linalg.norm(values, axis=1)
    if np.any((norms <= EPSILON) & ~allowed_zero):
        die("{} contains non-registered tiny/zero row".format(context))
    output = np.zeros_like(values, dtype="float32")
    keep = ~allowed_zero
    output[keep] = (values[keep] / norms[keep, None]).astype("float32")
    return output


def load_a0_module(np):
    path = repo_path(
        "scripts/analysis/c01_policy_contrast_a0.py", "A0 implementation"
    )
    spec = importlib.util.spec_from_file_location("c01_a0_probe_reference", path)
    if spec is None or spec.loader is None:
        die("cannot import A0 implementation for key parity")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.np = np
    return module, path


def local_endpoint_std(cache, allowed_zero, np, context):
    first_img = normalize_rows(
        cache["img"], allowed_zero, np, context + "/first_modality_l2/img"
    )
    first_text = normalize_rows(
        cache["text"], allowed_zero, np, context + "/first_modality_l2/text"
    )
    block_img = normalize_rows(
        first_img, allowed_zero, np, context + "/fuse_block_rel2/img"
    )
    block_text = normalize_rows(
        first_text, allowed_zero, np, context + "/fuse_block_rel2/text"
    )
    return normalize_rows(
        np.concatenate([block_img, block_text], axis=1),
        allowed_zero,
        np,
        context + "/concat_final_l2",
    )


def endpoint_std(cache, allowed_zero, np, a0, a0_sha256, context):
    first_img, _ = a0.l2_rows(
        cache["img"],
        EPSILON,
        context + "/a0_first_modality_l2/img",
        allowed_zero,
    )
    first_text, _ = a0.l2_rows(
        cache["text"],
        EPSILON,
        context + "/a0_first_modality_l2/text",
        allowed_zero,
    )
    imported_key = a0.fuse_modalities(
        first_img,
        first_text,
        EPSILON,
        context + "/a0_fuse_modalities",
        allowed_zero,
    )
    local_key = local_endpoint_std(cache, allowed_zero, np, context + "/local")
    imported_bytes = np.ascontiguousarray(imported_key).tobytes(order="C")
    local_bytes = np.ascontiguousarray(local_key).tobytes(order="C")
    parity = {
        "pass": bool(
            imported_key.dtype == local_key.dtype
            and imported_key.shape == local_key.shape
            and imported_bytes == local_bytes
        ),
        "a0_source_sha256": a0_sha256,
        "construction_steps": [
            "raw image/text modality row L2",
            "fuse_modalities re-L2 each already-normalized modality block",
            "concatenate image/text blocks",
            "final fused row L2",
        ],
        "imported_a0_key": {
            "dtype": imported_key.dtype.str,
            "shape": list(imported_key.shape),
            "sha256": sha256_bytes(imported_bytes),
        },
        "local_explicit_key": {
            "dtype": local_key.dtype.str,
            "shape": list(local_key.shape),
            "sha256": sha256_bytes(local_bytes),
        },
    }
    if not parity["pass"]:
        die("{} endpoint_std construction parity failed".format(context))
    return imported_key, parity


def search(memory, query, faiss, np, k):
    memory = np.ascontiguousarray(memory.astype("float32", copy=True))
    query = np.ascontiguousarray(query.astype("float32", copy=True))
    faiss.normalize_L2(memory)
    faiss.normalize_L2(query)
    index = faiss.IndexFlatIP(memory.shape[1])
    index.add(memory)
    similarities, neighbors = index.search(query, k)
    if (
        neighbors.shape != (len(query), k)
        or np.any(neighbors < 0)
        or np.any(neighbors >= len(memory))
        or not np.isfinite(similarities).all()
    ):
        die("FAISS returned malformed output")
    return (
        similarities.astype("float32", copy=False),
        neighbors.astype("int64", copy=False),
    )


def array_summary(values, np):
    contiguous = np.ascontiguousarray(values)
    raw = contiguous.tobytes(order="C")
    return {
        "dtype": values.dtype.str,
        "shape": list(values.shape),
        "nbytes": len(raw),
        "sha256": sha256_bytes(raw),
    }


def ordered_float32_bits(values, np):
    values = np.ascontiguousarray(values)
    if values.dtype != np.dtype("float32"):
        die("ordered ULP mapping requires float32")
    if not np.isfinite(values).all():
        die("ordered ULP mapping forbids NaN/Inf")
    bits = values.view("uint32")
    negative = (bits & np.uint32(0x80000000)) != 0
    ordered = np.where(
        negative,
        np.bitwise_not(bits),
        np.bitwise_xor(bits, np.uint32(0x80000000)),
    )
    return ordered.astype("uint64")


def float32_ulp_self_check(np):
    source_bits = np.asarray([
        0xbf800000,
        0x80000001,
        0x80000000,
        0x00000000,
        0x00000001,
        0x3f800000,
    ], dtype="uint32")
    values = source_bits.view("float32")
    observed = ordered_float32_bits(values, np)
    expected = np.asarray([
        0x407fffff,
        0x7ffffffe,
        0x7fffffff,
        0x80000000,
        0x80000001,
        0xbf800000,
    ], dtype="uint64")
    forbidden_bits = np.asarray(
        [0x7f800000, 0xff800000, 0x7fc00000], dtype="uint32"
    )
    forbidden_values = forbidden_bits.view("float32")
    checks = {
        "known_ordered_mapping_exact": bool(np.array_equal(observed, expected)),
        "strictly_monotone_minus1_to_plus1": bool(
            np.all(observed[:-1] < observed[1:])
        ),
        "negative_zero_to_positive_zero_ulp": int(
            observed[3] - observed[2]
        ),
        "negative_min_subnormal_to_positive_min_subnormal_ulp": int(
            observed[4] - observed[1]
        ),
        "nan_positive_inf_negative_inf_all_nonfinite": bool(
            not np.isfinite(forbidden_values).any()
        ),
    }
    passed = bool(
        checks["known_ordered_mapping_exact"]
        and checks["strictly_monotone_minus1_to_plus1"]
        and checks["negative_zero_to_positive_zero_ulp"] == 1
        and checks[
            "negative_min_subnormal_to_positive_min_subnormal_ulp"
        ] == 3
        and checks["nan_positive_inf_negative_inf_all_nonfinite"]
    )
    if not passed:
        die("float32 ordered-ULP known-bit self-check failed")
    return {
        "pass": True,
        "mapping": (
            "negative: bitwise_not(raw_uint32); nonnegative: "
            "raw_uint32 xor 0x80000000"
        ),
        "signed_zero_policy": "-0.0 and +0.0 are distinct adjacent codes",
        "cross_sign_policy": "distance crosses -0.0 then +0.0",
        "nonfinite_policy": "NaN and both infinities forbidden",
        "source_float32_hex": [
            "{:08x}".format(int(value)) for value in source_bits
        ],
        "ordered_uint32_hex": [
            "{:08x}".format(int(value)) for value in observed
        ],
        "checks": checks,
    }


def float32_ulp_distance(left, right, np):
    left_ordered = ordered_float32_bits(left, np)
    right_ordered = ordered_float32_bits(right, np)
    return np.maximum(left_ordered, right_ordered) - np.minimum(
        left_ordered, right_ordered
    )


def similarity_compare(left, right, query_ids, np):
    if left.shape != right.shape or left.dtype != right.dtype:
        return {
            "shape_dtype_exact": False,
            "left": array_summary(left, np),
            "right": array_summary(right, np),
        }
    left_bits = np.ascontiguousarray(left).view("uint32")
    right_bits = np.ascontiguousarray(right).view("uint32")
    element_diff = left_bits != right_bits
    absolute = np.abs(left.astype("float64") - right.astype("float64"))
    ulp = float32_ulp_distance(left, right, np)
    examples = [
        {
            "query_id": query_ids[int(row)],
            "rank_zero_based": int(rank),
            "with_similarity": float(left[row, rank]),
            "remove_similarity": float(right[row, rank]),
            "absolute_difference": float(absolute[row, rank]),
            "ulp_difference": int(ulp[row, rank]),
            "with_float32_hex": "{:08x}".format(int(left_bits[row, rank])),
            "remove_float32_hex": "{:08x}".format(int(right_bits[row, rank])),
        }
        for row, rank in np.argwhere(element_diff)[:EXAMPLE_LIMIT]
    ]
    return {
        "shape_dtype_exact": True,
        "c_order_bytes_exact": not bool(np.any(element_diff)),
        "element_byte_diff_count": int(np.sum(element_diff)),
        "max_abs_difference": float(np.max(absolute)),
        "max_ulp_difference": int(np.max(ulp)),
        "first_diff_examples": examples,
        "left": array_summary(left, np),
        "right": array_summary(right, np),
    }


def neighbor_compare(left, right, query_ids, train_ids, left_sim, right_sim, np):
    different = left != right
    order_queries = np.any(different, axis=1)
    set_queries = np.asarray([
        set(left[row].tolist()) != set(right[row].tolist())
        for row in range(len(left))
    ], dtype=bool)
    examples = []
    for row, rank in np.argwhere(different)[:EXAMPLE_LIMIT]:
        li = int(left[row, rank])
        ri = int(right[row, rank])
        examples.append({
            "query_id": query_ids[int(row)],
            "rank_zero_based": int(rank),
            "with_original_index": li,
            "remove_mapped_original_index": ri,
            "with_train_id": train_ids[li],
            "remove_train_id": train_ids[ri],
            "with_similarity": float(left_sim[row, rank]),
            "remove_similarity": float(right_sim[row, rank]),
        })
    return {
        "left": array_summary(left, np),
        "right": array_summary(right, np),
        "element_diff_count": int(np.sum(different)),
        "query_order_diff_count": int(np.sum(order_queries)),
        "query_set_diff_count": int(np.sum(set_queries)),
        "order_diff_query_ids_limited": [
            query_ids[index]
            for index in np.flatnonzero(order_queries)[:EXAMPLE_LIMIT]
        ],
        "set_diff_query_ids_limited": [
            query_ids[index]
            for index in np.flatnonzero(set_queries)[:EXAMPLE_LIMIT]
        ],
        "first_diff_examples": examples,
    }


def weighted_scores(similarities, original_neighbors, labels, np):
    weights = np.arange(TOPK, 0, -1, dtype="float64")
    signed = labels[original_neighbors].astype("float64") * 2.0 - 1.0
    values = np.sum(
        signed * similarities.astype("float64") * weights[None, :], axis=1
    )
    return values / float(np.sum(weights))


def roc_auc(gold, scores, np):
    positives = int(np.sum(gold == 1))
    negatives = int(np.sum(gold == 0))
    if not positives or not negatives:
        return None
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype="float64")
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and scores[order[stop]] == scores[order[start]]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    return float(
        (np.sum(ranks[gold == 1]) - positives * (positives + 1) / 2.0)
        / (positives * negatives)
    )


def metrics(gold, scores, np):
    predictions = (scores >= 0.0).astype("int64")
    f1 = []
    for label in (0, 1):
        tp = int(np.sum((predictions == label) & (gold == label)))
        fp = int(np.sum((predictions == label) & (gold != label)))
        fn = int(np.sum((predictions != label) & (gold == label)))
        denominator = 2 * tp + fp + fn
        f1.append(0.0 if not denominator else 2.0 * tp / denominator)
    return predictions, {
        "accuracy": float(np.mean(predictions == gold)),
        "macro_f1": float(np.mean(f1)),
        "roc_auc": roc_auc(gold, scores, np),
    }


def metric_payload(values):
    typed = {}
    for key in sorted(values):
        value = values[key]
        typed[key] = (
            {"type": "none"}
            if value is None
            else {
                "type": "ieee754_binary64",
                "big_endian_hex": struct.pack(">d", float(value)).hex(),
            }
        )
    payload = json.dumps(
        typed, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return {"typed": typed, "sha256": sha256_bytes(payload)}


def score_prediction_metric_compare(
    left_scores, right_scores, gold, query_ids, np
):
    left_pred, left_metrics = metrics(gold, left_scores, np)
    right_pred, right_metrics = metrics(gold, right_scores, np)
    score_bytes = (
        np.ascontiguousarray(left_scores).tobytes()
        == np.ascontiguousarray(right_scores).tobytes()
    )
    prediction_bytes = (
        np.ascontiguousarray(left_pred).tobytes()
        == np.ascontiguousarray(right_pred).tobytes()
    )
    left_metric_payload = metric_payload(left_metrics)
    right_metric_payload = metric_payload(right_metrics)
    absolute = np.abs(left_scores - right_scores)
    score_diff_rows = np.flatnonzero(
        np.ascontiguousarray(left_scores).view("uint64")
        != np.ascontiguousarray(right_scores).view("uint64")
    )
    prediction_diff_rows = np.flatnonzero(left_pred != right_pred)
    return {
        "scores": {
            "left": array_summary(left_scores, np),
            "right": array_summary(right_scores, np),
            "dtype_shape_bytes_exact": bool(
                left_scores.dtype == right_scores.dtype
                and left_scores.shape == right_scores.shape
                and score_bytes
            ),
            "element_numeric_diff_count": int(np.sum(left_scores != right_scores)),
            "max_abs_difference": float(np.max(absolute)),
            "first_diff_examples": [
                {
                    "query_id": query_ids[int(index)],
                    "with_score": float(left_scores[index]),
                    "remove_score": float(right_scores[index]),
                    "absolute_difference": float(absolute[index]),
                }
                for index in score_diff_rows[:EXAMPLE_LIMIT]
            ],
        },
        "predictions": {
            "left": array_summary(left_pred, np),
            "right": array_summary(right_pred, np),
            "dtype_shape_bytes_exact": bool(
                left_pred.dtype == right_pred.dtype
                and left_pred.shape == right_pred.shape
                and prediction_bytes
            ),
            "diff_count": int(np.sum(left_pred != right_pred)),
            "diff_query_ids_limited": [
                query_ids[int(index)]
                for index in prediction_diff_rows[:EXAMPLE_LIMIT]
            ],
        },
        "metrics": {
            "left": left_metrics,
            "right": right_metrics,
            "left_canonical": left_metric_payload,
            "right_canonical": right_metric_payload,
            "canonical_bytes_exact": (
                left_metric_payload["sha256"] == right_metric_payload["sha256"]
                and left_metric_payload["typed"] == right_metric_payload["typed"]
            ),
        },
    }


def stable_sort(similarities, original_neighbors, np):
    sorted_sim = np.empty_like(similarities)
    sorted_neighbors = np.empty_like(original_neighbors)
    for row in range(len(similarities)):
        order = np.lexsort((
            original_neighbors[row],
            -similarities[row],
        ))
        sorted_sim[row] = similarities[row, order]
        sorted_neighbors[row] = original_neighbors[row, order]
    return sorted_sim, sorted_neighbors


def tie_gap_summary(similarities_21, query_ids, np):
    adjacent = similarities_21[:, :-1] - similarities_21[:, 1:]
    exact_ties = adjacent == 0
    one_ulp_or_less = float32_ulp_distance(
        similarities_21[:, :-1], similarities_21[:, 1:], np
    ) <= 1
    boundary = similarities_21[:, TOPK - 1] - similarities_21[:, TOPK]
    boundary_tie_rows = boundary == 0
    return {
        "adjacent_exact_tie_count_top21": int(np.sum(exact_ties)),
        "adjacent_within_one_ulp_count_top21": int(np.sum(one_ulp_or_less)),
        "minimum_nonnegative_adjacent_gap": float(np.min(adjacent)),
        "rank20_minus_rank21_gap_min": float(np.min(boundary)),
        "rank20_minus_rank21_exact_tie_query_count": int(
            np.sum(boundary_tie_rows)
        ),
        "boundary_tie_query_ids_limited": [
            query_ids[index]
            for index in np.flatnonzero(boundary_tie_rows)[:EXAMPLE_LIMIT]
        ],
    }


def atomic_publish(path, payload):
    if path.exists() or not path.parent.is_dir():
        die("exclusive output target is not fresh")
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
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def main():
    args = parse_args()
    config, config_path = load_config(args.config)
    environment = runtime_guard(config)
    evidence = load_evidence()

    import numpy as np
    import torch
    import faiss

    threads = int(config["execution"]["required_cpus"])
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    faiss.omp_set_num_threads(threads)
    if torch.cuda.is_available():
        die("CUDA visible in CPU-only probe")

    ulp_self_check = float32_ulp_self_check(np)
    a0, a0_path = load_a0_module(np)
    a0_sha256 = sha256_file(a0_path)

    output_dir = repo_path(config["output"]["namespace"], "output directory")
    if output_dir.exists():
        die("probe namespace already exists; no-clobber")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir()

    train = load_cache(
        TRAIN_PATH, evidence["manifest_by_path"][TRAIN_PATH], torch
    )
    dev = load_cache(DEV_PATH, evidence["manifest_by_path"][DEV_PATH], torch)
    if train["ids"][NULL_INDEX] != NULL_ID:
        die("registered null ID/index mismatch")
    if int(train["labels"][NULL_INDEX]) != NULL_LABEL:
        die("registered null label-integrity mismatch")
    train_zero = np.zeros(len(train["ids"]), dtype=bool)
    train_zero[NULL_INDEX] = True
    dev_zero = np.zeros(len(dev["ids"]), dtype=bool)
    train_view, train_key_parity = endpoint_std(
        train,
        train_zero,
        np,
        a0,
        a0_sha256,
        "HateMM/train/standard",
    )
    dev_view, dev_key_parity = endpoint_std(
        dev,
        dev_zero,
        np,
        a0,
        a0_sha256,
        "HateMM/dev_seen/standard",
    )

    keep = np.ones(len(train["ids"]), dtype=bool)
    keep[NULL_INDEX] = False
    original_indices = np.flatnonzero(keep).astype("int64", copy=False)
    expected_mapping = np.arange(len(train["ids"]) - 1, dtype="int64")
    expected_mapping[NULL_INDEX:] += 1
    mapping_exact = bool(np.array_equal(original_indices, expected_mapping))

    full_sim, full_local = search(train_view, dev_view, faiss, np, TOPK)
    reduced_sim, reduced_local = search(
        train_view[keep], dev_view, faiss, np, TOPK
    )
    mapped_reduced = original_indices[reduced_local]
    full_sim21, full_local21 = search(
        train_view, dev_view, faiss, np, TOPK + 1
    )
    reduced_sim21, reduced_local21 = search(
        train_view[keep], dev_view, faiss, np, TOPK + 1
    )

    null_positions = np.argwhere(full_local == NULL_INDEX)
    null_examples = [
        {
            "query_id": dev["ids"][int(row)],
            "rank_zero_based": int(rank),
            "similarity": float(full_sim[row, rank]),
        }
        for row, rank in null_positions[:EXAMPLE_LIMIT]
    ]

    raw_neighbor = neighbor_compare(
        full_local,
        mapped_reduced,
        dev["ids"],
        train["ids"],
        full_sim,
        reduced_sim,
        np,
    )
    raw_similarity = similarity_compare(
        full_sim, reduced_sim, dev["ids"], np
    )
    raw_full_scores = weighted_scores(
        full_sim, full_local, train["labels"], np
    )
    raw_reduced_scores = weighted_scores(
        reduced_sim, mapped_reduced, train["labels"], np
    )
    raw_outcomes = score_prediction_metric_compare(
        raw_full_scores, raw_reduced_scores, dev["labels"], dev["ids"], np
    )

    stable_full_sim, stable_full_neighbors = stable_sort(
        full_sim, full_local, np
    )
    stable_reduced_sim, stable_reduced_neighbors = stable_sort(
        reduced_sim, mapped_reduced, np
    )
    stable_neighbor = neighbor_compare(
        stable_full_neighbors,
        stable_reduced_neighbors,
        dev["ids"],
        train["ids"],
        stable_full_sim,
        stable_reduced_sim,
        np,
    )
    stable_similarity = similarity_compare(
        stable_full_sim, stable_reduced_sim, dev["ids"], np
    )
    stable_full_scores = weighted_scores(
        stable_full_sim, stable_full_neighbors, train["labels"], np
    )
    stable_reduced_scores = weighted_scores(
        stable_reduced_sim, stable_reduced_neighbors, train["labels"], np
    )
    stable_outcomes = score_prediction_metric_compare(
        stable_full_scores,
        stable_reduced_scores,
        dev["labels"],
        dev["ids"],
        np,
    )

    full_ties = tie_gap_summary(full_sim21, dev["ids"], np)
    reduced_ties = tie_gap_summary(reduced_sim21, dev["ids"], np)
    if not mapping_exact:
        diagnosis = "MAPPING_BUG"
    elif len(null_positions):
        diagnosis = "REGISTERED_NULL_SELECTED_IN_TOP20"
    elif (
        raw_neighbor["element_diff_count"] > 0
        and stable_neighbor["element_diff_count"] == 0
        and stable_similarity.get("c_order_bytes_exact", False)
    ):
        diagnosis = "RAW_FAISS_TIE_ORDER"
    elif (
        stable_neighbor["element_diff_count"] == 0
        and not stable_similarity.get("c_order_bytes_exact", False)
    ):
        diagnosis = "FLOAT_VARIATION_WITH_STABLE_NEIGHBOR_IDENTITY"
    elif raw_neighbor["element_diff_count"] > 0:
        diagnosis = "MIXED_OR_FLOAT_VARIATION"
    elif not raw_similarity.get("c_order_bytes_exact", False):
        diagnosis = "FLOAT_VARIATION"
    else:
        diagnosis = "NO_MISMATCH_REPRODUCED"

    result = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "status": "DIAGNOSTIC_ONLY",
        "scope": (
            "HateMM train-memory/dev_seen-query standard endpoint_std only; "
            "no scientific decision"
        ),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "thread_environment": environment,
        "config": {
            "path": str(config_path.relative_to(REPO)),
            "sha256": sha256_file(config_path),
        },
        "endpoint_std_key_construction": {
            "source": str(a0_path.relative_to(REPO)),
            "source_sha256": a0_sha256,
            "train_parity": train_key_parity,
            "dev_seen_parity": dev_key_parity,
        },
        "float32_ordered_ulp": ulp_self_check,
        "evidence": {
            "manifest_path": MANIFEST_PATH,
            "manifest_sha256": MANIFEST_SHA256,
            "zero_probe_path": ZERO_PROBE_PATH,
            "zero_probe_sha256": ZERO_PROBE_SHA256,
            "authorized_null": {
                "dataset": "HateMM",
                "split": "train",
                "id": NULL_ID,
                "row_index": NULL_INDEX,
                "expected_label_integrity_only": NULL_LABEL,
                "modalities": ["img", "text"],
                "policy": "standard",
            },
        },
        "inputs": {
            "train": {
                "path": TRAIN_PATH,
                "sha256": train["sha256"],
                "bytes": train["bytes"],
                "n": len(train["ids"]),
            },
            "dev_seen": {
                "path": DEV_PATH,
                "sha256": dev["sha256"],
                "bytes": dev["bytes"],
                "n": len(dev["ids"]),
            },
            "test_like_attempt_count": 0,
            "test_like_open_count": 0,
            "feature_vectors_serialized": False,
        },
        "index_mapping": {
            "with_null_train_size": len(train["ids"]),
            "remove_null_train_size": int(np.sum(keep)),
            "removed_original_index": NULL_INDEX,
            "local_to_original_dtype_shape_hash": array_summary(
                original_indices, np
            ),
            "formula": "original=local if local<355 else local+1",
            "formula_exact_for_all_indices": mapping_exact,
            "limited_examples": [
                {"local": int(index), "original": int(original_indices[index])}
                for index in (0, 354, 355, len(original_indices) - 1)
            ],
        },
        "registered_null_top20": {
            "occurrence_count": int(len(null_positions)),
            "query_count": int(len(set(
                int(row) for row, _ in null_positions.tolist()
            ))),
            "query_id_rank_examples_limited": null_examples,
            "null_similarity_is_zero_by_exact_zero_vector": True,
        },
        "raw_faiss_order": {
            "neighbors_original_vs_mapped": raw_neighbor,
            "similarities": raw_similarity,
            "scores_predictions_metrics": raw_outcomes,
        },
        "stable_similarity_then_original_index_order": {
            "rule": "lexicographic (-FAISS_float32_similarity, original_train_index)",
            "neighbors_original_vs_mapped": stable_neighbor,
            "similarities": stable_similarity,
            "scores_predictions_metrics": stable_outcomes,
        },
        "ties_and_gaps": {
            "with_null": full_ties,
            "remove_null": reduced_ties,
        },
        "diagnosis": {
            "classification": diagnosis,
            "mapping_bug": not mapping_exact,
            "null_selected": bool(len(null_positions)),
            "raw_neighbor_set_diff_queries": raw_neighbor[
                "query_set_diff_count"
            ],
            "raw_neighbor_order_diff_queries": raw_neighbor[
                "query_order_diff_count"
            ],
            "stable_neighbor_set_diff_queries": stable_neighbor[
                "query_set_diff_count"
            ],
            "stable_neighbor_order_diff_queries": stable_neighbor[
                "query_order_diff_count"
            ],
            "raw_similarity_bytes_exact": raw_similarity.get(
                "c_order_bytes_exact", False
            ),
            "stable_similarity_bytes_exact": stable_similarity.get(
                "c_order_bytes_exact", False
            ),
        },
        "nonclaims": [
            "This probe does not modify or authorize A0.",
            "This probe does not produce a CONTINUE/KILL decision.",
            "No test split, feature vector, transcript, or full neighbor array is serialized.",
        ],
    }
    payload = (
        json.dumps(
            result, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n"
    ).encode("utf-8")
    if len(payload) > int(config["output"]["maximum_bytes"]):
        die("diagnostic JSON exceeds small-output budget")
    atomic_publish(output_dir / config["output"]["file"], payload)
    print(json.dumps({
        "run_id": RUN_ID,
        "status": "DIAGNOSTIC_ONLY",
        "diagnosis": diagnosis,
        "artifact": str(
            (output_dir / config["output"]["file"]).relative_to(REPO)
        ),
        "artifact_sha256": sha256_bytes(payload),
    }, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(
            "C01_RETRIEVAL_EQUIVALENCE_PROBE_FAIL_CLOSED: {}".format(exc),
            file=sys.stderr,
        )
        raise
