#!/usr/bin/env python3
"""C01 zero-row contract probe.

Slurm-only and read-only with respect to source data.  The probe opens exactly
the eight train/dev_seen caches registered by the immutable C01 full-SHA256
manifest.  It publishes metadata about zero/tiny rows, IDs, labels, and paired
endpoint masks; it never serializes feature values.
"""

import hashlib
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO / "configs" / "c01" / "c01_a0_v1.json"
RUN_ID = "C01-ZERO-PROBE-v1"
SCHEMA_VERSION = "c01_zero_contract_probe_v1"
NAMESPACE = (
    REPO
    / "artifacts"
    / "c01_policy_contrastive"
    / "v2"
    / "zero_contract_probe"
    / RUN_ID
)
OUTPUT_NAME = "zero_contract_probe.json"
REQUIRED_DATASETS = ("MHC_zh", "HateMM")
REQUIRED_SPLITS = ("train", "dev_seen")
REQUIRED_POLICIES = ("standard", "oneword")
MODALITIES = ("img", "text")
FEATURE_KEYS = {"ids", "img_feats", "text_feats", "labels"}
EPSILON = 1e-12
MAX_OUTPUT_BYTES = 1_000_000
APPROVED_MANIFEST_SHA256 = (
    "083275d39a1026bde3b6583bd5608d41"
    "cec5b431da9ffda87ae8ab1046cf2305"
)
ROW_STATE_ENUM = (
    "normal_nonzero",
    "exact_zero",
    "tiny_nonzero",
    "nonfinite",
)
EXACT_ZERO_SEMANTICS = (
    "observation_only_structural_interpretation_requires_external_"
    "evidence_and_review"
)
REQUIRED_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "",
    "OMP_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}
MANIFEST_TOP_KEYS = {
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
MANIFEST_FILE_KEYS = {
    "ordinal",
    "dataset",
    "split",
    "policy",
    "path",
    "bytes",
    "provenance_sha16",
    "sha256",
}
MANIFEST_LEDGER_KEYS = {
    "ordinal",
    "path",
    "dataset",
    "split",
    "policy",
    "open_attempted",
    "opened",
    "bytes_read",
    "test_like",
}


def die(message):
    raise RuntimeError(message)


def has_test_token(value):
    tokens = [
        token
        for token in re.split(r"[^a-z0-9]+", str(value).lower())
        if token
    ]
    return "test" in tokens


def repo_path(relative, context):
    if not isinstance(relative, str) or not relative:
        die("{} must be a non-empty repo-relative path".format(context))
    if has_test_token(relative):
        die("test-like path blocked before open: {}".format(relative))
    path = (REPO / relative).resolve()
    try:
        path.relative_to(REPO)
    except ValueError:
        die("{} escapes repository root".format(context))
    if has_test_token(path):
        die("test-like resolved path blocked before open: {}".format(path))
    return path


def enforce_runtime():
    job_id = os.environ.get("SLURM_JOB_ID")
    if not isinstance(job_id, str) or not re.fullmatch(r"[1-9][0-9]*", job_id):
        die("probe requires a positive-integer SLURM_JOB_ID")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        die("probe requires exactly one Slurm CPU")
    for key, expected in REQUIRED_ENVIRONMENT.items():
        if os.environ.get(key) != expected:
            die("{} must equal {!r}".format(key, expected))
    return job_id


def sha256_file(path):
    digest = hashlib.sha256()
    bytes_read = 0
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
            bytes_read += len(block)
    return digest.hexdigest(), bytes_read


def ensure_json_finite(value, location="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            ensure_json_finite(child, location + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            ensure_json_finite(child, location + "[{}]".format(index))
    elif isinstance(value, float) and not math.isfinite(value):
        die("non-finite JSON value at {}".format(location))


def json_bytes(value):
    ensure_json_finite(value)
    return (
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def exclusive_publish(path, payload):
    if not path.parent.is_dir() or path.exists():
        die("exclusive output target is not fresh: {}".format(path))
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


def expected_manifest_records(config):
    if config.get("schema_version") != 1 or config.get("experiment_id") != "C01_A0":
        die("unsupported C01 config identity")
    inputs = config.get("inputs")
    if not isinstance(inputs, dict):
        die("C01 input config is malformed")
    if tuple(inputs.get("allowed_datasets", ())) != REQUIRED_DATASETS:
        die("probe dataset binding changed")
    if tuple(inputs.get("allowed_splits", ())) != REQUIRED_SPLITS:
        die("probe split binding changed")
    if inputs.get("feature_dim") != 3584:
        die("probe is frozen to feature_dim=3584")
    if set(inputs.get("exact_contract_keys", ())) != FEATURE_KEYS:
        die("probe cache contract changed")
    if (
        inputs.get("standard_suffix") != "ro_L24"
        or inputs.get("oneword_suffix") != "ro_ow_L24"
    ):
        die("probe endpoint suffix binding changed")
    if set(inputs.get("datasets", {})) != set(REQUIRED_DATASETS):
        die("probe dataset config family changed")
    if (
        config.get("full_sha256_preflight", {}).get(
            "approved_manifest_sha256"
        )
        != APPROVED_MANIFEST_SHA256
    ):
        die("approved full-SHA256 manifest binding changed")
    decision_schema = config.get("output", {}).get("decision_schema", {})
    if (
        decision_schema.get(
            "zero_contract_probe_other_modality_state_enum"
        )
        != list(ROW_STATE_ENUM)
        or decision_schema.get(
            "zero_contract_probe_exact_zero_semantics"
        )
        != EXACT_ZERO_SEMANTICS
    ):
        die("zero-contract probe observation schema changed")

    records = []
    for dataset in REQUIRED_DATASETS:
        dataset_config = inputs["datasets"][dataset]
        for split in REQUIRED_SPLITS:
            if has_test_token(split):
                die("test-like split blocked: {}".format(split))
            expected = dataset_config["expected"][split]
            for policy in REQUIRED_POLICIES:
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
                            dataset_config["base_tag"],
                            suffix,
                        ),
                        "bytes": int(expected[policy + "_bytes"]),
                        "provenance_sha16": expected[
                            policy + "_provenance_sha16"
                        ],
                        "n": int(expected["n"]),
                    }
                )
    if len(records) != 8:
        die("probe requires exactly eight registered records")
    if any(has_test_token(record["path"]) for record in records):
        die("test-like record entered probe allowlist")
    return records


def validate_manifest(config, records):
    preflight = config["full_sha256_preflight"]
    manifest_path = repo_path(
        preflight["manifest_path"], "full_sha256_preflight.manifest_path"
    )
    if not manifest_path.is_file():
        die("required immutable full-SHA256 manifest is absent")
    raw = manifest_path.read_bytes()
    manifest_sha256 = hashlib.sha256(raw).hexdigest()
    if manifest_sha256 != APPROVED_MANIFEST_SHA256:
        die("manifest is not the exact approved full-SHA256 artifact")
    manifest = json.loads(raw.decode("utf-8"))
    if not isinstance(manifest, dict) or set(manifest) != MANIFEST_TOP_KEYS:
        die("full-SHA256 manifest top-level schema changed")
    if (
        manifest["schema_version"] != preflight["schema_version"]
        or manifest["run_id"] != preflight["run_id"]
        or manifest["source_set_id"] != preflight["source_set_id"]
        or manifest["producer"] != preflight["producer_script"]
        or manifest["complete"] is not True
        or manifest["expected_file_count"] != 8
    ):
        die("full-SHA256 manifest identity/completeness mismatch")
    if manifest["thread_environment"] != REQUIRED_ENVIRONMENT:
        die("hash manifest thread environment changed")
    manifest_job_id = manifest["slurm_job_id"]
    if (
        not isinstance(manifest_job_id, str)
        or not re.fullmatch(r"[1-9][0-9]*", manifest_job_id)
    ):
        die("hash manifest Slurm job ID is invalid")

    files = manifest["files"]
    ledger = manifest["access_ledger"]
    if (
        not isinstance(files, list)
        or not isinstance(ledger, list)
        or len(files) != len(records)
        or len(ledger) != len(records)
    ):
        die("hash manifest file/ledger cardinality mismatch")
    manifest_digests = {}
    for registered, observed_file, observed_access in zip(records, files, ledger):
        if (
            not isinstance(observed_file, dict)
            or set(observed_file) != MANIFEST_FILE_KEYS
        ):
            die("hash manifest file-entry schema changed")
        expected_file = {
            key: registered[key]
            for key in (
                "ordinal",
                "dataset",
                "split",
                "policy",
                "path",
                "bytes",
                "provenance_sha16",
            )
        }
        observed_identity = {
            key: observed_file[key] for key in expected_file
        }
        if observed_identity != expected_file:
            die("hash manifest file identity/order changed")
        digest = observed_file["sha256"]
        if (
            not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not digest.startswith(registered["provenance_sha16"])
        ):
            die("hash manifest digest is invalid for {}".format(registered["path"]))
        if registered["path"] in manifest_digests:
            die("duplicate cache path in hash manifest")
        manifest_digests[registered["path"]] = digest

        if (
            not isinstance(observed_access, dict)
            or set(observed_access) != MANIFEST_LEDGER_KEYS
        ):
            die("hash manifest access-ledger schema changed")
        expected_access = {
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
        if (
            type(observed_access["ordinal"]) is not int
            or type(observed_access["bytes_read"]) is not int
            or observed_access["open_attempted"] is not True
            or observed_access["opened"] is not True
            or observed_access["test_like"] is not False
            or observed_access != expected_access
        ):
            die("hash manifest access-ledger entry mismatch")
    if manifest["access_summary"] != {
        "opened_count": 8,
        "test_like_attempt_count": 0,
        "test_like_open_count": 0,
    }:
        die("hash manifest access summary changed")
    return {
        "path": str(manifest_path.relative_to(REPO)),
        "sha256": manifest_sha256,
        "run_id": manifest["run_id"],
        "slurm_job_id": manifest_job_id,
        "digests": manifest_digests,
    }


def normalize_ids(raw, context):
    if not isinstance(raw, (list, tuple)) or len(raw) != 1:
        die("{} IDs violate one-sublist cache contract".format(context))
    if not isinstance(raw[0], (list, tuple)):
        die("{} ids[0] is not a list".format(context))
    ids = list(raw[0])
    if (
        any(type(value) is not str or not value for value in ids)
        or len(set(ids)) != len(ids)
    ):
        die("{} IDs must be raw non-empty unique strings".format(context))
    return ids


def load_registered_cache(record, expected_digest, torch):
    path = repo_path(record["path"], "registered cache")
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        die("registered cache path/size mismatch: {}".format(record["path"]))
    digest, bytes_hashed = sha256_file(path)
    if bytes_hashed != record["bytes"] or digest != expected_digest:
        die("cache failed exact full-SHA256 check before torch.load")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict) or set(payload) != FEATURE_KEYS:
        die("cache payload contract changed: {}".format(record["path"]))
    context = "{}/{}/{}".format(
        record["dataset"], record["split"], record["policy"]
    )
    ids = normalize_ids(payload["ids"], context)
    img = payload["img_feats"]
    text = payload["text_feats"]
    labels = payload["labels"]
    if not all(torch.is_tensor(value) for value in (img, text, labels)):
        die("{} cache tensors are malformed".format(context))
    img = img.detach().cpu()
    text = text.detach().cpu()
    labels = labels.detach().cpu()
    n = record["n"]
    if (
        tuple(img.shape) != (n, 3584)
        or tuple(text.shape) != (n, 3584)
        or tuple(labels.shape) != (n,)
        or len(ids) != n
    ):
        die("{} cache shape/count contract changed".format(context))
    if not bool(torch.isfinite(labels).all()):
        die("{} labels contain NaN/Inf".format(context))
    rounded = torch.round(labels.to(torch.float64)).to(torch.int64)
    if (
        not bool(torch.equal(labels.to(torch.float64), rounded.to(torch.float64)))
        or not set(rounded.tolist()).issubset({0, 1})
    ):
        die("{} labels are not binary integers".format(context))
    return {
        "ids": ids,
        "img": img,
        "text": text,
        "labels": rounded,
        "path": record["path"],
        "sha256": digest,
        "bytes": bytes_hashed,
    }


def row_descriptor(index, ids, labels):
    return {
        "row_index": int(index),
        "id": ids[index],
        "label": int(labels[index].item()),
    }


def classify_modality_row(row, torch):
    if not bool(torch.isfinite(row).all()):
        return "nonfinite"
    if bool((row == 0).all()):
        return "exact_zero"
    norm = torch.linalg.vector_norm(row.to(torch.float64))
    if bool(norm <= EPSILON):
        return "tiny_nonzero"
    return "normal_nonzero"


def modality_summary(features, other_features, ids, labels, torch):
    finite_mask = torch.isfinite(features).all(dim=1)
    exact_zero_mask = (features == 0).all(dim=1)
    norms = torch.linalg.vector_norm(features.to(torch.float64), dim=1)
    tiny_nonzero_mask = (
        finite_mask & (~exact_zero_mask) & (norms <= EPSILON)
    )
    zero_rows = []
    for index in torch.nonzero(exact_zero_mask, as_tuple=False).flatten().tolist():
        row = row_descriptor(index, ids, labels)
        row["other_modality_state"] = classify_modality_row(
            other_features[index], torch
        )
        zero_rows.append(row)
    tiny_rows = [
        row_descriptor(index, ids, labels)
        for index in torch.nonzero(
            tiny_nonzero_mask, as_tuple=False
        ).flatten().tolist()
    ]
    nonfinite_rows = [
        row_descriptor(index, ids, labels)
        for index in torch.nonzero(
            ~finite_mask, as_tuple=False
        ).flatten().tolist()
    ]
    return {
        "zero_row_count": len(zero_rows),
        "zero_row_ids": [row["id"] for row in zero_rows],
        "zero_rows": zero_rows,
        "tiny_nonzero_row_count": len(tiny_rows),
        "tiny_nonzero_row_ids": [row["id"] for row in tiny_rows],
        "tiny_nonzero_rows": tiny_rows,
        "nonfinite_row_count": len(nonfinite_rows),
        "nonfinite_row_ids": [row["id"] for row in nonfinite_rows],
        "nonfinite_rows": nonfinite_rows,
    }, exact_zero_mask, tiny_nonzero_mask, finite_mask


def ids_from_mask(mask, ids, torch):
    return [
        ids[index]
        for index in torch.nonzero(mask, as_tuple=False).flatten().tolist()
    ]


def analyse_records(records, manifest, torch):
    cache_by_key = {}
    access_ledger = []
    for record in records:
        if has_test_token(record["split"]) or has_test_token(record["path"]):
            die("test-like input reached probe runtime")
        cache = load_registered_cache(
            record, manifest["digests"][record["path"]], torch
        )
        key = (record["dataset"], record["split"], record["policy"])
        cache_by_key[key] = cache
        access_ledger.append(
            {
                "ordinal": record["ordinal"],
                "dataset": record["dataset"],
                "split": record["split"],
                "policy": record["policy"],
                "path": record["path"],
                "bytes_hashed": cache["bytes"],
                "sha256_matched_before_torch_load": True,
                "torch_loaded": True,
                "test_like": False,
            }
        )

    datasets = {}
    all_endpoint_masks_match = True
    all_tiny_nonzero_absent = True
    all_nonfinite_absent = True
    for dataset in REQUIRED_DATASETS:
        dataset_output = {"splits": {}}
        for split in REQUIRED_SPLITS:
            standard = cache_by_key[(dataset, split, "standard")]
            oneword = cache_by_key[(dataset, split, "oneword")]
            if standard["ids"] != oneword["ids"]:
                die("{}/{} endpoint ID order mismatch".format(dataset, split))
            if not bool(torch.equal(standard["labels"], oneword["labels"])):
                die("{}/{} endpoint label mismatch".format(dataset, split))

            split_output = {
                "n": len(standard["ids"]),
                "endpoint_ids_exact_match": True,
                "endpoint_labels_exact_match": True,
                "policies": {},
                "endpoint_comparison": {},
            }
            masks = {}
            for policy, cache in (
                ("standard", standard),
                ("oneword", oneword),
            ):
                policy_output = {
                    "path": cache["path"],
                    "sha256": cache["sha256"],
                    "modalities": {},
                }
                masks[policy] = {}
                for modality, other_modality in (
                    ("img", "text"),
                    ("text", "img"),
                ):
                    summary, zero_mask, tiny_mask, finite_mask = modality_summary(
                        cache[modality],
                        cache[other_modality],
                        cache["ids"],
                        cache["labels"],
                        torch,
                    )
                    policy_output["modalities"][modality] = summary
                    masks[policy][modality] = {
                        "zero": zero_mask,
                        "tiny": tiny_mask,
                        "finite": finite_mask,
                    }
                    all_tiny_nonzero_absent = (
                        all_tiny_nonzero_absent
                        and summary["tiny_nonzero_row_count"] == 0
                    )
                    all_nonfinite_absent = (
                        all_nonfinite_absent
                        and summary["nonfinite_row_count"] == 0
                    )
                split_output["policies"][policy] = policy_output

            for modality in MODALITIES:
                standard_zero = masks["standard"][modality]["zero"]
                oneword_zero = masks["oneword"][modality]["zero"]
                exact_match = bool(torch.equal(standard_zero, oneword_zero))
                all_endpoint_masks_match = all_endpoint_masks_match and exact_match
                split_output["endpoint_comparison"][modality] = {
                    "zero_mask_exact_match": exact_match,
                    "matched_zero_ids": ids_from_mask(
                        standard_zero & oneword_zero,
                        standard["ids"],
                        torch,
                    ),
                    "standard_only_zero_ids": ids_from_mask(
                        standard_zero & (~oneword_zero),
                        standard["ids"],
                        torch,
                    ),
                    "oneword_only_zero_ids": ids_from_mask(
                        oneword_zero & (~standard_zero),
                        standard["ids"],
                        torch,
                    ),
                }
            dataset_output["splits"][split] = split_output
        datasets[dataset] = dataset_output
    return {
        "datasets": datasets,
        "access_ledger": access_ledger,
        "all_endpoint_zero_masks_exact_match": all_endpoint_masks_match,
        "all_tiny_nonzero_rows_absent": all_tiny_nonzero_absent,
        "all_nonfinite_rows_absent": all_nonfinite_absent,
    }


def main():
    job_id = enforce_runtime()
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = json.load(handle)
    records = expected_manifest_records(config)
    manifest = validate_manifest(config, records)

    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if torch.cuda.is_available():
        die("CUDA is visible despite CPU-only probe contract")

    analysis = analyse_records(records, manifest, torch)
    output = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "status": "DIAGNOSTIC_ONLY",
        "slurm_job_id": job_id,
        "thread_environment": REQUIRED_ENVIRONMENT,
        "config": str(CONFIG_PATH.relative_to(REPO)),
        "hash_manifest": {
            key: manifest[key]
            for key in ("path", "sha256", "run_id", "slurm_job_id")
        },
        "input_scope": {
            "datasets": list(REQUIRED_DATASETS),
            "splits": list(REQUIRED_SPLITS),
            "policies": list(REQUIRED_POLICIES),
            "modalities": list(MODALITIES),
            "expected_cache_count": 8,
            "actual_cache_count": len(analysis["access_ledger"]),
            "test_like_attempt_count": 0,
            "test_like_open_count": 0,
            "features_serialized": False,
            "other_modality_state_enum": list(ROW_STATE_ENUM),
            "exact_zero_semantics": EXACT_ZERO_SEMANTICS,
        },
        "datasets": analysis["datasets"],
        "access_ledger": analysis["access_ledger"],
        "v2_repair_preregistered_assessment": {
            "probe_observation_scope": EXACT_ZERO_SEMANTICS,
            "endpoint_zero_masks_exact_match": analysis[
                "all_endpoint_zero_masks_exact_match"
            ],
            "non_structural_tiny_absent": analysis[
                "all_tiny_nonzero_rows_absent"
            ],
            "nonfinite_rows_absent": analysis[
                "all_nonfinite_rows_absent"
            ],
            "historical_baseline_same_id_consumption": {
                "status": "REPORTED_EXTERNAL_NOT_VERIFIED_BY_PROBE",
                "not_verified_by_this_eight_cache_probe": True,
                "evidence": [
                    {
                        "path": "refine-logs/PROVENANCE_AUDIT_2026-07-28.md",
                        "lines": "187-193",
                    },
                    {
                        "path": "refine-logs/MNTP_S1_RECORD.md",
                        "lines": "183-185,227-230",
                    },
                ],
            },
            "allow_zero_block_in_a0": False,
            "authorization": (
                "This probe never authorizes an A0 change by itself. C01 v2 may "
                "retain a zero block only when endpoint zero masks align at the "
                "same ID and modality, no non-structural tiny row exists, and "
                "frozen evidence confirms the historical baseline consumed the "
                "same structural null. Any endpoint mismatch, non-structural "
                "tiny row, non-finite row, or absent baseline evidence remains "
                "fail-closed."
            ),
        },
    }
    payload = json_bytes(output)
    if len(payload) > MAX_OUTPUT_BYTES:
        die("zero-contract probe output exceeds small-artifact budget")
    if NAMESPACE.exists():
        die("probe namespace already exists; no-clobber requires a new run ID")
    NAMESPACE.parent.mkdir(parents=True, exist_ok=True)
    NAMESPACE.mkdir()
    destination = NAMESPACE / OUTPUT_NAME
    exclusive_publish(destination, payload)
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "artifact": str(destination.relative_to(REPO)),
                "artifact_sha256": hashlib.sha256(payload).hexdigest(),
                "endpoint_zero_masks_exact_match": analysis[
                    "all_endpoint_zero_masks_exact_match"
                ],
                "non_structural_tiny_absent": analysis[
                    "all_tiny_nonzero_rows_absent"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("C01_ZERO_CONTRACT_PROBE_FAIL_CLOSED: {}".format(exc), file=sys.stderr)
        raise
