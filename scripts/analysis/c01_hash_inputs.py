#!/usr/bin/env python3
"""Slurm-only, read-only full-SHA256 preflight for the eight C01 A0 inputs.

This preflight never imports torch and never opens a test-like path.  It verifies
the historical size+sha16 provenance guards, then publishes one immutable
full-SHA256 manifest in an exclusive run namespace.
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
SCHEMA_VERSION = "c01_full_sha256_manifest_v1"
RUN_ID = "C01-HASH-v1"
SOURCE_SET_ID = "c01_l24_train_dev_eight_files_v1"
NAMESPACE = (
    REPO
    / "artifacts"
    / "c01_policy_contrastive"
    / "v1"
    / "hash_preflight"
    / RUN_ID
)
MANIFEST_NAME = "full_sha256_manifest.json"
EXPECTED_FILES = (
    {
        "dataset": "MHC_zh",
        "split": "train",
        "policy": "standard",
        "path": "data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct-LoRA_HF-ro_L24.pt",
        "bytes": 16619920,
        "provenance_sha16": "1d33fe5d69083479",
    },
    {
        "dataset": "MHC_zh",
        "split": "train",
        "policy": "oneword",
        "path": "data/CLIP_Embedding/MHC_zh/train_Qwen2.5-VL-7B-Instruct-LoRA_HF-ro_ow_L24.pt",
        "bytes": 16619941,
        "provenance_sha16": "3ad1309dc7500182",
    },
    {
        "dataset": "MHC_zh",
        "split": "dev_seen",
        "policy": "standard",
        "path": "data/CLIP_Embedding/MHC_zh/dev_seen_Qwen2.5-VL-7B-Instruct-LoRA_HF-ro_L24.pt",
        "bytes": 2240677,
        "provenance_sha16": "a4cf072837e6fe6b",
    },
    {
        "dataset": "MHC_zh",
        "split": "dev_seen",
        "policy": "oneword",
        "path": "data/CLIP_Embedding/MHC_zh/dev_seen_Qwen2.5-VL-7B-Instruct-LoRA_HF-ro_ow_L24.pt",
        "bytes": 2240698,
        "provenance_sha16": "17c4efb2f7a0c2c0",
    },
    {
        "dataset": "HateMM",
        "split": "train",
        "policy": "standard",
        "path": "data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF-ro_L24.pt",
        "bytes": 21358913,
        "provenance_sha16": "6a44cce4f65d4a60",
    },
    {
        "dataset": "HateMM",
        "split": "train",
        "policy": "oneword",
        "path": "data/CLIP_Embedding/HateMM/train_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF-ro_ow_L24.pt",
        "bytes": 21358934,
        "provenance_sha16": "60054f3be1204ca7",
    },
    {
        "dataset": "HateMM",
        "split": "dev_seen",
        "policy": "standard",
        "path": "data/CLIP_Embedding/HateMM/dev_seen_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF-ro_L24.pt",
        "bytes": 3073494,
        "provenance_sha16": "92a17d42627cb4b1",
    },
    {
        "dataset": "HateMM",
        "split": "dev_seen",
        "policy": "oneword",
        "path": "data/CLIP_Embedding/HateMM/dev_seen_Qwen2.5-VL-7B-Instruct-LoRA-curric_HF-ro_ow_L24.pt",
        "bytes": 3073579,
        "provenance_sha16": "07c19096a054845a",
    },
)


def die(message):
    raise RuntimeError(message)


def has_test_token(value):
    tokens = [token for token in re.split(r"[^a-z0-9]+", str(value).lower()) if token]
    return "test" in tokens


def canonical_repo_path(relative):
    if not isinstance(relative, str) or not relative:
        die("input path must be a non-empty repo-relative string")
    path = (REPO / relative).resolve()
    try:
        path.relative_to(REPO)
    except ValueError:
        die("input path escapes repository root: {}".format(relative))
    if has_test_token(relative) or has_test_token(path):
        die("test-like input path blocked: {}".format(relative))
    return path


def sha256_file(path, ledger_entry):
    ledger_entry["open_attempted"] = True
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        ledger_entry["opened"] = True
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    ledger_entry["bytes_read"] = path.stat().st_size
    return digest.hexdigest()


def ensure_json_finite(value, location="root"):
    if isinstance(value, dict):
        for key, child in value.items():
            ensure_json_finite(child, location + "." + str(key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            ensure_json_finite(child, location + "[{}]".format(index))
    elif isinstance(value, float) and not math.isfinite(value):
        die("non-finite JSON at {}".format(location))


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


def validate_runtime():
    if not os.environ.get("SLURM_JOB_ID"):
        die("hash preflight is Slurm-only")
    if os.environ.get("SLURM_CPUS_PER_TASK") != "1":
        die("hash preflight requires exactly one Slurm CPU")
    required = {
        "CUDA_VISIBLE_DEVICES": "",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
    }
    for key, expected in required.items():
        if os.environ.get(key) != expected:
            die("{} must equal {!r}".format(key, expected))
    return required


def main():
    thread_environment = validate_runtime()
    if len(EXPECTED_FILES) != 8:
        die("canonical input binding must contain exactly eight files")
    keys = [
        (entry["dataset"], entry["split"], entry["policy"], entry["path"])
        for entry in EXPECTED_FILES
    ]
    if len(set(keys)) != len(keys):
        die("canonical input binding contains duplicates")
    if {entry["dataset"] for entry in EXPECTED_FILES} != {"MHC_zh", "HateMM"}:
        die("canonical dataset binding changed")
    if {entry["split"] for entry in EXPECTED_FILES} != {"train", "dev_seen"}:
        die("canonical split binding changed")
    if {entry["policy"] for entry in EXPECTED_FILES} != {"standard", "oneword"}:
        die("canonical policy binding changed")

    NAMESPACE.parent.mkdir(parents=True, exist_ok=True)
    try:
        NAMESPACE.mkdir()
    except FileExistsError:
        die(
            "hash-preflight run namespace already exists; no-clobber requires a new run_id"
        )

    access_ledger = []
    manifest_entries = []
    for ordinal, expected in enumerate(EXPECTED_FILES):
        path = canonical_repo_path(expected["path"])
        ledger = {
            "ordinal": ordinal,
            "path": expected["path"],
            "dataset": expected["dataset"],
            "split": expected["split"],
            "policy": expected["policy"],
            "open_attempted": False,
            "opened": False,
            "bytes_read": 0,
            "test_like": has_test_token(expected["path"]),
        }
        access_ledger.append(ledger)
        if not path.is_file():
            die("missing canonical cache: {}".format(expected["path"]))
        observed_bytes = path.stat().st_size
        if observed_bytes != expected["bytes"]:
            die("size provenance mismatch: {}".format(expected["path"]))
        digest = sha256_file(path, ledger)
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            die("invalid full SHA256: {}".format(expected["path"]))
        if not digest.startswith(expected["provenance_sha16"]):
            die("sha16 provenance mismatch: {}".format(expected["path"]))
        manifest_entries.append(
            {
                "ordinal": ordinal,
                "dataset": expected["dataset"],
                "split": expected["split"],
                "policy": expected["policy"],
                "path": expected["path"],
                "bytes": observed_bytes,
                "provenance_sha16": expected["provenance_sha16"],
                "sha256": digest,
            }
        )

    test_like_attempts = sum(1 for entry in access_ledger if entry["test_like"])
    opened_count = sum(1 for entry in access_ledger if entry["opened"])
    if test_like_attempts != 0 or opened_count != len(EXPECTED_FILES):
        die("access-ledger completeness/test guard failed")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "run_id": RUN_ID,
        "source_set_id": SOURCE_SET_ID,
        "complete": True,
        "producer": str(Path(__file__).resolve().relative_to(REPO)),
        "slurm_job_id": os.environ["SLURM_JOB_ID"],
        "thread_environment": thread_environment,
        "expected_file_count": len(EXPECTED_FILES),
        "files": manifest_entries,
        "access_ledger": access_ledger,
        "access_summary": {
            "opened_count": opened_count,
            "test_like_attempt_count": test_like_attempts,
            "test_like_open_count": sum(
                1
                for entry in access_ledger
                if entry["test_like"] and entry["opened"]
            ),
        },
    }
    payload = json_bytes(manifest)
    destination = NAMESPACE / MANIFEST_NAME
    exclusive_publish(destination, payload)
    print(
        json.dumps(
            {
                "run_id": RUN_ID,
                "manifest": str(destination.relative_to(REPO)),
                "manifest_sha256": hashlib.sha256(payload).hexdigest(),
                "files": len(manifest_entries),
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("C01_HASH_PREFLIGHT_FAIL_CLOSED: {}".format(exc), file=sys.stderr)
        raise
